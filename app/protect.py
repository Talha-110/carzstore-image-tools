"""Anti-copy protection for FINAL files (image-tools v1.2.0, Chat 6E, decision 07 / 2026-10-24).

Every final file made by /export and /compose/promo (output=webp) gets:
  1. An invisible pixel watermark carrying "Carzstore.pk" (secret key + magic number) and the
     product SKU. Tested: survives WEBP/JPEG re-saves (down to quality 50), resizing down to
     ~37 %, light blur. It does NOT survive crops or heavy edits.
  2. Copyright metadata: EXIF Artist / Copyright / ImageDescription and an XMP block
     (dc:creator, dc:rights, xmpRights:Marked, WebStatement, UsageTerms, SKU, alt text).
     Many sites strip metadata, so the pixel mark is the stronger proof.
/check-ownership reads both back from any image.

How the pixel mark works (short): the brightness of the image is blurred and shrunk to a
small grid (long side 256 px). Each 8x8 block of that grid is turned into frequencies (DCT)
and 7 low frequencies are nudged onto a secret lattice (QIM) that spells the payload bits,
each bit repeated many times. The change goes back into the full image as a soft
brightness ripple of about 1-3 levels out of 255, ONLY on textured areas: pure white
(clean background), plain flat colors (promo orange) and near-black stay untouched.
Reading = same blur + shrink + DCT, gain correction, majority vote, CRC check.

Every setting below is part of the "format": changing one makes older files unreadable.
Only WATERMARK_KEY comes from the environment, and it must never change once used.
"""
from __future__ import annotations

import binascii
import io
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, uniform_filter

OWNER = "Carzstore.pk"
WEBSITE = "https://carzstore.pk"
KEY = os.getenv("WATERMARK_KEY", "carzstore.pk-wm-v1")
WM_VERSION = "WM-1.0"

CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-"   # sku characters (03 s2a)
MAX_SKU = 40
MAGIC = 0xC25                                       # 12 bits

CANON = 256                                         # long side of the working grid
COEFS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)]
STEP = 14.0                                         # lattice step (strength)
BLUR = 1.0                                          # pre-blur, in grid pixels
SKIP_BRIGHT, SKIP_DARK = 236.0, 14.0                # block brightness left untouched
TEXTURE = 2.5                                       # local spread that counts as "textured"
ROOM_MIN = 0.75                                     # share of a block that must be textured
WHITE_KEEP = 250.0                                  # pixels this bright are never changed
GAINS = np.round(np.arange(0.70, 1.0601, 0.004), 3) # resize/blur gains tried when reading


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------
def _crc(sku: str) -> int:
    return binascii.crc_hqx((sku + "|" + KEY).encode(), 0xFFFF) & 0xFFFF


def _bits(value: int, n: int) -> list[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def nbits_for(n: int) -> int:
    """magic 12 + length 6 + 6 per sku character + crc 16. Short SKUs = fewer bits = more repeats."""
    return 12 + 6 + 6 * n + 16


def encode_payload(sku: str) -> np.ndarray:
    sku = (sku or "").strip().lower()
    if not sku or len(sku) > MAX_SKU or any(c not in CHARSET for c in sku):
        raise ValueError("sku must be 1-40 characters: a-z, 0-9 or hyphen")
    bits = _bits(MAGIC, 12) + _bits(len(sku), 6)
    for ch in sku:
        bits += _bits(CHARSET.index(ch) + 1, 6)
    bits += _bits(_crc(sku), 16)
    return np.array(bits, dtype=np.int8)


def decode_payload(bits: np.ndarray) -> str | None:
    def val(a, b):
        v = 0
        for x in bits[a:b]:
            v = (v << 1) | int(x)
        return v
    if len(bits) < nbits_for(1) or val(0, 12) != MAGIC:
        return None
    n = val(12, 18)
    if not 1 <= n <= MAX_SKU or len(bits) != nbits_for(n):
        return None
    chars = []
    for i in range(n):
        c = val(18 + 6 * i, 24 + 6 * i)
        if not 1 <= c <= len(CHARSET):
            return None
        chars.append(CHARSET[c - 1])
    sku = "".join(chars)
    return sku if val(len(bits) - 16, len(bits)) == _crc(sku) else None


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------
def canon_size(w: int, h: int) -> tuple[int, int]:
    if w >= h:
        return CANON, max(8, int(round(CANON * h / w / 8)) * 8)
    return max(8, int(round(CANON * w / h / 8)) * 8), CANON


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _shrink(y: np.ndarray, cw: int, ch: int, blur: bool = True) -> np.ndarray:
    """Blur by an amount tied to the shrink factor, then box-average onto the grid. The blur
    removes fine detail first, so a copy at another size gives (almost) the same grid."""
    sigma = BLUR * y.shape[1] / cw
    if blur and sigma > 0.3:
        y = gaussian_filter(y, sigma, mode="nearest")
    return np.asarray(Image.fromarray(y.astype(np.float32), "F").resize((cw, ch), Image.BOX),
                      dtype=np.float64)


def _grow(d: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(d.astype(np.float32), "F").resize((w, h), Image.BICUBIC),
                      dtype=np.float64)


def _blocks(a: np.ndarray) -> np.ndarray:            # (H, W) -> (by, bx, 8, 8)
    H, W = a.shape
    return a.reshape(H // 8, 8, W // 8, 8).swapaxes(1, 2)


def _unblocks(b: np.ndarray) -> np.ndarray:
    by, bx = b.shape[:2]
    return b.swapaxes(1, 2).reshape(by * 8, bx * 8)


def _room(y: np.ndarray, cw: int) -> np.ndarray:
    """Where the mark may go, per full-size pixel: 0 on pure white and on plain flat color
    (a ripple would show there), up to 1 on textured areas. Soft edges, no seams."""
    win = max(3, int(round(y.shape[1] / cw * 2)) | 1)
    mu = uniform_filter(y, win)
    local_sd = np.sqrt(np.maximum(uniform_filter(y * y, win) - mu * mu, 0))
    room = np.clip(local_sd / TEXTURE, 0, 1)
    room[y >= WHITE_KEEP] = 0.0
    return gaussian_filter(room, win / 2)


def _usable(yc: np.ndarray, roomc: np.ndarray) -> np.ndarray:
    """Blocks that carry the mark: mostly textured, not near-white, not near-black."""
    m = _blocks(yc).mean(axis=(2, 3))
    r = _blocks(roomc).mean(axis=(2, 3))
    return (m < SKIP_BRIGHT) & (m > SKIP_DARK) & (r > ROOM_MIN)


def _plan(by: int, bx: int):
    """Slot number (raster order: block by block, 7 frequencies each) + a keyed dither per slot.
    Raster order keeps every payload bit spread evenly, so even a small product area
    (promo, story) carries each bit several times."""
    rng = np.random.default_rng(binascii.crc32((KEY + f"|{by}x{bx}").encode()))
    slots = by * bx * len(COEFS)
    dither = rng.uniform(0, STEP, slots)
    return np.arange(slots).reshape(by, bx, len(COEFS)), dither.reshape(by, bx, len(COEFS))


def _bit_map(nb: int) -> np.ndarray:
    """Keyed shuffle of the payload bits (slot n carries bit _bit_map(nb)[n % nb])."""
    return np.random.default_rng(binascii.crc32((KEY + f"|bits{nb}").encode())).permutation(nb)


def _coefs(yc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blk = _blocks(yc)
    d = dctn(blk, axes=(2, 3), norm="ortho")
    c = np.stack([d[:, :, u, v] for u, v in COEFS], axis=-1)
    return d, c


# --------------------------------------------------------------------------
# embed / read
# --------------------------------------------------------------------------
def embed(im: Image.Image, sku: str, rounds: int = 4) -> Image.Image:
    """Returns an RGB copy with the invisible mark."""
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]
    if min(w, h) < 256:
        raise ValueError("image too small for a watermark")
    cw, ch = canon_size(w, h)
    payload = encode_payload(sku)
    y0 = _luma(rgb)
    room = _room(y0, cw)

    base = _shrink(y0, cw, ch)
    by, bx = ch // 8, cw // 8
    order, dither = _plan(by, bx)
    target_bits = payload[_bit_map(len(payload))[order % len(payload)]]
    active = _usable(base, _shrink(room, cw, ch, blur=False))

    delta_full = np.zeros_like(y0)
    for _ in range(rounds):
        # what a reader would see right now
        cur = _shrink(y0 + delta_full, cw, ch)
        d, c = _coefs(cur)
        off = dither + target_bits * STEP / 2
        q = np.round((c - off) / STEP) * STEP + off
        change = np.where(active[..., None], q - c, 0.0)
        if np.abs(change).max() < 0.05:
            break
        dd = np.zeros_like(d)
        for i, (u, v) in enumerate(COEFS):
            dd[:, :, u, v] = change[..., i]
        step_c = _unblocks(idctn(dd, axes=(2, 3), norm="ortho"))
        delta_full += _grow(step_c, w, h) * room

    out = np.clip(rgb + delta_full[..., None], 0, 255)
    return Image.fromarray(np.round(out).astype(np.uint8), "RGB")


def _read_once(y: np.ndarray, cw: int, ch: int) -> tuple[str | None, float]:
    yc = _shrink(y, cw, ch)
    by, bx = ch // 8, cw // 8
    order, dither = _plan(by, bx)
    _, c = _coefs(yc)
    act = _usable(yc, _shrink(_room(y, cw), cw, ch, blur=False))
    if act.sum() < 4:
        return None, 0.0
    # Resizing/blurring weakens each frequency a little (gain k). QIM reads absolute values,
    # so first find k per frequency, blind: at the right k, values sit on the lattice (|cos| ~ 1).
    for i in range(len(COEFS)):
        ci, di = c[..., i][act], dither[..., i][act]
        scores = [np.abs(np.cos(2 * np.pi * (ci / k - di) / STEP)).mean() for k in GAINS]
        c[..., i] = c[..., i] / GAINS[int(np.argmax(scores))]
    use = act[..., None].repeat(len(COEFS), axis=-1)
    v = np.cos(2 * np.pi * (c - dither) / STEP)[use]       # +1 = bit 0, -1 = bit 1
    o = order[use]
    best = (None, 0.0)
    for n in range(1, MAX_SKU + 1):                          # payload length is not known yet
        nb = nbits_for(n)
        idx = _bit_map(nb)[o % nb]
        votes = np.bincount(idx, weights=v, minlength=nb)
        counts = np.bincount(idx, minlength=nb)
        if counts.min() == 0:
            continue
        conf = float(np.mean(np.abs(votes) / counts))      # 1.0 = perfect, ~0 = no mark
        sku = decode_payload((votes < 0).astype(np.int8))
        if sku and len(sku) == n:
            return sku, round(conf, 3)
        if conf > best[1]:
            best = (None, round(conf, 3))
    return best


def read(im: Image.Image) -> dict:
    """Look for the mark. Tries the exact grid and its neighbours (rounding after resizes)."""
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]
    y = _luma(rgb)
    cw, ch = canon_size(w, h)
    tries = [(cw, ch)]
    for dx, dy in ((0, 8), (0, -8), (8, 0), (-8, 0)):
        nw, nh = cw + dx, ch + dy
        if CANON in (nw, nh) and min(nw, nh) >= 8 and (nw, nh) not in tries:
            tries.append((nw, nh))
    best = {"found": False, "sku": None, "confidence": 0.0}
    for tw, th in tries:
        sku, conf = _read_once(y, tw, th)
        if sku:
            return {"found": True, "sku": sku, "confidence": conf, "grid": f"{tw}x{th}"}
        if conf > best["confidence"]:
            best = {"found": False, "sku": None, "confidence": conf, "grid": f"{tw}x{th}"}
    return best


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------
def _ascii(t: str) -> str:
    t = t.replace("©", "(c)").replace("—", "-").replace("–", "-")
    return t.encode("ascii", "ignore").decode()


def copyright_line(year: int | None = None) -> str:
    return f"© {year or datetime.now().year} {OWNER}. All rights reserved."


def make_exif(sku: str, alt: str) -> Image.Exif:
    ex = Image.Exif()
    ex[0x013B] = OWNER                                     # Artist
    ex[0x8298] = _ascii(copyright_line())                  # Copyright
    ex[0x010E] = _ascii(alt or f"{OWNER} product image {sku}")[:250]   # ImageDescription
    ex[0x0131] = f"Carzstore image-tools {WM_VERSION}"     # Software
    return ex


def make_xmp(sku: str, alt: str) -> bytes:
    e = lambda t: escape(t or "", {'"': "&quot;"})
    xml = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmpRights:Marked="True"
    xmpRights:WebStatement="{e(WEBSITE)}"
    photoshop:Credit="{e(OWNER)}"
    photoshop:Source="{e(OWNER)} SKU {e(sku)}">
   <dc:creator><rdf:Seq><rdf:li>{e(OWNER)}</rdf:li></rdf:Seq></dc:creator>
   <dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{e(copyright_line())}</rdf:li></rdf:Alt></dc:rights>
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{e(sku)}</rdf:li></rdf:Alt></dc:title>
   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{e(alt)}</rdf:li></rdf:Alt></dc:description>
   <xmpRights:UsageTerms><rdf:Alt><rdf:li xml:lang="x-default">Do not copy or reuse without written permission from {e(OWNER)}.</rdf:li></rdf:Alt></xmpRights:UsageTerms>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xml.encode("utf-8")


def read_metadata(im: Image.Image) -> dict:
    out = {}
    try:
        ex = im.getexif()
        for tag, name in ((0x013B, "artist"), (0x8298, "copyright"), (0x010E, "description")):
            if ex.get(tag):
                out[name] = str(ex.get(tag))
    except Exception:
        pass
    xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")
    if xmp:
        out["xmp_marked"] = 'xmpRights:Marked="True"' in xmp
        out["xmp_owner"] = OWNER if OWNER in xmp else None
        m = re.search(r"SKU ([a-z0-9-]+)", xmp)
        if m:
            out["xmp_sku"] = m.group(1)
    return out


# --------------------------------------------------------------------------
# final file = mark + metadata + self-check
# --------------------------------------------------------------------------
def finalize(im: Image.Image, sku: str, alt: str = "", fmt: str = "WEBP",
             quality: int = 90, watermark: bool = True) -> tuple[bytes, dict]:
    """Encode a final file. Returns (bytes, meta). meta.watermark_ok = the mark was read
    back from the encoded bytes with the right SKU."""
    im = im.convert("RGB")
    meta = {"watermark": False, "watermark_ok": None, "metadata": True, "wm_version": WM_VERSION}
    if watermark:
        im = embed(im, sku)
        meta["watermark"] = True
    buf = io.BytesIO()
    fmt = fmt.upper()
    exif, xmp = make_exif(sku, alt), make_xmp(sku, alt)
    if fmt == "WEBP":
        im.save(buf, "WEBP", quality=quality, method=6, exif=exif.tobytes(), xmp=xmp)
    elif fmt in ("JPEG", "JPG"):
        im.save(buf, "JPEG", quality=quality, optimize=True, exif=exif.tobytes(), xmp=xmp)
    else:
        im.save(buf, "PNG", exif=exif.tobytes())
    data = buf.getvalue()
    if watermark:
        r = read(Image.open(io.BytesIO(data)))
        meta["watermark_ok"] = bool(r["found"] and r["sku"] == sku.strip().lower())
        meta["wm_confidence"] = r["confidence"]
    return data, meta
