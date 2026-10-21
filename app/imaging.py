"""Carzstore image-tools: all no-AI image work.

Specs come from the project files:
  01 brand colors, 04 s9 clean spec, 04 s10 promo spec, 04 s11 logo,
  03 s6d photo checks, 05 s8a golden set, 05 s9 code checks.
"""
from __future__ import annotations

import io
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy import ndimage

try:  # iPhone HEIC photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

ASSETS = Path(__file__).resolve().parent.parent / "assets"
FONT_BOLD = str(ASSETS / "fonts" / "Poppins-Bold.ttf")
FONT_REGULAR = str(ASSETS / "fonts" / "Poppins-Regular.ttf")
LOGO_PATH = Path(os.getenv("LOGO_PATH", str(ASSETS / "logo.png")))

# Brand colors (01 section 2)
ORANGE_BANNER = (254, 138, 2)      # #FE8A02
LIGHT_ORANGE = (251, 190, 114)     # #FBBE72
DARK_GREEN = (0, 31, 16)           # #001F10
ASPHALT = (44, 44, 44)             # #2C2C2C
WHITE = (255, 255, 255)

SIZES = {  # 01 section 6
    "STORE": (2048, 2048),
    "POST": (1080, 1350),
    "STORY": (1080, 1920),
    "BANNER": (1200, 600),
}
SIZE_LABEL = {"STORE": "2048", "POST": "post-1080x1350",
              "STORY": "story-1080x1920", "BANNER": "banner-1200x600"}

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
CONVERT_FORMATS = {"HEIF", "HEIC", "AVIF"}


class ToolError(Exception):
    """A known problem with a code from 03 section 8 (E..) or H.. reasons."""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Loading / saving
# --------------------------------------------------------------------------
def open_image(data: bytes) -> tuple[Image.Image, str]:
    """Open by content (not file name). Returns upright image + real format."""
    if not data:
        raise ToolError("E05", "Empty file")
    try:
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").upper()
        im.load()
    except Exception:
        raise ToolError("E09", "Not an image file, or a type we can't read")
    im = ImageOps.exif_transpose(im)  # phone photos come in sideways otherwise
    return im, fmt


def to_rgba(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGBA" else im.convert("RGBA")


def encode(im: Image.Image, fmt: str = "PNG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt in ("JPEG", "JPG"):
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, WHITE)
            bg.paste(im, mask=im.getchannel("A") if im.mode == "RGBA" else None)
            im = bg
        im.save(buf, "JPEG", quality=quality, optimize=True)
    elif fmt == "WEBP":
        im.save(buf, "WEBP", quality=quality, method=6)
    else:
        im.save(buf, "PNG", optimize=False, compress_level=6)
    return buf.getvalue()


# --------------------------------------------------------------------------
# /inspect — photo checks (03 section 6d)
# --------------------------------------------------------------------------
def inspect(data: bytes) -> dict:
    max_mb = env_float("MAX_FILE_MB", 20)
    min_side = env_float("MIN_SHORT_SIDE_PX", 800)
    low_res = env_float("LOW_RES_BELOW_PX", 1200)

    mb = len(data) / 1_048_576
    if mb > max_mb:
        raise ToolError("E07", f"File is {mb:.1f} MB, limit is {max_mb:.0f} MB")

    im, fmt = open_image(data)
    flags: list[str] = []
    converted = None
    if fmt in CONVERT_FORMATS:
        flags.append("CONVERTED_FORMAT")
        converted = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    elif fmt not in ALLOWED_FORMATS:
        raise ToolError("E09", f"File type {fmt or 'unknown'} is not supported (use JPG, PNG or WEBP)")

    w, h = im.size
    short = min(w, h)
    ratio = w / h
    looks_like_banner = (abs(w - 1200) <= 60 and abs(h - 630) <= 60) or ratio > 1.7 or ratio < 0.55
    if short < min_side and looks_like_banner:
        raise ToolError("H04", f"Image is {w}x{h}: looks like a social/banner crop, not the product photo")
    if short < min_side:
        raise ToolError("E08", f"Image is {w}x{h}; shortest side must be at least {int(min_side)} px")
    if short < low_res:
        flags.append("LOW_RES")

    gray = np.asarray(im.convert("L").resize((256, max(1, int(256 * h / w)))), dtype=np.float32)
    if gray.std() < 2.0:
        raise ToolError("E05", "Image is blank (one flat color)")

    return {
        "format": fmt, "width": w, "height": h, "size": f"{w}x{h}",
        "file_mb": round(mb, 2), "flags": flags,
        "looks_like_banner": looks_like_banner,
        "converted": converted,
    }


# --------------------------------------------------------------------------
# /cutout — background removal + cutout check (02 WF1 step 6–7, H01)
# --------------------------------------------------------------------------
_sessions: dict = {}
_session_lock = threading.Lock()
DEFAULT_MODEL = os.getenv("CUTOUT_MODEL", "isnet-general-use")
# birefnet models were tested (Chat 6) and need >7 GB RAM per photo on CPU: not allowed.
ALLOWED_MODELS = {"isnet-general-use", "u2net"}


def get_session(model: str):
    from rembg import new_session
    with _session_lock:
        if model not in _sessions:
            _sessions[model] = new_session(model)
        return _sessions[model]


def loaded_models() -> list[str]:
    return sorted(_sessions)


@dataclass
class CutoutResult:
    image: Image.Image            # trimmed RGBA cutout
    meta: dict = field(default_factory=dict)


def drop_small_pieces(cut: Image.Image) -> tuple[Image.Image, int]:
    """Remove tiny loose pieces the cutout model kept by mistake — mostly caption
    text on supplier photos ("Suzuki Swift 2023") and specks. Big separate pieces
    (e.g. 4 floor mats in one photo) are kept.

    A piece is dropped if it is smaller than CUTOUT_DROP_SMALL_PCT (default 2%)
    of the biggest piece. Safety: if the small pieces together are more than
    CUTOUT_DROP_MAX_SHARE (default 15%) of the product, nothing is dropped —
    they are probably real parts (clips, screws) and a human should look.
    """
    pct = env_float("CUTOUT_DROP_SMALL_PCT", 2) / 100
    if pct <= 0:
        return cut, 0
    alpha = np.asarray(cut.getchannel("A"))
    solid = alpha > 10
    # join pieces that touch or nearly touch (soft edges, thin straps)
    joined = ndimage.binary_dilation(solid, iterations=2)
    labels, n = ndimage.label(joined)
    if n <= 1:
        return cut, 0
    sizes = np.asarray(ndimage.sum(solid, labels, range(1, n + 1)))
    small = sizes < pct * sizes.max()
    if not small.any():
        return cut, 0
    if sizes[small].sum() > env_float("CUTOUT_DROP_MAX_SHARE", 0.15) * sizes[~small].sum():
        return cut, 0
    remove_ids = np.flatnonzero(small) + 1
    kill = np.isin(labels, remove_ids) & solid
    new_alpha = alpha.copy()
    new_alpha[kill] = 0
    out = cut.copy()
    out.putalpha(Image.fromarray(new_alpha))
    return out, int(small.sum())


def cutout(data: bytes, model: str | None = None) -> CutoutResult:
    from rembg import remove

    model = model or DEFAULT_MODEL
    if model not in ALLOWED_MODELS:
        raise ToolError("E01", f"Unknown cutout model '{model}'")
    im, _ = open_image(data)
    im = im.convert("RGB")
    try:
        out = remove(im, session=get_session(model), post_process_mask=True)
    except Exception as e:  # model crash, out of memory...
        raise ToolError("E11", f"Background removal failed: {e}", status=500)
    out = to_rgba(out)
    out, dropped = drop_small_pieces(out)

    alpha = np.asarray(out.getchannel("A"))
    mask = alpha > 128
    total = mask.size
    area = int(mask.sum())
    if area == 0:
        raise ToolError("E11", "Background removal found no product")

    ys, xs = np.where(alpha > 10)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = out.size
    fill_pct = 100.0 * area / total
    box_w_pct = 100.0 * (x1 - x0) / w
    box_h_pct = 100.0 * (y1 - y0) / h

    # Holes: transparent pockets fully inside the product (e.g. a bad cutout of a mat)
    filled = ndimage.binary_fill_holes(mask)
    hole_ratio = float((filled.sum() - area) / max(filled.sum(), 1))
    # Bits: many separate pieces usually means background noise was kept
    labels, n = ndimage.label(mask)
    big_pieces = 0
    if n:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        big_pieces = int((np.asarray(sizes) > 0.01 * area).sum())
    touches_edge = bool(x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1)

    problems = []
    if fill_pct < env_float("CUTOUT_MIN_FILL_PCT", 3):
        problems.append(f"product too small ({fill_pct:.1f}% of photo)")
    if fill_pct > env_float("CUTOUT_MAX_FILL_PCT", 92) and box_w_pct > 97 and box_h_pct > 97:
        problems.append("background was probably not removed (cutout fills the whole photo)")
    if hole_ratio > env_float("CUTOUT_MAX_HOLE_RATIO", 0.5):
        problems.append(f"big holes inside the product ({hole_ratio:.0%})")
    if big_pieces > int(env_float("CUTOUT_MAX_PIECES", 6)):
        problems.append(f"cutout is in {big_pieces} separate pieces")

    pad = int(0.02 * max(x1 - x0, y1 - y0))
    box = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
    trimmed = out.crop(box)

    return CutoutResult(trimmed, {
        "model": model,
        "cutout_fill_pct": round(fill_pct, 1),
        "box_pct": [round(box_w_pct, 1), round(box_h_pct, 1)],
        "hole_ratio": round(hole_ratio, 3),
        "pieces": big_pieces,
        "small_pieces_removed": dropped,
        "flags": ["SMALL_PIECES_REMOVED"] if dropped else [],
        "touches_edge": touches_edge,
        "check_ok": not problems,
        "check_code": None if not problems else "H01",
        "problems": problems,
        "width": trimmed.width, "height": trimmed.height,
    })


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def clean_edges(cut: Image.Image) -> Image.Image:
    """Shrink the edge 1 px and soften 1 px (removes white/colored halo, 04 s9)."""
    cut = to_rgba(cut)
    a = cut.getchannel("A").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
    cut = cut.copy()
    cut.putalpha(a)
    return cut


def product_box(cut: Image.Image) -> tuple[int, int, int, int]:
    box = cut.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
    return box or (0, 0, cut.width, cut.height)


def tight(cut: Image.Image) -> Image.Image:
    return to_rgba(cut).crop(product_box(cut))


def drop_shadow(cut: Image.Image, opacity=0.20, blur=30, offset=12) -> Image.Image:
    """Shadow layer the same size as `cut` + margin. Returns (layer, margin)."""
    m = blur * 3 + offset
    layer = Image.new("RGBA", (cut.width + 2 * m, cut.height + 2 * m), (0, 0, 0, 0))
    a = cut.getchannel("A").point(lambda v: int(v * opacity))
    sh = Image.new("RGBA", cut.size, (0, 0, 0, 255))
    sh.putalpha(a)
    layer.alpha_composite(sh, (m, m + offset))
    return layer.filter(ImageFilter.GaussianBlur(blur)), m


def paste_with_shadow(canvas: Image.Image, cut: Image.Image, x: int, y: int,
                      opacity=0.20, blur=30, offset=12) -> None:
    sh, m = drop_shadow(cut, opacity, blur, offset)
    _safe_composite(canvas, sh, x - m, y - m)   # shadow may hang over the canvas edge
    _safe_composite(canvas, cut, x, y)


def _safe_composite(canvas: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tmp.paste(layer, (x, y), layer)
    canvas.alpha_composite(tmp)


# --------------------------------------------------------------------------
# /compose/clean (04 section 9)
# --------------------------------------------------------------------------
def compose_clean(cut_bytes: bytes, size: int = 2048) -> tuple[Image.Image, dict]:
    cut, _ = open_image(cut_bytes)
    if cut.mode != "RGBA":
        raise ToolError("E01", "Clean image needs the transparent cutout PNG")
    cut = tight(clean_edges(cut))
    target = int(round(size * 0.80))           # longest side = 80% of canvas
    scale = target / max(cut.size)
    new = (max(1, round(cut.width * scale)), max(1, round(cut.height * scale)))
    cut = cut.resize(new, Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), WHITE + (255,))
    x = (size - cut.width) // 2
    y = int(size * 0.48 - cut.height / 2)       # product center at 48% height
    s = size / 2048
    paste_with_shadow(canvas, cut, x, y, 0.20, max(1, int(30 * s)), max(1, int(12 * s)))

    flags = ["LOW_RES"] if scale > 2 else []
    longest_pct = 100.0 * max(new) / size
    edge_touch = x <= 0 or y <= 0 or x + cut.width >= size or y + cut.height >= size
    ok = 75 <= longest_pct <= 85 and not edge_touch
    return canvas.convert("RGB"), {
        "fill_longest_pct": round(longest_pct, 1),
        "upscale": round(scale, 2), "flags": flags,
        "check_ok": ok, "check_code": None if ok else "H01",
    }


# --------------------------------------------------------------------------
# Text helpers (promo)
# --------------------------------------------------------------------------
def font(path: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(8, int(px)))


def text_w(f: ImageFont.FreeTypeFont, t: str) -> int:
    return int(math.ceil(f.getlength(t)))


def wrap_words(text: str, f, max_w: int, max_lines: int) -> list[str] | None:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if text_w(f, w) > max_w:
            return None
        trial = f"{cur} {w}".strip()
        if text_w(f, trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines if len(lines) <= max_lines else None


def fit_headline(text: str, max_w: int, img_h: int, start_pct=0.085, min_pct=0.045,
                 max_lines: int = 2):
    """Start big and shrink until it fits (2 lines; STORY 3). Never below 4.5% of height."""
    px = int(img_h * start_pct)
    min_px = int(math.ceil(img_h * min_pct))
    while px >= min_px:
        f = font(FONT_BOLD, px)
        lines = wrap_words(text, f, max_w, max_lines)
        if lines:
            return f, lines, px
        px -= max(1, px // 40)
    raise ToolError("E01", "promo_headline too long to fit")


def is_latin_text(t: str) -> bool:
    """English / Roman Urdu only (Poppins can't draw Urdu script or emojis)."""
    for ch in t:
        o = ord(ch)
        if ch.isspace() or 0x20 <= o <= 0x7E or 0xA0 <= o <= 0x24F or ch in "–—‘’“”…•₨":
            continue
        return False
    return True


def format_price(p: str) -> str:
    digits = "".join(ch for ch in str(p) if ch.isdigit())
    if not digits:
        raise ToolError("E01", "promo_price must be a whole number")
    return f"Rs {int(digits):,}"


def rounded_box(draw: ImageDraw.ImageDraw, box, radius, fill) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def logo_card(width: int, opacity: float = 1.0) -> Image.Image:
    """Logo on a white rounded card (current logo has a white background)."""
    logo = Image.open(LOGO_PATH)
    logo = to_rgba(logo)
    has_alpha = logo.getchannel("A").getextrema()[0] < 250
    pad = int(width * 0.06)
    inner = width - 2 * pad
    lh = int(logo.height * inner / logo.width)
    logo = logo.resize((inner, lh), Image.LANCZOS)
    if has_alpha and os.getenv("LOGO_CARD", "auto") == "off":
        return logo
    card = Image.new("RGBA", (width, lh + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=int(width * 0.07),
                        fill=(255, 255, 255, int(255 * opacity)))
    card.alpha_composite(logo, (pad, pad))
    return card


# --------------------------------------------------------------------------
# /compose/promo (04 section 10, 05 section 9)
# --------------------------------------------------------------------------
@dataclass
class PromoLayout:
    road_pct: float
    text_x_pct: float = 0.055
    text_max_w_pct: float = 0.48
    logo_w_pct: float = 0.22
    top_safe_pct: float = 0.05
    product_area: tuple = (0.50, 0.95)   # x range for product (fractions of W)
    product_max_w: float = 0.50
    product_max_h: float = 0.55
    stacked: bool = False                # STORY: text top, product bottom
    headline_start: float = 0.085        # first try, as share of image height
    price_pct: float = 0.048
    offer_pct: float = 0.032
    text_limit: float = 1.0              # text must end above this share of height
    max_lines: int = 2


LAYOUTS = {
    "STORE": PromoLayout(road_pct=0.16),
    "POST": PromoLayout(road_pct=0.15),
    "BANNER": PromoLayout(road_pct=0.15, text_max_w_pct=0.50, logo_w_pct=0.17,
                          product_area=(0.56, 0.96), product_max_w=0.42, product_max_h=0.66,
                          headline_start=0.10, price_pct=0.055, offer_pct=0.037),
    # STORY: text out of the top 14% (Instagram buttons); road at the bottom has no text.
    "STORY": PromoLayout(road_pct=0.14, text_max_w_pct=0.86, logo_w_pct=0.26,
                         top_safe_pct=0.14, product_area=(0.08, 0.92),
                         product_max_w=0.84, product_max_h=0.36, stacked=True,
                         headline_start=0.065, price_pct=0.038, offer_pct=0.027,
                         text_limit=0.64, max_lines=3),
}

# Side-by-side layouts: wide products (mat sets, body kits) may use up to 62% of the
# width when they sit below/next to short text lines (04 s10 said 50%; Chat 6 change).
PROMO_SIDE_MAX_W = 0.62

STACKED_FALLBACK = {
    "STORE": PromoLayout(road_pct=0.16, text_max_w_pct=0.88, logo_w_pct=0.20,
                         product_area=(0.10, 0.90), product_max_w=0.70, product_max_h=0.45,
                         stacked=True, headline_start=0.075, price_pct=0.042,
                         offer_pct=0.029, text_limit=0.55),
    "POST": PromoLayout(road_pct=0.15, text_max_w_pct=0.88, logo_w_pct=0.22,
                        product_area=(0.10, 0.90), product_max_w=0.70, product_max_h=0.42,
                        stacked=True, headline_start=0.068, price_pct=0.040,
                        offer_pct=0.028, text_limit=0.57),
}


def _draw_background(W: int, H: int, road_h: int) -> Image.Image:
    bg = Image.new("RGBA", (W, H), ORANGE_BANNER + (255,))
    lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lines)
    stroke = max(2, int(min(W, H) * 0.006))
    alpha = int(255 * 0.40)
    cx, cy = int(W * 1.05), int(H * 1.0)          # sweep from bottom-right
    for r in (0.55, 0.75, 0.95):
        rr = int(max(W, H) * r)
        d.arc((cx - rr, cy - rr, cx + rr, cy + rr), start=180, end=270,
              fill=LIGHT_ORANGE + (alpha,), width=stroke)
    bg.alpha_composite(lines)
    # road strip with white dashed center line
    d = ImageDraw.Draw(bg)
    top = H - road_h
    d.rectangle((0, top, W, H), fill=ASPHALT + (255,))
    mid = top + road_h // 2
    dash, gap = int(W * 0.06), int(W * 0.04)
    lw = max(2, int(road_h * 0.06))
    x = int(W * 0.02)
    while x < W:
        d.rectangle((x, mid - lw // 2, min(W, x + dash), mid + lw // 2), fill=WHITE + (255,))
        x += dash + gap
    return bg


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def compose_promo(cut_bytes: bytes, headline: str, price: str | None,
                  offer: str | None, size_key: str = "STORE") -> tuple[Image.Image, dict]:
    """Normal layout first (text left, product right). If a long headline can't fit
    that way, STORE/POST fall back to text-on-top, product-below."""
    key = (size_key or "STORE").upper()
    try:
        return _compose_promo(cut_bytes, headline, price, offer, key, LAYOUTS.get(key))
    except ToolError as e:
        fallback = STACKED_FALLBACK.get(key)
        retry = e.code == "H01" or e.message in ("promo_headline too long to fit",
                                                  "promo text too long to fit above the product area")
        if not (fallback and retry):
            raise
        im, meta = _compose_promo(cut_bytes, headline, price, offer, key, fallback)
        meta["note"] = "long headline: stacked layout used"
        return im, meta


def _compose_promo(cut_bytes: bytes, headline: str, price: str | None,
                   offer: str | None, size_key: str, L: "PromoLayout | None") -> tuple[Image.Image, dict]:
    size_key = (size_key or "STORE").upper()
    if size_key not in SIZES:
        raise ToolError("E01", f"Unknown size '{size_key}'")
    headline = " ".join((headline or "").split())
    offer = " ".join((offer or "").split()) or None
    if not headline:
        raise ToolError("E01", "promo_headline is empty")
    if len(headline) > 40:
        raise ToolError("E01", "promo_headline is longer than 40 characters")
    if offer and len(offer) > 25:
        raise ToolError("E01", "promo_offer is longer than 25 characters")
    for name, val in (("promo_headline", headline), ("promo_offer", offer or "")):
        if not is_latin_text(val):
            raise ToolError("E01", f"{name} has unsupported characters")
    price_text = format_price(price) if price not in (None, "") else None

    cut, _ = open_image(cut_bytes)
    if cut.mode != "RGBA":
        raise ToolError("E01", "Promo image needs the transparent cutout PNG")
    cut = tight(clean_edges(cut))

    W, H = SIZES[size_key]
    road_h = int(H * L.road_pct)
    road_top = H - road_h
    canvas = _draw_background(W, H, road_h)

    # Logo top-left
    margin_x = int(W * L.text_x_pct)
    logo = logo_card(int(W * L.logo_w_pct))
    logo_y = int(H * L.top_safe_pct)
    canvas.alpha_composite(logo, (margin_x, logo_y))
    y = logo_y + logo.height + int(H * (0.03 if L.stacked else 0.04))

    # Headline
    max_w = int(W * L.text_max_w_pct)
    hf, lines, hpx = fit_headline(headline, max_w, H, start_pct=L.headline_start,
                                  max_lines=L.max_lines)
    d = ImageDraw.Draw(canvas)
    line_h = int(hpx * 1.15)
    # Every text piece keeps its own box, so the product can use the free space
    # next to short lines and pills (not just right of the widest line).
    rects = [(margin_x, logo_y, margin_x + logo.width, logo_y + logo.height)]
    for ln in lines:
        d.text((margin_x, y), ln, font=hf, fill=WHITE + (255,))
        rects.append((margin_x, y, margin_x + text_w(hf, ln), y + line_h))
        y += line_h
    y += int(hpx * 0.30)

    # Price pill (dark green on white) and offer pill (white on dark green)
    def pill(text, fpath, px, fg, bg_col):
        nonlocal y
        f = font(fpath, px)
        tw = text_w(f, text)
        asc, desc = f.getmetrics()
        padx, pady = int(px * 0.55), int(px * 0.28)
        box = (margin_x, y, margin_x + tw + 2 * padx, y + asc + desc + 2 * pady)
        rounded_box(d, box, radius=(box[3] - box[1]) // 2, fill=bg_col + (255,))
        d.text((margin_x + padx, y + pady), text, font=f, fill=fg + (255,))
        rects.append(box)
        y = box[3] + int(px * 0.35)

    if price_text:
        pill(price_text, FONT_BOLD, int(H * L.price_pct), DARK_GREEN, WHITE)
    if offer:
        pill(offer, FONT_REGULAR, int(H * L.offer_pct), WHITE, DARK_GREEN)
    text_bottom = y

    limit_bottom = min(road_top, int(H * L.text_limit))
    if text_bottom > limit_bottom:
        raise ToolError("E01", "promo text too long to fit above the product area")
    # STORY: product goes in the space between the text and the road
    stack_top = text_bottom + int(H * 0.02)
    stack_room = road_top - int(H * 0.015) - stack_top
    if L.stacked and stack_room < H * 0.18:
        raise ToolError("E01", "promo text leaves too little room for the product")

    gap = int(min(W, H) * 0.025)
    def hits_text(box):
        return any(_overlap((r[0] - gap, r[1] - gap, r[2] + gap, r[3] + gap), box) for r in rects)

    def sized(sc):
        return cut.resize((max(1, int(cut.width * sc)), max(1, int(cut.height * sc))), Image.LANCZOS)

    # Product
    note = None
    rest_y = road_top - int(H * 0.015)          # resting just above the road
    if L.stacked:
        ax0, ax1 = int(W * L.product_area[0]), int(W * L.product_area[1])
        max_w_px = min(W * L.product_max_w, ax1 - ax0)
        max_h_px = min(H * L.product_max_h, stack_room)
        sc = min(max_w_px / cut.width, max_h_px / cut.height)
        pc = sized(sc)
        px_ = ax0 + (ax1 - ax0 - pc.width) // 2
        py_ = rest_y - pc.height
    else:
        # Biggest size that fits bottom-right without touching any text box.
        right = W - int(W * 0.04)
        sc_max = min(W * PROMO_SIDE_MAX_W / cut.width, H * L.product_max_h / cut.height)
        sc, found = sc_max, None
        while sc >= sc_max * 0.30:
            w_, h_ = int(cut.width * sc), int(cut.height * sc)
            box = (right - w_, rest_y - h_, right, rest_y)
            if not hits_text(box) and box[0] >= int(W * 0.04):
                found = sc
                break
            sc *= 0.97
        too_small = found is not None and found < sc_max * 0.55
        if found is None or (too_small and size_key in STACKED_FALLBACK
                             and L is LAYOUTS.get(size_key)):
            # caller retries STORE/POST with the stacked layout
            raise ToolError("H01", "Promo text and product overlap", status=422)
        pc = sized(found)
        sc = found
        px_, py_ = right - pc.width, rest_y - pc.height
        if found < sc_max * 0.999:
            note = f"product sized to {found / sc_max:.0%} of max to stay clear of the text"

    s = W / 2048
    paste_with_shadow(canvas, pc, px_, py_, 0.30, max(1, int(28 * s)), max(1, int(14 * s)))

    meta = {"size": size_key, "width": W, "height": H, "headline_px": hpx,
            "headline_pct": round(100 * hpx / H, 2), "lines": lines,
            "price_text": price_text, "upscale": round(sc, 2),
            "flags": ["LOW_RES"] if sc > 2 else []}
    if note:
        meta["note"] = note
    return canvas.convert("RGB"), meta


# --------------------------------------------------------------------------
# /overlay/logo (04 section 11)
# --------------------------------------------------------------------------
def overlay_logo(im: Image.Image, width_pct=0.14, margin_pct=0.03, opacity=0.90) -> Image.Image:
    im = to_rgba(im)
    # 14% of width on square images; scaled by area on wide/tall ones (banner stays small)
    w = min(im.width * width_pct, math.sqrt(im.width * im.height) * width_pct)
    card = logo_card(int(w), opacity)
    m = int(im.width * margin_pct)
    im.alpha_composite(card, (im.width - card.width - m, im.height - card.height - m))
    return im.convert("RGB")


# --------------------------------------------------------------------------
# /export — sizes + SEO names (02 s6, 03 s9)
# --------------------------------------------------------------------------
def fit_to(im: Image.Image, W: int, H: int, mode: str, pad_color=WHITE) -> Image.Image:
    im = im.convert("RGB")
    if mode == "pad":   # keep everything (clean image): letterbox on white
        sc = min(W / im.width, H / im.height)
        r = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
        out = Image.new("RGB", (W, H), pad_color)
        out.paste(r, ((W - r.width) // 2, (H - r.height) // 2))
        return out
    return ImageOps.fit(im, (W, H), Image.LANCZOS, centering=(0.5, 0.5))  # crop to fill


def export(im: Image.Image, sizes: list[str], slug: str, img_type: str,
           mode: str = "crop", logo: bool = False, fmt: str = "WEBP",
           quality: int = 90) -> list[dict]:
    out = []
    ext = "webp" if fmt.upper() == "WEBP" else ("jpg" if fmt.upper() in ("JPG", "JPEG") else "png")
    for key in sizes:
        key = key.strip().upper()
        if key not in SIZES:
            raise ToolError("E01", f"sizes_needed has an unknown word: {key}")
        W, H = SIZES[key]
        r = fit_to(im, W, H, mode)
        if logo:
            r = overlay_logo(r)
        name = f"{slug}-{img_type}-{SIZE_LABEL[key]}.{ext}"
        out.append({"size": key, "name": name, "image": r, "fmt": fmt, "quality": quality,
                    "width": W, "height": H})
    return out


# --------------------------------------------------------------------------
# /review-prep (05 section 2): longest side 1024, JPEG q90
# --------------------------------------------------------------------------
def review_prep(im: Image.Image, longest: int = 1024) -> Image.Image:
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if im.mode == "RGBA":     # cutouts: put on white so the reviewer sees the product
        bg = Image.new("RGB", im.size, WHITE)
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    sc = longest / max(im.size)
    if sc < 1:
        im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    return im


# --------------------------------------------------------------------------
# /golden (05 section 8a): broken copies of a good lifestyle image
# --------------------------------------------------------------------------
def golden_set(good: Image.Image, extra_cut: Image.Image | None = None) -> list[tuple[str, str, Image.Image]]:
    """Returns (kind, expected verdict, image)."""
    good = good.convert("RGB")
    W, H = good.size
    out = [("untouched", "PASS", good.copy())]

    # color drift: lift darks (black -> dark grey) and warm slightly
    arr = np.asarray(good).astype(np.float32)
    drift = np.clip(arr * 0.75 + 255 * 0.25, 0, 255)
    drift[..., 0] = np.clip(drift[..., 0] * 1.06, 0, 255)
    out.append(("color_drift", "FAIL (G1 or fidelity<=3)", Image.fromarray(drift.astype(np.uint8))))

    # text added
    t = good.copy()
    d = ImageDraw.Draw(t)
    f = font(FONT_BOLD, int(min(W, H) * 0.07))
    x0, y0 = int(W * 0.04), int(H * 0.04)
    l, tp, r, b = d.textbbox((x0 + 20, y0 + 10), "SALE Rs 999", font=f)
    d.rectangle((x0, y0, r + 20, b + 14), fill=(200, 0, 0))
    d.text((x0 + 20, y0 + 10), "SALE Rs 999", font=f, fill=WHITE)
    out.append(("text_added", "FAIL (G4)", t))

    # number plate
    p = good.copy()
    d = ImageDraw.Draw(p)
    pf = font(FONT_BOLD, int(min(W, H) * 0.06))
    x, y = int(W * 0.30), int(H * 0.86)
    l, tp, r, b = d.textbbox((x, y), "LEA 2024", font=pf)
    pad = int(min(W, H) * 0.015)
    d.rectangle((l - pad, tp - pad, r + pad, b + pad), fill=WHITE, outline=(0, 0, 0), width=max(2, pad // 3))
    d.text((x, y), "LEA 2024", font=pf, fill=(0, 0, 0))
    out.append(("number_plate", "FAIL (G4)", p))

    # left-hand drive: mirror the whole image
    out.append(("lhd_flip", "FAIL (steering_side=left)", ImageOps.mirror(good)))

    # extra product
    if extra_cut is not None:
        e = good.convert("RGBA")
        c = tight(to_rgba(extra_cut))
        sc = (W * 0.22) / max(c.size)
        c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))), Image.LANCZOS)
        e.alpha_composite(c, (int(W * 0.05), int(H * 0.55)))
        out.append(("extra_product", "FAIL (G6)", e.convert("RGB")))

    # cartoon: posterize + bold edges
    c = ImageOps.posterize(good, 3).filter(ImageFilter.SMOOTH_MORE)
    edges = good.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda v: 0 if v > 40 else 255)
    c = ImageChops.multiply(c, Image.merge("RGB", (edges, edges, edges)))
    out.append(("cartoon", "FAIL (G7)", c))

    # small product: shrink the scene to ~35% and place on grey
    s = Image.new("RGB", (W, H), (128, 128, 128))
    small = good.resize((int(W * 0.35), int(H * 0.35)), Image.LANCZOS)
    s.paste(small, ((W - small.width) // 2, (H - small.height) // 2))
    out.append(("small_product", "FAIL (G8 or visibility<=2)", s))
    return out
