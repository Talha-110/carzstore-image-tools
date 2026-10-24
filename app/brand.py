"""Corner logo for lifestyle finals (v1.2.1, Chat 6E — Talha: the white logo card looked boring).

New look: the logo itself, no card. The white background of the current logo file is removed
in code, the tagline line is dropped (unreadable at this size), and the dark-green wordmark
text turns white with a soft shadow — the orange car arc keeps its colour. On a bright corner
(sky, white wall) the original dark-green colours are used instead, with a soft white glow,
so the logo is always readable.

When the HD transparent logo arrives, drop it in assets/logo.png: a file that already has
transparency is used as it is (no white removal).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from . import imaging as T

WIDTH_PCT = 0.16       # logo width on a square image (scaled by area on wide/tall ones)
MARGIN_PCT = 0.035     # gap to the right and bottom edge
BRIGHT_CORNER = 165    # corner brightness (0-255) above which the dark logo is used


@lru_cache(maxsize=1)
def _logo_variants() -> tuple[Image.Image, Image.Image]:
    """(light version for dark/busy corners, dark version for bright corners), RGBA, trimmed."""
    src = Image.open(T.LOGO_PATH)
    src = T.to_rgba(src)
    a = np.asarray(src, dtype=np.float64)
    if a[..., 3].min() >= 250:                        # no transparency: key out the white
        rgb = a[..., :3]
        alpha = np.clip(((255 - rgb.min(axis=2)) / 255 - 0.04) / 0.96, 0, 1)
        safe = np.maximum(alpha[..., None], 1e-3)
        rgb = np.where(alpha[..., None] > 0, 255 - (255 - rgb) / safe, 255)
        a = np.dstack([np.clip(rgb, 0, 255), alpha * 255])
    logo = Image.fromarray(a.astype(np.uint8), "RGBA")
    box = logo.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
    if box:
        logo = logo.crop(box)

    # drop the tagline: cut at the widest empty band in the lower half
    solid = np.asarray(logo.getchannel("A")) > 40
    rows = solid.sum(axis=1)
    h = len(rows)
    best, start = None, None
    for i in range(h // 2, h):
        if rows[i] == 0 and start is None:
            start = i
        if rows[i] != 0 and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best and best[1] - best[0] >= 2:
        logo = logo.crop((0, 0, logo.width, best[0]))
        logo = logo.crop(logo.getchannel("A").getbbox())

    dark = logo
    arr = np.asarray(logo).astype(np.float64)
    textish = arr[..., :3].mean(axis=-1) < 110        # dark green text -> white, orange stays
    arr[textish, :3] = 255
    light = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return light, dark


def _halo(layer: Image.Image, color, blur: int, opacity: float, dy: int = 0) -> tuple[Image.Image, int]:
    pad = blur * 3
    base = Image.new("RGBA", (layer.width + 2 * pad, layer.height + 2 * pad), (0, 0, 0, 0))
    solid = Image.new("RGBA", layer.size, color + (255,))
    solid.putalpha(layer.getchannel("A").point(lambda v: int(v * opacity)))
    base.alpha_composite(solid, (pad, pad + dy))
    return base.filter(ImageFilter.GaussianBlur(blur)), pad


def overlay_logo(im: Image.Image, width_pct: float = WIDTH_PCT, margin_pct: float = MARGIN_PCT,
                 opacity: float = 1.0) -> Image.Image:
    """Same call as imaging.overlay_logo (used by /export logo=true and /overlay/logo)."""
    im = T.to_rgba(im)
    light, dark = _logo_variants()
    w = int(min(im.width * width_pct, math.sqrt(im.width * im.height) * width_pct))
    h = max(1, int(light.height * w / light.width))
    m = int(min(im.width, im.height) * margin_pct)
    x, y = im.width - w - m, im.height - h - m

    corner = im.crop((max(0, x - m), max(0, y - m), im.width, im.height)).convert("L")
    bright = np.asarray(corner, dtype=np.float64).mean() > BRIGHT_CORNER
    logo = (dark if bright else light).resize((w, h), Image.LANCZOS)
    if opacity < 1:
        logo.putalpha(logo.getchannel("A").point(lambda v: int(v * opacity)))

    blur = max(2, w // 45)
    if bright:
        halo, pad = _halo(logo, (255, 255, 255), blur, 0.85)
        im.alpha_composite(halo, (x - pad, y - pad))
    else:
        halo, pad = _halo(logo, (0, 0, 0), blur, 0.60, dy=max(1, blur // 2))
        im.alpha_composite(halo, (x - pad, y - pad))
    im.alpha_composite(logo, (x, y))
    return im.convert("RGB")
