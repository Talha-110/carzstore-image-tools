"""image-tools HTTP API for the Carzstore n8n workflows.

Every image endpoint takes multipart form data and returns JSON:
  {"ok": true, "files": [{"name", "mime", "width", "height", "b64"}], "meta": {...}}
Known problems return HTTP 422 with:
  {"ok": false, "code": "E08", "message": "..."}      (codes from 03 section 8)
n8n turns "b64" into a binary file with one Code node line.

v1.2.0 (Chat 6E): final files from /export and /compose/promo (output=webp) get an invisible
pixel watermark (Carzstore.pk + SKU) and (c) metadata (app/protect.py). New: /check-ownership.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from . import imaging as T
from . import protect as W

VERSION = "1.2.0"
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_MB = T.env_float("MAX_UPLOAD_MB", 30)
# Heavy jobs (cutout) one at a time: protects VPS memory and n8n next to it.
_heavy = threading.Semaphore(int(T.env_float("HEAVY_JOBS", 1)))



@asynccontextmanager
async def lifespan(_app):
    """Download + load the default cutout model in the background (first start ~1 min)."""
    if os.getenv("WARMUP", "1") == "1":
        threading.Thread(target=lambda: T.get_session(T.DEFAULT_MODEL), daemon=True).start()
    yield


app = FastAPI(title="Carzstore image-tools", version=VERSION, lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Wrong or missing X-API-Key")


@app.exception_handler(T.ToolError)
async def tool_error(_: Request, e: T.ToolError):
    return JSONResponse(status_code=e.status, content={"ok": False, "code": e.code, "message": e.message})


async def read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_MB * 1_048_576:
        raise T.ToolError("E07", f"Upload is larger than {MAX_UPLOAD_MB:.0f} MB")
    return data


MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg", "WEBP": "image/webp"}


def file_entry(im, name: str, fmt: str = "PNG", quality: int = 90) -> dict:
    data = T.encode(im, fmt, quality)
    return {"name": name, "mime": MIME[fmt.upper()], "width": im.width, "height": im.height,
            "bytes": len(data), "b64": base64.b64encode(data).decode()}


def yes(v) -> bool:
    return str(v).strip().lower() not in ("false", "0", "no", "off", "")


def check_sku(sku: str) -> str:
    sku = (sku or "").strip().lower()
    if not sku or len(sku) > W.MAX_SKU or any(c not in W.CHARSET for c in sku):
        raise T.ToolError("E01", "sku is needed for final files (3-40 lowercase letters, numbers or hyphens)")
    return sku


def final_entry(im, name: str, sku: str, alt: str, fmt: str = "WEBP", quality: int = 90,
                watermark: bool = True) -> dict:
    """Final file: invisible watermark + (c) metadata, read back once to prove the mark."""
    data, meta = W.finalize(im, sku, alt, fmt, quality, watermark)
    return {"name": name, "mime": MIME[fmt.upper()], "width": im.width, "height": im.height,
            "bytes": len(data), "b64": base64.b64encode(data).decode(),
            "watermark_ok": meta.get("watermark_ok"), "wm_confidence": meta.get("wm_confidence")}


def ok(files: list[dict], meta: dict, started: float) -> dict:
    meta = {**meta, "ms": int((time.time() - started) * 1000)}
    return {"ok": True, "files": files, "meta": meta}


def wm_summary(files: list[dict]) -> dict:
    marked = [f for f in files if f.get("watermark_ok") is not None]
    weak = [f["name"] for f in marked if not f["watermark_ok"]]
    return {"protected": bool(marked), "watermark_all_ok": bool(marked) and not weak,
            "watermark_weak": weak, "wm_version": W.WM_VERSION}


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "default_cutout_model": T.DEFAULT_MODEL,
            "models_loaded": T.loaded_models(), "watermark": W.WM_VERSION}


# ---------------------------------------------------------------------------
@app.post("/inspect", dependencies=[Depends(check_key)])
async def inspect(file: UploadFile = File(...)):
    """WF1 photo checks: real type, size, resolution, blank, banner-like. HEIC/AVIF -> PNG."""
    t0 = time.time()
    r = T.inspect(await read(file))
    conv = r.pop("converted")
    files = [file_entry(conv, "converted.png")] if conv is not None else []
    return ok(files, r, t0)


@app.post("/cutout", dependencies=[Depends(check_key)])
async def cutout(file: UploadFile = File(...), sku: str = Form("product"),
                 model: Optional[str] = Form(None)):
    """Background removal -> trimmed transparent PNG + cutout check (H01)."""
    t0 = time.time()
    data = await read(file)
    with _heavy:
        res = T.cutout(data, model)
    return ok([file_entry(res.image, f"{sku}-cutout.png")], res.meta, t0)


@app.post("/compose/clean", dependencies=[Depends(check_key)])
async def compose_clean(cutout: UploadFile = File(...), slug: str = Form("product"),
                        size: int = Form(2048)):
    t0 = time.time()
    im, meta = T.compose_clean(await read(cutout), size)
    return ok([file_entry(im, f"{slug}-clean-work.png")], meta, t0)


@app.post("/compose/promo", dependencies=[Depends(check_key)])
async def compose_promo(cutout: UploadFile = File(...), headline: str = Form(...),
                        price: Optional[str] = Form(None), offer: Optional[str] = Form(None),
                        sizes: str = Form("STORE"), slug: str = Form("product"),
                        output: str = Form("png"), sku: str = Form(""), alt: str = Form(""),
                        protect: str = Form("true")):
    """One promo image per size (each size has its own layout, so no cropping).
    output=png for the working copy (WF2), webp for finals (WF4: watermark + (c) metadata,
    needs sku; protect=false skips both)."""
    t0 = time.time()
    data = await read(cutout)
    keys = [s.strip().upper() for s in sizes.split(",") if s.strip()] or ["STORE"]
    fmt = "WEBP" if output.lower() == "webp" else "PNG"
    ext = "webp" if fmt == "WEBP" else "png"
    final = fmt == "WEBP" and yes(protect)
    sku_ok = check_sku(sku) if final else ""
    files, metas = [], []
    for k in keys:
        im, meta = T.compose_promo(data, headline, price, offer, k)
        label = T.SIZE_LABEL.get(k, k.lower())
        name = f"{slug}-promo-{label}.{ext}"
        entry = final_entry(im, name, sku_ok, alt) if final else file_entry(im, name, fmt)
        files.append(entry | {"size": k})
        metas.append(meta)
    return ok(files, {"items": metas, **(wm_summary(files) if final else {})}, t0)


@app.post("/overlay/logo", dependencies=[Depends(check_key)])
async def overlay_logo(file: UploadFile = File(...), name: str = Form("image-logo.png")):
    t0 = time.time()
    im, _ = T.open_image(await read(file))
    return ok([file_entry(T.overlay_logo(im), name)], {}, t0)


@app.post("/export", dependencies=[Depends(check_key)])
async def export(file: UploadFile = File(...), sizes: str = Form("STORE"),
                 slug: str = Form(...), type: str = Form(...),
                 mode: str = Form("crop"), logo: bool = Form(False),
                 format: str = Form("webp"), quality: int = Form(90),
                 sku: str = Form(""), alt: str = Form(""), protect: str = Form("true")):
    """Final files for WF4. type = clean | lifestyle | promo.
    mode: crop (fill the frame; lifestyle) or pad (white letterbox; clean).
    logo=true adds the corner logo AFTER resizing, so it's never cropped off.
    protect=true (default): invisible watermark (Carzstore.pk + sku) + (c) metadata, needs sku."""
    t0 = time.time()
    im, _ = T.open_image(await read(file))
    keys = [s for s in sizes.split(",") if s.strip()] or ["STORE"]
    fmt = format.upper()
    protect = yes(protect)   # a text field: FastAPI reads a missing bool form field as false
    sku_ok = check_sku(sku) if protect else ""
    items = T.export(im, keys, slug, type.lower(), mode, logo, fmt, quality)
    files = []
    for i in items:
        entry = (final_entry(i["image"], i["name"], sku_ok, alt, i["fmt"], i["quality"]) if protect
                 else file_entry(i["image"], i["name"], i["fmt"], i["quality"]))
        files.append(entry | {"size": i["size"]})
    return ok(files, {"count": len(files), **(wm_summary(files) if protect else {})}, t0)


@app.post("/check-ownership", dependencies=[Depends(check_key)])
async def check_ownership(file: UploadFile = File(...)):
    """Is this image ours? Reads the invisible watermark and the (c) metadata.
    Works on re-saved and resized copies; not on cropped or heavily edited ones."""
    t0 = time.time()
    data = await read(file)
    im, fmt = T.open_image(data)
    try:
        meta = W.read_metadata(Image.open(io.BytesIO(data)))
    except Exception:
        meta = {}
    mark = W.read(im)
    meta_ours = W.OWNER in (meta.get("artist", "") + meta.get("copyright", "")) or bool(meta.get("xmp_owner"))
    if mark["found"]:
        verdict = (f"YES - {W.OWNER} image. Hidden mark found: SKU {mark['sku']} "
                   f"(strength {mark['confidence']:.2f}).")
    elif meta_ours:
        verdict = (f"PROBABLY - {W.OWNER} copyright metadata found, but no readable hidden mark "
                   "(image may be cropped or edited).")
    else:
        verdict = (f"NOT FOUND - no {W.OWNER} mark or metadata. The image may not be ours, or it "
                   "was cropped / heavily edited. Compare it with the Drive originals.")
    return ok([], {"owner": W.OWNER if (mark["found"] or meta_ours) else None,
                   "watermark_found": mark["found"], "sku": mark["sku"],
                   "strength": mark["confidence"], "metadata": meta, "verdict": verdict,
                   "image": f"{im.width}x{im.height} {fmt}", "wm_version": W.WM_VERSION}, t0)


@app.post("/review-prep", dependencies=[Depends(check_key)])
async def review_prep(file: UploadFile = File(...), name: str = Form("review.jpg"),
                      longest: int = Form(1024)):
    """Resize to 1024 px JPEG q90 before sending to the review model (05 section 2)."""
    t0 = time.time()
    im, _ = T.open_image(await read(file))
    out = T.review_prep(im, longest)
    return ok([file_entry(out, name, "JPEG", 90)], {}, t0)


@app.post("/golden", dependencies=[Depends(check_key)])
async def golden(file: UploadFile = File(...), extra_cutout: Optional[UploadFile] = File(None),
                 slug: str = Form("golden")):
    """Broken copies of a GOOD lifestyle image to test the reviewer (05 section 8a)."""
    t0 = time.time()
    good, _ = T.open_image(await read(file))
    extra = None
    if extra_cutout is not None:
        extra, _ = T.open_image(await read(extra_cutout))
    items = T.golden_set(good, extra)
    files = [file_entry(im, f"{slug}-{kind}.jpg", "JPEG", 92) | {"kind": kind, "expected": exp}
             for kind, exp, im in items]
    return ok(files, {"count": len(files)}, t0)
