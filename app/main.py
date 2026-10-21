"""image-tools HTTP API for the Carzstore n8n workflows.

Every image endpoint takes multipart form data and returns JSON:
  {"ok": true, "files": [{"name", "mime", "width", "height", "b64"}], "meta": {...}}
Known problems return HTTP 422 with:
  {"ok": false, "code": "E08", "message": "..."}      (codes from 03 section 8)
n8n turns "b64" into a binary file with one Code node line.
"""
from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import imaging as T

VERSION = "1.0.0"
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


def file_entry(im, name: str, fmt: str = "PNG", quality: int = 90) -> dict:
    data = T.encode(im, fmt, quality)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg",
            "WEBP": "image/webp"}[fmt.upper()]
    return {"name": name, "mime": mime, "width": im.width, "height": im.height,
            "bytes": len(data), "b64": base64.b64encode(data).decode()}


def ok(files: list[dict], meta: dict, started: float) -> dict:
    meta = {**meta, "ms": int((time.time() - started) * 1000)}
    return {"ok": True, "files": files, "meta": meta}


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "default_cutout_model": T.DEFAULT_MODEL,
            "models_loaded": T.loaded_models()}


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
                        output: str = Form("png")):
    """One promo image per size (each size has its own layout, so no cropping).
    output=png for the working copy (WF2), webp for finals (WF4)."""
    t0 = time.time()
    data = await read(cutout)
    keys = [s.strip().upper() for s in sizes.split(",") if s.strip()] or ["STORE"]
    fmt = "WEBP" if output.lower() == "webp" else "PNG"
    ext = "webp" if fmt == "WEBP" else "png"
    files, metas = [], []
    for k in keys:
        im, meta = T.compose_promo(data, headline, price, offer, k)
        label = T.SIZE_LABEL.get(k, k.lower())
        files.append(file_entry(im, f"{slug}-promo-{label}.{ext}", fmt))
        metas.append(meta)
    return ok(files, {"items": metas}, t0)


@app.post("/overlay/logo", dependencies=[Depends(check_key)])
async def overlay_logo(file: UploadFile = File(...), name: str = Form("image-logo.png")):
    t0 = time.time()
    im, _ = T.open_image(await read(file))
    return ok([file_entry(T.overlay_logo(im), name)], {}, t0)


@app.post("/export", dependencies=[Depends(check_key)])
async def export(file: UploadFile = File(...), sizes: str = Form("STORE"),
                 slug: str = Form(...), type: str = Form(...),
                 mode: str = Form("crop"), logo: bool = Form(False),
                 format: str = Form("webp"), quality: int = Form(90)):
    """Final files for WF4. type = clean | lifestyle | promo.
    mode: crop (fill the frame; lifestyle) or pad (white letterbox; clean).
    logo=true adds the corner logo AFTER resizing, so it's never cropped off."""
    t0 = time.time()
    im, _ = T.open_image(await read(file))
    keys = [s for s in sizes.split(",") if s.strip()] or ["STORE"]
    fmt = format.upper()
    items = T.export(im, keys, slug, type.lower(), mode, logo, fmt, quality)
    files = [file_entry(i["image"], i["name"], i["fmt"], i["quality"]) | {"size": i["size"]}
             for i in items]
    return ok(files, {"count": len(files)}, t0)


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
