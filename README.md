# Carzstore image-tools

Small private web service that does all the **no-AI** image work for the Carzstore n8n workflows:
cutout (background removal), clean product image, promo image, logo overlay, final sizes, review-prep and the golden test set.

Runs as one Docker container next to n8n on the Hostinger VPS. Not open to the internet — only n8n calls it at `http://image-tools:8000`.

## How it gets to the VPS (no terminal)
1. Push to `main` → GitHub Actions builds the image → `ghcr.io/<github-name>/carzstore-image-tools:latest`.
2. Hostinger hPanel → Docker Manager → Compose manually → paste `docker-compose.yml` → Deploy.
3. Code change later: push to GitHub, wait for the green tick in **Actions**, then Docker Manager → image-tools → **Update**.

## Endpoints
All `POST` endpoints take multipart form data and need header `X-API-Key`.
Success: `{"ok": true, "files": [{"name","mime","width","height","bytes","b64"}], "meta": {...}}`
Known problem: HTTP 422 `{"ok": false, "code": "E08", "message": "..."}` (codes from 03_SHEETS_SCHEMA section 8).

| Endpoint | Form fields | Returns |
|---|---|---|
| `GET /health` | — | status, version, loaded models |
| `/inspect` | `file` | format, size, flags (`LOW_RES`, `CONVERTED_FORMAT`), `looks_like_banner`; PNG if HEIC/AVIF was converted. Errors E05, E07, E08, E09, H04 |
| `/cutout` | `file`, `sku`, `model` (opt.) | trimmed transparent PNG + `cutout_fill_pct`, `check_ok`, `problems` (H01) |
| `/compose/clean` | `cutout`, `slug` | 2048 white image, product 80%, soft shadow (04 s9) |
| `/compose/promo` | `cutout`, `headline`, `price`, `offer`, `sizes` (STORE,POST,STORY,BANNER), `slug`, `output` (png/webp) | one promo per size (04 s10) |
| `/overlay/logo` | `file` | logo bottom-right, 14% width (04 s11) |
| `/export` | `file`, `sizes`, `slug`, `type`, `mode` (crop/pad), `logo` (true/false), `format` | final WEBP files with SEO names |
| `/review-prep` | `file` | JPEG, longest side 1024 (05 s2) |
| `/golden` | `file`, `extra_cutout` (opt.) | 8 test copies (05 s8a) |

## Settings (environment variables)
`API_KEY`, `CUTOUT_MODEL` (default `isnet-general-use`), `HEAVY_JOBS` (1), `MAX_FILE_MB` (20), `MIN_SHORT_SIDE_PX` (800), `LOW_RES_BELOW_PX` (1200), `CUTOUT_MIN_FILL_PCT` (3), `CUTOUT_MAX_HOLE_RATIO` (0.5), `LOGO_PATH`.

## Notes
- birefnet cutout models were tested and need more than 7 GB RAM per photo on CPU, so only `isnet-general-use` (≈1.2 GB, ≈1.5 s) and `u2net` are allowed.
- To swap in the HD logo: replace `assets/logo.png` and push. A transparent PNG is used without the white card when `LOGO_CARD=off`.
- Fonts: Poppins (SIL Open Font License), in `assets/fonts`.
