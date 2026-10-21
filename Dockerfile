# Carzstore image-tools — built automatically by GitHub Actions (see .github/workflows/build.yml)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REMBG_HOME=/models \
    OMP_NUM_THREADS=2

# libgomp: needed by onnxruntime; curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the cutout model into the image (~180 MB) so the first photo doesn't wait for a download
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY app ./app
COPY assets ./assets

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fs http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
