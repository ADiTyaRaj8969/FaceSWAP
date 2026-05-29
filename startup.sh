#!/bin/bash
set -e

echo ""
echo "======================================"
echo "  DeepFace Studio — Starting Up"
echo "======================================"
echo ""

# Download models in background so Flask starts immediately
# (HF Spaces kills containers that don't respond within ~60s)
(
  MODEL_PATH="models/inswapper_128.onnx"
  mkdir -p models

  python3 - <<'PYEOF'
import requests, os

# (filename, url) — inswapper is required; GFPGAN restores the 128x128 swap
# blur on CPU. RealESRGAN is skipped on the CPU deploy (super_res uses Lanczos
# there), so we don't waste startup time/disk downloading it.
MODELS = [
    ("models/inswapper_128.onnx",
     "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"),
    ("models/GFPGANv1.4.pth",
     "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"),
]
os.makedirs("models", exist_ok=True)

for dest, url in MODELS:
    name = os.path.basename(dest)
    if os.path.exists(dest):
        print(f"[bg] {name} already present.")
        continue
    tmp = dest + ".tmp"
    try:
        print(f"[bg] Downloading {name} ...")
        resp  = requests.get(url, stream=True, timeout=600)
        total = int(resp.headers.get("content-length", 0))
        done  = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    mb  = done // 1024 // 1024
                    print(f"\r[bg]   {name}: {mb}/{total//1024//1024} MB ({pct}%)", end="", flush=True)
        # verify completeness — a truncated file would crash model loading later
        if total and os.path.getsize(tmp) < total:
            raise IOError(f"truncated: {os.path.getsize(tmp)}/{total} bytes")
        os.replace(tmp, dest)
        print(f"\n[bg] {name} ready.")
    except Exception as e:
        print(f"\n[bg] WARNING: {name} download failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
PYEOF

  echo "[bg] Pre-loading InsightFace buffalo_l..."
  python3 - <<'PYEOF'
try:
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[bg] buffalo_l ready.")
except Exception as e:
    print(f"[bg] WARNING: buffalo_l pre-load failed: {e}")
PYEOF
) &

echo "Starting Flask server on port ${PORT:-7860}..."
echo ""
exec python web_app.py
