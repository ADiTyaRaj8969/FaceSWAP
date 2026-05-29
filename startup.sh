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

  if [ ! -f "$MODEL_PATH" ]; then
    echo "[bg] Downloading inswapper_128.onnx (~528 MB)..."
    python3 - <<'PYEOF'
import requests, os

url  = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
dest = "models/inswapper_128.onnx"
tmp  = dest + ".tmp"

try:
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
                print(f"\r[bg]   {mb}/{total//1024//1024} MB ({pct}%)", end="", flush=True)
    os.replace(tmp, dest)
    print("\n[bg] inswapper_128.onnx ready.")
except Exception as e:
    print(f"\n[bg] WARNING: model download failed: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
PYEOF
  else
    echo "[bg] inswapper_128.onnx already present."
  fi

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
