#!/bin/bash
set -e

echo ""
echo "======================================"
echo "  DeepFace Studio — Starting Up"
echo "======================================"
echo ""

# Download inswapper_128.onnx if not already present
MODEL_PATH="models/inswapper_128.onnx"
MODEL_URL="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
MODEL_SIZE_MB=528

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading face swap model (~${MODEL_SIZE_MB} MB)..."
    mkdir -p models
    python3 - <<'PYEOF'
import os, requests, sys

url  = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
dest = "models/inswapper_128.onnx"

resp  = requests.get(url, stream=True, timeout=300)
total = int(resp.headers.get("content-length", 0))
done  = 0

with open(dest, "wb") as f:
    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
        f.write(chunk)
        done += len(chunk)
        if total:
            pct = done * 100 // total
            mb  = done // 1024 // 1024
            print(f"\r  {mb} / {total // 1024 // 1024} MB  ({pct}%)", end="", flush=True)

print("\n  Model downloaded successfully.")
PYEOF
else
    echo "Model already present — skipping download."
fi

echo ""
echo "Pre-downloading InsightFace buffalo_l models..."
python3 - <<'PYEOF'
try:
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("  buffalo_l models ready.")
except Exception as e:
    print(f"  Warning: could not pre-load buffalo_l: {e}")
PYEOF

echo ""
echo "Starting Flask server on port ${PORT:-7860}..."
echo ""

exec python web_app.py
