"""
Download pretrained model weights required by the pipeline.
Run: python scripts/download_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Model registry: name -> (download_url, notes)
MODELS = {
    "inswapper_128.onnx": (
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "InsightFace face-swap model (required)"
    ),
}

# Note: BiSeNet and RetinaFace weights are downloaded automatically by
# their respective Python packages on first use. Only the InsightFace
# inswapper model requires a manual download.


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:.1f}%")
        sys.stdout.flush()


def download_model(name: str, url: str, notes: str):
    dest = MODELS_DIR / name
    if dest.exists():
        print(f"  [OK] {name} already exists - skipping.")
        return

    print(f"\nDownloading {name}  ({notes})")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print(f"\n  Saved -> {dest}")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(
            f"  Please download manually and place at: {dest}\n"
            f"  URL: {url}"
        )


if __name__ == "__main__":
    print("Face Swap - Model Downloader")
    print("=" * 40)
    for model_name, (url, notes) in MODELS.items():
        download_model(model_name, url, notes)

    print("\nDone. Place any additional model files in the models/ directory.")
    print("See the README for the full list of supported models.")
