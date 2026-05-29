"""
Download pretrained model weights required by the pipeline.
Run: python scripts/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Model registry: name -> (download_url, notes)
MODELS = {
    "inswapper_128.onnx": (
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "InsightFace face-swap model (required)",
    ),
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN x4 upscaler — enables 4K output (optional but recommended)",
    ),
    "GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "GFPGAN face restoration — fixes InsightFace blur (optional but recommended)",
    ),
}

# Note: BiSeNet and RetinaFace weights are downloaded automatically by
# their respective Python packages on first use. Only the models listed
# above require a manual download.


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
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        _, headers = urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        # Verify the download is complete — urlretrieve can report success on a
        # connection that closed mid-stream, leaving a truncated file.
        expected = int(headers.get("Content-Length", 0))
        actual = tmp.stat().st_size
        if expected and actual < expected:
            tmp.unlink(missing_ok=True)
            raise IOError(
                f"truncated download: got {actual} bytes, expected {expected}"
            )
        tmp.replace(dest)
        print(f"\n  Saved -> {dest}  ({actual/1e6:.0f} MB)")
    except Exception as e:
        tmp.unlink(missing_ok=True)
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
