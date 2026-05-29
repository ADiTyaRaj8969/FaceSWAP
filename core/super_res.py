import cv2
import numpy as np
import os

_realesrgan_instance = None
_gfpgan_instance = None

MODELS_DIR = "models"
REALESRGAN_MODEL = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")
GFPGAN_MODEL     = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")


def _load_realesrgan():
    global _realesrgan_instance
    if _realesrgan_instance is not None:
        return _realesrgan_instance
    if not os.path.exists(REALESRGAN_MODEL):
        print(f"[super_res] RealESRGAN model not found: {REALESRGAN_MODEL}")
        return None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        _realesrgan_instance = RealESRGANer(
            scale=4,
            model_path=REALESRGAN_MODEL,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=False,  # CPU-safe — no float16
        )
        print("[super_res] RealESRGAN x4 loaded OK")
        return _realesrgan_instance
    except Exception as e:
        print(f"[super_res] RealESRGAN load failed: {e}")
        return None


def _load_gfpgan():
    global _gfpgan_instance
    if _gfpgan_instance is not None:
        return _gfpgan_instance
    if not os.path.exists(GFPGAN_MODEL):
        print(f"[super_res] GFPGAN model not found: {GFPGAN_MODEL}")
        return None
    try:
        from gfpgan import GFPGANer

        _gfpgan_instance = GFPGANer(
            model_path=GFPGAN_MODEL,
            upscale=1,          # restore only — upscaling done by RealESRGAN
            arch="clean",
            channel_multiplier=2,
        )
        print("[super_res] GFPGAN loaded OK")
        return _gfpgan_instance
    except Exception as e:
        print(f"[super_res] GFPGAN load failed: {e}")
        return None


def enhance_resolution(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Two-stage quality enhancement:
      1. GFPGAN — restores face detail lost in InsightFace's 128x128 resize
         (no resolution change, just sharpness/texture restoration on faces).
      2. RealESRGAN x4 — upscales the whole image to ~4K.
    Falls back to Lanczos resize when models are not downloaded.

    scale: 2 or 4 (4 = ~4K from a 1024px input)
    """
    result = image.copy()

    # --- Stage 1: face restoration ----------------------------------------
    gfpgan = _load_gfpgan()
    if gfpgan is not None:
        try:
            _, _, restored = gfpgan.enhance(
                result,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            if restored is not None and restored.shape == result.shape:
                result = restored
                print("[super_res] GFPGAN face restoration applied")
        except Exception as e:
            print(f"[super_res] GFPGAN enhance failed: {e}")

    # --- Stage 2: full-image upscaling ------------------------------------
    upsampler = _load_realesrgan()
    if upsampler is not None:
        try:
            output, _ = upsampler.enhance(result, outscale=scale)
            if output is not None:
                h, w = output.shape[:2]
                print(f"[super_res] RealESRGAN {scale}x done → {w}×{h}")
                return output
        except Exception as e:
            print(f"[super_res] RealESRGAN enhance failed: {e}")

    # --- Fallback: Lanczos resize -----------------------------------------
    h, w = result.shape[:2]
    tw, th = w * scale, h * scale
    print(f"[super_res] Lanczos fallback {scale}x: {w}×{h} → {tw}×{th}")
    return cv2.resize(result, (tw, th), interpolation=cv2.INTER_LANCZOS4)
