"""
Face restoration + super-resolution.

InsightFace swaps at 128x128 internally, so the pasted face is soft. GFPGAN
restores that lost facial detail; RealESRGAN upscales the whole frame to ~4K.

basicsr (a dependency of gfpgan/realesrgan) imports
`torchvision.transforms.functional_tensor`, which was removed in torchvision
0.17+. We shim it back before importing those packages so they load on modern
torchvision.
"""
import os
import sys
import types

import cv2
import numpy as np

# -- torchvision.functional_tensor shim (must run before basicsr import) -------
# torchvision >=0.17 removed this submodule; basicsr still imports it. Recreate
# it from the current functional API. Static checkers flag the missing module —
# that's expected; the shim exists only at runtime.
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        import torchvision.transforms.functional as _tvF  # type: ignore
        _shim = types.ModuleType("torchvision.transforms.functional_tensor")
        setattr(_shim, "rgb_to_grayscale", _tvF.rgb_to_grayscale)
        sys.modules["torchvision.transforms.functional_tensor"] = _shim
    except Exception:
        pass


_gfpgan_instance = None
_realesrgan_instance = None
_gfpgan_failed = False
_realesrgan_failed = False

MODELS_DIR       = "models"
REALESRGAN_MODEL = os.path.join(MODELS_DIR, "RealESRGAN_x4plus.pth")
GFPGAN_MODEL     = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -- model loaders -------------------------------------------------------------

def _load_gfpgan():
    global _gfpgan_instance, _gfpgan_failed
    if _gfpgan_instance is not None:
        return _gfpgan_instance
    if _gfpgan_failed or not os.path.exists(GFPGAN_MODEL):
        return None
    try:
        from gfpgan import GFPGANer
        _gfpgan_instance = GFPGANer(
            model_path=GFPGAN_MODEL,
            upscale=1,            # restore detail only; upscaling done by RealESRGAN
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=_device(),
        )
        print(f"[super_res] GFPGAN loaded OK ({_device()})")
        return _gfpgan_instance
    except Exception as e:
        print(f"[super_res] GFPGAN load failed: {e}")
        _gfpgan_failed = True
        return None


def _load_realesrgan():
    global _realesrgan_instance, _realesrgan_failed
    if _realesrgan_instance is not None:
        return _realesrgan_instance
    if _realesrgan_failed or not os.path.exists(REALESRGAN_MODEL):
        return None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        use_gpu = _device() == "cuda"
        _realesrgan_instance = RealESRGANer(
            scale=4,
            model_path=REALESRGAN_MODEL,
            model=model,
            tile=400,             # tiled inference avoids GPU/CPU OOM on big frames
            tile_pad=10,
            pre_pad=0,
            half=use_gpu,         # fp16 on GPU for speed
        )
        print(f"[super_res] RealESRGAN x4 loaded OK ({_device()})")
        return _realesrgan_instance
    except Exception as e:
        print(f"[super_res] RealESRGAN load failed: {e}")
        _realesrgan_failed = True
        return None


# -- public API ----------------------------------------------------------------

def restore_faces(image: np.ndarray) -> np.ndarray:
    """
    Restore facial detail lost in the 128x128 swap using GFPGAN.
    Same resolution in/out — GFPGAN detects faces, restores them, and pastes
    them back seamlessly. Returns the original image unchanged if GFPGAN is
    unavailable. This is the key step that fixes blur, so it runs in the main
    swap path (affects the on-screen preview, not just the download).
    """
    gfpgan = _load_gfpgan()
    if gfpgan is None:
        return image
    try:
        _, _, restored = gfpgan.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        if restored is not None and restored.shape == image.shape:
            return restored
    except Exception as e:
        print(f"[super_res] GFPGAN enhance failed: {e}")
    return image


def upscale_image(image: np.ndarray, scale: int = 4,
                  realesrgan_weight: float = 0.5) -> np.ndarray:
    """
    Upscale the whole frame ~scale x for the high-resolution download.

    RealESRGAN adds resolution but, being a general-purpose model, it over-sharpens
    skin into a plastic/waxy texture on faces. So we BLEND it with a plain Lanczos
    upscale (realesrgan_weight controls the mix: 0 = pure natural Lanczos, 1 = full
    RealESRGAN) — keeping most of the detail while killing the artificial texture.

    RealESRGAN is only used on GPU — on CPU (e.g. the HuggingFace free tier) a 4x
    pass is too slow, so we fall back to pure Lanczos there.
    """
    h, w = image.shape[:2]
    lanczos = cv2.resize(image, (w * scale, h * scale),
                         interpolation=cv2.INTER_LANCZOS4)

    if _device() == "cuda":
        upsampler = _load_realesrgan()
        if upsampler is not None:
            try:
                output, _ = upsampler.enhance(image, outscale=scale)
                if output is not None:
                    if output.shape[:2] != lanczos.shape[:2]:
                        output = cv2.resize(output, (lanczos.shape[1], lanczos.shape[0]),
                                            interpolation=cv2.INTER_LANCZOS4)
                    blended = cv2.addWeighted(output, realesrgan_weight,
                                              lanczos, 1.0 - realesrgan_weight, 0)
                    print(f"[super_res] RealESRGAN+Lanczos {scale}x -> "
                          f"{blended.shape[1]}x{blended.shape[0]} (re={realesrgan_weight})")
                    return blended
            except Exception as e:
                print(f"[super_res] RealESRGAN enhance failed: {e}")

    print(f"[super_res] Lanczos {scale}x -> {w*scale}x{h*scale}")
    return lanczos


def enhance_resolution(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """Convenience: GFPGAN face restore, then RealESRGAN upscale to ~4K."""
    return upscale_image(restore_faces(image), scale=scale)
