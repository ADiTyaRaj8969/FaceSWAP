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
                  realesrgan_weight: float = 0.5,
                  focus_bbox=None) -> np.ndarray:
    """
    Upscale ~scale x for the high-resolution download.

    RealESRGAN adds resolution but, being a general-purpose model, it over-sharpens
    skin into a plastic/waxy texture on faces. So we BLEND it with a plain Lanczos
    upscale (realesrgan_weight controls the mix: 0 = pure natural Lanczos, 1 = full
    RealESRGAN) — keeping most of the detail while killing the artificial texture.

    focus_bbox restricts RealESRGAN to the head region. Everything outside it is
    now composited from the untouched full-resolution target (see
    utils.image_io.composite_onto_original), so only the head — which came
    through the swap's 128x128 bottleneck — has detail worth reconstructing.
    That matters most on CPU: measured on this pipeline a full 1.57MP frame
    takes ~20 minutes, while the head crop alone is seconds, which is what makes
    real super-resolution affordable on the free CPU tier instead of GPU-only.
    Passing no bbox on CPU keeps the old Lanczos-only behaviour.
    """
    h, w = image.shape[:2]
    lanczos = cv2.resize(image, (w * scale, h * scale),
                         interpolation=cv2.INTER_LANCZOS4)

    on_gpu = _device() == "cuda"
    if not on_gpu and focus_bbox is None:
        print(f"[super_res] Lanczos {scale}x -> {w*scale}x{h*scale}")
        return lanczos

    upsampler = _load_realesrgan()
    if upsampler is None:
        print(f"[super_res] Lanczos {scale}x -> {w*scale}x{h*scale}")
        return lanczos

    try:
        if focus_bbox is None:
            output, _ = upsampler.enhance(image, outscale=scale)
            if output is None:
                return lanczos
            if output.shape[:2] != lanczos.shape[:2]:
                output = cv2.resize(output, (lanczos.shape[1], lanczos.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)
            blended = cv2.addWeighted(output, realesrgan_weight,
                                      lanczos, 1.0 - realesrgan_weight, 0)
            print(f"[super_res] RealESRGAN+Lanczos {scale}x -> "
                  f"{blended.shape[1]}x{blended.shape[0]} (re={realesrgan_weight})")
            return blended

        # -- head-only super-resolution ------------------------------------
        x1, y1, x2, y2 = [int(v) for v in focus_bbox]
        bw, bh = x2 - x1, y2 - y1
        # Generous margin so hair and jaw are reconstructed with the face, and
        # the feathered seam lands on background rather than on skin.
        cx1 = max(0, x1 - int(bw * 1.0)); cy1 = max(0, y1 - int(bh * 1.2))
        cx2 = min(w, x2 + int(bw * 1.0)); cy2 = min(h, y2 + int(bh * 1.6))
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return lanczos

        out_crop, _ = upsampler.enhance(crop, outscale=scale)
        if out_crop is None:
            return lanczos
        tw, th = (cx2 - cx1) * scale, (cy2 - cy1) * scale
        if (out_crop.shape[1], out_crop.shape[0]) != (tw, th):
            out_crop = cv2.resize(out_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        region = lanczos[cy1 * scale:cy2 * scale, cx1 * scale:cx2 * scale]
        mixed = cv2.addWeighted(out_crop, realesrgan_weight,
                                region, 1.0 - realesrgan_weight, 0)

        # Feather the crop edge so the enhanced head doesn't end on a hard line.
        m = np.zeros((th, tw), np.float32)
        pad = max(2, int(min(th, tw) * 0.06))
        m[pad:th - pad, pad:tw - pad] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), pad / 2.0)[..., None]

        out = lanczos.copy()
        out[cy1 * scale:cy2 * scale, cx1 * scale:cx2 * scale] = (
            mixed.astype(np.float32) * m + region.astype(np.float32) * (1.0 - m)
        ).astype(np.uint8)
        print(f"[super_res] RealESRGAN on head {crop.shape[1]}x{crop.shape[0]} "
              f"+ Lanczos {scale}x -> {out.shape[1]}x{out.shape[0]}")
        return out
    except Exception as e:
        print(f"[super_res] RealESRGAN enhance failed: {e}")
        return lanczos


def enhance_resolution(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """Convenience: GFPGAN face restore, then RealESRGAN upscale to ~4K."""
    return upscale_image(restore_faces(image), scale=scale)
