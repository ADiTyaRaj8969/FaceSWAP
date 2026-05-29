import cv2
import numpy as np
import os

from core.detector import _get_insightface  # reuse shared buffalo_l instance

_swapper_model = None
_swapper_load_attempted = False


def _load_swapper():
    global _swapper_model, _swapper_load_attempted
    if _swapper_load_attempted:
        return _swapper_model
    _swapper_load_attempted = True

    model_path = "models/inswapper_128.onnx"
    if not os.path.exists(model_path):
        print("[swapper] inswapper_128.onnx not found")
        return None

    try:
        import insightface
        _swapper_model = insightface.model_zoo.get_model(
            model_path, providers=["CPUExecutionProvider"]
        )
        print("[swapper] inswapper_128 loaded OK")
        return _swapper_model
    except Exception as e:
        print(f"[swapper] load_swapper failed: {e}")
        return None


def swap_face_insightface(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Swap the face from source into the target image using InsightFace inswapper.
    Returns the target image with the source face inserted.
    Falls back to a simple alpha-blend paste if the model is unavailable.
    """
    swapper = _load_swapper()
    app = _get_insightface()

    if swapper is None or app is None:
        return _fallback_swap(source, target)

    try:
        # InsightFace expects BGR (OpenCV native) — do NOT convert to RGB
        src_faces = app.get(source)
        tgt_faces = app.get(target)

        if not src_faces or not tgt_faces:
            print(f"[swapper] face detection failed: src={len(src_faces)} tgt={len(tgt_faces)}")
            return _fallback_swap(source, target)

        result = target.copy()
        src_face = src_faces[0]
        for tgt_face in tgt_faces:
            result = swapper.get(result, tgt_face, src_face, paste_back=True)

        return result
    except Exception as e:
        print(f"[swapper] swap failed: {e}")
        return _fallback_swap(source, target)


def _fallback_swap(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Minimal fallback: detect face in source, warp it onto detected face in target.
    Uses OpenCV seamlessClone for blending.
    """
    gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces_src = cascade.detectMultiScale(gray_src, 1.1, 5, minSize=(64, 64))
    faces_tgt = cascade.detectMultiScale(gray_tgt, 1.1, 5, minSize=(64, 64))

    if len(faces_src) == 0 or len(faces_tgt) == 0:
        return target.copy()

    sx, sy, sw, sh = faces_src[0]
    tx, ty, tw, th = faces_tgt[0]

    face_patch = source[sy:sy+sh, sx:sx+sw]
    face_patch_resized = cv2.resize(face_patch, (tw, th))

    result = target.copy()

    # Create elliptical mask for the face patch
    mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask, (tw // 2, th // 2), (tw // 2, th // 2), 0, 0, 360, 255, -1)

    center = (tx + tw // 2, ty + th // 2)
    try:
        result = cv2.seamlessClone(
            face_patch_resized, result, mask, center, cv2.NORMAL_CLONE
        )
    except Exception:
        # Direct paste as last resort
        result[ty:ty+th, tx:tx+tw] = (
            face_patch_resized * 0.7 + result[ty:ty+th, tx:tx+tw] * 0.3
        ).astype(np.uint8)

    return result
