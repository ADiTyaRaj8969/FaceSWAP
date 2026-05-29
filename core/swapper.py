import os
import cv2
import cv2.data
import numpy as np

from .detector import _get_insightface, _ORT_PROVIDERS  # reuse shared instance + providers

_swapper_model = None
_swapper_failed = False   # True only after a real load error (not "file not there yet")


def _load_swapper():
    global _swapper_model, _swapper_failed

    if _swapper_model is not None:
        return _swapper_model          # already loaded

    if _swapper_failed:
        return None                    # previously errored — don't retry

    model_path = "models/inswapper_128.onnx"
    if not os.path.exists(model_path):
        return None                    # not downloaded yet — retry next call

    try:
        import insightface
        _swapper_model = insightface.model_zoo.get_model(
            model_path, providers=_ORT_PROVIDERS
        )
        print("[swapper] inswapper_128 loaded OK")
        return _swapper_model
    except Exception as e:
        print(f"[swapper] load failed: {e}")
        _swapper_failed = True
        return None


def swap_face_insightface(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    swapper = _load_swapper()
    app     = _get_insightface()

    if swapper is None or app is None:
        return _fallback_swap(source, target)

    try:
        src_faces = app.get(source)   # InsightFace expects BGR
        tgt_faces = app.get(target)

        if not src_faces or not tgt_faces:
            print(f"[swapper] faces not detected: src={len(src_faces)} tgt={len(tgt_faces)}")
            return _fallback_swap(source, target)

        result   = target.copy()
        src_face = src_faces[0]
        for tgt_face in tgt_faces:
            result = swapper.get(result, tgt_face, src_face, paste_back=True)

        # Sharpen the swapped face region to recover detail lost in 128x128 internal resize
        result = _sharpen_face_region(result, tgt_faces)
        return result
    except Exception as e:
        print(f"[swapper] swap error: {e}")
        return _fallback_swap(source, target)


def _sharpen_face_region(image: np.ndarray, tgt_faces: list) -> np.ndarray:
    """
    Sharpen + even skin tone in each swapped face region.
    - Unsharp mask restores detail lost in InsightFace's 128x128 internal resize.
    - Bilateral filter smooths out skin blotchiness while preserving edges.
    - Colour correction at the face boundary blends the two skin tones.
    """
    result = image.copy()
    h, w   = image.shape[:2]

    for face in tgt_faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        pad = int((x2 - x1) * 0.15)
        x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad);  y2p = min(h, y2 + pad)

        crop = result[y1p:y2p, x1p:x2p].copy()
        if crop.size == 0:
            continue

        # 1. Even skin tone — bilateral filter (smooth blotches, keep edges)
        smooth = cv2.bilateralFilter(crop, d=9, sigmaColor=60, sigmaSpace=60)

        # 2. Sharpen — unsharp mask on top of smoothed
        blur   = cv2.GaussianBlur(smooth, (0, 0), sigmaX=2.5)
        sharp  = cv2.addWeighted(smooth, 1.6, blur, -0.6, 0)

        # 3. Feathered paste — fade near the crop edges so no hard border
        fh, fw = crop.shape[:2]
        feather = np.ones((fh, fw), dtype=np.float32)
        border  = max(4, pad // 2)
        for i in range(border):
            v = (i + 1) / (border + 1)
            feather[i, :]      *= v
            feather[fh-1-i, :] *= v
            feather[:, i]      *= v
            feather[:, fw-1-i] *= v
        feather = np.stack([feather] * 3, axis=-1)

        result[y1p:y2p, x1p:x2p] = (
            sharp.astype(np.float32) * feather +
            result[y1p:y2p, x1p:x2p].astype(np.float32) * (1 - feather)
        ).astype(np.uint8)

    return result


def _fallback_swap(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces_src = cascade.detectMultiScale(gray_src, 1.1, 5, minSize=(64, 64))
    faces_tgt = cascade.detectMultiScale(gray_tgt, 1.1, 5, minSize=(64, 64))

    if len(faces_src) == 0 or len(faces_tgt) == 0:
        return target.copy()

    sx, sy, sw, sh = faces_src[0]
    tx, ty, tw, th = faces_tgt[0]

    face_patch         = source[sy:sy+sh, sx:sx+sw]
    face_patch_resized = cv2.resize(face_patch, (tw, th))
    result             = target.copy()

    mask   = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask, (tw // 2, th // 2), (tw // 2, th // 2), 0, 0, 360, 255, -1)
    center = (tx + tw // 2, ty + th // 2)

    try:
        result = cv2.seamlessClone(face_patch_resized, result, mask, center, cv2.NORMAL_CLONE)
    except Exception:
        result[ty:ty+th, tx:tx+tw] = (
            face_patch_resized * 0.7 + result[ty:ty+th, tx:tx+tw] * 0.3
        ).astype(np.uint8)

    return result
