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
            # If target has glasses, restore that region from original target
            result = _restore_glasses_region(result, target, tgt_face)

        # Sharpen the swapped face region to recover detail lost in 128x128 internal resize
        result = _sharpen_face_region(result, tgt_faces)
        return result
    except Exception as e:
        print(f"[swapper] swap error: {e}")
        return _fallback_swap(source, target)


def _restore_glasses_region(swapped: np.ndarray, original_target: np.ndarray, face) -> np.ndarray:
    """
    Restore the glasses/eye region from the original target so spectacles
    (frames + lenses) are preserved cleanly regardless of the source face.
    Uses InsightFace 5-keypoint or 68/106-landmark eye positions.
    """
    h, w = swapped.shape[:2]
    result = swapped.copy()

    try:
        # Get eye centre positions from keypoints [0]=left eye, [1]=right eye
        kps = face.kps  # shape (5,2): left_eye, right_eye, nose, left_mouth, right_mouth
        le, re = kps[0], kps[1]

        # Estimate glasses bounding box: wide enough to cover frames + nose bridge
        eye_dist   = float(np.linalg.norm(re - le))
        cx         = int((le[0] + re[0]) / 2)
        cy         = int((le[1] + re[1]) / 2)
        half_w     = int(eye_dist * 0.80)
        half_h     = int(eye_dist * 0.38)

        gx1 = max(0, cx - half_w);  gx2 = min(w, cx + half_w)
        gy1 = max(0, cy - half_h);  gy2 = min(h, cy + half_h)

        if gx2 <= gx1 or gy2 <= gy1:
            return result

        rh, rw = gy2 - gy1, gx2 - gx1

        # Feathered blend: original target fully in centre, fade to swapped at edges
        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rw * 0.12)
        mask = np.stack([mask] * 3, axis=-1)

        tgt_crop = original_target[gy1:gy2, gx1:gx2].astype(np.float32)
        swp_crop = swapped[gy1:gy2, gx1:gx2].astype(np.float32)
        blended  = tgt_crop * mask + swp_crop * (1 - mask)

        result[gy1:gy2, gx1:gx2] = np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        pass  # no landmarks → skip silently

    return result


def _sharpen_face_region(image: np.ndarray, tgt_faces: list) -> np.ndarray:
    """
    Recover detail lost in InsightFace's 128x128 internal resize.
    Strategy:
      - Mild bilateral filter to remove compression artefacts (not texture).
      - Unsharp mask on the ORIGINAL crop (not the blurred version) for true
        high-frequency recovery; 2.3x strength gives clean edges without halos.
      - Blend sharp + smooth so skin stays natural while edges are crisp.
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

        # 1. Mild bilateral — removes compression blotches while keeping edges
        smooth = cv2.bilateralFilter(crop, d=5, sigmaColor=30, sigmaSpace=30)

        # 2. Unsharp mask on ORIGINAL crop — true detail recovery
        blur  = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.5)
        sharp = cv2.addWeighted(crop, 2.3, blur, -1.3, 0)

        # 3. Composite: 55% sharp detail + 45% smooth skin base
        enhanced = cv2.addWeighted(sharp, 0.55, smooth, 0.45, 0)

        # 4. Feathered paste — no hard border at crop edges
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
            enhanced.astype(np.float32) * feather +
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
