import os
import cv2
import cv2.data
import numpy as np
try:
    import onnxruntime as ort
    _ORT_PROVIDERS = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
except ImportError:
    _ORT_PROVIDERS = ["CPUExecutionProvider"]

_face_cascade = None
_insightface_app = None
_insightface_failed = False   # True only after a real load error (not "not downloaded yet")


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _buffalo_ready() -> bool:
    """Return True if buffalo_l model files are already on disk."""
    path = os.path.expanduser("~/.insightface/models/buffalo_l")
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def _get_insightface():
    global _insightface_app, _insightface_failed

    if _insightface_app is not None:
        return _insightface_app          # already loaded

    if _insightface_failed:
        return None                      # previously errored — don't retry

    # NOTE: we no longer short-circuit when buffalo_l is missing. FaceAnalysis()
    # auto-downloads the model on first use, so this works on a fresh machine
    # (previously it returned None forever and silently fell back to a crude
    # Haar-cascade paste — no real swap, no head swap, no glasses).
    try:
        import insightface
        if not _buffalo_ready():
            print("[detector] buffalo_l not found — downloading (~300 MB, one-time)…")
        _insightface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=_ORT_PROVIDERS,
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[detector] InsightFace buffalo_l loaded OK")
    except Exception as e:
        print(f"[detector] InsightFace load failed: {e}")
        _insightface_failed = True
        _insightface_app = None

    return _insightface_app


def _clahe_enhance(image: np.ndarray) -> np.ndarray:
    """
    Boost contrast with CLAHE on the L channel so face detectors can find
    faces in poorly-lit, backlit, or underexposed photos.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _detect_insightface(image: np.ndarray) -> list:
    app = _get_insightface()
    if app is None:
        return []
    try:
        faces = app.get(image)
        if not faces:
            return []
        bboxes = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                bboxes.append((x1, y1, x2, y2))
        return bboxes
    except Exception:
        return []


def detect_faces(image: np.ndarray) -> list:
    if image is None or image.size == 0:
        return []

    # First attempt on the original image.
    result = _detect_insightface(image)
    if result:
        return result

    # Low-light / poor-contrast retry: CLAHE enhances the luminance channel
    # before re-running InsightFace and the Haar fallback. Handles phone photos
    # taken in dim rooms, harsh backlit shots, and heavily shadowed faces.
    enhanced = _clahe_enhance(image)
    result = _detect_insightface(enhanced)
    if result:
        return result

    # Final fallback: Haar cascade (works on both original and CLAHE image).
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(detections) == 0:
        return []
    return [(x, y, x + w, y + h) for (x, y, w, h) in detections]


def get_insightface_faces(image: np.ndarray):
    app = _get_insightface()
    if app is None:
        return []
    try:
        return app.get(image)
    except Exception:
        return []
