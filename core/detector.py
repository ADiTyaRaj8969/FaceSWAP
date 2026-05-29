import os
import cv2
import numpy as np

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

    if not _buffalo_ready():
        return None                      # models not downloaded yet — retry next call

    try:
        import insightface
        _insightface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[detector] InsightFace buffalo_l loaded OK")
    except Exception as e:
        print(f"[detector] InsightFace load failed: {e}")
        _insightface_failed = True
        _insightface_app = None

    return _insightface_app


def detect_faces(image: np.ndarray) -> list:
    if image is None or image.size == 0:
        return []

    app = _get_insightface()
    if app is not None:
        try:
            faces = app.get(image)
            if faces:
                bboxes = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    if (x2 - x1) >= 64 and (y2 - y1) >= 64:
                        bboxes.append((x1, y1, x2, y2))
                if bboxes:
                    return bboxes
        except Exception:
            pass

    # Fallback: Haar cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
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
