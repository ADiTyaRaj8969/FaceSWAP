import cv2
import numpy as np

_face_cascade = None
_insightface_app = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


_insightface_load_attempted = False

def _get_insightface():
    global _insightface_app, _insightface_load_attempted
    if _insightface_load_attempted:
        return _insightface_app  # None if it failed before — don't retry
    _insightface_load_attempted = True
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
        _insightface_app = None
    return _insightface_app


def detect_faces(image: np.ndarray) -> list:
    """
    Detect faces in image. Returns list of (x1, y1, x2, y2) bounding boxes.
    Tries InsightFace first, falls back to OpenCV Haar cascade.
    """
    if image is None or image.size == 0:
        return []

    # Try InsightFace
    app = _get_insightface()
    if app is not None:
        try:
            faces = app.get(image)  # InsightFace expects BGR (OpenCV native)
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

    # Fallback: OpenCV Haar cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    detections = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
    )
    if len(detections) == 0:
        return []
    bboxes = []
    for (x, y, w, h) in detections:
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def get_insightface_faces(image: np.ndarray):
    """Return raw InsightFace face objects (for use with swapper)."""
    app = _get_insightface()
    if app is None:
        return []
    try:
        return app.get(image)  # InsightFace expects BGR
    except Exception:
        return []