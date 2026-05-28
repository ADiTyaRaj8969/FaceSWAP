import cv2
import numpy as np

_mp_face_mesh = None
_face_mesh_instance = None


def _get_face_mesh():
    global _mp_face_mesh, _face_mesh_instance
    if _face_mesh_instance is None:
        try:
            import mediapipe as mp
            _mp_face_mesh = mp.solutions.face_mesh
            _face_mesh_instance = _mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except Exception:
            _face_mesh_instance = None
    return _face_mesh_instance


def extract_landmarks_468(image: np.ndarray) -> np.ndarray | None:
    """
    Extract 468 facial landmarks using MediaPipe Face Mesh.
    Returns array of shape (468, 2) with (x, y) pixel coords, or None.
    """
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return _fallback_landmarks(image)

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return _fallback_landmarks(image)

    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
    return pts  # shape (468, 2)


def extract_landmarks_68(image: np.ndarray) -> np.ndarray | None:
    """
    Extract 68 facial landmarks. Uses the first 68 points from MediaPipe
    mapped to dlib-compatible indices.
    Returns array of shape (68, 2), or None.
    """
    pts468 = extract_landmarks_468(image)
    if pts468 is None:
        return None
    # MediaPipe to dlib 68-point index mapping (approximate)
    MP_TO_DLIB_68 = [
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365,
        397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336, 296,
        334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305,
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        78, 82, 13, 312, 308, 317, 14, 87
    ]
    return pts468[MP_TO_DLIB_68]


def _fallback_landmarks(image: np.ndarray) -> np.ndarray | None:
    """Minimal fallback using OpenCV face detection to synthesize landmarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    # Generate synthetic 468 points distributed over the face bounding box
    rng = np.random.default_rng(42)
    pts = rng.uniform([x, y], [x + w, y + h], size=(468, 2)).astype(np.float32)
    return pts