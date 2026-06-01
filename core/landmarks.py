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
    """
    Fallback when MediaPipe is unavailable: use InsightFace 5-point keypoints
    (left eye, right eye, nose, left mouth, right mouth) mapped to a sparse
    468-point array so downstream callers get real anatomical positions.
    Returns None if InsightFace is also unavailable — callers must handle None.

    The previous implementation returned random uniform noise inside the face
    bounding box, which silently corrupted alignment transforms and quality
    scores whenever MediaPipe failed.
    """
    from .detector import _get_insightface
    app = _get_insightface()
    if app is None:
        return None
    try:
        faces = app.get(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps = np.asarray(face.kps, dtype=np.float32)  # shape (5, 2)

        # Map the 5 InsightFace keypoints to their nearest MediaPipe indices so
        # _get_key_indices() in aligner.py selects them correctly.
        #   kps[0] = left eye   → MP 33
        #   kps[1] = right eye  → MP 263
        #   kps[2] = nose tip   → MP 1
        #   kps[3] = left mouth → MP 61
        #   kps[4] = right mouth→ MP 291
        pts = np.zeros((468, 2), dtype=np.float32)
        for mp_idx, kp in zip([33, 263, 1, 61, 291], kps):
            pts[mp_idx] = kp
        return pts
    except Exception:
        return None