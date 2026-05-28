import cv2
import numpy as np


def align_faces(
    source: np.ndarray,
    target: np.ndarray,
    src_landmarks: np.ndarray,
    tgt_landmarks: np.ndarray
) -> np.ndarray:
    """
    Align source face to match the pose/position of the target face.
    Uses Procrustes + affine transform on corresponding landmarks.
    Returns warped source image (same size as target).
    """
    if src_landmarks is None or tgt_landmarks is None:
        return source

    n = min(len(src_landmarks), len(tgt_landmarks))
    src_pts = src_landmarks[:n].astype(np.float32)
    tgt_pts = tgt_landmarks[:n].astype(np.float32)

    # Use only a stable subset of landmarks for the transform
    # Eyes + nose + mouth corners give a robust affine estimate
    key_indices = _get_key_indices(n)
    src_key = src_pts[key_indices]
    tgt_key = tgt_pts[key_indices]

    M, _ = cv2.estimateAffinePartial2D(src_key, tgt_key, method=cv2.RANSAC)
    if M is None:
        M = cv2.estimateAffine2D(src_key, tgt_key)[0]
    if M is None:
        return source

    h, w = target.shape[:2]
    aligned = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned


def procrustes_align(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray
) -> np.ndarray:
    """
    Compute Procrustes transformation (rotation + scale + translation).
    Returns transformation matrix (2x3).
    """
    src_c = src_pts - src_pts.mean(0)
    tgt_c = tgt_pts - tgt_pts.mean(0)

    src_scale = np.sqrt((src_c ** 2).sum() / len(src_c))
    tgt_scale = np.sqrt((tgt_c ** 2).sum() / len(tgt_c))

    src_n = src_c / (src_scale + 1e-6)
    tgt_n = tgt_c / (tgt_scale + 1e-6)

    H = src_n.T @ tgt_n
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    scale = tgt_scale / (src_scale + 1e-6)
    t = tgt_pts.mean(0) - scale * (R @ src_pts.mean(0))

    M = np.array([
        [scale * R[0, 0], scale * R[0, 1], t[0]],
        [scale * R[1, 0], scale * R[1, 1], t[1]],
    ], dtype=np.float32)
    return M


def _get_key_indices(n: int) -> list:
    """Return stable landmark indices for affine estimation."""
    if n >= 468:
        # MediaPipe: left eye, right eye, nose tip, mouth corners
        return [33, 263, 1, 61, 291, 199]
    elif n >= 68:
        # dlib 68: eyes (36,45), nose (30), mouth corners (48,54)
        return [36, 45, 30, 48, 54, 8]
    else:
        step = max(1, n // 6)
        return list(range(0, min(n, 6 * step), step))
