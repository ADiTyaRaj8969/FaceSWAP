import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def dilate_mask(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Morphologically dilate a binary mask."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Morphologically erode a binary mask."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def feather_mask(mask: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Apply Gaussian feathering to a binary mask.
    Returns float32 mask in [0, 1].
    """
    f = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    feathered = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return feathered


def merge_masks(masks: list, weights: list | None = None) -> np.ndarray:
    """
    Merge multiple masks by weighted average.
    All masks should be same shape, uint8 or float32 in [0,255] or [0,1].
    """
    if not masks:
        raise ValueError("No masks provided")
    if weights is None:
        weights = [1.0] * len(masks)

    result = np.zeros_like(masks[0], dtype=np.float32)
    total_w = sum(weights)
    for m, w in zip(masks, weights):
        f = m.astype(np.float32) / 255.0 if m.max() > 1 else m.astype(np.float32)
        result += f * (w / total_w)
    return np.clip(result, 0, 1)


def get_boundary_mask(mask: np.ndarray, dilation: int = 5, erosion: int = 3) -> np.ndarray:
    """Extract the boundary region of a binary mask."""
    dilated = dilate_mask(mask, dilation)
    eroded = erode_mask(mask, erosion)
    return cv2.subtract(dilated, eroded)


def create_face_ellipse_mask(h: int, w: int,
                             cx: int, cy: int,
                             rx: int, ry: int) -> np.ndarray:
    """Create an elliptical face mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask
