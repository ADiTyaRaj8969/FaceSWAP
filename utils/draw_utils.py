import cv2
import numpy as np


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: tuple = (0, 255, 0),
    radius: int = 1
) -> np.ndarray:
    """Draw facial landmark points on a copy of the image."""
    out = image.copy()
    if landmarks is None:
        return out
    for (x, y) in landmarks.astype(int):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1)
    return out


def draw_bounding_boxes(
    image: np.ndarray,
    bboxes: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw bounding boxes on a copy of the image. Accepts (x1,y1,x2,y2) tuples."""
    out = image.copy()
    for bbox in bboxes:
        if hasattr(bbox, 'bbox'):
            # InsightFace face object
            x1, y1, x2, y2 = bbox.bbox.astype(int)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 120, 255),
    alpha: float = 0.35
) -> np.ndarray:
    """Overlay a coloured semi-transparent mask on the image."""
    out = image.copy().astype(np.float32)
    overlay = np.zeros_like(image, dtype=np.float32)
    m = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    overlay[:] = color[::-1]  # BGR
    mask_3 = np.stack([m] * 3, axis=-1)
    out = out * (1 - mask_3 * alpha) + overlay * mask_3 * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_quality_metrics(
    image: np.ndarray,
    metrics: dict
) -> np.ndarray:
    """Draw quality metric text overlay on the image."""
    out = image.copy()
    lines = [
        f"Align: {metrics.get('alignment', 0):.1f}/100",
        f"Blend: {metrics.get('blend', 0):.1f}/100",
        f"dE:    {metrics.get('delta_e', 0):.2f}",
        f"Natural: {metrics.get('naturalness', 0):.1f}/100",
    ]
    y = 20
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out
