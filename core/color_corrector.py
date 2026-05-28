import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def harmonize_colors(
    swapped: np.ndarray,
    target: np.ndarray,
    masks: dict
) -> np.ndarray:
    """
    Harmonize the colors and lighting of the swapped result with the target image.
    Applies LAB mean/std transfer to the face region and adjusts shadows/highlights.
    """
    result = swapped.copy()
    face_mask = masks.get("face_mask")

    if face_mask is None:
        return result

    mask_f = face_mask.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)

    # LAB transfer scoped to face region
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_vals = result_lab[:, :, ch][face_mask > 128]
        tgt_vals = target_lab[:, :, ch][face_mask > 128]
        if src_vals.std() < 1e-6 or tgt_vals.std() < 1e-6:
            continue
        corrected = (
            (result_lab[:, :, ch] - src_vals.mean()) *
            (tgt_vals.std() / src_vals.std()) +
            tgt_vals.mean()
        )
        # Blend only in face mask area
        result_lab[:, :, ch] = (
            corrected * mask_f[:, :, 0] +
            result_lab[:, :, ch] * (1 - mask_f[:, :, 0])
        )

    result = cv2.cvtColor(
        np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    # Soft shadow correction: match luminance distribution at boundary
    result = _correct_boundary_lighting(result, target, face_mask)
    return result


def _correct_boundary_lighting(
    result: np.ndarray,
    target: np.ndarray,
    face_mask: np.ndarray
) -> np.ndarray:
    """
    Blend lighting at the mask boundary using a feathered alpha.
    Prevents sudden brightness shifts at the swap edge.
    """
    feather = cv2.GaussianBlur(face_mask.astype(np.float32), (31, 31), 15)
    feather = feather / 255.0
    alpha = np.stack([feather] * 3, axis=-1)

    blended = (result.astype(np.float32) * alpha +
               target.astype(np.float32) * (1 - alpha))
    return blended.astype(np.uint8)
