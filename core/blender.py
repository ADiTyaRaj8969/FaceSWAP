import cv2
import numpy as np


def laplacian_blend(img1: np.ndarray, img2: np.ndarray,
                    mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Multi-scale Laplacian pyramid blending.
    Blends high-frequency detail (hair strands) separately from
    low-frequency color/lighting - prevents visible seams.
    """
    mask_f = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)

    # Build Gaussian pyramids
    gp1, gp2, gpm = [img1.astype(np.float32)], [img2.astype(np.float32)], [mask_f]
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpm.append(cv2.pyrDown(gpm[-1]))

    # Build Laplacian pyramids
    lp1 = [gp1[levels]]
    lp2 = [gp2[levels]]
    for i in range(levels, 0, -1):
        lp1.append(gp1[i - 1] - cv2.pyrUp(gp1[i], dstsize=gp1[i - 1].shape[:2][::-1]))
        lp2.append(gp2[i - 1] - cv2.pyrUp(gp2[i], dstsize=gp2[i - 1].shape[:2][::-1]))

    # Blend each level
    blended = []
    for l1, l2, m in zip(lp1, lp2, reversed(gpm)):
        if m.shape[:2] != l1.shape[:2]:
            m = cv2.resize(m, (l1.shape[1], l1.shape[0]))
        blended.append(l1 * m + l2 * (1 - m))

    # Reconstruct
    result = blended[0]
    for b in blended[1:]:
        result = cv2.pyrUp(result, dstsize=b.shape[:2][::-1]) + b

    return np.clip(result, 0, 255).astype(np.uint8)


def poisson_blend(src: np.ndarray, dst: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """
    OpenCV seamless clone (Poisson blending).
    Best for removing visible paste edges.
    """
    mask_u8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)

    # Find center of mask
    M = cv2.moments(mask_u8)
    if M["m00"] == 0:
        return src  # fallback

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)

    try:
        result = cv2.seamlessClone(src, dst, mask_u8, center, cv2.NORMAL_CLONE)
    except Exception:
        result = src  # fallback if clone fails at edges
    return result
