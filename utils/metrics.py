import cv2
import numpy as np


def compute_delta_e(img1: np.ndarray, img2: np.ndarray,
                    mask: np.ndarray | None = None) -> float:
    """
    Compute mean CIE76 ΔE between two images in CIE LAB colour space.
    Optional mask restricts the computation to a region.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"

    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Rescale to physical LAB values
    lab1[:, :, 0] *= 100.0 / 255.0
    lab2[:, :, 0] *= 100.0 / 255.0
    lab1[:, :, 1:] -= 128
    lab2[:, :, 1:] -= 128

    de = np.sqrt(((lab1 - lab2) ** 2).sum(axis=2))

    if mask is not None:
        region = de[mask > 0]
        return float(region.mean()) if region.size > 0 else 0.0
    return float(de.mean())


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union for two binary masks.
    Masks can be uint8 (0/255) or bool.
    """
    m1 = (mask1 > 0)
    m2 = (mask2 > 0)
    intersection = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(intersection) / float(union + 1e-6)


def compute_alignment_error(lm1: np.ndarray, lm2: np.ndarray) -> float:
    """
    Compute mean pixel error between two landmark sets.
    Returns mean Euclidean distance in pixels.
    """
    if lm1 is None or lm2 is None:
        return float("inf")
    n = min(len(lm1), len(lm2))
    return float(np.linalg.norm(lm1[:n] - lm2[:n], axis=1).mean())


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)
    except Exception:
        return 0.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = ((img1.astype(np.float64) - img2.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))
