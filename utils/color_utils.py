import cv2
import numpy as np


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to float32 CIE LAB (L: 0-100, a/b: -128..127)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return lab


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Convert float32 CIE LAB back to BGR uint8."""
    lab_cv = lab.copy()
    lab_cv[:, :, 0] = lab_cv[:, :, 0] * 255.0 / 100.0
    lab_cv[:, :, 1:] += 128.0
    return cv2.cvtColor(np.clip(lab_cv, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to float32 HSV (H: 0-360, S/V: 0-100)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] *= 2.0          # OpenCV H: 0-180 -> 0-360
    hsv[:, :, 1:] /= 255.0 * 100 # S,V to percent
    return hsv


def hsv_to_bgr(hsv: np.ndarray) -> np.ndarray:
    """Convert float32 HSV (H:0-360, S/V:0-100) back to BGR uint8."""
    hsv_cv = hsv.copy()
    hsv_cv[:, :, 0] /= 2.0
    hsv_cv[:, :, 1:] *= 255.0 / 100.0
    return cv2.cvtColor(np.clip(hsv_cv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the histogram of source image to reference image.
    Operates channel-by-channel on BGR images.
    """
    matched = np.empty_like(source)
    for ch in range(source.shape[2]):
        src_ch = source[:, :, ch].ravel()
        ref_ch = reference[:, :, ch].ravel()

        src_hist, _ = np.histogram(src_ch, 256, [0, 256])
        ref_hist, _ = np.histogram(ref_ch, 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        lut = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched[:, :, ch] = lut[source[:, :, ch]]

    return matched.astype(np.uint8)


def compute_mean_lab(image: np.ndarray, mask: np.ndarray | None = None) -> tuple:
    """Return mean L*, a*, b* for an image (optionally masked)."""
    lab = bgr_to_lab(image)
    if mask is not None:
        region = lab[mask > 0]
    else:
        region = lab.reshape(-1, 3)
    return tuple(region.mean(axis=0))
