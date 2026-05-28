import cv2
import numpy as np


def compute_quality_score(
    swapped: np.ndarray,
    target: np.ndarray,
    src_landmarks: np.ndarray,
    tgt_landmarks: np.ndarray
) -> dict:
    """
    Compute quality metrics for the swap result.
    Returns dict: alignment, blend, delta_e, naturalness (all 0-100 scale).
    """
    alignment = _alignment_score(src_landmarks, tgt_landmarks)
    blend = _blend_quality(swapped, target)
    delta_e = _compute_delta_e(swapped, target)
    naturalness = _naturalness_score(swapped)

    return {
        "alignment": alignment,
        "blend": blend,
        "delta_e": delta_e,
        "naturalness": naturalness,
    }


def _alignment_score(src_lm: np.ndarray, tgt_lm: np.ndarray) -> float:
    """Mean pixel deviation between landmark sets, converted to 0-100 score."""
    if src_lm is None or tgt_lm is None:
        return 50.0
    n = min(len(src_lm), len(tgt_lm))
    diff = np.linalg.norm(src_lm[:n] - tgt_lm[:n], axis=1)
    mean_err = diff.mean()
    # Map: 0px -> 100, 10px -> 0
    score = max(0.0, 100.0 - mean_err * 10.0)
    return round(float(score), 1)


def _blend_quality(swapped: np.ndarray, target: np.ndarray) -> float:
    """
    Estimate blend quality by measuring gradient discontinuities at the face region boundary.
    Higher = smoother boundary.
    """
    if swapped.shape != target.shape:
        return 75.0

    diff = cv2.absdiff(swapped, target).astype(np.float32)
    edges = cv2.Laplacian(diff, cv2.CV_32F)
    discontinuity = np.abs(edges).mean()
    # Map: 0 -> 100, 30 -> 0
    score = max(0.0, 100.0 - discontinuity * (100.0 / 30.0))
    return round(float(score), 1)


def _compute_delta_e(swapped: np.ndarray, target: np.ndarray) -> float:
    """
    Compute mean CIE ΔE (L*a*b* Euclidean) between swapped and target images.
    Lower is better (target < 10).
    """
    if swapped.shape != target.shape:
        return 15.0

    lab1 = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(target,  cv2.COLOR_BGR2LAB).astype(np.float32)

    # Rescale L* from 0-255 to 0-100, a* b* from 0-255 to -128..127
    lab1[:, :, 0] *= 100.0 / 255.0
    lab2[:, :, 0] *= 100.0 / 255.0
    lab1[:, :, 1:] -= 128
    lab2[:, :, 1:] -= 128

    de = np.sqrt(((lab1 - lab2) ** 2).sum(axis=2))
    return round(float(de.mean()), 2)


def _naturalness_score(swapped: np.ndarray) -> float:
    """
    Heuristic naturalness score: checks for colour clipping, excessive noise,
    and unnatural saturation levels.
    """
    clipped = (
        (swapped == 0).sum() + (swapped == 255).sum()
    ) / swapped.size
    clip_penalty = min(clipped * 500, 30.0)

    hsv = cv2.cvtColor(swapped, cv2.COLOR_BGR2HSV)
    mean_sat = hsv[:, :, 1].mean()
    # Penalise if saturation is very low (<20) or very high (>200)
    sat_penalty = max(0, 20 - mean_sat) + max(0, mean_sat - 200) * 0.5

    noise = cv2.Laplacian(cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()
    noise_penalty = min(noise / 1000.0, 20.0)

    score = max(0.0, 100.0 - clip_penalty - sat_penalty - noise_penalty)
    return round(float(score), 1)
