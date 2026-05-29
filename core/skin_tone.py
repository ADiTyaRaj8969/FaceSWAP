import cv2
import numpy as np


def analyze_skin_tone(image: np.ndarray, face_bbox) -> dict:
    """
    Analyze skin tone from forehead, cheeks, chin, nose regions.
    Returns L*, a*, b* (CIE LAB), hue, saturation, undertone, category.
    face_bbox: (x1, y1, x2, y2) tuple or InsightFace bbox array.
    """
    # Normalise bbox to (x1, y1, x2, y2) ints
    if hasattr(face_bbox, '__len__') and len(face_bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
    else:
        # InsightFace face object
        x1, y1, x2, y2 = [int(v) for v in face_bbox.bbox]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return _default_tone()

    fh, fw = face_crop.shape[:2]

    # Sample 5 key regions
    regions = {
        'forehead':    face_crop[int(fh*0.05):int(fh*0.15), int(fw*0.3):int(fw*0.7)],
        'left_cheek':  face_crop[int(fh*0.4):int(fh*0.6),  int(fw*0.05):int(fw*0.3)],
        'right_cheek': face_crop[int(fh*0.4):int(fh*0.6),  int(fw*0.7):int(fw*0.95)],
        'chin':        face_crop[int(fh*0.8):int(fh*0.95), int(fw*0.3):int(fw*0.7)],
        'nose':        face_crop[int(fh*0.3):int(fh*0.55), int(fw*0.4):int(fw*0.6)],
    }

    lab_values = []
    hsv_values = []
    for region in regions.values():
        if region.size == 0:
            continue
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab_values.append(cv2.mean(lab)[:3])
        hsv_values.append(cv2.mean(hsv)[:3])

    if not lab_values:
        return _default_tone()

    # Average across regions (OpenCV LAB: L 0-255, a/b 0-255 shifted)
    L   = np.mean([v[0] for v in lab_values]) * (100 / 255)
    a   = np.mean([v[1] for v in lab_values]) - 128
    b   = np.mean([v[2] for v in lab_values]) - 128
    hue = np.mean([v[0] for v in hsv_values]) * 2          # 0-360°
    sat = np.mean([v[1] for v in hsv_values]) / 255 * 100  # 0-100%

    # Undertone classification
    if a > 5 and b < 5:
        undertone = "Cool (Pinkish)"
    elif b > 10:
        undertone = "Warm (Yellow/Golden)"
    else:
        undertone = "Neutral"

    # Fitzpatrick-like category
    if L > 75:      category = "Fair"
    elif L > 65:    category = "Light"
    elif L > 55:    category = "Medium"
    elif L > 45:    category = "Olive/Tan"
    else:           category = "Deep"

    return dict(L=L, a=a, b=b, hue=hue, saturation=sat,
                undertone=undertone, category=category)


def match_skin_tone(
    swapped_img: np.ndarray,
    target_img: np.ndarray,
    src_tone: dict,  # noqa: ARG001 — kept for API compatibility
    tgt_tone: dict,  # noqa: ARG001 — kept for API compatibility
    strength: float = 0.9,
    face_mask=None,  # type: np.ndarray | None
) -> np.ndarray:
    """
    Adjust swapped face skin tone to match target, confined to face_mask region.
    Passing face_mask prevents hair/neck/background from being colour-shifted.
    When face_mask is None the transfer falls back to the whole image (legacy).
    """
    src_lab = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img,  cv2.COLOR_BGR2LAB).astype(np.float32)

    # Determine which pixels to compute statistics from
    if face_mask is not None and face_mask.size > 0:
        stat_mask = face_mask > 128
    else:
        stat_mask = np.ones(src_lab.shape[:2], dtype=bool)

    corrected_lab = src_lab.copy()
    for ch in range(3):
        src_vals = src_lab[:, :, ch][stat_mask]
        tgt_vals = tgt_lab[:, :, ch][stat_mask]
        if src_vals.std() < 1e-6 or tgt_vals.std() < 1e-6:
            continue
        corrected_ch = (
            (src_lab[:, :, ch] - src_vals.mean()) *
            (tgt_vals.std() / src_vals.std()) +
            tgt_vals.mean()
        )
        corrected_lab[:, :, ch] = (
            src_lab[:, :, ch] * (1 - strength) + corrected_ch * strength
        )

    # Apply correction only inside the face mask (feathered at boundary)
    if face_mask is not None and face_mask.size > 0:
        alpha = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (21, 21), 7)
        alpha = np.stack([alpha] * 3, axis=-1)
        result_lab = src_lab * (1.0 - alpha) + corrected_lab * alpha
    else:
        result_lab = corrected_lab

    result = cv2.cvtColor(
        np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )
    return result


def _default_tone() -> dict:
    return dict(L=55.0, a=5.0, b=10.0, hue=25.0, saturation=35.0,
                undertone="Neutral", category="Medium")
