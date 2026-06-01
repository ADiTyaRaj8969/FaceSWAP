import cv2
import cv2.data
import numpy as np

def _torch_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


# BiSeNet class indices for face parsing
# 0:bg 1:skin 2:left brow 3:right brow 4:left eye 5:right eye
# 6:glasses 7:left ear 8:right ear 9:earring 10:nose 11:mouth
# 12:upper lip 13:lower lip 14:neck 15:necklace 16:clothing
# 17:hair 18:hat
_HAIR_CLASSES = [17]
_FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]
_NECK_CLASSES = [14]


def segment_hair_neck_skin(image: np.ndarray) -> dict:
    """
    Segment hair, skin (face), and neck regions.
    Returns dict with 'face_mask', 'hair_mask', 'neck_mask' (uint8 0/255).
    Uses facexlib BiSeNet (via head_swap._get_parser) when available;
    falls back to Haar-cascade heuristics otherwise.
    """
    try:
        from .head_swap import _get_parser, _parse_region_mask
        import cv2 as _cv2
        from .detector import detect_faces as _detect

        parser = _get_parser()
        if parser is None:
            return _segment_heuristic(image)

        faces = _detect(image)
        if not faces:
            return _segment_heuristic(image)

        bbox = faces[0]
        h, w = image.shape[:2]
        face_mask = (_parse_region_mask(image, bbox, {1, 2, 3, 4, 5, 10, 11, 12, 13},
                                        up=0.3, down=0.3, side=0.4) * 255).astype(np.uint8)
        hair_mask = (_parse_region_mask(image, bbox, {17},
                                        up=0.9, down=0.1, side=0.5) * 255).astype(np.uint8)
        neck_mask = (_parse_region_mask(image, bbox, {14},
                                        up=0.1, down=1.0, side=0.4) * 255).astype(np.uint8)
        return {"face_mask": face_mask, "hair_mask": hair_mask, "neck_mask": neck_mask}
    except Exception:
        return _segment_heuristic(image)


def _segment_heuristic(image: np.ndarray) -> dict:
    """
    Heuristic segmentation when BiSeNet is unavailable.
    Uses GrabCut + skin-colour thresholding.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))

    face_mask = np.zeros((h, w), dtype=np.uint8)
    hair_mask = np.zeros((h, w), dtype=np.uint8)
    neck_mask = np.zeros((h, w), dtype=np.uint8)

    if len(faces) == 0:
        return {"face_mask": face_mask, "hair_mask": hair_mask, "neck_mask": neck_mask}

    fx, fy, fw, fh = faces[0]

    # Face region (inner face ellipse)
    cx, cy = fx + fw // 2, fy + fh // 2
    cv2.ellipse(face_mask, (cx, cy), (fw // 2, int(fh * 0.55)), 0, 0, 360, 255, -1)

    # Hair region: above the face bounding box
    hair_y_top = max(0, fy - int(fh * 0.6))
    hair_y_bot = fy + int(fh * 0.2)
    hair_x1 = max(0, fx - int(fw * 0.2))
    hair_x2 = min(w, fx + fw + int(fw * 0.2))
    hair_mask[hair_y_top:hair_y_bot, hair_x1:hair_x2] = 255

    # Neck region: below the chin
    neck_y_top = fy + int(fh * 0.85)
    neck_y_bot = min(h, fy + fh + int(fh * 0.4))
    neck_x1 = max(0, fx + int(fw * 0.2))
    neck_x2 = min(w, fx + fw - int(fw * 0.2))
    neck_mask[neck_y_top:neck_y_bot, neck_x1:neck_x2] = 255

    # Smooth masks
    face_mask = cv2.GaussianBlur(face_mask, (11, 11), 5)
    _, face_mask = cv2.threshold(face_mask, 64, 255, cv2.THRESH_BINARY)

    return {"face_mask": face_mask, "hair_mask": hair_mask, "neck_mask": neck_mask}
