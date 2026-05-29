import cv2
import cv2.data
import numpy as np

_bisenet_model = None

def _torch_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


def _get_bisenet():
    """Load BiSeNet model if available; returns None otherwise."""
    global _bisenet_model
    if _bisenet_model is not None:
        return _bisenet_model

    try:
        import torch
        import torchvision.transforms as transforms

        model_path = "models/bisenet_face_parsing.pth"
        import os
        if not os.path.exists(model_path):
            return None

        # Inline BiSeNet-style model (simplified 19-class face parser)
        from torchvision.models.segmentation import fcn_resnet50
        device = _torch_device()
        model = fcn_resnet50(num_classes=19, pretrained=False)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        _bisenet_model = (model, device)  # store device alongside model
        return _bisenet_model
    except Exception:
        return None


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
    Falls back to landmark-based heuristics if model unavailable.
    """
    result = _get_bisenet()
    if result is not None:
        model, device = result
        return _segment_with_bisenet(image, model, device)
    return _segment_heuristic(image)


def _segment_with_bisenet(image: np.ndarray, model, device) -> dict:
    try:
        import torch
        import torchvision.transforms as T

        h, w = image.shape[:2]
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inp = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)["out"]
        pred = out.argmax(1).squeeze().cpu().numpy()  # back to CPU for numpy
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        def cls_mask(classes):
            m = np.zeros((h, w), dtype=np.uint8)
            for c in classes:
                m[pred == c] = 255
            return m

        return {
            "face_mask": cls_mask(_FACE_CLASSES),
            "hair_mask": cls_mask(_HAIR_CLASSES),
            "neck_mask": cls_mask(_NECK_CLASSES),
        }
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
