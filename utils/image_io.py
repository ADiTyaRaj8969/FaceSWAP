import cv2
import numpy as np
from PIL import Image
import io


def load_image(source) -> np.ndarray:
    """
    Load an image from a file path, PIL Image, bytes, BytesIO,
    or Streamlit UploadedFile. Returns BGR numpy array.
    """
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img

    # BytesIO or file-like (Streamlit UploadedFile)
    if hasattr(source, "read"):
        data = source.read()
    elif isinstance(source, (bytes, bytearray)):
        data = source
    else:
        # Try reading as bytes
        data = bytes(source.getvalue()) if hasattr(source, "getvalue") else bytes(source)

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback via PIL
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def save_image(image: np.ndarray, path: str, quality: int = 95) -> None:
    """Save a BGR numpy array to disk."""
    ext = path.rsplit(".", 1)[-1].lower()
    params = []
    if ext in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    cv2.imwrite(path, image, params)


def resize_keep_aspect(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image so the longest side <= max_size, maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale == 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert BGR numpy array to encoded bytes."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
