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


def composite_onto_original(result: np.ndarray, work_target: np.ndarray,
                            orig_target: np.ndarray, threshold: int = 6,
                            feather: float = 0.004) -> np.ndarray:
    """
    Put everything the pipeline CHANGED back onto the full-resolution target.

    The swap runs at a reduced working resolution so the models stay affordable,
    and the 4x download then interpolates that whole frame back up. For a
    1086x1448 photo processed at 768x1024 that discards half the real detail —
    including the background and body, which the swap never touches. Here we
    diff the result against the working-resolution target to find what actually
    changed (the head, plus any skin the tone match reached), and paste only
    that onto the untouched original. Everything else stays pixel-exact.

    A diff-driven mask is used rather than a parsed head mask so that changes
    outside the head — match_skin_to_source also recolours neck, arms and hands
    — are carried over too instead of being silently dropped.
    """
    if orig_target is None or result is None:
        return result
    oh, ow = orig_target.shape[:2]
    rh, rw = result.shape[:2]
    if (oh, ow) == (rh, rw):
        return result                      # nothing was downscaled
    if work_target is None or work_target.shape[:2] != result.shape[:2]:
        return cv2.resize(result, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

    diff = cv2.absdiff(result, work_target).max(axis=2)
    # Blur before thresholding so JPEG-level speckle doesn't fragment the mask.
    diff = cv2.GaussianBlur(diff, (0, 0), max(1.0, min(rh, rw) * 0.002))
    mask = (diff > threshold).astype(np.float32)
    if mask.max() <= 0:
        return orig_target.copy()          # pipeline was a no-op
    # Close holes, then grow slightly so the seam sits outside the changed area.
    k = max(3, int(min(rh, rw) * 0.01) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))

    up_result = cv2.resize(result, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
    up_mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_LINEAR)
    up_mask = cv2.GaussianBlur(up_mask, (0, 0), max(1.5, min(oh, ow) * feather))
    a = np.clip(up_mask, 0.0, 1.0)[..., None]

    out = up_result.astype(np.float32) * a + orig_target.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def image_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert BGR numpy array to encoded bytes."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
