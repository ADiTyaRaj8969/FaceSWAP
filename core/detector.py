import os
import sys
import glob
import cv2
import cv2.data
import numpy as np


def _ensure_cuda_dlls():
    """
    Windows: make CUDA 12 + cuDNN 9 DLLs discoverable so onnxruntime-gpu can load
    CUDAExecutionProvider — otherwise it silently falls back to CPU (LoadLibrary
    error 126). cuDNN 9 ships its CUDA-12 DLLs in a versioned subfolder
    (…\\CUDNN\\vX\\bin\\12.x) that isn't on PATH by default. We auto-discover the
    CUDA-toolkit bin + the cuDNN 12.x bin and prepend them to PATH.
    No-op on Linux (the HF Docker/CUDA base image already puts CUDA on the path).
    """
    if sys.platform != "win32":
        return
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    patterns = [
        os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA", "v12*", "bin"),
        os.path.join(pf, "NVIDIA", "CUDNN", "v*", "bin", "12*"),  # cuDNN9 for CUDA12
        os.path.join(pf, "NVIDIA", "CUDNN", "v*", "bin"),
    ]
    added = set()
    for pat in patterns:
        for d in sorted(glob.glob(pat), reverse=True):  # newest version first
            if os.path.isdir(d) and d not in added:
                added.add(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass


_ensure_cuda_dlls()

try:
    import onnxruntime as ort
    _ORT_PROVIDERS = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
except ImportError:
    _ORT_PROVIDERS = ["CPUExecutionProvider"]

_face_cascade = None
_insightface_app = None
_insightface_failed = False   # True only after a real load error (not "not downloaded yet")


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _buffalo_ready() -> bool:
    """Return True if buffalo_l model files are already on disk."""
    path = os.path.expanduser("~/.insightface/models/buffalo_l")
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def _get_insightface():
    global _insightface_app, _insightface_failed

    if _insightface_app is not None:
        return _insightface_app          # already loaded

    if _insightface_failed:
        return None                      # previously errored — don't retry

    # NOTE: we no longer short-circuit when buffalo_l is missing. FaceAnalysis()
    # auto-downloads the model on first use, so this works on a fresh machine
    # (previously it returned None forever and silently fell back to a crude
    # Haar-cascade paste — no real swap, no head swap, no glasses).
    try:
        import insightface
        if not _buffalo_ready():
            print("[detector] buffalo_l not found — downloading (~300 MB, one-time)…")
        _insightface_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=_ORT_PROVIDERS,
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[detector] InsightFace buffalo_l loaded OK")
    except Exception as e:
        print(f"[detector] InsightFace load failed: {e}")
        _insightface_failed = True
        _insightface_app = None

    return _insightface_app


def align_face_upright(image: np.ndarray, max_roll: float = 3.0) -> np.ndarray:
    """
    De-roll a (possibly tilted) photo so the face's eyes are horizontal.

    Why: InsightFace's detector misses faces rolled more than ~15-20 deg, so a
    tilted selfie silently falls back to a crude paste. We:
      1. try to detect the face; if that fails (heavy tilt), brute-force rotate
         the image through candidate angles until a face is found;
      2. measure the eye-line angle from the 5 keypoints and rotate the whole
         image so the eyes are level.
    The swapper then aligns this upright source to the target's pose as usual.
    Returns the original image unchanged if no face can be found.
    """
    import math
    app = _get_insightface()
    if app is None:
        return image

    def _eye_angle(face):
        le, re = face.kps[0], face.kps[1]      # left eye, right eye
        return math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))

    def _largest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    faces = app.get(image)
    base_angle = 0.0
    if not faces:
        # Heavy tilt: rotate the image to bring the face into the detector's range.
        for ang in (15, -15, 25, -25, 35, -35, 45, -45, 10, -10, 20, -20):
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            rot = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
            f = app.get(rot)
            if f:
                faces = f
                base_angle = ang
                break
    if not faces:
        return image

    roll = _eye_angle(_largest(faces))         # residual roll after base rotation
    total = base_angle + roll
    if abs(total) < max_roll:
        return image                           # already upright enough
    M = cv2.getRotationMatrix2D((cx, cy), total, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))


def _clahe_enhance(image: np.ndarray) -> np.ndarray:
    """
    Boost contrast with CLAHE on the L channel so face detectors can find
    faces in poorly-lit, backlit, or underexposed photos.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _detect_insightface(image: np.ndarray) -> list:
    app = _get_insightface()
    if app is None:
        return []
    try:
        faces = app.get(image)
        if not faces:
            return []
        bboxes = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                bboxes.append((x1, y1, x2, y2))
        return bboxes
    except Exception:
        return []


def detect_faces(image: np.ndarray) -> list:
    if image is None or image.size == 0:
        return []

    # First attempt on the original image.
    result = _detect_insightface(image)
    if result:
        return result

    # Low-light / poor-contrast retry: CLAHE enhances the luminance channel
    # before re-running InsightFace and the Haar fallback. Handles phone photos
    # taken in dim rooms, harsh backlit shots, and heavily shadowed faces.
    enhanced = _clahe_enhance(image)
    result = _detect_insightface(enhanced)
    if result:
        return result

    # Final fallback: Haar cascade (works on both original and CLAHE image).
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(detections) == 0:
        return []
    return [(x, y, x + w, y + h) for (x, y, w, h) in detections]


def get_insightface_faces(image: np.ndarray):
    app = _get_insightface()
    if app is None:
        return []
    try:
        return app.get(image)
    except Exception:
        return []


def pad_until_detectable(image: np.ndarray, fracs=(0.3, 0.5, 0.8)) -> np.ndarray:
    """
    Make a close-up face detectable.

    A selfie where the face fills the whole frame (no margin) defeats the
    RetinaFace detector inside InsightFace — it returns 0 faces, so the swap
    silently falls back to a crude paste (no real swap, no hair/skin/neck). We
    pad a replicated margin around the image until a face is found; the extra
    border gives the detector the context it needs.

    Safe for the SOURCE image, which is only used to READ the face landmarks —
    the output is always drawn on the target canvas, so the border never shows.
    Returns the padded image, or the original if a face is already detectable
    (or none can be found at any padding).
    """
    app = _get_insightface()
    if app is None:
        return image
    try:
        if app.get(image):
            return image                       # already detectable — no change
        for frac in fracs:
            p = int(max(image.shape[:2]) * frac)
            padded = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_REPLICATE)
            if app.get(padded):
                return padded
    except Exception:
        pass
    return image
