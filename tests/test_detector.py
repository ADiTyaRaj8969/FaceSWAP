import numpy as np
import pytest
from core.detector import detect_faces


def _synthetic_face_image(h=256, w=256):
    """Create a simple synthetic image with a skin-toned rectangle as a face stub."""
    img = np.full((h, w, 3), (200, 160, 120), dtype=np.uint8)
    return img


def test_detect_faces_returns_list():
    img = _synthetic_face_image()
    result = detect_faces(img)
    assert isinstance(result, list)


def test_detect_faces_empty_on_blank():
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    result = detect_faces(blank)
    assert result == []


def test_detect_faces_none_input():
    result = detect_faces(None)
    assert result == []
