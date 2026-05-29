import numpy as np
from core.skin_tone import analyze_skin_tone, match_skin_tone


def _face_image(h=128, w=128, color=(120, 160, 200)):
    """BGR image filled with a flat skin-like colour."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def _bbox(h=128, w=128):
    return (0, 0, w, h)


def test_analyze_skin_tone_keys():
    img = _face_image()
    tone = analyze_skin_tone(img, _bbox())
    for key in ("L", "a", "b", "hue", "saturation", "undertone", "category"):
        assert key in tone


def test_analyze_skin_tone_lightness_range():
    img = _face_image()
    tone = analyze_skin_tone(img, _bbox())
    assert 0 <= tone["L"] <= 100


def test_analyze_skin_tone_empty_crop():
    img = _face_image()
    # bbox outside image - should return default without crashing
    tone = analyze_skin_tone(img, (200, 200, 300, 300))
    assert "L" in tone


def test_match_skin_tone_output_shape():
    src = _face_image(color=(120, 160, 200))
    tgt = _face_image(color=(80, 110, 140))
    src_tone = analyze_skin_tone(src, _bbox())
    tgt_tone = analyze_skin_tone(tgt, _bbox())
    result = match_skin_tone(src, tgt, src_tone, tgt_tone, strength=0.8)
    assert result.shape == src.shape
    assert result.dtype == np.uint8
