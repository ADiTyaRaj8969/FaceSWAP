import numpy as np
from pipeline.full_pipeline import run_full_pipeline
from core.super_res import enhance_resolution
from core.skin_tone import match_skin_tone, analyze_skin_tone


def _synthetic_image(h=256, w=256, color=(180, 150, 120)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_pipeline_returns_error_on_faceless_input():
    """Blank images with no faces should return an error key."""
    src = np.zeros((128, 128, 3), dtype=np.uint8)
    tgt = np.zeros((128, 128, 3), dtype=np.uint8)
    result = run_full_pipeline(src, tgt)
    assert "error" in result


def test_pipeline_progress_callback_reports_detection_stage():
    """
    On faceless input the pipeline must still invoke the callback for the
    first 'Detecting faces' stage before returning the error — this verifies
    the callback contract, not just that *some* call happened.
    """
    src = _synthetic_image()
    tgt = _synthetic_image()
    calls = []

    def cb(pct, msg):
        calls.append((pct, msg))

    result = run_full_pipeline(src, tgt, progress_callback=cb)

    assert len(calls) > 0
    assert calls[0][0] == 5                      # first stage is 5%
    assert "detect" in calls[0][1].lower()       # ...and it's the detect step
    assert "error" in result                     # faceless → error returned


def test_super_res_lanczos_fallback_upscales_4x():
    """
    With no SR model weights present, enhance_resolution must fall back to a
    Lanczos resize and still deliver a 4x larger image (never silently no-op).
    """
    img = _synthetic_image(h=128, w=128)
    out = enhance_resolution(img, scale=4)
    assert out.shape[0] == 128 * 4
    assert out.shape[1] == 128 * 4
    assert out.dtype == np.uint8


def test_match_skin_tone_confined_to_face_mask():
    """
    A face_mask covering only the left half must leave the right half pixels
    untouched while shifting the masked region toward the target tone.
    """
    src = _synthetic_image(color=(120, 160, 200))
    tgt = _synthetic_image(color=(60, 90, 130))
    src_tone = analyze_skin_tone(src, (0, 0, 256, 256))

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:, :100] = 255                          # left columns only (no feather overlap on far right)

    out = match_skin_tone(src, tgt, src_tone, src_tone, strength=0.9, face_mask=mask)

    # Far-right region (well outside mask + feather) must be unchanged
    assert np.array_equal(out[:, 200:], src[:, 200:])
    # Masked region must have moved away from the source colour
    assert not np.array_equal(out[:, :80], src[:, :80])
