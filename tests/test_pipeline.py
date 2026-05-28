import numpy as np
import pytest
from pipeline.full_pipeline import run_full_pipeline


def _synthetic_image(h=256, w=256, color=(180, 150, 120)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_pipeline_returns_error_on_faceless_input():
    """Blank images with no faces should return an error key."""
    src = np.zeros((128, 128, 3), dtype=np.uint8)
    tgt = np.zeros((128, 128, 3), dtype=np.uint8)
    result = run_full_pipeline(src, tgt)
    assert "error" in result


def test_pipeline_progress_callback():
    """Progress callback is called at least once."""
    src = _synthetic_image()
    tgt = _synthetic_image()
    calls = []

    def cb(pct, msg):
        calls.append(pct)

    run_full_pipeline(src, tgt, progress_callback=cb)
    # Callback should be called at least at initialisation
    assert len(calls) > 0
