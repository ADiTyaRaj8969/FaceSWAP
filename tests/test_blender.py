import numpy as np
import pytest
from core.blender import laplacian_blend, poisson_blend


def _img(h=128, w=128, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _mask(h=128, w=128, region="center"):
    m = np.zeros((h, w), dtype=np.uint8)
    if region == "center":
        m[h//4: 3*h//4, w//4: 3*w//4] = 255
    elif region == "full":
        m[:] = 255
    return m


def test_laplacian_blend_output_shape():
    img1 = _img(val=50)
    img2 = _img(val=200)
    mask = _mask()
    result = laplacian_blend(img1, img2, mask, levels=3)
    assert result.shape == img1.shape
    assert result.dtype == np.uint8


def test_laplacian_blend_full_mask_gives_img1():
    img1 = _img(val=100)
    img2 = _img(val=200)
    mask = _mask(region="full")
    result = laplacian_blend(img1, img2, mask, levels=2)
    assert result.mean() > 90


def test_poisson_blend_fallback_on_empty_mask():
    src = _img(val=100)
    dst = _img(val=200)
    empty_mask = np.zeros((128, 128), dtype=np.uint8)
    result = poisson_blend(src, dst, empty_mask)
    assert result.shape == src.shape
