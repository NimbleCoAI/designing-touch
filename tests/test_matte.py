"""Tests for the subject-agnostic matte operators (no person assumption)."""
import numpy as np

from dtouch.matte import SaliencyMatte, EdgeMatte, LumaMatte, MotionMatte, make_matte


def _frame_with_square():
    img = np.zeros((180, 320, 3), np.uint8)
    img[60:120, 130:190] = 230          # a bright square on black
    return img


def test_luma_matte_highlights_bright_region():
    m = LumaMatte().compute(_frame_with_square())
    assert m.shape == (180, 320) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert m[90, 160] > 0.5 and m[10, 10] < 0.2


def test_saliency_matte_shape_range_and_structure():
    m = SaliencyMatte().compute(_frame_with_square())
    assert m.shape == (180, 320) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0 + 1e-6
    # spectral-residual saliency responds to content: not a flat/constant field
    assert m.std() > 0.01
    # the square's boundary (novelty) is more salient than its uniform interior
    assert m[60, 160] > m[90, 160]


def test_edge_matte_fires_on_borders_not_interior():
    m = EdgeMatte().compute(_frame_with_square())
    assert m.shape == (180, 320)
    assert m[60, 160] > m[90, 160]      # top border brighter than flat interior


def test_motion_matte_detects_a_moving_square():
    mm = MotionMatte()
    bg = np.zeros((180, 320, 3), np.uint8)
    for _ in range(15):                 # warm up on static background
        mm.compute(bg)
    moved = bg.copy()
    moved[60:120, 130:190] = 255        # square appears -> motion
    m = mm.compute(moved)
    assert mm.warm
    assert m[90, 160] > m[10, 10]


def test_make_matte_factory():
    for kind in ["auto", "saliency", "motion", "edges", "luma"]:
        assert make_matte(kind) is not None
