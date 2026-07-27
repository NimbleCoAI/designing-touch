"""Tests for dtouch.dither — Bayer ordered dithering and Floyd-Steinberg error diffusion."""
import numpy as np
import pytest

from dtouch.dither import bayer_dither, floyd_steinberg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp(h=32, w=32, channels=None):
    """Horizontal luminance ramp [0, 1] as float32."""
    r = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    if channels is not None:
        r = np.stack([r] * channels, axis=2)
    return r


def _uniform(h=16, w=16, v=0.5, channels=None):
    arr = np.full((h, w), v, dtype=np.float32)
    if channels is not None:
        arr = np.stack([arr] * channels, axis=2)
    return arr


# ---------------------------------------------------------------------------
# Bayer dither
# ---------------------------------------------------------------------------

class TestBayerDither:
    def test_output_shape_2d(self):
        img = _ramp(32, 32)
        out = bayer_dither(img, bits=2)
        assert out.shape == img.shape

    def test_output_shape_3d(self):
        img = _ramp(32, 32, channels=3)
        out = bayer_dither(img, bits=2)
        assert out.shape == img.shape

    def test_output_dtype(self):
        out = bayer_dither(_ramp(), bits=2)
        assert out.dtype == np.float32

    def test_output_range(self):
        img = _ramp(32, 32)
        out = bayer_dither(img, bits=2)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_quantised_levels(self):
        """With bits=2, output should have at most 4 distinct values (0, 1/3, 2/3, 1)."""
        img = _ramp(32, 64)
        out = bayer_dither(img, bits=2)
        unique = np.unique(np.round(out, 5))
        assert len(unique) <= 4

    def test_uniform_black_stays_black(self):
        img = _uniform(v=0.0)
        out = bayer_dither(img, bits=2)
        assert np.allclose(out, 0.0)

    def test_uniform_white_stays_white(self):
        img = _uniform(v=1.0)
        out = bayer_dither(img, bits=2)
        assert np.allclose(out, 1.0)

    def test_higher_bits_finer_levels(self):
        img = _ramp(32, 64)
        out2 = bayer_dither(img, bits=2)
        out4 = bayer_dither(img, bits=4)
        levels2 = len(np.unique(np.round(out2, 5)))
        levels4 = len(np.unique(np.round(out4, 5)))
        assert levels4 > levels2

    def test_different_matrix_sizes(self):
        img = _ramp(32, 32)
        for sz in (2, 4, 8):
            out = bayer_dither(img, bits=2, matrix_size=sz)
            assert out.shape == img.shape
            assert out.min() >= 0.0 and out.max() <= 1.0

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=0)
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=9)

    def test_invalid_matrix_size_raises(self):
        with pytest.raises(ValueError):
            bayer_dither(_ramp(), bits=2, matrix_size=3)

    def test_average_of_uniform_mid_grey(self):
        """Dithered mid-grey should average close to 0.5 over a large area."""
        img = _uniform(h=64, w=64, v=0.5)
        out = bayer_dither(img, bits=1)   # only 0 and 1 possible
        assert abs(out.mean() - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Floyd-Steinberg dither
# ---------------------------------------------------------------------------

class TestFloydSteinberg:
    def test_output_shape_2d(self):
        img = _ramp(16, 32)
        out = floyd_steinberg(img, bits=2)
        assert out.shape == img.shape

    def test_output_shape_3d(self):
        img = _ramp(16, 32, channels=3)
        out = floyd_steinberg(img, bits=2)
        assert out.shape == img.shape

    def test_output_dtype(self):
        out = floyd_steinberg(_ramp(8, 16), bits=2)
        assert out.dtype == np.float32

    def test_output_range(self):
        img = _ramp(16, 32)
        out = floyd_steinberg(img, bits=2)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_uniform_black_stays_black(self):
        img = _uniform(8, 16, v=0.0)
        out = floyd_steinberg(img, bits=2)
        assert np.allclose(out, 0.0)

    def test_uniform_white_stays_white(self):
        img = _uniform(8, 16, v=1.0)
        out = floyd_steinberg(img, bits=2)
        assert np.allclose(out, 1.0)

    def test_average_close_to_input_grey(self):
        """Error diffusion preserves average luminance: mean of output ≈ mean of input."""
        img = _uniform(16, 32, v=0.5)
        out = floyd_steinberg(img, bits=1)  # 0/1 only
        assert abs(out.mean() - 0.5) < 0.1

    def test_quantised_levels(self):
        """With bits=2, output values come from {0, 1/3, 2/3, 1}."""
        img = _ramp(8, 16)
        out = floyd_steinberg(img, bits=2)
        unique = np.unique(np.round(out, 5))
        assert len(unique) <= 4

    def test_does_not_mutate_input(self):
        img = _ramp(8, 16)
        orig = img.copy()
        floyd_steinberg(img, bits=2)
        assert np.array_equal(img, orig)

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            floyd_steinberg(_ramp(4, 4), bits=0)

    def test_3d_per_channel_independence(self):
        """Channels should be dithered independently — red ramp vs blue ramp."""
        h, w = 8, 16
        img = np.zeros((h, w, 3), np.float32)
        img[:, :, 0] = np.linspace(0.0, 1.0, w)   # red ramp
        img[:, :, 2] = np.linspace(1.0, 0.0, w)   # blue ramp (reversed)
        out = floyd_steinberg(img, bits=2)
        # first and last pixel of red/blue channels should differ
        assert not np.allclose(out[:, 0, 0], out[:, -1, 0])
        assert not np.allclose(out[:, 0, 2], out[:, -1, 2])
