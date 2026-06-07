"""TDD for the deterministic point-field transforms (pure NumPy, no GPU)."""
import numpy as np
import pytest

from dtouch import make_grid, displace_z, random_scale, random_euler, pack_instances


def test_make_grid_count_and_order():
    g = make_grid(gx=4, gy=3, extent=2.0)
    assert g.shape == (12, 3)
    assert g.dtype == np.float32
    assert np.allclose(g[0:4, 1], g[0, 1])
    assert not np.isclose(g[4, 1], g[0, 1])


def test_make_grid_centered_and_z_zero():
    g = make_grid(gx=5, gy=5, extent=2.0)
    assert abs(float(g[:, 0].mean())) < 1e-6
    assert abs(float(g[:, 1].mean())) < 1e-6
    assert np.isclose(g[:, 0].min(), -1.0, atol=1e-6)
    assert np.isclose(g[:, 0].max(), 1.0, atol=1e-6)
    assert np.allclose(g[:, 2], 0.0)


def test_displace_z_maps_luma_to_depth_in_order():
    g = make_grid(gx=3, gy=2, extent=2.0)
    luma = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]], dtype=np.float32)
    out = displace_z(g, luma, depth_scale=2.0)
    assert np.allclose(out[:, :2], g[:, :2])
    assert np.allclose(out[:, 2], luma.reshape(-1) * 2.0)


def test_displace_z_rejects_mismatched_luma():
    g = make_grid(gx=3, gy=2, extent=2.0)
    with pytest.raises(ValueError):
        displace_z(g, np.zeros((4, 4), dtype=np.float32), depth_scale=1.0)


def test_random_scale_deterministic_and_bounded():
    a = random_scale(100, seed=7, lo=0.5, hi=1.5)
    b = random_scale(100, seed=7, lo=0.5, hi=1.5)
    c = random_scale(100, seed=8, lo=0.5, hi=1.5)
    assert a.shape == (100, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0.5 and a.max() <= 1.5


def test_random_euler_deterministic_and_bounded():
    a = random_euler(100, seed=3)
    b = random_euler(100, seed=3)
    assert a.shape == (100, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() < 2 * np.pi + 1e-6


def test_pack_instances_layout():
    pos = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    scale = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    euler = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    buf = pack_instances(pos, scale, euler).reshape(2, 9)
    assert buf.dtype == np.float32
    assert np.allclose(buf[0], [1, 2, 3, 1, 1, 1, 0.1, 0.2, 0.3])
    assert np.allclose(buf[1], [4, 5, 6, 2, 2, 2, 0.4, 0.5, 0.6])
