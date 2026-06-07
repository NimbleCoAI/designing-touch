"""TDD for the deterministic point-field transforms (TOP -> POP -> displace -> Copy SOP attrs).

These are pure NumPy functions: no GPU, no I/O. They are the load-bearing logic
of the displacement system, so they get tested. Rendering is smoke-tested separately.
"""
import numpy as np
import pytest

from ops.field import make_grid, displace_z, random_scale, random_euler, pack_instances


def test_make_grid_count_and_order():
    # gx columns, gy rows -> gx*gy points, row-major (matches a (gy, gx) image)
    g = make_grid(gx=4, gy=3, extent=2.0)
    assert g.shape == (12, 3)
    assert g.dtype == np.float32
    # row-major: first gx points share the same Y (top row)
    assert np.allclose(g[0:4, 1], g[0, 1])
    # next row has a different Y
    assert not np.isclose(g[4, 1], g[0, 1])


def test_make_grid_centered_and_z_zero():
    g = make_grid(gx=5, gy=5, extent=2.0)
    # centered around origin in XY
    assert abs(float(g[:, 0].mean())) < 1e-6
    assert abs(float(g[:, 1].mean())) < 1e-6
    # spans roughly [-1, 1] for extent 2.0
    assert np.isclose(g[:, 0].min(), -1.0, atol=1e-6)
    assert np.isclose(g[:, 0].max(), 1.0, atol=1e-6)
    # flat before displacement
    assert np.allclose(g[:, 2], 0.0)


def test_displace_z_maps_luma_to_depth_in_order():
    g = make_grid(gx=3, gy=2, extent=2.0)
    luma = np.array([[0.0, 0.5, 1.0],
                     [0.25, 0.75, 1.0]], dtype=np.float32)  # shape (gy=2, gx=3)
    out = displace_z(g, luma, depth_scale=2.0)
    # XY untouched
    assert np.allclose(out[:, :2], g[:, :2])
    # Z = flattened luma * depth_scale, row-major
    expected_z = luma.reshape(-1) * 2.0
    assert np.allclose(out[:, 2], expected_z)


def test_displace_z_rejects_mismatched_luma():
    g = make_grid(gx=3, gy=2, extent=2.0)
    bad = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        displace_z(g, bad, depth_scale=1.0)


def test_random_scale_deterministic_and_bounded():
    a = random_scale(100, seed=7, lo=0.5, hi=1.5)
    b = random_scale(100, seed=7, lo=0.5, hi=1.5)
    c = random_scale(100, seed=8, lo=0.5, hi=1.5)
    assert a.shape == (100, 3) and a.dtype == np.float32
    assert np.array_equal(a, b)          # same seed -> identical (stable across frames)
    assert not np.array_equal(a, c)      # different seed -> different
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
    buf = pack_instances(pos, scale, euler)
    # 2 instances x 9 floats (offset3 + scale3 + euler3)
    assert buf.dtype == np.float32
    assert buf.shape == (2 * 9,)
    flat = buf.reshape(2, 9)
    assert np.allclose(flat[0], [1, 2, 3, 1, 1, 1, 0.1, 0.2, 0.3])
    assert np.allclose(flat[1], [4, 5, 6, 2, 2, 2, 0.4, 0.5, 0.6])
