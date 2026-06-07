"""Tests for the particle-flow simulation."""
import numpy as np

from dtouch.particles import ParticleFlow


def _square_matte(gw, gh):
    m = np.zeros((gh, gw), np.float32)
    m[gh // 3: 2 * gh // 3, gw // 3: 2 * gw // 3] = 1.0
    return m


def test_particles_flow_into_the_matte_region():
    gw, gh = 128, 72
    pf = ParticleFlow(n=8000, gw=gw, gh=gh, seed=0)
    matte = _square_matte(gw, gh)
    gray = np.zeros((gh, gw), np.float32)

    def frac_inside():
        inside = ((pf.px >= gw // 3) & (pf.px < 2 * gw // 3) &
                  (pf.py >= gh // 3) & (pf.py < 2 * gh // 3))
        return float(inside.mean())

    start = frac_inside()
    for _ in range(40):
        pf.update(matte, gray)
    end = frac_inside()
    # particles should concentrate in the matte square (it covers ~1/9 of the area)
    assert end > start + 0.3, f"particles did not gather: {start:.2f} -> {end:.2f}"
    assert end > 0.6


def test_render_data_shape_and_bounds():
    gw, gh = 96, 54
    pf = ParticleFlow(n=5000, gw=gw, gh=gh, seed=1)
    pf.update(_square_matte(gw, gh), np.zeros((gh, gw), np.float32))
    data = pf.render_data()
    assert data.shape == (5000 * 7,)
    assert np.all(np.isfinite(data))
    inst = data.reshape(5000, 7)
    assert inst[:, 0].min() >= -1.5 and inst[:, 0].max() <= 1.5   # x_ndc
    assert inst[:, 1].min() >= -1.5 and inst[:, 1].max() <= 1.5   # y_ndc
    assert (inst[:, 4:7] >= 0).all() and (inst[:, 4:7] <= 1.0001).all()  # color
