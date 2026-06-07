"""TDD for the stable-fluids core."""
import numpy as np

from dtouch.fluid import bilinear, divergence, project, advect, Fluid2D


def test_bilinear_recovers_grid_values():
    f = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.isclose(bilinear(f, np.array([2.0]), np.array([1.0]))[0], f[1, 2])
    # midpoint interpolation
    mid = bilinear(f, np.array([0.5]), np.array([0.0]))[0]
    assert np.isclose(mid, (f[0, 0] + f[0, 1]) / 2)


def test_advect_shifts_a_blob_with_constant_velocity():
    h = w = 16
    field = np.zeros((h, w))
    field[8, 8] = 1.0
    u = np.full((h, w), 2.0)   # +x flow
    v = np.zeros((h, w))
    out = advect(field, u, v, dt=1.0)
    # mass moved in +x: column 10 now carries more than column 6
    assert out[8, 10] > out[8, 6]
    assert out[8, 10] > 0.0


def test_project_reduces_divergence():
    # A smooth, mostly-divergent (curl-free) field: gradient of a Gaussian bump.
    # Projection should flatten its divergence strongly.
    h = w = 32
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = cy = 15.5
    phi = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 6.0 ** 2))
    u = np.zeros((h, w)); v = np.zeros((h, w))
    u[1:-1, 1:-1] = 0.5 * (phi[1:-1, 2:] - phi[1:-1, :-2])
    v[1:-1, 1:-1] = 0.5 * (phi[2:, 1:-1] - phi[:-2, 1:-1])
    before = np.abs(divergence(u, v)[2:-2, 2:-2]).mean()
    u2, v2 = project(u, v, iters=200)
    after = np.abs(divergence(u2, v2)[2:-2, 2:-2]).mean()
    assert after < before * 0.15, f"divergence not reduced: {before} -> {after}"


def test_fluid_step_conserves_finiteness_and_advects_density():
    f = Fluid2D(32, 32)
    f.density[16, 16] = 10.0
    f.add_force(np.full((32, 32), 1.5), np.zeros((32, 32)))  # push +x
    for _ in range(5):
        f.step(dt=1.0)
    assert np.all(np.isfinite(f.u)) and np.all(np.isfinite(f.density))
    # density center of mass moved in +x from column 16
    cols = np.arange(32)
    com = (f.density.sum(axis=0) * cols).sum() / (f.density.sum() + 1e-9)
    assert com > 16.0


def test_sample_velocity_matches_field():
    f = Fluid2D(8, 8)
    f.u[:] = 3.0
    f.v[:] = -1.0
    vx, vy = f.sample_velocity(np.array([2.5]), np.array([4.5]))
    assert np.isclose(vx[0], 3.0) and np.isclose(vy[0], -1.0)
