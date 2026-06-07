"""Integration smoke test: synthetic source -> field -> GPU render produces a real image.

No camera, no files — runs anywhere with a GL context. Guards the whole pipeline against
regressions (shader compile errors, attribute layout drift, empty renders).
"""
import numpy as np

from ops.sources import SyntheticSource
from ops.field import make_grid, displace_z, random_scale, random_euler, pack_instances
from ops.render import Renderer


def test_pipeline_renders_nonblack_frame():
    gx, gy = 64, 36
    n = gx * gy
    src = SyntheticSource(gx, gy, frames=1)
    grid = make_grid(gx, gy)
    scale = random_scale(n, seed=0)
    euler = random_euler(n, seed=1)

    luma = src.read()
    assert luma is not None and luma.shape == (gy, gx)

    positions = displace_z(grid, luma, depth_scale=1.2)
    buf = pack_instances(positions, scale, euler)

    r = Renderer(320, 180, n, base_size=0.7 / max(gx, gy), depth_scale=1.2)
    try:
        frame = r.render(buf)
    finally:
        r.release()

    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    # background is black; the dots must light up a meaningful number of pixels
    lit = int((frame.max(axis=2) > 20).sum())
    assert lit > 200, f"expected many lit pixels, got {lit}"
