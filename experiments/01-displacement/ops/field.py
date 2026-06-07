"""Point-field operators — the code equivalent of TouchDesigner's TOP->POP->displace->Copy SOP.

Each function is a small, pure transform on NumPy arrays. Compose them in a pipeline.
This is the "node graph as code" seed: deterministic, testable, no hidden state.

Conventions
-----------
- A luminance grid is a float32 array of shape (gy, gx) in [0, 1] (rows, cols) like an image.
- A point field is a float32 array of shape (N, 3) of XYZ positions, N = gx*gy, row-major
  so point index i == (row=i//gx, col=i%gx) -> lines up with luma.reshape(-1).
"""
from __future__ import annotations

import numpy as np


def make_grid(gx: int, gy: int, extent: float = 2.0) -> np.ndarray:
    """Base XY grid centered on the origin, Z=0. Spans [-extent/2, +extent/2] in X.

    Row-major: the first gx points are the top row (shared Y), matching a (gy, gx) image.
    """
    half = extent / 2.0
    xs = np.linspace(-half, half, gx, dtype=np.float32)
    # top row first (largest Y) so row 0 of the image is the top of the field
    ys = np.linspace(half, -half, gy, dtype=np.float32)
    gxx, gyy = np.meshgrid(xs, ys)  # both (gy, gx)
    pos = np.stack([gxx.reshape(-1), gyy.reshape(-1), np.zeros(gx * gy)], axis=1)
    return pos.astype(np.float32)


def displace_z(positions: np.ndarray, luma: np.ndarray, depth_scale: float) -> np.ndarray:
    """Displace each point along +Z by its luminance * depth_scale (the 'height' control)."""
    n = positions.shape[0]
    if luma.size != n:
        raise ValueError(
            f"luma has {luma.size} values but field has {n} points; "
            f"resize the source grid to match make_grid(gx, gy)."
        )
    out = positions.copy()
    out[:, 2] = luma.reshape(-1).astype(np.float32) * np.float32(depth_scale)
    return out


def random_scale(n: int, seed: int, lo: float = 0.4, hi: float = 1.2) -> np.ndarray:
    """Per-instance random scale (one of the two TD randoms). Stable for a given seed."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(lo, hi, size=(n, 3))
    return s.astype(np.float32)


def random_euler(n: int, seed: int) -> np.ndarray:
    """Per-instance random rotation as Euler angles in radians [0, 2pi) (the other TD random)."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.0, 2.0 * np.pi, size=(n, 3))
    return e.astype(np.float32)


def pack_instances(positions: np.ndarray, scale: np.ndarray, euler: np.ndarray) -> np.ndarray:
    """Interleave per-instance attributes into one flat float32 buffer for a GPU VBO.

    Layout per instance: [ox, oy, oz, sx, sy, sz, ex, ey, ez] (9 floats).
    """
    n = positions.shape[0]
    buf = np.empty((n, 9), dtype=np.float32)
    buf[:, 0:3] = positions
    buf[:, 3:6] = scale
    buf[:, 6:9] = euler
    return buf.reshape(-1)
