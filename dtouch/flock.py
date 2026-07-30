"""Boids steering for the live particle cloud — Reynolds' three rules, at 200k particles.

`experiments/06-flocking` computes the classic formulation: every boid against every other,
which materialises an n x n neighbour matrix. That is fine for the ~1-3k boids it renders and
completely impossible for the live instrument, which pushes 200,000 particles per frame. A
straight port would have been O(n^2) — around 4x10^10 pair terms.

So the neighbourhood is approximated with a **coarse spatial grid**: particles are binned into
cells, per-cell aggregates are accumulated with `np.bincount`, and each particle then steers
against its own cell's aggregate. That is O(n) with a handful of vectorised passes, and it
holds the thing that actually matters aesthetically — that a particle reacts to *its local
neighbourhood* rather than to the global mean. Cells are the neighbourhood radius.

The aggregates **exclude the particle itself** (subtract-self, then divide by count-1). With
coarse cells and sparse regions a cell can hold very few particles, and self-inclusion there
makes a particle chase its own position — cohesion pulls it toward itself (no-op at best) and
separation pushes it away from itself (a divide-by-near-zero kick in a random direction). A
lone particle in a cell now simply receives no flocking force, which is correct.

Forces are returned, not applied, so `ParticleFlow` keeps ownership of integration order.
"""
from __future__ import annotations

import numpy as np

# Below this many neighbours a cell has nothing meaningful to say about local structure.
_MIN_NEIGHBOURS = 1


def flock_forces(px, py, vx, vy, gw, gh, cell=12.0,
                 cohesion=0.0, alignment=0.0, separation=0.0):
    """Reynolds steering forces from a grid-approximated neighbourhood.

    Parameters
    ----------
    px, py, vx, vy : float32 (n,)
        Particle positions in grid coordinates ``[0, gw) x [0, gh)`` and velocities.
    gw, gh : int
        Grid extent the positions live in.
    cell : float
        Neighbourhood size in grid units — the side of one bin. Larger = broader,
        smoother flocks; smaller = tighter, more local structure.
    cohesion : float
        Steer toward the local centroid.
    alignment : float
        Steer toward the local mean velocity.
    separation : float
        Steer away from the local centroid, scaled by crowding.

    Returns
    -------
    (fx, fy) : float32 (n,) arrays. Zero everywhere when all three gains are 0.
    """
    n = px.shape[0]
    if n == 0 or (cohesion == 0.0 and alignment == 0.0 and separation == 0.0):
        return np.zeros(n, np.float32), np.zeros(n, np.float32)

    cell = max(float(cell), 1e-3)
    cw = max(1, int(np.ceil(gw / cell)))
    ch = max(1, int(np.ceil(gh / cell)))
    ncells = cw * ch

    # Bin. Positions can drift outside the grid between reseeds, so clip rather than
    # trusting the caller — an out-of-range index would corrupt an unrelated cell.
    ix = np.clip((px / cell).astype(np.int32), 0, cw - 1)
    iy = np.clip((py / cell).astype(np.int32), 0, ch - 1)
    idx = iy * cw + ix

    counts = np.bincount(idx, minlength=ncells).astype(np.float32)
    sum_px = np.bincount(idx, weights=px, minlength=ncells)
    sum_py = np.bincount(idx, weights=py, minlength=ncells)
    sum_vx = np.bincount(idx, weights=vx, minlength=ncells)
    sum_vy = np.bincount(idx, weights=vy, minlength=ncells)

    c_self = counts[idx]                      # neighbours incl. self
    others = c_self - 1.0                     # excl. self
    has = others >= _MIN_NEIGHBOURS
    denom = np.where(has, others, 1.0).astype(np.float32)

    # Exclude-self means: (sum - own) / (count - 1). See the module docstring.
    mean_px = ((sum_px[idx] - px) / denom).astype(np.float32)
    mean_py = ((sum_py[idx] - py) / denom).astype(np.float32)
    mean_vx = ((sum_vx[idx] - vx) / denom).astype(np.float32)
    mean_vy = ((sum_vy[idx] - vy) / denom).astype(np.float32)

    fx = np.zeros(n, np.float32)
    fy = np.zeros(n, np.float32)

    if alignment != 0.0:
        fx += alignment * (mean_vx - vx)
        fy += alignment * (mean_vy - vy)

    dx = mean_px - px
    dy = mean_py - py

    if cohesion != 0.0:
        # Normalised by cell size so the gain means the same thing at any cell scale.
        fx += cohesion * dx / cell
        fy += cohesion * dy / cell

    if separation != 0.0:
        # Push directly away from the local centroid, harder the closer it is — the 1/r
        # falloff of the classic rule, softened by `cell` so it cannot explode when two
        # particles coincide.
        dist2 = dx * dx + dy * dy + (0.05 * cell) ** 2
        scale = (cell * cell) / dist2
        fx -= separation * dx * scale / cell
        fy -= separation * dy * scale / cell

    # Particles alone in their cell get nothing — no neighbourhood, no opinion.
    fx *= has
    fy *= has
    return fx.astype(np.float32), fy.astype(np.float32)
