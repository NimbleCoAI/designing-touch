"""Fluid operator — a 2D stable-fluids solver (Jos Stam, "Stable Fluids").

Eulerian velocity field on a grid, made (approximately) incompressible by Jacobi pressure
projection, advected semi-Lagrangianly. Pure NumPy so the core is unit-testable and runs
headless. Couple it to the particle field by advecting particle positions through the velocity
(see experiment 04) or by displacing height with the advected density.

Arrays are shaped (H, W) = (gy, gx). `u` is x-velocity, `v` is y-velocity. Coordinates for
sampling are in grid units: x in [0, W-1], y in [0, H-1].
"""
from __future__ import annotations

import numpy as np


def bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample `field` (H, W) at fractional (x, y) with clamping. x,y any shape -> same shape."""
    h, w = field.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    fx = x - x0; fy = y - y0
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy) +
            field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = 0.5 * ((u[1:-1, 2:] - u[1:-1, :-2]) + (v[2:, 1:-1] - v[:-2, 1:-1]))
    return div


def project(u: np.ndarray, v: np.ndarray, iters: int = 40):
    """Remove divergence via Jacobi pressure solve (Neumann boundaries)."""
    div = divergence(u, v)
    p = np.zeros_like(u)
    for _ in range(iters):
        p[1:-1, 1:-1] = 0.25 * (p[1:-1, 2:] + p[1:-1, :-2] +
                                p[2:, 1:-1] + p[:-2, 1:-1] - div[1:-1, 1:-1])
        # Neumann: pressure gradient zero at walls
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
    u = u.copy(); v = v.copy()
    u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    return u, v


def advect(field: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Semi-Lagrangian advection: trace each cell back along the velocity and sample."""
    h, w = field.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    bx = xs - dt * u
    by = ys - dt * v
    return bilinear(field, bx, by)


class Fluid2D:
    def __init__(self, h: int, w: int, dissipation: float = 0.99, iters: int = 40):
        self.h, self.w = h, w
        self.u = np.zeros((h, w), dtype=np.float64)
        self.v = np.zeros((h, w), dtype=np.float64)
        self.density = np.zeros((h, w), dtype=np.float64)
        self.dissipation = dissipation
        self.iters = iters

    def add_force(self, fx: np.ndarray, fy: np.ndarray):
        self.u += fx
        self.v += fy

    def add_density(self, d: np.ndarray):
        self.density += d

    def step(self, dt: float = 1.0):
        self.u, self.v = project(self.u, self.v, self.iters)
        u0, v0 = self.u.copy(), self.v.copy()
        self.u = advect(u0, u0, v0, dt)
        self.v = advect(v0, u0, v0, dt)
        self.u, self.v = project(self.u, self.v, self.iters)
        self.density = advect(self.density, self.u, self.v, dt) * self.dissipation

    def sample_velocity(self, x: np.ndarray, y: np.ndarray):
        """Velocity at grid coords (x in [0,W-1], y in [0,H-1]). Returns (vx, vy)."""
        return bilinear(self.u, x, y), bilinear(self.v, x, y)
