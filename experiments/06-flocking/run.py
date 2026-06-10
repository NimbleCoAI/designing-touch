#!/usr/bin/env python3
"""Flocking system — emergent order from local rules, rendered as oriented instances.

Reynolds boids (separation, alignment, cohesion) in 3D, with a soft spherical
boundary and a slow global swirl. No central controller — every agent sees only
its neighbours within a radius, and coherent motion falls out of the three local
forces. The same shape as a swarm of agents tending a shared commons: order that
is produced, not imposed.

Pipeline (TouchDesigner graph, as code):
    boid state (POP: pos+vel) -> neighbour forces (separation/alignment/cohesion)
    -> integrate -> heading euler + speed->scale -> pack -> instanced cubes (Copy SOP)
    -> directional light + depth (Light/Render) -> MP4 + PNG

Examples:
    python run.py                              # 700 boids, 240 frames -> out/flock.mp4
    python run.py --boids 1200 --frames 600
    python run.py --neighbor 0.5 --separation 1.8 --swirl 0.25
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from dtouch import pack_instances, Renderer


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def _limit(v, max_len):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_len / np.maximum(n, 1e-9))
    return v * scale


def heading_to_euler(vel):
    """Euler (x_pitch, y, z_yaw) so a cube points along its velocity. Yaw from the
    XY heading, pitch from the Z climb — enough for the instances to read as a
    directed shoal rather than a static cloud."""
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    yaw = np.arctan2(vy, vx)
    pitch = np.arctan2(vz, np.sqrt(vx * vx + vy * vy) + 1e-9)
    euler = np.zeros((vel.shape[0], 3), dtype=np.float32)
    euler[:, 0] = pitch
    euler[:, 2] = yaw
    return euler


def step(pos, vel, p):
    """One flocking update. O(n^2) neighbour pass — fine for a few thousand boids,
    and honest about where the cost is rather than hiding it behind a grid."""
    n = pos.shape[0]
    # pairwise offsets + distances
    diff = pos[:, None, :] - pos[None, :, :]          # (n, n, 3): i minus j
    dist2 = np.sum(diff * diff, axis=2)               # (n, n)
    np.fill_diagonal(dist2, np.inf)
    within = dist2 < (p.neighbor * p.neighbor)        # neighbour mask

    counts = np.maximum(within.sum(axis=1, keepdims=True), 1)

    # cohesion: steer toward neighbour centroid
    centroid = (within[:, :, None] * pos[None, :, :]).sum(axis=1) / counts
    cohesion = _normalize(centroid - pos)

    # alignment: match neighbour mean heading
    mean_vel = (within[:, :, None] * vel[None, :, :]).sum(axis=1) / counts
    alignment = _normalize(mean_vel)

    # separation: push from too-close neighbours, weighted by 1/dist
    close = dist2 < (p.sep_radius * p.sep_radius)
    np.fill_diagonal(close, False)
    w = close / (dist2 + 1e-6)
    sep = (w[:, :, None] * diff).sum(axis=1)          # diff already points i<-j
    separation = _normalize(sep)

    # slow global swirl about Z — a breath of shared weather over local rules
    swirl = np.stack([-pos[:, 1], pos[:, 0], np.zeros(n)], axis=1)
    swirl = _normalize(swirl) * p.swirl

    # soft spherical containment
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    inward = -_normalize(pos) * np.maximum(0.0, r - p.bound) * 2.5

    acc = (p.cohesion * cohesion + p.alignment * alignment +
           p.separation * separation + swirl + inward)
    vel = _limit(vel + acc * p.dt, p.max_speed)
    vel = np.where(np.linalg.norm(vel, axis=1, keepdims=True) < p.min_speed,
                   _normalize(vel) * p.min_speed, vel)
    pos = pos + vel * p.dt
    if p.recenter:
        pos = pos - pos.mean(axis=0, keepdims=True)  # camera tracks the shoal's centroid
    return pos, vel


def main():
    ap = argparse.ArgumentParser(description="Emergent flocking -> instanced particle render")
    ap.add_argument("--boids", type=int, default=700)
    ap.add_argument("--res", default="960x540", help="output resolution WxH")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--neighbor", type=float, default=0.45, help="neighbour radius")
    ap.add_argument("--sep-radius", type=float, default=0.2, help="separation radius")
    ap.add_argument("--cohesion", type=float, default=0.55)
    ap.add_argument("--alignment", type=float, default=1.25)
    ap.add_argument("--separation", type=float, default=2.0)
    ap.add_argument("--swirl", type=float, default=0.13)
    ap.add_argument("--bound", type=float, default=1.5, help="containment sphere radius")
    ap.add_argument("--max-speed", type=float, default=0.9)
    ap.add_argument("--min-speed", type=float, default=0.3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=True,
                    help="keep the shoal centroid framed (camera tracks it)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "flock.mp4"))
    p = ap.parse_args()

    rw, rh = (int(x) for x in p.res.lower().split("x"))
    n = p.boids
    os.makedirs(os.path.dirname(p.out) or ".", exist_ok=True)

    rng = np.random.default_rng(p.seed)
    pos = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)
    vel = _normalize(rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)) * p.min_speed

    base_size = 1.4 / np.sqrt(n)            # cube size shrinks as the shoal grows
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=1.0, extent=1.7)

    writer = imageio.get_writer(p.out, fps=p.fps, macro_block_size=8)
    png_path = os.path.splitext(p.out)[0] + "_frame0.png"
    mid_png = os.path.splitext(p.out)[0] + "_mid.png"

    for i in range(p.frames):
        pos, vel = step(pos, vel, p)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        # length-stretch along heading via per-axis scale: faster boids read longer
        s = np.repeat(0.7 + 1.6 * (speed / p.max_speed), 3, axis=1).astype(np.float32)
        s[:, 0] *= 1.8                       # elongate the leading axis
        euler = heading_to_euler(vel)
        frame = renderer.render(pack_instances(pos.astype(np.float32), s, euler))
        writer.append_data(frame)
        if i == 0:
            imageio.imwrite(png_path, frame)
        if i == p.frames // 2:
            imageio.imwrite(mid_png, frame)

    writer.close()
    renderer.release()
    print(f"rendered {p.frames} frames of {n} boids -> {p.out}")
    print(f"frames -> {png_path}, {mid_png}")


if __name__ == "__main__":
    main()
