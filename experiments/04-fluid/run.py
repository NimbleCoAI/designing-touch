#!/usr/bin/env python3
"""Experiment 04 — fluid dynamics.

The particle field flows through a 2D stable-fluids velocity field. Bright, moving regions of
the video inject swirling force (perpendicular to the luminance gradient, so flow curls around
them) and density. Each frame the particles are advected by the fluid velocity; their height
comes from the advected density. A weak spring pulls them home so the field stays bounded.

    python run.py --frames 120
    python run.py --source clip.mp4 --grid 110x110
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from dtouch import (make_source, random_scale, random_euler, pack_instances,
                    Renderer, Fluid2D)


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--grid", default="100x100")
    ap.add_argument("--res", default="720x720")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", type=float, default=1.1)
    ap.add_argument("--force", type=float, default=9.0, help="swirl injection strength")
    ap.add_argument("--spring", type=float, default=0.015, help="pull particles home")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/fluid.mp4")
    args = ap.parse_args()

    gx, gy = parse_wh(args.grid)
    rw, rh = parse_wh(args.res)
    n = gx * gy
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fluid = Fluid2D(gy, gx, dissipation=0.985, iters=30)
    home_y, home_x = np.mgrid[0:gy, 0:gx].astype(np.float64)
    px = home_x.reshape(-1).copy()
    py = home_y.reshape(-1).copy()
    home_px, home_py = home_x.reshape(-1), home_y.reshape(-1)

    scale0 = random_scale(n, seed=args.seed)
    euler0 = random_euler(n, seed=args.seed + 1)
    base_size = 1.1 / max(gx, gy)

    source = make_source(args.source, gx, gy, args.frames)
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=args.depth)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)

    save_at = {0: "frame0", args.frames // 2: "mid", args.frames - 1: "end"}
    saved = {}
    i = 0
    while True:
        luma = source.read()
        if luma is None:
            break
        L = luma.astype(np.float64)
        gyl, gxl = np.gradient(L)
        # force along iso-contours of luminance -> swirl around bright regions
        fluid.add_force(-gyl * args.force * L, gxl * args.force * L)
        fluid.add_density(L)
        fluid.step(dt=1.0)

        vx, vy = fluid.sample_velocity(px, py)
        px += vx + args.spring * (home_px - px)
        py += vy + args.spring * (home_py - py)
        np.clip(px, 0, gx - 1.001, out=px)
        np.clip(py, 0, gy - 1.001, out=py)

        from dtouch import bilinear
        dens = bilinear(fluid.density, px, py)
        dmax = float(dens.max()) or 1.0
        z = (dens / dmax) * args.depth

        wx = (px / (gx - 1)) * 2.0 - 1.0
        wy = 1.0 - (py / (gy - 1)) * 2.0
        positions = np.stack([wx, wy, z], axis=1).astype(np.float32)
        frame = renderer.render(pack_instances(positions, scale0, euler0))
        writer.append_data(frame)

        if i in save_at and save_at[i] not in saved:
            p = os.path.splitext(args.out)[0] + f"_{save_at[i]}.png"
            imageio.imwrite(p, frame)
            saved[save_at[i]] = p
        i += 1

    writer.close()
    renderer.release()
    print(f"rendered {i} frames -> {args.out}")
    for tag, p in saved.items():
        print(f"  {tag} -> {p}")


if __name__ == "__main__":
    main()
