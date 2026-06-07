#!/usr/bin/env python3
"""Experiment 02 — shadows.

Same displacement field as experiment 01, but rendered with a depth-from-light shadow map
and a ground plane so the displaced cubes cast real shadows (TouchDesigner Light + Shadow).

    python run.py --frames 90
    python run.py --source webcam --grid 200x112
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio

from dtouch import (make_source, make_grid, displace_z, random_scale,
                    random_euler, pack_instances, ShadowRenderer)


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--grid", default="110x62")
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/shadows.mp4")
    args = ap.parse_args()

    gx, gy = parse_wh(args.grid)
    rw, rh = parse_wh(args.res)
    n = gx * gy
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    grid = make_grid(gx, gy)
    scale = random_scale(n, seed=args.seed)
    euler = random_euler(n, seed=args.seed + 1)
    base_size = 1.5 / max(gx, gy)

    source = make_source(args.source, gx, gy, args.frames)
    renderer = ShadowRenderer(rw, rh, n, base_size=base_size, depth_scale=args.depth)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)
    png = os.path.splitext(args.out)[0] + "_frame0.png"

    count = 0
    while True:
        luma = source.read()
        if luma is None:
            break
        buf = pack_instances(displace_z(grid, luma, args.depth), scale, euler)
        frame = renderer.render(buf)
        writer.append_data(frame)
        if count == 0:
            imageio.imwrite(png, frame)
        count += 1

    writer.close()
    renderer.release()
    print(f"rendered {count} frames -> {args.out}")
    print(f"first frame -> {png}")


if __name__ == "__main__":
    main()
