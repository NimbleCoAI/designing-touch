#!/usr/bin/env python3
"""Displacement system — end-to-end CLI.

Pipeline (TouchDesigner graph, as code):
    source (TOP) -> luma grid -> make_grid + displace_z (TOP->POP, height) ->
    random scale + random rotation (two randoms) -> pack -> instanced cubes (Copy SOP) ->
    directional light + depth (Light/Render) -> MP4 + PNG

Examples:
    python run.py                                  # synthetic source, writes out/displace.mp4 + frame0
    python run.py --source webcam --frames 120     # live webcam (needs Camera permission)
    python run.py --source clip.mp4 --grid 200x112
    python run.py --source photo.jpg --frames 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from dtouch import (make_source, make_grid, displace_z, random_scale,
                    random_euler, pack_instances, Renderer)


def parse_grid(s: str):
    gx, gy = s.lower().split("x")
    return int(gx), int(gy)


def main():
    ap = argparse.ArgumentParser(description="Video/synthetic -> displaced particle system")
    ap.add_argument("--source", default="synthetic",
                    help="'synthetic' | 'webcam' | path to image/video")
    ap.add_argument("--grid", default="160x90", help="point grid WxH, e.g. 160x90")
    ap.add_argument("--res", default="960x540", help="output resolution WxH")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", type=float, default=1.2, help="Z displacement scale (height)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/displace.mp4")
    args = ap.parse_args()

    gx, gy = parse_grid(args.grid)
    rw, rh = parse_grid(args.res)
    n = gx * gy
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # static field data: base grid + the two stable randoms (generated once)
    grid = make_grid(gx, gy, extent=2.0)
    scale = random_scale(n, seed=args.seed)
    euler = random_euler(n, seed=args.seed + 1)
    base_size = 0.7 / max(gx, gy)  # cube size relative to grid spacing

    source = make_source(args.source, gx, gy, args.frames)
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=args.depth)

    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)
    png_path = os.path.splitext(args.out)[0] + "_frame0.png"

    count = 0
    while True:
        luma = source.read()
        if luma is None:
            break
        positions = displace_z(grid, luma, depth_scale=args.depth)
        buf = pack_instances(positions, scale, euler)
        frame = renderer.render(buf)
        writer.append_data(frame)
        if count == 0:
            imageio.imwrite(png_path, frame)
        count += 1

    writer.close()
    renderer.release()
    print(f"rendered {count} frames -> {args.out}")
    print(f"first frame -> {png_path}")


if __name__ == "__main__":
    main()
