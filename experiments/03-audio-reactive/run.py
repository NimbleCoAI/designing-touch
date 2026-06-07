#!/usr/bin/env python3
"""Experiment 03 — audio-reactive displacement (the CHOP driving the geometry).

A static luminance mound is displaced into a particle relief; sound modulates it per frame:
  bass   -> displacement height (the mound pumps with the kick)
  mid    -> cube size
  treble -> rotation speed (hi-hats spin the dots)

Audio defaults to a deterministic synthetic signal so the whole thing runs and self-verifies
with no mic and no file. Pass a WAV path to react to real audio.

    python run.py --frames 90
    python run.py --audio song.wav --frames 300
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from dtouch import (make_grid, displace_z, random_scale, random_euler,
                    pack_instances, Renderer, make_audio)


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def radial_mound(gx, gy):
    """A static bright-center luminance grid; bass will pump its height."""
    ys, xs = np.mgrid[0:gy, 0:gx].astype(np.float32)
    x = xs / max(gx - 1, 1) - 0.5
    y = ys / max(gy - 1, 1) - 0.5
    r2 = x * x + y * y
    return np.exp(-r2 / (2 * 0.16 ** 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="synthetic", help="'synthetic' or path to a .wav")
    ap.add_argument("--grid", default="150x150")
    ap.add_argument("--res", default="720x720")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth", type=float, default=0.6, help="base displacement (bass adds to it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/audio.mp4")
    args = ap.parse_args()

    gx, gy = parse_wh(args.grid)
    rw, rh = parse_wh(args.res)
    n = gx * gy
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    grid = make_grid(gx, gy)
    luma = radial_mound(gx, gy)
    scale0 = random_scale(n, seed=args.seed)
    euler0 = random_euler(n, seed=args.seed + 1)
    base_size = 0.9 / max(gx, gy)

    audio = make_audio(args.audio, args.frames, args.fps)
    bands = []
    while True:
        b = audio.read()
        if b is None:
            break
        bands.append(b)
    frames = len(bands)

    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=1.0)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)

    bass_series = np.array([b["bass"] for b in bands])
    peak_i = int(bass_series.argmax())
    trough_i = int(bass_series.argmin())
    saved = {}

    spin = 0.0
    for i, b in enumerate(bands):
        depth = args.depth * (1.0 + 1.6 * b["bass"])
        renderer.prog["u_base_size"].value = float(base_size * (1.0 + 0.9 * b["mid"]))
        spin += 0.35 * b["treble"]
        euler = euler0 + np.float32(spin)
        buf = pack_instances(displace_z(grid, luma, depth), scale0, euler)
        frame = renderer.render(buf)
        writer.append_data(frame)
        for tag, idx in (("frame0", 0), ("peak", peak_i), ("trough", trough_i)):
            if i == idx and tag not in saved:
                p = os.path.splitext(args.out)[0] + f"_{tag}.png"
                imageio.imwrite(p, frame)
                saved[tag] = p

    writer.close()
    renderer.release()
    print(f"rendered {frames} frames -> {args.out}")
    print(f"bass peak frame {peak_i}, trough frame {trough_i}")
    for tag, p in saved.items():
        print(f"  {tag} -> {p}")


if __name__ == "__main__":
    main()
