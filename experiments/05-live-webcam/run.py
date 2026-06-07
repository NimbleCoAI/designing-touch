#!/usr/bin/env python3
"""Experiment 05 — live webcam preview (real-time, interactive).

Turns the live camera into a displaced particle system in a window. This is the interactive
counterpart to the file-rendering experiments.

    python run.py                      # live webcam
    python run.py --source clip.mp4    # a video file, looped through the same pipeline
    python run.py --grid 110x62        # coarser grid -> higher fps

Controls: q/ESC quit · +/- depth · [ ] dot size · m mirror · i invert · space freeze
"""
from __future__ import annotations

import argparse

from dtouch.live import live


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="webcam index (e.g. 0) or a path to a video file")
    ap.add_argument("--grid", default="130x73")
    ap.add_argument("--res", default="1024x576")
    ap.add_argument("--depth", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    device = int(args.source) if args.source.isdigit() else args.source
    live(device=device, grid=parse_wh(args.grid), res=parse_wh(args.res),
         depth=args.depth, seed=args.seed, mirror=not args.no_mirror)


if __name__ == "__main__":
    main()
