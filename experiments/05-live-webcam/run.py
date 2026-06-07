#!/usr/bin/env python3
"""Experiment 05 — real-time live preview (interactive).

Default mode 'flow': camera -> subject-agnostic matte (whatever moves/stands out) -> a smoothly
flowing cloud of glowing particles. Works for a dancer, a crowd, a boat — not person-only.
Mode 'grid' is the older luminance-displaced grid.

    python run.py                       # flow, built-in laptop camera, auto matte
    python run.py --matte motion        # key on motion only (great for a dancer)
    python run.py --matte person        # multi-person segmentation
    python run.py --device 1            # a specific camera index
    python run.py --mode grid           # the old displacement-grid effect

Controls (flow): q quit · n cycle matte · m mirror · [ ] trail length · -/= glow · space freeze
"""
from __future__ import annotations

import argparse

from dtouch.live import live, live_flow
from dtouch.camera import list_cameras


def parse_wh(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="flow", choices=["flow", "grid"])
    ap.add_argument("--ui", default="panel", choices=["panel", "keys", "gui"],
                    help="panel = render + slider window (default); keys = render only; "
                         "gui = experimental Tk panel")
    ap.add_argument("--audio", action="store_true", help="start with mic reactivity on")
    ap.add_argument("--device", default="builtin",
                    help="'builtin' (laptop cam), an index, or a name substring")
    ap.add_argument("--matte", default="auto",
                    choices=["auto", "motion", "saliency", "person", "edges", "luma"])
    ap.add_argument("--preset", default=None,
                    help="named look: abstract | portrait | textured | embers | aurora | <your saved>")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--grid", default="416x234")
    ap.add_argument("--particles", type=int, default=200000,
                    help="max particles allocated; the Count slider scales how many render")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("  (AVFoundation enumeration unavailable)")
        for i, name, dtype in cams:
            tag = dtype.replace("AVCaptureDeviceType", "")
            print(f"  [{i}] {name}  ({tag})")
        return

    device = int(args.device) if args.device.isdigit() else args.device
    if args.mode == "grid":
        live(device=device, res=parse_wh(args.res), mirror=not args.no_mirror)
    elif args.ui == "gui":
        from dtouch.gui import run_gui
        run_gui(device=device, res=parse_wh(args.res), grid=parse_wh(args.grid),
                n=args.particles, preset=args.preset or "abstract")
    else:
        live_flow(device=device, matte=args.matte, res=parse_wh(args.res),
                  grid=parse_wh(args.grid), n=args.particles, preset=args.preset or "abstract",
                  audio=args.audio, panel=(args.ui == "panel"), mirror=not args.no_mirror)


if __name__ == "__main__":
    main()
