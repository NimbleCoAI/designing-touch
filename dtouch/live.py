"""Real-time live preview — webcam (or video) -> particle displacement, in a window.

Reuses the offscreen GPU `Renderer`: each frame is captured, turned into a displaced
instanced-cube render, and blitted to an OpenCV window. Interactive, ~30 fps at a modest grid.

Controls (window focused):
  q / ESC   quit
  + / -     more / less displacement depth
  [ / ]     smaller / larger dots
  m         toggle mirror (selfie view)
  i         invert (dark drives height instead of bright)
  space     freeze / unfreeze the incoming video

This is the interactive counterpart to the file-rendering experiments. It needs a display and
(for webcam) camera permission, so it is deliberately separate from the headless engine.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import imageio.v2 as imageio

from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
from .render import Renderer
from .camera import open_camera
from .matte import make_matte
from .particles import ParticleFlow
from .glow import GlowRenderer


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


_MATTE_CYCLE = ["auto", "motion", "saliency", "person", "edges"]


def live_flow(device="builtin", matte="auto", res=(1280, 720), grid=(256, 144),
              n=45000, fade=0.90, exposure=1.4, mirror=True, seed=1,
              show=True, max_frames=None):
    """Real-time: camera -> subject-agnostic matte -> flowing particle cloud (glow).

    `show=False`/`max_frames=N` is the headless smoke mode used to verify before launching.
    Returns (frames_processed, last_frame_or_None).
    """
    rw, rh = res
    gw, gh = grid
    mw, mh = 320, 180  # matte working resolution (cheap)

    cap, cam_name = open_camera(device)
    matte_kind = matte
    mat = make_matte(matte_kind)
    pf = ParticleFlow(n=n, gw=gw, gh=gh, seed=seed)
    glow = GlowRenderer(rw, rh, n, fade=fade, exposure=exposure)

    win = "dtouch — flow"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, rw, rh)

    frozen = False
    last_small = None
    t0 = time.time(); fps = 0.0; count = 0
    out = None
    writer = None; rec_path = None
    os.makedirs("out", exist_ok=True)
    try:
        while True:
            if not frozen or last_small is None:
                ok, frame = cap.read()
                if not ok:
                    if max_frames is None:
                        continue
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                last_small = cv2.resize(frame, (mw, mh))
            small = last_small

            m = cv2.resize(mat.compute(small), (gw, gh))
            gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                              (gw, gh))
            pf.update(m, gray)
            out = glow.render(pf.render_data())
            if writer is not None:
                writer.append_data(out)
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            count += 1
            if count % 10 == 0:
                now = time.time(); fps = 10.0 / (now - t0); t0 = now

            if show:
                cv2.putText(bgr, f"{fps:4.1f}fps  matte={matte_kind}  color={pf.palette}  "
                                 f"cam={cam_name[:16]}  fade={glow.fade:.2f}",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 220, 120), 1, cv2.LINE_AA)
                cv2.putText(bgr, "q quit  n matte  c color  m mirror  [ ] trail  -/= glow  r REC  space freeze",
                            (12, rh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 180, 110), 1,
                            cv2.LINE_AA)
                if writer is not None:
                    cv2.circle(bgr, (rw - 24, 24), 8, (60, 60, 235), -1)
                    cv2.putText(bgr, "REC", (rw - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (60, 60, 235), 2, cv2.LINE_AA)
                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('n'):
                    matte_kind = _MATTE_CYCLE[(_MATTE_CYCLE.index(matte_kind) + 1) % len(_MATTE_CYCLE)]
                    mat = make_matte(matte_kind)
                elif key == ord('c'):
                    pf.cycle_palette()
                elif key == ord('m'):
                    mirror = not mirror
                elif key == ord('['):
                    glow.fade = max(glow.fade - 0.02, 0.5)
                elif key == ord(']'):
                    glow.fade = min(glow.fade + 0.02, 0.985)
                elif key in (ord('-'), ord('_')):
                    glow.exposure = max(glow.exposure - 0.1, 0.3)
                elif key in (ord('='), ord('+')):
                    glow.exposure = min(glow.exposure + 0.1, 4.0)
                elif key == ord('r'):
                    if writer is None:
                        rec_path = os.path.join("out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
                        writer = imageio.get_writer(rec_path, fps=24, macro_block_size=8)
                    else:
                        writer.close(); print("saved recording ->", rec_path); writer = None
                elif key == ord(' '):
                    frozen = not frozen
            if max_frames is not None and count >= max_frames:
                break
    finally:
        if writer is not None:
            writer.close(); print("saved recording ->", rec_path)
        cap.release()
        glow.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out


def live(device=0, grid=(140, 79), res=(1100, 620), depth=1.3, seed=0,
         mirror=True, show=True, max_frames=None):
    """Run the live displacement preview. Returns the number of frames processed.

    `show=False` / `max_frames=N` is a headless smoke mode: it runs the real capture +
    render path for N frames without opening a window (used to verify before launching).
    """
    gx, gy = grid
    rw, rh = res
    n = gx * gy

    cap = _open_capture(device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open capture device {device!r}")

    grid_pos = make_grid(gx, gy)
    scale = random_scale(n, seed=seed)
    euler = random_euler(n, seed=seed + 1)
    base_size = 0.8 / max(gx, gy)
    renderer = Renderer(rw, rh, n, base_size=base_size, depth_scale=depth)

    win = "dtouch — live"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, rw, rh)

    invert = False
    frozen = False
    last_luma = None
    t0 = time.time()
    fps = 0.0
    count = 0
    try:
        while True:
            if not frozen or last_luma is None:
                ok, frame = cap.read()
                if not ok:
                    if max_frames is None:
                        continue
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (gx, gy), interpolation=cv2.INTER_AREA)
                luma = small.astype(np.float32) / 255.0
                if invert:
                    luma = 1.0 - luma
                last_luma = luma
            else:
                luma = last_luma

            buf = pack_instances(displace_z(grid_pos, luma, depth), scale, euler)
            img = renderer.render(buf)              # RGB
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            count += 1
            if count % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - t0)
                t0 = now

            if show:
                cv2.putText(bgr, f"{fps:4.1f} fps  depth={depth:.2f}  "
                                 f"[q]uit +/- depth  [ ] size  m mirror  i invert  space freeze",
                            (12, rh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1,
                            cv2.LINE_AA)
                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key in (ord('+'), ord('=')):
                    depth = min(depth + 0.1, 4.0)
                elif key in (ord('-'), ord('_')):
                    depth = max(depth - 0.1, 0.0)
                elif key == ord(']'):
                    base_size *= 1.1; renderer.prog["u_base_size"].value = float(base_size)
                elif key == ord('['):
                    base_size /= 1.1; renderer.prog["u_base_size"].value = float(base_size)
                elif key == ord('m'):
                    mirror = not mirror
                elif key == ord('i'):
                    invert = not invert
                elif key == ord(' '):
                    frozen = not frozen

            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()
        renderer.release()
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    return count
