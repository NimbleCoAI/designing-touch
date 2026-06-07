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

import time

import cv2
import numpy as np

from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
from .render import Renderer


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


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
