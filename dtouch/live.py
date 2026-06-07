"""Real-time live preview — camera -> flowing particle cloud, in ONE OpenCV window.

The cv2 window paints reliably on macOS (unlike Tk launched headless). The control panel is an
in-frame collapsible sidebar (dtouch.overlay_ui) drawn onto the render with mouse hit-testing,
so it's one window, mouse-driven, and self-verifiable. Quitting is via the window's close (red
X) button — ESC is intentionally ignored (it's muscle-memory for leaving a maximized window).

Detects all-black camera frames (the symptom of iPhone Continuity stealing the built-in camera)
and says so on screen instead of showing a silent blank.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import imageio.v2 as imageio

from .camera import open_camera
from .matte import make_matte
from .particles import ParticleFlow, PALETTES
from .glow import GlowRenderer
from .audio import LiveMic
from .overlay_ui import OverlayUI
from . import presets as _presets

MATTES = ["auto", "motion", "saliency", "person", "edges", "luma"]


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


def live_flow(device="builtin", matte="auto", res=(1280, 720), grid=(384, 216),
              n=90000, mirror=True, seed=1, preset="abstract", audio=False,
              panel=True, show=True, max_frames=None):
    rw, rh = res
    gw, gh = grid
    mw, mh = 400, 225

    cap, cam_name = open_camera(device)
    matte_kind = matte
    mat = make_matte(matte_kind)
    pf = ParticleFlow(n=n, gw=gw, gh=gh, seed=seed)
    glow = GlowRenderer(rw, rh, n, fade=0.90, exposure=1.4)
    all_presets = _presets.load()
    preset_names = list(all_presets.keys())

    def apply_preset(name):
        nonlocal matte_kind, mat
        cfg = all_presets.get(name)
        if not cfg:
            return
        if cfg.get("matte") and cfg["matte"] != matte_kind:
            matte_kind = cfg["matte"]; mat = make_matte(cfg["matte"])
        if "palette" in cfg: pf.palette = cfg["palette"]
        if "spark" in cfg: pf.spark = cfg["spark"]
        if "curl_amp" in cfg: pf.curl_amp = cfg["curl_amp"]
        if "reseed_frac" in cfg: pf.reseed_frac = cfg["reseed_frac"]
        if "base_size" in cfg: pf.base_size = cfg["base_size"]
        if "fade" in cfg: glow.fade = cfg["fade"]
        if "exposure" in cfg: glow.exposure = cfg["exposure"]

    if preset in all_presets:
        apply_preset(preset)

    ui = None
    win = "dtouch - flow"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, rw, rh)
        if panel:
            ui = OverlayUI(rw, rh, preset_names, PALETTES, MATTES,
                           preset=preset, matte=matte_kind, palette=pf.palette)
            ui.sync_from(glow.fade, glow.exposure, pf.spark, pf.curl_amp, pf.base_size,
                         matte_kind, pf.palette)
            ui.mirror = mirror
            ui.audio = audio
            cv2.setMouseCallback(win, ui.on_mouse)

    mic = None
    if audio:
        mic = LiveMic(); mic.start()
    writer = None; rec_path = None
    os.makedirs("out", exist_ok=True)

    t0 = time.time(); fps = 0.0; count = 0
    black_streak = 0
    out = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if max_frames is None:
                    continue
                break
            black_streak = black_streak + 1 if float(frame.mean()) < 3.0 else 0

            if ui is not None:
                if ui.pending_preset:
                    apply_preset(ui.pending_preset)
                    ui.sync_from(glow.fade, glow.exposure, pf.spark, pf.curl_amp,
                                 pf.base_size, matte_kind, pf.palette)
                    ui.pending_preset = None
                if ui.pending_save:
                    name = "mine_%s" % time.strftime("%H%M%S")
                    _presets.save(name, dict(matte=matte_kind, palette=pf.palette,
                                  fade=glow.fade, exposure=ui.exposure, spark=ui.spark,
                                  curl_amp=ui.curl, reseed_frac=pf.reseed_frac, base_size=ui.dot))
                    all_presets = _presets.load()
                    preset_names = list(all_presets.keys())
                    ui.presets = preset_names
                    if name in preset_names:
                        ui.preset_idx = preset_names.index(name)
                    print("saved preset", name)
                    ui.pending_save = False
                if ui.matte_name != matte_kind:
                    matte_kind = ui.matte_name; mat = make_matte(matte_kind)
                pf.palette = ui.palette_name
                pf.curl_amp = ui.curl
                pf.base_size = ui.dot
                glow.fade = ui.fade
                mirror = ui.mirror
                if ui.audio and mic is None:
                    mic = LiveMic(); mic.start()
                elif not ui.audio and mic is not None:
                    mic.stop(); mic = None
                if ui.record and writer is None:
                    rec_path = os.path.join("out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
                    writer = imageio.get_writer(rec_path, fps=24, macro_block_size=8)
                elif not ui.record and writer is not None:
                    writer.close(); print("saved", rec_path); writer = None

            if mirror:
                frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, (mw, mh))
            m = cv2.resize(mat.compute(small), (gw, gh))
            gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                              (gw, gh))
            color = cv2.cvtColor(cv2.resize(small, (gw, gh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # base look from the panel; audio modulates glow/spark on top for this frame
            glow.exposure = ui.exposure if ui is not None else glow.exposure
            pf.spark = ui.spark if ui is not None else pf.spark
            if mic is not None and mic.available:
                lv = mic.levels(); sens = ui.sens if ui is not None else 1.0
                glow.exposure = glow.exposure * (1.0 + 1.6 * sens * lv["bass"])
                pf.spark = pf.spark + 1.2 * sens * lv["treble"]

            pf.update(m, gray, color)
            out = glow.render(pf.render_data())
            if writer is not None:
                writer.append_data(out)
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            count += 1
            if count % 10 == 0:
                now = time.time(); fps = 10.0 / (now - t0); t0 = now

            if show:
                status = f"{fps:4.1f}fps  matte={matte_kind}  color={pf.palette}  cam={cam_name[:16]}"
                if ui is not None:
                    ui.draw(bgr, {"status": status, "black": black_streak > 15})
                else:
                    cv2.putText(bgr, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (90, 220, 120), 1, cv2.LINE_AA)
                cv2.imshow(win, bgr)
                cv2.waitKey(1)  # pump GUI + mouse; ESC intentionally does NOT quit
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed via the red X
            if max_frames is not None and count >= max_frames:
                break
    finally:
        if writer is not None:
            writer.close(); print("saved", rec_path)
        if mic is not None:
            mic.stop()
        cap.release()
        glow.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out


def live(device="builtin", res=(1024, 576), grid=(130, 73), depth=1.3, mirror=True,
         show=True, max_frames=None):
    """Legacy luminance-displaced grid effect (kept as a simple fallback)."""
    from .field import make_grid, displace_z, random_scale, random_euler, pack_instances
    from .render import Renderer
    gx, gy = grid; rw, rh = res; n = gx * gy
    cap, _ = open_camera(device)
    g = make_grid(gx, gy); s = random_scale(n, 0); e = random_euler(n, 1)
    r = Renderer(rw, rh, n, base_size=0.8 / max(gx, gy), depth_scale=depth)
    if show:
        cv2.namedWindow("dtouch - grid", cv2.WINDOW_NORMAL)
    count = 0; out = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if max_frames is None: continue
                break
            if mirror: frame = cv2.flip(frame, 1)
            luma = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (gx, gy)).astype(np.float32) / 255.0
            out = r.render(pack_instances(displace_z(g, luma, depth), s, e))
            if show:
                cv2.imshow("dtouch - grid", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                if cv2.getWindowProperty("dtouch - grid", cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(1)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release(); r.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out
