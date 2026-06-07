"""Real-time live preview — camera -> flowing particle cloud, in an OpenCV window.

Uses the proven cv2 render window (which paints reliably on macOS) plus an on-window slider
panel (cv2 trackbars) so everything is mouse-driven, not keyboard-only. Detects all-black
camera frames (the classic symptom of iPhone Continuity Camera stealing the built-in) and says
so on screen instead of showing a silent blank.
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
from . import presets as _presets

MATTES = ["auto", "motion", "saliency", "person", "edges", "luma"]


def _open_capture(device):
    if isinstance(device, int):
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


# ---- trackbar <-> value mappings (cv2 trackbars are integer, min 0) ----
def _fade_from(p): return 0.50 + p / 100.0          # p 0..48  -> 0.50..0.98
def _fade_to(v): return int(round((v - 0.50) * 100))
def _exp_from(p): return 0.3 + p / 10.0             # p 0..37  -> 0.3..4.0
def _exp_to(v): return int(round((v - 0.3) * 10))
def _unit_from(p): return p / 100.0                 # p 0..100 -> 0..1
def _unit_to(v): return int(round(v * 100))
def _dot_from(p): return (4 + p) / 1000.0           # p 0..16  -> 0.004..0.020
def _dot_to(v): return int(round(v * 1000 - 4))


def live_flow(device="builtin", matte="auto", res=(1280, 720), grid=(256, 144),
              n=45000, mirror=True, seed=1, preset="abstract", audio=False,
              panel=True, show=True, max_frames=None):
    rw, rh = res
    gw, gh = grid
    mw, mh = 320, 180

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
            matte_kind = cfg["matte"]; mat = make_matte(matte_kind)
        if "palette" in cfg: pf.palette = cfg["palette"]
        if "spark" in cfg: pf.spark = cfg["spark"]
        if "curl_amp" in cfg: pf.curl_amp = cfg["curl_amp"]
        if "reseed_frac" in cfg: pf.reseed_frac = cfg["reseed_frac"]
        if "base_size" in cfg: pf.base_size = cfg["base_size"]
        if "fade" in cfg: glow.fade = cfg["fade"]
        if "exposure" in cfg: glow.exposure = cfg["exposure"]

    if preset in all_presets:
        apply_preset(preset)

    win = "dtouch — flow"
    ctl = "dtouch — controls"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, rw, rh)
    if show and panel:
        cv2.namedWindow(ctl, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctl, 420, 360)
        cv2.createTrackbar("preset", ctl, preset_names.index(preset) if preset in preset_names else 0,
                           len(preset_names) - 1, lambda v: None)
        cv2.createTrackbar("matte", ctl, MATTES.index(matte_kind), len(MATTES) - 1, lambda v: None)
        cv2.createTrackbar("color", ctl, PALETTES.index(pf.palette), len(PALETTES) - 1, lambda v: None)
        cv2.createTrackbar("trails", ctl, _fade_to(glow.fade), 48, lambda v: None)
        cv2.createTrackbar("glow", ctl, _exp_to(glow.exposure), 37, lambda v: None)
        cv2.createTrackbar("spark", ctl, _unit_to(pf.spark), 100, lambda v: None)
        cv2.createTrackbar("flow", ctl, _unit_to(pf.curl_amp), 100, lambda v: None)
        cv2.createTrackbar("dot", ctl, _dot_to(pf.base_size), 16, lambda v: None)
        cv2.createTrackbar("mirror", ctl, 1 if mirror else 0, 1, lambda v: None)
        cv2.createTrackbar("audio", ctl, 1 if audio else 0, 1, lambda v: None)
        cv2.createTrackbar("sensitivity", ctl, 10, 30, lambda v: None)
        cv2.createTrackbar("record", ctl, 0, 1, lambda v: None)

    def sync_panel():
        if not (show and panel):
            return
        cv2.setTrackbarPos("matte", ctl, MATTES.index(matte_kind))
        cv2.setTrackbarPos("color", ctl, PALETTES.index(pf.palette))
        cv2.setTrackbarPos("trails", ctl, _fade_to(glow.fade))
        cv2.setTrackbarPos("glow", ctl, _exp_to(glow.exposure))
        cv2.setTrackbarPos("spark", ctl, _unit_to(pf.spark))
        cv2.setTrackbarPos("flow", ctl, _unit_to(pf.curl_amp))
        cv2.setTrackbarPos("dot", ctl, _dot_to(pf.base_size))

    mic = None
    if audio:
        mic = LiveMic(); mic.start()
    writer = None; rec_path = None
    os.makedirs("out", exist_ok=True)

    last_preset_pos = preset_names.index(preset) if preset in preset_names else 0
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

            if show and panel:
                pp = cv2.getTrackbarPos("preset", ctl)
                if pp != last_preset_pos:
                    apply_preset(preset_names[pp]); sync_panel(); last_preset_pos = pp
                mk = MATTES[cv2.getTrackbarPos("matte", ctl)]
                if mk != matte_kind:
                    matte_kind = mk; mat = make_matte(mk)
                pf.palette = PALETTES[cv2.getTrackbarPos("color", ctl)]
                glow.fade = _fade_from(cv2.getTrackbarPos("trails", ctl))
                glow.exposure = _exp_from(cv2.getTrackbarPos("glow", ctl))
                pf.spark = _unit_from(cv2.getTrackbarPos("spark", ctl))
                pf.curl_amp = _unit_from(cv2.getTrackbarPos("flow", ctl))
                pf.base_size = _dot_from(cv2.getTrackbarPos("dot", ctl))
                mirror = bool(cv2.getTrackbarPos("mirror", ctl))
                want_audio = bool(cv2.getTrackbarPos("audio", ctl))
                if want_audio and mic is None:
                    mic = LiveMic(); mic.start()
                elif not want_audio and mic is not None:
                    mic.stop(); mic = None
                want_rec = bool(cv2.getTrackbarPos("record", ctl))
                if want_rec and writer is None:
                    rec_path = os.path.join("out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
                    writer = imageio.get_writer(rec_path, fps=24, macro_block_size=8)
                elif not want_rec and writer is not None:
                    writer.close(); print("saved", rec_path); writer = None

            if mirror:
                frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, (mw, mh))
            m = cv2.resize(mat.compute(small), (gw, gh))
            gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                              (gw, gh))
            color = cv2.cvtColor(cv2.resize(small, (gw, gh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            exposure_base, spark_base = glow.exposure, pf.spark
            if mic is not None and mic.available:
                lv = mic.levels(); sens = (cv2.getTrackbarPos("sensitivity", ctl) / 10.0
                                           if (show and panel) else 1.0)
                glow.exposure = exposure_base * (1.0 + 1.6 * sens * lv["bass"])
                pf.spark = spark_base + 1.2 * sens * lv["treble"]

            pf.update(m, gray, color)
            out = glow.render(pf.render_data())
            glow.exposure, pf.spark = exposure_base, spark_base
            if writer is not None:
                writer.append_data(out)
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            count += 1
            if count % 10 == 0:
                now = time.time(); fps = 10.0 / (now - t0); t0 = now

            if show:
                cv2.putText(bgr, f"{fps:4.1f}fps  matte={matte_kind}  color={pf.palette}  "
                                 f"cam={cam_name[:16]}", (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 220, 120), 1, cv2.LINE_AA)
                if black_streak > 15:
                    cv2.putText(bgr, "CAMERA IS BLACK — disable iPhone Continuity Camera",
                                (12, rh // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (60, 60, 240), 2, cv2.LINE_AA)
                    cv2.putText(bgr, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                                (12, rh // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (120, 120, 240), 1, cv2.LINE_AA)
                if writer is not None:
                    cv2.circle(bgr, (rw - 24, 24), 8, (60, 60, 235), -1)
                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
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
        cv2.namedWindow("dtouch — grid", cv2.WINDOW_NORMAL)
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
                cv2.imshow("dtouch — grid", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release(); r.release()
        if show:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return count, out
