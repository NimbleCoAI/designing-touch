"""Tkinter control panel for the live flow — collapsible sections instead of keyboard-only.

One window: the live particle render on the left, a sidebar of collapsible sections on the
right (Templates, Source, Look, Record). Sliders/dropdowns/buttons drive the same engine the
keyboard build uses; the render loop runs on Tk's event loop via `after`, so it's single
threaded and responsive. Templates = presets (apply / save your own).

Falls back gracefully: if Tk can't open a display, raise so the caller can use the cv2 UI.
"""
from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk
import imageio.v2 as imageio

from .camera import open_camera, list_cameras
from .matte import make_matte
from .particles import ParticleFlow, PALETTES
from .glow import GlowRenderer
from .audio import LiveMic
from . import presets as _presets

MATTES = ["auto", "motion", "saliency", "person", "edges", "luma"]


class FlowGUI:
    def __init__(self, root, device="builtin", res=(1100, 620), grid=(256, 144),
                 n=45000, preset="abstract", display_w=820):
        self.root = root
        self.rw, self.rh = res
        self.gw, self.gh = grid
        self.n = n
        self.mw, self.mh = 320, 180
        self.display_w = display_w
        self.display_h = int(display_w * self.rh / self.rw)

        self.cap, self.cam_name = open_camera(device)
        self.matte_kind = "auto"
        self.mat = make_matte(self.matte_kind)
        self.pf = ParticleFlow(n=n, gw=self.gw, gh=self.gh, seed=1)
        self.glow = GlowRenderer(self.rw, self.rh, n, fade=0.90, exposure=1.4)
        self.presets = _presets.load()

        self.mirror = tk.BooleanVar(value=True)
        self.var_matte = tk.StringVar(value=self.matte_kind)
        self.var_palette = tk.StringVar(value="ice")
        self.var_cam = tk.StringVar(value=self.cam_name)
        self.var_preset = tk.StringVar(value=preset)
        self.fade = tk.DoubleVar(value=0.90)
        self.exposure = tk.DoubleVar(value=1.4)
        self.spark = tk.DoubleVar(value=0.35)
        self.curl = tk.DoubleVar(value=0.5)
        self.dotsize = tk.DoubleVar(value=0.011)
        self.fps_text = tk.StringVar(value="")

        self.mic = None
        self.audio_on = tk.BooleanVar(value=False)
        self.audio_sens = tk.DoubleVar(value=1.0)
        self.writer = None
        self.rec_path = None
        self._t0 = time.time()
        self._count = 0
        self._frozen = False
        self._last_small = None

        self._build_ui()
        if preset in self.presets:
            self._apply_preset(preset)
        # size the window and force a first layout/paint BEFORE starting the heavy loop,
        # otherwise the render loop can starve Tk's initial draw (blank window).
        self.root.geometry(f"{self.display_w + 320}x{self.display_h + 70}")
        self.root.update_idletasks()
        self.root.after(300, self._step)
        if os.environ.get("DTOUCH_SELFTEST"):
            self.root.after(2500, self._selftest)

    def _selftest(self):
        v, s = self.video, self.sidebar
        print("SELFTEST video mapped=%s %dx%d | sidebar mapped=%s %dx%d kids=%d | frames=%d"
              % (v.winfo_ismapped(), v.winfo_width(), v.winfo_height(),
                 s.winfo_ismapped(), s.winfo_width(), s.winfo_height(),
                 len(s.winfo_children()), self._count), flush=True)

    # ---------- UI ----------
    def _section(self, parent, title, expanded=True):
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=(0, 6))
        body = ttk.Frame(outer, padding=(10, 4))
        state = {"open": expanded}
        btn = ttk.Button(outer)

        def toggle():
            state["open"] = not state["open"]
            if state["open"]:
                body.pack(fill="x"); btn.config(text="▾  " + title)
            else:
                body.forget(); btn.config(text="▸  " + title)
        btn.config(text="▾  " + title, command=toggle, style="Section.TButton")
        btn.pack(fill="x")
        if expanded:
            body.pack(fill="x")
        return body

    def _slider(self, parent, label, var, lo, hi):
        row = ttk.Frame(parent); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=10).pack(side="left")
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal").pack(
            side="left", fill="x", expand=True)

    def _build_ui(self):
        self.root.title("dtouch — flow")
        self.root.configure(bg="#0c0c0e")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Section.TButton", anchor="w", font=("Helvetica", 12, "bold"))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = tk.Label(left, bg="black")
        self.video.pack()
        ttk.Label(left, textvariable=self.fps_text).pack(anchor="w", pady=(4, 0))

        side = ttk.Frame(main, width=300)
        side.pack(side="right", fill="y", padx=(10, 0))
        self.sidebar = side

        # Templates
        s = self._section(side, "Templates")
        ttk.Label(s, text="Preset").pack(anchor="w")
        self.preset_box = ttk.Combobox(s, textvariable=self.var_preset,
                                       values=list(self.presets.keys()), state="readonly")
        self.preset_box.pack(fill="x", pady=2)
        self.preset_box.bind("<<ComboboxSelected>>",
                             lambda e: self._apply_preset(self.var_preset.get()))
        ttk.Button(s, text="Save current as…", command=self._save_preset).pack(fill="x", pady=2)

        # Source
        s = self._section(side, "Source")
        ttk.Label(s, text="Matte (what becomes particles)").pack(anchor="w")
        mb = ttk.Combobox(s, textvariable=self.var_matte, values=MATTES, state="readonly")
        mb.pack(fill="x", pady=2)
        mb.bind("<<ComboboxSelected>>", lambda e: self._set_matte(self.var_matte.get()))
        ttk.Label(s, text="Camera").pack(anchor="w")
        cams = [n for _, n, _ in list_cameras()] or [self.cam_name]
        cb = ttk.Combobox(s, textvariable=self.var_cam, values=cams, state="readonly")
        cb.pack(fill="x", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_camera(self.var_cam.get()))
        ttk.Checkbutton(s, text="Mirror (selfie)", variable=self.mirror).pack(anchor="w", pady=2)

        # Look
        s = self._section(side, "Look")
        ttk.Label(s, text="Color").pack(anchor="w")
        pb = ttk.Combobox(s, textvariable=self.var_palette, values=PALETTES, state="readonly")
        pb.pack(fill="x", pady=2)
        pb.bind("<<ComboboxSelected>>", lambda e: setattr(self.pf, "palette", self.var_palette.get()))
        self._slider(s, "Trails", self.fade, 0.5, 0.985)
        self._slider(s, "Glow", self.exposure, 0.3, 3.0)
        self._slider(s, "Spark", self.spark, 0.0, 1.0)
        self._slider(s, "Flow", self.curl, 0.0, 1.0)
        self._slider(s, "Dot size", self.dotsize, 0.004, 0.02)

        # Audio
        s = self._section(side, "Audio", expanded=True)
        ttk.Checkbutton(s, text="React to sound (mic)", variable=self.audio_on,
                        command=self._toggle_audio).pack(anchor="w", pady=2)
        self._slider(s, "Sensitivity", self.audio_sens, 0.0, 3.0)
        self.audio_label = ttk.Label(s, text="off")
        self.audio_label.pack(anchor="w")

        # Record
        s = self._section(side, "Record", expanded=True)
        self.rec_btn = ttk.Button(s, text="● Start recording", command=self._toggle_record)
        self.rec_btn.pack(fill="x", pady=2)
        self.rec_label = ttk.Label(s, text="")
        self.rec_label.pack(anchor="w")

        ttk.Button(side, text="Quit", command=self._quit).pack(fill="x", pady=(8, 0))

    # ---------- actions ----------
    def _set_matte(self, kind):
        self.matte_kind = kind
        self.mat = make_matte(kind)

    def _set_camera(self, name):
        try:
            cap, resolved = open_camera(name)
        except Exception:
            return
        if self.cap is not None:
            self.cap.release()
        self.cap, self.cam_name = cap, resolved

    def _apply_preset(self, name):
        cfg = self.presets.get(name)
        if not cfg:
            return
        if cfg.get("matte"):
            self.var_matte.set(cfg["matte"]); self._set_matte(cfg["matte"])
        if "palette" in cfg:
            self.var_palette.set(cfg["palette"]); self.pf.palette = cfg["palette"]
        if "fade" in cfg: self.fade.set(cfg["fade"])
        if "exposure" in cfg: self.exposure.set(cfg["exposure"])
        if "spark" in cfg: self.spark.set(cfg["spark"])
        if "curl_amp" in cfg: self.curl.set(cfg["curl_amp"])
        if "base_size" in cfg: self.dotsize.set(cfg["base_size"])
        if "reseed_frac" in cfg: self.pf.reseed_frac = cfg["reseed_frac"]

    def _save_preset(self):
        name = simpledialog.askstring("Save preset", "Name this look:", parent=self.root)
        if not name:
            return
        cfg = dict(matte=self.matte_kind, palette=self.var_palette.get(),
                   fade=self.fade.get(), exposure=self.exposure.get(), spark=self.spark.get(),
                   curl_amp=self.curl.get(), reseed_frac=self.pf.reseed_frac,
                   base_size=self.dotsize.get())
        _presets.save(name, cfg)
        self.presets[name] = cfg
        self.preset_box.config(values=list(self.presets.keys()))
        self.var_preset.set(name)

    def _toggle_audio(self):
        if self.audio_on.get():
            self.mic = LiveMic()
            if self.mic.start():
                self.audio_label.config(text="listening")
            else:
                self.audio_on.set(False)
                self.audio_label.config(text="mic unavailable — grant Microphone permission")
        else:
            if self.mic is not None:
                self.mic.stop()
            self.audio_label.config(text="off")

    def _toggle_record(self):
        if self.writer is None:
            os.makedirs("out", exist_ok=True)
            self.rec_path = os.path.join("out", "rec_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
            self.writer = imageio.get_writer(self.rec_path, fps=24, macro_block_size=8)
            self.rec_btn.config(text="■ Stop recording")
            self.rec_label.config(text="● REC")
        else:
            self.writer.close()
            self.rec_label.config(text="saved %s" % os.path.basename(self.rec_path))
            self.writer = None
            self.rec_btn.config(text="● Start recording")

    def _quit(self):
        try:
            if self.writer is not None:
                self.writer.close()
            self.cap.release()
            self.glow.release()
        finally:
            self.root.destroy()

    # ---------- render loop ----------
    def _step(self):
        try:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.mirror.get():
                    frame = cv2.flip(frame, 1)
                small = cv2.resize(frame, (self.mw, self.mh))
                m = cv2.resize(self.mat.compute(small), (self.gw, self.gh))
                gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                                  (self.gw, self.gh))
                color = cv2.cvtColor(cv2.resize(small, (self.gw, self.gh)),
                                     cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                self.pf.spark = self.spark.get()
                self.pf.curl_amp = self.curl.get()
                self.pf.base_size = self.dotsize.get()
                self.glow.fade = self.fade.get()
                self.glow.exposure = self.exposure.get()
                self.pf.update(m, gray, color)
                rgb = self.glow.render(self.pf.render_data())
                if self.writer is not None:
                    self.writer.append_data(rgb)
                disp = cv2.resize(rgb, (self.display_w, self.display_h))
                img = ImageTk.PhotoImage(Image.fromarray(disp))
                self.video.configure(image=img)
                self.video.image = img
                self._count += 1
                if self._count % 10 == 0:
                    now = time.time()
                    self.fps_text.set(f"{10.0/(now-self._t0):.1f} fps   camera: {self.cam_name}")
                    self._t0 = now
        except Exception as e:
            self.fps_text.set(f"err: {e}")
            print("STEP ERR:", e, flush=True)
        self.root.after(15, self._step)


def run_gui(device="builtin", res=(1100, 620), grid=(256, 144), n=45000, preset="abstract"):
    root = tk.Tk()
    FlowGUI(root, device=device, res=res, grid=grid, n=n, preset=preset)
    root.mainloop()
