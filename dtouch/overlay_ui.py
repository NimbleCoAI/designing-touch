"""In-frame control panel — a collapsible sidebar drawn onto the render, one window.

cv2 highgui can't host a real side menu, and Tk won't paint when launched headless on macOS.
So this is an immediate-mode GUI drawn directly onto the rendered frame with cv2 primitives,
with mouse hit-testing via setMouseCallback. It lives in the same window as the video, collapses
to a small button, and because it's just drawn pixels it can be rendered headless and inspected.

The panel owns the tunable values (fade, exposure, spark, curl, dot, mirror, audio, sens,
record) and the selected preset/matte/palette; the live loop reads them each frame and applies.
Preset clicks set `pending_preset` for the loop to apply (it touches the engine), then the loop
calls `sync_from` to reflect the applied values back into the sliders.
"""
from __future__ import annotations

import cv2
import numpy as np

BG = (24, 22, 20)
PANEL = (34, 32, 30)
INK = (210, 220, 215)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
TRACK = (70, 74, 72)
HANDLE = (150, 220, 165)
RED = (70, 70, 235)


class OverlayUI:
    def __init__(self, w, h, presets, palettes, mattes,
                 preset="abstract", matte="auto", palette="ice"):
        self.w, self.h = w, h
        self.presets = presets
        self.palettes = palettes
        self.mattes = mattes
        self.preset_idx = presets.index(preset) if preset in presets else 0
        self.matte_idx = mattes.index(matte) if matte in mattes else 0
        self.palette_idx = palettes.index(palette) if palette in palettes else 0
        self.fade = 0.90
        self.exposure = 1.4
        self.spark = 0.35
        self.curl = 0.5
        self.dot = 0.011
        self.mirror = True
        self.audio = False
        self.sens = 1.0
        self.record = False
        self.open = True
        self.pending_preset = None
        self.pending_save = False
        self.panel_w = 280
        self._hot = []
        self._drag = None  # (attr, x0, x1, lo, hi)

    # ----- public values -----
    @property
    def matte_name(self): return self.mattes[self.matte_idx]
    @property
    def palette_name(self): return self.palettes[self.palette_idx]
    @property
    def preset_name(self): return self.presets[self.preset_idx]

    def sync_from(self, fade, exposure, spark, curl, dot, matte, palette):
        """Reflect engine state back into the panel (after a preset is applied)."""
        self.fade, self.exposure, self.spark, self.curl, self.dot = fade, exposure, spark, curl, dot
        if matte in self.mattes: self.matte_idx = self.mattes.index(matte)
        if palette in self.palettes: self.palette_idx = self.palettes.index(palette)

    # ----- drawing -----
    def _text(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _slider(self, img, label, attr, x, y, w, lo, hi):
        val = getattr(self, attr)
        self._text(img, label, x, y - 6, DIM, 0.42)
        tx0, tx1 = x, x + w
        cv2.line(img, (tx0, y + 8), (tx1, y + 8), TRACK, 3, cv2.LINE_AA)
        hx = int(tx0 + (val - lo) / (hi - lo) * w)
        cv2.circle(img, (hx, y + 8), 6, HANDLE, -1, cv2.LINE_AA)
        self._text(img, f"{val:.2f}", tx1 - 30, y - 6, DIM, 0.4)
        self._hot.append(((tx0 - 6, y - 2, tx1 + 6, y + 18), "slider", (attr, tx0, tx1, lo, hi)))
        return y + 30

    def _cycle(self, img, label, value, key, x, y, w):
        self._text(img, label, x, y, DIM, 0.42)
        self._text(img, f"< {value} >", x + 70, y, INK, 0.46)
        self._hot.append(((x + 60, y - 14, x + 86, y + 6), "cycle", (key, -1)))
        self._hot.append(((x + w - 26, y - 14, x + w, y + 6), "cycle", (key, +1)))
        return y + 26

    def _button(self, img, label, key, x, y, w, active=False):
        col = ACC if active else PANEL
        cv2.rectangle(img, (x, y - 16), (x + w, y + 8), col, -1)
        cv2.rectangle(img, (x, y - 16), (x + w, y + 8), TRACK, 1)
        self._text(img, label, x + 8, y, (20, 20, 20) if active else INK, 0.44)
        self._hot.append(((x, y - 16, x + w, y + 8), "button", key))
        return y + 32

    def draw(self, frame, info):
        self._hot = []
        h, w = frame.shape[:2]
        if not self.open:
            # collapsed: a small hamburger button top-right
            cv2.rectangle(frame, (w - 46, 12), (w - 12, 40), PANEL, -1)
            for yy in (20, 26, 32):
                cv2.line(frame, (w - 40, yy), (w - 18, yy), INK, 2, cv2.LINE_AA)
            self._hot.append(((w - 46, 12, w - 12, 40), "collapse", None))
            self._draw_status(frame, info)
            return frame

        px = w - self.panel_w
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, 0), (w, h), PANEL, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        x = px + 16
        cw = self.panel_w - 32
        y = 30
        self._text(frame, "dtouch", x, y, ACC, 0.6, 2)
        # collapse arrow (click to fold the panel away)
        cv2.rectangle(frame, (w - 34, 14), (w - 14, 34), TRACK, -1)
        self._text(frame, ">", w - 30, 30, INK, 0.6, 2)
        self._hot.append(((w - 40, 12, w - 12, 36), "collapse", None))
        y += 18

        self._text(frame, "TEMPLATES", x, y, DIM, 0.4); y += 16
        for i, name in enumerate(self.presets):
            active = (i == self.preset_idx)
            self._text(frame, ("• " if active else "  ") + name, x, y,
                       ACC if active else INK, 0.46)
            self._hot.append(((x, y - 12, px + self.panel_w - 16, y + 6), "preset", i))
            y += 20
        y = self._button(frame, "Save current look", "save", x, y + 4, cw)

        self._text(frame, "SOURCE", x, y, DIM, 0.4); y += 18
        y = self._cycle(frame, "matte", self.matte_name, "matte", x, y, cw)
        y += 4

        self._text(frame, "LOOK", x, y, DIM, 0.4); y += 18
        y = self._cycle(frame, "color", self.palette_name, "color", x, y, cw)
        y += 6
        y = self._slider(frame, "Trails", "fade", x, y, cw, 0.50, 0.985)
        y = self._slider(frame, "Glow", "exposure", x, y, cw, 0.3, 4.0)
        y = self._slider(frame, "Spark", "spark", x, y, cw, 0.0, 1.0)
        y = self._slider(frame, "Flow", "curl", x, y, cw, 0.0, 1.0)
        y = self._slider(frame, "Dot size", "dot", x, y, cw, 0.004, 0.020)

        self._text(frame, "AUDIO", x, y, DIM, 0.4); y += 16
        y = self._button(frame, "React to sound: " + ("ON" if self.audio else "off"),
                         "audio", x, y, cw, active=self.audio)
        y = self._slider(frame, "Sensitivity", "sens", x, y, cw, 0.0, 3.0)

        y += 6
        y = self._button(frame, ("■ Stop recording" if self.record else "● Record"),
                         "record", x, y, cw, active=self.record)
        y = self._button(frame, "Mirror: " + ("on" if self.mirror else "off"),
                         "mirror", x, y, cw, active=self.mirror)

        self._draw_status(frame, info)
        return frame

    def _draw_status(self, frame, info):
        self._text(frame, info.get("status", ""), 12, 24, ACC, 0.5)
        if info.get("black"):
            h, w = frame.shape[:2]
            self._text(frame, "CAMERA IS BLACK — disable iPhone Continuity Camera",
                       12, h // 2, RED, 0.8, 2)
            self._text(frame, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                       12, h // 2 + 28, (140, 140, 240), 0.5)
        if self.record:
            h, w = frame.shape[:2]
            cv2.circle(frame, (w - (self.panel_w + 24 if self.open else 70), 24), 7, RED, -1)

    # ----- mouse -----
    def on_mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            for (x0, y0, x1, y1), kind, payload in self._hot:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._activate(kind, payload, x)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and self._drag:
            attr, x0, x1, lo, hi = self._drag
            t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
            setattr(self, attr, lo + t * (hi - lo))
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _activate(self, kind, payload, x):
        if kind == "collapse":
            self.open = not self.open
        elif kind == "preset":
            self.preset_idx = payload
            self.pending_preset = self.presets[payload]
        elif kind == "cycle":
            key, d = payload
            if key == "matte":
                self.matte_idx = (self.matte_idx + d) % len(self.mattes)
            elif key == "color":
                self.palette_idx = (self.palette_idx + d) % len(self.palettes)
        elif kind == "slider":
            attr, x0, x1, lo, hi = payload
            self._drag = payload
            t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
            setattr(self, attr, lo + t * (hi - lo))
        elif kind == "button":
            if payload == "audio": self.audio = not self.audio
            elif payload == "record": self.record = not self.record
            elif payload == "mirror": self.mirror = not self.mirror
            elif payload == "save": self.pending_save = True
