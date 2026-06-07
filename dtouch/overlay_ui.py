"""In-frame control panel — a collapsible sidebar drawn onto the render, one window.

Immediate-mode GUI drawn with cv2 primitives (Tk won't paint when launched headless on macOS;
cv2 windows do). Everything is a drawn, boxed, hover-highlighting control with click feedback,
and aligned hit-testing via setMouseCallback. Because it's just pixels, it can be rendered
headless and inspected. ASCII glyphs only — cv2's font can't render •/■/≡/— (they show as ???).
"""
from __future__ import annotations

import cv2

PANEL = (34, 32, 30)
BTN = (54, 52, 50)
HOVER = (82, 86, 82)
INK = (215, 222, 218)
DIM = (140, 150, 145)
ACC = (120, 215, 140)
TRACK = (70, 74, 72)
HANDLE = (160, 230, 175)
RED = (70, 70, 235)
DARK = (20, 22, 20)


def _in(rect, p):
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


class OverlayUI:
    def __init__(self, w, h, presets, palettes, mattes,
                 preset="abstract", matte="auto", palette="ice"):
        self.w, self.h = w, h
        self.presets, self.palettes, self.mattes = presets, palettes, mattes
        self.preset_idx = presets.index(preset) if preset in presets else 0
        self.matte_idx = mattes.index(matte) if matte in mattes else 0
        self.palette_idx = palettes.index(palette) if palette in palettes else 0
        self.fade, self.exposure, self.spark, self.curl, self.dot = 0.90, 1.4, 0.35, 0.5, 0.011
        self.count = 0.6   # fraction of the allocated particles to render
        self.mirror, self.audio, self.sens, self.record = True, False, 1.0, False
        self.res_options = [("720p", 1280, 720), ("1080p", 1920, 1080),
                            ("1440p", 2560, 1440), ("4K", 3840, 2160)]
        self.res_idx = next((i for i, (_, rw, rh) in enumerate(self.res_options)
                             if (rw, rh) == (w, h)), 1)
        self.open = True
        self._tooltip = None
        self.pending_preset = None
        self.pending_save = False
        self.panel_w = 290
        self.mouse = (-1, -1)
        self._hot = []
        self._drag = None
        self._flash = 0
        self._flash_rect = None

    @property
    def res_name(self): return self.res_options[self.res_idx][0]
    @property
    def res_wh(self): return self.res_options[self.res_idx][1:3]
    @property
    def matte_name(self): return self.mattes[self.matte_idx]
    @property
    def palette_name(self): return self.palettes[self.palette_idx]
    @property
    def preset_name(self): return self.presets[self.preset_idx]

    def sync_from(self, fade, exposure, spark, curl, dot, matte, palette):
        self.fade, self.exposure, self.spark, self.curl, self.dot = fade, exposure, spark, curl, dot
        if matte in self.mattes: self.matte_idx = self.mattes.index(matte)
        if palette in self.palettes: self.palette_idx = self.palettes.index(palette)

    # ----- primitives -----
    def _text(self, img, s, x, y, color=INK, scale=0.46, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _box(self, img, rect, fill, border=TRACK):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)

    def _row(self, img, label, key, x, y, w, h=24, active=False, payload=None):
        rect = (x, y, x + w, y + h)
        hovered = _in(rect, self.mouse)
        fill = ACC if active else (HOVER if hovered else BTN)
        self._box(img, rect, fill)
        self._text(img, label, x + 10, y + h - 8, DARK if active else INK, 0.46)
        self._hot.append(((x, y, x + w, y + h), key, payload))
        return rect

    def _slider(self, img, label, attr, x, y, w, info=None):
        lo, hi = _RANGES[attr]
        val = getattr(self, attr)
        rect = (x, y, x + w, y + 26)
        hovered = _in(rect, self.mouse)
        self._box(img, rect, HOVER if hovered else BTN)
        # info badge
        ic = (x + 14, y + 13)
        irect = (x + 4, y + 3, x + 24, y + 23)
        cv2.circle(img, ic, 8, (90, 110, 150), -1, cv2.LINE_AA)
        self._text(img, "i", ic[0] - 2, ic[1] + 5, (235, 240, 245), 0.42, 1)
        if info and _in(irect, self.mouse):
            self._tooltip = (info, x, y)
        self._text(img, label, x + 30, y + 17, DIM, 0.42)
        tx0, tx1 = x + 96, x + w - 44
        cv2.line(img, (tx0, y + 13), (tx1, y + 13), TRACK, 3, cv2.LINE_AA)
        hx = int(tx0 + (val - lo) / (hi - lo) * (tx1 - tx0))
        cv2.circle(img, (hx, y + 13), 6, HANDLE, -1, cv2.LINE_AA)
        self._text(img, f"{val:.2f}", x + w - 38, y + 17, INK, 0.42)
        self._hot.append(((tx0 - 8, y, tx1 + 8, y + 26), "slider", (attr, tx0, tx1, lo, hi)))
        return y + 30

    def _cycle(self, img, label, value, key, x, y, w):
        self._text(img, label, x, y + 10, DIM, 0.42)
        ry = y + 16
        lb = (x, ry, x + 28, ry + 24)
        rb = (x + w - 28, ry, x + w, ry + 24)
        self._box(img, lb, HOVER if _in(lb, self.mouse) else BTN)
        self._box(img, rb, HOVER if _in(rb, self.mouse) else BTN)
        self._text(img, "<", x + 9, ry + 17, INK, 0.5, 2)
        self._text(img, ">", x + w - 19, ry + 17, INK, 0.5, 2)
        self._text(img, str(value), x + 40, ry + 17, ACC, 0.5)
        self._hot.append((lb, "cycle", (key, -1)))
        self._hot.append((rb, "cycle", (key, +1)))
        return ry + 32

    # ----- layout -----
    def draw(self, frame, info):
        self._hot = []
        self._tooltip = None
        h, w = frame.shape[:2]
        if not self.open:
            r = (w - 48, 12, w - 12, 42)
            self._box(frame, r, HOVER if _in(r, self.mouse) else PANEL)
            for yy in (21, 27, 33):
                cv2.line(frame, (w - 40, yy), (w - 20, yy), INK, 2, cv2.LINE_AA)
            self._hot.append((r, "collapse", None))
            self._draw_status(frame, info)
            return frame

        px = w - self.panel_w
        ov = frame.copy()
        cv2.rectangle(ov, (px, 0), (w, h), PANEL, -1)
        cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
        x, cw, y = px + 16, self.panel_w - 32, 30
        self._text(frame, "dtouch", x, y, ACC, 0.62, 2)
        cr = (w - 40, 12, w - 12, 36)
        self._box(frame, cr, HOVER if _in(cr, self.mouse) else BTN)
        self._text(frame, ">", w - 33, 30, INK, 0.55, 2)
        self._hot.append((cr, "collapse", None))
        y += 16

        self._text(frame, "TEMPLATES", x, y, DIM, 0.4); y += 8
        for i, name in enumerate(self.presets):
            self._row(frame, name, "preset", x, y, cw, active=(i == self.preset_idx), payload=i)
            y += 28
        self._row(frame, "+ Save current look", "save", x, y, cw); y += 34

        self._text(frame, "SOURCE", x, y, DIM, 0.4); y += 6
        y = self._cycle(frame, "matte", self.matte_name, "matte", x, y, cw) + 2
        y = self._cycle(frame, "output", self.res_name, "res", x, y, cw) + 6

        self._text(frame, "LOOK", x, y, DIM, 0.4); y += 6
        y = self._cycle(frame, "color", self.palette_name, "color", x, y, cw) + 4
        for label, attr, tip in _SLIDERS:
            y = self._slider(frame, label, attr, x, y, cw, tip)

        y += 4
        self._row(frame, "Sound react: " + ("ON" if self.audio else "off"),
                  "audio", x, y, cw, active=self.audio); y += 28
        y = self._slider(frame, "Sens", "sens", x, y, cw,
                         "How strongly sound drives the visuals.") + 4
        self._row(frame, ("Stop recording" if self.record else "Record"),
                  "record", x, y, cw, active=self.record); y += 28
        self._row(frame, "Mirror: " + ("on" if self.mirror else "off"),
                  "mirror", x, y, cw, active=self.mirror)

        # click feedback: flash the last-clicked control
        if self._flash > 0 and self._flash_rect is not None:
            self._box(frame, self._flash_rect, BTN, border=(255, 255, 255))
            self._flash -= 1
        self._draw_tooltip(frame)
        self._draw_status(frame, info)
        return frame

    def _draw_tooltip(self, frame):
        if not self._tooltip:
            return
        text, sx, sy = self._tooltip
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 30:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        if cur:
            lines.append(cur)
        bw, bh = 250, 16 * len(lines) + 16
        bx = max(sx - bw - 14, 10)
        by = max(min(sy, self.h - bh - 10), 10)
        self._box(frame, (bx, by, bx + bw, by + bh), (44, 48, 56), border=(120, 140, 170))
        for i, ln in enumerate(lines):
            self._text(frame, ln, bx + 10, by + 20 + i * 16, INK, 0.42)

    def _draw_status(self, frame, info):
        self._text(frame, info.get("status", ""), 12, 24, ACC, 0.5)
        if info.get("black"):
            h, w = frame.shape[:2]
            self._text(frame, "CAMERA IS BLACK - disable iPhone Continuity Camera",
                       12, h // 2, RED, 0.8, 2)
            self._text(frame, "iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off",
                       12, h // 2 + 28, (140, 140, 240), 0.5)
        if self.record:
            h, w = frame.shape[:2]
            cv2.circle(frame, (w - (self.panel_w + 22 if self.open else 70), 24), 7, RED, -1)

    # ----- mouse -----
    def on_mouse(self, event, x, y, flags, param=None):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for rect, kind, payload in self._hot:
                if _in(rect, (x, y)):
                    self._flash_rect, self._flash = rect, 4
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
            elif key == "res":
                self.res_idx = (self.res_idx + d) % len(self.res_options)
        elif kind == "slider":
            attr, x0, x1, lo, hi = payload
            self._drag = payload
            t = min(max((x - x0) / max(x1 - x0, 1), 0.0), 1.0)
            setattr(self, attr, lo + t * (hi - lo))
        elif kind == "audio":
            self.audio = not self.audio
        elif kind == "record":
            self.record = not self.record
        elif kind == "mirror":
            self.mirror = not self.mirror
        elif kind == "save":
            self.pending_save = True


_RANGES = {
    "fade": (0.50, 0.985), "exposure": (0.3, 4.0), "spark": (0.0, 1.0),
    "curl": (0.0, 1.0), "dot": (0.004, 0.020), "sens": (0.0, 3.0),
    "count": (0.1, 1.0),
}

_SLIDERS = [
    ("Trails", "fade", "How long particle motion trails linger before fading."),
    ("Glow", "exposure", "Overall brightness and bloom of the particles."),
    ("Spark", "spark", "How much fast motion scatters particles into bursts of sparks."),
    ("Flow", "curl", "Swirling, turbulent drift of the particles."),
    ("Size", "dot", "Size of each individual particle dot."),
    ("Count", "count", "How many particles (density of the cloud)."),
]
