"""4K scaling — the control panel must scale proportionally to the output resolution.

Issue #3: at 4K the panel kept its 1080p pixel dimensions, so it rendered tiny. The
panel now scales by max(1.0, h/1080) — floored at 1.0 so 720p/1080p are unchanged
(the scroll tests rely on that), and growing to 2x at 4K so it stays the same fraction
of the frame. Geometry is read back from the drawn hit-rects, mirroring the overlay
scroll-test pattern."""
import numpy as np

from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui_drawn(w, h):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _quit_rect(ui):
    return next(r for r, k, _ in ui._hot if k == "quit")


def test_panel_unchanged_at_1080p():
    # scale floors at 1.0, so the 1080p baseline is byte-identical to before the change
    x0, y0, x1, y1 = _quit_rect(_ui_drawn(1920, 1080))
    assert x1 - x0 == 258            # column width: panel_w(290) - 2*16 margin
    assert x0 == 1920 - 290 + 16     # panel left edge + margin
    assert y1 - y0 == 24             # default row height


def test_panel_wider_at_4k():
    q1080 = _quit_rect(_ui_drawn(1920, 1080))
    q4k = _quit_rect(_ui_drawn(3840, 2160))
    assert (q4k[2] - q4k[0]) > (q1080[2] - q1080[0])


def test_panel_taller_rows_at_4k():
    q1080 = _quit_rect(_ui_drawn(1920, 1080))
    q4k = _quit_rect(_ui_drawn(3840, 2160))
    assert (q4k[3] - q4k[1]) > (q1080[3] - q1080[1])


def test_panel_scales_proportionally_at_4k():
    # 4K is exactly 2x the 1080p baseline, so panel widths and row heights double
    q1080 = _quit_rect(_ui_drawn(1920, 1080))
    q4k = _quit_rect(_ui_drawn(3840, 2160))
    assert (q4k[2] - q4k[0]) == 2 * (q1080[2] - q1080[0])
    assert (q4k[3] - q4k[1]) == 2 * (q1080[3] - q1080[1])


def test_panel_pixel_width_scales_with_resolution():
    assert _ui_drawn(1920, 1080)._panel_px == 290    # 1080p baseline
    assert _ui_drawn(3840, 2160)._panel_px == 580    # 4K: 2x


def test_1440p_scales_between_1080p_and_4k():
    w1080 = (lambda r: r[2] - r[0])(_quit_rect(_ui_drawn(1920, 1080)))
    w1440 = (lambda r: r[2] - r[0])(_quit_rect(_ui_drawn(2560, 1440)))
    w4k = (lambda r: r[2] - r[0])(_quit_rect(_ui_drawn(3840, 2160)))
    assert w1080 < w1440 < w4k
