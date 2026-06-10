"""Panel scrolling — at small outputs (720p) the control column is taller than the
window; wheel / drag-on-empty-panel scrolling must make every control reachable."""
import cv2
import numpy as np

from dtouch.overlay_ui import OverlayUI
from dtouch.particles import PALETTES
from dtouch.live import MATTES

PRESETS = ["abstract", "portrait", "textured", "embers", "aurora", "sigil"]


def _ui(w=1280, h=720):
    ui = OverlayUI(w, h, PRESETS, list(PALETTES), MATTES)
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return ui


def _quit_bottom(ui, w=1280, h=720):
    ui.draw(np.zeros((h, w, 3), np.uint8), {"status": ""})
    return next(r for r, k, _ in ui._hot if k == "quit")[3]


def test_wheel_scrolls_panel_down():
    ui = _ui()
    before = _quit_bottom(ui)
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, -120)   # wheel down (negative delta)
    after = _quit_bottom(ui)
    assert after < before


def test_scroll_clamps_to_zero_at_top():
    ui = _ui()
    ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, +120)    # wheel up at the top
    assert _quit_bottom(ui) == _quit_bottom(_ui())        # unchanged


def test_quit_reachable_at_720p_after_scrolling():
    ui = _ui()
    for _ in range(60):                                   # scroll all the way down
        ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1100, 300, -120)
    assert _quit_bottom(ui) <= 720


def test_no_scroll_when_content_fits_1080p():
    ui = _ui(1920, 1080)
    before = _quit_bottom(ui, 1920, 1080)
    for _ in range(20):
        ui.on_mouse(cv2.EVENT_MOUSEWHEEL, 1700, 300, -120)
    assert _quit_bottom(ui, 1920, 1080) == before         # clamped: content fits


def test_drag_on_empty_panel_scrolls():
    ui = _ui()
    before = _quit_bottom(ui)
    # a spot inside the panel that hits no control: just under the collapse button
    px = 1280 - 6
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, px, 400, 0)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, px, 250, cv2.EVENT_FLAG_LBUTTON)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, px, 250, 0)
    assert _quit_bottom(ui) < before


def test_slider_drag_still_works_with_scrolling():
    ui = _ui()
    payload = next(p for _, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    attr, x0, x1, lo, hi = payload
    rect = next(r for r, k, p in ui._hot if k == "slider" and p[0] == "video_mix")
    ymid = (rect[1] + rect[3]) // 2
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, x1, ymid, 0)
    assert abs(ui.video_mix - hi) < 1e-6
