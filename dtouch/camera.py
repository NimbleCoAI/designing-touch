"""Camera selection — pick the right capture device by name, not a fragile index.

On macOS, Continuity Camera makes an iPhone show up as a video device alongside the built-in
camera, and the index order isn't guaranteed. We enumerate devices via AVFoundation (their
order matches OpenCV's AVFoundation backend indices) and select by name so we always get the
built-in laptop camera unless told otherwise.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import cv2

_EXTERNAL_HINTS = ("iphone", "ipad", "continuity", "obs", "virtual")


def list_cameras() -> List[Tuple[int, str]]:
    """Return [(index, name), ...] in OpenCV/AVFoundation order. Empty if AVFoundation absent."""
    try:
        import AVFoundation as AV  # pyobjc, macOS only
    except Exception:
        return []
    devs = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
    return [(i, str(d.localizedName())) for i, d in enumerate(devs)]


def find_builtin_index(default: int = 0) -> int:
    """Index of the built-in laptop camera (first device that isn't an external/phone)."""
    cams = list_cameras()
    if not cams:
        return default
    for i, name in cams:
        if not any(h in name.lower() for h in _EXTERNAL_HINTS):
            return i
    return default


def open_camera(device="builtin") -> Tuple[cv2.VideoCapture, str]:
    """Open a capture. device: 'builtin' | int index | device-name substring.

    Returns (capture, resolved_name). Raises if it can't open.
    """
    cams = list_cameras()
    if device == "builtin":
        idx = find_builtin_index()
    elif isinstance(device, int):
        idx = device
    else:  # name substring
        idx = next((i for i, n in cams if str(device).lower() in n.lower()), 0)
    name = next((n for i, n in cams if i == idx), f"index {idx}")
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {device!r} (resolved index {idx}, '{name}')")
    return cap, name
