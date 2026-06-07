"""Camera selection — lock onto the built-in laptop camera, never Continuity (iPhone).

macOS Continuity Camera makes an iPhone appear/disappear as a video device dynamically, and
its ordering isn't stable. We select strictly by AVFoundation **device type**: only the
BuiltInWideAngle (laptop FaceTime) camera is accepted; Continuity/External devices are rejected.
A device's position in `devicesWithMediaType` is the index OpenCV's AVFoundation backend uses,
so that position is the capture index.

For a 100% guarantee that the phone never appears at all, also disable Continuity Camera on the
iPhone: Settings > General > AirPlay & Handoff > Continuity Camera > Off.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2

_EXTERNAL_TYPES = ("continuity", "external")


def _video_devices():
    import AVFoundation as AV  # pyobjc, macOS only
    return list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))


def list_cameras() -> List[Tuple[int, str, str]]:
    """Return [(index, name, device_type), ...] in OpenCV/AVFoundation order."""
    try:
        devs = _video_devices()
    except Exception:
        return []
    out = []
    for i, d in enumerate(devs):
        try:
            dt = str(d.deviceType())
        except Exception:
            dt = "?"
        out.append((i, str(d.localizedName()), dt))
    return out


def _is_builtin(device_type: str) -> bool:
    t = device_type.lower()
    if any(x in t for x in _EXTERNAL_TYPES):
        return False
    return "builtinwideangle" in t


def find_builtin_index(default: int = 0) -> int:
    """Index of the built-in laptop camera (BuiltInWideAngle), rejecting Continuity/External."""
    cams = list_cameras()
    if not cams:
        return default
    for i, _name, dt in cams:
        if _is_builtin(dt):
            return i
    # fallback: first device that is explicitly not Continuity/External
    for i, _name, dt in cams:
        if not any(x in dt.lower() for x in _EXTERNAL_TYPES):
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
        idx = next((i for i, n, _ in cams if str(device).lower() in n.lower()), 0)
    name = next((n for i, n, _ in cams if i == idx), f"index {idx}")
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {device!r} (resolved index {idx}, '{name}')")
    return cap, name
