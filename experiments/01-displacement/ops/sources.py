"""Source operators — the TOP (texture in). Every source yields a luminance grid.

A luminance grid is float32 (gy, gx) in [0, 1]. Downsampling to the field resolution
happens here so the rest of the pipeline only ever sees grid-sized data.

Sources
-------
- SyntheticSource: procedural animated luminance. The zero-dependency default, so the
  whole pipeline can run and be verified with no camera permission and no input file.
- ImageSource: a single still image (any format OpenCV reads), held for N frames.
- VideoSource: a video file or live webcam via OpenCV's AVFoundation backend (macOS).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _to_luma_grid(bgr: np.ndarray, gx: int, gy: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (gx, gy), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0)


class SyntheticSource:
    """A moving Gaussian blob over faint rings — clear luminance structure for verification."""

    def __init__(self, gx: int, gy: int, frames: int = 90):
        self.gx, self.gy, self.frames = gx, gy, frames
        self._i = 0
        ys, xs = np.mgrid[0:gy, 0:gx].astype(np.float32)
        self._x = xs / max(gx - 1, 1)  # 0..1
        self._y = ys / max(gy - 1, 1)

    def read(self) -> Optional[np.ndarray]:
        if self._i >= self.frames:
            return None
        t = self._i / max(self.frames, 1)
        # blob orbits the center
        cx = 0.5 + 0.28 * np.cos(2 * np.pi * t)
        cy = 0.5 + 0.28 * np.sin(2 * np.pi * t)
        d2 = (self._x - cx) ** 2 + (self._y - cy) ** 2
        blob = np.exp(-d2 / (2 * 0.06 ** 2))
        rings = 0.25 * (0.5 + 0.5 * np.cos(40.0 * np.sqrt(
            (self._x - 0.5) ** 2 + (self._y - 0.5) ** 2) - 6.0 * t))
        luma = np.clip(blob + rings, 0.0, 1.0).astype(np.float32)
        self._i += 1
        return luma


class ImageSource:
    """A single still image, downsampled and held for `frames` frames."""

    def __init__(self, path: str, gx: int, gy: int, frames: int = 1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        self._luma = _to_luma_grid(img, gx, gy)
        self.frames = frames
        self._i = 0

    def read(self) -> Optional[np.ndarray]:
        if self._i >= self.frames:
            return None
        self._i += 1
        return self._luma


class VideoSource:
    """A video file (path) or live webcam (integer index) via OpenCV.

    On macOS the AVFoundation backend is used explicitly. Live webcam requires the
    terminal app to hold Camera permission (System Settings -> Privacy -> Camera),
    otherwise frames read back black.
    """

    def __init__(self, spec, gx: int, gy: int, frames: int = 90):
        self.gx, self.gy = gx, gy
        self.frames = frames
        self._i = 0
        if isinstance(spec, int):
            self._cap = cv2.VideoCapture(spec, cv2.CAP_AVFOUNDATION)
        else:
            self._cap = cv2.VideoCapture(spec)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source: {spec!r}")

    def read(self) -> Optional[np.ndarray]:
        if self._i >= self.frames:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._i += 1
        return _to_luma_grid(frame, self.gx, self.gy)

    def __del__(self):
        try:
            self._cap.release()
        except Exception:
            pass


def make_source(kind: str, gx: int, gy: int, frames: int):
    """Factory: 'synthetic' | 'webcam' | <path-to-image-or-video>."""
    if kind == "synthetic":
        return SyntheticSource(gx, gy, frames)
    if kind == "webcam":
        return VideoSource(0, gx, gy, frames)
    lower = kind.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
        return ImageSource(kind, gx, gy, frames)
    return VideoSource(kind, gx, gy, frames)
