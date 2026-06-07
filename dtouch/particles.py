"""Particle-flow simulation — turn a per-frame matte into a smoothly flowing particle cloud.

A fixed pool of N particles is attracted to fill the matte's foreground and carried by the
subject's actual motion (optical flow) plus organic curl drift. Particles persist frame to
frame (they flow, never teleport); only particles that drift far outside the matte are recycled
to fresh foreground samples, so the cloud dissolves and reforms instead of popping.

Pure NumPy + a little OpenCV (distance transform, optical flow), all on a small working grid.
Subject-agnostic: it consumes any matte (saliency, motion, person, ...), not a person mask.
"""
from __future__ import annotations

import cv2
import numpy as np

PALETTES = ["ice", "video", "rainbow", "fire", "aurora", "mono"]


def _hsv2rgb(h, s, v):
    """Vectorized HSV->RGB. h,s,v are (N,) arrays in [0,1]. Returns (N,3)."""
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _bilinear(grid, x, y, gw, gh):
    x = np.clip(x, 0, gw - 1.001)
    y = np.clip(y, 0, gh - 1.001)
    x0 = x.astype(np.int32); y0 = y.astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    tx = x - x0; ty = y - y0
    return (grid[y0, x0] * (1 - tx) * (1 - ty) + grid[y0, x1] * tx * (1 - ty) +
            grid[y1, x0] * (1 - tx) * ty + grid[y1, x1] * tx * ty)


class ParticleFlow:
    def __init__(self, n=40000, gw=256, gh=144, seed=0, tint=(0.45, 0.72, 1.0),
                 K=0.20, attract_speed=4.5, drift_inside=0.15, pull_falloff=22.0,
                 flow_gain=0.6, curl_amp=0.5, damp=0.90, reseed_sdf=-22.0,
                 reseed_frac=0.06, base_size=0.011, speed_ref=4.0, spark=0.35):
        self.n, self.gw, self.gh = n, gw, gh
        rng = np.random.default_rng(seed)
        self.px = rng.uniform(0, gw, n).astype(np.float32)
        self.py = rng.uniform(0, gh, n).astype(np.float32)
        self.vx = np.zeros(n, np.float32)
        self.vy = np.zeros(n, np.float32)
        self.tint = np.array(tint, np.float32)
        # params
        self.K, self.attract_speed, self.drift_inside = K, attract_speed, drift_inside
        self.pull_falloff, self.flow_gain, self.curl_amp = pull_falloff, flow_gain, curl_amp
        self.damp, self.reseed_sdf, self.reseed_frac = damp, reseed_sdf, reseed_frac
        self.base_size, self.speed_ref = base_size, speed_ref
        self.spark = spark
        self.palette = "ice"
        self._rng = rng
        self._prev_gray = None
        self._z = np.zeros(n, np.float32)
        self._vcol = np.ones((n, 3), np.float32)   # per-particle source-video color
        # static curl field (divergence-free swirl) for organic drift
        pot = cv2.GaussianBlur(rng.standard_normal((gh, gw)).astype(np.float32), (0, 0), 9.0)
        cy, cx = np.gradient(pot)
        self._curl_x, self._curl_y = cy, -cx

    def update(self, matte, gray, color=None):
        """matte, gray: float32 (gh, gw). color: optional (gh, gw, 3) RGB in [0,1] for the
        'video' palette (particles painted with the real footage -> recognizable subject)."""
        gw, gh = self.gw, self.gh
        mbin = (matte > 0.35).astype(np.uint8)
        if mbin.any():
            dt_in = cv2.distanceTransform(mbin, cv2.DIST_L2, 3)
            dt_out = cv2.distanceTransform(1 - mbin, cv2.DIST_L2, 3)
        else:
            dt_in = np.zeros((gh, gw), np.float32); dt_out = np.zeros((gh, gw), np.float32)
        sdf = (dt_in - dt_out).astype(np.float32)
        gy, gx = np.gradient(sdf)
        gmag = np.hypot(gx, gy) + 1e-6
        fx_grid = (gx / gmag).astype(np.float32)
        fy_grid = (gy / gmag).astype(np.float32)
        z_grid = (dt_in / (dt_in.max() + 1e-6)).astype(np.float32)

        # optical flow (subject motion) on uint8 gray
        g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, g8, None,
                                                0.5, 2, 15, 2, 5, 1.1, 0)
            flow_x = flow[..., 0].astype(np.float32)
            flow_y = flow[..., 1].astype(np.float32)
        else:
            flow_x = np.zeros((gh, gw), np.float32); flow_y = np.zeros((gh, gw), np.float32)
        self._prev_gray = g8

        px, py = self.px, self.py
        fxg = _bilinear(fx_grid, px, py, gw, gh)
        fyg = _bilinear(fy_grid, px, py, gw, gh)
        sdf_v = _bilinear(sdf, px, py, gw, gh)
        mval = _bilinear(matte, px, py, gw, gh)
        fwx = _bilinear(flow_x, px, py, gw, gh)
        fwy = _bilinear(flow_y, px, py, gw, gh)
        cnx = _bilinear(self._curl_x, px, py, gw, gh)
        cny = _bilinear(self._curl_y, px, py, gw, gh)

        # Attraction pulls ONLY outside particles back toward the shape (no skeleton collapse).
        # Inside particles are sustained by density-weighted reseeding + drift, so they FILL.
        pull = np.clip(-sdf_v / self.pull_falloff, 0.0, 1.0)   # 0 inside, ->1 outside
        v_des_x = self.attract_speed * fxg * pull
        v_des_y = self.attract_speed * fyg * pull
        self.vx += self.K * (v_des_x - self.vx)
        self.vy += self.K * (v_des_y - self.vy)
        # flow carries motion; curl gives organic drift everywhere (life inside the shape)
        self.vx += self.flow_gain * fwx + self.curl_amp * cnx
        self.vy += self.flow_gain * fwy + self.curl_amp * cny
        # motion-spark: fast local motion scatters particles into energetic sparks
        if self.spark > 0:
            fmag = np.hypot(fwx, fwy)
            kick = self.spark * fmag
            self.vx += self._rng.standard_normal(self.n).astype(np.float32) * kick
            self.vy += self._rng.standard_normal(self.n).astype(np.float32) * kick
        self.vx *= self.damp
        self.vy *= self.damp
        self.px += self.vx
        self.py += self.vy

        # recycle particles that have LEFT the shape (low matte) or gone offscreen.
        offscreen = (self.px < 0) | (self.px >= gw) | (self.py < 0) | (self.py >= gh)
        mnow = _bilinear(matte, np.clip(self.px, 0, gw - 1.001),
                         np.clip(self.py, 0, gh - 1.001), gw, gh)
        left = mnow < 0.12
        recycle = np.where(offscreen | left)[0]
        budget = int(self.reseed_frac * self.n)
        if recycle.size > budget:
            recycle = self._rng.choice(recycle, budget, replace=False)
        if recycle.size and mbin.any():
            rx, ry = self._sample_matte(matte, recycle.size)
            self.px[recycle] = rx; self.py[recycle] = ry
            self.vx[recycle] = _bilinear(flow_x, rx, ry, gw, gh) * self.flow_gain
            self.vy[recycle] = _bilinear(flow_y, rx, ry, gw, gh) * self.flow_gain

        self._z = _bilinear(z_grid, self.px, self.py, gw, gh).astype(np.float32)
        if color is not None:
            cx = np.clip(self.px, 0, gw - 1.001); cy = np.clip(self.py, 0, gh - 1.001)
            self._vcol = np.stack([_bilinear(color[..., 0], cx, cy, gw, gh),
                                   _bilinear(color[..., 1], cx, cy, gw, gh),
                                   _bilinear(color[..., 2], cx, cy, gw, gh)], axis=1).astype(np.float32)

    def _sample_matte(self, matte, k):
        w = matte.ravel().astype(np.float64)
        w *= w
        s = w.sum()
        if s <= 0:
            return (self._rng.uniform(0, self.gw, k).astype(np.float32),
                    self._rng.uniform(0, self.gh, k).astype(np.float32))
        cdf = np.cumsum(w) / s
        idx = np.searchsorted(cdf, self._rng.random(k))
        gy_i, gx_i = np.divmod(idx, self.gw)
        return ((gx_i + self._rng.random(k)).astype(np.float32),
                (gy_i + self._rng.random(k)).astype(np.float32))

    def render_data(self) -> np.ndarray:
        """Per-particle instance buffer: [x_ndc, y_ndc, size, bright, r, g, b] x N."""
        x_ndc = self.px / self.gw * 2.0 - 1.0
        y_ndc = 1.0 - self.py / self.gh * 2.0
        speed = np.hypot(self.vx, self.vy)
        sp = np.clip(speed / self.speed_ref, 0.0, 1.0)
        z = self._z
        if self.palette == "video":
            bright = 0.75 + 0.25 * z          # let the footage's own colors carry luminance
        else:
            bright = (0.30 + 0.70 * sp) * (0.45 + 0.55 * z)
        size = self.base_size * (0.7 + 1.0 * z)
        col = self._colorize(sp, z)
        out = np.empty((self.n, 7), np.float32)
        out[:, 0] = x_ndc; out[:, 1] = y_ndc; out[:, 2] = size; out[:, 3] = bright
        out[:, 4:7] = col
        return out.reshape(-1)

    def _colorize(self, sp, z):
        n = self.n
        if self.palette == "video":
            return self._vcol
        if self.palette == "mono":
            return np.ones((n, 3), np.float32)
        if self.palette == "rainbow":
            hue = (np.arctan2(self.vy, self.vx) / (2 * np.pi)) + 0.5
            return _hsv2rgb(hue, np.full(n, 0.65, np.float32), np.ones(n, np.float32))
        if self.palette == "fire":
            hue = 0.02 + 0.12 * np.clip(sp + 0.4 * z, 0, 1)        # red -> yellow
            return _hsv2rgb(hue, 1.0 - 0.35 * sp, np.ones(n, np.float32))
        if self.palette == "aurora":
            hue = 0.33 + 0.5 * (self.py / self.gh)                  # green -> magenta by height
            return _hsv2rgb(hue, np.full(n, 0.7, np.float32), np.ones(n, np.float32))
        # ice (default): cool tint whitening with speed
        return self.tint[None, :] * (1 - 0.6 * sp)[:, None] + (0.6 * sp)[:, None]

    def cycle_palette(self) -> str:
        self.palette = PALETTES[(PALETTES.index(self.palette) + 1) % len(PALETTES)]
        return self.palette
