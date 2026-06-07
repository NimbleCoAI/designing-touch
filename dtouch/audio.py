"""Audio operators — the CHOP. Turn sound into per-frame control signals.

`analyze_block` is the DSP core: an rFFT of a sample block -> RMS amplitude + bass/mid/treble
band energy. It's pure and unit-tested. Sources build a per-frame *timeline* of these signals,
normalized to [0, 1] so visual mappings are robust regardless of input loudness.

- SyntheticAudio: a deterministic kick + tone + hi-hat signal. No file, no mic, no deps -> the
  headless-verifiable default (matches the repo's self-verifying loop).
- WavAudio: a mono/stereo WAV file (stdlib `wave`, no extra dependency).
- Live mic capture is intentionally not here: it needs sounddevice + macOS mic permission,
  which breaks unattended runs. Add it behind an explicit flag when driving live.
"""
from __future__ import annotations

import wave
from typing import Optional, Dict

import numpy as np

BANDS = {"bass": (20, 250), "mid": (250, 4000), "treble": (4000, 20000)}


def analyze_block(samples: np.ndarray, sr: int) -> Dict[str, float]:
    """RMS amplitude + per-band magnitude for one block of mono samples."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return {"amp": 0.0, "bass": 0.0, "mid": 0.0, "treble": 0.0}
    amp = float(np.sqrt(np.mean(x ** 2)))
    win = x * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    out = {"amp": amp}
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = float(mag[sel].mean()) if sel.any() else 0.0
    return out


def _timeline(signal: np.ndarray, sr: int, fps: int, frames: int, win: int = 2048):
    """Per-frame normalized band/amp envelopes for a whole signal."""
    hop = max(int(sr / fps), 1)
    raw = {k: np.zeros(frames, dtype=np.float64) for k in ("amp", "bass", "mid", "treble")}
    for i in range(frames):
        start = i * hop
        block = signal[start:start + win]
        if block.size < win:
            block = np.pad(block, (0, win - block.size))
        a = analyze_block(block, sr)
        for k in raw:
            raw[k][i] = a[k]
    norm = {}
    for k, v in raw.items():
        ref = np.percentile(v, 95) if np.any(v) else 1.0
        norm[k] = np.clip(v / (ref + 1e-9), 0.0, 1.5).astype(np.float32)
    return norm


class _TimelineSource:
    def __init__(self, timeline, frames):
        self._t = timeline
        self.frames = frames
        self._i = 0

    def read(self) -> Optional[Dict[str, float]]:
        if self._i >= self.frames:
            return None
        i = self._i
        self._i += 1
        return {k: float(self._t[k][i]) for k in self._t}


class SyntheticAudio(_TimelineSource):
    """Deterministic kick (bass) + tone (mid) + hi-hat (treble) — a lively reactive signal."""

    def __init__(self, frames: int, fps: int = 30, sr: int = 44100):
        n = int(frames / fps * sr) + sr
        t = np.arange(n, dtype=np.float64) / sr
        sig = np.zeros(n, dtype=np.float64)
        beat = 0.5  # 120 BPM
        for k in range(int(t[-1] / beat) + 1):
            t0 = k * beat
            env = np.exp(-np.maximum(t - t0, 0) * 18.0) * (t >= t0)
            sig += 0.9 * env * np.sin(2 * np.pi * 55.0 * t)           # kick
            ht0 = t0 + beat / 2
            henv = np.exp(-np.maximum(t - ht0, 0) * 60.0) * (t >= ht0)
            sig += 0.25 * henv * np.sin(2 * np.pi * 8000.0 * t)       # hi-hat
        melody = (0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) *
                  np.sin(2 * np.pi * 440.0 * t))                       # mid tone, tremolo
        sig += melody
        super().__init__(_timeline(sig, sr, fps, frames), frames)


class WavAudio(_TimelineSource):
    def __init__(self, path: str, frames: int, fps: int = 30):
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        if nch > 1:
            data = data.reshape(-1, nch).mean(axis=1)
        super().__init__(_timeline(data, sr, fps, frames), frames)


def make_audio(kind: str, frames: int, fps: int = 30):
    if kind == "synthetic":
        return SyntheticAudio(frames, fps)
    return WavAudio(kind, frames, fps)
