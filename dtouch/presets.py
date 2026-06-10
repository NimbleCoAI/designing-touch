"""Presets — named bundles of live settings you can switch between and save.

A preset sets the matte, color palette, and the live-tunable look parameters. `n` (particle
count) and resolution are start-up only and not part of a preset. Built-ins ship in code; your
own saved presets live in a JSON file (default ./presets.json in the launch dir) and are merged
on top, so you can capture a look you like and come back to it.
"""
from __future__ import annotations

import json
import os
from typing import Dict

# live-applicable keys a preset may set. video/audio keys are saved by the panel but
# absent from built-ins on purpose: a preset only touches the live video-bg / sound-react
# state when it explicitly recorded one, so template-hopping doesn't reset your toggles.
KEYS = ("matte", "palette", "fade", "exposure", "spark", "curl_amp", "reseed_frac",
        "base_size", "damp", "pull_falloff", "attract_speed",
        "video_bg", "video_mix", "audio", "sens")

BUILTIN: Dict[str, dict] = {
    # the loved abstract cloud
    "abstract": dict(matte="auto", palette="ice", fade=0.90, exposure=1.4,
                     spark=0.35, curl_amp=0.5, reseed_frac=0.06, base_size=0.011),
    # recognizable: particles painted with the real footage, calm so the shape holds
    "portrait": dict(matte="person", palette="video", fade=0.74, exposure=2.1,
                     spark=0.0, curl_amp=0.05, reseed_frac=0.16, base_size=0.0045),
    # textured but subject-agnostic (no person model) — colors of whatever's salient/moving
    "textured": dict(matte="auto", palette="video", fade=0.76, exposure=2.1,
                     spark=0.04, curl_amp=0.10, reseed_frac=0.14, base_size=0.005),
    "embers": dict(matte="auto", palette="fire", fade=0.93, exposure=1.6,
                   spark=0.5, curl_amp=0.6, reseed_frac=0.06, base_size=0.012),
    "aurora": dict(matte="motion", palette="aurora", fade=0.92, exposure=1.5,
                   spark=0.4, curl_amp=0.5, reseed_frac=0.06, base_size=0.011),
    # "sigil": sharp inward pull + very low damping make particles overshoot into glowing
    # contour bands along the silhouette, with caustic streaks — cyber-sigil / techcore.
    "sigil": dict(matte="person", palette="mono", fade=0.96, exposure=1.5, spark=0.0,
                  curl_amp=0.08, reseed_frac=0.008, base_size=0.006, damp=0.975,
                  pull_falloff=12.0, attract_speed=5.0),
}


def _read(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load(path: str = "presets.json") -> Dict[str, dict]:
    """Built-in presets merged with user presets from `path` (user wins on name clash)."""
    presets = {k: dict(v) for k, v in BUILTIN.items()}
    for name, cfg in _read(path).items():
        presets[name] = {k: cfg[k] for k in KEYS if k in cfg}
    return presets


def user_names(path: str = "presets.json") -> set:
    """Names of the user-saved presets (the ones that can be renamed/deleted)."""
    return set(_read(path).keys())


def save(name: str, cfg: dict, path: str = "presets.json") -> str:
    """Write/replace a user preset, preserving other user presets in the file."""
    existing = _read(path)
    existing[name] = {k: cfg[k] for k in KEYS if k in cfg}
    _write(existing, path)
    return path


def delete(name: str, path: str = "presets.json") -> bool:
    """Remove a user preset. Built-ins live in code, so they can't be deleted."""
    existing = _read(path)
    if name not in existing:
        return False
    del existing[name]
    _write(existing, path)
    return True


def rename(old: str, new: str, path: str = "presets.json") -> bool:
    """Rename a user preset. Refuses built-ins, name clashes, and empty names."""
    new = new.strip()
    existing = _read(path)
    if old not in existing or not new or new == old:
        return False
    if new in existing or new in BUILTIN:
        return False
    existing[new] = existing.pop(old)
    _write(existing, path)
    return True
