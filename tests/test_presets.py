"""Tests for presets and the video-color palette."""
import os
import numpy as np

from dtouch import presets
from dtouch.particles import ParticleFlow


def test_builtin_presets_present_and_well_formed():
    p = presets.load("does_not_exist.json")
    for name in ("abstract", "portrait", "textured"):
        assert name in p
        assert p[name]["palette"] in ("ice", "video", "rainbow", "fire", "aurora", "mono")


def test_save_and_reload_user_preset(tmp_path):
    path = str(tmp_path / "presets.json")
    cfg = dict(matte="auto", palette="fire", fade=0.91, exposure=1.7,
               spark=0.4, curl_amp=0.5, reseed_frac=0.06, base_size=0.012)
    presets.save("mine", cfg, path=path)
    loaded = presets.load(path)
    assert "mine" in loaded and loaded["mine"]["palette"] == "fire"
    assert "abstract" in loaded                      # built-ins still present


def test_video_palette_paints_particles_with_footage_color():
    gw, gh = 64, 36
    pf = ParticleFlow(n=3000, gw=gw, gh=gh, seed=0)
    pf.palette = "video"
    matte = np.ones((gh, gw), np.float32)
    gray = np.zeros((gh, gw), np.float32)
    color = np.zeros((gh, gw, 3), np.float32); color[..., 0] = 1.0   # pure red footage
    for _ in range(5):
        pf.update(matte, gray, color)
    inst = pf.render_data().reshape(3000, 7)
    # particles should be tinted red (R high, G/B low) from the footage
    assert inst[:, 4].mean() > 0.6
    assert inst[:, 5].mean() < 0.2 and inst[:, 6].mean() < 0.2
