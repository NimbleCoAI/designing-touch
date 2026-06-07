"""TDD for the audio DSP core."""
import numpy as np

from dtouch.audio import analyze_block, SyntheticAudio, LiveMic


def _tone(freq, sr=44100, n=2048, amp=1.0):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_pure_bass_tone_excites_bass_band():
    a = analyze_block(_tone(60), 44100)
    assert a["bass"] > a["mid"] * 5
    assert a["bass"] > a["treble"] * 5


def test_pure_mid_tone_excites_mid_band():
    a = analyze_block(_tone(1000), 44100)
    assert a["mid"] > a["bass"] * 5
    assert a["mid"] > a["treble"] * 5


def test_pure_treble_tone_excites_treble_band():
    a = analyze_block(_tone(8000), 44100)
    assert a["treble"] > a["bass"] * 5
    assert a["treble"] > a["mid"] * 5


def test_amplitude_tracks_rms():
    quiet = analyze_block(_tone(1000, amp=0.1), 44100)
    loud = analyze_block(_tone(1000, amp=1.0), 44100)
    assert loud["amp"] > quiet["amp"] * 5
    assert abs(loud["amp"] - 0.707) < 0.05   # RMS of unit sine


def test_empty_block_is_silent():
    a = analyze_block(np.array([]), 44100)
    assert a == {"amp": 0.0, "bass": 0.0, "mid": 0.0, "treble": 0.0}


def test_synthetic_audio_timeline_is_normalized_and_reactive():
    src = SyntheticAudio(frames=60, fps=30)
    vals = []
    while True:
        v = src.read()
        if v is None:
            break
        vals.append(v)
    assert len(vals) == 60
    bass = np.array([v["bass"] for v in vals])
    # normalized into ~[0,1.5] and actually time-varying (the beat)
    assert bass.max() <= 1.5 + 1e-6
    assert bass.std() > 0.1


def test_live_mic_band_math_and_normalization():
    mic = LiveMic(sr=44100, block=2048)
    t = np.arange(2048) / 44100
    bass_block = np.sin(2 * np.pi * 60 * t).reshape(-1, 1).astype(np.float32)
    for _ in range(5):
        mic._callback(bass_block, 2048, None, None)
    lv = mic.levels()
    assert set(lv) == {"amp", "bass", "mid", "treble"}
    assert all(0.0 <= v <= 1.0 for v in lv.values())
    assert lv["bass"] > lv["treble"]          # bass tone -> bass dominates


def test_live_mic_start_is_graceful():
    mic = LiveMic()
    ok = mic.start()                          # may be True/False depending on permission
    assert isinstance(ok, bool)
    mic.stop()
