# AGENTS.md — working on designing-touch

Guide for AI agents (and humans) developing this repo. Read this before changing code.

## What this is

A real-time, modular engine for turning **video and/or sound into flowing particle visuals** —
recreating TouchDesigner-style effects as text-first, CLI/GUI-drivable Python instead of a GUI
node graph. The headline experience is **experiment 05** (`experiments/05-live-webcam/`): a live
webcam → particle instrument with an in-frame control panel and saved looks ("templates").

## Architecture

The engine is the `dtouch/` package (installable via `pip install -e .`). Each module is one
small, composable operator — the "node graph as code":

| Module             | Role                                                                |
|--------------------|---------------------------------------------------------------------|
| `sources.py`       | video/webcam/image/synthetic → luminance grids (TOP)                |
| `matte.py`         | subject-agnostic interest field: motion / saliency / edges / luma / person |
| `field.py`         | grid, luminance Z-displacement, seeded randoms, packing (TOP→POP)   |
| `particles.py`     | `ParticleFlow` — particles fill a matte, advect by optical flow + curl |
| `render.py`        | `Renderer` — headless instanced-cube GPU renderer (lights + depth)  |
| `shadow.py`        | `ShadowRenderer` — adds a depth-from-light shadow map               |
| `glow.py`          | `GlowRenderer` — additive soft-particle renderer with trail feedback |
| `audio.py`         | `analyze_block`, `SyntheticAudio`/`WavAudio`, `LiveMic` (CHOP)       |
| `fluid.py`         | `Fluid2D` — 2D stable-fluids solver                                 |
| `camera.py`        | select the built-in laptop camera by device type (macOS-safe)       |
| `overlay_ui.py`    | `OverlayUI` — in-frame collapsible control panel (drawn on the render) |
| `presets.py`       | named looks (templates) + load/save                                 |
| `live.py`          | `live_flow` (the live instrument) and `live` (legacy grid)          |
| `pipeline.py`      | `Op`/`Graph` — wire operators by threading a context dict           |

Rendering is **headless moderngl** (`create_standalone_context`) → renders to an offscreen FBO,
read back as a NumPy array, then blitted to an OpenCV window. This is why it's all self-verifiable.

## The core convention: headless self-verification

See `docs/autonomy-pattern.md`. Every effect must render to a file you can inspect, and every
input has a synthetic/file fallback so it runs with no camera/mic/display. When changing visuals,
**render a frame and look at it** (read the PNG) — don't claim it works from code alone. TDD the
deterministic NumPy cores (`field`, `audio`, `fluid`, `particles`, `matte`); smoke-test rendering.

## Run & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # engine
pip install -e ".[person]"  # + mediapipe for the person matte
pytest tests/ -q            # 37 tests
python experiments/05-live-webcam/run.py     # the live instrument (or double-click start.command)
```

## How to extend

- **New color palette:** add its name to `PALETTES` in `particles.py` and a branch in
  `ParticleFlow._colorize`. It auto-appears in the panel's color cycler.
- **New matte (subject source):** add a class with `.compute(frame_bgr) -> float (H,W) [0,1]` in
  `matte.py`, register it in `make_matte` and the `MATTES` list in `live.py`.
- **New template (look):** add an entry to `BUILTIN` in `presets.py` using the keys in `KEYS`.
  Users save their own to `presets.json` (gitignored) via the panel.
- **New control slider:** add the value + range to `OverlayUI` (`_RANGES`, `_SLIDERS`) and read it
  each frame in `live.py`. If it's a `ParticleFlow`/`GlowRenderer` field, also wire it in
  `apply_preset` + `sync_from`.
- **New experiment:** `experiments/NN-name/` with a `run.py` that composes `dtouch` operators and
  writes `out/*.mp4` + a `_frame0.png`; commit a sample frame under `docs/`.

## Gotchas (hard-won)

- **macOS camera:** an iPhone Continuity Camera forces the built-in camera to return **all-black**
  frames. `camera.py` selects the built-in by device type; the live app detects black frames and
  says so. The real fix is disabling Continuity Camera on the iPhone.
- **cv2 window is `WINDOW_AUTOSIZE`** (fixed at render res). Maximizing a smaller window upscales
  every frame (kills fps) and breaks mouse-coordinate mapping (controls go unresponsive). "Bigger"
  = switch the `output` resolution, not OS-maximize.
- **Overlay panel is ASCII-only.** cv2's Hershey font renders `•/■/≡/—` as `???`; use ASCII or
  draw shapes.
- **Tkinter doesn't paint** when launched headless on macOS — that's why the panel is drawn on the
  cv2 frame, not a Tk window. (`gui.py` is a deprecated Tk attempt; not the default.)
- **Quit** is the panel's Quit button, `q`, or the window's close box (detected via
  `WND_PROP_VISIBLE < 0`, which does NOT fire on minimize). ESC is intentionally ignored.
- **`presets.json` and `dtouch/assets/*.tflite`** are gitignored (user data / downloaded model).
