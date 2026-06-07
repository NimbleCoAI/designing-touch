# designing-touch

Experiments in recreating features of [TouchDesigner](https://derivative.ca/) — the node-based,
real-time visual programming environment — as a **modular engine that can be built, run, and
iterated on by Claude Code** from a terminal.

## Why

TouchDesigner is GUI-first: you wire operators (TOPs, CHOPs, SOPs) on a canvas. That's hard for
an agent to drive. This repo inverts it: **express the same primitives as code** — composable,
text-first, version-controlled, and headless — so an agent can author effects, render them, and
*verify the result itself* without a display or device permissions.

That last property is the whole point. See **[docs/autonomy-pattern.md](docs/autonomy-pattern.md)**
for the self-verifying loop the repo is built around.

## The engine: `dtouch/`

A small Python package of GPU/NumPy operators plus a node-graph-as-code spine:

| Module             | Role (TouchDesigner analog)                                   |
|--------------------|---------------------------------------------------------------|
| `dtouch.sources`   | TOP — synthetic / image / webcam / video luminance sources    |
| `dtouch.field`     | TOP→POP — grid, luminance displacement, seeded randoms, packing |
| `dtouch.render`    | Copy SOP + Light + Camera — headless instanced GPU renderer    |
| `dtouch.audio`     | CHOP — audio file/mic → amplitude + frequency bands            |
| `dtouch.fluid`     | GPU stable-fluids velocity field (advection)                   |
| `dtouch.pipeline`  | `Op` / `Graph` — wire operators by threading a context dict    |

## Experiments

| #  | Experiment        | TouchDesigner feature ported                          |
|----|-------------------|-------------------------------------------------------|
| 01 | displacement      | TOP→POP, displace by luminance, instanced boxes, light/depth |
| 02 | shadows           | + depth-from-light shadow map                          |
| 03 | audio-reactive    | CHOP → modulate displacement/scale by sound            |
| 04 | fluid             | stable-fluids advection of the particle field          |
| 05 | live-webcam       | **real-time interactive** webcam → particle displacement in a window |

Each `experiments/NN-name/` has its own README and a `run.py` CLI; a sample frame is committed
under `docs/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # installs the dtouch engine + deps

cd experiments/01-displacement
python run.py --frames 90             # synthetic source, no camera needed
python run.py --source webcam         # live (grant terminal Camera permission first)
```

Outputs land in each experiment's `out/` (`*.mp4` + `*_frame0.png`).

## Tests

```bash
pytest tests/ -q          # engine: deterministic transforms + graph + render smoke test
```

Verified on Apple Silicon (M4 Max, Metal-backed GL 4.1). Requires a GPU/GL context for the
render smoke test.
