# 05 — Live preview (real-time, interactive)

Camera → a smoothly flowing cloud of **glowing particles** that forms whatever is in the frame.
This is the interactive payoff of the engine, and the answer to "turn a dancer into particles."

## Two big ideas (and why the first attempt was wrong)

The first version displaced a **fixed grid** of dots by luminance — so you saw a flat rectangular
sheet vaguely reacting, not *you*. This version is built differently:

1. **Subject-agnostic matte, not "person".** A `dtouch.matte` extracts an interest field from
   the frame — `auto` (motion ∪ saliency), `motion` (background subtraction), `saliency`,
   `edges`, or `person` (optional multi-person segmentation). It keys on *whatever moves or
   stands out* — a dancer, a crowd, a boat — not a hardcoded person model.

2. **Particles flow into that shape.** `dtouch.particles.ParticleFlow` keeps a pool of ~45k
   particles that fill the matte (density-weighted reseeding), are carried by the subject's
   real motion (optical flow) plus organic curl, and persist frame-to-frame so they *flow*
   instead of snapping. `dtouch.glow.GlowRenderer` draws them as soft additive points with a
   ping-pong trail buffer → luminous flowing dust.

## Camera selection

Uses the **built-in laptop camera by name** (`dtouch.camera`), so macOS Continuity Camera never
hijacks it with your iPhone. `--device builtin` (default) | an index | a name substring.
`python run.py --list-cameras` shows what's available.

## Run

```bash
pip install -e ../..            # engine (default)
pip install -e "../..[person]"  # add the person matte (mediapipe)

python run.py                   # flow, built-in camera, auto matte
python run.py --matte motion    # key purely on movement — best for dancing
python run.py --matte person    # multi-person segmentation
python run.py --mode grid       # the older luminance-grid effect
```

Controls: `q` quit · `n` cycle matte · `m` mirror · `[` `]` trail length · `-` `=` glow · `space` freeze

## Sample (person matte, from a still)

Figures rendered as flowing particle clouds filling their body shapes:

![flow](../../docs/05-flow-person.png)

## Notes / next

- Runs ~20 fps at 1280×720 with 45k particles on an M4. Optical flow (Farneback) is the main
  cost; drop `--particles` or grid res for more speed.
- Next: live audio reactivity (mic → particle energy), per-particle color from flow direction,
  a record-to-MP4 key, and a GPU port of the matte/flow for higher particle counts.
