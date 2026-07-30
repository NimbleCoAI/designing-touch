# 07 — Circuit bent

**TouchDesigner feature targeted:** the glitch/datamosh chain you'd wire from
`Channel Mix TOP` + `Displace TOP` + `Lookup TOP` with noise CHOPs driving the
offsets — plus the ordered-dither `Lookup` pass people reach for to fake a lo-fi
sensor. Here the whole chain is code: stochastic processes in NumPy, composited in
image space before the frame ever reaches a renderer.

## What it is

A software emulation of a **circuit-bent camera** — hardware deliberately
short-circuited so its video decoder misbehaves.

The hard part is that real bent hardware is *unpredictable*. A clean, deterministic
"glitch filter" reads as a Photoshop preset: the same wobble every frame, obviously
authored. Bent hardware instead drifts, catches, holds, and lets go, because the
faults are analogue and stateful. So none of this is a per-frame random number:

- **Chroma bleed** — R and B are displaced independently, but the offsets are
  IIR-filtered (`alpha = 0.85`), so the colour separation *drifts* rather than
  jittering. This is the colour decoder losing lock slowly.
- **Scan drift** — every row is displaced by two sine envelopes at incommensurate
  frequencies, with a randomised phase increment so the pattern never settles into a
  loop, plus small per-row noise so neighbouring rows aren't perfectly coupled.
  Occasionally (~4% of frames) a whole band slips at once: a **sync tear**.
- **Glitch blocks** — a rectangle of the frame is frozen and replayed somewhere else,
  and *held* for several frames. Holding is what makes it read as hardware: a
  one-frame flash looks like a dropped frame, a held block looks like a stuck buffer.
- **Bit crush + dithering** — hard quantisation, then Bayer or Floyd-Steinberg
  dithering, optionally computed at reduced resolution and scaled back up with
  nearest-neighbour for a chunky lo-fi sensor look.

Everything shares one RNG, so a fixed `seed` reproduces a session exactly — you can
tune a look, then get it back.

## Run it

```bash
python experiments/07-circuit-bent/run.py              # live webcam, all effects on
python experiments/07-circuit-bent/run.py --device 1   # pick a camera
python experiments/07-circuit-bent/run.py --source synthetic   # no camera needed
python experiments/07-circuit-bent/run.py --frames 60 --no-show  # render to out/
```

## Live controls

| key | does |
|-----|------|
| `q` / close window | quit |
| `d` | cycle dither: bayer → floyd-steinberg → none |
| `b` | toggle bit-crush |
| `s` | toggle CRT scan lines |
| `g` | toggle glitch blocks |
| `+` / `-` | chroma shift up / down |
| `[` / `]` | scan drift down / up |

## A note on speed

`bayer` is fully vectorised and runs comfortably in real time — measured **~140 fps**
on a 720p frame with the default reduced-resolution dither (~90 fps dithering at full
resolution).

**Floyd-Steinberg is error diffusion, which is inherently sequential**, so it runs a
Python loop per pixel: measured **~6 fps** on the same frame. That is expected, not a
bug — it is in the chain for the *look*, and for offline renders (`--no-show`). If the
window suddenly crawls after you press `d`, that is why; press `d` again to move on to
`none`, or twice to return to `bayer`.
