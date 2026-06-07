# designing-touch

Experiments in recreating features of [TouchDesigner](https://derivative.ca/) — the node-based,
real-time visual programming environment — in a form that can be **built, run, and iterated on by
Claude Code**.

## Why

TouchDesigner is a GUI-first tool: you wire operators (TOPs, CHOPs, SOPs, DATs) on a canvas to build
real-time generative visuals, audio-reactive systems, and interactive installations. That GUI-first
model is hard for an agent to drive. This repo explores the inverse: **express the same primitives as
code** — composable, text-first, version-controlled — so an agent can author and evolve them.

## Goal

Pick individual TouchDesigner capabilities and rebuild them as small, self-contained experiments:

- **Operator model** — a node graph of typed operators (texture / channel / geometry / data) with
  a pull-based evaluation model.
- **Real-time rendering** — GLSL/WebGL pipelines for procedural textures and feedback loops.
- **Audio-reactive** — FFT/CHOP-style signal processing driving visual parameters.
- **Procedural geometry** — SOP-style geometry generation and instancing.

Each experiment stands alone, documents what TD feature it targets, and how close it gets.

## Structure

```
experiments/   # one directory per experiment, self-contained
docs/          # notes on TouchDesigner concepts being recreated
```

## Status

Early. No experiments yet — scaffolding only.
