# The self-verifying autonomy loop

This repo is built to be driven by an AI agent (Claude Code) with minimal human steering.
The thing that makes that work is **headless self-verification**: the agent can build an
effect, render it, *look at the result itself*, and iterate — with no display, no device
permissions, and no human in the loop until the work is actually done.

## The loop

```
research → design an operator → build it (TDD the deterministic core)
      → render a frame to PNG (headless GPU)
      → AGENT READS THE PNG and judges it           ← the load-bearing step
      → wrong? fix and re-render.  right? commit.
      → next experiment
```

The agent closing the loop on its *own* output (reading the rendered image) is what lets it
run unattended. A pipeline that can only be checked by a human watching a live window cannot
be iterated autonomously. So every effect here must be renderable to a file and inspectable.

## Conventions that keep the loop closed

1. **Headless GPU only.** Rendering uses `moderngl.create_standalone_context()` — on macOS
   this is a CGL context (Metal-backed GL 4.1), no window, no X server. The frame is read
   back with `fbo.read()` into a NumPy array and written to PNG/MP4. This is the single
   reason `moderngl` was chosen over pyrender/Open3D/vispy, whose headless paths are
   Linux-only or broken on macOS.

2. **Every input has a synthetic/file fallback.** Live webcam and live mic need macOS TCC
   permission (a human granting access), which breaks unattended runs. So every source
   defaults to a deterministic synthetic generator (e.g. `SyntheticSource`, a generated WAV)
   that needs nothing. `--source webcam` / live mic exist for real use; verification never
   depends on them.

3. **TDD the deterministic core; smoke-test the GPU.** Pure NumPy transforms (grid,
   displacement, randoms, audio analysis, fluid math) get real unit tests. Rendering gets an
   end-to-end smoke test that asserts the frame isn't empty. Visual *correctness* is judged
   by the agent reading the PNG, not asserted in code.

4. **Operators, composed as code.** Each stage is a small pure function in `dtouch/`. The
   `dtouch.pipeline.Graph` wires them by threading a context dict — the node-graph-as-code
   equivalent of a TD network. New effects reuse operators; they don't fork the engine.

5. **Commit per working increment, decide small things yourself.** Each experiment that
   renders correctly is committed. Config edits, branches, and PRs are routine and handled
   without asking. Surface to the human only for: a finished product, or a true blocker
   (e.g. a GitHub org transfer that needs their acceptance).

## What a new experiment looks like

```
experiments/NN-name/
  README.md      # which TD feature this ports, how to run
  run.py         # CLI: wires dtouch operators, writes out/*.mp4 + *_frame0.png
docs/NN-name.png # a committed sample frame (the agent's own verification artifact)
```

Build it, render frame 0, read the PNG, fix until it's right, commit. Repeat.
