# Reproduce the signal walkthrough

[Complete pipeline and source map](PIPELINE.md) · [Back to the runnable example](../README.md#run-the-signal-walkthrough)

The README animation is an original visualization of actual retained numerical routines on synthetic input. It is separate from the research recordings in the reference guide. There is no trained pose model, hardware, network, camera or broker in this command.

```bash
python -m pip install -r requirements-showcase.txt
python poseborne.py showcase --out signal-demo
```

The output directory must be new. It contains:

- `signal-path.gif`: 64 frames at 1120 × 700, for the desktop diagnostic.
- `signal-path-mobile.gif`: the same 64 source times in a separate 420 × 650 layout with readable labels on narrow screens.
- `signal-path.png` and `signal-path-mobile.png`: reduced-motion views of the actual rejected window.
- `signal-path.json`: seed, numerical-input/window hashes, links, dimensions, rejection counts and presentation timing.

Both animations loop. Ordinary frames last 70 ms; the rejected window lasts 900 ms. White backgrounds, graphite text, cobalt signal values and red missing-data warnings are functional plot encodings. Brightness maps amplitude values from 18 to 61, clipped for display only; stored feature windows are unchanged. The narrow view keeps the link plot, selected tensor, gate result and motion energy, rather than shrinking the desktop labels.

The default Pillow font works across platforms. An optional local `--font /path/to/font.ttf` changes typography only. The checked-in images use macOS Helvetica and Pillow 12.3.0; the font itself is not included. NumPy 2.5.4 produced the current numerical record. Different library versions may change bytes; recorded hashes describe the actual run, not a cross-version promise.

## What is executed

1. Seed `20261014` generates four seconds of 100 Hz, signed 8-bit I/Q pairs for nine links and 56 carriers. I/Q are the two components of the synthetic radio samples. The sinusoidal envelope illustrates a changing signal; it is not a physical room/body simulator.
2. `csi_pipe.align.amplitude` computes `sqrt(I² + Q²)` from those pairs.
3. Six consecutive samples are removed from RX3/TX3. `fill_gaps` retains that interval as a break because it exceeds the two-sample interpolation limit; `grid_block` computes amplitudes and the actual missing-data mask.
4. `cut_windows` makes 80 feature windows of shape `(280, 3, 3)`, stored as `float16`. It rejects a window if any link has at least two masked slots among its five samples. In this fixture, 79 pass and one fails.
5. `csi_rt.presence.MotionEnergy` computes the mean of per-link amplitude-mean standard deviations over its trailing half-second history. This is model-free signal variability; it is not a presence or pose prediction.
6. Pillow draws the resulting arrays and gate state in desktop and narrow-screen diagnostic layouts. Both views explicitly label the input synthetic; no pose model is loaded.

The trace is an offline walkthrough. The cursor marks the end of the displayed five-sample window; the entire four-second input is visible for context. Playback pauses on the actual rejected window at source time **1.85 s**, without changing its data or cursor, then continues. The numerical record lists each frame's playback duration; the loop takes 5.31 seconds. Playback timing is a presentation choice, not a processing-throughput benchmark.

## Verify it

```bash
python -m unittest discover -s tests -v
```

The tests repeat the seeded computation, compare numerical records, check tensor shape/type/finite values, require the missing-data gate to reject a window and confirm an existing output directory remains untouched. The entry-point test checks every command's argument and failure-code propagation without launching hardware routes.

The source generator is [scripts/showcase.py](../scripts/showcase.py). Edit visual styling there, regenerate to a fresh directory, inspect the animation and still image, then replace both GIFs, both stills and their matching JSON record together. Do not use source-reported hardware results or fabricated skeletons as evidence for this walkthrough.
