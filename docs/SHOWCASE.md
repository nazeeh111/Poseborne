# Reproduce the signal walkthrough

The README animation is an original visualization of actual retained numerical routines on synthetic input. It is separate from the research recordings in the reference guide. There is no trained pose model, hardware, network, camera or broker in this command.

```bash
python -m pip install -r requirements-showcase.txt
python wipose.py showcase --out signal-demo
```

The output directory must be new. It contains:

- `signal-path.gif`: 64 frames, 1120 × 660 pixels, 70 ms per frame, looping.
- `signal-path.png`: still view for reduced motion and image previews.
- `signal-path.json`: seed, numerical-input/window hashes, links, dimensions and rejection counts.

The default Pillow font works across platforms. An optional local `--font /path/to/font.ttf` changes typography only. The checked-in images were rendered with macOS Helvetica and Pillow 12.3.0; the font itself is not included. NumPy 2.3.5 produced the committed numerical record. Different library versions may change bytes; the script records hashes of the actual run rather than promising cross-version identity.

## What is executed

1. Seed `20261014` generates four seconds of 100 Hz, signed 8-bit I/Q pairs for nine links and 56 carriers. I/Q are the two components of the synthetic radio samples. The sinusoidal envelope illustrates a changing signal; it is not a physical room/body simulator.
2. `csi_pipe.align.amplitude` computes `sqrt(I² + Q²)` from those pairs.
3. Six consecutive samples are removed from RX3/TX3. `fill_gaps` retains that interval as a break because it exceeds the two-sample interpolation limit; `grid_block` computes amplitudes and the actual missing-data mask.
4. `cut_windows` makes 80 feature windows of shape `(280, 3, 3)`, stored as `float16`. It rejects a window if any link has at least two masked slots among its five samples. In this fixture, 79 pass and one fails.
5. `csi_rt.presence.MotionEnergy` computes the mean of per-link amplitude-mean standard deviations over its trailing half-second history. This is model-free signal variability; it is not a presence or pose prediction.
6. Pillow draws the resulting arrays and gate state. The inference stage remains visually inactive, labeled “trained weights required.”

The trace is an offline walkthrough. The cursor marks the end of the displayed five-sample window; the entire four-second input is visible for context. The GIF's playback time is a presentation choice, not a processing-throughput benchmark.

## Verify it

```bash
python -m unittest discover -s tests -v
```

The tests repeat the seeded computation, compare numerical records, check tensor shape/type/finite values, require the missing-data gate to reject a window and confirm an existing output directory remains untouched. The entry-point test checks every command's argument and failure-code propagation without launching hardware routes.

The source generator is [scripts/showcase.py](../scripts/showcase.py). Edit visual styling there, regenerate to a fresh directory, inspect the animation and still image, then replace the three matching README assets together. Do not use source-reported hardware results or fabricated skeletons as evidence for this walkthrough.
