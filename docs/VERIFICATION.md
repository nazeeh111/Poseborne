# Verification

Original packaging checks executed on 2026-09-22. Python 3.12.13, PyTorch 2.14.0, NumPy 1.26.4 on macOS ARM64.

## Signal presentation update: 2026-10-14

Executed locally with Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0. The original computational kernels, firmware, research media and license notices remain unchanged by this update.

- `python wipose.py showcase` renders a deterministic synthetic I/Q trace through the actual amplitude, gap-filling, alignment, feature-window and motion-energy routines. The numerical record is [signal-path.json](../figures/signal-path.json): 80 windows, 79 accepted, one rejected by the deliberately injected six-packet gap. No pose model is loaded.
- Three unit tests pass: argument/failure propagation for all five CLI routes; seeded numerical repeatability, tensor shape/type/finite values and missing-data rejection; refusal to overwrite an existing output directory.
- The GIF decodes to 64 distinct 1120 × 660 frames, each lasting 70 ms. All frames are decoded for verification; the first, missing-data gate and final views are inspected. The default Pillow font is inspected separately to verify legible cross-platform labels. An optimized GIF disposal mode preserves every decoded pixel while reducing the file size.
- README and walkthrough relative file links and reference-guide heading anchors resolve. Both the main CLI and `showcase --help` work before optional numerical imports. `git diff --check` passes.

These checks establish the synthetic walkthrough, not hardware sensing, trained pose accuracy, real-time throughput, or fall-detection reliability. The optional font changes visual rendering only. Numerical and image hashes describe the checked-in assets and are not guaranteed across dependency versions. Reproduction instructions are in [SHOWCASE.md](SHOWCASE.md).

## Original packaging checks: passed

- 107 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 67 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- Two seeded CPU model variants (PAM and vector heads) produced byte-identical arrays against the baseline. Presence-gate and motion-energy outputs matched; CRC16 passed the standard 123456789 fixture. Models used reduced width and random weights, not trained checkpoints. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/previous-checkout
python scripts/verify_kernels.py --baseline-dir /path/to/previous-checkout
```

Parity commands take an explicit separate prior checkout; they do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

ESP32 hardware/firmware build and flashing, serial links, MQTT, teacher labeling, model downloads, training datasets, full trained inference, fall-detection accuracy, and cross-session evaluation were not executed. The reference figures/results are recorded project material, not new measurements.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. Original numerical code, calibration, and recorded scientific images were preserved.
