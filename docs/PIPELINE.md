# Follow the Poseborne pipeline

Trace the implemented capture, training and inference paths to their source. The [verification record](VERIFICATION.md) distinguishes executed checks from retained research results.

[Runnable walkthrough](../README.md#run-the-signal-walkthrough) · [Hardware setup](REFERENCE.md#hardware-requirements) · [Research reference](REFERENCE.md)

## Capture and sample construction

<a href="../figures/pipeline-capture.svg">
<picture>
  <source media="(max-width: 600px)" srcset="../figures/pipeline-capture-mobile.svg">
  <img src="../figures/pipeline-capture.svg" alt="Capture and offline sample construction, from six radios through recorded sessions and aligned feature windows.">
</picture>
</a>

<details>
<summary>Editable flow source</summary>

```text
flowchart TD
    classDef default fill:#ffffff,stroke:#747a7e,color:#20272d
    linkStyle default stroke:#20272d
    TX[Three ESP32-S3 transmitters] -->|ESP-NOW beacons| RX[Three ESP32-S3 receivers]
    RX -->|CSI frames over serial| Host[Host bridge: timestamp, frame and publish]
    Host --> Raw[Raw logs]
    Host -->|MQTT| Session[Recorded HDF5 session]
    Session --> Clock[Clock fitting and epoch separation]
    Clock --> Mask[Amplitude, gap filling and missing-data mask]
    Mask --> Grid[Corrected 100 Hz grid]
    Grid --> Windows[Five-sample feature windows and validity]
```

</details>

The default amplitude route combines **five time samples × 56 carriers × nine radio links** into a `float16` tensor shaped **`(280, 3, 3)`**. The last two axes index receiver and transmitter. A short sequence gap of at most two samples can be interpolated, and its provenance stays in the mask. Longer gaps remain breaks. The window gate rejects a window when any link has at least two masked slots among its five samples. The synthetic walkthrough deliberately exercises a six-packet gap.

The guide shows the offline sample path. Live inference uses its own [ring buffer](../rt/csi_rt/ringbuf.py) to cut windows, while [replay sources](../rt/csi_rt/sources.py) read the recorded link data. These paths share an input contract, not one identical execution graph.

| Boundary | Source |
| --- | --- |
| Radio capture and serial frames | [TX firmware](../firmware/tx/main/main.c), [RX firmware](../firmware/rx/main/main.c) |
| Host timestamps, framing, raw log and MQTT | [host bridge](../host/bridge/bridge.py), [bridge core](../host/csi_host/bridge_core.py) |
| Session recording | [recorder](../host/recorder/recorder.py) |
| Clock correction | [clock fitting](../host/csi_pipe/clockfit.py) |
| Epochs, amplitudes, gap mask, grid and windows | [alignment routines](../host/csi_pipe/align.py), [sample builder](../host/csi_pipe/samples.py) |
| Synthetic data through actual routines | [showcase generator](../scripts/showcase.py) |

## Training: where the checkpoint comes from

<a href="../figures/pipeline-training.svg">
<picture>
  <source media="(max-width: 600px)" srcset="../figures/pipeline-training-mobile.svg">
  <img src="../figures/pipeline-training.svg" alt="Camera teacher labels join recorded CSI windows to train a saved checkpoint.">
</picture>
</a>

<details>
<summary>Editable flow source</summary>

```text
flowchart TD
    classDef default fill:#ffffff,stroke:#747a7e,color:#20272d
    linkStyle default stroke:#20272d
    Video[Synchronized camera recording] --> Teacher[RTMDet and RTMPose teacher]
    Teacher --> Labels[18-joint labels]
    CSI[Recorded CSI] --> Samples[Time-aligned CSI windows]
    Samples --> Pair[Timestamp join, label QA and presence targets]
    Labels --> Pair
    Pair --> Fit[Fit normalization and train pose regression]
    Fit --> Checkpoint[Checkpoint: weights, configuration, statistics]
```

</details>

The camera supplies training labels through third-party ONNX models, downloaded when teacher labeling runs. Training needs synchronized recordings and configuration; no trained checkpoint is bundled.

The model maps CSI features into a pose representation. Its default pose adjacency matrix (PAM) has shape `(3, 18, 18)`; inference extracts the diagonal to obtain 18 joint coordinates and a per-joint confidence channel. An optional vector head outputs `(3, 18)` directly. The confidence channel is a model output, not a calibrated probability of correctness.

| Operation | Source |
| --- | --- |
| Camera teacher and ONNX dependencies | [runner](../teacher/csi_teacher/runner.py) |
| Label construction | [labels](../teacher/csi_teacher/labels.py), [18-joint mapping](../teacher/csi_teacher/body18.py) |
| Window storage | [sample builder](../host/csi_pipe/samples.py) |
| Timestamp join, label QA and presence targets | [pose target builder](../teacher/csi_teacher/pam.py) |
| Normalization and channel convention | [training data](../train/csi_train/data.py) |
| Model and alternate heads | [model](../train/csi_train/model.py) |
| Fitting and checkpoint persistence | [training loop](../train/csi_train/fit.py) |

## Inference and replay

<a href="../figures/pipeline-inference.svg">
<picture>
  <source media="(max-width: 600px)" srcset="../figures/pipeline-inference-mobile.svg">
  <img src="../figures/pipeline-inference.svg" alt="Valid windows run checkpoint-loaded inference. Invalid windows retain the prior displayed pose and state.">
</picture>
</a>

<details>
<summary>Editable flow source</summary>

```text
flowchart TD
    classDef default fill:#ffffff,stroke:#747a7e,color:#20272d
    linkStyle default stroke:#20272d
    Source[Live or recorded CSI packets] --> Buffer[Ring buffer: build window and validity]
    Buffer --> Gate{Valid window?}
    Gate -->|yes| Normalize[Per-link L2 and stored normalization]
    Checkpoint[Supplied trained checkpoint] --> Normalize
    Normalize --> Model[WiSPPN model: extract 18 joints]
    Model --> Presence[Presence gate and pose smoothing]
    Presence --> Fall[Experimental fall-state rules]
    Fall --> Display[Render pose and state]
    Gate -->|no| Hold[Count drop; retain previous displayed state]
    Hold --> Display
```

</details>

[PoseEstimator](../rt/csi_rt/infer.py) loads the checkpoint, checks the channel convention, applies per-link L2 normalization followed by the saved mean and scale, and runs the model. Its coordinate output is `(18, 2)`, with a separate `(18,)` confidence array. A camera is not required as model input; optional synchronized video can be used as the display background.

[Engine.tick](../rt/demo.py) runs the estimator only on a valid window. It applies the [presence gate](../rt/csi_rt/presence.py), [smoother](../rt/csi_rt/smooth.py) and [fall rules](../rt/csi_rt/fall.py). On an invalid window it increments the drop counter and retains the previous displayed pose/state. A visible skeleton can therefore be stale; it is not evidence that every window yielded a new estimate. Motion energy is a separate model-free variability measure used by the fall rules and display; it is not itself a pose or human-presence estimate.

The scheduled window boundary is 50 ms. That describes the pipeline's time grid, not verified live throughput on your hardware. The headless replay command writes a performance summary for the run actually executed. `--random-weights` is explicitly a smoke-test route, not trained inference.

## Reading the demonstrations

The [signal walkthrough](SHOWCASE.md) executes preprocessing on synthetic samples and stops before pose inference. The [original research recording](REFERENCE.md#what-it-detects) shows the source-reported full system. See [evidence and limits](../README.md#evidence-and-limits) before interpreting either demonstration.

The [scientific overview](../scripts/system_figure.py) and [guide layouts](../scripts/presentation.py) are original vector sources. Run `python scripts/presentation.py` to regenerate their desktop and narrow-screen SVGs with the standard library. The README uses 2× PNG exports of the scientific figure to preserve its DIN Alternate and Avenir Next typography without redistributing font files. Collapsed text retains each guide diagram’s flow relationships; keep artwork tied to the linked implementation.
