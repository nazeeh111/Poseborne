# Reproduce the checkpoint input contract

No trained checkpoint or compatible labeled ESP32-S3 recording is distributed.
These checks execute actual WiSPPN forwards on generated CSI with temporary,
seeded **random weights**. They test software contracts, not learned pose accuracy,
human presence, fall detection, or physical six-board operation.

```bash
python -m venv .venv-contract
source .venv-contract/bin/activate
python -m pip install torch numpy h5py PyYAML
python -m unittest discover -s tests -p test_inference_contract.py -v
```

The independent reference normalizes each packet and radio link separately,
optionally applies `10**(RSSI/20)`, applies the checkpoint's saved statistics,
then uses the same float16 storage boundary as training and evaluation. The
checks compare both model heads against this reference and repeat with a
changed carrier ordering. They reject mismatched channel conventions, invalid
statistics, missing phase/RSSI and malformed inputs. No model, dataset or camera
is downloaded or opened by the checks.

## Supported paths

| Input | Supported checkpoint features |
| --- | --- |
| Packet replay / live `poseborne.py infer` | amplitude only |
| Built HDF5 samples through `poseborne.py train eval` | amplitude, optional RSSI and sanitized phase |
| `PoseEstimator(window, phase=..., rssi=...)` | features declared by the supplied checkpoint |

For the Python estimator, amplitude and sanitized phase each have shape
`(280,3,3)`, and RSSI has shape `(5,3,3)`. Channel order is five past-to-present
packets, 56 low-to-high subcarriers, receiver then transmitter. Phase is appended
after amplitude. Pass only the optional features declared in the checkpoint.
The packet source/ring buffer currently retains amplitude alone, so its CLI
rejects RSSI/phase checkpoints before starting replay or live acquisition.
A missing `config.features` means legacy amplitude-only input. Checkpoints must
carry the existing exact `channel_convention` and finite `(3,3)` means and
strictly positive scales, including separate statistics for phase.

```bash
# With your own legally obtained compatible checkpoint and labeled recordings:
cp configs/train.example.yaml configs/train.yaml
# Set the session paths and explicit train/val/test roles in configs/train.yaml.
python poseborne.py train eval --config configs/train.yaml \
  --ckpt /path/to/trained.pt --split test --device cpu
```

The existing evaluator loads checkpoint statistics without refitting them on the
test set and writes `report-test.json` beside the checkpoint. Its joint errors
are in camera pixels and its labels are camera-teacher outputs, unless separately
annotated and verified. A time split from one recording does not establish
cross-room, cross-person or cross-session quality.

## Evidence and missing assets

The previous estimator ignored checkpoint RSSI requirements and failed with a
channel-count exception on phase models. Even amplitude-only replay skipped the
training float16 boundary. The shared preprocessing now matches the saved
training contract and unsupported packet inputs fail with an actionable error.

The retained upstream implementation is [sel00000/csi-pose](https://github.com/sel00000/csi-pose).
Its source reports a single-session experiment, while the checked upstream and
Poseborne release lists contain no downloadable model or recording. Other
Wi-Fi pose datasets and original Intel 5300 WiSPPN weights use different radio,
feature and architecture contracts; they cannot serve as a compatible ESP32
checkpoint without a separately justified adaptation and evaluation.

To unlock real held-out inference, supply licensed/consented nine-link HDF5
recordings with synchronized teacher labels, `valid`, `label_ok`, `presence`,
`Y` and `labels.W/H`, plus a checkpoint saved by this training code with matching
features and statistics. Retain dataset/subject/session split provenance, model
training history, licenses and SHA-256 digests. Neither this fix nor synthetic
tests resolve trained pose accuracy or independent hardware validation.
