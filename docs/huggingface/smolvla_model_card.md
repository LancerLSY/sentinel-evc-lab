---
license: apache-2.0
library_name: pytorch
base_model: lerobot/smolvla_base
datasets:
  - lerobot/svla_so100_pickplace
tags:
  - robotics
  - smolvla
  - lerobot
  - action-prediction
  - trainable-overlay
  - sentinel-evc
---

# Sentinel SmolVLA SO100

An evaluated **trainable overlay** for the pinned SmolVLA base. It adapts the
action expert and state/action projections to real SO100 PickPlace recordings.
The vision-language backbone stays frozen. This bundle requires the original
base/backbone; it is not a standalone `from_pretrained` policy directory.

## Recorded-data result

| Policy | Native action MAE | MAE / training action standard deviation |
|---|---:|---:|
| Pinned original base | 13.321098 | 0.673166 |
| Dev-selected fine-tune | 4.472850 | 0.245984 |

Normalized action MAE decreased **63.46%** in one paired held-out evaluation.
Five held-out episodes contain 1,926 overlapping windows; the independent unit
is the episode. Each prediction is a 50×6 action chunk; padded targets are
excluded. Dataset-native physical units are undeclared. These values measure
action reconstruction, not task completion or physical robot accuracy.

Training used 5,000 optimizer updates, batch 8, bfloat16 autocast and 99,880,992
trainable parameters. Dev selected update 3750, loss 0.161077. The split was
30 train / 5 dev / 10 reserved calibration / 5 test episodes; normalization fit
only training data. Calibration episodes were untouched by this policy run.

Actual hardware: **RTX 4090 D, 24 GB**. RTX 5090 is a future target configuration
and has no measurements here. Four-thread, batch-one, in-memory image-to-action
inference measured P50/P95 **229.26/236.90 ms**, including preprocessing and
unnormalization, excluding video decoding, transport and actuation.

## Pinned inputs and checkpoint

- Dataset: `lerobot/svla_so100_pickplace`, revision `728583b5eaf9e739a7f119e2def466fa1d552402`.
- Base: `lerobot/smolvla_base`, revision `d9f33c94a60fb382c90dea2164c96845bd955e28`.
- Backbone: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- `trainable_state.pt`: SHA256 `786fb0b1bdd89824960d57fb88564c0b7bfbd36c011da8e7cf721a7905dca30b`.
- Training source: SHA256 `bd6533682e152dab24b55fcec1730e49b04f7ea764f7b2af0826ee6c3d18cb70`.

The bundle includes the selected overlay, original feature metadata, train-only
statistics, split/run identities, evaluation and loading source. It excludes
optimizer/RNG state, original videos and frozen upstream weights.

## Load and predict

Use the isolated experiment environment with `lerobot[smolvla]==0.6.1`, PyTorch
CUDA, `av>=15,<16`, datasets, pandas and pyarrow as described in the
[training instructions](https://github.com/LancerLSY/sentinel-evc-lab/tree/main/experiments/gpu).
After obtaining this bundle, download exactly the upstream files recorded in
its identity:

```python
import json
from huggingface_hub import snapshot_download

identity = json.load(open("run_identity.json"))
for key, destination in [("base_model", "base"), ("vlm_backbone", "vlm")]:
    source = identity["bindings"]["downloads"][key]
    snapshot_download(source["repo_id"], revision=source["revision"],
                      allow_patterns=[f["path"] for f in source["files"]],
                      local_dir=destination)
```

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python load_smolvla_overlay.py \
  --bundle . --base base --vlm vlm \
  --example example_inputs.pt --out offline_prediction.pt
```

The example uses one saved **dev-only** input. The loader verifies upstream
file hashes, source/checkpoint bindings, train-only normalization and the exact
155-tensor overlay. It emits one finite `[1,50,6]` chunk and never calls a driver.
For another input preserve camera order `observation.images.top`, then
`observation.images.wrist`, CHW RGB images, six recorded state fields and task
language. The example's zero action placeholder is a preprocessing field, not
a motor command or future observation.

## Scope and attribution

This is an offline research model. No robot task-success rate, multi-object
tracking result or live Sentinel permit/runtime integration is established by
this evaluation. The separate UR5e simulation study uses its own controller;
its task results must not be attributed to this SO100 policy.

Upstream SmolVLA, SmolVLM2 and dataset: Apache-2.0, retained in `LICENSE` and
`NOTICE`. Sentinel loading/training source: MIT, retained in `LICENSE-CODE-MIT`.
[Full results and limitations](https://github.com/LancerLSY/sentinel-evc-lab/blob/main/docs/gpu_training_results.md)
and [original artifacts](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/gpu-experiments-20261002).
