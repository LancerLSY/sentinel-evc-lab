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

Actual hardware: **RTX 4090 D, 24 GB**. Four-thread, batch-one, in-memory image-to-action
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

Use Python **3.12 or newer** with `lerobot[smolvla]==0.6.1` and CUDA-enabled
PyTorch. The [GPU experiment report](https://github.com/LancerLSY/sentinel-evc-lab/blob/main/docs/gpu_training_results.md)
records Python 3.12.3 and **PyTorch 2.8.0+cu128** for the GPU host. Install the data and
policy dependencies with:

```bash
python -m pip install 'lerobot[smolvla]==0.6.1' 'datasets>=4.8,<5' \
  'pandas>=2,<3' 'pyarrow>=21,<30' 'av>=15,<16'
```

Use a PyTorch CUDA build compatible with the GPU host, as described in the
[training instructions](https://github.com/LancerLSY/sentinel-evc-lab/tree/main/experiments/gpu).
Download this model bundle into a local directory first:

```python
from huggingface_hub import snapshot_download

bundle = snapshot_download("LancerLSY/sentinel-smolvla-so100",
                           local_dir="sentinel-smolvla-so100")
```

Run the following commands from that directory. Download exactly the upstream
files recorded in its identity:

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

## Product integration

The [Sentinel EVC repository](https://github.com/LancerLSY/sentinel-evc-lab)
now includes a managed closed-loop entry for model inference, Panda motion
checking, one-time execution permits, MuJoCo writes, environment feedback and
signed run export. See the [project page](https://lansiyao.com/research/sentinel-vla/project/#managed-loop),
[integration guide](https://github.com/LancerLSY/sentinel-evc-lab/blob/59d3f1a469c7d53482f8bc651e0bfbef33defa6c/docs/managed_closed_loop.md)
and [validation record](https://github.com/LancerLSY/sentinel-evc-lab/blob/59d3f1a469c7d53482f8bc651e0bfbef33defa6c/docs/managed_closed_loop_validation.md).

That managed runtime is a separate supported profile: the official SmolVLA
LIBERO checkpoint with Panda tasks on Linux and MuJoCo 3.3.7. This SO100 overlay
cannot replace its Panda checkpoint. The integration validation covers local
software lifecycle, evidence and interface behavior; it adds no new GPU task-
success result and does not establish live SO100 execution.

## Scope and attribution

This is an offline research model. No robot task-success rate, multi-object
tracking result or live Sentinel permit/runtime integration is established for
this SO100 policy by its recorded-data evaluation. The separate UR5e simulation
study and the managed SmolVLA/Panda product profile use their own controllers
and checkpoints; their task results must not be attributed to this SO100 policy.

The ten reserved calibration episodes were not used in training, development
selection or the reported five-episode test evaluation. The reported evaluation
used the original dataset instructions. Instruction paraphrases, extra clauses
and different-object instructions have not yet been evaluated on those episodes,
so no language-robustness claim is made here.

Upstream SmolVLA, SmolVLM2 and dataset: Apache-2.0, retained in `LICENSE` and
`NOTICE`. Sentinel loading/training source: MIT, retained in `LICENSE-CODE-MIT`.
[Full results and limitations](https://github.com/LancerLSY/sentinel-evc-lab/blob/main/docs/gpu_training_results.md)
and [original artifacts](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/gpu-experiments-20261002).
