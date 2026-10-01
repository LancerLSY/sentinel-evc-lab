# GPU experiment mission — 2026-10-02

The user authorized a ten-hour training and research budget, ending at
2026-10-02 10:01 Asia/Shanghai. These scripts run in an isolated experimental
environment. The installable Sentinel package keeps its existing dependencies.
No new software tests are part of this mission; the evidence comes from actual
training, disjoint evaluation, calibration, ablations and saved model artifacts.

## Fixed inputs and material passport

| Input | Owner / source | Immutable revision | Use / license |
|---|---|---|---|
| SO100 PickPlace | [LeRobot dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | `728583b5eaf9e739a7f119e2def466fa1d552402` | Real recorded robot demonstrations; Apache-2.0 |
| SmolVLA base | [LeRobot model](https://huggingface.co/lerobot/smolvla_base) | `d9f33c94a60fb382c90dea2164c96845bd955e28` | Actual VLA checkpoint; Apache-2.0 |
| SmolVLM2 backbone | [Hugging Face model](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | Frozen VLA backbone; Apache-2.0 |
| ResNet18 V1 | [PyTorch official weights](https://download.pytorch.org/models/resnet18-f37072fd.pth) | SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | Frozen spatial image features; ImageNet pretraining |
| Numeric plant | Sentinel source | Baseline `cb79c3ae42120f7f9375ddb0ef63b40dbe37b696` plus per-script SHA256 | Synthetic observable-state experiment |
| XYZ tray fixture | Sentinel procedural MJCF / [MuJoCo](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjstate) | MuJoCo 3.14.0; MJCF/state hashes per root | Actual simulation contact dynamics; no robot hardware |

Official Hub API metadata is obtained independently. `fetch_snapshot.py` verifies
each file against that metadata, using LFS SHA256 or the Git blob ID. A transport
mirror can supply bytes; it does not establish provenance. Weights, original
data, private design documents and credentials are kept outside Git.

The SO100 snapshot contains 50 episodes, 19,631 frames at 30 Hz, six joint/gripper
fields and two recorded cameras. The metadata does not declare physical units
or controller acknowledgments. Its logged action targets are not Cartesian
plans, and it lacks object-pose, force and incident annotations.

## Experiments and success criteria

| Run | Model / data | Frozen evaluation |
|---|---|---|
| W0 | 48D residual GRU; 8 past samples, 40 future steps, 4 sibling plans; 3 members plus separately trained no-action members | 800 train / 120 dev / 199 cal / 300 test roots; physical-ID, fixed physical, persistence and action/history interventions |
| W0 calibration correction | Reuse the first run's dev-selected weights and normalization | Keep the initial 90% protocol results; evaluate design alpha=.05 on fresh cal/test roots without retraining or selecting on old test results |
| W0 strict v4 | No hidden plant constants in forward; bounded learned acceleration plus known integration; member-level root bootstrap and four siblings per sampled root | Reuse fixed train/dev; fresh 199 cal / 300 test roots; frozen earlier variant and observable-history physical ID comparisons |
| W2 | 128D GRU; 22 observed state fields and actual MuJoCo-generated futures | 1000 / 200 / 299 / 500 roots; four complete-state paired siblings; separate full-state and XY-risk calibration; geometry-earliest / fixed-slow / learned selection |
| W2 visual object | Actual EGL-rendered recorded histories, frozen spatial ResNet18/PCA; robot state and images estimate object state | Reuse W2 labels without re-integrating; exclude true payload input/initial state; robot-only and no-action models; frozen-calibration camera-layout stress |
| W1 real state | Causal action-conditioned joint-state GRU | Shared episode split: 30 train / 5 dev / 10 cal / 5 test; train-only scaling; action-target and persistence baselines |
| W1 real visual | Frozen ResNet18 spatial grids, train-only PCA, historical features plus joint state | Same episodes and paired training seeds; state-only comparison and separately trained no-future-action model; future images never enter the predictor |
| SmolVLA | Actual base checkpoint; frozen VLM; action expert and state/action projections fine-tuned | Same episode split; dev-selected checkpoint reloaded; base/fine action errors per joint and training scale; ten hashed shadow windows |

Train/dev/cal/test are different root or episode groups. Every simulated sibling
stays with its root. Normalization/PCA fit only on training data; checkpoint
selection and residual scales use dev; conformal ranks use independent cal.
With ten real calibration episodes, alpha=.1 permits rank 10; alpha=.05 would
require rank 11 and yields no finite envelope. This is recorded explicitly.

W2 forecasts two seconds. The additional half-second hold is a separate outcome
diagnostic, not covered by the two-second envelope. Hidden mass/friction generate
labels and remain outside neural inputs. The direct offline control schedule has
no product ACK/prepare intervals, so it does not establish product closed-loop
behavior. All difficult roots and negative results remain in the artifacts.

## Minimal environment and entrypoints

Reuse the GPU host's PyTorch/CUDA installation in a venv with system site packages.
The official [LeRobot 0.6.1 dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)
and [SmolVLA API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py)
are pinned for this experiment. PyAV supplies video decoding.

```bash
python -m pip install 'lerobot[smolvla]==0.6.1' 'datasets>=4.8,<5' \
  'pandas>=2,<3' 'pyarrow>=21,<30' 'av>=15,<16' mujoco==3.14.0

python experiments/gpu/train_numeric_gru.py --out runs/w0 --device cuda
python experiments/gpu/recalibrate_numeric.py --run runs/w0 \
  --out runs/w0-design95 --device cuda
python experiments/gpu/train_numeric_strict.py --reference-run runs/w0 \
  --out runs/w0-strict --device cuda --train-no-action
python experiments/gpu/train_mujoco_world.py --out runs/w2 --device cuda \
  --workers 8 --train-no-action
MUJOCO_GL=egl python experiments/gpu/train_mujoco_visual.py --dataset-run runs/w2 \
  --weights "$WEIGHTS/resnet18-f37072fd.pth" --out runs/w2-visual \
  --device cuda --source-commit "$(git rev-parse HEAD)"
MUJOCO_GL=egl python experiments/gpu/recalibrate_mujoco_camera.py --dataset-run runs/w2 \
  --source-run runs/w2-visual --weights "$WEIGHTS/resnet18-f37072fd.pth" \
  --out runs/w2-camera-recal --device cuda
python experiments/gpu/cache_visual_features.py --data-dir "$DATA" \
  --out runs/visual-cache --weights "$WEIGHTS/resnet18-f37072fd.pth" --device cuda
python experiments/gpu/train_real_world.py --data-dir "$DATA" \
  --visual-cache runs/visual-cache/visual_features.npz --out runs/w1 \
  --device cuda --source-commit "$(git rev-parse HEAD)"
python experiments/gpu/train_smolvla.py --dataset-root "$DATA" \
  --dataset-revision 728583b5eaf9e739a7f119e2def466fa1d552402 \
  --model-path "$WEIGHTS/smolvla_base" --vlm-path "$WEIGHTS/SmolVLM2-500M-Video-Instruct" \
  --model-revision d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --vlm-revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --output-dir runs/smolvla --steps 5000
```

Each run saves its configuration, source hash, input identities, training log,
selected weights, calibration and held-out metrics. Run completion establishes
that those steps executed; scientific acceptance additionally requires the
reported comparisons. Real-data inference is shadow evidence. Device motion,
object-state supervision and hardware task success require their own evidence.

Actual measurements and negative results are recorded in
[the GPU training report](../../docs/gpu_training_results.md). The first W0 script
is an engineering variant with fixed nominal residual dynamics and branch-level
training batches; the strict script is the separate architecture/batching
alignment experiment. Neither is silently substituted for the other.

EGL rendering requires the host's OpenGL/EGL dispatcher and NVIDIA driver. The
measured GPU host needed Ubuntu's `libegl1`; this is a renderer system package,
not an added Python/core dependency. The fixed-camera visual object experiment
observed 0/500 unsafe selections, while shifting only the test cameras produced
47/500. Preserve that comparison when using the model.
