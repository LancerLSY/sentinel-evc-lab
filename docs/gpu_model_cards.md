# Trained experimental models

These models belong to the isolated GPU experiment profile. Their weights,
normalization, input identities and evaluation artifacts travel together. The
CLI/App retains its existing numerical predictor until a profile-specific
integration has been evaluated. See [actual measurements](gpu_training_results.md)
and [training commands](../experiments/gpu/README.md).

## W0 numerical ensembles

`w0-numeric-v1` is the earlier nominal-plant residual architecture; its six
compressed NPZ files contain three action-conditioned and three independently
trained no-action members. The design-alpha recalibration is a separate directory
`w0-numeric-design95` and refers back to those frozen weights.

`w0-strict-v4` contains six separately trained members using bounded learned
acceleration and known integration, with root bootstrap preserving all four
candidate siblings. Its `manifest.json` supplies train-only normalization,
state-dictionary tensor names, each weight digest and calibration identities.
Use `train_numeric_strict._model_class(torch)` to construct the matching model,
then map numeric NPZ tensor keys through the member's `tensor_names`. The existing
loader in `recalibrate_numeric.py` demonstrates digest verification, strict state
loading and ensemble inference for the earlier architecture.

Input: eight observed `(r,v,a)` samples and four final 40-step acceleration
sequences at 50 ms intervals. Output: per-candidate relative position and velocity.
Meters, meters/second and meters/second² have explicit meanings in this synthetic
profile. Numeric weights do not apply to SO100 joint fields or Cartesian
MuJoCo targets.

## W2 observed-object MuJoCo ensemble

`w2-mujoco-v2` saves six NPZ members, a dataset/manifest, calibration results,
per-root predictions and training logs. The state-only model is constructed by
`train_mujoco_world._model_class(torch)(manifest["normalizers"])`. For each member,
map its numeric NPZ tensor keys using `tensor_names` and call
`load_state_dict(..., strict=True)`.

Input: eight 22D observed tray/payload/support states, past XYZ targets and four
40-step future XYZ target sequences. Output: payload-relative XYZ position and
velocity, rotation-6D and angular velocity. Position uses meters; the full 15D
aggregate combines different quantities. Mass and friction are label-generation
metadata and are never neural inputs.

The full-state and XY-risk envelopes are separate calibration objects. The
two-second forecast does not cover the additional half-second outcome hold.
Reuse requires the same observations, timing, candidate construction, controller
schedule and calibration rule.

## W2 image-to-object WorldGuard

`w2-mujoco-visual-v1` adds nine learned members: visual object prediction,
robot-only control and independently trained visual/no-action ablation. It
reuses W2 labels and reconstructs historical poses only for rendering. Two frozen
camera definitions, crop, encoder identity, train-only per-camera PCA and feature
cache are recorded in its manifest. Ten original rendered PNG samples accompany
the weights.

Ordered channels are `overview, front`, with the shifted stress layout ordered
`overview_shift, front_shift`. Preserve this order explicitly; the manifest's
sorted camera dictionary alone does not preserve it. All nine NPZ files passed
strict dev-only reload, and [the independent audit](gpu/2026-10-02/w2_visual_independent_audit.json)
records exact source/PCA/split/label identities and input isolation. Training
uses branch minibatches with grouped root splits; it is not root bootstrap.

Input: eight 128D spatial image features, six observed tray position/velocity
fields, past XYZ targets and candidate future XYZ targets. Output: 40 predicted
15D payload states. The model never reads true historical/current payload fields,
true `initial_output` or future images. An initial-state head estimates the
starting payload state. Construct
`train_mujoco_visual._model_class(torch)(manifest["normalizers"], visual_flag)`
and use each member's tensor mapping to strictly load its NPZ.

The same profile with shifted cameras fails the XY risk comparison on 47/500
roots. Layout, image processing, frozen encoder, PCA and calibration identity
are part of the supported input contract. Fixed spatial grid regions do not
provide persistent multi-object identities. The earlier observed-object W2
model has additional true state information and is listed as an oracle reference.

## W1 real SO100 visual joint predictor

`w1-real-visual-v1` trains three state/action members, three visual/state/action
members and three visual members without future action input. Numeric NPZ keys
are actual PyTorch state-dictionary names and load with `allow_pickle=False`.
The model constructor is `train_real_world._model_class(torch)`; use
`normalization.json`, recorded hidden size and the family's future-action/visual
flags before strict state loading.

Input: states `x[t-7:t]`, preceding logged targets `u[t-8:t-1]`, historical
128D image features and future logged targets `u[t:t+39]`. Output:
states `x[t+1:t+40]`. The `u[t] → x[t+1]` relation follows a recording-stream
assumption. Targets are logged teleoperation/processor values rather than
confirmed sent actuator commands. Dataset-native field units are undeclared, so compare per-joint errors
and train-standard-deviation-normalized errors rather than treating a pooled
number as meters or robot accuracy.

The `visual-cache-v1` companion contains two train-only PCA mappings, feature
identity/alignment metadata and the cached features. The frozen encoder is
official ImageNet ResNet18; its two-camera 2×2 spatial grids retain regions but
do not establish tracked object identities. Frames align through the original
video timestamps. Future images are excluded.

All nine saved NPZ checkpoints passed an independent strict reload and finite
inference on ten dev-only windows. No test window was used to select a model
during that check. [The evidence](gpu/2026-10-02/w1_reload_verification.json) has
SHA256 `a32930ba1f41150d441f9ce9c1490492bce3192346c1c6b493a203a4a78a7ae3`.

The real-video cache was produced by source SHA256 `9afc79d73e9d4e4e2c8c32d8377befe4cba42ae9f73d3059b791caa62ecdac66`. Its exact executed source is retained with the artifacts. The published cache producer (`837171f70348d44d311e825e74b971c8d264865bd7306447510d5b6b2d995abf`) adds full encoder-digest and dataset-manifest checks; it was not used to regenerate these features. The original encoder digest was independently verified in full. Numerical preprocessing is unchanged.

## SmolVLA fine-tune

`smolvla-so100-v1` uses the actual pinned LeRobot SmolVLA base and SmolVLM2
backbone. The frozen VLM stays outside the optimizer; 99,880,992 action-expert,
state/action projection and action-time parameters are trainable. The registered
run uses batch 8, bfloat16 and 5000 optimizer steps.

`checkpoints/best/trainable_state.pt` is an overlay for the pinned base, not a
standalone full policy. `checkpoint.json` binds its hash, the immutable run
identity and train-only statistics. Rebuild the base using
`train_smolvla._build_policy`, convert `train_only_stats.json` entries to float32
tensors, then load the overlay with `torch.load(..., weights_only=True)` and
`train_smolvla._load_trainable_state`. The training entrypoint performs this fresh
reload before comparing base and fine-tuned predictions.

Input: the two current RGB camera observations, six recorded state fields and
task language. Output: one 50×6 action chunk from `predict_action_chunk`, followed
by the LeRobot postprocessor returning dataset-native values. The ten retained
shadow windows save original images, state, fixed sampling noise, recorded target
chunks, base/fine outputs and individual tensor hashes.

The model's original Apache-2.0 licensing and dataset attribution accompany the
checkpoint overlay; Sentinel's code license remains MIT. Download the pinned
original base/backbone from their owners. Large third-party weights and recorded
video are not duplicated in repository history.

## Loading and retention

Use the source scripts at the recorded SHA rather than an arbitrary later model
definition. Check manifest sizes and SHA256 before loading. Keep numeric-only NPZ
loading and the trainable overlay's `weights_only=True` path. Optimizer/RNG files
are for resuming this run and are separate from inference weights.

The local full artifact backup is under ignored `runs/gpu-20261002/`. Compact
measured JSON is retained under `docs/gpu/2026-10-02/`; distributable trained
weights and supporting configuration belong to the experimental release assets.
