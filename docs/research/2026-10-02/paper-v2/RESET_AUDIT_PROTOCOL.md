# LIBERO reset fidelity audit protocol

## Question

The native SmolVLA evaluation recorded 0/10 successes on LIBERO-Spatial task 5, where the black bowl begins on the ramekin. An unresolved upstream report suggests that MuJoCo version changes may alter settled object placement. This audit tests the narrower factual question: with policy inference and actions removed, does the same fixed LIBERO reset produce different pre-action simulator state under MuJoCo 3.8.1 and 3.3.7?

The upstream report motivates the comparison; it is not evidence that a version effect occurs here. A measured reset difference also cannot establish that MuJoCo caused the separate task-5 policy result.

## Frozen design

- **Versions:** current MuJoCo 3.8.1 and an isolated MuJoCo 3.3.7 prefix. The 3.3.7 package is installed with `pip --no-deps --target` and selected only in a fresh process through `PYTHONPATH`; the 3.8.1 environment is not modified.
- **Software held fixed:** Python 3.12, LeRobot 0.6.1, hf-libero 0.1.4, and robosuite 1.4.0.
- **Environment held fixed:** LIBERO-Spatial, fixed init states, hard reset, relative control, 20 Hz, one synchronous environment, and 360×360 observations.
- **Tasks:** task 5 is the reported failure task. Task 0 is the control because it uses the same black-bowl, ramekin, and plate object family without starting the target bowl on the ramekin.
- **Formal grid:** task IDs 0 and 5, init-state indices 10–19, and seeds `41021 + state_index`, under both MuJoCo versions. This is 20 resets per version and 40 resets total.
- **Disjointness:** native policy evaluation used states 0–9. Compatibility preflight uses task 5/state 47/seed 41068. Other existing preflights used states 48 and 49. None enter the formal comparison.
- **No policy path:** no checkpoint is loaded, no model inference runs, and no action is submitted.

Before any formal capture, both versions must complete the frozen task-5/state-47 preflight. Each formal process verifies the exact version label, task/state/seed, pre-action/no-policy/zero-action record, asset tree, native protocol, environment, and every output hash. It then records the absolute path and SHA-256 of both preflight manifests.

## Retained evidence

Each reset asserts the requested init-state index against the environment's post-reset index and asserts the requested seed against the environment's post-reset seed. It retains:

- exact `uint8` RGB arrays for `agentview_image` and `robot0_eye_in_hand_image`;
- `qpos`, `qvel`, `act`, `ctrl`, and simulator time;
- named-body position and quaternion for `akita_black_bowl_1_main` and `glazed_rim_porcelain_ramekin_1_main`;
- bowl-minus-ramekin relative position and each object's tilt from world vertical;
- asset-tree, BDDL, init-state, native-protocol, runner, protocol, and output hashes;
- effective package versions and the imported MuJoCo module path.

All measurements are taken immediately after reset and before any action.

## Analysis and claims

Comparisons are paired by task, state index, and seed. The report includes exact camera equality and absolute pixel differences; absolute state-vector differences; body translation and quaternion-angle differences; relative-position differences; tilt differences; and a descriptive task-0 versus task-5 contrast. There is no post-hoc pass threshold and no parameter tuning after the formal run begins.

This is a parameter-fixed reset diagnostic. It does not measure task success, policy quality, intervention efficacy, safety, or training behavior. Pose or pixel differences cannot by themselves explain the native task-5 result.

## Commands

Current-version preflight:

```bash
LIBERO_CONFIG_PATH=/path/to/libero-config \
python experiments/vla/audit_libero_reset.py \
  --mode preflight --version-label current \
  --protocol experiments/vla/reset_audit_protocol.json \
  --native-protocol experiments/vla/libero_protocol.json \
  --libero-config /path/to/libero-config \
  --output-dir runs/paper-v2/reset-audit-preflight/current
```

Legacy-version preflight uses the same command with the isolated prefix first on `PYTHONPATH`, `--version-label legacy`, and a different empty output directory. Formal capture changes `--mode` to `final` and supplies both preflight manifests with `--preflight-current` and `--preflight-legacy`. After both 20-reset captures finish, `--mode compare` consumes their manifests and writes the paired comparison into a third empty directory.
