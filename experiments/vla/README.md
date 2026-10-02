# Reproducing the frozen SmolVLA LIBERO evaluation

This evaluation uses a separate Python environment and does not add LIBERO to
the product's core dependencies.  The verified runtime was Python 3.12.3,
PyTorch 2.8.0+cu128, LeRobot 0.6.1, `hf-libero` 0.1.4, robosuite 1.4.0,
MuJoCo 3.8.1, and num2words 0.5.14 on an RTX 4090 D 24 GB GPU.

```bash
python3.12 -m venv /path/to/libero-env
/path/to/libero-env/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
/path/to/libero-env/bin/pip install 'lerobot[libero]==0.6.1' hf-libero==0.1.4 robosuite==1.4.0 mujoco==3.8.1 num2words==0.5.14
```

Download the immutable policy and backbone revisions to local directories.
The runner verifies all action-producing files and the complete 15-file
backbone tree before constructing an environment.

```bash
hf download lerobot/smolvla_libero \
  --revision 31d453f7edd78c839a8bbc39744a292686daf0de \
  --local-dir /path/to/models/smolvla_libero-31d453f
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir /path/to/models/SmolVLM2-500M-Video-Instruct
```

The official asset source is
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
The measured runtime used the asset directory inside the isolated
`site-packages/libero/libero/assets` tree.  It contains package-bundled files
plus the following required official asset families copied from that pinned
dataset revision:

```text
stable_scanned_objects/akita_black_bowl/**
stable_scanned_objects/glazed_rim_porcelain_ramekin/**
stable_scanned_objects/plate/**
stable_hope_objects/cookies/**
articulated_objects/wooden_cabinet/**
articulated_objects/flat_stove/**
textures/**
```

This is deliberately described as a mixed installed tree, not as a pristine
dataset snapshot.  Before either run, the runner requires exactly 955 files,
188,829,695 bytes, and canonical tree SHA-256
`f04f72afb7503afb071f9de7c6734b0eca147f87549df0a9080cc987afb17781`.
It also verifies the BDDL and fixed-init-state file hash for every one of the
ten LIBERO-Spatial tasks.  A different installed tree stops the run; do not
edit the protocol to accept it after seeing outcomes.

This historical directory digest includes download-cache metadata and interrupted
downloads. The original directory was not archived and cannot be reconstructed
from a fresh Hub download. The formal protocols retain that identity. New
diagnostic replays instead freeze and verify the actual official model files
individually; see the [portable-asset reproducibility note](../../docs/research/2026-10-02/paper-v2/REPRODUCIBILITY_NOTE.md).

Set `LIBERO_CONFIG_PATH` to a directory containing `config.yaml`.  The file
must point `assets`, `bddl_files`, and `init_states` to the resolved isolated
installation.  Then run the disjoint compatibility episode first:

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/path/to/libero-config \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/libero-env/bin/python experiments/vla/run_libero_closedloop.py \
  --protocol experiments/vla/libero_protocol.json \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --output-dir /path/to/runs/libero-preflight-state49 \
  --mode preflight
```

The preflight is task 0, fixed state 49, seed 41020, and is excluded from all
formal metrics.  Final mode refuses to start without its exact version-matched
manifest:

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/path/to/libero-config \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/libero-env/bin/python experiments/vla/run_libero_closedloop.py \
  --protocol experiments/vla/libero_protocol.json \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --preflight-manifest /path/to/runs/libero-preflight-state49/manifest.json \
  --output-dir /path/to/runs/libero-spatial-final-100 \
  --mode final
```

The formal grid is all ten `libero_spatial` tasks at fixed initial states 0
through 9, with seed `41021 + initial_state_index`.  Each episode explicitly
sets and verifies its state and seed.  There are no retries or replacement
episodes.  The Sentinel hook is an observe-only pass-through logger, so this
experiment measures the official SmolVLA checkpoint in native LIBERO and does
not estimate intervention benefit, real-robot safety, or performance outside
this fixed 100-episode grid.

## Paired execution-horizon ablation

The frozen ablation keeps the official prediction chunk at 50 actions and
changes only `n_action_steps`, comparing execution horizons 50 and 10.  It
uses task 5 as the primary failure case and task 0 as a control, with fixed
states 10 through 19.  Each branch receives the same hard reset, seed, first
state, camera frames, and first ten predicted actions.  Run the excluded
task-0/state-48 preflight before the formal 20-pair grid:

```bash
PYTHONPATH=/path/to/frozen-native-runner \
/path/to/libero-env/bin/python experiments/vla/run_replan_ablation.py \
  --protocol experiments/vla/replan_protocol.json \
  --native-runner experiments/vla/run_libero_closedloop.py \
  --native-protocol experiments/vla/libero_protocol.json \
  --native-manifest /path/to/libero-spatial-final-100/manifest.json \
  --native-trace /path/to/libero-spatial-final-100/action_trace.jsonl \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --output-dir /path/to/libero-replan-preflight \
  --mode preflight

PYTHONPATH=/path/to/frozen-native-runner \
/path/to/libero-env/bin/python experiments/vla/run_replan_ablation.py \
  --protocol experiments/vla/replan_protocol.json \
  --native-runner experiments/vla/run_libero_closedloop.py \
  --native-protocol experiments/vla/libero_protocol.json \
  --native-manifest /path/to/libero-spatial-final-100/manifest.json \
  --native-trace /path/to/libero-spatial-final-100/action_trace.jsonl \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --preflight-manifest /path/to/libero-replan-preflight/manifest.json \
  --output-dir /path/to/libero-replan-final-40 \
  --mode formal
```

## Reset and MuJoCo-backend diagnostics

The reset audit was frozen in commit `c7bfa7d`; the policy backend comparison
was frozen in commit `eebfb7a`.  MuJoCo 3.3.7 was installed into a separate
target directory and selected only for a fresh legacy process.  The base
environment and its MuJoCo 3.8.1 installation were not modified:

```bash
/path/to/libero-env/bin/pip install \
  --target /path/to/reset-audit-mujoco-3.3.7 'mujoco==3.3.7'

MUJOCO_GL=egl LIBERO_CONFIG_PATH=/path/to/libero-config \
/path/to/libero-env/bin/python experiments/vla/audit_libero_reset.py \
  --mode preflight --version-label current \
  --protocol experiments/vla/reset_audit_protocol.json \
  --native-protocol experiments/vla/libero_protocol.json \
  --libero-config /path/to/libero-config \
  --output-dir /path/to/reset-audit-preflight/current

PYTHONPATH=/path/to/reset-audit-mujoco-3.3.7 \
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/path/to/libero-config \
/path/to/libero-env/bin/python experiments/vla/audit_libero_reset.py \
  --mode preflight --version-label legacy \
  --protocol experiments/vla/reset_audit_protocol.json \
  --native-protocol experiments/vla/libero_protocol.json \
  --libero-config /path/to/libero-config \
  --output-dir /path/to/reset-audit-preflight/legacy
```

For each backend, rerun that command with `--mode final`, a new output
directory, and both frozen receipts:

```text
--preflight-current /path/to/reset-audit-preflight/current/manifest.json
--preflight-legacy  /path/to/reset-audit-preflight/legacy/manifest.json
```

Then compare the two captures without loading the policy:

```bash
/path/to/libero-env/bin/python experiments/vla/audit_libero_reset.py \
  --mode compare \
  --protocol experiments/vla/reset_audit_protocol.json \
  --output-dir /path/to/reset-audit-final/comparison \
  --current-manifest /path/to/reset-audit-final/current/manifest.json \
  --legacy-manifest /path/to/reset-audit-final/legacy/manifest.json
```

The backend policy runner takes the same frozen native/model inputs plus all
three reset-audit results.  Run a disjoint preflight for each version, then
repeat with `--mode formal`, both `--preflight-current` and
`--preflight-legacy`, and fresh output directories.  The legacy invocation
must keep the isolated prefix first on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/reset-audit-mujoco-3.3.7 \
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/path/to/libero-config \
/path/to/libero-env/bin/python experiments/vla/run_backend_ablation.py \
  --mode preflight --version-label legacy \
  --protocol experiments/vla/backend_protocol.json \
  --native-runner experiments/vla/run_libero_closedloop.py \
  --native-protocol experiments/vla/libero_protocol.json \
  --native-manifest /path/to/libero-spatial-final-100/manifest.json \
  --native-trace /path/to/libero-spatial-final-100/action_trace.jsonl \
  --reset-runner experiments/vla/audit_libero_reset.py \
  --reset-protocol experiments/vla/reset_audit_protocol.json \
  --reset-current /path/to/reset-audit-final/current/manifest.json \
  --reset-legacy /path/to/reset-audit-final/legacy/manifest.json \
  --reset-comparison /path/to/reset-audit-final/comparison/comparison.json \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --libero-config /path/to/libero-config \
  --output-dir /path/to/backend-preflight/legacy
```

Use the same command without the `PYTHONPATH` prefix and with
`--version-label current` for the current backend.  After both formal
captures, generate the paired comparison without policy inference:

```bash
/path/to/libero-env/bin/python experiments/vla/run_backend_ablation.py \
  --mode compare \
  --protocol experiments/vla/backend_protocol.json \
  --output-dir /path/to/backend-final/comparison \
  --current-manifest /path/to/backend-final/current/manifest.json \
  --legacy-manifest /path/to/backend-final/legacy/manifest.json
```

## Full-suite MuJoCo 3.3.7 confirmation

The confirmation source, protocol, and evidence logger were frozen in commit
`9b1a48d`.  It keeps the official 50-action prediction chunk, executes ten
actions before replanning, and uses all ten tasks at fixed states 30 through
39.  Its task-0/state-44/seed-44020 preflight is excluded from formal metrics.

The command deliberately lists all fourteen bound inputs.  The runner checks
their exact protocol hashes before constructing the environment:

```bash
PYTHONPATH=/path/to/reset-audit-mujoco-3.3.7 \
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/libero-env/bin/python experiments/vla/run_libero_confirmation.py \
  --mode preflight \
  --protocol experiments/vla/confirmation_protocol.json \
  --native-runner /path/to/frozen/run_libero_closedloop.py \
  --native-protocol /path/to/frozen/libero_protocol.json \
  --native-manifest /path/to/libero-spatial-final-100/manifest.json \
  --native-trace /path/to/libero-spatial-final-100/action_trace.jsonl \
  --reset-comparison /path/to/reset-audit-final/comparison/comparison.json \
  --horizon-runner /path/to/frozen/run_replan_ablation.py \
  --horizon-protocol /path/to/frozen/replan_protocol.json \
  --horizon-manifest /path/to/libero-replan-final-40/manifest.json \
  --horizon-trace /path/to/libero-replan-final-40/action_trace.jsonl \
  --backend-runner /path/to/frozen/run_backend_ablation.py \
  --backend-protocol /path/to/frozen/backend_protocol.json \
  --backend-current /path/to/backend-final/current/manifest.json \
  --backend-legacy /path/to/backend-final/legacy/manifest.json \
  --backend-comparison /path/to/backend-final/comparison/comparison.json \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --backbone /path/to/models/SmolVLM2-500M-Video-Instruct \
  --libero-config /path/to/libero-config \
  --output-dir /path/to/libero-confirmation-preflight
```

Formal mode uses the identical command with `--mode formal`, a new output
directory, and this additional version-matched evidence gate:

```text
--preflight-manifest /path/to/libero-confirmation-preflight/manifest.json
```

Each episode retains the official reward, terminated/truncated flags, and a
nullable `is_success` value with its `info` or `final_info` provenance.  The
formal denominator is always 100: a crash is a failure, receives no retry,
and cannot be replaced by another state.

## Exact-action failure replay

`replay_libero_failure.py` reconstructs the original task-5/state-0 failure
from the 280 actions already stored in the native trace.  It does not load the
policy or add another benchmark episode.  The manifest binds the source
trace, initial state and camera hashes, every action hash, object poses, raw
RGB video, and a bilingual 20 Hz presentation video.

```bash
PYTHONPATH=/path/to/frozen-native-runner \
/path/to/libero-env/bin/python experiments/vla/replay_libero_failure.py \
  --native-protocol experiments/vla/libero_protocol.json \
  --native-manifest /path/to/libero-spatial-final-100/manifest.json \
  --native-trace /path/to/libero-spatial-final-100/action_trace.jsonl \
  --checkpoint /path/to/models/smolvla_libero-31d453f \
  --font /path/to/SourceHanSerifSC-Regular.otf \
  --output-dir /path/to/libero-task5-state0-exact-replay
```

## Result-derived figures

Figures only read retained manifests; they never load a policy or simulator.
The confirmation figure records its actual plotting runtime in its manifest
(Python 3.12.14, NumPy 2.5.3, Matplotlib 3.11.2).

```bash
python experiments/vla/plot_confirmation_results.py \
  --manifest /path/to/libero-confirmation-final/manifest.json \
  --output-dir /path/to/new-confirmation-figure
python experiments/vla/plot_backend_results.py \
  --backend-dir /path/to/backend-final \
  --runner experiments/vla/run_backend_ablation.py \
  --protocol experiments/vla/backend_protocol.json \
  --output-dir /path/to/new-backend-figure
```

The original release distributed `replay_backend_comparison.py` as a statically
reviewed reproduction utility. No executed paired-film capture was included
in the release; use the individual verified native-success and failure films.

The separate [backend replay companion](../../docs/research/2026-10-02/paper-v2/backend/BACKEND_REPLAY.md)
now retains an executed paired capture, its portable official asset binding,
runtime receipt and 1080p bilingual video. It does not change the formal grid.
