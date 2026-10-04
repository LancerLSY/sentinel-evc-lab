# Reproducing the A5 and A6 repair runs

This guide reproduces the repaired A5 baseline and the prospective A6 safe-prefix policy. It uses the official `lerobot/smolvla_libero` checkpoint and LIBERO's native Panda controller. It does not use the SO100 fine-tuned overlay.

## Immutable inputs

Use the archived sources and config for each run rather than the current working tree:

| Artifact | SHA-256 |
|---|---|
| `A5-run.tar.gz` | `33f96e76d69e6cb861ed4e1b7389e1c1f2ea562875933b6b0d5a57406ef465ea` |
| `A6-run.tar.gz` | `83c5509c947e0e2455e625cab70b152a0088ab956b8a353ad4314909a375892e` |
| A5 config | `2704b7a048c6b17bb9e5f1338d72f0f05199ee3ce8f1b81fcbbcc21f74b61a56` |
| A6 config | `53d09ea749705e0297b9f512eff6e5776da405e6e5cc94365e7648d6e35a938f` |

The archives contain the exact source snapshot, config, manifest, JSONL logs, per-episode arrays, and incremental outputs retained from each run. The manifest is authoritative for package versions, checkpoint hashes, source hashes, resolved allowed-contact pairs, task names, and artifact hashes. It also records strict checkpoint loading: 500 checkpoint keys, 500 loaded model keys, zero missing keys, and zero unexpected keys.

Publication paths:

- [`evidence/A5-run.tar.gz`](evidence/A5-run.tar.gz)
- [`evidence/A6-run.tar.gz`](evidence/A6-run.tar.gz)
- [`receipts/A5-evidence-validation.json`](receipts/A5-evidence-validation.json)
- [`receipts/A6-evidence-validation.json`](receipts/A6-evidence-validation.json)
- [`receipts/rollout-state-forensic-summary.json`](receipts/rollout-state-forensic-summary.json)

## Protocol

Both runs use the same six formal roots:

- task IDs `0`, `4`, and `5` from `libero_spatial`;
- initial-state indices `40` and `41`;
- seeds `2026100400` and `2026100401`, respectively;
- branches `parent_only`, `full_final`, and `delta_evc`;
- a maximum of 280 native environment writes per episode.

Each branch receives a fresh reset to the declared root and then runs its own policy/environment loop. Do not reuse observations, forecasts, certificates, actions, or terminal state between branches. Report six trials per branch, not 18 pooled samples. The task order differs between the archived A5 and A6 configs but the root set is identical.

Both configs also declare a separate one-step preflight at task 0, state 49, seed `2026100399`. Exclude those three preflight episodes from formal outcome counts.

## Environment preparation

Start from the package versions and source revision recorded in the selected archive manifest. Place the complete `lerobot/smolvla_libero` checkpoint and its pinned backbone files at the paths referenced by the config, or update those two paths before freezing a new config. A changed path or deadline produces a new config hash and must be reported as a new reproduction run.

Set the repository on `PYTHONPATH` and force offline checkpoint loading after the files are present:

```bash
export PYTHONPATH=/path/to/repo/src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
```

The runner fails closed if strict checkpoint-key validation, native-state capture, forecast restoration, certificate binding, or exact gateway-write binding fails.

## Run A5

A5 is the repaired baseline. Its config enables full native `MjData` restoration, safe traversal from the LeRobot environment root, restoration through the captured top-level mutable references, and nine-coordinate Panda tracking.

```bash
python experiments/vla/run_unified_libero.py \
  --config experiments/vla/config_unified_libero_A5_20261004.json
```

Relevant declarations:

- [`config_unified_libero_A5_20261004.json`](../../../../experiments/vla/config_unified_libero_A5_20261004.json)
- [`unified_libero_amendment_A5.md`](../../../../experiments/vla/unified_libero_amendment_A5.md)

Expected formal outcome from the archived run: `2/6` successes for each branch, with zero crashes, zero tracking-drift failures, and zero observed forbidden contacts.

## Run A6

A6 retains A5's state-restoration and nine-coordinate checks. For `full_final` and `delta_evc` only, it enables the declared unchanged-prefix sequence `5, 2, 1`. A prefix must receive its own full certificate over the exact shortened bytes and retained substep trajectory before dispatch. Failure to certify every candidate terminates without a write. `parent_only` remains unchanged.

```bash
python experiments/vla/run_unified_libero.py \
  --config experiments/vla/config_unified_libero_A6_20261004.json
```

Relevant declarations:

- [`config_unified_libero_A6_20261004.json`](../../../../experiments/vla/config_unified_libero_A6_20261004.json)
- [`unified_libero_amendment_A6.md`](../../../../experiments/vla/unified_libero_amendment_A6.md)
- [`safe_prefix_recovery.py`](../../../../experiments/vla/safe_prefix_recovery.py)

Expected formal outcome from the archived run: `2/6` for `parent_only`, `3/6` for `full_final`, and `3/6` for `delta_evc`, with zero crashes, zero tracking-drift failures, and zero observed forbidden contacts. The additional success is task 5/state 40 in each certified final-action branch.

## Validate retained evidence

Before summarizing outcomes:

1. Verify the archive SHA-256 listed above.
2. Verify the internal artifact hashes in `manifest.json`.
3. Confirm there are 21 per-episode rows: three excluded preflights and 18 formal episodes.
4. Confirm every formal root appears exactly once per branch.
5. Confirm every successful or denied episode has its matching step log and candidate-array archive.
6. Confirm actual gateway write hashes match the independently computed exact request hashes.
7. Confirm all executed substep trajectories remain within the declared hinge and slide tracking tolerances.
8. Count unwanted contacts with the archived exact allowed-contact policy; do not treat allowed target-bowl contacts as forbidden.

The retained validation receipts report:

| Check | A5 | A6 |
|---|---:|---:|
| Validation complete | yes | yes |
| Per-episode/event/NPZ records | 21 each | 21 each |
| Actual native writes | 1,116 | 1,130 |
| Verified actual-qpos records | 744 | 758 |
| Certificates | 350 | 366 |
| Denial records | 12 | 10 |

These counts include the declared preflight records where applicable; formal success denominators exclude them.

## Restore-path forensic check

The excluded forensic diagnostic replays four retained A3 native-action chunks on task 5 at states 40 and 41, restoring after each forecast and then executing the same actions. Its pass condition is exact equality of the integration/native/Python/RNG fingerprints plus equality of forecast and actual qpos and raw contact signatures at every retained physics substep.

The archived receipt records maximum qpos error `0.0`, no differing contact substep, and exact restored fingerprints for the tested chunks. It also records the remaining limitation: nested alias-group identity is not claimed universally. This diagnostic corrects the historical A3 record, whose earlier walker captured zero Python objects.

## Interpretation limits

- The continuous geometry result covers the nine Panda coordinates against a scene frozen at each chunk root. It does not continuously model carried-object or other non-arm scene motion.
- Native substep contact traces are observed simulator evidence. An optional A8 unwanted-contact veto is a separate mechanism and is not part of A5 or A6.
- The six-root grid is too small for a general success-rate estimate.
- Timing fields are retained for audit, but A5/A6 do not support a clean speed comparison because branch work and counterfactual diagnostics differ.
- These are MuJoCo/LIBERO results. They do not establish continuous physical safety, functional safety, or deployment readiness.

## A7 and A8 artifacts

A7 is complete: use its exact archive, config and receipt. A8 consists of the original deadline-stopped run and a separately frozen task-0 completion shard. Reproduce each source/config snapshot separately; do not silently combine their manifests or replace timed-out episodes with the excluded live follow-up.

- [A7 archive](evidence/A7-run.tar.gz) and [receipt](receipts/A7-evidence-validation.json)
- [A8 original archive](evidence/A8-run.tar.gz) and [partial receipt](receipts/A8-evidence-validation.json)
- [A8 task-0 completion archive](evidence/A8-completion-run.tar.gz) and [partial receipt](receipts/A8-completion-evidence-validation.json)
- [Excluded A8 live follow-up](evidence/A8-state41-live.tar.gz) and [receipt](receipts/excluded-live-A8-task05-state041-delta-evidence-validation.json)
- [Native direct control](evidence/native-direct-control.tar.gz) and [receipt](receipts/excluded-native-direct-F-evidence-validation.json)

The native direct control explicitly pins the installed selective backbone configuration/tokenizer tree and installed assets while retaining every checkpoint weight digest and all BDDL/init-state digests. It reloads the full pinned checkpoint with `strict=True`. Its identity variant is restricted to the excluded baseline mode; existing official modes keep the original identity protocol. No standalone backbone weights were downloaded. The exported `demo.public` key is for this demonstration, not customer PKI.

The control is unaggregated and excluded from formal statistics. Startup failures before any episode are retained in the control archive. File hashes for all final archives and presentation files are listed in [repair-artifacts.sha256](receipts/repair-artifacts.sha256).
