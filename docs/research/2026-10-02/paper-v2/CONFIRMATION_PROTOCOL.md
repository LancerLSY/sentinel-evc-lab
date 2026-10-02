# Prospective full-suite LIBERO confirmation

## Question

The native 100-episode run measured the official `lerobot/smolvla_libero`
checkpoint with a 50-action execution horizon under MuJoCo 3.8.1.  Subsequent
diagnostics separately studied a 10-action execution horizon and a
benchmark-compatible MuJoCo 3.3.7 backend on two tasks.  This prospective
lane asks for whole-suite performance under one declared configuration:
MuJoCo 3.3.7, 50-action prediction chunks, and a 10-action execution horizon.

This is an independent post-diagnostic confirmation.  It is not a paired
comparison against the original 58/100 result because the fixed states,
backend, and execution horizon differ.  It does not measure Sentinel
intervention efficacy and does not evaluate the repository's SO100 overlay.

## Frozen grid

- **Policy:** immutable official `lerobot/smolvla_libero` checkpoint and the
  same checkpoint normalizers, backbone, tokenizer, cameras, action units,
  processors, and LIBERO asset tree used by the native run.
- **Backend:** isolated MuJoCo 3.3.7 in a fresh process, with LeRobot 0.6.1,
  hf-libero 0.1.4, and robosuite 1.4.0 unchanged.
- **Treatment configuration:** the policy still predicts `[1, 50, 7]` action
  chunks.  `n_action_steps=10` refreshes the observation and prediction after
  every ten executed actions.
- **Formal grid:** all ten LIBERO-Spatial tasks at fixed initial-state indices
  30 through 39, using seed `44021 + (state_index - 30)`.  This is exactly 100
  episodes with no retries or replacement cells.
- **Preflight:** task 0, state 44, seed 44020.  It is excluded from every
  formal result and must use the exact final source and protocol.

The grid is disjoint from the native states 0–9, reset/horizon diagnostic
states 10–29, and prior compatibility states 46–49.

## Evidence gates

Before preflight, the protocol must replace all pending backend-result hashes
and enter `frozen_before_formal_run`.  It binds:

1. the original native runner, protocol, 100-episode manifest, and full trace;
2. the completed reset comparison;
3. the 40-episode execution-horizon runner, protocol, manifest, and trace;
4. the backend runner and protocol; and
5. both completed backend-version manifests and their paired comparison.

The runner independently rechecks checkpoint files, normalizers, full
backbone tree, the 955-file mixed LIBERO asset tree, all task BDDL/init-state
files, and effective software versions.  Formal mode consumes a task-0/state-44
preflight receipt whose source and protocol hashes must match exactly.

## Failure and reporting rules

Every episode explicitly sets and verifies the requested fixed-state index and
seed.  The full trace retains raw 50-action predictions, selected normalized
actions, official postprocessed actions, and the exact actions supplied to
`env.step`.  After each call it also retains the reward, terminated and
truncated flags, and `is_success` value returned by the official environment,
with task, fixed-state, and action-step identifiers.  `is_success` is nullable:
the trace separately records whether the key was present and whether it came
from `info` or `final_info`, rather than treating missing evidence as failure.
The result wrapper returns the exact original result object unchanged.  The
observe-only recorder never changes an action or outcome.

An episode exception counts as failure without retry.  If a fatal error stops
the process-level loop, every remaining declared task/state cell is added to
the manifest as a failure with null reset evidence.  The formal denominator
therefore remains 100.  Report the overall success rate and Wilson interval,
every task's ten-cell result and Wilson interval, crashes, action-chunk count,
environment steps, and runtime.  Do not extrapolate full-suite performance
from either earlier two-task diagnostic.

No preflight or formal command is authorized while the protocol contains a
`PENDING_` binding.  After backend evidence is complete, the source and
protocol require independent review and a Git freeze before the excluded
preflight.  The formal 100-episode run requires a separate explicit GO after
that preflight is reviewed.
