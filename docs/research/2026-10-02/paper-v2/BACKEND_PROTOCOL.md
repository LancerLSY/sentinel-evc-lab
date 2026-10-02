# MuJoCo backend compatibility protocol

## Question

The no-policy reset audit found a task-conditional difference between MuJoCo 3.8.1 and 3.3.7 for the LIBERO task where the black bowl begins on the ramekin. This prospective study asks the next direct question: with the official SmolVLA weights and all benchmark inputs held fixed, does changing the complete MuJoCo version change actual closed-loop task success?

This is a benchmark-version compatibility comparison. It does not designate MuJoCo 3.3.7 as physically more accurate or MuJoCo 3.8.1 as buggy. MuJoCo's official 3.4.0 changelog documents a box-box distance fix, and LeRobot issue 4390 discusses legacy LIBERO contact behavior. They motivate the treatment but do not predetermine the result. Because the treatment changes the complete MuJoCo package, reset-mediated causation remains a hypothesis unless contact behavior is isolated separately.

## Frozen design

- **Primary and control:** task 5, where the target bowl begins on the ramekin, is primary. Task 0 is the same-object-family control.
- **Versions:** MuJoCo 3.8.1 in the current environment and MuJoCo 3.3.7 from the existing isolated `--target` prefix. Each version runs in a fresh process, sequentially.
- **Formal grid:** states 20–29 for both tasks and both versions, giving 20 paired state/task cases and 40 rollouts. Seeds are `43021 + (state_index - 20)`.
- **Policy:** the official frozen SmolVLA LIBERO checkpoint and SmolVLM backbone. Prediction chunks remain 50 actions; execution horizon is fixed at 10 actions. The policy is reset before every episode.
- **Benchmark inputs:** task, BDDL, official init-state file and index, seed, assets, checkpoint, weights, processors, cameras, control mode, and horizon are matched. Official init states are neither patched nor regenerated.
- **Preflight:** task 0/state 46/seed 43020 runs once per version and is excluded. Both exact preflight manifests must be hash-bound in each formal manifest.
- **Failures:** a crash counts as failure and receives no retry or replacement. State and seed remain mandatory when reset was observed. A crash before reset retains null actual state/seed and is reported as missing reset evidence rather than reading stale environment values.

The model and backbone hashes, complete asset tree, every LIBERO BDDL/init-state hash, native 100-episode evidence, reset-audit sources and results, runner, protocol, full trace, and preflight receipts are retained or hash-bound.

## Measurements

The primary endpoint is task-5 paired success difference, MuJoCo 3.3.7 minus 3.8.1. Secondary endpoints are task-0 paired difference, per-version Wilson intervals, a 10,000-sample paired bootstrap interval, discordant pairs, crashes, and episode runtime.

The full policy/action trace is retained. For each episode, the manifest also records the first reset-state hash, both first camera hashes, the first 50-action raw chunk hash, and the first predicted normalized action. These inputs and actions may differ because simulator version is the treatment; differences are reported and are not protocol failures.

## Interpretation

The result estimates a complete-version effect within this frozen benchmark stack. It does not isolate one MuJoCo contact change, prove that a reset difference caused a success difference, establish physical accuracy, or support real-robot and safety claims.

Formal execution requires a Git-frozen runner/protocol, successful version-specific preflights, and explicit authorization after the separate horizon study has completed so GPU inference runs do not overlap.
