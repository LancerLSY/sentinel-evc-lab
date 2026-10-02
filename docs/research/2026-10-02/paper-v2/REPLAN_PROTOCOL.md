# Frozen paired replanning-horizon ablation

## Question

The native 100-episode evaluation succeeded in 58 cases, while task 5 failed
in all ten fixed states.  This ablation asks whether refreshing observations
and predicting a new action chunk every 10 executed actions improves success
relative to executing 50 actions from each prediction.  Both branches use the
same official `lerobot/smolvla_libero` weights and still predict a 50-action
chunk.  Only the number of queued actions executed before replanning changes.

In LeRobot 0.6.1, `SmolVLA.select_action` predicts a new chunk only when its
action deque is empty, and enqueues the first `config.n_action_steps` entries.
Thus runtime `n_action_steps=10` gives a new 50-action prediction every ten
environment steps; it does not change `chunk_size`, weights, normalizers,
cameras, action units, or postprocessing.

## Paired design

The primary task is task 5, “pick up the black bowl on the ramekin and place it
on the plate.”  Task 0 is a successful control.  Each task uses fresh fixed
states 10 through 19, disjoint from the native formal states 0 through 9 and
the native preflight state 49.  Each task/state root is evaluated once under
each branch, for 20 paired roots and 40 formal episodes.

Both branches hard-reset the official environment to the same fixed state and
seed.  The random generators and policy queue are reset before each branch.
Branch order alternates by pair ordinal.  Before accepting a pair, the runner
requires identical first-chunk state and camera hashes, a raw prediction shape
of `[1, 50, 7]`, and bitwise equality of the first ten predicted actions.  This
shows that the intended treatment is the later replanning boundary rather than
a different initial observation or first prediction.

The formal seed for state `s` is `42021 + (s - 10)`.  The excluded preflight is
task 0, state 48, seed 42020, under both branches.  Formal execution remains
blocked until the exact source, protocol, and preflight receipt are committed.

## Endpoints

The primary endpoint is task 5 paired success delta, execute-10 minus
execute-50.  We report both raw paired outcomes, per-branch Wilson intervals,
and a deterministic paired bootstrap interval with 20,000 resamples.  Task 0,
raw chunks per step, and wall time are secondary.  With ten pairs per task,
all intervals are descriptive.

A crash counts as failure without retry or replacement.  The trace retains
raw prediction chunks, selected actions, official postprocessed actions, and
the exact actions passed to `env.step`.  This is a native policy ablation; the
Sentinel observer never changes an action.

## Bound evidence

The runner binds the frozen native source (`9d5103e...`), protocol
(`431ea1...`), formal manifest (`b5d8e9...`), and formal trace (`346097...`).
It rechecks the same checkpoint, normalizers, full backbone tree, mixed
955-file asset tree, task BDDL/init-state files, and software versions before
running.

## Post-hoc task-5/state-0 replay plan

A separate diagnostic may replay the already recorded 280 postprocessed
actions from native task 5/state 0 into the same official state and seed while
rendering RGB and object/robot state.  It must verify the recorded initial
hashes, action hashes, zero reward, and failed terminal outcome.  This replay
does not call the policy, change the benchmark result, select a new case, or
claim a causal failure mechanism.  It should visualize where the recorded
trajectory went without guessing from the success bit alone.

