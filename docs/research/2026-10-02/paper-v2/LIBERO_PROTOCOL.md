# Frozen SmolVLA LIBERO-Spatial closed-loop protocol

## Claim under test

This experiment measures whether the official `lerobot/smolvla_libero`
checkpoint completes LIBERO-Spatial tasks when its postprocessed actions are
actually applied to the official simulated environment.  It does not evaluate
our SO100 overlay, does not map SO100 actions to UR5e, and does not establish a
real-robot or safety result.

The Sentinel hook is deliberately a shadow observer.  The runner records the
raw policy chunk, the selected normalized action, the official LeRobot
postprocessor output, and the exact array supplied to `env.step`.  It asserts
bit-for-bit equality between the latter two arrays.  Therefore this run can
show integration and traceability, but it cannot show benefit from an
intervention.
The completed manifest binds the full JSONL action trace and every reported
rollout video by SHA-256 and byte length.

At every chunk boundary the trace also stores the natural-language
instruction, the unnormalized 8-dimensional state values and hash, and hashes
of both simulator camera tensors.  Camera pixels are not duplicated in the
JSONL trace; actual RGB pixels are retained only in the predeclared final
video.

## Frozen evaluation grid

The final evaluation uses the official LeRobot `libero_spatial` suite, all ten
tasks, and the first ten official fixed initial states for each task.  Each
task is evaluated sequentially with one environment, hard resets, relative
control, 20 Hz control frequency, and the checkpoint's 50-action execution
horizon.  This produces 100 rollouts.  Before each single-episode `eval_one`
call, the runner explicitly sets and then verifies the underlying LIBERO
`init_state_id` against the frozen index 0 through 9.  Episode seed is
`41021 + init_state_id`.  This prevents Gymnasium's automatic reset from
silently advancing the next evaluated initial state.  It also verifies that
each task has enough stored states to avoid LIBERO's modulo lookup silently
reusing an earlier state.  No soft reset,
asynchronous environment, or parallel task execution is allowed.

The checkpoint is pinned to Hub commit
`31d453f7edd78c839a8bbc39744a292686daf0de`.  Its SmolVLM backbone is pinned to
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`.  LeRobot is pinned to 0.6.1 and
the LIBERO integration to `hf-libero==0.1.4`.  Local paths replace the two Hub
names only after their immutable revisions and file hashes have been checked;
this prevents network aliases from changing the run.
The action-producing contract binds the policy weights, both serialized
normalizer files, the complete 15-file local backbone snapshot tree, and
individual tokenizer and processor configuration files by SHA-256 before
environment construction.
The official 3D asset dataset is pinned separately to
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
The resolved runtime asset directory is explicitly labeled a mixed
`site-packages` tree rather than a pristine dataset snapshot.  The runner
verifies its 955 files, 188,829,695 bytes, and canonical tree digest
`f04f72afb7503afb071f9de7c6734b0eca147f87549df0a9080cc987afb17781`.
It separately verifies the BDDL and fixed-init-state file digest for each of
the ten LIBERO-Spatial tasks.

The official LIBERO adapter names its two images `image` and `image2`, while
the checkpoint names the corresponding inputs `camera1` and `camera2`.  The
frozen LeRobot `rename_map` performs only those two key renames.  The checkpoint
also declares `camera3`; SmolVLA's official image preparation accepts missing
declared image keys and processes the two present cameras.  The protocol does
not fabricate a third image.

The runtime contract is frozen as an 8-dimensional state, a 7-dimensional
action, and exactly those two simulator camera keys.  The runner checks this
contract at every policy-chunk boundary and before every `env.step`.  It also
checks the SHA-256 digests of both serialized normalizer files; each digest is
`b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9`.

## Endpoints and reporting

The primary endpoint is task success over all 100 fixed rollouts.  We report
the count, percentage, and two-sided 95% Wilson interval.  Per-task counts and
Wilson intervals are secondary descriptive results; ten trials per task are
too few for fine-grained ranking.  We also report wall time, raw chunk count,
and the maximum absolute difference between the official postprocessor output
and the action passed to `env.step`.

A crashed or missing final rollout is retained and counted as a failure.  The
runner writes the partial action trace and an episode-level failure receipt,
preserves every completed episode, and counts the failed episode and every
subsequent uncompleted episode in the frozen grid as failures without retry or
replacement.  A
compatibility failure before action execution is reported as a protocol-blocking
result rather than repaired after seeing task outcomes.  The single preflight
rollout is only for dependency, rendering, checkpoint, and runtime validation;
it is stored separately and excluded from every final metric.  The preflight
uses task 0, fixed initial-state 49, and seed 41020, so it is disjoint from the
final task-0 states 0 through 9 and final seed sequence beginning at 41021.
Final mode requires that preflight manifest and verifies its script digest,
protocol digest, completion status, exclusion flag, task, state, seed, and
completed episode count before constructing the formal evaluation.

## Video rule

The video is selected before the run: task 0, fixed initial-state 0, from the
final evaluation.  It is rendered from the actual RGB frames returned by the
official LIBERO simulator.  Success or failure does not affect selection.

## Resolved checkpoint metadata discrepancy

At the frozen Hub revision, `config.json` declares a 6-dimensional state and
three cameras, while the serialized normalizer contains 8-dimensional state
statistics and the official LIBERO adapter exposes two cameras plus an
8-dimensional end-effector/gripper state.  The disjoint preflight confirmed
that the official LeRobot path consumes the unmodified 8-dimensional state
using the checkpoint's serialized 8-dimensional normalizer statistics;
SmolVLA then pads the generic state projection internally.  The raw predicted
chunk was `[1, 50, 7]`, and the official postprocessor produced the 7 values
passed unchanged to LIBERO.  The runner does not slice state, synthesize a
third camera, edit weights, change gripper signs, or convert action units.

Primary references: the official
[LeRobot LIBERO guide](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)
and the official
[SmolVLA LIBERO checkpoint](https://huggingface.co/lerobot/smolvla_libero).
