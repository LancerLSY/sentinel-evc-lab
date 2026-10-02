# Native raw-request compatibility: paired engineering evaluation

This protocol supersedes the bounded-request configuration for a separate
evaluation. The original v2 failures remain reported under their original
profile; their outcomes are not recomputed as successful runs.

## Reason for the new contract

The signed v2 baseline contains 5,659 unchanged official requests. Forty requests
in nine of its 50 task/state cells exceeded the declared motion envelope
`[-1,1]`. Active mode stopped on task 0/state 41 at a raw translation value
`-1.0298006534576416`; the fault lane later stopped at `-1.0027008056640625`.
These are compatibility failures of the raw-request envelope.

The actual pinned Panda OSC controller clips its six inputs to `[-1,1]`, then
maps translation to `[-0.05,0.05]` and rotation to `[-0.5,0.5]`. Its gripper
uses the request sign to advance a bounded internal command. This mapping
occurs inside the upstream environment, after the `env.step` API. See the
[controller receipt](controller-contract-receipt.json),
[gripper receipt](gripper-contract-receipt.json), and upstream
[Controller.scale_action](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/controllers/base_controller.py).
Gripper example calls in the receipt are sequential on a stateful instance.
These are mapping provenance, not collision or dynamic validation.

The new [raw float32 profile](../../../../experiments/vla/profiles/libero_native_raw_float32.json)
accepts exactly shape `(1,7)`, dtype `float32`, and finite values in the complete
IEEE float32 representation domain. The domain is fixed by the representation;
it is not an envelope fitted to observed maxima. Sentinel preserves the raw
request bytes and does not move clipping into its write path. The native
controller's mapping and physical checks remain separate from request integrity.
This contract depends on the named pinned environment and must not be reused
as a raw-input contract for an arbitrary robot driver.

## Cells fixed before execution

| Lane | Task IDs | Fixed state indices | Episodes | Role |
|---|---|---|---:|---|
| Qualification | 0 | 41 | 1 | previously observed envelope failure; excluded |
| Baseline | 0–9 | 46–49 | 40 | direct-submission comparator, disjoint from the native-v2 formal grid |
| Active | 0–9 | 46–49 | 40 | independently inferred, one-use request authorization |
| Fault | 0–9 | 45 | 10 | repeatable software authorization faults; separate from task-performance claims |

The paired grid is disjoint from native-v2's formal states 40–44. Four cells
appear in earlier diagnostic protocols: task0/state46 (backend preflight),
task5/state47 (reset audit), task0/state48 (replan preflight), and task0/state49
(official preflight). The 40-cell grid is therefore an engineering comparison,
not a wholly untouched holdout. The remaining 36 cells may be identified
separately; they are not a new task-distribution benchmark. Both lanes use `seed_base=44046`, with seed
`44046 + state_index - 46`; fault uses `44045`. Both perform actual inference.
No recorded action sequence replaces an active policy run.

MuJoCo **3.3.7**, official checkpoint/processors/assets, 20 Hz simulation control,
360×360 cameras, four CPU threads, execute **10** of each **50** predicted
actions, feedback age **1500 ms**, and one-use permit **250 ms** remain fixed.
Hardware is **RTX 4090 D (24 GB)**. Qualification and source/configuration
identities must be retained before formal execution. No bounds or timing budget
may be changed after these paired outcomes are observed.

Predeclared scene/video records are **task 0, 4 and 5 / state 46**, whether they
succeed or fail. Original recorder videos play independently and are not
asserted to match the stored pose clock. Companion 20 Hz visualizations must
use the retained requests/initial states and verify their source trajectory;
they add no benchmark episodes.

## Endpoints and interpretation

Report every planned cell, success/failure/crash, per-task counts and descriptive
Wilson intervals. Compare paired action bytes, trajectory lengths and outcomes,
retaining any divergence. Report actual submitted/accepted/observed counts.
Six invalid-authorization attempts per fault episode have a separate denominator:
action replacement, replay, expiry, old feedback, changed context, and revoked
generation. A denial passes only when its expected reason is recorded and the
instrumented forbidden writer is called zero times.

Authorization plus admission cost excludes model inference, environment stepping
and rendering; these boundaries are reported separately. Request authorization
does not prove Panda collision/dynamics/WorldGuard validation or physical stop.
No competitor superiority is inferred from this self-comparison.
