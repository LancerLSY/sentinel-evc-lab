# Native VLA product qualification and paired comparison

The evaluation separates native request authorization from physical validation.
The checkpoint is the frozen official `lerobot/smolvla_libero` Panda policy;
the separately published SO100 fine-tune is not substituted into this environment.

## Preregistered cells

| Lane | LIBERO spatial tasks | Fixed initial-state indices | Episodes | Purpose |
|---|---|---|---:|---|
| Qualification | 5 | 21 | 1 | Schema, component ranges, feedback age and runtime compatibility; excluded from comparison |
| Baseline | 0–9 | 40–44 | 50 | Official inference and recorder, direct environment submission |
| Active | 0–9 | 40–44 | 50 | The same inference/configuration with one-use native permits |
| Fault | 0–9 | 45 | 10 | Six declared invalid-authorization attempts per episode; then normal execution |

Use `seed_base=44040` in the paired lanes, with each state seed equal to
`seed_base + state_index - 40`. Fault uses `seed_base=44045`. Model state is
reset for every episode. Both paired lanes perform real inference; recorded
baseline actions cannot replace active inference. All tasks have 50 available
fixed states. Earlier studies used indices 0–39; the qualification cell is not
a formal sample.

The environment uses MuJoCo **3.3.7**, 20 Hz simulation control, relative
7-D requests, 360×360 cameras, horizon **10** out of the model's 50-action
prediction, and four CPU threads. Runtime hardware is measured, rather than
renamed. The intended current device is **RTX 4090 D (24 GB)**.

The initial request envelope is seven components in `[-1,1]`; it is a software
request profile, not a physical protection limit. Qualification must record
actual shape, dtype, component minima/maxima and freshness timing before the
formal profile is frozen. A changed profile or timing budget requires a new
qualification receipt before formal execution. It cannot be widened after
seeing formal outcomes. Configuration and source digests are retained.

## Fault attempts

Each fault episode attempts `action_replacement`, `lease_replay`,
`expired_permit`, `old_feedback`, `context_changed`, and `revoked_generation`.
Each attempted request traverses the native gateway with an instrumented
forbidden-writer callback. A passing guard must emit its exact expected denial
code and call that callback zero times. The rejected candidate is not sent to
the physical simulator. These are synthetic authorization failures, not
measured collision hazards. Task success and the 60 guard attempts use separate
denominators. A new valid permit allows the unchanged ordinary policy action
to continue.

## Outputs and decision criteria

- Retain every declared episode, including unsuccessful tasks and crashes.
- Compare final-request bytes, trajectory lengths and official success on
  matched task/state/seed cells. Report every divergence.
- Report per-task and aggregate task success. A descriptive binomial Wilson
  interval does not establish performance on other tasks or hardware.
- Measure authorization plus admission overhead before the environment writer;
  report inference, environment step, rendering and total run time separately.
- Check submitted → accepted → observed linkage and signed artifact integrity.
- Bind and authenticate the official model, processor, asset, task and runtime
  identities using the frozen identity protocol; missing or mismatched required
  files terminate qualification.

Predeclared 3-D/video examples are **task 0, task 4 and task 5 / state 40**.
Every step in those examples retains real compiled-model body poses. Other
episodes keep a pose stride of 10 and full actions/state summaries. Missing
task geometry falls back to the action timeline. Browser replay shows stored
poses; it does not run new dynamics or interpolate invented motion.

Panda collision/dynamics/physical-stop verification and Panda WorldGuard
prediction remain outside this native profile. Existing UR5e and tray results
retain their own support conditions. Successful request authorization must not
be presented as completion of those physical checks.
