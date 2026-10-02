# Native VLA execution gateway

The native gateway is the product integration path for an existing robot
policy and environment. It keeps the policy's native action schema instead of
coercing a seven-dimensional LIBERO request into Sentinel's numerical 3-D
`Plan` contract.

The first supported profile is the official `lerobot/smolvla_libero`
checkpoint driving the official LIBERO/Panda relative-control environment.
The request reaching the gateway is the exact seven-value output of the
official LeRobot postprocessor. Heavy LeRobot, PyTorch, NumPy, LIBERO and
MuJoCo imports remain in the experiment runner; `sentinel_evc.native_gateway`
uses the standard library and the package's existing event/evidence code.

## What active mode enforces

For each native request, active mode:

1. receives actual feedback returned by `env.reset` or the preceding
   `env.step`;
2. checks the declared 7-D schema, finite values and component envelope;
3. binds the exact array-byte digest to the feedback digest, environment step,
   execution context, queue revision and revocation generation;
4. issues a short-lived one-use permit after policy inference and official
   postprocessing;
5. makes `NativeGateway.submit` the only path to the captured native
   `env.step` writer;
6. records submitted, accepted and observed cursors separately; and
7. signs the result, replay, action trace, configuration and optional video/3-D
   assets with the existing `EventLog` and `build_bundle` path.

The gateway actively blocks an exact-action substitution, permit reuse,
permit expiry, stale or replaced feedback, execution-context change and a
revoked generation before the native writer is called. Fault mode records
these as synthetic guard attempts and then submits the unchanged normal action
with a fresh permit. A blocked guard attempt is not a task success and does not
call `env.step`.

Revocation prevents new admissions from the old generation. A writer already
entered may finish its synchronous `env.step`; this gateway does not interrupt
in-flight simulation, cancel a physical controller, or prove a physical stop.

The active profile does **not** perform collision clearance, self-collision,
dynamics, controller-mapped joint-limit, contact-force, physical-stop or
WorldGuard validation. Result bundles therefore mark `physical_validation`
and `worldguard` as `unsupported`. The profile envelope is an environment
request contract, not a physical safety limit or functional-safety
certification.

## Run configuration

The runner consumes one JSON object and refuses an existing output directory:

```bash
python experiments/vla/run_sentinel_libero.py \
  --config /path/to/native-active.json \
  --output-dir /path/to/new-output
```

Before loading the policy, the runner verifies every checkpoint and backbone
file against the frozen public `libero_protocol.json`, verifies the complete
backbone and LIBERO asset trees, all ten task BDDL/init-state files, the
declared software stack, the fixed observation mapping and the explicitly
selected preregistered MuJoCo backend. Hashing an arbitrary local directory is
recordkeeping only and cannot establish that it is the official checkpoint;
an identity mismatch stops the run.

A minimal active qualification configuration is:

```json
{
  "mode": "active",
  "checkpoint": "/models/smolvla_libero",
  "backbone": "/models/smolvlm2_500m_video_instruct",
  "identity_protocol": "experiments/vla/libero_protocol.json",
  "expected_mujoco": "3.3.7",
  "profile": "experiments/vla/profiles/libero_native_qualified.json",
  "task_ids": [0],
  "initial_state_indices": [21],
  "seed_base": 44021,
  "execution_horizon": 10,
  "lease_ttl_ms": 250.0,
  "max_feedback_age_ms": 1500.0,
  "record_pose_stride": 10,
  "render_episode_keys": [[0, 21]],
  "render_episodes": 1
}
```

`mode` is one of `baseline`, `shadow`, `active`, or `fault`:

- `baseline` retains the same recorder but submits directly, for paired product
  overhead measurement.
- `shadow` authorizes and proves byte-identical postprocessor-to-environment
  passage but does not place the permit in the write path.
- `active` makes the one-use permit a prerequisite for the real native call.
- `fault` runs declared guard attempts and then continues the real episode.

The supported fault names are `action_replacement`, `lease_replay`,
`expired_permit`, `old_feedback`, `context_changed`, and
`revoked_generation`. Example:

```json
{
  "mode": "fault",
  "checkpoint": "/models/smolvla_libero",
  "backbone": "/models/smolvlm2_500m_video_instruct",
  "identity_protocol": "experiments/vla/libero_protocol.json",
  "expected_mujoco": "3.3.7",
  "profile": "experiments/vla/profiles/libero_native_qualified.json",
  "max_feedback_age_ms": 1500.0,
  "lease_ttl_ms": 250.0,
  "task_ids": [0],
  "initial_state_indices": [45],
  "faults": [
    "action_replacement",
    "lease_replay",
    "expired_permit",
    "old_feedback",
    "context_changed",
    "revoked_generation"
  ]
}
```

Before a formal paired run, freeze and hash the JSON configurations and use a
disjoint qualification episode. A recommended new grid is tasks 0–9 and fixed
initial-state indices 40–44, with the same task/state/seed cells for baseline
and active modes. Existing studies already use lower index ranges. Record the
actual postprocessed component minima/maxima in qualification before freezing
the profile; do not widen bounds after seeing formal outcomes. Deterministic
CUDA mode, policy reset and fixed seeds are enabled, but paired actions must
still be compared and any numerical difference reported rather than replaced
with recorded actions.

## Signed product artifacts

The portable ZIP has no outer run directory:

```text
bundle/events.jsonl
bundle/manifest.json
bundle/manifest.sig
bundle/result.json
bundle/replay.json
bundle/native_trace.jsonl
bundle/config.json
bundle/viewer-model.json                 # optional task-model index
bundle/viewer-model-taskN.json           # optional task-specific geometry
bundle/episode-taskNN-stateNN.mp4        # optional predeclared native video
anchors/demo.public
```

`result.json` uses schema `sentinel-native-vla-run-v1`. It contains the run
mode/status, declared profile and limitations, per-episode official LIBERO
outcomes, gateway/fault metrics, software/model identities and explicit
`physical_validation`/`worldguard` support status.

Latency fields name their measurement boundary. `gateway_authorize` covers
profile/context/feedback checks and permit creation. `gateway_submit_admission_before_writer`
ends immediately before the native writer is entered. `native_env_step` is the
simulator call. `active_submit_including_env_step` includes both admission and
the simulator, and therefore must not be reported as gateway overhead. The
bundle also retains policy-input-to-chunk inference time and feedback age at
authorization, including P50/P95/P99 summaries.

`replay.json` uses schema `sentinel-native-vla-replay-v1`. Every episode has a
stable `episode_id`, task and initial-state identifiers, an optional
task-specific `model_asset`, an optional `video_asset`, and frames. Every step
retains the final 7-D request, exact byte digest, decision, permit, feedback
digest, actual state summary, official outcome and submitted/accepted/observed
cursors. Body positions and row-major 3x3 rotations are arrays indexed by the
task model's `scene.bodies[].bodyIndex`; each frame also carries the matching
`modelId`.
Full poses are retained for every step of predeclared demonstration episodes;
other episodes use the configured pose stride while retaining actions, state
and outcomes at every step.

The generated `anchors/demo.public` is a demo trust source. Bundle integrity
does not prove that a sensor was truthful, that physical motion occurred, or
that a robot stopped within a distance.
