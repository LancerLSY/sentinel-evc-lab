# Product foundation contracts

This increment is a local single-user numeric workbench. Core DTOs live only in `contracts.py`; predictions live only in `prediction.py`. HTTP accepts a bounded Scenario, never a low-level motion permit or arbitrary driver call. The old v0.1 schemas and sample bundle remain supported.

## Core interface

- `Snapshot(obs_id, position, capture_mono_ns, valid=True, supported=None)` is an immutable newly sampled actual controller position. An unchanged executed cursor or position may have a new sampling time; repeated/old sampling timestamps cannot authorize new dispatch. Targets are not observations.
- `Context` retains robot/boot/epoch/scene/controller/task_phase/queue_rev/committed_prefix_hash. All fields are compared. Executor owns revocation generation, and Authority invalidates previously issued permits when that generation advances.
- `Executor.commit(lease, plan, snapshot, live_context, now_ns)` rejects a running prefix and rechecks state age, all context, generation and binding.
- `Executor.tick(now_ns, snapshot=None, live_context=None)` requires fresh Snapshot and live Context before each new dispatch; absent/stale inputs fail closed. Tick may advance already submitted controller work while idle/faulted without approving a new action.
- `Executor.try_recover(operator_approved, snapshot=None, live_context=None, now_ns=None)` requires confirmed cancel, drained driver prefix, a valid observation sampled after revoke, current local epoch, and explicit approval.
- SimController exposes read-only `read_feedback(now_ns)` with actual position, submitted/accepted/observed cursors and gripper state. Only its observed command list changes the numeric plant; no next target or merely submitted command changes observed state.
- A numerical `open` event requires `Snapshot.supported is True` and a supported task phase, and is dispatched by Executor only. It never establishes physical support sensing.
- Authority accepts registered prediction objects via `register_prediction(prediction)`; `prepare(..., prediction=None, require_prediction=False)` binds any required prediction to the final plan and its validity. Lease carries `prediction_hash` when supplied. Core snapshots `Prediction.plan_hash`, `hash`, `deadline_mono_ns`, and `allowed`, and denies preparation/dispatch if a registered object's fields change; numeric code defines its immutable object. Missing/unregistered/denied/expired required predictions deny preparation and dispatch. The registry remains a trusted local-process boundary.
- `EventLog(run_id, maxlen=100000, schema_version="v0.1")` preserves legacy event types; `product-v1` adds `PREDICTION`, `LEASE`, `CANCEL_REQUEST`, `CANCEL_ACCEPTED`. `CANCEL_ACK` carries confirmed semantics. Overflow emits a visible sticky LOG_GAP and denies further motion approvals in that run.

## Numeric interfaces

The selected numeric Plan receives one v2 root certificate. Each permission uses
the exact unexecuted suffix from the actual observed cursor, including after
stop/resume. The certificate adapter preserves its exact-plan digest, exact scene
digest, proof scope and parent/root lineage before Authority registration. Changed
inputs require dependency validation or full rechecking.

New `Scene.hash` values bind exact float64 content. `Scene.legacy_hash` retains the
rounded historical identity for reading older records. Existing signed bundles
are not rewritten. Third-party callers must supply `Context.scene_hash` to enable
scene-content comparison in the compatible Authority interface. The pipeline's
committed-prefix hash records positions. Individual plans and permits bind gripper
events.

`NumericHistory` contains only observable past (r, v, a) samples. Hidden plant parameters are evaluator-only. Fixed candidates use 0.6/0.9/1.2/1.6-second movement durations, 40 future intervals at dt=0.05, four final 3D position Plans from the same root. Candidate generation, physical checks and consequence predictions precede deterministic choice from the allowed subset. All four outcomes remain in saved results.

Numeric APIs expose `make_numeric_case(seed)`, `generate_candidates(displacement=0.35, start=(0,0,0), dt=0.05, horizon=40)`, prediction/model/calibration APIs and a lightweight `evaluate_candidates(...)` orchestration helper. The concrete API signatures live in the numeric modules. Predictions bind final plan, history, model, calibration, candidate rule, lower/upper consequence envelopes, expiry, root prediction hash and suffix offset. `slice_prediction` compares exact suffix positions/dt/descriptor/gripper events; modified actions or stale evidence cannot inherit a learned forecast.

The first trainable model is explicitly a standard-library low-dimensional residual baseline, not the v4 GRU. Calibration uses independent root groups and the finite-sample conformal rank; too few calibration roots produce no finite envelope, never a convenient finite substitute. Model/calibration/candidate mismatches fail closed. Experimental performance is reported only from actual artifacts.

## HTTP and storage

- JSON errors: `{ "error": { "code": "...", "message": "..." } }`; messages do not expose paths, secrets or tracebacks.
- `GET /api/session` returns startup CSRF token, profile and templates; browser uses `X-Sentinel-Token` for mutations. Host/Origin/Fetch-Site checks precede token delivery. No CORS or URL tokens.
- `GET/POST /api/runs`; `GET /api/runs/{id}`; `GET /api/runs/{id}/events`; `POST /api/runs/{id}/stop`; `POST /api/runs/{id}/resume` with `{ "operator_approved": true }`; `POST /api/runs/{id}/export`; `GET /api/runs/{id}/download`; `GET /api/experiments`.
- Scenario: schema_version product-v1; name; seed; displacement; risk_limit; prediction_mode physical/residual; optional strict scene. Numeric values are bounded and finite. Unknown/duplicate JSON fields and invalid UTF-8 are rejected. No file paths or user-selected run IDs in creation input.
- Run IDs are generated identifiers; one active run; status queued/preparing/running/stopping/stopped/completed/rejected/failed. Worker owns controller lifecycle; transport only requests stop/resume. Index is rebuildable from persisted assets. Restart marks interrupted sessions explicitly, never completed.
- Results contain all candidates, selected final digest, runtime cursors, observed load history, model/calibration IDs, reason codes, scope and actual wall time. Replay is saved data, not recomputed results.
- Signed manifests cover scenario, result, model, calibration, predictions/replay and event bytes. ZIP includes a clearly labelled demo public key; independent verification requires an explicitly chosen trust source and expected run ID. External anchors remain an experiment.

## Experimental status

The [managed unified loop](managed_closed_loop.md) adds `loop-configure`,
`loop-status`, `loop-run` and the App's live execution page. Operator-side CLI
configuration binds a source digest and a separate Python/configuration;
HTTP only starts that binding or requests stop. `GET /api/loop-runtime` reports
readiness. `GET/POST /api/loop-jobs`, `GET /api/loop-jobs/{id}` and its `live`,
`model`, `download` resources expose the lifecycle. `POST .../{id}/stop` accepts
only `{}`. Live data is provisional; terminal data and downloads reverify signed
post-exit assets. A receipt covers the raw gateway events without changing their
schema. Simulator process cleanup does not establish physical cancellation.

The native VLA request gateway has a separate [contract](native_vla_gateway.md).
It owns a copy of the submitted array and revalidates current feedback/context,
generation and timing at the writer's `entered()` acknowledgement. Writers are
trusted adapters and must acknowledge immediately before dispatch and propagate
denials. This contract does not grant physical-motion authority through HTTP.

Pending/blocked experiments have prerequisites, runnable entrypoints or explicit integration requirements, acceptance criteria and output expectations, but no metrics. Completed experiments require command, source/environment, data/model/calibration identifiers, acceptance output and artifact digests. Real VLA/vision/robot/GRU experiments are distinct from this numerical product.


Successful completion and signed rejection are published only after controller cleanup
and signed asset finalization. A restart or finalization failure can publish an
operational `failed` record without a complete bundle; it has verification false
(`UNFINALIZED_RECORD` or `EVIDENCE_FINALIZATION_FAILED`), cannot export and never
establishes successful execution or a confirmed physical stop.
Resume responds `preparing` so clients continue polling; a newer stop request is never
cleared by recovery. Finalization failure becomes `EVIDENCE_FINALIZATION_FAILED`,
invalidates completion confirmation and permits a later independent run. Terminal
reads and lists reverify finalized signed assets and compare all displayed
status/result/error/scope data; removing a verification cache field does not bypass
verification. Downloads regenerate exports from verified source assets, and exports
allow flat member names only. A shutdown timeout retains workspace ownership while
workers remain alive. A stop during preparation ends with no approved dispatch and
requires a new run. The supplied public key remains
a demo trust source. Product scenario/event schemas are envelopes; core constructors
and regressions enforce semantic cross-field conditions. Dedicated run/prediction/
experiment schemas are a subsequent contract-documentation increment.
