# Mechanism implementation matrix — product foundation

This matrix separates the numerical/contact core from the native VLA request
gateway. Repository mechanism IDs M01–M12 are distinct from the original v4
requirement IDs. See [design_alignment.md](design_alignment.md) for the F01–F12
trace and remaining integrations.

The [2026-10-02 GPU experiments](gpu_training_results.md) add actual trained
numerical/visual/object predictors and real recorded data. The table describes
the product runtime, which keeps those experimental model profiles separate.

## Native Panda integration

The [native gateway](native_vla_gateway.md) adds a separate seven-dimensional
LIBERO/Panda request contract. Its authorization must not be interpreted as
the physical checks in the numerical table below.

| Product operation | Implemented native behavior | Support boundary |
|---|---|---|
| Policy to environment | official checkpoint/processors, raw chunk, selected action, postprocessed ndarray and actual `env.step` identity retained | pinned SmolVLA/LIBERO runtime |
| Request authorization | exact dtype/shape/bytes bound to feedback, task context, queue revision, step and revocation generation; expiring one-use permit | in-process request integrity, not collision/dynamics validation |
| Dispatch and observation | gateway-owned action copy; state and timing rechecked at writer entry; submitted/accepted/observed cursors use actual transitions | trusted writer acknowledges immediately before dispatch; an entered synchronous step may finish after revoke |
| Launch qualification | fixed-input camera/state/cursor/final-action comparison, observed wiring hints, signed CPU reproducer and guarded native writer | finite probes; trusted loaded identity; concurrent hot-swap needs caller lock |
| Portable evidence | independent public-key selection, bounded staged ZIP import, fresh verification and signed export | demo trust source, no customer PKI or sensor authenticity |
| App replay | exact task-model binding, real visual meshes and stored body poses, request/decision/cursor timeline | no new browser dynamics or interpolated motion |

The [raw-request paired protocol](research/2026-10-03/native-product/RAW_REQUEST_PROTOCOL.md)
separates normal task outcomes from six synthetic invalid-authorization attempts.

## Numerical product mechanisms

The numeric CLI/App lifecycle now uses one exact-scene v2 root certificate and
certificates for its actual remaining suffixes. The fixed seed7 comparison
preserved all 40 observed actions and reused 220 segments with no suffix full
check. Stop/resume was independently reviewed. This integration does not connect
the separate UR5e checker to the native VLA gateway.
[Current audit and retained evidence](research/2026-10-03/v2/AUDIT_RESULTS.zh-CN.md).

| Mechanism | Implemented behavior | Remaining experiment |
|---|---|---|
| M01 final action binding | immutable descriptor/Plan digest; finite 3D position profile | upstream VLA action adapters and normalization versions |
| M02 full physical validation | exact segment/sphere and box geometry; model cannot override rejection | link geometry, controller feasibility and recoverable robot terminal sets |
| M03 incremental certificate | parent/child/scene/time/event bindings; cumulative margins; full fallback; paired local total-cost measurement | realistic transform distribution and complex geometry; current simple-profile delta path is slower |
| M04 lease | bounded prefix, HMAC, one-use, full Context, expiry and prediction registry | process/key isolation and device trust |
| M05 single writer | only Executor writes controller commands | real driver capability enforcement |
| M06 commit/runtime state | fresh actual feedback before dispatch; start/tracking tube; context checks | sensors, synchronization and measured feedback delays |
| M07 revoke/recover | invalidates unused permits; cancel confirmed + drain + fresh feedback + current epoch + approval | device cancel and physical braking characterization |
| M08 final-candidate prediction | all four final Plans evaluated; deterministic earliest allowed; strict exact suffix; separate GPU GRU/visual models trained | import and bind trained model/action/normalization/calibration profiles to this runtime |
| M09 calibrated consequence | root-max rank, independent root groups, model/rule digest; unsupported action family unknown | large calibration/test sets and distribution shift studies |
| M10 local correction | bounded fixed-end geometric repair, original full recheck; disabled by default | v4 SciPy SQP and complete post-repair prediction pipeline |
| M11 gripper release | numerical open requires support assertion + allowed phase; actual simulated gripper feedback | hardware support sensing, clamp/force/slip trials |
| M12 evidence/replay | sticky gap gate; every run asset signed; independent verifier; persisted data replay | external anchors/customer PKI/forensic custody |
| Workbench | create/import scenario, four-candidate outcomes, actual load chart, event filters, step replay, stop/resume, ZIP | multi-user auth, remote deployment, device/VLA management |

The numerical plant uses `v_next=v+dt*(-a-k*r-d*v-beta*r³)` and `r_next=r+dt*v_next`. Observable history contains eight `(r,v,a)` samples. Hidden simulator parameters are confined to label/evaluator channels. Absolute position Plans convert to acceleration using successive discrete velocities and initial zero future tool velocity. Four movement durations share 40 observation intervals at dt=0.05.

The bundled model/calibration covers the fixed 0.35m four-duration family. Other displacement, dt, horizon, event or action changes produce `MODEL_UNKNOWN` and empty envelopes. Position translation preserves the action sequence. A learned forecast can be reused only by exact suffix comparison, retaining root identity and envelope offsets. Runtime observed r must remain inside its corresponding root envelope; stale predictions stop permission preparation.

Host wall time and numerical command time are separate. Numeric time advances only for observed controller commands. Holding during operator pause freezes this numerical clock; it does not model continuous physical load motion during a real stop. The simulated accepted tail is drained and recorded before recovery. Real stop dynamics remain an experiment.

Model scales in the basic ridge baseline are fitted on training roots, without test data. Dev roots are kept disjoint but this increment does not run hyperparameter selection or reproduce the v4 development-scale/GRU protocol. Independent root-max calibration remains explicit. Results are small deterministic evaluations, not safety guarantees.

## Added physical and SSH profile

See [physics_ssh_design.md](physics_ssh_design.md) for the full design review, scoped implementation and unresolved scientific gates. `physics` adds a real MuJoCo XYZ open-tray/free-payload contact fixture with complete-state pairing, fixed 2ms gateway cadence, full compiled-model binding, continuing stop dynamics, and a signed aggregate index. `remote-physics` launches the same trusted local runtime through strict OpenSSH and verifies the returned evidence against its submitted source and authenticated transfer receipt.

This profile does not inherit the numeric predictor/calibration, incremental geometry certificates or workbench pause semantics. Geometry-only selection can slip; fastest-branch terminal-pose convergence currently fails. This increment is an experimental contact runner, not completion of v4 robot/VLA/GRU integration.
