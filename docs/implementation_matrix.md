# Mechanism implementation matrix — product foundation

This is the first usable local numerical product increment. Design mechanisms are tracked individually; the historical v4 GRU/dataset/robot claims are not substituted with prototype results.

| Mechanism | Implemented behavior | Remaining experiment |
|---|---|---|
| F01 final action binding | immutable descriptor/Plan digest; finite 3D position profile | upstream VLA action adapters and normalization versions |
| F02 full physical validation | exact segment/sphere and box geometry; model cannot override rejection | link geometry, controller feasibility and recoverable robot terminal sets |
| F03 incremental certificate | parent/child/scene/time/event bindings; cumulative margins; full fallback | realistic transform distribution, total cost measurement |
| F04 lease | bounded prefix, HMAC, one-use, full Context, expiry and prediction registry | process/key isolation and device trust |
| F05 single writer | only Executor writes controller commands | real driver capability enforcement |
| F06 commit/runtime state | fresh actual feedback before dispatch; start/tracking tube; context checks | sensors, synchronization and measured feedback delays |
| F07 revoke/recover | invalidates unused permits; cancel confirmed + drain + fresh feedback + current epoch + approval | device cancel and physical braking characterization |
| F08 final-candidate prediction | all four final Plans evaluated; deterministic earliest allowed; strict exact suffix | v4 GRU weights/training, visual/action-conditional model |
| F09 calibrated consequence | root-max rank, independent root groups, model/rule digest; unsupported action family unknown | large calibration/test sets and distribution shift studies |
| F10 local correction | bounded fixed-end geometric repair, original full recheck; disabled by default | v4 SciPy SQP and complete post-repair prediction pipeline |
| F11 gripper release | numerical open requires support assertion + allowed phase; actual simulated gripper feedback | hardware support sensing, clamp/force/slip trials |
| F12 evidence/replay | sticky gap gate; every run asset signed; independent verifier; persisted data replay | external anchors/customer PKI/forensic custody |
| Workbench | create/import scenario, four-candidate outcomes, actual load chart, event filters, step replay, stop/resume, ZIP | multi-user auth, remote deployment, device/VLA management |

The numerical plant uses `v_next=v+dt*(-a-k*r-d*v-beta*r³)` and `r_next=r+dt*v_next`. Observable history contains eight `(r,v,a)` samples. Hidden simulator parameters are confined to label/evaluator channels. Absolute position Plans convert to acceleration using successive discrete velocities and initial zero future tool velocity. Four movement durations share 40 observation intervals at dt=0.05.

The bundled model/calibration covers the fixed 0.35m four-duration family. Other displacement, dt, horizon, event or action changes produce `MODEL_UNKNOWN` and empty envelopes. Position translation preserves the action sequence. A learned forecast can be reused only by exact suffix comparison, retaining root identity and envelope offsets. Runtime observed r must remain inside its corresponding root envelope; stale predictions stop permission preparation.

Host wall time and numerical command time are separate. Numeric time advances only for observed controller commands. Holding during operator pause freezes this numerical clock; it does not model continuous physical load motion during a real stop. The simulated accepted tail is drained and recorded before recovery. Real stop dynamics remain an experiment.

Model scales in the basic ridge baseline are fitted on training roots, without test data. Dev roots are kept disjoint but this increment does not run hyperparameter selection or reproduce the v4 development-scale/GRU protocol. Independent root-max calibration remains explicit. Results are small deterministic evaluations, not safety guarantees.

## Added physical and SSH profile

See [physics_ssh_design.md](physics_ssh_design.md) for the full design review, scoped implementation and unresolved scientific gates. `physics` adds a real MuJoCo XYZ open-tray/free-payload contact fixture with complete-state pairing, fixed 2ms gateway cadence, full compiled-model binding, continuing stop dynamics, and a signed aggregate index. `remote-physics` launches the same trusted local runtime through strict OpenSSH and verifies the returned evidence against its submitted source and authenticated transfer receipt.

This profile does not inherit the numeric predictor/calibration, incremental geometry certificates or workbench pause semantics. Geometry-only selection can slip; fastest-branch terminal-pose convergence currently fails. This increment is an experimental contact runner, not completion of v4 robot/VLA/GRU integration.
