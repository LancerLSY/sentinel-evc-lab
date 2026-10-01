# Design review and optimized physics/SSH increment

This review compares the v4 technical design with executable repository behavior. The private design document is not included in the public source package. The numerical workbench remains a separate profile; the new MuJoCo profile is a CLI experiment runner, with an isolated SSH job launcher and verified download. No remote experiment is claimed until an authenticated job completes.

## Review findings and implemented changes

| Finding | Consequence | Implemented correction / remaining work |
|---|---|---|
| Numeric feedback could be relabelled by a caller Snapshot | A fresh-looking snapshot could mask replayed actual controller feedback | Commit/runtime checks now require actual feedback capture time to equal Snapshot time, independently fresh, valid and finite; malformed/stale feedback regression tests |
| Previous numerical plant had one-dimensional hidden load dynamics | Cannot establish three-dimensional contact behavior | Separate optional MuJoCo 3.14.0 XYZ tray servo, free six-DOF payload, gravity, friction and floor contact; measured qpos/qvel/contact feedback |
| Model digest covered selected arrays only | Changed actuator/joint/solver parameters could leave an old proof valid | Digest of full compiled MJB plus source XML, workspace and explicit profile; mutation regressions include joint axes, flags and actuator dynamics |
| Same-root pairing can omit solver/actuator state | Candidate comparison and replay cease to be controlled | `mjSTATE_INTEGRATION` snapshot after settling; restore complete state before issuing any command; byte-identical complete trace replay gate |
| Refining physics also changed gateway cadence | Convergence comparison mixed numerical and authorization changes | Public gateway tick fixed at 2ms; physics substeps independently 2/1/.5/.25/.125ms; same 50ms command interpolation contract |
| Stop could freeze the simulated load | Revocation would overstate physical stopping | Invalidate new submissions, retain one accepted tail, confirm cancel/drain, continue dynamics at least .5s; physical payload motion is recorded |
| Evaluator labels could choose the winner | Offline oracle can masquerade as online safety | Integrated policy selects fastest geometry-feasible final Plan before any sibling future evaluation; it genuinely slips in the fixed fixture |
| Signed trials alone left aggregate claims mutable | Summary could misrepresent valid individual trials | Signed experiment index binds summary, complete source/test manifest, installed distribution versions, configuration and all trial anchors |
| Network calls inside Executor would make stop timing unbounded | SSH delay could block local dispatch/revoke | SSH only launches/transports isolated jobs. Authority, Executor, controller and simulation clock run together on the remote host |
| Self-contained signing keys are not external identity | Whole experiment substitution remains possible without retained pins | SSH strict known-host authentication plus local transfer receipt and hashes; customer PKI and independently retained receipts remain deployment work |

## Explicit physics scope

The fixture has an open rectangular tray actuated along three slide joints, a freely moving box payload and a floor plane. All X/Y/Z targets vary along the motion. There are **no physical workspace walls, articulated robot links, cameras feeding decisions, real VLA checkpoint, learned three-dimensional consequence predictor or physical device**. Rendered views replay measured states; they never provide future observations to authorization.

The local analytic certificate uses a conservative 0.17m tool projection inside a virtual workspace and actuator target ranges. Its profile is `mujoco-tray-geometry-v1`; it never proves that a free payload stays supported. Full model/scene identity is checked at commit and runtime. Incremental inheritance and the one-dimensional ridge calibration are not reused in this profile. All final candidates are rechecked through this trusted local verifier.

The numerical workbench's scalar load model and freeze-on-pause clock remain documented in `implementation_matrix.md`. They cannot be used as evidence for physical braking or 3D contact safety. The new physical runner keeps evolving after revoke and does not offer operator recovery during a physics trial.

## Frozen experimental protocol

1. Settle each physics resolution for .3s; require support, last-.1s drift <1mm, speed <1mm/s, no engine warnings. Save the complete integration root.
2. From that same root at each resolution run .6/.9/1.2/1.6s minimum-jerk motions, translating (.35,.10,.08)m within a common 40-command horizon. No release events or gripper profile are inferred. The v4 illustrative H=16 is deliberately extended to 40 observation intervals in this contact fixture, while each lease still approves at most four commands; this is a scoped benchmark profile, not reproduction of the full v4 policy configuration.
3. Run a separate geometry-only selected branch before collecting sibling outcomes. Retain its slip/failure alongside the slower branch; neither substitutes for a learned WorldGuard.
4. Compare .25ms with .125ms: classification/contact agreement, peak offset and final position within 1mm, final speed difference within 1mm/s, tracking peak difference within 1mm, quaternion angle within .01rad. Coarser levels remain in the evidence. Thresholds are fixed before reporting the result.
5. Test revoke after six submissions, complete-root replay, and scene geometry/friction change after proof issuance. No old-generation new submissions; one accepted tail must finish while physics keeps moving.
6. Save 25 individually signed trial bundles and the signed experiment index. The last-written `COMPLETE.json` means the job finished and packaged evidence; it does **not** mean the scientific gates passed.

Slip means maximum payload/tray XY offset exceeds .06m in this fixture. This is a task outcome metric, not a general robot safety threshold. Contact classification is recorded separately from command execution completion. Every result carries source, environment, root, Plan, model/configuration, actual feedback trace and events.

Local development found a real negative result: the .6s candidate slips, while the slower fixture candidates remain supported. With the strict multi-metric convergence gate, the .6s candidate's terminal position and orientation are not converged at .25/.125ms. The .9/1.2/1.6s candidates passed this gate in the recorded local refinement. This is incomplete numerical validation of the fast contact branch; retain it, investigate contact parameter sensitivity and refine time steps before publishing its precise terminal pose as a stable measurement. Development pilots remain outside the repository rather than being presented as final acceptance artifacts.

## SSH job and evidence contract

`remote-physics` uses an existing OpenSSH alias, keys/agent, `BatchMode=yes` and `StrictHostKeyChecking=yes`. It accepts no passwords, reads no password comments, does not edit SSH configuration or known_hosts, and does not weaken host verification. A target with a changed host key fails closed. Python 3.10+ with venv/pip and network access to install optional dependencies is required. Optional rendering sets `MUJOCO_GL=egl` before MuJoCo imports and requires a working EGL driver; physics itself needs no GPU.

The submitted archive contains only public source, tests, pyproject, README and LICENSE. It excludes `.git`, private design documents, local experiment outputs and SSH files. A unique remote directory under `~/.local/share/sentinel-evc/ssh-jobs/` holds a private venv, logs and results. The job installs `.[physics,test]`, runs the full tests, executes the physics CLI, validates the signed index and trials, then returns one completion marker. A completed negative experiment (exit 3) is downloadable; installation/test/partial-output failures are not success.

The local launcher bounds archive size/count, rejects traversal and links, validates the transferred archive hash, verifies signed source/configuration bindings against the submitted tree, and atomically publishes a complete downloaded result with `REMOTE_RECEIPT.json`. `verify-physics` detects later changes relative to that receipt. The receipt records the SSH alias/known-host fingerprints and authenticated transfer pins, not credentials. Keep it separately if it is to serve as an external anchor. A self-contained demo signing key proves internal integrity, not that a hostile remote operator told the truth.

## Staged next work

| Stage | Required evidence | Current status |
|---|---|---|
| L0 numeric product | Four final candidates, calibration, revoke, signed replay | Implemented earlier; profile limitations explicit |
| L1 contact fixture | True physics, causal gateway, complete state, convergence, negative branch | Implemented; fast full-pose convergence remains failed |
| L1 remote reproduction | Authenticated SSH host, job, returned and independently checked evidence | Runner implemented; live authentication/host selection still required |
| L2 robot/visual/VLA shadow | Frozen robot and camera model, actual checkpoint/action normalization, raw/final shadow logs | Pending; no invented integration |
| L3 trained 3D WorldGuard | Root-group data split, train/dev/cal/test, action ablations, held-out consequence coverage | Pending; numeric predictor does not authorize L1 |
| L4 physical controller | Measured queue/cancel/braking/sensor bounds and approved device cell | Pending |

A stronger production design additionally needs process/key separation for CertificateStore and prediction registration, bounded driver calls, durable external signer/anchors, schema/version migration, configuration/environment locks and an authenticated multi-user service. The current trusted local Python process is not an isolation boundary. Keep arbitrary remote certificate/prediction registration out of its API until those boundaries exist.

## Primary references

- [MuJoCo Python bindings and Renderer](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo state and numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MuJoCo mj_getState, mj_setState and mj_saveModel](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
- [MuJoCo MJCF actuators and contact parameters](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
