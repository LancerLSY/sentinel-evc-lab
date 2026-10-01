# v4 design alignment review

This review compares the Sentinel EVC v4 technical design with the shipped
implementation. Requirement descriptions below summarize the design; the private
document is not part of the public repository. Its targets
are comparison criteria, not evidence that an experiment has been completed.

## Assessment and release boundary

**GPU experiments, 2026-10-02:** numerical GRU, observed-object MuJoCo,
image-to-object WorldGuard and SO100 joint predictors have independent evaluation
records. SmolVLA training uses 5000 fine-tuning updates, with paired held-out evaluation
and saved-weight/10-window verification. [Training results](gpu_training_results.md) and
[model cards](gpu_model_cards.md) supersede the missing-training statements below;
the remainder of this review describes the previously shipped product profile.
Its live VLA transform/permit bindings, device execution and external trust
integration still require their own implementation and evidence.

The local numeric product and fixed MuJoCo fixture implement useful, testable
parts of the mechanism. Full v4 product completion remains incomplete at live
VLA/learned-predictor integration, the real motion driver and external trust
integration. The current foundation has a **WATCH** architecture assessment:
its bounded profiles can be released with the limitations below visible.

The numeric profile operates on fixed Cartesian position Plans and a one-dimensional
load model. The MuJoCo profile uses a three-axis open tray and free payload with
actual contact integration, five timesteps and scripted candidate Plans. Model
asset import and Universal Robots read-only diagnostics provide integration ports;
neither enables arbitrary-model actuation or real robot motion.

## Original requirements F01–F12

The IDs here match section 03 of the supplied design. The older repository-specific
mechanism list is now labelled M01–M12 in [implementation_matrix.md](implementation_matrix.md).
“Scoped” means verified behavior within the stated implemented profile, not full
completion of the requirement across v4 integrations.

| ID | Design requirement | Current implementation and evidence | Remaining scope |
|---|---|---|---|
| F01 | Retain raw model actions and final executable actions | Partial: all four numeric final Plans, predictions, selection and replay retained; physics records scripted Plan and actual feedback | Real VLA raw outputs, decoding/normalization and per-step raw-to-final alignment |
| F02 | Explicit units, frame, mode, timing and gripper semantics | Scoped: immutable ActionDescriptor/Plan, finite constructors, dt and gripper contract tests | Joint ordering, normalization versions, camera/robot frames and actual adapter contracts |
| F03 | Register action transforms with new digests and dependencies | Partial: TransformRecord binds parent/child, kind, parameters and version in delta demo | Product upstream transform chain, affected ranges and dependency classifications |
| F04 | Full original-constraint validation and delta inheritance | Scoped numeric static geometry: cumulative margins, scene/time/event bindings, independent full comparison and full fallback; physics uses full compiled-model binding | Robot link/dynamic/recoverable-terminal constraints; physical delta inheritance |
| F05 | Separate prepare/commit and recheck live context | Scoped local implementation: all current numeric Context fields, fresh feedback, generation, prediction validity and single-use lease | Normalization/calibration/firmware identity, distributed CAS/transport and hardware timing qualification |
| F06 | Single command writer; revoke blocks new old-generation sends | Scoped numeric/MuJoCo Executor; revocation, cancel/drain and fault regressions | Enforced hardware driver capabilities and process/key isolation |
| F07 | Record irrevocable work and submitted/accepted/observed cursors | Scoped simulated controller and MuJoCo actual trace; accepted tail retained | Device acceptance semantics, queue bounds and measured physical braking tail |
| F08 | World model evaluates final candidates; changed actions invalidate prediction | Scoped numeric four-final-Plan evaluation, exact suffix inheritance, root offsets and runtime envelope; registry field mutations now deny approval | Actual VLA and three-dimensional WorldGuard, repair-to-reprediction chain |
| F09 | Explicit unknown/no-solution/timeout fallback; no stale success replay | Scoped numeric rejection, MODEL_UNKNOWN, expiry, preparation cancellation, fault drain and approved recovery; new independent runs after failures | Device-specific approved fallback/braking policy and terminal-set validation |
| F10 | Independent evidence verification tied to version | Scoped Ed25519 manifests, events, source/config/environment index, reread/list/export verification and tamper refusal | Customer PKI, external anchors, sensor authenticity and evidence custody |
| F11 | Group evaluation by root; retain all rounds and outcomes | Scoped disjoint train/dev/cal/test root groups, root-max calibration, all four numeric branches; complete-state physics pairing | Large held-out sets, distribution shifts, actual VLA/robot task coverage |
| F12 | Reproduction, manifests, rollback and data permissions | Partial: CLI/App installer, source manifests, performance/scenario commands, strict SSH transport package | Qualified release rollback, customer data authorization/retention and completed authenticated SSH reproduction |

Relevant code is in `contracts.py`, `authority.py`, `executor.py`, `delta_cert.py`,
`prediction.py`, `product_pipeline.py`, `runstore.py`, `physics_experiment.py`,
`physics_jobs.py`, `installation.py` and `ssh_experiment.py`. Regression evidence
and additional scenarios are described in [reliability_results.md](reliability_results.md).

## Substitutions and boundaries

- The ridge residual baseline does not reproduce the v4 GRU training protocol.
- Bounded geometric repair, disabled by default, does not implement SQP or a
  complete repaired-action prediction pipeline.
- Scripted physics candidates do not establish VLA inference. The XYZ tray does
  not model an articulated arm, camera perception or real support sensing.
- Imported OBJ/STL/MJCF/URDF assets remain inspection assets. Compile/settle
  success leaves `physics_authorized: false`.
- Local HMAC and prediction/certificate registries trust the local process.
  Mutation checks protect registered bindings; they do not isolate an untrusted
  plugin that can access Authority memory or keys.
- Ed25519 bundles establish byte integrity under the chosen key. Included demo
  public keys do not establish customer identity or sensor truth.
- After the worker records REVOKE, no new old-generation dispatch is allowed.
  HTTP `stopping` acknowledges a request; `stopped` confirms simulated cancel/drain.
  Previously accepted work may
  still execute. Numeric waiting freezes logical load time; MuJoCo continues
  physical integration during stop/drain. Neither is a hardware braking proof.

## Design improvements and next acceptance gates

1. Keep each profile's action domain, constraint family, prediction requirement,
   clock, cancellation semantics and trust source explicit. Cross-profile data
   cannot silently inherit a forecast or certificate.
2. Before accepting an external VLA plugin, place signing/driver authority in a
   separate process and define canonical raw/final/transform manifests. Test
   normalization changes, modified gripper actions, replay and adapter timeouts.
3. Begin VLA integration in read-only shadow mode with an actual checkpoint and
   licensed task data. Retain every root's proposals and refusals; promote only
   after final-action alignment and calibrated held-out evidence.
4. Before motion, qualify the device adapter's submitted/accepted/executed queues,
   cancellation ACK, feedback freshness and maximum accepted/braking tail. Define
   approved fallback/terminal sets from measured device behavior.
5. Qualify deployment separately: dependency hashes or a wheelhouse, SSH disk
   retention, release rollback, external trust custody and customer data policy.
   Current SSH jobs deliberately retain source/venv/log/evidence; operator cleanup
   and package-index dependency resolution remain deployment limitations.

These are product integration gates. The separately reported GPU experiments
establish authenticated SSH training, actual checkpoint loading and scoped model
evaluation; they do not supply hardware execution or external trust results.
