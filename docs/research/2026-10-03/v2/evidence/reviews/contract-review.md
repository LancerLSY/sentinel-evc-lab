# D1 / D4 contract review — final working-tree pass

Review target: base `45cc883`, patch HEAD `fbac932`, plus the current uncommitted
hardening changes in `repo`. Scope is `contracts.py`, `authority.py`,
`delta_cert.py`, `events.py`, `executor.py`, `executor_pipeline.py`,
`native_gateway.py`, the D4 fixture/benchmark, and their contract wording. This
reviewer did not edit product sources.

## Verdict

**COMMENT — code path has no remaining CRITICAL/HIGH finding.** The original
revocation, byte-identity, scene-identity, D4-fixture, signed-zero, and native
evidence counterexamples are closed on the current working tree. The focused
suite is green (`93 passed, 9 skipped`; all skips require MuJoCo), the product
suite is green with loopback access (`40 passed`), and the D4 rerun is
byte-identical to the retained artifact.

Two deliberate compatibility/scope boundaries remain and must stay explicit:
scene-content binding is conditional on a supplied `scene_hash`, and the chained
prefix digest commits positions rather than gripper events. The updated v2 report
now states the atomic evidence reservation, exact `Scene.hash` migration, and the
real-domain versus observed floating-point boundary correctly.

## Remaining findings

### [MEDIUM] Scene-content binding remains an optional integration profile

Files: `src/sentinel_evc/contracts.py:386`, `src/sentinel_evc/authority.py:214`

`Context.scene_hash` remains optional, and Authority compares certificate scene
content only when the caller provides it. The shipped demo, product, physics,
and corrected D4 callers now provide it, so their execution path is bound. A
third-party integration can still omit it without an explicit strict-profile
error.

Fix or claim condition: require a non-null scene hash in the public v2/strict
execution entry point, while retaining an explicitly named legacy reader/profile
if compatibility is needed. Until then every external claim must say that scene
binding requires the caller-supplied `scene_hash`.

### [MEDIUM] Chained-prefix digest commits positions, not the full action prefix

File: `src/sentinel_evc/executor_pipeline.py:55`

`prefix_digest()` folds only each `<3d` position. Gripper events are bound by the
individual plan/lease, so this is not a demonstrated queue bypass, but
`committed_prefix_hash` cannot independently identify two histories with equal
positions and different gripper events.

Fix or claim condition: hash a typed step record including the gripper event, or
name/document this field as a committed **position** prefix digest.

## Original blockers now resolved

- **Shared-Authority revocation race:** `Authority.admit_submission()` and
  `advance_generation()` use the same re-entrant lock. In the widened race,
  revocation blocked at final admission; event order was `submit(gen=0)`, then
  `revoke(gen=1)`, and old submissions after revoke were zero.
- **Hash-only certificate bypass:** execution coverage now fails closed when a
  plan exact digest is supplied. A byte-different plan with the same canonical
  hash returned `REJECTED:CERTIFICATE_MISS`.
- **Scene hash collision:** `Scene.hash` now uses exact float64 bytes. The two
  one-ULP scenes still collide under `legacy_hash`, produce opposite geometry
  verdicts, and now have different execution hashes.
- **D4 fixture gap:** the benchmark fixture now supplies `scene_hash` and adapts
  `plan_exact`. The rerun is byte-identical to the retained pilot.
- **Signed-zero continuity:** chained plan starts are compared via packed float64
  bytes, so `+0.0` and `-0.0` no longer satisfy the stated bit-identical check.
- **Native evidence-capacity TOCTOU:** EventLog reservations pin COMMIT and
  DISPATCH and retain space for ACK. Capacity 7 rejects before writer entry;
  capacity 9 with one concurrent revoke retains all events without a gap; three
  concurrent revokes create a visible sticky gap while retaining the admitted
  write's COMMIT, DISPATCH, and ACK.
- **Writer exceptional exit:** the outer native writer handler catches
  `BaseException`, writes a reserved negative ACK, and re-raises. The final probe
  produced one ACK with `accepted=false`, `status=writer_raised`, and
  `error_type=KeyboardInterrupt`.
- **Auditable exact-write linkage:** the MAC'd lease payload and COMMIT contain
  `plan_exact`; numeric DISPATCH contains exact little-endian action bytes and
  their SHA-256 digest. This supports semantic reconstruction from the trusted
  recorder. It is evidence from a trusted software writer, not hardware
  attestation of motor motion.
- **Bounded randomized native drill:** eight frozen classes × 200 independent
  seeded actions/contexts exercised the public NativeGateway API. All 1,600
  expected outcomes matched; denied target submissions made zero writer entries,
  and all 200 legal controls entered exactly once. This is a deterministic
  contract drill, not a statistical robustness rate or competitor comparison.

## Validation evidence

```text
Ad hoc final probes
  exact certificate collision: REJECTED:CERTIFICATE_MISS
  exact scene hashes equal: false; legacy hashes equal: true
  geometry verdicts: PASS / REJECT
  shared Authority old submissions after revoke: 0
  maxlen=7 native writer calls: 0; REJECTED:EVIDENCE_GAP
  maxlen=9 + one revoke: COMMIT/DISPATCH/ACK retained; no gap
  maxlen=9 + three revokes: COMMIT/DISPATCH/ACK retained; sticky gap visible
  writer KeyboardInterrupt: negative CONTROLLER_ACK retained

bounded NativeGateway randomized drill
  8 frozen classes x 200 cases = 1,600 unique seeded cases
  invalid entries: 0
  denied target writer calls: 0
  legal target writer calls: 200
  raw JSONL reproducible byte-for-byte on a second run
  raw JSONL SHA-256 05ae7d406de0d35c8676f29d8ce8ab48f0b157c092ea68e509dafa4e015a4187

pytest (contract/geometry/physics-related selection)
  93 passed, 9 skipped in 3.70s

pytest (product integration/storage/numeric/performance, loopback enabled)
  40 passed in 8.58s

D4 benchmark
  output SHA-256 2197fd15fbd55b08bc9be0e289b4e1ad4fae4623f37b3878c8ef1782997069f5
  byte-identical to docs/research/2026-10-03/v2/pilot/pipeline_ticks.json

git diff --check: PASS
py_compile of all reviewed modified modules: PASS
```

No Python language-server executable is installed in this environment, so LSP
diagnostics were unavailable. Fresh compilation, focused tests, product tests,
static exception/secret-pattern inspection, and adversarial runtime probes were
used instead.

Stop condition for this code review is met. The ARM experiments are reviewed
separately. Carry or enforce the two remaining integration boundaries in any
public product claim.
