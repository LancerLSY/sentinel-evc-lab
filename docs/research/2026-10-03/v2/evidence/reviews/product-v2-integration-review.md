# Numeric product Δ-Cert v2 integration review

Review target: `src/sentinel_evc/product_pipeline.py` working-tree change on HEAD
`fbac932`. Scope is the shared numeric CLI/App lifecycle only. UR5e, native VLA,
and ARM experiment results are outside this review.

## Verdict

**APPROVE. No CRITICAL, HIGH, or MEDIUM correctness finding.** The selected
numeric plan receives one exact-scene v2 root certificate, and every execution
lease uses a certificate for the exact unexecuted suffix. The existing
Authority still supplies the final registered-certificate identity, exact plan
byte, scene-content, one-time, deadline, prediction, and generation checks.

A completed fast product run and a real stop/resume run both preserved the
expected certificate lineage and exact bindings. Direct adversarial probes also
showed that a byte-different canonical-hash collision is rejected, a changed
root plan forces a full fallback, and a checker from another exact scene fails
closed.

## Contract analysis

### Stop and recovery

Files: `src/sentinel_evc/product_pipeline.py:232-280`

Stopping revokes the current Authority generation and drains/cancels the
controller before recovery. `_observe()` advances `observed_index` for any
accepted tail that really becomes observed during cancellation. Recovery then
updates the context epoch while preserving `scene_hash`, and the next suffix is
constructed from `plan.knots[observed_index:]`. Thus the reused segments are the
actual remaining suffix, not the suffix that was current when stop was clicked.

Observed stop/resume probe:

```text
observed at stop: 2
final status: completed
certificate horizons: 40, 38, 34, 30, ..., 6, 2
lease generations: 0, then 1 for every post-resume lease
scene hashes across certificates: 1
exact digests present on every certificate: true
broken parent references: 0
```

### Root-certificate stability

Files: `src/sentinel_evc/product_pipeline.py:216-227`,
`src/sentinel_evc/delta_cert_v2.py:142-193`

The root covers the immutable selected `Plan`; `Scenario` is frozen and its
scene reconstruction has the same exact scene hash. Before inheritance,
`deps_v2()` checks the root's exact plan digest, exact scene digest, proof scope,
dt, descriptor, offset, gripper-event translation, depth, and numeric scale.
The product passes transformation kind `shift` and constructs suffix knots and
events directly from the root plan.

A parent whose bytes were changed by one ULP produced
`reason_code=parent_plan_mismatch`, `full_checks_used=1`, verdict `FULL`. Passing
a checker built for another exact scene raised `ValueError` before certificate
reuse. Both are fail-closed outcomes.

### Exact suffix and registered certificate

Files: `src/sentinel_evc/product_pipeline.py:252-271`,
`src/sentinel_evc/delta_cert_v2.py:112-123`,
`src/sentinel_evc/authority.py:205-216`

`validate_or_inherit_v2()` emits a certificate whose `plan_exact` is the suffix
`Plan.exact_hash`. `to_store_certificate()` preserves the exact plan digest,
scene hash, margins, parent/root IDs, depth, and proof scope. The adapted object
is registered before `Authority.prepare()`, and that same object is passed to
prepare. Authority then compares both exact coverage and complete registered
object identity.

Probe results:

```text
suffix exact digest == certificate exact digest: true
suffix exact digest == lease exact digest: true
one-ULP replacement has same canonical hash: true
one-ULP replacement has same exact hash: false
prepare replacement result: REJECTED:CERTIFICATE_MISS
```

### Event lineage

File: `src/sentinel_evc/product_pipeline.py:259-270`

The first offset-zero lease logs the FULL root certificate. A root row has
`parent_id=null` and `root_id=null`, which is the existing schema's root-node
sentinel rather than an unresolved reference. Every derived suffix certificate
has `parent_id=<root cert id>` and `root_id=<root cert id>`, and the root event
appears earlier in the signed sequence. Normal and stop/resume probes had zero
broken parent references.

Normal run:

```text
root: cert2-...001, parent_id=null, root_id=null, horizon=40
children: 9; every parent_id/root_id resolves to the root
certificate / lease / commit counts: 10 / 10 / 10
proof scopes observed: numeric-sphere-box-L1-active-v2 only
```

The independent bundle verifier proves event/file integrity, not the geometry
semantics of this lineage. That is an existing evidence-scope boundary: it does
not independently recompute v2 margins from a bundled full plan asset. Product
wording should continue to distinguish signed recorder integrity from physical
or mathematical re-verification by the bundle verifier.

## Validation

```text
Existing targeted product/storage/numeric tests
  21 passed in 4.54s

Fresh completed product run
  completed; root_full_checks=1; suffix_full_checks=0
  segments_reused=220; all certificate exact/scene/profile bindings present

Fresh stop/resume run
  completed after stop at observed step 2
  root_full_checks=1; suffix_full_checks=0; segments_reused=240
  one root, ten derived suffix certificates, zero broken lineage references

Adversarial direct probes
  canonical-hash / exact-byte collision: rejected CERTIFICATE_MISS
  changed root plan: parent_plan_mismatch -> full fallback
  changed exact scene with old checker: failed closed

product_pipeline.py py_compile: PASS
git diff --check: PASS
```

No Python language server is installed, so LSP diagnostics were unavailable.
Compilation, existing product tests, two end-to-end lifecycle runs, event-lineage
inspection, and direct adversarial probes were used instead.
