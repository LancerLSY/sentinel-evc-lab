# Geometry review — HEAD eb7b681 final-state refresh

Independent reviewer: v2_geometry_review. Read-only final refresh of specified
source and retained artifacts. The reviewer did not rerun experiments.

## Current status

- **D2 numeric CLI/App lifecycle: CLEAR.** The selected plan receives `full_v2`
  root verification. Each actual remaining suffix passes
  `validate_or_inherit_v2`, preserves exact plan/scene/profile and lineage, and is
  registered before Authority prepares a permit. The independent product review
  approved complete, stop/resume, one-ULP, changed-root and changed-scene probes.
- **D3 static UR5e experiment and evidence: WATCH.** It has bound immutable
  certificates, numerical guard, work budget, independent MuJoCo contact checks
  and frozen counterbalanced timing evidence.
- **D3 as an integrated VLA/arm product capability: BLOCK.** Its experiment
  certificate does not enter core Authority or an actual robot writer. Numeric
  D2 integration does not close this separate boundary.

## Corrected geometry and identity issues

Exact float64 scene identity closes the rounded-hash collision. Coordinates of
the parent, child, scene and index parameters share a 1000 m implementation
limit. Larger values cause full checking. The saved large-coordinate counterexample
now rejects by full fallback. Query ranges use outward rounding and a 1 nm
guard, closing the saved index-boundary false acceptance.

ARM certificates copy plan and lower-bound arrays into immutable byte storage.
The digest binds body frames, joints/limits, collision proxies, home, rho and
self-pairs. Checker identity includes depth, tightness, FK budget, self-collision
and numerical guard. Invalid model/profile/content/shape/seal causes full checking.
The process-local HMAC establishes issuance inside a trusted Python process. It
does not isolate arbitrary code executing inside that process.

Final frozen hashes verified by the reviewer:

```text
arm_delta.py
b9d22607db4fc87332d3fee938284fb40c437d1517ca4312f70477186ffb3f25
ur5e_kin.py
ff3ffd09b6a0fd8f2025c670d7c3977fa624f1765f4d9ab577a06b8acd4f95dd
```

The final rho uses `nextafter` toward positive infinity. Continuous leaves and
inherited lower bounds subtract a 1 nm guard. FK budget exhaustion returns unknown
and does not accept unresolved paths. These are bounded engineering safeguards,
not a proof of all IEEE-754 operations.

## Independent contacts and timing

The crosscheck uses its own sampler, stable geom names, and checks body, type,
group, radius and half-length. Contacts are filtered to robot–robot and
robot–sphere. `mj_forward` evaluates 482,073 dense poses across 360 paths.
All 237 proxy-certified-free paths have no sampled MuJoCo contact. This is dense
discrete contact validation, not MuJoCo continuous collision detection. The official
asset receipt and pinned XML hash match the frozen protocol.

The performance protocol freezes traces before timing. The continuous reference
chooses the future trace, so delta does not control its own workload. Three method
orders rotate across three processes. Binding, certificate copy/hash/seal and
amortized root cost are timed. The pre-summary statistical correction uses 12 root
clusters, with processes kept as technical repeats.

N4 speedups are 1.619 [1.276,1.755] and 1.092 [1.035,1.253]. N1 is slower.
N128 retains 50 ms misses. The final results replace intermediate fixed-order
pilot performance claims.

## Product boundary

The historic demo remains on v1. The accurate integration claim applies to the
numeric ProductManager reached by CLI and App/server. D3 covers a bare six-joint
UR5e with static capsule proxies, spheres and configured self-pairs. Tools,
grippers, ground/table, mesh obstacles, dynamics and tracking require separate
profiles and execution integration.

The smallest remaining joint experiment is VLA chunk → ARM checker → persistent
core certificate/Authority → one-use controller write → feedback → new cycle.
