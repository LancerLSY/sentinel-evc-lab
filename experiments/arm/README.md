# UR5e incremental-validation experiment

This experiment runs the official Google DeepMind MuJoCo Menagerie UR5e model
at pinned upstream commit `4d038b3feae26ec82b46a4d586379114012a8ac7`.
The upstream model and meshes are BSD-3-Clause and remain external assets; they
are identified by per-file hashes in each run manifest.

It compares a parent-only decision, a full final-plan validation, bound static-prefix
reuse with full fallback, and a conservative reject-transforms baseline.
Reuse requires an allowed parent record and matching exact prefix, initial integration
state, model/assets, time/control settings and context. A missing or invalid record
causes full final-plan validation; that fallback can allow a safe final plan even
when its parent was rejected. Dynamic rollout always starts from frame zero.
This is an in-process verification record, not an externally signed v4 certificate.
Six deterministic scenario classes cover a clear
path, late suffix collision, narrow passage, joint limit, degraded tracking,
and an environment change. Obstacles are deliberately constructed from plan
kinematics. All roots are retained; these are mechanism fixtures, not samples
from a natural deployment distribution. The repaired run uses the same 180
roots as its historical version, so it is not a new statistical holdout.

The static pass interpolates joint configurations so every sampled joint step
is at most 0.01 rad. This is **discrete configuration checking**, not a proof of
continuous collision freedom. A separate cloned-state `mj_step` rollout uses
25 physics steps per 50 ms target frame. The gate script's own reference replay
shares its implementation and is diagnostic only. Formal comparison labels come
from `verify_ur5e_roots.py`: a separately implemented scanner (at most 0.005 rad)
and new model/data instances with 1 ms physics steps, 50 per target frame.
It does not import the gate scanner or rollout. Both use the MuJoCo engine;
neither provides a continuous collision proof or hardware validation.

```bash
python experiments/arm/run_ur5e_guard.py \
  --asset-dir /path/to/mujoco_menagerie/universal_robots_ur5e \
  --out runs/arm-20261002/full-v3 --roots-per-scenario 30

python experiments/arm/verify_ur5e_roots.py \
  --run runs/arm-20261002/full-v3 \
  --asset-dir /path/to/mujoco_menagerie/universal_robots_ur5e \
  --media-out runs/arm-20261002/media-v4
```

`manifest.json` records source/assets, protocol constants, environment, and
same-script reference metrics. `per_root.json` retains decisions, parent-record
bindings, fallback reasons and observed costs. The separate `review/` directory
contains reviewed outcomes, metrics and higher-resolution joint trajectories.
`verification_manifest.json` binds the exact executed inputs and outputs. Archive
hashes and the independent review confirm this publication's input consistency.

MP4/GIF/PNG are rendered from **stored higher-resolution joint trajectories**,
not full integration-state snapshots. Cases are selected by the disclosed,
outcome-conditioned rule: the first unsafe case per scenario and a safe narrow
passage control. The complete metrics include all 180 roots. GIF metadata separates
logical frames from encoded frames after duplicate-frame compression. The release
also contains a portable model directory with all 26 pinned upstream files and its
wrapper; its standalone loading was verified.

Timing is a fixed-order single-run diagnostic with early static exits. Reported
incremental totals include parent validation; they establish no algorithm speedup.
The legacy `complete_validation_calls_per_root` metric counts validator invocations,
not complete static scans. Parent records are scoped to this in-process validator
configuration; they do not independently bind validator source or tracking threshold.

[Public results and review](../../docs/research/2026-10-02/ur5e/v3/independent_review.json) ·
[Official asset verification](../../docs/research/2026-10-02/ur5e/upstream_verification.json)

The reference execution uses an **RTX 4090 D (24 GB)** host for EGL rendering;
physics is CPU MuJoCo.
