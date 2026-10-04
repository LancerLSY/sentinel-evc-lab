# Panda geometry bridge interface

`panda_geometry.py` checks the compiled MuJoCo geometries already used by the
live LIBERO/robosuite Panda environment. It does not remap UR5e dimensions and
does not interpret the native seven-dimensional Cartesian/gripper action as
joint angles.

Build one `PandaGeometryProfile.from_model(...)` from the live `MjModel`, the
ordered seven Panda joint qpos indices, and exact task-specific
`AllowedContact` geom pairs. The profile exposes `arm_geom_ids`,
`scene_geom_ids`, `checked_pairs`, and `allowed_contacts`. By default, arm geoms
are every geom on a descendant body of the seven joint bodies; scene geoms are
the remaining non-ignored geoms. An allowed pair with `max_penetration=None` is
excluded. A finite maximum keeps the pair checked against that penetration.

At each fixed root, call `capture_scene_snapshot(model, data, profile,
external_snapshot=...)`. Then call:

```python
decision = certify_trajectory(
    model,
    data,
    profile,
    qpos_samples,                 # floating (N, 7), current live qpos first
    action_bytes=action.tobytes(),# exact native 7D request bytes
    scene_snapshot=snapshot,
    parent=parent_certificate,    # optional
)
```

A successful continuous decision has `status == "certified"`,
`scope == "continuous_joint_linear_frozen_scene_lipschitz"`, and a non-null certificate. The certificate
binds `action_sha256`, exact trajectory bytes/dtype/shape, root live-state and
scene digests, compiled model/profile/config digests, and exposes `.digest`.
Incremental reuse is allowed only for an identical trajectory or when every
parent pair clearance exceeds the conservative joint-space child-deviation
bound. Every other case runs the full checker. Collision and unknown results do
not inherit a rejection from the parent.

`GeometryConfig.tracking_reserve_rad` adds a symmetric per-joint tracking tube
around every forecast sample. Each pair clearance is reduced by the dot product
of its seven-joint motion sensitivity and that full tube radius, and joint-limit
margin is reduced by the same radius. A nominal collision remains `collision`;
when only the tracking reserve consumes the available clearance, the result is
`unknown`. The reserve is included in the config and profile digests, and all
certificate margins are already reserve-adjusted.

The continuous checker evaluates exact compiled MuJoCo geom distances and
first tries to certify groups of retained substep chords from their joint-space
bounding box. Ambiguous groups split recursively; adjacent samples use the
same conservative link-point segment bound. Reaching the subdivision limit or an
unsupported distance query returns `unknown` and no certificate. The `work`,
`min_margin`, `unknown_pairs`, `violations`, and `latency_ns` fields are intended
for experiment logs.

MuJoCo native CCD can return zero distance for separated mesh pairs
([upstream issue #3383](https://github.com/google-deepmind/mujoco/issues/3383)).
A native zero without a matching contact is therefore not treated as collision
or clearance. The checker independently searches for a separating projection
using actual compiled world mesh vertices and analytic primitive supports along
world axes, both geom frames, center direction, and the native witness direction.
After subtracting the configured minimum pad (at least 1 nm) plus a scene-scale
floating-point budget, only a strictly positive
projection gap becomes a clearance lower bound;
otherwise the pair is `unknown` and fails closed.
For every native query, exact-pair contact distances also constrain the result:
the smaller finite distance wins, and a nonfinite contact produces `unknown`.

`current_unwanted_collisions(model, data, profile)` reports actual violating
contacts from a live `env.step` using the same exact geom-pair policy.
`compare_execution(certificate, executed_qpos_samples)` binds the actual trace
and reports aligned RMSE, maximum, final, and per-joint tracking errors. This is
a tracking report, not a retroactive certificate; the runner still records
live unwanted contacts after every `env.step`.
`dense_review(...)` is a diagnostic high-resolution point scan using the same
geometry implementation. Its scope is
named `finite_resolution_joint_linear_frozen_scene_max_joint_step_rad=...`; it never issues a continuous
certificate and must not be described as continuous collision safety.

The forecast holds every non-arm qpos and mocap pose in the captured scene
fixed. MuJoCo collision geoms may themselves be simplified proxies selected by
the LIBERO/robosuite model author. The profile fingerprints those exact compiled
geoms; it does not claim that they are exact visual or physical object surfaces.
The continuous scope covers the piecewise joint-linear interpolation between
the supplied forecast samples under that fixed-scene approximation. Forecast
samples should therefore include every reversible-rollout state needed to make
that interpolation the intended checked path. It is not a certificate for an
unobserved dynamics path between sparse control-rate samples.

The bridge is an experiment-level geometry constraint check. The configured
joint-space tube covers only its stated bounded tracking error; it does not
cover other dynamics, contact force, physical stopping, human protection, or
functional-safety certification.
