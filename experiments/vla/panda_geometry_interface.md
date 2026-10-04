# Panda geometry bridge interface

`panda_geometry.py` checks the compiled MuJoCo geometries already used by the
live LIBERO/robosuite Panda environment. It does not remap UR5e dimensions and
does not interpret the native seven-dimensional Cartesian/gripper action as
joint angles.

Build one `PandaGeometryProfile.from_model(...)` from the live `MjModel`, the
ordered seven Panda arm qpos indices, and exact task-specific `AllowedContact`
geom pairs. To check controlled finger motion too, pass the two finger slide
indices as `controlled_extra_qpos_indices`; their joint ids may be passed as
`controlled_extra_joint_ids` or inferred. The profile rejects non-hinge arm
joints, non-slide extras, and extras outside the Panda arm body subtree, then
exposes the combined ordered
`moving_qpos_indices`, `moving_joint_ids`, and `coordinate_units`. The first
seven coordinates are radians and every controlled extra is metres. Omitting
the extras preserves the original seven-coordinate interface. The profile also exposes `arm_geom_ids`,
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
    qpos_samples,                 # floating (N, len(profile.moving_qpos_indices))
    action_bytes=action.tobytes(),# exact native 7D request bytes
    scene_snapshot=snapshot,
    parent=parent_certificate,    # optional
)
```

A successful continuous decision has `status == "certified"`,
`scope == "continuous_joint_linear_frozen_scene_lipschitz"`, and a non-null certificate. The certificate
binds `action_sha256`, exact trajectory bytes/dtype/shape, root live-state and
scene digests, compiled model/profile/config digests, `coordinate_units`, and
exposes `.digest`. The opaque native action binding stays the exact 7D action
bytes even when the forecast trajectory has nine controlled coordinates.
Incremental reuse is allowed only for an identical trajectory or when every
parent pair clearance exceeds the conservative joint-space child-deviation
bound. Every other case runs the full checker. Collision and unknown results do
not inherit a rejection from the parent.

`GeometryConfig.tracking_reserve_rad` adds a symmetric tube to each of the seven
hinge coordinates; `tracking_reserve_slide_m` independently applies to every
controlled slide coordinate. Each pair clearance is reduced by the dot product
of its coordinate-wise motion sensitivity and these reserves. Hinge sensitivity
has metres-per-radian units; slide sensitivity has metres-per-metre units.
Hinge and slide joint-limit margins are reduced by their matching reserve and
are reported separately as `min_joint_limit_margin_rad` and
`min_slide_limit_margin_m`. A nominal collision remains `collision`;
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

`GeometryConfig.mesh_separator_axes` defaults to `False`, preserving the
historical checker. When explicitly enabled, a native-zero query that the cheap
axes cannot separate additionally tries world-space normals from actual compiled
mesh triangles. If those still fail, it tries crosses between compiled mesh
triangle edges and the other geom frame's three axes. Directions are generated
in bounded batches and deduplicated. At most 2,048 unique extra directions are
evaluated in deterministic compiled-face order, and processing stops after a
batch proves positive clearance. Exhausting that budget without a proof remains
`unknown`. Support still projects every compiled mesh
vertex or uses the same analytic primitive support. A strictly positive gap
after the existing metre and floating-point pads is a conservative clearance
proof, including for a nonconvex mesh. These extra axes are incomplete and are
not claimed as a complete separating-axis test. They never override a negative
native or exact-contact distance. The flag is bound into the config and profile
digests.

`current_unwanted_collisions(model, data, profile)` reports actual violating
contacts from a live `env.step` using the same exact geom-pair policy.
`compare_execution(certificate, executed_qpos_samples)` binds the actual trace
and reports hinge RMSE/maximum/final/per-joint errors in radians separately from
slide RMSE/maximum/final/per-slide errors in metres. This is
a tracking report, not a retroactive certificate; the runner still records
live unwanted contacts after every `env.step`.
`dense_review(...)` is a diagnostic high-resolution point scan using the same
geometry implementation. Its scope records both `max_joint_step_rad` and
`max_slide_step_m`; it never issues a continuous
certificate and must not be described as continuous collision safety.

The forecast holds every qpos outside `profile.moving_qpos_indices` and every
mocap pose in the captured scene fixed. With no controlled extras, finger slides
remain frozen and the certificate does not cover independent finger motion.
With both slides configured and included in every sample, their actual piecewise
linear traces are checked with their own metre-valued bounds. MuJoCo collision
geoms may themselves be simplified proxies selected by
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
