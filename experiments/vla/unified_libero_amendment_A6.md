# Unified LIBERO amendment A6: certified unchanged-prefix recovery

A6 adds one conditional execution policy to the A5 fixed-snapshot, nine-coordinate
geometry experiment. When the final ten-action forecast is not certified because
it is geometrically unproved or colliding, the recovery helper checks the unchanged
candidate prefixes of lengths 5, 2, and 1, in that order. The first certified prefix
is eligible for dispatch. If none certifies, the cycle dispatches no fallback motion.

For a prefix of length `n`, certification receives exactly:

- `final_native[:, :n, :]`, with its native dtype and values unchanged;
- `final_qpos[:final_action_end_indices[n - 1] + 1]`, including the root sample and
  every retained physics substep through the last prefix action;
- `final_action_end_indices[:n]` as the returned execution boundaries; and
- the same live model, data, nine-coordinate profile, frozen scene snapshot, geometry
  configuration, exact allowed-contact policy, and cabinet-denial policy as the denied
  ten-action candidate.

The certificate binds `final_native[:, :n, :].tobytes(order="C")`. Recovery does not
reinterpret the native Cartesian/gripper action as joint motion, generate a replacement
action, widen a tolerance, permit cabinet contact, or move after all three attempts fail.
Each attempt records its status, method, reason, clearance, unknown pairs, violating geom
pairs, work counters, latency, action hash, and certificate digest. Aggregate work and
wall latency include every attempted prefix.

After a prefix dispatch, the runner re-observes and replans. Raw-parent and incremental
comparisons must be recomputed for the same shortened horizon. Any previous raw proposal
may be reused only at the actual consumed action offset; it must not be indexed as though
all ten actions were executed. The parent baseline retains its original stop behavior.
The full and delta product branches enable this policy only through an explicit
`recover_safe_prefix` condition. The episode limit remains 280 actual environment actions;
replanning cycles do not increase that limit, and task success remains the official LIBERO
success signal.

A6 is a prospective execution-policy comparison against A5. Until measured on the fixed
evaluation grid, it supports no claim of improved success, safety, efficiency, or geometric
coverage. Prefix certification retains the geometry checker's frozen-scene and proxy-geom
scope and does not establish collision freedom for unobserved dynamics or later replans.
