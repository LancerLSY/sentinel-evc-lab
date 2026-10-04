# Unified LIBERO amendment A7: fixed certified Cartesian single-step recovery

A7 prospectively adds one optional execution policy after A6. It is disabled by
default through `recover_cartesian_motion: false`. When enabled for the full and
delta product branches, it may run only after the original final candidate and
the A6 unchanged-prefix recovery fail to produce a certifiable next action. The
parent baseline retains its stop behavior.

The helper receives the first final native action after the official LeRobot
postprocessor and optional declared native-profile clipping. It constructs the
following fixed one-step candidates while preserving the original gripper value:

1. `xyz_keep_rot_zero`: preserve XYZ translation and set rotation to zero;
2. `xyz_half_rot_zero`: multiply XYZ translation by one half and set rotation to zero;
3. `x_only_rot_zero`: preserve X, set Y/Z and rotation to zero;
4. `y_only_rot_zero`: preserve Y, set X/Z and rotation to zero;
5. `z_only_rot_zero`: preserve Z, set X/Y and rotation to zero; and
6. `xyz_retreat_half_rot_zero`: multiply XYZ translation by negative one half and set rotation to zero.

Every candidate is a copied contiguous `float32` array of shape `(1, 1, 7)`.
An exact duplicate of the original action, a duplicate generated candidate, and
any candidate with zero motion in its first six components are excluded. The
native request bounds remain unchanged. Candidates are sorted by Euclidean L2
distance to the original seven-dimensional native action; fixed name order above
breaks ties. This distance is only a deterministic proximity rule. It is not a
reward, task-progress, or safety score.

In sorted order, each candidate undergoes a fresh reversible native forecast.
The forecast must return the root plus all 25 physics-substep samples in the
nine-coordinate Panda arm-and-gripper graph, shape `(26, 9)`. Full geometry
certification then binds the exact candidate bytes. The first certified candidate
is returned to the caller. If all candidates fail, no action is returned. Forecast
or restore failures propagate and fail closed. The helper never calls the gateway,
controller, or environment writer.

Each attempt records the candidate name and transformed values, exact byte hex and
SHA-256, dtype and shape, L2 distance, fixed transform configuration, forecast
restore metadata, forecast trajectory identity, geometry decision, and forecast,
certification, and total latency. Exclusions and total helper latency are also
recorded. Root integration must recompute the raw one-step parent and final
incremental diagnostic for the selected replacement, bind its exact new bytes to
the certificate and one-use permit, and retain the original policy proposals in
the evidence record.

A7 does not widen the native controller profile, geometry tolerance, tracking
reserve, allowed-contact policy, or cabinet-contact policy. It does not use a
goal or reward oracle, add a zero-motion hold, silently remove a denied episode,
or increase the 280 actual-environment-action limit. Scene binding, gateway
freshness, exact writer-byte checks, re-observation after dispatch, and all A6
scope limits remain unchanged.

A7 is a new execution policy rather than a claim about default SmolVLA behavior.
Until it is measured on a frozen protocol, it supports no claim of improved task
success, safety, efficiency, or geometric coverage.

The frozen run caps selected one-step motion repairs at 32 per episode. Exhaustion
retains the geometric denial and dispatches no fallback motion. This additional
recovery budget does not extend the 280 actual-action limit. Overlapping independent
workers share the RTX 4090 D; timings cannot establish a clean throughput advantage.
