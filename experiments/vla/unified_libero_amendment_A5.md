# Unified LIBERO amendment A5: complete controller restore and moving fingers

A5 reruns all six A1 roots (tasks 0, 4, 5; states 40 and 41) with the same
checkpoint, seeds, 50/50 two-chunk aggregation, ten-action horizon, 280-action
limit, native controller bounds, allowed-contact pairs and LIBERO success signal.

Two implementation defects are corrected before measuring new outcomes:

1. The earlier state walker rejected the root `lerobot.envs.libero.LiberoEnv`
   because its package prefix was absent. It therefore captured zero Python
   objects. The old controller/RNG fingerprint claim was incomplete. A5 traverses
   the root, nested package containers, NumPy scalars and observables, and requires
   wrapper, native environment, controller and configured interpolator coverage.
2. `DeltaBuffer.push` changes array aliasing. Restoring into the post-forecast
   references could overwrite distinct saved `current` and `last` values. A5
   restores values into the captured original top-level mutable references and
   reattaches those references. The inventory and value identities are hashed.

The complete native `MjData` restore from A3 remains. An excluded model-free
forensic replays four retained A3 action chunks through the corrected state path,
including the previously failing grasp cycle. Its snapshot and motion/contact
comparisons are reported separately from formal model rollouts. No universal
claim about nested Python object identity is made.

The geometry profile optionally adds both actual finger slide joints after the
seven arm hinge coordinates. A5 enables this profile explicitly: radian and metre
step sizes, tracking budgets and joint-limit margins are separate. Certificates
bind the nine-coordinate layout while native controller action bytes remain an
opaque seven-dimensional Cartesian/gripper request. Both extra joints must be
inside the Panda arm subtree. Exact intended bowl-grasp contact pairs stay the
same; cabinet, table and other obstacles receive no new allowance.

The slide interpolation step is 1e-3 m and its tracking reserve/tolerance is
1e-8 m. The arm tracking reserve/tolerance stays 1e-5 rad. This experiment still
uses compiled collision proxies and a non-arm scene frozen at each chunk root;
future carried-object motion is not continuously certified.

A5 is a repaired implementation, not a clean isolated ablation of either fix.
A1-A3 records remain available as historical executions. Their Python-state
restore statements must be read with the root-walker correction above.
