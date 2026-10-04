# A3: complete native simulator data restoration

Declared before A3 results. Repeat the six A2 contact diagnostics: task 5, states
40/41, the same seeds and all three branches. Preserve every prior run separately.
Model, aggregation, clipping, contact policy, 1e-5 rad geometric reserve and
execution tolerance remain unchanged. Stop deadline remains 05:45 UTC.

A2 recorded exact forecast/execution agreement through cycles 0-2. Differences
started only within grasp cycle 3 and grew after contact activation/release.
The old restore used MJSTATE_INTEGRATION plus mj_forward. This restores the
integration variables and recomputes derived contact/constraint data. The old
fingerprint could not detect a difference in those derived fields.

A3 snapshots MjData with copy.copy and restores it in place through the official
mj_copyData C function, without another mj_forward. Python controller/container
and RNG restoration remain unchanged. The pinned Linux MuJoCo 3.3.7 Python
bindings do not expose mj_copyData, so the helper calls the bundled shared
library through standard-library ctypes. Version, library count, object addresses
and destination return address are checked. Existing wrapper/data identities
remain stable. The digest additionally binds acceleration, constraint forces,
contact and selected derived-state arrays.

Reference: https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-copydata

Primary diagnostic: forecast/execution agreement and task/denial/crash outcomes
on the same six contact episodes. A reduction in replay divergence supports this
local restoration correction. It does not establish collision or speed superiority.
