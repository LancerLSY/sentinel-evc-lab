# Post-hoc diagnosis of the discrete-check disagreement

This diagnosis was run after the frozen timing outcomes. It reads only the frozen `N=128, jitter=0.01` trace and does not rerun or alter timing.

The unique disagreement is:

- Root `2`, cycle `37`, candidate `32` (zero-based indices).
- The 0.01 rad discrete checker **accepted** the candidate.
- The fresh continuous checker **rejected it as collision** after 811 FK evaluations, matching the stored frozen reference label.
- An independently constructed 0.0005 rad grid sampled 1,264 configurations. Its minimum capsule-to-obstacle clearance was `−9.881554e-7 m` (about `−0.99 µm`); minimum self-clearance remained positive at `0.037226 m`.
- MuJoCo found robot/obstacle contact at 5 of the 1,264 dense samples. The first contact was between `audit_robot_6` (`wrist_2_link`) and `audit_sphere_3`, with reported distance `−4.235313e-7 m`.

The evidence establishes that this particular 0.01 rad discrete acceptance missed a narrow sampled collision. The MuJoCo result is dense `mj_forward` contact evaluation, not MuJoCo continuous collision detection. The contact depth is sub-micrometre, so it should be reported with its magnitude rather than presented as a large collision.

Artifacts:

- `result.json` — location, direction, continuous result, dense proxy minimum, MuJoCo contact and geometry binding.
- `trajectory_instance.npz` — exact 41-knot candidate, four obstacles, indices and labels.
- `diagnostic_collision_model.xml` — official collision geometry with four audit obstacle spheres.
- `diagnose.py` — complete post-hoc diagnostic.
- `ten-cell-metrics.md` / `.json` — complete ten-cell performance and comparison-count table.
