# A2: bounded tracking error diagnostic

Declared after A1 outcomes and before A2 execution. This diagnostic repeats only
task 5, states 40 and 41, all three branches (six episodes). Seeds, model,
aggregation, contact policy, native clipping and maximum steps stay unchanged.
A1 and the original run remain separate; A2 does not replace their denominators.

A1 aborted all six task-5 episodes because forecast/execution joint error exceeded
1e-8 rad (observed errors 4.87302424801e-8 and 1.59272687739e-6 rad).
A2 reserves rho=1e-5 rad per joint in the geometry proof and permits execution
tracking error up to that same value. Pair clearance subtracts rho times the sum
of all seven joint sensitivity bounds. Joint-limit margins also subtract rho.
Reserve exhaustion returns UNKNOWN; actual predicted penetration stays COLLISION.
The profile/certificate digest binds the reserve. This is a stronger clearance
requirement, not an unaccounted relaxation of the execution check.

Executed substep arrays are now retained before any tracking abort. A1's six
aborted-cycle array gaps remain documented and are not reconstructed.

Primary diagnostic: whether the six contact-rich episodes continue under an
explicit geometric error budget, and their task/contact/denial outcomes.
No improvement claim is predeclared. All failures and denials remain outcomes.
The same frozen-scene and moving-finger/carried-object scope limits apply.
Stop deadline remains 05:45 UTC, 2026-10-04.
