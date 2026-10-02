# Broad physical fallback stress protocol

## Research question

The earlier low-friction profile showed that a 4.8 s candidate could complete its fixed 100-root profile without an observed unsafe outcome. The integrated WorldGuard study then rejected many mass-shift and goal-shift roots because those conditions were outside its supported decision route. This prospective experiment asks whether the physical fallback remains useful across a broader, explicitly stratified simulator profile before any new product route is added.

This is a bounded MuJoCo stress study. It does not qualify every point in a continuous rectangular domain, validate hardware, or use a learned WorldGuard model.

## Frozen factorial profile

The study has 27 cells:

- friction: `[0.015, 0.05]`, `[0.05, 0.2]`, `[0.2, 0.8]`;
- payload mass: `[0.02, 0.06]`, `[0.06, 0.13]`, `[0.13, 0.2]` kg;
- displacement scale: `[0.8, 1.0]`, `[1.0, 1.2]`, `[1.2, 1.5]` times `(0.35, 0.10, 0.08)` m.

Each formal cell contains eight exact parameter corners and four independently seeded random interior points. Random friction is log-uniform; mass and displacement scale are uniform. The 324 formal roots use `730000 + cell_index*100 + local_index`. The 27 preflight roots use the disjoint `830000` series.

Every root captures one complete `mjSTATE_INTEGRATION` snapshot after the same eight-frame observation history. The 1.6, 3.2, and 4.8 s siblings restore that exact snapshot, run for a 5.0 s candidate window, and then hold for another 0.5 s.

## Policies and information boundary

The three frozen policies are fixed 1.6 s, fixed 4.8 s, and the shortest candidate satisfying the declared `mu=0.015` target-acceleration bound. If no candidate satisfies the bound, that policy rejects the root.

All policies may read whether the payload is supported at the captured root. The floor policy additionally reads only the actual candidate target sequence, gravity, and the declared floor. It cannot read root friction, payload mass, or future outcome labels. The floor is a trusted profile setting, not a measured friction estimate.

## Outcomes

The primary outcome covers the complete 5.5 s candidate-plus-hold window. Unsafe means the sampled payload-relative XY offset exceeds 0.06 m or the payload meets the existing drop condition at any sampled frame in that window. Completion requires selection, no posthold unsafe/drop outcome, and posthold endpoint tracking error at most 0.01 m. Rejections remain non-completions in the all-root denominator. The 5.0 s forecast result is retained as a secondary diagnostic, together with the count of roots that first become unsafe during the additional 0.5 s hold.

The study reports root-level aggregate and per-cell primary unsafe, drop, completion, reject, and additional-hold-risk rates with Wilson 95% intervals, plus the secondary forecast outcomes. Aggregate policy differences use 10,000 paired root bootstrap replicates. Servo force saturation, force peaks, and tracking error are sampled at 50 ms frame endpoints; they are not transient substep maxima.

## Frozen acceptance gate

For the floor policy, the observed research gate requires zero full-window unsafe roots, zero full-window drops, and at least 95% full-window completion across all 324 roots. Failure is reported directly and does not trigger parameter or threshold tuning.

The source hashes and exact machine-readable protocol are committed before the formal roots are executed. This establishes prospective ordering inside the repository; it is not an external preregistration.
