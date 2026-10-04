# Unified LIBERO amendment A8

Frozen before A8 formal execution on 2026-10-04. Same official SmolVLA model, six paired task/state/seed roots, three branches, official task success, native action bounds, and 280 actual actions per episode as A7. Results remain separate from all earlier stages.

A7 terminal replay found two independent reasons for refusal: compiled-mesh separation axes were incomplete for a contact-free retreat, and a frozen-scene geometry decision could omit cabinet contact present in the full native simulator forecast. A8 adds optional triangle face normals and edge/frame cross directions to the conservative support-projection fallback. Every positive bound still projects the entire compiled mesh with the same numerical pad. These directions improve separation search; they are not a complete continuous collision algorithm.

A8 also vetoes any known unwanted contact in the retained native simulator forecast before dispatch, in all three branches and in every recovery candidate. A shorter unchanged prefix only checks its retained substeps. Negative native contacts are never overridden. Contact permissions, penetration tolerance, and native bounds are unchanged.

A5/A6 actual trajectories had zero measured forecast-to-execution error after full controller-state restoration. A8 restores the original strict 1e-8 rad tracking reserve and abort tolerance; the slide reserve and abort tolerance remain 1e-8 m. This change is prospective, not a reclassification of earlier results. The same six Cartesian transformations and 32 selected-repair limit apply. No goal or reward oracle selects actions.

Known scope: arm and two fingers are continuously checked against the scene frozen at the chunk root. The additional native contact veto checks simulated physics substeps, not all continuous motion of scene objects or carried objects. This is not a physical-robot safety guarantee.

Runs may overlap on the same RTX 4090 D 24GB. Timing is not a clean real-time performance comparison. Deadline 08:05 UTC; incomplete episodes stay marked incomplete. Engineering captures and preflights are excluded from formal statistics.
