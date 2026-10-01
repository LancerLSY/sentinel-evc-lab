# Numeric prediction and the future visual WorldGuard

The v0.1 interface-only placeholder is superseded for the **numeric domain** by
`src/sentinel_evc/prediction.py`, `calibration.py`, `numeric_world.py` and `data.py`.
The local workbench compares a history-only physics identification baseline and a
trainable standard-library ridge residual, using independent root calibration.
This is not the historical v4 GRU or a visual world model.

The visual/VLA branch remains pending. Activation requires a frozen checkpoint,
observable inputs only, an equally informed physical baseline, action shuffle and
no-action ablations, independent calibration/test roots, and measured closed-loop
benefits including inference, wait, refusal and recovery costs.

A task requiring a consequence prediction rejects missing, unknown, unregistered,
expired or mismatched evidence. It does not silently bypass the prediction gate.

See `docs/implementation_matrix.md`, `docs/experiment_plan.md` and the actual small
baseline artifacts in `examples/numeric_baseline/`. Historical document metrics are
not current measurements and are not reproduced by these ridge models.
