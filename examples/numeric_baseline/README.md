# Reproducible small numerical baseline

These are actual generated artifacts for two baselines, using 32 training, 8 development, 19 independent calibration and 12 held-out test roots. Each test root has all four action branches. Reproduce in two empty directories:

```bash
python -m sentinel_evc train-baseline --mode physical --seed 10000000 --risk-limit 0.12 --out runs/physical-baseline
python -m sentinel_evc train-baseline --mode residual --seed 10000000 --risk-limit 0.12 --out runs/residual-baseline
```

| Mode | Mean absolute r error | Jointly covered test roots | Allowed branches |
|---|---:|---:|---:|
| History-identified physical | 0.0319811 | 12/12 | 0/48 |
| Physical plus ridge residual | 0.0335074 | 12/12 | 0/48 |

All branches were refused at the 0.12 risk threshold. Zero false allows is therefore vacuous here; it does not demonstrate useful availability or a low accident rate. The residual baseline did not improve mean error on this small split. Keep these negative results when expanding the experiment.

Each directory contains the dataset split identities, model, calibration, detailed per-root/per-branch evaluation and a manifest with artifact digests, source digest, Python version and exact reproduction command. The manifests describe the source used to generate these files; development roots are separated but no hyperparameter search is performed. Workbench default seed 7 uses a different scenario and calibration seed and can select the slow candidate; it is not one of these held-out test roots.

These results do not reproduce the v4 GRU/dataset protocol or validate a real robot. See [pending experiments](../../docs/experiment_plan.md).
