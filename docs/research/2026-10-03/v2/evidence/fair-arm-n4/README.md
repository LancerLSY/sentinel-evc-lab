# N=4 product-scale extension

This extension applies the same frozen-trace, counterbalanced protocol to four candidates per control cycle. The wrapper and underlying runner hashes were recorded before timing. Statistical analysis uses 12 independent root clusters and three technical process repeats.

| Jitter (rad) | Full / delta speedup (95% CI) | Discrete / delta speedup (95% CI) |
|---:|---:|---:|
| 0.002 | 1.619 (1.276–1.755) | 2.776 (2.132–2.954) |
| 0.010 | 1.092 (1.035–1.253) | 2.588 (2.288–3.027) |

Both cells show a root-cluster confidence interval above 1.0 against full continuous checking. Delta and full produced identical decisions, with zero FK-budget unknowns and zero 50 ms deadline misses. The frozen input traces remain on the experiment host under `/root/autodl-tmp/sentinel-v2/fair-arm-n4/traces` (8.9 MiB) and are bound by the included hashes.

The complete frozen NPZ inputs were subsequently retrieved and hash-verified. They are included in the full experiment delivery archive. This Git evidence directory keeps the protocols, raw timing outputs and input hashes.
