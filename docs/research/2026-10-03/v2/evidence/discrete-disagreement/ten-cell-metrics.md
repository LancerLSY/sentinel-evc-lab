# Complete ten-cell metrics

Twelve independent roots per cell; three process runs are technical repeats. Delta medians include root-certificate amortization and selected-certificate issuance.

| N | Jitter | Median ms D / F / Δ | F / Δ speedup (95% root-cluster CI) | D / Δ speedup (95% CI) | Unique disagreements D/F; Δ/F | FK-budget unknown | 50 ms misses D / F / Δ (of 1,440) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002 | 1.784 / 1.209 / 1.614 | 0.749 (0.663–0.804) | 1.106 (0.967–1.186) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 1 | 0.010 | 1.336 / 1.069 / 1.428 | 0.749 (0.676–0.819) | 0.936 (0.613–1.302) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 4 | 0.002 | 3.180 / 1.855 / 1.145 | 1.619 (1.276–1.755) | 2.776 (2.132–2.954) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 4 | 0.010 | 4.735 / 1.998 / 1.830 | 1.092 (1.035–1.253) | 2.588 (2.288–3.027) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 8 | 0.002 | 6.090 / 3.437 / 1.337 | 2.571 (1.867–2.859) | 4.554 (3.304–5.056) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 8 | 0.010 | 9.275 / 3.523 / 2.729 | 1.291 (1.197–1.575) | 3.398 (2.957–4.142) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 32 | 0.002 | 23.794 / 13.220 / 3.692 | 3.580 (1.832–4.509) | 6.444 (3.166–8.129) | 0; 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 32 | 0.010 | 35.853 / 13.205 / 8.148 | 1.621 (1.251–1.914) | 4.400 (2.896–5.255) | 0; 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| 128 | 0.002 | 105.773 / 62.862 / 20.510 | 3.065 (1.835–5.029) | 5.157 (2.933–8.443) | 0; 0 | 0 / 0 / 0 | 1440 / 1440 / 235 |
| 128 | 0.010 | 179.196 / 66.078 / 26.391 | 2.504 (1.499–3.303) | 6.790 (3.750–8.964) | 1; 0 | 0 / 0 / 0 | 1440 / 1440 / 344 |

## Comparison counts

- Unique frozen candidate decisions: **249,120**.
- Delta versus full: **249,120** unique comparisons, **0** disagreements.
- Discrete versus full: **249,120** unique comparisons, **1** disagreement.
- Counting the three technical timing repeats: **747,360** comparisons per method pair, **1,494,720** across both pairs.
- Measured timing cycles: **14,400** per method and **43,200** across the three methods.

The sole discrete/full difference is diagnosed in `result.json`; it is not treated as an independent difference three times merely because the same frozen trace was timed in three processes.
