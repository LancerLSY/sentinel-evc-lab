# Counterbalanced ARM timing evidence

This experiment freezes every candidate trace before timing, pins BLAS to one thread and the process to CPU 0–2, and rotates the three method orders across three independent Python processes. Each cell uses 12 independent roots, 60 cycles per root, and excludes 20 warm-up cycles. The three process runs are technical timing repeats; they are not treated as 36 independent roots. Confidence intervals use a paired bootstrap over the 12 root clusters (10,000 resamples).

Delta timing includes certificate validation, selected-certificate creation, and one-fortieth of the root-certificate cost. The selection policy is frozen independently of delta margins: it commits the first candidate accepted by the reference continuous checker.

## Results

| Candidates | Jitter (rad) | Full / delta speedup (95% CI) | Discrete / delta speedup (95% CI) |
|---:|---:|---:|---:|
| 1 | 0.002 | 0.749 (0.663–0.804) | 1.106 (0.967–1.186) |
| 1 | 0.010 | 0.749 (0.676–0.819) | 0.936 (0.613–1.302) |
| 8 | 0.002 | 2.571 (1.867–2.859) | 4.554 (3.304–5.056) |
| 8 | 0.010 | 1.291 (1.197–1.575) | 3.398 (2.957–4.142) |
| 32 | 0.002 | 3.580 (1.832–4.509) | 6.444 (3.166–8.129) |
| 32 | 0.010 | 1.621 (1.251–1.914) | 4.400 (2.896–5.255) |
| 128 | 0.002 | 3.065 (1.835–5.029) | 5.157 (2.933–8.443) |
| 128 | 0.010 | 2.504 (1.499–3.303) | 6.790 (3.750–8.964) |

Delta and full continuous checking produced identical candidate decisions in every cell, with zero FK-budget unknowns. The discrete checker differed on one unique candidate in the `N=128, jitter=0.01` trace (the same difference appears in each technical repeat). `N=1` is a measured negative result: delta is slower than full continuous checking.

At `N=128`, the 50 ms deadline was missed in 235/1,440 measured delta cycles at jitter 0.002 and 344/1,440 at jitter 0.01. Full and discrete checking missed it in 1,440/1,440 cycles in both cells. All lower-N delta cells had zero deadline misses.

The frozen trace files remain on the experiment host under `/root/autodl-tmp/sentinel-v2/fair-arm/traces` (373 MiB). `trace-files.sha256` and `protocol.frozen.json` bind their files and exact array bytes. The local package contains the full per-root timing results, environment, protocols, hashes, pre-summary statistical amendment, and root-cluster summary.

The evidence covers the static UR5e capsule model and CPU timing. The 1 nm numerical guard is an empirical floating-point margin, not a formal floating-point proof.

The complete frozen NPZ inputs were subsequently retrieved and hash-verified. They are included in the full experiment delivery archive. This Git evidence directory keeps the protocols, raw timing outputs and input hashes.
