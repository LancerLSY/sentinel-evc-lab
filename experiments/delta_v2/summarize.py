#!/usr/bin/env python3
"""Summarize numeric_v2.json (W1 + W2).

Per W2 cell (H, M, jitter; 3 reps x 40 cycles):
  median / P95 per-cycle ms of each method (pooled over reps),
  v2 Δ with the root certificate (index build + first full_v2) amortized over the run's cycles,
  speed-up of v2 Δ (amortized) over the faster full check, on medians and on P95,
  decision agreement with v1 full_check, and segment accounting.
"""
import json
import math
import statistics
import sys
from collections import defaultdict


def pct(v, q):
    v = sorted(v)
    return v[max(0, math.ceil(q * len(v)) - 1)]


def summarize(r):
    cells = defaultdict(list)
    for x in r["W2"]:
        cells[(x["H"], x["M"], x["jitter"])].append(x)
    rows = []
    for (h, m, j), v in sorted(cells.items()):
        pooled = {k: [t for x in v for t in x["per_cycle_ms"][k]] for k in ("v1_full", "v1_delta", "v2_full", "v2_delta")}
        amort = [t + x["root_ms_v2"] / x["cycles"] for x in v for t in x["per_cycle_ms"]["v2_delta"]]
        med = {k: statistics.median(p) for k, p in pooled.items()}
        p95 = {k: pct(p, 0.95) for k, p in pooled.items()}
        med["v2_delta_amortized"] = statistics.median(amort)
        p95["v2_delta_amortized"] = pct(amort, 0.95)
        best_med = min(med["v1_full"], med["v2_full"])
        best_p95 = min(p95["v1_full"], p95["v2_full"])
        rows.append({
            "H": h, "M": m, "jitter": j, "cycles": sum(x["cycles"] for x in v), "rejected": sum(x["rejected"] for x in v),
            "agree": {a: sum(x["agree"][a] for x in v) for a in ("v1_delta", "v2_full", "v2_delta")},
            "root_ms_v2_mean": statistics.fmean(x["root_ms_v2"] for x in v),
            "median_ms": med, "p95_ms": p95,
            "speedup_median_vs_best_full": best_med / med["v2_delta_amortized"],
            "speedup_p95_vs_best_full": best_p95 / p95["v2_delta_amortized"],
            "speedup_median_vs_v1_delta": med["v1_delta"] / med["v2_delta_amortized"],
            "segments_reused_inherited_rechecked": [sum(x["segments_reused_inherited_rechecked"][i] for x in v) for i in range(3)],
        })
    return rows


def main():
    r = json.load(open(sys.argv[1]))
    print("W1", json.dumps(r["W1"]))
    rows = summarize(r)
    print("H M jitter | cyc rej | agree v1d/v2f/v2d | median ms v1_full v1_delta v2_full v2_delta(+root) | P95 v2Δ(+root) best_full"
          " | speed-up med / P95 vs best full | vs v1Δ | reuse/inh/recheck")
    for x in rows:
        md, pp = x["median_ms"], x["p95_ms"]
        print(f"{x['H']:4d} {x['M']:4d} {x['jitter']:<6} | {x['cycles']} {x['rejected']:3d} | "
              f"{x['agree']['v1_delta']}/{x['agree']['v2_full']}/{x['agree']['v2_delta']} | "
              f"{md['v1_full']:.3f} {md['v1_delta']:.3f} {md['v2_full']:.3f} {md['v2_delta_amortized']:.3f} | "
              f"{pp['v2_delta_amortized']:.3f} {min(pp['v1_full'], pp['v2_full']):.3f} | "
              f"{x['speedup_median_vs_best_full']:.2f}x / {x['speedup_p95_vs_best_full']:.2f}x | "
              f"{x['speedup_median_vs_v1_delta']:.2f}x | {x['segments_reused_inherited_rechecked']}")
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
