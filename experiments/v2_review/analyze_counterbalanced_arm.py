#!/usr/bin/env python3
"""Root-cluster analysis for frozen counterbalanced arm timing results."""

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("discrete_0.01", "certified_full", "delta_arm")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def ratio_ci(a, b, rng):
    point = float(np.median(a) / np.median(b))
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(a), len(a))
        draws.append(np.median(a[idx]) / np.median(b[idx]))
    return {"point": point, "paired_root_cluster_bootstrap95":
            [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs=3, required=True)
    ap.add_argument("--amendment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reps = [json.load(open(p)) for p in args.inputs]
    amendment = json.load(open(args.amendment))
    if not amendment.get("created_before_summary_or_outcome_inspection"):
        raise RuntimeError("analysis amendment was not frozen before outcome inspection")
    if sorted(r["repeat_index"] for r in reps) != [0, 1, 2]:
        raise RuntimeError("expected the three distinct technical repeats")
    out = {
        "schema": "sentinel-arm-root-cluster-summary-v1",
        "independent_roots_per_cell": 12,
        "technical_process_repeats_per_root": 3,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "analysis_amendment": amendment,
        "cells": {},
    }
    for ci, name in enumerate(reps[0]["cells"]):
        exact = {r["cells"][name]["trace_exact_sha256"] for r in reps}
        if len(exact) != 1:
            raise RuntimeError(f"trace changed across repeats: {name}")
        roots = len(reps[0]["cells"][name]["methods"][METHODS[0]])
        if roots != 12:
            raise RuntimeError(f"expected 12 roots, got {roots}: {name}")
        aggregate = {m: [] for m in METHODS}
        technical = {m: [] for m in METHODS}
        misses = {m: 0 for m in METHODS}
        disagreement = {m: 0 for m in METHODS}
        budget = {m: 0 for m in METHODS}
        for root in range(roots):
            for method in METHODS:
                values = []
                for rep in reps:
                    row = rep["cells"][name]["methods"][method][root]
                    value = row["median_ms"]
                    if method == "delta_arm":
                        value += row["root_certificate_ns"] / len(row["timing_ns"]) / 1e6
                    values.append(value)
                    misses[method] += row["deadline_50ms_misses"]
                    disagreement[method] += row["disagreements_all_cycles"]
                    budget[method] += row["budget_unknown_candidates_all_cycles"]
                technical[method].append(values)
                aggregate[method].append(float(np.median(values)))
        arrays = {m: np.asarray(v) for m, v in aggregate.items()}
        rng = np.random.default_rng(BOOTSTRAP_SEED + ci)
        out["cells"][name] = {
            "N": reps[0]["cells"][name]["N"],
            "jitter": reps[0]["cells"][name]["jitter"],
            "trace_exact_sha256": next(iter(exact)),
            "root_aggregate_ms": aggregate,
            "technical_repeat_ms_by_root": technical,
            "median_ms": {m: float(np.median(v)) for m, v in arrays.items()},
            "speedup_full_over_delta": ratio_ci(arrays["certified_full"], arrays["delta_arm"], rng),
            "speedup_discrete_over_delta": ratio_ci(arrays["discrete_0.01"], arrays["delta_arm"], rng),
            "deadline_50ms_misses_across_technical_repeats": misses,
            "disagreements_across_all_cycles_and_repeats": disagreement,
            "budget_unknown_candidates_across_all_cycles_and_repeats": budget,
        }
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
