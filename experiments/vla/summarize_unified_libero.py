#!/usr/bin/env python3
"""Summarize retained episodes without loading a policy or executing a simulator."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values) / 1e6,
            "p95_ms": ordered[math.ceil(.95 * len(values)) - 1] / 1e6,
            "p95_rule": "nearest rank"}


def summarize(run):
    manifest = json.loads((run / "manifest.json").read_text())
    episodes = [r for r in manifest["results"] if r["phase"] == "formal"]
    records = [json.loads(line) for line in (run / "steps.jsonl").open()]
    records = [r for r in records if r["episode_id"].startswith("formal-")]
    summary = {"run": run.name, "status": manifest["status"],
               "formal_episodes": len(episodes), "unfinished_cases": manifest["unfinished_cases"],
               "manifest_metrics": manifest["metrics"], "branches": {}}
    rows = []
    for episode in episodes:
        rows.append({key: episode.get(key) for key in (
            "episode_id", "task_id", "state_index", "seed", "branch", "steps", "success",
            "crashed", "denied_candidates", "unwanted_collisions", "message")})
    for branch in ("parent_only", "full_final", "delta_evc"):
        selected = [r for r in records if r["episode_id"].endswith(branch)]
        candidates = [r for r in selected if r.get("event") == "candidate"]
        actual = [r for r in selected if r.get("event") == "actual_step"]
        summary["branches"][branch] = {
            "candidates": len(candidates), "actual_writes": len(actual),
            "naturally_revised_candidates": sum(r["raw_to_aggregate_l2"] > 0 for r in candidates),
            "incremental_methods": dict(Counter(r["incremental"]["method"] for r in candidates)),
            "full_geometry": distribution([r["timing_ns"]["full_geometry"] for r in candidates]),
            "incremental_geometry": distribution([r["timing_ns"]["incremental_geometry"] for r in candidates]),
            "parent_plus_incremental_geometry": distribution([
                r["timing_ns"]["parent_geometry"] + r["timing_ns"]["incremental_geometry"] for r in candidates]),
            "cycle_elapsed_at_write": distribution([r["full_cycle_ns"] for r in actual]),
            "maximum_tracking_error_rad": max((r["forecast_tracking_max_abs_rad"] for r in actual), default=None),
        }
    summary["timing_note"] = "Instrumented diagnostic timings include counterfactual checks. cycle_elapsed_at_write measures time from the cycle start through each write, rather than an independent completed cycle. No deployment throughput estimate."
    return summary, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"run": summary["run"], "formal_episodes": len(rows)}))
