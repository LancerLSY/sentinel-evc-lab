#!/usr/bin/env python3
"""Plot the retained fresh-state confirmation; never run a policy or simulator."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_libero_results import digest, wilson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "complete" or manifest["mode"] != "formal":
        raise ValueError("A complete formal manifest is required")
    episodes = manifest["episodes"]
    grid = {(task, state) for task in range(10) for state in range(30, 40)}
    if len(episodes) != 100 or {(row["task_id"], row["initial_state_index"]) for row in episodes} != grid:
        raise ValueError("Unexpected fixed task/state grid")
    counts = [sum(row["success"] for row in episodes if row["task_id"] == task) for task in range(10)]
    total = sum(counts)
    if total != manifest["metrics"]["overall"]["successes"]:
        raise ValueError("Aggregate success count differs from episode records")
    for task, count in enumerate(counts):
        if count != manifest["metrics"][str(task)]["successes"]:
            raise ValueError("Task success count differs from episode records")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(13, 6.2))
    p = np.array(counts) / 10
    intervals = np.array([wilson(count, 10) for count in counts])
    errors = np.maximum(0, np.stack([p - intervals[:, 0], intervals[:, 1] - p]))
    ax.bar(range(10), p * 100, width=.67, color=["#c86148" if count < 10 else "#228b78" for count in counts])
    ax.errorbar(range(10), p * 100, yerr=errors * 100, fmt="none", ecolor="#35465a", capsize=4)
    for task, count in enumerate(counts):
        ax.text(task, intervals[task, 1] * 100 + 2, f"{count}/10", ha="center", fontsize=11)
    low, high = wilson(total, 100)
    fig.suptitle(f"Fresh-state full-suite confirmation: {total}/100 successes", fontsize=18, fontweight="bold", y=.96)
    ax.set_title(f"Official SmolVLA LIBERO · MuJoCo 3.3.7 · reobserve every 10 actions\nOverall descriptive 95% Wilson interval: {100 * low:.2f}–{100 * high:.2f}%", fontsize=12, pad=16)
    ax.set_xticks(range(10), [str(task) for task in range(10)])
    ax.set_xlabel("LIBERO-Spatial task ID · ten fresh initial states 30–39 per task")
    ax.set_ylabel("Official task success (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8edf2")
    ax.set_axisbelow(True)
    crashes = manifest["metrics"]["overall"]["crashes"]
    fig.text(.075, .045, f"Frozen 100-cell denominator; all failures/crashes retained (crashes: {crashes}). Task bars: descriptive 95% Wilson intervals.\nNew states and changed configuration: not a paired gain over the original 58/100 grid.\nObserve-only; no Sentinel intervention; not the published SO100 overlay or hardware performance.", fontsize=10, color="#42556b")
    fig.subplots_adjust(left=.075, right=.98, top=.76, bottom=.23)
    outputs = [args.output_dir / f"full-suite-confirmation.{suffix}" for suffix in ("svg", "png")]
    for path in outputs:
        fig.savefig(path, dpi=160)
    plt.close(fig)
    helper = Path(__file__).with_name("plot_libero_results.py")
    receipt = {
        "schema": "sentinel-libero-confirmation-figure-v1",
        "script_sha256": digest(Path(__file__)),
        "helper_sha256": digest(helper),
        "render_runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "inputs": [{"path": str(args.manifest), "sha256": digest(args.manifest)}],
        "outputs": [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in outputs],
        "successes": total,
        "denominator": 100,
        "task_success_counts": counts,
        "no_policy_or_simulator_execution": True,
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
