#!/usr/bin/env python3
"""Plot retained native-checkpoint results without rerunning any policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - spread, center + spread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--replan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    replan = json.loads(args.replan.read_text())
    if native["status"] != "complete" or replan["status"] != "complete":
        raise ValueError("Only complete, retained formal studies may be plotted")
    if native["results"]["episodes"] != 100 or replan["episodes_accounted"] != 40:
        raise ValueError("Unexpected frozen denominators")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.patch.set_facecolor("white")
    counts = [native["results"]["tasks"][str(task)]["successes"] for task in range(10)]
    proportions = np.array(counts) / 10
    intervals = np.array([wilson(count, 10) for count in counts])
    errors = np.maximum(0, np.stack([proportions - intervals[:, 0], intervals[:, 1] - proportions]))
    colors = ["#d9534f" if task == 5 else "#4479b7" for task in range(10)]
    axes[0].bar(range(10), 100 * proportions, color=colors, width=.7)
    axes[0].errorbar(range(10), 100 * proportions, yerr=100 * errors, fmt="none", ecolor="#35465a", capsize=3)
    for task, count in enumerate(counts):
        axes[0].text(task, 100 * intervals[task, 1] + 2, f"{count}/10", ha="center", fontsize=10)
    axes[0].set_xticks(range(10), [str(task) for task in range(10)])
    axes[0].set_xlabel("LIBERO-Spatial task ID (states 0–9)")
    axes[0].set_ylabel("Task success (%)")
    axes[0].set_title("A. Original fixed-state evaluation: 58/100", loc="left", fontsize=13, fontweight="bold")
    positions = np.arange(2)
    for branch_index, (branch, label, color) in enumerate([
        ("execute_50", "Reobserve every 50 actions", "#8698aa"),
        ("execute_10", "Reobserve every 10 actions", "#228b78"),
    ]):
        branch_counts = [replan["metrics"][str(task)]["branches"][branch]["successes"] for task in [0, 5]]
        x = positions + (branch_index - .5) * .34
        axes[1].bar(x, np.array(branch_counts) * 10, width=.32, color=color, label=label)
        for xx, count in zip(x, branch_counts):
            axes[1].text(xx, count * 10 + 3, f"{count}/10", ha="center", fontsize=10)
    axes[1].set_xticks(positions, ["Task 0\ncontrol", "Task 5\nbowl on ramekin"])
    axes[1].set_title("B. Matched horizon ablation", loc="left", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Fresh states 10–19; 20 pairs / 40 rollouts")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=10)
    for ax in axes:
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8edf2")
        ax.set_axisbelow(True)
    fig.suptitle("Official SmolVLA checkpoint · native LIBERO/Panda · MuJoCo 3.8.1", fontsize=16, fontweight="bold")
    fig.text(.06, .04, "Panel A: descriptive 95% Wilson intervals on fixed task/state sets.\nPanel B: same weights, seeds, initial observations and first 10 predicted actions; 280-step cap.\nNo Sentinel intervention. These rollouts do not evaluate the published SO100 overlay.", fontsize=10, color="#42556b")
    fig.subplots_adjust(left=.065, right=.98, top=.82, bottom=.29, wspace=.22)
    paths = [args.output_dir / f"native-policy-and-horizon.{suffix}" for suffix in ["svg", "png"]]
    for path in paths:
        fig.savefig(path, dpi=160)
    plt.close(fig)
    receipt = {
        "schema": "sentinel-native-libero-figure-v1",
        "script_sha256": digest(Path(__file__)),
        "inputs": [{"path": str(path), "sha256": digest(path)} for path in [args.native, args.replan]],
        "outputs": [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths],
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
