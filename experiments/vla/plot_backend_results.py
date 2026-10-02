#!/usr/bin/env python3
"""Plot the frozen paired MuJoCo backend compatibility results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison_path = args.backend_dir / "comparison/comparison.json"
    paired_path = args.backend_dir / "comparison/paired_episodes.json"
    current_path = args.backend_dir / "current/manifest.json"
    legacy_path = args.backend_dir / "legacy/manifest.json"
    comparison = load_object(comparison_path)
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    current = load_object(current_path)
    legacy = load_object(legacy_path)
    protocol = load_object(args.protocol)

    if comparison.get("status") != "complete" or comparison.get("paired_episode_count") != 20:
        raise ValueError("Expected the complete frozen 20-pair backend comparison")
    if not isinstance(paired, list) or len(paired) != 20:
        raise ValueError("Expected exactly 20 paired backend rows")
    if digest(paired_path) != comparison["paired_episodes"]["sha256"]:
        raise ValueError("Paired-row digest mismatch")
    for label, path, manifest, version in (
        ("current", current_path, current, "3.8.1"),
        ("legacy", legacy_path, legacy, "3.3.7"),
    ):
        if digest(path) != comparison[f"{label}_manifest"]["sha256"]:
            raise ValueError(f"{label} manifest digest mismatch")
        if manifest.get("status") != "complete" or manifest.get("episodes_accounted") != 20:
            raise ValueError(f"{label} capture is incomplete")
        if manifest["software"]["mujoco"] != version:
            raise ValueError(f"{label} MuJoCo version mismatch")
        if manifest["script_sha256"] != digest(args.runner):
            raise ValueError(f"{label} runner binding mismatch")
        if manifest["protocol_sha256"] != digest(args.protocol):
            raise ValueError(f"{label} protocol binding mismatch")
    if current["model_artifacts"] != legacy["model_artifacts"]:
        raise ValueError("Non-treatment model artifacts differ")
    if current["asset_tree"] != legacy["asset_tree"]:
        raise ValueError("Non-treatment asset tree differs")
    if protocol["evaluation"]["task_ids"] != [5, 0]:
        raise ValueError("Unexpected frozen task order")

    metrics = comparison["metrics"]
    task_order = [5, 0]
    for task_id in task_order:
        rows = [row for row in paired if row["task_id"] == task_id]
        if len(rows) != 10:
            raise ValueError(f"Task {task_id} denominator is not 10")
        observed = metrics[str(task_id)]
        if sum(row["current_success"] for row in rows) != observed["current_successes"]:
            raise ValueError("Current success count mismatch")
        if sum(row["legacy_success"] for row in rows) != observed["legacy_successes"]:
            raise ValueError("Legacy success count mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5), gridspec_kw={"width_ratios": [1.28, 1]})
    fig.patch.set_facecolor("white")

    x = np.arange(2)
    width = 0.34
    series = [
        ("current_successes", "MuJoCo 3.8.1", "#4e79a7"),
        ("legacy_successes", "MuJoCo 3.3.7", "#e07a3f"),
    ]
    for index, (key, label, color) in enumerate(series):
        counts = np.array([metrics[str(task)][key] for task in task_order])
        positions = x + (index - 0.5) * width
        axes[0].bar(positions, counts, width=width * 0.92, color=color, label=label)
        for position, count in zip(positions, counts):
            axes[0].text(position, count + 0.28, f"{count}/10", ha="center", fontweight="bold")
    axes[0].set_xticks(x, ["Task 5\nbowl on ramekin", "Task 0\ncontrol"])
    axes[0].set_ylim(0, 11.6)
    axes[0].set_yticks(range(0, 11, 2))
    axes[0].set_ylabel("Successful rollouts (of 10)")
    axes[0].set_title("A. Fixed-state success counts", loc="left", fontsize=13)
    axes[0].legend(frameon=False, loc="upper left")

    deltas = np.array([metrics[str(task)]["paired_delta_legacy_minus_current"] for task in task_order])
    intervals = np.array([metrics[str(task)]["paired_bootstrap_95"] for task in task_order])
    errors = np.stack([deltas - intervals[:, 0], intervals[:, 1] - deltas])
    colors = ["#c7534f", "#5b7796"]
    axes[1].axvline(0, color="#586675", linewidth=1.2)
    for row, task_id in enumerate(task_order):
        axes[1].errorbar(
            deltas[row] * 100,
            row,
            xerr=np.array([[errors[0, row] * 100], [errors[1, row] * 100]]),
            fmt="o",
            markersize=9,
            color=colors[row],
            ecolor=colors[row],
            elinewidth=2.4,
            capsize=5,
        )
        low, high = 100 * intervals[row]
        label_x = 108 if high >= 90 else high + 5
        axes[1].text(
            label_x,
            row + (0.10 if high >= 90 else 0),
            f"{100 * deltas[row]:+.0f} pp  [{low:+.0f}, {high:+.0f}]",
            ha="right" if high >= 90 else "left",
            va="center",
            fontsize=10,
            color="#26384a",
        )
    axes[1].set_yticks([0, 1], ["Task 5", "Task 0"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-12, 112)
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xlabel("Paired success difference (3.3.7 − 3.8.1), percentage points")
    axes[1].set_title("B. Descriptive paired 95% bootstrap interval", loc="left", fontsize=13)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e8edf2")
        axis.set_axisbelow(True)

    fig.suptitle(
        "SmolVLA LIBERO backend compatibility · complete MuJoCo version treatment",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.885,
        "20 matched task/state pairs total · n=10 per task · execution horizon 10 · RTX 4090 D 24GB",
        ha="center",
        fontsize=11,
        color="#42556b",
    )
    fig.text(
        0.055,
        0.03,
        "Treatment changes the complete MuJoCo package version while checkpoint, assets, init-state index, seed, and policy settings remain fixed.\n"
        "Intervals are descriptive for these paired states. The comparison does not isolate a reset mechanism or identify either version as physically more accurate.",
        fontsize=9.5,
        color="#42556b",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.23, wspace=0.30)

    outputs = []
    for suffix in ("png", "svg"):
        path = args.output_dir / f"backend-version-success.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        outputs.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    plt.close(fig)

    manifest = {
        "schema": "sentinel-libero-backend-figure-v1",
        "script": receipt(Path(__file__)),
        "sources": [receipt(args.runner), receipt(args.protocol)],
        "inputs": [
            receipt(comparison_path),
            receipt(paired_path),
            receipt(current_path),
            receipt(legacy_path),
        ],
        "study": {
            "gpu_label": "RTX 4090 D 24GB",
            "paired_cells": 20,
            "pairs_per_task": 10,
            "tasks": task_order,
            "treatment": "complete MuJoCo package version: 3.3.7 versus 3.8.1",
            "interval": "descriptive paired bootstrap 95%, 10,000 samples",
            "claim_boundary": "No isolated reset-mechanism or physical-accuracy claim.",
        },
        "outputs": outputs,
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
