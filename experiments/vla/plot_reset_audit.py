#!/usr/bin/env python3
"""Show the first ordered reset pair from each task, with retained RGB only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison_path = args.audit_dir / "comparison/comparison.json"
    comparison = json.loads(comparison_path.read_text())
    if comparison["status"] != "complete" or comparison["paired_reset_count"] != 20:
        raise ValueError("Expected the complete frozen 20-pair audit")
    inputs = [{"path": str(comparison_path), "sha256": digest(comparison_path)}]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.8))
    for column, (label, version) in enumerate([("current", "3.8.1"), ("legacy", "3.3.7")]):
        path = args.audit_dir / label / "manifest.json"
        manifest = json.loads(path.read_text())
        if digest(path) != comparison[label + "_manifest"]["sha256"]:
            raise ValueError("Audit manifest binding mismatch")
        if manifest["software"]["mujoco"] != version:
            raise ValueError("Unexpected actual MuJoCo version")
        inputs.append({"path": str(path), "sha256": digest(path)})
        for row, task in enumerate([0, 5]):
            record = next(x for x in manifest["records"] if x["task_id"] == task and x["initial_state_index"] == 10)
            receipt = record["camera_artifact"]
            source = path.parent / receipt["path"]
            if digest(source) != receipt["sha256"] or source.stat().st_size != receipt["bytes"]:
                raise ValueError("Camera artifact mismatch")
            inputs.append({"path": str(source), "sha256": digest(source)})
            with np.load(source, allow_pickle=False) as cameras:
                pixels = cameras["agentview_image"]
                if pixels.dtype != np.uint8 or pixels.shape != (360, 360, 3):
                    raise ValueError("Unexpected retained RGB layout")
                axes[row, column].imshow(pixels)
            axes[row, column].axis("off")
            axes[row, column].set_title(f"Task {task} · state 10 · MuJoCo {version}", pad=9)
    fig.suptitle("Version-dependent initial state, before any policy action", fontsize=17, fontweight="bold", y=.97)
    fig.text(.5, .495, "Task 0 control: mean bowl shift 0.688 mm over 10 pairs", ha="center", color="#42556b")
    fig.text(.5, .09, "Task 5 (bowl on ramekin): bowl shift 35.327 mm in all 10 pairs", ha="center", color="#b84943", fontweight="bold")
    fig.text(.5, .045, "Same official assets/init files and paired seeds. Original 360×360 RGB; no generated scene detail.\nImages illustrate the first ordered formal state, not additional success trials or a causal policy diagnosis.", ha="center", fontsize=10, color="#42556b")
    fig.subplots_adjust(left=.04, right=.96, top=.89, bottom=.13, hspace=.25, wspace=.08)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for suffix in ("png", "svg"):
        path = args.output_dir / f"reset-version-comparison.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        outputs.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
    plt.close(fig)
    receipt = {
        "schema": "sentinel-libero-reset-figure-v1",
        "script_sha256": digest(Path(__file__)),
        "selection": "First ordered formal initial state 10 for each declared task 0 and 5",
        "presentation_only": True,
        "inputs": inputs,
        "outputs": outputs,
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
