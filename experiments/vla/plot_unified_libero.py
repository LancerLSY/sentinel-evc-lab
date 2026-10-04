#!/usr/bin/env python3
"""Plot retained unified-study outcomes and descriptive component costs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.run_dir / "manifest.json").read_text())
episodes = [r for r in manifest["results"] if r["phase"] == "formal"]
branches = ["parent_only", "full_final", "delta_evc"]
tasks = manifest["config"]["formal"]["task_ids"]
rows = [[next((r for r in episodes if r["task_id"] == t and r["state_index"] == s and r["branch"] == b), None)
         for b in branches] for t in tasks for s in manifest["config"]["formal"]["initial_state_indices"]]
matrix = np.array([[(-1 if r is None else 2 if r.get("success") else 1 if r.get("denied_candidates", 0) else 0)
                    for r in row] for row in rows])
labels = [f"task {t}, state {s}" for t in tasks for s in manifest["config"]["formal"]["initial_state_indices"]]
candidates = []
for line in (args.run_dir / "steps.jsonl").open():
    r = json.loads(line)
    if r.get("event") == "candidate" and r["episode_id"].startswith("formal-") and r["episode_id"].endswith("delta_evc"):
        candidates.append(r)
full = np.array([r["timing_ns"]["full_geometry"] / 1e6 for r in candidates])
inc = np.array([r["timing_ns"]["incremental_geometry"] / 1e6 for r in candidates])
total = np.array([(r["timing_ns"]["parent_geometry"] + r["timing_ns"]["incremental_geometry"]) / 1e6 for r in candidates])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
from matplotlib.colors import ListedColormap, BoundaryNorm
colors = ["#d7dce3", "#d7534f", "#e7ad42", "#208876"]
axes[0].imshow(matrix, cmap=ListedColormap(colors), norm=BoundaryNorm([-1.5, -.5, .5, 1.5, 2.5], 4), aspect="auto")
axes[0].set_xticks(range(3), ["Parent only", "Full final", "Delta + EVC"])
axes[0].set_yticks(range(len(labels)), labels)
axes[0].set_title("A  Matched-root outcomes", loc="left", fontweight="bold")
names = {-1:"unfinished", 0:"failed", 1:"stopped", 2:"success"}
for i in range(matrix.shape[0]):
    for j in range(3):
        axes[0].text(j, i, names[int(matrix[i,j])], ha="center", va="center", fontsize=9,
                     color="white" if matrix[i,j] in (0,2) else "#242c36")
if len(candidates):
    x = np.arange(3)
    medians = [np.median(a) for a in (full, inc, total)]
    tails = [np.percentile(a, 95, method="higher") for a in (full, inc, total)]
    axes[1].bar(x-.18, medians, width=.36, label="Median", color="#208876")
    axes[1].bar(x+.18, tails, width=.36, label="P95", color="#375779")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ["Full final", "Incremental\nonly", "Parent +\nincremental"])
    for positions, values in ((x-.18, medians), (x+.18, tails)):
        for p, v in zip(positions, values):
            axes[1].text(p, v*1.12, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].legend(frameon=False, loc="upper left")
axes[1].set_ylabel("Measured geometry component cost (ms, log scale)")
axes[1].set_title(f"B  Component costs ({len(candidates)} candidates)", loc="left", fontweight="bold")
axes[1].spines[["top", "right"]].set_visible(False)
fig.suptitle("SmolVLA / Panda: unified geometry and execution authorization", fontsize=13, fontweight="bold")
fig.text(.5, -.045, "Incremental-only omits parent construction. Instrumented diagnostic costs are not deployment throughput.", ha="center", fontsize=9)
args.output_dir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    path = args.output_dir / f"unified-outcomes-and-costs.{ext}"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
print(json.dumps({"formal_episodes":len(episodes),"delta_candidates":len(candidates),"output":str(args.output_dir)}))
