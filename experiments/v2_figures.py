#!/usr/bin/env python3
"""Figures for the v2 pilot results (optional; needs matplotlib).

usage: python experiments/v2_figures.py <pilot_out_dir> <fig_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- F1 numeric streaming
sys.path.insert(0, str(Path(__file__).resolve().parent / "delta_v2"))
from summarize import summarize  # noqa: E402

rows = summarize(json.loads((src / "numeric_v2.json").read_text()))
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
for ax, jit in zip(axes, (0.0005, 0.003)):
    for H, mk in ((16, "o"), (40, "s"), (128, "^")):
        sel = sorted((r for r in rows if r["H"] == H and r["jitter"] == jit), key=lambda r: r["M"])
        ax.plot([r["M"] for r in sel], [r["speedup_median_vs_best_full"] for r in sel], marker=mk, label=f"H={H}")
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("obstacles M")
    ax.set_title(f"per-cycle jitter {jit} m")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("speed-up of Δ-Cert v2 (root amortized)\nover the faster full check, median")
axes[0].legend(frameon=False)
fig.savefig(dst / "F1_numeric_streaming_speedup.png")
plt.close(fig)

# ---------------------------------------------------------------- F2 pipeline Pareto
pipe = json.loads((src / "pipeline_ticks.json").read_text())
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for mode, mk, col in (("stop_and_go", "o", "tab:gray"), ("pipelined", "s", "tab:blue")):
    xs, ys, ls = [], [], []
    for K in (1, 2, 4, 8):
        r = pipe[f"cap=2,K={K}"][mode]
        xs.append(r["auth_age_ms_mean"]); ys.append(r["ticks"]); ls.append(K)
    ax.plot(xs, ys, marker=mk, color=col, label=mode.replace("_", "-"))
    for x, y, K in zip(xs, ys, ls):
        ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=col)
ax.set_xlim(0, 620)
ax.set_xlabel("mean authorization age at write (ms)")
ax.set_ylabel("control cycles for 40 steps")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
fig.savefig(dst / "F2_pipeline_pareto.png")
plt.close(fig)

# ---------------------------------------------------------------- F3 fault matrix
pat = json.loads((src / "evc_patterns.json").read_text())
names = list(pat)
faults = list(pat[names[0]])
fig, ax = plt.subplots(figsize=(5.6, 5.0))
grid = [[1 if pat[n][f].startswith("BLOCKED") else 0 for n in names] for f in faults]
ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=7)
ax.set_yticks(range(len(faults)))
ax.set_yticklabels(faults, fontsize=7)
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, "blocked" if v else "ADMITTED", ha="center", va="center", fontsize=6)
ax.set_title(f"{len(faults)} injected contract faults × gating pattern (one constructed instance each)", fontsize=9)
fig.savefig(dst / "F3_fault_matrix.png")
plt.close(fig)

# ---------------------------------------------------------------- F4 arm streaming
arm = json.loads((src / "arm_pilot.json").read_text())["C_streaming"]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
meth = (("waypoints_only", "waypoints only (not continuous)"), ("discrete_0.01", "discrete 0.01 rad"),
        ("certified_full", "certified, from scratch"), ("delta_tight=inf", "Δ-Cert-arm"))
jits = list(arm)
for key, lab in meth:
    axes[0].plot([float(j) for j in jits], [arm[j]["methods"][key]["fk_mean"] for j in jits], marker="o", label=lab)
    axes[1].plot([float(j) for j in jits], [arm[j]["methods"][key]["ms_mean"] for j in jits], marker="o", label=lab)
axes[0].set_ylabel("FK evaluations per cycle (mean)")
axes[1].set_ylabel("wall-clock per cycle, ms (mean, numpy)")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([float(j) for j in jits])
    ax.set_xticklabels(jits)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("per-cycle joint jitter (rad)")
    ax.grid(alpha=0.3, which="major")
axes[0].legend(frameon=False, fontsize=7)
fig.savefig(dst / "F4_arm_streaming.png")
plt.close(fig)
# ---------------------------------------------------------------- F5 multi-candidate
cand_path = src / "arm_candidates.json"
if cand_path.exists():
    cand = json.loads(cand_path.read_text())["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for ax, jit in zip(axes, (0.002, 0.01)):
        rows = [c for c in cand if c["jitter"] == jit]
        Ns = [c["N"] for c in rows]
        for key, lab in (("discrete_0.01", "discrete 0.01 rad"), ("certified_full", "certified, from scratch"),
                         ("delta_arm", "Δ-Cert-arm (+ amortized root)")):
            ys = [c["methods"][key]["ms_median"] + (c["root_certificate_ms_per_cycle_amortized"] if key == "delta_arm" else 0)
                  for c in rows]
            ax.plot(Ns, ys, marker="o", label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("candidate chunks verified per cycle N")
        ax.set_title(f"joint jitter {jit} rad")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("wall-clock per cycle, ms (median)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(dst / "F5_arm_multicandidate.png")
    plt.close(fig)
print("figures ->", dst)
