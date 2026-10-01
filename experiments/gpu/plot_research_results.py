"""Render publication figures from saved metrics; never rerun or select experiments."""
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


def wilson(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, min(p, center - half)), min(1.0, max(p, center + half))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldguard", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white",
                         "svg.hashsalt": "sentinel-saved-metrics"})
    colors = ["#94a3b8", "#64748b", "#0284c7", "#0f766e"]
    outputs = []

    def save(fig, name: str, footer: str) -> None:
        fig.text(.02, .01, footer, fontsize=9, color="#475569")
        fig.tight_layout(rect=(0, .07, 1, .97))
        for extension in ("png", "svg"):
            path = args.out / f"{name}.{extension}"
            fig.savefig(path, dpi=180, metadata={"Creator": "Sentinel saved-metrics plotter", "Date": None}
                        if extension == "svg" else {"Software": "Sentinel saved-metrics plotter"})
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs.append({"file": path.name, "bytes": path.stat().st_size,
                            "sha256": digest(path)})
        plt.close(fig)

    wg = json.loads(args.worldguard.read_text())["scenarios"]
    keys = ["nominal", "low_friction", "mass_low", "mass_high", "camera_shift", "displacement_shift"]
    labels = ["Nominal", "Low friction", "Low mass", "High mass", "Camera shift", "1.35x move\n(out of contract)"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    methods = ["geometry_only_fastest_0p6", "fixed_1p6", "state_worldguard", "image_worldguard"]
    names = ["Fast geometry baseline", "Fixed 1.6 s", "State WorldGuard", "Image WorldGuard"]
    x = np.arange(len(keys))
    for i, (method, name) in enumerate(zip(methods, names)):
        policies = [wg[k][method].get("policy", wg[k][method]) for k in keys]
        counts = [p["unsafe_selected"] for p in policies]
        n = [p["roots"] for p in policies]
        rates = np.array(counts) / n
        intervals = [wilson(c, nn) for c, nn in zip(counts, n)]
        errors = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(rates, intervals)]).T * 100
        bars = axes[0].bar(x + (i - 1.5) * .18, rates * 100, .18, color=colors[i], label=name,
                           yerr=errors, capsize=2, error_kw={"elinewidth": .8})
        bars[-1].set_hatch("///")
        for index, (bar, c, nn) in enumerate(zip(bars, counts, n)):
            if keys[index] == "low_friction":
                if i == 1:
                    axes[0].text(x[index], 110, "All four: 100/100", ha="center", fontsize=10)
            elif c:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                             f"{c}/{nn}", ha="center", fontsize=8)
    axes[0].set(title="WorldGuard: failures remain visible across six fresh scenarios",
                ylabel="Unsafe selections / all roots (%)", ylim=(0, 119))
    axes[0].legend(ncol=2, loc="upper right", fontsize=10)
    for i, (method, name, color) in enumerate(zip(methods[2:], names[2:], colors[2:])):
        values = [100 * wg[k][method]["xy_joint_root_coverage"]["value"] for k in keys]
        axes[1].plot(x, values, "o-", color=color, label=name)
        for xx, value, key in zip(x, values, keys):
            if i and value == 100 * wg[key][methods[2]]["xy_joint_root_coverage"]["value"]:
                continue
            axes[1].text(xx + (i - .5) * .14, max(4, value + (4 if i == 0 else -8)),
                         f"{value:.0f}%", fontsize=9, color=color)
    axes[1].axhline(95, color="#dc2626", linestyle="--", linewidth=1, label="Nominal 95% target")
    axes[1].set(ylabel="Joint XY root coverage (%)", ylim=(-3, 108), xticks=x, xticklabels=labels)
    axes[1].legend(ncol=3, loc="lower right", fontsize=9)
    for ax in axes:
        ax.axvspan(4.5, 5.5, color="#f1f5f9", alpha=.6, zorder=-1)
        ax.grid(axis="y", alpha=.15)
    save(fig, "worldguard-scenarios", "100 fresh roots/scenario. Error bars: 95% Wilson intervals for sampled root counts.\n"
         "Hatched displacement results are bare-model diagnostics; the product action contract rejects all 100. RTX 4090 D measured.")

    fallback = json.loads(args.fallback.read_text())["metrics"]
    durations = ["1.6", "3.2", "4.8"]
    counts = [fallback["candidate_label_counts"][d]["unsafe_forecast"] for d in durations]
    rates = np.array(counts, dtype=float)
    intervals = [wilson(c, 100) for c in counts]
    errors = np.array([[v - lo * 100, hi * 100 - v] for v, (lo, hi) in zip(rates, intervals)]).T
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    bars = axes[0].bar(durations, rates, color=["#dc2626", "#64748b", "#0f766e"],
                       yerr=errors, capsize=4)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count + 6, f"{count}/100", ha="center")
    axes[0].set(title="Low-friction outcomes", ylabel="Unsafe candidate roots (%)",
                xlabel="Candidate motion duration (s)", ylim=(0, 119))
    diagnostics = fallback["plan_bound"]["duration_diagnostics"]
    accelerations = [diagnostics[d]["peak_horizontal_acceleration_m_s2"] for d in durations]
    bars = axes[1].bar(durations, accelerations, color=["#dc2626", "#64748b", "#0f766e"])
    for bar, acceleration in zip(bars, accelerations):
        axes[1].text(bar.get_x() + bar.get_width() / 2, acceleration + .025, f"{acceleration:.3f}", ha="center")
    axes[1].set(title="Frozen profile rule selects 4.8 s", ylabel="Peak target horizontal acceleration (m/s²)",
                xlabel="3.2 s was safe here, but the rule rejected it", ylim=(0, 1))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    save(fig, "low-friction-fallback", "100 new paired roots, separate 5 s profile. Known friction floor mu >= 0.015 is an external assumption.\n"
         "Selected 4.8 s: 100/100 tray tasks, 0 drops; 3x motion duration. Target screening is not a safety proof or a deployed repair.")

    arm_manifest = json.loads(args.arm.read_text())
    arm = arm_manifest["gate_reference_summary"]
    reviewed = json.loads(args.arm_review.read_text())
    methods = ["parent_only", "full", "incremental", "conservative"]
    names = ["Parent only", "Full final", "Incremental", "Reject transforms"]
    values = [reviewed["methods"][m] | {"gate_wall_ms_mean": arm["methods"][m]["gate_wall_ms_mean"]}
              for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for ax, count_key, denom_key, title in [
        (axes[0], "false_allow_count", "false_allow_denominator_unsafe", "False allows among 139 unsafe roots"),
        (axes[1], "false_reject_count", "false_reject_denominator_safe", "False rejects among 41 safe roots")]:
        rates = [100 * v[count_key] / v[denom_key] for v in values]
        bars = ax.bar(names, rates, color=colors)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                    f"{v[count_key]}/{v[denom_key]}", ha="center", fontsize=10)
        ax.set(title=title, ylabel="Root rate (%)", ylim=(0, 111))
    ms = [v["gate_wall_ms_mean"] for v in values]
    bars = axes[2].bar(names, ms, color=colors)
    for bar, value in zip(bars, ms):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value:.1f}", ha="center")
    axes[2].set(title="Mean validation cost, including parent", ylabel="Wall time per root (ms)", ylim=(0, 181))
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=22)
        ax.grid(axis="y", alpha=.15)
    save(fig, "ur5e-comparison", "180 constructed UR5e roots; separately implemented 1 ms / 0.005 rad replay checks collision, joint limits and tracking.\n"
         "106 static-prefix reuses / 74 full fallbacks; full dynamic rollout. Fixed-order timing mixes early exits; no speedup or continuous-safety claim.")
    manifest = {"schema": "sentinel-saved-metrics-figures-v1", "source_sha256": digest(Path(__file__)),
                "inputs": [{"path": str(p), "sha256": digest(p)} for p in (args.worldguard, args.fallback, args.arm, args.arm_review)],
                "outputs": outputs, "selection": "all six scenarios and all four declared arm methods; no new experiment"}
    (args.out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
