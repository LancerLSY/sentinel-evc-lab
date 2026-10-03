#!/usr/bin/env python3
"""Plot all frozen ARM cells from the retained root-cluster summaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs=2, required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    cells = {}
    for path in args.summaries:
        cells.update(json.loads(path.read_text())['cells'])
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    counts = [1, 4, 8, 32, 128]
    colors = {0.002: '#2379b5', 0.01: '#dc7b26'}
    methods = {'discrete_0.01': ('Discrete 0.01 rad', '#9c5252'),
               'certified_full': ('Full continuous', '#667283'),
               'delta_arm': ('Delta + certificate cost', '#167c70')}
    for jitter, offset in [(0.002, -.09), (0.01, .09)]:
        rows = [cells[f'N{n}-j{jitter}'] for n in counts]
        scores = [r['speedup_full_over_delta'] for r in rows]
        points = np.array([s['point'] for s in scores])
        low = np.array([s['paired_root_cluster_bootstrap95'][0] for s in scores])
        high = np.array([s['paired_root_cluster_bootstrap95'][1] for s in scores])
        x = np.arange(len(counts)) + offset
        axes[0].errorbar(x, points, yerr=np.stack([points-low, high-points]),
                         fmt='o', capsize=4, color=colors[jitter],
                         label=f'Change {jitter:.3f} rad')
        for method, (label, color) in methods.items():
            axes[1].plot(np.arange(len(counts)), [r['median_ms'][method] for r in rows],
                         'o-' if jitter == .002 else 's--', color=color,
                         label=label if jitter == .002 else None, alpha=.9)
    axes[0].axhline(1, color='#5e6874', linewidth=1, linestyle='--')
    axes[0].set_title('A. Speedup vs full continuous checking')
    axes[0].set_ylabel('Full / delta latency (95% root-cluster CI)')
    axes[0].set_ylim(0, 5.6)
    axes[0].legend(frameon=False, loc='upper left')
    axes[1].set_title('B. Verification latency, all methods')
    axes[1].axhline(50, color='#5e6874', linestyle=':', linewidth=1)
    axes[1].text(.05, 53, '50 ms cycle budget', color='#5e6874', fontsize=10)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Median verification latency (ms)')
    axes[1].legend(frameon=False, loc='upper left', fontsize=9)
    for ax in axes:
        ax.set_xticks(np.arange(len(counts)), counts)
        ax.set_xlabel('Candidates per cycle')
        ax.grid(axis='y', alpha=.15)
    axes[1].set_xlabel('Candidates per cycle\nSolid: change 0.002 rad | Dashed: change 0.010 rad')
    fig.suptitle('UR5e static geometry | Xeon CPU | 12 roots, 3 technical repeats', fontsize=13)
    fig.savefig(args.out / 'arm_speedup.png', dpi=220)
    fig.savefig(args.out / 'arm_speedup.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
