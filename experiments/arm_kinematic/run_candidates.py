#!/usr/bin/env python3
"""Multi-candidate verification per control cycle (pilot, not frozen).

Every cycle the policy proposes N candidate chunks (shifted committed plan +
independent smooth perturbations).  All N must be verified before one is chosen.

Methods (each verifies all N candidates in one numpy batch):
  discrete_0.01   sampled check at 0.01 rad (common practice; no bound between samples)
  certified_full  continuous certified check from scratch
  delta_arm       Δ-Cert-arm: inherit from the committed plan's certificate, re-certify
                  only segments whose inherited bound is not positive (the one-off root
                  certificate per task is timed separately and reported amortized)

Reported per cell: wall-clock per cycle (median / p95), FK evaluations per cycle,
per-candidate decision agreement with certified_full, and the effect of choosing
among accepted candidates by certified margin (only the certified methods produce
a margin): true minimum clearance of the chosen chunk, measured by a dense
0.002 rad sweep.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from arm_delta import _issue_arm_cert, certify_full, certify_full_batch, discrete_batch, inherit, inherit_batch, scene_digest  # noqa: E402
from run_arm_pilot import GOAL_LO, place_obstacles  # noqa: E402
from scenarios import OBSTACLE_RADIUS, interpolate  # noqa: E402
from ur5e_kin import UR5eModel, densify  # noqa: E402


def true_min_clearance(m, plan, obs):
    o, s = m.clearances(densify(plan, 0.002), obs, OBSTACLE_RADIUS)
    return float(min(o.min(), s.min() if s.size else np.inf))


def cell(m, N, jitter, roots, cycles, seed, check_single=False):
    rng = np.random.default_rng(seed)
    t = {"discrete_0.01": [], "certified_full": [], "delta_arm": []}
    fk = {k: [] for k in t}
    agree = {"discrete_0.01": 0, "delta_arm": 0}
    missed_by_discrete = 0
    cand_total = 0
    seg = np.zeros(3, dtype=int)
    budget = {"certified_full_cycles": 0, "certified_full_candidates": 0,
              "delta_arm_cycles": 0, "delta_arm_candidates": 0}
    choice = {"by_certified_margin": [], "uniform_among_accepted": []}
    single_mismatch = 0
    root_ms = []
    for root in range(roots):
        goal = m.home + rng.uniform(GOAL_LO, -GOAL_LO)
        base = interpolate(m.home, goal, 41)
        obs = place_obstacles(m, base, rng)
        r0 = time.perf_counter()
        cert = certify_full(m, base, obs, OBSTACLE_RADIUS).cert
        root_ms.append((time.perf_counter() - r0) * 1e3)
        plan, offset = base, 0
        for c in range(cycles):
            offset += 1
            proposal = np.concatenate([plan[offset:], np.repeat(plan[-1:], offset, 0)])
            d = 4
            w = np.clip((np.arange(41) - d) / (40 - d), 0, 1)[None, :, None]
            kids = proposal[None] + w * np.cumsum(rng.normal(0, jitter, (N, 41, 6)), 1) * 0.3
            kids[:, :d] = proposal[None, :d]
            t0 = time.perf_counter(); ok_d, fk_d = discrete_batch(m, kids, obs, OBSTACLE_RADIUS, 0.01); t1 = time.perf_counter()
            ok_f, lo_f, ls_f, fk_f, meta_f = certify_full_batch(
                m, kids, obs, OBSTACLE_RADIUS, return_meta=True); t2 = time.perf_counter()
            parent_cert = cert
            ok_x, lo_x, ls_x, fk_x, sg, meta_x = inherit_batch(
                m, parent_cert, kids, obs, OBSTACLE_RADIUS, offset, return_meta=True)
            budget["certified_full_cycles"] += int(meta_f["budget_exhausted"])
            budget["certified_full_candidates"] += meta_f["unknown_candidates"]
            budget["delta_arm_cycles"] += int(meta_x["budget_exhausted"])
            budget["delta_arm_candidates"] += meta_x["unknown_candidates"]
            acc = np.where(ok_x)[0]
            next_cert = None
            if len(acc):
                margin = np.minimum(lo_x[acc].min(axis=(1, 2)), ls_x[acc].min(axis=(1, 2)) if ls_x.shape[2] else np.inf)
                best = acc[int(np.argmax(margin))]
                next_cert = _issue_arm_cert(m, kids[best], scene_digest(obs, OBSTACLE_RADIUS),
                                            lo_x[best], ls_x[best], parent_cert.depth + 1)
            t3 = time.perf_counter()
            t["discrete_0.01"].append((t1 - t0) * 1e3); t["certified_full"].append((t2 - t1) * 1e3); t["delta_arm"].append((t3 - t2) * 1e3)
            fk["discrete_0.01"].append(fk_d); fk["certified_full"].append(fk_f); fk["delta_arm"].append(fk_x)
            agree["discrete_0.01"] += int(np.sum(ok_d == ok_f)); agree["delta_arm"] += int(np.sum(ok_x == ok_f))
            missed_by_discrete += int(np.sum(ok_d & ~ok_f))
            cand_total += N
            seg += np.array(sg)
            if check_single and N <= 8:
                for i in range(N):
                    single_mismatch += int(inherit(m, parent_cert, kids[i], obs, OBSTACLE_RADIUS, offset).ok != ok_x[i])
            if len(acc):
                uni = acc[int(rng.integers(len(acc)))]
                if len(acc) > 1:
                    choice["by_certified_margin"].append(true_min_clearance(m, kids[best], obs))
                    choice["uniform_among_accepted"].append(true_min_clearance(m, kids[uni], obs))
                plan, offset, cert = kids[best], 0, next_cert
    pct = lambda a, q: float(np.percentile(np.array(a), q))
    out = {"N": N, "jitter": jitter, "cycles": roots * cycles, "candidates": cand_total,
           "segments_reused_inherited_rechecked": seg.tolist(),
           "root_certificate_ms_mean": float(np.mean(root_ms)),
           "root_certificate_ms_per_cycle_amortized": float(np.sum(root_ms) / (roots * cycles)),
           "max_fk_evals_per_checker": 200000, "budget_unknown": budget,
           "agreement_with_certified_full": {k: f"{v}/{cand_total}" for k, v in agree.items()},
           "discrete_allows_that_certified_rejects": missed_by_discrete,
           "methods": {k: {"ms_median": pct(t[k], 50), "ms_p95": pct(t[k], 95), "fk_mean": float(np.mean(fk[k]))} for k in t},
           "choice_true_min_clearance_m": {k: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                                               "p5": pct(v, 5) if v else None} for k, v in choice.items()}}
    if check_single:
        out["batch_vs_single_mismatches"] = single_mismatch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True, help="mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=20)
    args = ap.parse_args()
    m = UR5eModel(Path(args.mjcf))
    res = {"schema": "sentinel-arm-candidates-pilot-v1", "python": platform.python_version(), "numpy": np.__version__,
           "note": "pilot, not frozen; static capsule geometry; single CPU thread numpy", "cells": []}
    for jit in (0.002, 0.01):
        for N in (1, 8, 32, 128):
            r = cell(m, N, jit, args.roots, args.cycles, 1000 + N + int(jit * 1e4), check_single=(N <= 8))
            res["cells"].append(r)
            md = r["methods"]
            print(f"jit={jit} N={N:3d} | ms med disc {md['discrete_0.01']['ms_median']:.2f} full {md['certified_full']['ms_median']:.2f} "
                  f"delta {md['delta_arm']['ms_median']:.2f} | fk {md['discrete_0.01']['fk_mean']:.0f}/{md['certified_full']['fk_mean']:.0f}/{md['delta_arm']['fk_mean']:.0f}"
                  f" | agree {r['agreement_with_certified_full']} missed_disc {r['discrete_allows_that_certified_rejects']}"
                  f" | choice {r['choice_true_min_clearance_m']['by_certified_margin']['mean']} vs {r['choice_true_min_clearance_m']['uniform_among_accepted']['mean']}"
                  + (f" | single-mismatch {r.get('batch_vs_single_mismatches')}" if 'batch_vs_single_mismatches' in r else ""), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
