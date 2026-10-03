#!/usr/bin/env python3
"""Pilot experiments on the kinematic UR5e verifier (not a frozen protocol).

A. Fidelity: reproduce the 180 stored MuJoCo roots (obstacle placement and
   static labels at 0.01 / 0.005 rad).
B. Accuracy ("graze"): obstacles placed beside the swept capsule path with a
   constructed penetration (unsafe) or clearance (safe) of 0.1-20 mm.  Safe
   candidates are kept only if a depth-24 certification proves them free
   (the count of dropped candidates is reported as dropped_safe_not_certified).
C. Streaming re-verification: RTC-style chunk revision every control cycle;
   compare per-cycle verification work of discrete full re-check, certified
   full re-check and Δ-Cert-arm.
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
from arm_delta import certify_full, inherit  # noqa: E402
from scenarios import OBSTACLE_RADIUS, SCENARIOS, interpolate, make_case  # noqa: E402
from ur5e_kin import BatchedCertifiedChecker, UR5eModel, densify, discrete_check  # noqa: E402

GOAL_LO = np.array([-.55, -.35, -.45, -.35, -.3, -.4])


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t) * 1000


# ----------------------------------------------------------------------------- A
def fidelity(m, review_path):
    rows = json.load(open(review_path))
    err = 0.0
    agree01 = agree005 = 0
    for r in rows:
        c = make_case(m, r["scenario"], r["root"])
        err = max(err, float(np.abs(np.array(r["final_obstacles"]) - c["final_obstacles"]).max()))
        d1 = discrete_check(m, c["final"], c["final_obstacles"], OBSTACLE_RADIUS, 0.01)
        d2 = discrete_check(m, c["final"], c["final_obstacles"], OBSTACLE_RADIUS, 0.005)
        agree01 += d1["ok"] == r["discrete_truth"]["ok"]
        agree005 += d2["ok"] == r["highres_static"]["ok"]
    return {"roots": len(rows), "max_obstacle_position_error_m": err,
            "static_label_agreement_0p01": agree01, "static_label_agreement_0p005": agree005}


# ----------------------------------------------------------------------------- B
def graze_case(m, rng, unsafe, r_obs, speed):
    goal = m.home + speed * rng.uniform(GOAL_LO, -GOAL_LO)
    plan = interpolate(m.home, goal, 41)
    Q = densify(plan, 0.0005)
    j = int(rng.integers(5, len(Q) - 5))
    g = int(rng.choice([2, 4, 5, 6, 7, 8]))
    A, B = m.capsule_segments(Q[j - 1:j + 2])
    t = rng.uniform(0, 1)
    pt = A[1, g] + t * (B[1, g] - A[1, g])
    v = (A[2, g] + t * (B[2, g] - A[2, g])) - (A[0, g] + t * (B[0, g] - A[0, g]))
    axis = B[1, g] - A[1, g]; axis /= np.linalg.norm(axis)
    n = np.cross(axis, v)
    if np.linalg.norm(n) < 1e-9:
        n = np.cross(axis, rng.normal(size=3))
    n = n / np.linalg.norm(n) * rng.choice([-1.0, 1.0])
    depth = float(np.exp(rng.uniform(np.log(0.0001), np.log(0.02))))   # 0.1-20 mm, log-uniform
    gap = -depth if unsafe else depth
    center = pt + n * (m.capsules[g].radius + r_obs + gap)
    return plan, center[None, :], depth


def waypoint_check(m, plan, obs, r_obs):
    """Check only the commanded waypoints (each action of the chunk), not the motion between them."""
    from ur5e_kin import joint_ok
    if not joint_ok(m, plan):
        return {"ok": False, "fk_evals": 0}
    o, s = m.clearances(plan, obs, r_obs)
    return {"ok": bool(o.min() >= 0 and (s.size == 0 or s.min() >= 0)), "fk_evals": len(plan)}


def accuracy(m, n_each, seed):
    rng = np.random.default_rng(seed)
    keys = ["waypoints_only", "discrete_0.05", "discrete_0.01", "discrete_0.005", "certified"]
    out = {"cells": {}}
    for r_obs in (0.055, 0.02, 0.008):
        for speed in (1.0, 2.0):
            res = {k: [] for k in keys}
            labels, depths = [], []
            dropped = 0
            while len(labels) < 2 * n_each:
                unsafe_target = len(labels) % 2 == 0
                plan, obs, depth = graze_case(m, rng, unsafe_target, r_obs, speed)
                if not unsafe_target:
                    deep = BatchedCertifiedChecker(m, obs, r_obs, max_depth=24).check_plan(plan)
                    if deep["status"] != "free":
                        dropped += 1
                        continue
                labels.append(not unsafe_target); depths.append(depth)
                for key in keys:
                    if key == "waypoints_only":
                        r, ms = timed(lambda: waypoint_check(m, plan, obs, r_obs)); fk = r["fk_evals"]
                    elif key.startswith("discrete"):
                        step = float(key.split("_")[1])
                        r, ms = timed(lambda: discrete_check(m, plan, obs, r_obs, step)); fk = r["fk_evals"]
                    else:
                        chk = BatchedCertifiedChecker(m, obs, r_obs, max_depth=14)
                        r, ms = timed(lambda: chk.check_plan(plan)); fk = chk.fk_evals
                    res[key].append((r["ok"], fk, ms))
            labels = np.array(labels); depths = np.array(depths)
            cell = {"unsafe": int((~labels).sum()), "safe": int(labels.sum()), "dropped_safe_not_certified": dropped,
                    "methods": {}}
            for key, rr in res.items():
                ok = np.array([x[0] for x in rr]); fk = np.array([x[1] for x in rr]); ms = np.array([x[2] for x in rr])
                fa = int((ok & ~labels).sum()); fr = int((~ok & labels).sum())
                bins = {}
                for lo, hi in [(0.0001, 0.001), (0.001, 0.005), (0.005, 0.02)]:
                    sel = (~labels) & (depths >= lo) & (depths < hi)
                    bins[f"{lo*1000:g}-{hi*1000:g}mm"] = [int((ok & sel).sum()), int(sel.sum())]
                cell["methods"][key] = {"false_allow": fa, "false_allow_rate": fa / max(1, int((~labels).sum())),
                                        "false_allow_wilson": wilson(fa, int((~labels).sum())),
                                        "false_reject": fr, "missed_by_depth": bins,
                                        "fk_mean": float(fk.mean()), "ms_mean": float(ms.mean())}
            out["cells"][f"r_obs={r_obs},speed={speed}"] = cell
            print("B", r_obs, speed, {k: (v["false_allow"], round(v["fk_mean"], 1)) for k, v in cell["methods"].items()}, flush=True)
    return out


# ----------------------------------------------------------------------------- C
def place_obstacles(m, base, rng, n=4, tries=200):
    """Spheres near the swept arm (1-15 cm clearance) that the base plan clears."""
    obs = []
    caps = [2, 4, 5, 6, 7, 8]
    A, B = m.capsule_segments(base)
    for _ in range(tries):
        if len(obs) == n:
            break
        f = int(rng.integers(5, len(base))); g = int(rng.choice(caps))
        pt = A[f, g] + rng.uniform() * (B[f, g] - A[f, g])
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        c = pt + d * (m.capsules[g].radius + OBSTACLE_RADIUS + rng.uniform(0.01, 0.15))
        trial = np.array(obs + [c])
        if certify_full(m, base, trial, OBSTACLE_RADIUS).ok:
            obs.append(c)
    return np.array(obs).reshape(-1, 3)


def streaming(m, roots, cycles, seed, jitter, tights=(np.inf, 0.03, 0.01)):
    rng = np.random.default_rng(seed)
    keys = ["waypoints_only", "discrete_0.01", "certified_full"] + [f"delta_tight={t:g}" for t in tights]
    rec = {k: {"fk": [], "ms": []} for k in keys}
    seg = {f"delta_tight={t:g}": [0, 0, 0] for t in tights}
    disagree = {f"delta_tight={t:g}": 0 for t in tights}
    missed = {"waypoints_only": 0, "discrete_0.01": 0}
    n_cycles = rejected = 0
    for root in range(roots):
        goal = m.home + rng.uniform(GOAL_LO, -GOAL_LO)
        base = interpolate(m.home, goal, 41)
        obs = place_obstacles(m, base, rng)
        chains = {}
        for t in tights:
            v0 = certify_full(m, base, obs, OBSTACLE_RADIUS, tight=t)
            chains[t] = [v0.cert, base, 0]
        plan, offset = base, 0
        for c in range(cycles):
            offset += 1
            proposal = np.concatenate([plan[offset:], np.repeat(plan[-1:], offset, 0)])
            d = 4
            w = np.clip((np.arange(41) - d) / (40 - d), 0, 1)[:, None]
            child = proposal + w * np.cumsum(rng.normal(0, jitter, (41, 6)), 0) * 0.3
            child[:d] = proposal[:d]
            n_cycles += 1
            r_w, ms_w = timed(lambda: waypoint_check(m, child, obs, OBSTACLE_RADIUS))
            r_d, ms_d = timed(lambda: discrete_check(m, child, obs, OBSTACLE_RADIUS, 0.01))
            chk = BatchedCertifiedChecker(m, obs, OBSTACLE_RADIUS, max_depth=14)
            r_f, ms_f = timed(lambda: chk.check_plan(child))
            rec["waypoints_only"]["fk"].append(r_w["fk_evals"]); rec["waypoints_only"]["ms"].append(ms_w)
            rec["discrete_0.01"]["fk"].append(r_d["fk_evals"]); rec["discrete_0.01"]["ms"].append(ms_d)
            rec["certified_full"]["fk"].append(chk.fk_evals); rec["certified_full"]["ms"].append(ms_f)
            truth_collision = (not r_f["ok"]) and r_f["status"] in ("collision", "joint_limit")
            missed["waypoints_only"] += bool(r_w["ok"] and truth_collision)
            missed["discrete_0.01"] += bool(r_d["ok"] and truth_collision)
            for t in tights:
                cert, cplan, coff = chains[t]
                r_x, ms_x = timed(lambda: inherit(m, cert, child, obs, OBSTACLE_RADIUS, offset, tight=t))
                k = f"delta_tight={t:g}"
                rec[k]["fk"].append(r_x.fk_evals); rec[k]["ms"].append(ms_x)
                seg[k][0] += r_x.reused; seg[k][1] += r_x.inherited; seg[k][2] += r_x.rechecked
                disagree[k] += bool(r_x.ok) != bool(r_f["ok"])
                if r_x.ok:
                    chains[t] = [r_x.cert, child, 0]
            if r_f["ok"]:
                plan, offset = child, 0
            else:
                rejected += 1
    out = {"cycles": n_cycles, "rejected_cycles": rejected, "missed_collisions_vs_certified": missed,
           "delta_disagreements_vs_certified": disagree,
           "delta_segments_reused_inherited_rechecked": seg, "methods": {}}
    for k, v in rec.items():
        fk = np.array(v["fk"]); ms = np.array(v["ms"])
        out["methods"][k] = {"fk_mean": float(fk.mean()), "fk_p95": float(np.percentile(fk, 95)),
                             "ms_mean": float(ms.mean()), "ms_p95": float(np.percentile(ms, 95)), "ms_max": float(ms.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True, help="mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    ap.add_argument("--review", default=str(Path(__file__).resolve().parents[2] / "docs/research/2026-10-02/ur5e/v3/review/reviewed_per_root.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--graze", type=int, default=300)
    ap.add_argument("--roots", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=30)
    args = ap.parse_args()
    m = UR5eModel(Path(args.mjcf))
    res = {"schema": "sentinel-arm-kinematic-pilot-v1", "python": platform.python_version(),
           "numpy": np.__version__, "cpu": platform.processor() or platform.machine(),
           "note": "pilot, not a frozen protocol; static capsule geometry only"}
    t = time.time(); res["A_fidelity"] = fidelity(m, args.review); print("A", res["A_fidelity"], flush=True)
    res["B_accuracy"] = accuracy(m, args.graze, 20261003)
    res["C_streaming"] = {}
    for jit in (0.002, 0.01, 0.03):
        res["C_streaming"][str(jit)] = streaming(m, args.roots, args.cycles, 7 + int(jit * 1000), jit)
        print("C", jit, json.dumps(res["C_streaming"][str(jit)], indent=1), flush=True)
    res["wall_s"] = time.time() - t
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
