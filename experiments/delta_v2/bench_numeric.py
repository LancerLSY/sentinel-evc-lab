#!/usr/bin/env python3
"""Numeric Δ-Cert v1 vs v2 pilot benchmark (not a frozen protocol).

Workloads
  W1  the repository's 1000 constructed cases (mix / perturb), parent certificate charged to both sides
  W2  streaming RTC-style revisions over H in {16,40,128} x M in {1,4,8,16,64,256} x two jitters:
      each cycle shifts the plan by one step, freezes a short prefix, blends the suffix toward a
      new proposal and appends one new knot

Methods
  v1_full   geometry.full_check from the pristine main branch (baseline the repo already ships)
  v1_delta  delta_cert.validate_or_inherit from main (v1 Δ-Cert, whole-plan fallback)
  v2_full   IndexedChecker exact full verification (sweep index + active set)
  v2_delta  validate_or_inherit_v2 (segment-local, active set, exact identity)
Every method's decision is compared against v1_full.
"""
from __future__ import annotations

import argparse, json, math, os, platform, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# v1 baseline = main's src/sentinel_evc copied as package "sentinel_evc_v1", e.g.
#   git worktree add ../evc-main main && mkdir -p ../v1pkg && cp -r ../evc-main/src/sentinel_evc ../v1pkg/sentinel_evc_v1
sys.path.insert(0, os.environ.get("SENTINEL_V1_PKG_DIR", str(ROOT.parent / "v1pkg")))

import sentinel_evc_v1.contracts as C1  # noqa: E402
import sentinel_evc_v1.delta_cert as D1  # noqa: E402
import sentinel_evc_v1.geometry as G1  # noqa: E402
import sentinel_evc_v1.scenarios as S1  # noqa: E402
from sentinel_evc.contracts import Plan as Plan2, Scene as Scene2, Sphere as Sphere2  # noqa: E402
from sentinel_evc.delta_cert_v2 import full_v2, validate_or_inherit_v2  # noqa: E402
from sentinel_evc.scene_index import IndexedChecker  # noqa: E402


def to2(plan1):
    return Plan2(plan1.plan_id, plan1.knots, plan1.dt, plan1.gripper_events)


def scene2(sc1):
    return Scene2(sc1.scene_id, tuple(Sphere2(o.center, o.radius) for o in sc1.obstacles), sc1.ws_lo, sc1.ws_hi,
                  sc1.tool_radius, sc1.tracking_reserve)


def tm(fn, reps=1):
    best = []
    for _ in range(reps):
        t = time.perf_counter_ns(); r = fn(); best.append((time.perf_counter_ns() - t) / 1e6)
    return r, statistics.median(best)


# ------------------------------------------------------------------ W1
def w1(cases):
    out = {"cases": cases, "agree": {"v1_delta": 0, "v2_full": 0, "v2_delta": 0}}
    t = {"v1_full": 0.0, "v1_delta": 0.0, "v2_full": 0.0, "v2_delta": 0.0}
    for i in range(cases):
        sc1 = S1.make_scene(i % 50); p1, p2 = S1.make_parent_pair(i)
        child, rec = (S1.mix(p1, p2, plan_id=f"MIX-{i}") if i % 2 == 0 else S1.perturb(p1, seed=i, plan_id=f"NEAR-{i}"))
        # fresh objects so cached digests cannot help either side
        sc2 = scene2(sc1); P2, Ch2 = to2(p1), to2(child)
        (ok, _, _), a = tm(lambda: G1.full_check(child, sc1))
        root1, b0 = tm(lambda: D1.establish_root(p1, sc1))
        v1, b = tm(lambda: D1.validate_or_inherit(child, sc1, p1, root1.certificate, rec))
        def root2():                     # index build + scene hashing + root certificate, all charged
            ck = IndexedChecker(sc2)
            return ck, full_v2(P2, sc2, ck)
        (chk, r2), c0 = tm(root2)
        f2, c = tm(lambda: full_v2(Ch2, sc2, IndexedChecker(sc2)))
        kind = "perturb" if rec.kind == "perturb" else "unknown"
        v2, d = tm(lambda: validate_or_inherit_v2(Ch2, sc2, P2, r2.certificate, kind, 0, chk))
        t["v1_full"] += b0 + a; t["v1_delta"] += b0 + b; t["v2_full"] += c0 + c; t["v2_delta"] += c0 + d
        out["agree"]["v1_delta"] += (v1.verdict != "REJECTED") == ok
        out["agree"]["v2_full"] += f2.accepted == ok
        out["agree"]["v2_delta"] += v2.accepted == ok
    out["total_ms_incl_parent"] = t
    return out


# ------------------------------------------------------------------ W2
def rand_scene(rng, m):
    obs = [C1.Sphere((rng.uniform(0.05, 0.75), rng.uniform(-0.35, 0.35), rng.uniform(0.05, 0.55)), rng.uniform(0.005, 0.02))
           for _ in range(m)]
    return C1.Scene(f"s{m}-{rng.random()}", tuple(obs), (0.0, -0.4, 0.0), (0.8, 0.4, 0.6), 0.02, 0.005)


def free_path(rng, sc1, h):
    for _ in range(400):
        a = (0.12, rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.4)); b = (0.68, rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.4))
        amp = rng.uniform(-0.15, 0.15)
        ks = tuple((a[0] + (b[0] - a[0]) * i / h, a[1] + (b[1] - a[1]) * i / h + amp * math.sin(math.pi * i / h),
                    a[2] + (b[2] - a[2]) * i / h) for i in range(h + 1))
        p = C1.Plan("p", ks, 0.05)
        if G1.full_check(p, sc1)[0]:
            return p
    return None


def w2(h, m, cycles, jitter, seed):
    rng = random.Random(seed)
    sc1 = rand_scene(rng, m)
    sc2 = scene2(sc1)
    plan = free_path(rng, sc1, h)
    if plan is None:
        return None
    cur2 = to2(plan)
    t0 = time.perf_counter_ns()
    chk = IndexedChecker(sc2)                       # index build + root certificate: charged to v2 Δ
    cert2 = full_v2(cur2, sc2, chk).certificate
    root_ms = (time.perf_counter_ns() - t0) / 1e6
    cur1 = plan; cert1 = D1.establish_root(plan, sc1).certificate
    times = {k: [] for k in ("v1_full", "v1_delta", "v2_full", "v2_delta")}
    agree = {"v1_delta": 0, "v2_full": 0, "v2_delta": 0}
    seg = [0, 0, 0]
    rejected = 0
    d = 3
    drift = [0.0, 0.0, 0.0]
    n = 0
    for c in range(cycles):
        ks = [list(k) for k in cur1.knots[1:]]
        last = ks[-1]
        ks.append([last[0], last[1], last[2]])
        drift = [x + rng.gauss(0, jitter) for x in drift]
        for i in range(d, len(ks)):
            w = (i - d) / (len(ks) - 1 - d)
            for j in range(3):
                ks[i][j] += w * drift[j]
        child1 = C1.Plan(f"c{c}", tuple(tuple(k) for k in ks), 0.05)
        child2 = to2(child1)
        # v1 needs same-grid correspondence: it cannot express a shift, so its parent is the shifted
        # suffix of the previous plan with a registered identity-like record (best case for v1)
        (ok, _, _), a = tm(lambda: G1.full_check(child1, sc1))
        par_shift = C1.Plan("ps", cur1.knots[1:] + (cur1.knots[-1],), 0.05)
        rec = C1.TransformRecord("t", "perturb", par_shift.hash, child1.hash, {})
        root_shift, _ = tm(lambda: D1.establish_root(par_shift, sc1))   # v1 must re-establish (not charged: best case for v1)
        v1, b = tm(lambda: D1.validate_or_inherit(child1, sc1, par_shift, root_shift.certificate, rec))
        f2, cc = tm(lambda: full_v2(child2, sc2, IndexedChecker(sc2, index=chk.index)))
        v2, dd = tm(lambda: validate_or_inherit_v2(child2, sc2, cur2, cert2, "rtc_suffix", 1, chk))
        times["v1_full"].append(a); times["v1_delta"].append(b); times["v2_full"].append(cc); times["v2_delta"].append(dd)
        agree["v1_delta"] += (v1.verdict != "REJECTED") == ok
        agree["v2_full"] += f2.accepted == ok
        agree["v2_delta"] += v2.accepted == ok
        seg[0] += v2.segments_reused; seg[1] += v2.segments_inherited; seg[2] += v2.segments_rechecked
        n += 1
        if v2.accepted:
            cur1, cur2, cert2 = child1, child2, v2.certificate
        else:
            rejected += 1          # keep the last certified plan; the next revision starts from it
            drift = [0.0, 0.0, 0.0]
    return {"H": h, "M": m, "jitter": jitter, "cycles": n, "rejected": rejected, "agree": agree, "segments_reused_inherited_rechecked": seg,
            "root_ms_v2": root_ms,
            "per_cycle_ms": {k: v for k, v in times.items()},
            "median_ms": {k: statistics.median(v) for k, v in times.items() if v},
            "p95_ms": {k: sorted(v)[max(0, math.ceil(0.95 * len(v)) - 1)] for k, v in times.items() if v}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--cases", type=int, default=1000)
    a = ap.parse_args()
    res = {"schema": "sentinel-delta-v2-numeric-pilot-v1", "python": platform.python_version(),
           "note": "pilot, single host, not frozen; v1 code imported from pristine main worktree"}
    res["W1"] = w1(a.cases); print("W1", json.dumps(res["W1"]), flush=True)
    res["W2"] = []
    for h in (16, 40, 128):
        for m in (1, 4, 8, 16, 64, 256):
            for jit in (0.0005, 0.003):
                for rep in range(3):
                    r = w2(h, m, 40, jit, seed=h * 100000 + m * 100 + int(jit * 1e4) + rep * 7)
                    if r:
                        r["rep"] = rep
                        res["W2"].append(r)
                print("W2 cell", h, m, jit, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
