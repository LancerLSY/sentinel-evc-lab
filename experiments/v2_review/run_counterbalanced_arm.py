#!/usr/bin/env python3
"""Frozen, counterbalanced CPU timing protocol for the hardened arm checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ARM = Path(__file__).resolve().parents[1] / "arm_kinematic"
sys.path.insert(0, str(ARM))
from arm_delta import (  # noqa: E402
    _issue_arm_cert,
    certify_full,
    certify_full_batch,
    discrete_batch,
    inherit_batch,
    scene_digest,
)
from run_arm_pilot import GOAL_LO, place_obstacles  # noqa: E402
from scenarios import OBSTACLE_RADIUS, interpolate  # noqa: E402
from ur5e_kin import UR5eModel  # noqa: E402

METHODS = ("discrete_0.01", "certified_full", "delta_arm")
ORDERS = (
    METHODS,
    ("certified_full", "delta_arm", "discrete_0.01"),
    ("delta_arm", "discrete_0.01", "certified_full"),
)
NS = (1, 8, 32, 128)
JITTERS = (0.002, 0.01)
MAX_FK = 200_000
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_trace_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256(b"sentinel-arm-fair-trace-v1\0")
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        label = name.encode("utf-8")
        h.update(len(label).to_bytes(4, "little")); h.update(label)
        dtype = a.dtype.str.encode("ascii")
        h.update(len(dtype).to_bytes(4, "little")); h.update(dtype)
        shape = np.asarray(a.shape, dtype="<i8").tobytes()
        h.update(len(shape).to_bytes(4, "little")); h.update(shape)
        h.update(a.tobytes())
    return h.hexdigest()


def cell_name(n: int, jitter: float) -> str:
    return f"N{n}-j{jitter:g}"


def prepare(args) -> None:
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    model = UR5eModel(Path(args.mjcf))
    source_files = [Path(__file__), ARM / "arm_delta.py", ARM / "ur5e_kin.py"]
    protocol = {
        "schema": "sentinel-arm-counterbalanced-protocol-v1",
        "status": "FROZEN_BEFORE_TIMING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": {
            "roots": args.roots,
            "cycles_per_root": args.cycles,
            "warmup_cycles_excluded": args.warmup,
            "measured_cycles_per_root": args.cycles - args.warmup,
            "N": list(NS),
            "jitter_rad": list(JITTERS),
            "process_repeats": 3,
            "method_orders": [list(x) for x in ORDERS],
            "deadline_ms": 50.0,
            "max_fk_evals_per_checker": MAX_FK,
            "blas_threads": 1,
        },
        "selection_rule": (
            "Before any timing, certified_full labels every frozen candidate. The first accepted "
            "candidate index is committed for the next trace step; if none is accepted, the prior "
            "plan remains. Timed methods cannot alter this selection or the future trace."
        ),
        "timing_rule": (
            "Run three independent Python processes with rotated method order. Exclude the first 20 "
            "cycles per root. Delta timing includes content/model/profile checks and selected-certificate "
            "copy/hash/seal; its one root certificate is amortized over 40 measured cycles."
        ),
        "analysis": {
            "unit": "root-process pair",
            "point": "ratio of paired unit medians",
            "interval": "paired percentile bootstrap 95%",
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "report_all_cells": True,
        },
        "scope": "static UR5e capsule model; CPU timing; not dynamics, functional safety or real-robot latency",
        "mjcf_sha256": sha256(Path(args.mjcf)),
        "source_sha256": {p.name: sha256(p) for p in source_files},
        "trace_files": {},
    }
    (trace_dir / "protocol.pre-timing.json").write_text(json.dumps(protocol, indent=2))
    for jitter in JITTERS:
        for n in NS:
            bases, obstacles, candidates, selections, reference = [], [], [], [], []
            for root in range(args.roots):
                rng = np.random.default_rng(20261003 + root * 10007 + n * 101 + int(jitter * 1e6))
                base = interpolate(model.home, model.home + rng.uniform(GOAL_LO, -GOAL_LO), 41)
                obs = place_obstacles(model, base, rng)
                if obs.shape != (4, 3):
                    raise RuntimeError(f"trace construction produced {len(obs)} obstacles")
                plan, offset = base, 0
                root_candidates, root_select, root_ref = [], [], []
                for _ in range(args.cycles):
                    offset += 1
                    proposal = np.concatenate([plan[offset:], np.repeat(plan[-1:], offset, 0)])
                    d = 4
                    w = np.clip((np.arange(41) - d) / (40 - d), 0, 1)[None, :, None]
                    kids = proposal[None] + w * np.cumsum(rng.normal(0, jitter, (n, 41, 6)), 1) * 0.3
                    kids[:, :d] = proposal[None, :d]
                    ok, _, _, _, meta = certify_full_batch(
                        model, kids, obs, OBSTACLE_RADIUS, max_fk_evals=MAX_FK, return_meta=True)
                    if meta["budget_exhausted"]:
                        raise RuntimeError("reference checker exhausted its budget during trace construction")
                    accepted = np.flatnonzero(ok)
                    selected = int(accepted[0]) if len(accepted) else -1
                    root_candidates.append(kids)
                    root_select.append(selected)
                    root_ref.append(ok)
                    if selected >= 0:
                        plan, offset = kids[selected], 0
                bases.append(base); obstacles.append(obs); candidates.append(root_candidates)
                selections.append(root_select); reference.append(root_ref)
            arrays = {
                "base": np.asarray(bases, dtype="<f8"),
                "obstacles": np.asarray(obstacles, dtype="<f8"),
                "candidates": np.asarray(candidates, dtype="<f8"),
                "selection": np.asarray(selections, dtype="<i8"),
                "reference_ok": np.asarray(reference, dtype=np.bool_),
            }
            path = trace_dir / f"trace-{cell_name(n, jitter)}.npz"
            np.savez_compressed(path, **arrays)
            protocol["trace_files"][path.name] = {
                "file_sha256": sha256(path),
                "exact_array_bytes_sha256": exact_trace_digest(arrays),
                "shape_candidates": list(arrays["candidates"].shape),
                "selected_cycles": int(np.sum(arrays["selection"] >= 0)),
            }
            print("prepared", path.name, protocol["trace_files"][path.name]["exact_array_bytes_sha256"], flush=True)
    protocol["status"] = "TRACES_FROZEN_BEFORE_TIMING"
    frozen = trace_dir / "protocol.frozen.json"
    frozen.write_text(json.dumps(protocol, indent=2))
    requested = Path(args.protocol)
    if requested.resolve() != frozen.resolve():
        requested.write_text(json.dumps(protocol, indent=2))


def run_method(method, model, arrays, warmup):
    roots, cycles, n = arrays["candidates"].shape[:3]
    rows = []
    for root in range(roots):
        base = arrays["base"][root]
        obs = arrays["obstacles"][root]
        selected = arrays["selection"][root]
        reference = arrays["reference_ok"][root]
        cert = None
        root_ns = 0
        offset = 0
        if method == "delta_arm":
            t0 = time.perf_counter_ns()
            verdict = certify_full(model, base, obs, OBSTACLE_RADIUS, max_fk_evals=MAX_FK)
            root_ns = time.perf_counter_ns() - t0
            if not verdict.ok:
                raise RuntimeError(f"root {root} failed certification: {verdict.status}")
            cert = verdict.cert
        timings, fks = [], []
        disagreements = deadline_misses = budget_unknown = 0
        for cycle in range(cycles):
            kids = arrays["candidates"][root, cycle]
            pick = int(selected[cycle])
            if method == "delta_arm":
                offset += 1
            t0 = time.perf_counter_ns()
            if method == "discrete_0.01":
                ok, fk = discrete_batch(model, kids, obs, OBSTACLE_RADIUS, 0.01)
                meta = {"budget_exhausted": False, "unknown_candidates": 0}
            elif method == "certified_full":
                ok, _, _, fk, meta = certify_full_batch(
                    model, kids, obs, OBSTACLE_RADIUS, max_fk_evals=MAX_FK, return_meta=True)
            else:
                ok, lo, ls, fk, _, meta = inherit_batch(
                    model, cert, kids, obs, OBSTACLE_RADIUS, offset,
                    max_fk_evals=MAX_FK, return_meta=True)
                if pick >= 0:
                    if ok[pick]:
                        cert = _issue_arm_cert(
                            model, kids[pick], scene_digest(obs, OBSTACLE_RADIUS), lo[pick], ls[pick],
                            cert.depth + 1, max_fk_evals=MAX_FK)
                    else:
                        fallback = certify_full(
                            model, kids[pick], obs, OBSTACLE_RADIUS, max_fk_evals=MAX_FK)
                        if fallback.ok:
                            cert = fallback.cert
                    offset = 0
            elapsed = time.perf_counter_ns() - t0
            disagreements += int(np.sum(ok != reference[cycle]))
            budget_unknown += int(meta["unknown_candidates"])
            if cycle >= warmup:
                timings.append(elapsed)
                fks.append(int(fk))
                deadline_misses += int(elapsed > 50_000_000)
        rows.append({
            "root": root,
            "timing_ns": timings,
            "fk_evals": fks,
            "root_certificate_ns": root_ns,
            "disagreements_all_cycles": disagreements,
            "deadline_50ms_misses": deadline_misses,
            "budget_unknown_candidates_all_cycles": budget_unknown,
            "median_ms": float(np.median(timings) / 1e6),
            "p95_ms": float(np.percentile(timings, 95) / 1e6),
        })
    return rows


def run_repeat(args) -> None:
    protocol_path = Path(args.trace_dir) / "protocol.frozen.json"
    protocol = json.load(open(protocol_path))
    if protocol.get("status") != "TRACES_FROZEN_BEFORE_TIMING":
        raise RuntimeError("timing requires a fully frozen trace protocol")
    model = UR5eModel(Path(args.mjcf))
    order = ORDERS[args.repeat_index]
    out = {
        "schema": "sentinel-arm-counterbalanced-repeat-v1",
        "repeat_index": args.repeat_index,
        "method_order": list(order),
        "host": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "protocol_sha256": sha256(protocol_path),
        "cells": {},
    }
    for jitter in JITTERS:
        for n in NS:
            name = cell_name(n, jitter)
            trace_path = Path(args.trace_dir) / f"trace-{name}.npz"
            expected = protocol["trace_files"].get(trace_path.name)
            if expected and sha256(trace_path) != expected["file_sha256"]:
                raise RuntimeError(f"trace file hash changed: {trace_path}")
            with np.load(trace_path) as z:
                arrays = {k: z[k] for k in z.files}
            exact = exact_trace_digest(arrays)
            if expected and exact != expected["exact_array_bytes_sha256"]:
                raise RuntimeError(f"trace bytes changed: {trace_path}")
            out["cells"][name] = {"N": n, "jitter": jitter, "trace_exact_sha256": exact, "methods": {}}
            for method in order:
                out["cells"][name]["methods"][method] = run_method(
                    method, model, arrays, args.warmup)
                print("repeat", args.repeat_index, name, method, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def bootstrap_ratio(a, b, rng):
    point = float(np.median(a) / np.median(b))
    idx = rng.integers(0, len(a), (BOOTSTRAP_REPS, len(a)))
    draws = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    return {"point": point, "bootstrap95": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def summarize(args) -> None:
    repeats = [json.load(open(p)) for p in args.inputs]
    summary = {
        "schema": "sentinel-arm-counterbalanced-summary-v1",
        "analysis_unit": "paired root-process",
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "cells": {},
    }
    for ci, name in enumerate(repeats[0]["cells"]):
        units = {m: [] for m in METHODS}
        misses = {m: 0 for m in METHODS}
        disagreement = {m: 0 for m in METHODS}
        budget = {m: 0 for m in METHODS}
        for rep in repeats:
            for method in METHODS:
                for row in rep["cells"][name]["methods"][method]:
                    value = row["median_ms"]
                    if method == "delta_arm":
                        value += row["root_certificate_ns"] / 40 / 1e6
                    units[method].append(value)
                    misses[method] += row["deadline_50ms_misses"]
                    disagreement[method] += row["disagreements_all_cycles"]
                    budget[method] += row["budget_unknown_candidates_all_cycles"]
        rng = np.random.default_rng(BOOTSTRAP_SEED + ci)
        arrays = {m: np.asarray(v) for m, v in units.items()}
        summary["cells"][name] = {
            "N": repeats[0]["cells"][name]["N"],
            "jitter": repeats[0]["cells"][name]["jitter"],
            "root_process_units": len(arrays["delta_arm"]),
            "unit_median_ms": {m: float(np.median(v)) for m, v in arrays.items()},
            "speedup_full_over_delta": bootstrap_ratio(arrays["certified_full"], arrays["delta_arm"], rng),
            "speedup_discrete_over_delta": bootstrap_ratio(arrays["discrete_0.01"], arrays["delta_arm"], rng),
            "deadline_50ms_misses": misses,
            "disagreements": disagreement,
            "budget_unknown_candidates": budget,
        }
    Path(args.out).write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--mjcf", required=True); p.add_argument("--trace-dir", required=True); p.add_argument("--protocol", required=True)
    p.add_argument("--roots", type=int, default=12); p.add_argument("--cycles", type=int, default=60); p.add_argument("--warmup", type=int, default=20)
    r = sub.add_parser("run")
    r.add_argument("--mjcf", required=True); r.add_argument("--trace-dir", required=True); r.add_argument("--out", required=True)
    r.add_argument("--repeat-index", type=int, choices=(0, 1, 2), required=True); r.add_argument("--warmup", type=int, default=20)
    s = sub.add_parser("summarize")
    s.add_argument("--inputs", nargs=3, required=True); s.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run_repeat(args)
    else: summarize(args)


if __name__ == "__main__":
    main()
