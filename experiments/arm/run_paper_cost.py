#!/usr/bin/env python3
"""Frozen-protocol, paired cost study for the UR5e verification gate.

Obstacle fields are sampled before either the parent or final plan is formed.
The script refuses a non-empty output directory and will only execute against
a frozen protocol that binds this source and both gate/reviewer sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import run_ur5e_guard as gate
import verify_ur5e_roots as reviewer


SCENARIOS = gate.SCENARIOS
METHODS = ("parent_only", "full", "bound_incremental", "conservative")
FINAL_ROOT_BASE = 720_000
PREFLIGHT_ROOT_BASE = 820_000
ROOT_SCENE_STRIDE = 1_000
TIMED_REPEATS = 3
BOOTSTRAP_REPLICATES = 10_000
AMORTIZATION_K = (1, 2, 4, 8)
SOURCE_KEYS = (
    "experiments/arm/run_paper_cost.py",
    "experiments/arm/run_ur5e_guard.py",
    "experiments/arm/verify_ur5e_roots.py",
)


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=native) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protocol_sources(protocol: dict[str, Any]) -> dict[str, str]:
    sources = protocol.get("frozen_sources")
    if isinstance(sources, dict):
        return {str(key): str(value) for key, value in sources.items()}
    if isinstance(sources, list):
        result = {}
        for entry in sources:
            if isinstance(entry, dict) and "path" in entry and "sha256" in entry:
                result[str(entry["path"])] = str(entry["sha256"])
        return result
    return {}


def verify_protocol(protocol_path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, str]]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol status must be 'frozen'")
    frozen = protocol_sources(protocol)
    verified = {}
    for relative in SOURCE_KEYS:
        expected = next((value for key, value in frozen.items()
                         if key == relative or key.endswith("/" + relative)), None)
        if expected is None:
            raise RuntimeError(f"frozen_sources does not bind {relative}")
        actual = sha256(repo / relative)
        if actual != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")
        verified[relative] = actual
    return protocol, verified


def obstacle_field(scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample scene context without consulting any plan or FK trajectory."""
    far = np.array([[1.55, 1.45, 1.55], [1.75, 1.55, 1.65]], dtype=np.float64)
    parent = far.copy()
    final = far.copy()
    rule = "far_control"
    if scenario in ("late_suffix", "narrow_passage", "environment_change"):
        # Fixed UR5e workspace distribution, sampled independently of motion.
        first = rng.uniform([-.62, -.62, .14], [.62, .62, .92])
        second = rng.uniform([-.62, -.62, .14], [.62, .62, .92])
        for _ in range(16):
            if np.linalg.norm(second - first) >= .16:
                break
            second = rng.uniform([-.62, -.62, .14], [.62, .62, .92])
        final = np.stack([first, second])
        parent = far.copy() if scenario == "environment_change" else final.copy()
        rule = "workspace_uniform_before_plan"
    return parent, final, rule


def make_case(lay: gate.Layout, scenario: str, scene_index: int,
              local_index: int, root_base: int) -> dict[str, Any]:
    root = root_base + scene_index * ROOT_SCENE_STRIDE + local_index
    sequence = np.random.SeedSequence(root)
    obstacle_seed, plan_seed = sequence.spawn(2)
    obstacle_rng = np.random.default_rng(obstacle_seed)
    plan_rng = np.random.default_rng(plan_seed)

    # This call deliberately precedes all plan construction.
    parent_obstacles, final_obstacles, obstacle_rule = obstacle_field(scenario, obstacle_rng)

    q0 = lay.home.copy()
    goal = q0 + plan_rng.uniform(
        [-.58, -.38, -.48, -.38, -.32, -.42],
        [.58, .38, .48, .38, .32, .42],
    )
    parent = gate.interpolate(q0, goal)
    final = parent.copy()
    split = 20
    disturbance = 1.0
    transformed = scenario != "clear"
    if scenario in ("late_suffix", "environment_change"):
        delta = plan_rng.uniform(
            [-.48, -.34, -.28, -.25, -.22, -.25],
            [.48, .34, .28, .25, .22, .25],
        )
        final[split:] += np.linspace(0, 1, len(final) - split)[:, None] * delta
    elif scenario == "narrow_passage":
        delta = plan_rng.uniform(
            [-.42, -.28, -.24, -.20, -.18, -.20],
            [.42, .28, .24, .20, .18, .20],
        )
        final[split:] += np.sin(np.linspace(0, np.pi, len(final) - split))[:, None] * delta
    elif scenario == "joint_limit":
        final[split:, 2] = np.linspace(final[split, 2], lay.high[2] + .18, len(final) - split)
    elif scenario == "tracking_disturbance":
        final[split:] += np.linspace(0, 1, len(final) - split)[:, None] * np.array(
            [[.70, -.50, .60, .55, -.45, .45]])
        disturbance = .16
    return {
        "root": root, "scenario": scenario, "parent": parent, "final": final,
        "split": split, "parent_obstacles": parent_obstacles,
        "final_obstacles": final_obstacles, "disturbance": disturbance,
        "transformed": transformed, "obstacle_sampling_rule": obstacle_rule,
        "obstacle_seed_state": obstacle_seed.generate_state(4).tolist(),
        "plan_seed_state": plan_seed.generate_state(4).tolist(),
        "generation_order": ["obstacles", "parent_plan", "final_plan"],
    }


def execute_method(method: str, model: mujoco.MjModel, lay: gate.Layout,
                   case: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    if method == "parent_only":
        started = time.perf_counter_ns()
        result = gate.validate(model, lay, case["parent"], case["parent_obstacles"], 1.0)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return {"decision": bool(result["allow"]), "total_ms": elapsed,
                "static_samples": result["static_samples"],
                "dynamic_rollouts": result["dynamic_rollouts"]}
    if method == "full":
        started = time.perf_counter_ns()
        result = gate.validate(model, lay, case["final"], case["final_obstacles"], case["disturbance"])
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return {"decision": bool(result["allow"]), "total_ms": elapsed,
                "static_samples": result["static_samples"],
                "dynamic_rollouts": result["dynamic_rollouts"]}
    if method == "conservative":
        started = time.perf_counter_ns()
        decision = not case["transformed"]
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return {"decision": decision, "total_ms": elapsed,
                "static_samples": 0, "dynamic_rollouts": 0}
    if method != "bound_incremental":
        raise ValueError(method)

    parent_started = time.perf_counter_ns()
    parent_gate = gate.validate(model, lay, case["parent"], case["parent_obstacles"], 1.0)
    record = gate.parent_record(model, lay, case, parent_gate, binding)
    parent_ms = (time.perf_counter_ns() - parent_started) / 1e6

    delta_started = time.perf_counter_ns()
    reuse, reasons = gate.check_parent_record(model, lay, case, record, binding)
    if reuse:
        delta_gate = gate.validate(
            model, lay, case["final"], case["final_obstacles"], case["disturbance"], case["split"])
        decision = bool(parent_gate["allow"] and record is not None and delta_gate["allow"])
    else:
        delta_gate = gate.validate(
            model, lay, case["final"], case["final_obstacles"], case["disturbance"])
        decision = bool(delta_gate["allow"])
    delta_ms = (time.perf_counter_ns() - delta_started) / 1e6
    return {"decision": decision, "total_ms": parent_ms + delta_ms,
            "c_parent_ms": parent_ms, "c_delta_ms": delta_ms,
            "reuse_authorized": reuse, "full_fallback": not reuse,
            "fallback_reasons": reasons,
            "static_samples": parent_gate["static_samples"] + delta_gate["static_samples"],
            "dynamic_rollouts": parent_gate["dynamic_rollouts"] + delta_gate["dynamic_rollouts"],
            "parent_static_samples": parent_gate["static_samples"],
            "parent_dynamic_rollouts": parent_gate["dynamic_rollouts"],
            "delta_static_samples": delta_gate["static_samples"],
            "delta_dynamic_rollouts": delta_gate["dynamic_rollouts"]}


def timed_root(model: mujoco.MjModel, lay: gate.Layout, case: dict[str, Any],
               binding: dict[str, Any], root_position: int) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    warmup = {method: execute_method(method, model, lay, case, binding) for method in METHODS}
    decisions = {method: bool(result["decision"]) for method, result in warmup.items()}
    repeats = []
    for repeat in range(TIMED_REPEATS):
        offset = (root_position + repeat) % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        results = {}
        for position, method in enumerate(order):
            result = execute_method(method, model, lay, case, binding)
            if bool(result["decision"]) != decisions[method]:
                raise RuntimeError(f"non-deterministic decision for root {case['root']} method {method}")
            results[method] = {**result, "execution_position": position}
        repeats.append({"repeat": repeat, "method_order": order, "results": results})
    return decisions, repeats


def distribution(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def paired_bootstrap(delta: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    sampled = np.mean(delta[indices], axis=1)
    return {"estimate_mean_ms": float(np.mean(delta)),
            "ci95_percentile_ms": [float(np.percentile(sampled, 2.5)),
                                    float(np.percentile(sampled, 97.5))],
            "replicates": BOOTSTRAP_REPLICATES, "unit": "independent root"}


def paired_ratio_bootstrap(numerator: np.ndarray, denominator: np.ndarray,
                           rng: np.random.Generator) -> dict[str, Any]:
    indices = rng.integers(0, len(numerator), size=(BOOTSTRAP_REPLICATES, len(numerator)))
    sampled = np.mean(numerator[indices], axis=1) / np.mean(denominator[indices], axis=1)
    return {"estimate_ratio_of_means": float(np.mean(numerator) / np.mean(denominator)),
            "ci95_percentile": [float(np.percentile(sampled, 2.5)),
                                float(np.percentile(sampled, 97.5))],
            "replicates": BOOTSTRAP_REPLICATES, "unit": "independent root"}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_root = {method: np.asarray([
        np.median([repeat["results"][method]["total_ms"] for repeat in row["timing_repeats"]])
        for row in rows], dtype=np.float64) for method in METHODS}
    c_parent = np.asarray([
        np.median([repeat["results"]["bound_incremental"]["c_parent_ms"]
                   for repeat in row["timing_repeats"]]) for row in rows])
    c_delta = np.asarray([
        np.median([repeat["results"]["bound_incremental"]["c_delta_ms"]
                   for repeat in row["timing_repeats"]]) for row in rows])
    rng = np.random.default_rng(910_2026)
    unsafe = sum(row["review_unsafe"] for row in rows)
    safe = len(rows) - unsafe
    safety = {}
    for method in METHODS:
        false_allow = sum(row["decisions"][method] and row["review_unsafe"] for row in rows)
        false_reject = sum((not row["decisions"][method]) and
                           (not row["review_unsafe"]) for row in rows)
        safety[method] = {
            "false_allow_count": false_allow, "false_allow_denominator_unsafe": unsafe,
            "false_reject_count": false_reject, "false_reject_denominator_safe": safe,
        }
    marginal = {
        "c_delta_ms": distribution(c_delta), "c_full_ms": distribution(per_root["full"]),
        "paired_difference_c_delta_minus_c_full_ms": paired_bootstrap(
            c_delta - per_root["full"], rng),
        "paired_ratio_c_delta_over_c_full": paired_ratio_bootstrap(
            c_delta, per_root["full"], rng),
    }
    amortized = {}
    for k in AMORTIZATION_K:
        incremental_total = c_parent / k + c_delta
        full_total = c_parent / k + per_root["full"]
        amortized[str(k)] = {
            "incremental_definition": f"C_parent/{k} + C_delta",
            "full_definition": f"C_parent/{k} + C_full",
            "incremental_total_ms": distribution(incremental_total),
            "full_total_ms": distribution(full_total),
            "paired_difference_incremental_minus_full_ms": paired_bootstrap(
                incremental_total - full_total, rng),
            "paired_ratio_incremental_over_full": paired_ratio_bootstrap(
                incremental_total, full_total, rng),
        }
    counters = {}
    for method in METHODS:
        counters[method] = {
            "static_samples_total": int(sum(
                row["timing_repeats"][0]["results"][method]["static_samples"] for row in rows)),
            "dynamic_rollouts_total": int(sum(
                row["timing_repeats"][0]["results"][method]["dynamic_rollouts"] for row in rows)),
        }
    return {
        "roots": len(rows), "timed_repeats_per_method": TIMED_REPEATS,
        "methods_ms_from_per_root_medians": {method: distribution(values)
                                              for method, values in per_root.items()},
        "bound_incremental_components_ms": {"c_parent": distribution(c_parent),
                                             "c_delta": distribution(c_delta)},
        "marginal_cost_comparison": marginal,
        "amortized_bound_incremental": amortized,
        "decision_counts": {method: {"allow": sum(row["decisions"][method] for row in rows),
                                      "reject": sum(not row["decisions"][method] for row in rows)}
                            for method in METHODS},
        "independent_review_safety": safety,
        "review_unsafe": unsafe, "review_safe": safe,
        "full_incremental_decision_disagreement_count": sum(
            row["decisions"]["full"] != row["decisions"]["bound_incremental"] for row in rows),
        "validation_work_first_timed_repeat_totals": counters,
        "fallback_roots": sum(row["timing_repeats"][0]["results"]["bound_incremental"]["full_fallback"]
                              for row in rows),
        "bootstrap": {"kind": "paired nonparametric percentile bootstrap",
                      "seed": 910_2026, "replicates": BOOTSTRAP_REPLICATES},
    }


def load_gate_model(asset_dir: Path, out: Path) -> tuple[mujoco.MjModel, gate.Layout, str]:
    wrapper = gate.make_wrapper(asset_dir, out / "model" / "gate_scene.xml")
    temporary = asset_dir / ".sentinel_paper_cost.xml"
    temporary.write_text(wrapper.read_text())
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)
    return model, gate.layout(model), sha256(wrapper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--preflight", action="store_true", help="run 2 roots per scenario")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("output directory must be absent or empty")
    protocol, frozen_sources = verify_protocol(args.protocol.resolve(), args.repo.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    shutil.copy2(args.protocol, args.out / "frozen_protocol.json")
    (args.out / "model").mkdir(exist_ok=True)
    source_dir = args.out / "source"
    source_dir.mkdir(exist_ok=True)
    for relative in SOURCE_KEYS:
        shutil.copy2(args.repo.resolve() / relative, source_dir / Path(relative).name)

    asset_dir = args.asset_dir.resolve()
    model, lay, wrapper_sha = load_gate_model(asset_dir, args.out)
    asset_sha, _ = gate.tree_hash(asset_dir)
    binding = gate.model_binding(model, lay, asset_sha, wrapper_sha)
    review_model = reviewer.make_model(asset_dir, args.out / "model" / "review_scene.xml")
    review_ids = reviewer.model_ids(review_model)

    roots_per_scene = 2 if args.preflight else 10
    root_base = PREFLIGHT_ROOT_BASE if args.preflight else FINAL_ROOT_BASE
    rows = []
    cases = {}
    for scene_index, scenario in enumerate(SCENARIOS):
        for local_index in range(roots_per_scene):
            case = make_case(lay, scenario, scene_index, local_index, root_base)
            decisions, timing = timed_root(model, lay, case, binding, len(rows))
            static = reviewer.highres_static(
                review_model, review_ids, case["final"], case["final_obstacles"])
            replay = reviewer.highres_replay(
                review_model, review_ids, case["final"], case["final_obstacles"], case["disturbance"])
            review_unsafe = not (static["ok"] and replay["ok"])
            row = {
                "root": case["root"], "scenario": scenario,
                "generation_order": case["generation_order"],
                "obstacle_sampling_rule": case["obstacle_sampling_rule"],
                "obstacle_seed_state": case["obstacle_seed_state"],
                "plan_seed_state": case["plan_seed_state"],
                "parent_obstacles": case["parent_obstacles"],
                "final_obstacles": case["final_obstacles"],
                "disturbance_gain_scale": case["disturbance"],
                "decisions": decisions, "timing_repeats": timing,
                "review_unsafe": review_unsafe,
                "review_reason": static["reason"] if not static["ok"] else (
                    "dynamic_collision" if replay["obstacle"] or replay["self_collision"] else
                    "tracking" if not replay["ok"] else "safe"),
                "review_static": static,
                "review_replay": {key: value for key, value in replay.items()
                                  if key not in ("qpos", "site")},
            }
            rows.append(row)
            cases[f"root_{case['root']}_parent_plan"] = case["parent"]
            cases[f"root_{case['root']}_final_plan"] = case["final"]
            print(json.dumps({"root": case["root"], "scenario": scenario,
                              "unsafe": review_unsafe, "decisions": decisions}), flush=True)

    plans_path = args.out / "plans.npz"
    np.savez_compressed(plans_path, **cases)
    rows_path = args.out / "per_root.json"
    summary_path = args.out / "summary.json"
    write_json(rows_path, rows)
    write_json(summary_path, summarize(rows))
    manifest = {
        "schema": "sentinel-ur5e-paper-cost-v1",
        "run_kind": "development_preflight" if args.preflight else "frozen_final",
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "protocol_sha256": sha256(args.protocol),
        "frozen_sources_verified": frozen_sources,
        "source_sha256": sha256(Path(__file__)),
        "archived_source_sha256": {path.name: sha256(path)
                                    for path in sorted(source_dir.iterdir())},
        "asset_tree_sha256": asset_sha, "gate_wrapper_sha256": wrapper_sha,
        "model_binding_sha256": gate.object_sha256(binding), "model_binding": binding,
        "root_formula": f"{root_base} + scenario_index*1000 + local_index",
        "root_base": root_base,
        "scenario_order": SCENARIOS, "roots_per_scenario": roots_per_scene,
        "scenario_design": {
            "clear": "unchanged plan with far obstacle controls",
            "late_suffix": "perturbed suffix with workspace-uniform obstacles sampled before plans",
            "narrow_passage": "legacy label; perturbed suffix with independent workspace-uniform obstacles, not constructed narrow geometry",
            "joint_limit": "suffix exceeds a joint limit with far obstacle controls",
            "tracking_disturbance": "suffix perturbation plus low actuator gain with far obstacle controls",
            "environment_change": "workspace-uniform final obstacles replace far parent controls",
        },
        "obstacle_independence": "obstacle RNG is spawned and sampled before plan RNG is read; no FK or plan state is used",
        "timing": {"warmups_per_method": 1, "timed_repeats_per_method": TIMED_REPEATS,
                   "order": "Latin-style cyclic rotation by root position and repeat",
                   "c_parent": "parent validation plus bound-record creation",
                   "c_delta": "record checks plus suffix validation or full fallback; excludes parent cost"},
        "review": {"implementation": "verify_ur5e_roots independent high-resolution functions",
                   "static_step_rad": reviewer.MAX_STATIC_STEP, "dt_seconds": reviewer.DT,
                   "substeps_per_frame": reviewer.SUBSTEPS},
        "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                        "numpy": np.__version__, "host": platform.node()},
        "outputs": {"per_root_sha256": sha256(rows_path), "summary_sha256": sha256(summary_path),
                    "plans_sha256": sha256(plans_path)},
        "protocol_identity": protocol.get("protocol_id"),
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(json.loads(summary_path.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
