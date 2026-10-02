#!/usr/bin/env python3
"""Prospective stratified MuJoCo stress study for the physical fallback route.

This is a simulator study, not W2 inference and not rectangular-domain
qualification. Policy decisions never read root friction, payload mass, or
future outcome labels.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402
import train_mujoco_world as w2  # noqa: E402


SCHEMA = "sentinel-broad-physical-fallback-v1"
PROTOCOL_SCHEMA = "sentinel-broad-physical-fallback-protocol-v1"
FRICTION_BINS = ((.015, .05), (.05, .2), (.2, .8))
MASS_BINS_KG = ((.02, .06), (.06, .13), (.13, .2))
DISPLACEMENT_SCALE_BINS = ((.8, 1.0), (1.0, 1.2), (1.2, 1.5))
BASE_DISPLACEMENT_M = (.35, .10, .08)
DURATIONS_S = (1.6, 3.2, 4.8)
HORIZON = 100
HOLD_FRAMES = 10
MU_FLOOR = .015
GRAVITY = 9.81
COMPLETION_TOLERANCE_M = .01
DROP_ABSOLUTE_XY_M = (.14, .12)
FORMAL_ROOT_BASE = 730_000
PREFLIGHT_ROOT_BASE = 830_000
FORMAL_ROOTS_PER_CELL = 12
PREFLIGHT_ROOTS_PER_CELL = 1
BOOTSTRAP_REPLICATES = 10_000
SATURATION_RELATIVE_TOLERANCE = 1e-6
POLICIES = ("fixed_1p6", "fixed_4p8", "floor_0p015_shortest")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def native(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=native,
                               allow_nan=False) + "\n")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    expected = {
        "schema": PROTOCOL_SCHEMA, "status": "frozen",
        "friction_bins": [list(x) for x in FRICTION_BINS],
        "payload_mass_bins_kg": [list(x) for x in MASS_BINS_KG],
        "displacement_scale_bins": [list(x) for x in DISPLACEMENT_SCALE_BINS],
        "base_displacement_xyz_m": list(BASE_DISPLACEMENT_M),
        "durations_s": list(DURATIONS_S), "forecast_frames": HORIZON,
        "frame_dt_s": w2.FRAME_DT, "posthold_frames": HOLD_FRAMES,
        "mu_floor": MU_FLOOR, "gravity_m_s2": GRAVITY,
        "risk_threshold_xy_m": w2.RISK_LIMIT,
        "drop_absolute_xy_m": list(DROP_ABSOLUTE_XY_M),
        "completion_endpoint_tolerance_m": COMPLETION_TOLERANCE_M,
        "formal_root_base": FORMAL_ROOT_BASE,
        "preflight_root_base": PREFLIGHT_ROOT_BASE,
        "formal_roots_per_cell": FORMAL_ROOTS_PER_CELL,
        "preflight_roots_per_cell": PREFLIGHT_ROOTS_PER_CELL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "policies": list(POLICIES),
        "acceptance_gate": {"unsafe_count": 0, "drop_count": 0,
                            "minimum_completion_rate_all_roots": .95,
                            "scope": "observed research gate only",
                            "outcome_window": "full_5p5_seconds"},
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol mismatch for {key}")
    frozen = protocol.get("frozen_sources", {})
    actual = {
        "experiments/gpu/run_broad_fallback.py": sha256(Path(__file__)),
        "experiments/gpu/train_mujoco_world.py": sha256(Path(w2.__file__)),
        "src/sentinel_evc/physics.py": sha256(ROOT / "src/sentinel_evc/physics.py"),
    }
    if frozen != actual:
        raise ValueError(f"frozen source mismatch: expected {frozen}, got {actual}")
    return protocol, sha_bytes(raw)


def candidate_targets(start, duration: float, scale: float, np):
    displacement = np.asarray(BASE_DISPLACEMENT_M, dtype=np.float64) * scale
    rows = []
    for step in range(1, HORIZON + 1):
        u = min(step * w2.FRAME_DT / duration, 1.0)
        smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
        rows.append(np.asarray(start) + displacement * smooth)
    return np.asarray(rows, dtype=np.float32)


def plan_bound(targets, start, np) -> dict[str, Any]:
    positions = np.vstack((np.asarray(start), np.asarray(start), targets.astype(np.float64)))
    acceleration = np.diff(positions, n=2, axis=0) / (w2.FRAME_DT**2)
    horizontal = np.linalg.norm(acceleration[:, :2], axis=1)
    normal = GRAVITY + acceleration[:, 2]
    margin = MU_FLOOR * np.maximum(normal, 0.0) - horizontal
    return {
        "compliant": bool(np.all((normal > 0.0) & (margin >= 0.0))),
        "minimum_margin_m_s2": float(np.min(margin)),
        "peak_target_horizontal_acceleration_m_s2": float(np.max(horizontal)),
        "minimum_effective_normal_acceleration_m_s2": float(np.min(normal)),
    }


def cell_parameters(cell_index: int, local_index: int, seed: int,
                    preflight: bool) -> dict[str, Any]:
    friction_index = cell_index // 9
    mass_index = (cell_index // 3) % 3
    displacement_index = cell_index % 3
    friction_bin = FRICTION_BINS[friction_index]
    mass_bin = MASS_BINS_KG[mass_index]
    displacement_bin = DISPLACEMENT_SCALE_BINS[displacement_index]
    parameter_rng = random.Random(seed ^ 0x5A17C3)
    perturb_rng = random.Random(seed)
    if preflight:
        friction = math.exp(parameter_rng.uniform(
            math.log(friction_bin[0]), math.log(friction_bin[1])))
        mass = parameter_rng.uniform(*mass_bin)
        scale = parameter_rng.uniform(*displacement_bin)
        design_point = "preflight_random"
    elif local_index < 8:
        friction = friction_bin[(local_index >> 2) & 1]
        mass = mass_bin[(local_index >> 1) & 1]
        scale = displacement_bin[local_index & 1]
        design_point = f"corner_{local_index}"
    else:
        friction = math.exp(parameter_rng.uniform(
            math.log(friction_bin[0]), math.log(friction_bin[1])))
        mass = parameter_rng.uniform(*mass_bin)
        scale = parameter_rng.uniform(*displacement_bin)
        design_point = f"random_{local_index - 8}"
    perturb = (perturb_rng.uniform(-.008, .008), perturb_rng.uniform(-.008, .008))
    return {
        "friction_bin_index": friction_index,
        "mass_bin_index": mass_index,
        "displacement_bin_index": displacement_index,
        "friction": friction, "payload_mass_kg": mass,
        "displacement_scale": scale, "initial_payload_xy_m": perturb,
        "design_point": design_point,
    }


def generate_root(task: tuple[int, int, int, bool]) -> dict[str, Any]:
    cell_index, local_index, seed, preflight = task
    import mujoco
    import numpy as np

    parameters = cell_parameters(cell_index, local_index, seed, preflight)
    config = PhysicsConfig(timestep=w2.DT, friction=parameters["friction"],
                           payload_mass=parameters["payload_mass_kg"], seed=seed)
    xml = model_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    payload_address = model.joint("payload_free").qposadr[0]
    data.qpos[payload_address:payload_address + 2] = parameters["initial_payload_xy_m"]
    mujoco.mj_forward(model, data)
    base = np.asarray((0.0, 0.0, .45))
    for _ in range(round(.3 / w2.DT)):
        data.ctrl[:] = (0.0, 0.0, 0.0)
        mujoco.mj_step(model, data)
    history = []
    prior = base.copy()
    for frame in range(w2.HISTORY):
        if frame < 6:
            phase = math.pi * (frame + 1) / 6.0
            target = base + np.asarray((.008 * math.sin(phase)**2,
                                        .004 * math.sin(2 * phase),
                                        .002 * math.sin(phase)**2))
        else:
            target = base.copy()
        w2._advance_frame(mujoco, model, data, prior, target)
        prior = target
        history.append(w2._observe(mujoco, model, data)[0])
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    root_start = np.asarray(data.xpos[model.body("tray").id], dtype=np.float64)
    observed = np.asarray(history[-1])
    observed_supported = bool(observed[21] > .5)
    candidates = []
    for duration in DURATIONS_S:
        branch = mujoco.MjData(model)
        mujoco.mj_setState(model, branch, state, spec)
        mujoco.mj_forward(model, branch)
        targets = candidate_targets(root_start, duration, parameters["displacement_scale"], np)
        bound = plan_bound(targets, root_start, np)
        offsets, endpoint_errors = [], []
        dropped = False
        saturation_sample_axes = 0
        saturation_sample_frames = 0
        peak_sampled_force = 0.0
        maximum_sampled_tracking_error = 0.0
        prior_target = base.copy()
        for target in targets:
            w2._advance_frame(mujoco, model, branch, prior_target, target)
            prior_target = target
            _, output, _, floor = w2._observe(mujoco, model, branch)
            offset = float(np.linalg.norm(output[:2]))
            offsets.append(offset)
            dropped = (dropped or floor or abs(output[0]) > DROP_ABSOLUTE_XY_M[0] or
                       abs(output[1]) > DROP_ABSOLUTE_XY_M[1])
            tray_position = np.asarray(branch.xpos[model.body("tray").id], dtype=np.float64)
            tracking = float(np.linalg.norm(tray_position - target))
            endpoint_errors.append(tracking)
            sampled_force = np.abs(np.asarray(branch.actuator_force, dtype=np.float64))
            force_limits = np.max(np.abs(model.actuator_forcerange), axis=1)
            saturated = sampled_force >= force_limits * (1.0 - SATURATION_RELATIVE_TOLERANCE)
            saturation_sample_axes += int(np.sum(saturated))
            saturation_sample_frames += int(np.any(saturated))
            peak_sampled_force = max(peak_sampled_force, float(np.max(sampled_force)))
            maximum_sampled_tracking_error = max(maximum_sampled_tracking_error, tracking)
        forecast_unsafe = max(offsets) > w2.RISK_LIMIT or dropped
        forecast_drop = dropped
        forecast_endpoint_error = endpoint_errors[-1]
        for _ in range(HOLD_FRAMES):
            w2._advance_frame(mujoco, model, branch, prior_target, prior_target)
            _, output, _, floor = w2._observe(mujoco, model, branch)
            offset = float(np.linalg.norm(output[:2]))
            offsets.append(offset)
            dropped = (dropped or floor or abs(output[0]) > DROP_ABSOLUTE_XY_M[0] or
                       abs(output[1]) > DROP_ABSOLUTE_XY_M[1])
            tray_position = np.asarray(branch.xpos[model.body("tray").id], dtype=np.float64)
            tracking = float(np.linalg.norm(tray_position - prior_target))
            sampled_force = np.abs(np.asarray(branch.actuator_force, dtype=np.float64))
            force_limits = np.max(np.abs(model.actuator_forcerange), axis=1)
            saturated = sampled_force >= force_limits * (1.0 - SATURATION_RELATIVE_TOLERANCE)
            saturation_sample_axes += int(np.sum(saturated))
            saturation_sample_frames += int(np.any(saturated))
            peak_sampled_force = max(peak_sampled_force, float(np.max(sampled_force)))
            maximum_sampled_tracking_error = max(maximum_sampled_tracking_error, tracking)
        posthold_unsafe = max(offsets) > w2.RISK_LIMIT or dropped
        candidates.append({
            "duration_s": duration, "action_sha256": sha_bytes(targets.tobytes()),
            "plan_bound": bound, "forecast_unsafe": forecast_unsafe,
            "posthold_unsafe": posthold_unsafe, "forecast_drop": forecast_drop,
            "posthold_drop": dropped,
            "additional_hold_risk": bool((not forecast_unsafe) and posthold_unsafe),
            "forecast_maximum_xy_offset_m": max(offsets[:-HOLD_FRAMES]),
            "posthold_maximum_xy_offset_m": max(offsets),
            "forecast_endpoint_tracking_error_m": forecast_endpoint_error,
            "posthold_endpoint_tracking_error_m": float(np.linalg.norm(
                np.asarray(branch.xpos[model.body("tray").id]) - prior_target)),
            "sampled_servo_force": {
                "cadence_s": w2.FRAME_DT, "includes_posthold": True,
                "saturated_axis_samples": saturation_sample_axes,
                "saturated_frame_samples": saturation_sample_frames,
                "peak_absolute_actuator_force": peak_sampled_force,
                "force_limit": float(np.max(force_limits)),
            },
            "maximum_sampled_tracking_error_m": maximum_sampled_tracking_error,
        })
    return {
        "root_id": f"broad-fallback-{seed}", "seed": seed,
        "cell_index": cell_index, "local_index": local_index,
        **parameters, "observed_supported": observed_supported,
        "integration_state_spec": int(spec),
        "integration_state_sha256": sha_bytes(state.tobytes()),
        "integration_state": state, "mjcf_sha256": sha_bytes(xml.encode()),
        "all_siblings_restored_from_same_integration_state": True,
        "candidates": candidates,
    }


def wilson(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"count": successes, "denominator": total, "rate": None, "wilson95": None}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": successes, "denominator": total, "rate": p,
            "wilson95": [max(0.0, center - radius), min(1.0, center + radius)]}


def policy_outcome(row: dict[str, Any], policy: str) -> dict[str, Any]:
    selected = None
    if row["observed_supported"]:
        if policy == "fixed_1p6":
            selected = 0
        elif policy == "fixed_4p8":
            selected = 2
        else:
            selected = next((index for index, candidate in enumerate(row["candidates"])
                             if candidate["plan_bound"]["compliant"]), None)
    if selected is None:
        return {"selected_index": None, "rejected": True, "unsafe": False,
                "drop": False, "completed": False, "forecast_unsafe": False,
                "forecast_drop": False, "forecast_completed": False,
                "additional_hold_risk": False}
    candidate = row["candidates"][selected]
    forecast_completed = (
        not candidate["forecast_unsafe"] and not candidate["forecast_drop"] and
        candidate["forecast_endpoint_tracking_error_m"] <= COMPLETION_TOLERANCE_M)
    completed = (not candidate["posthold_unsafe"] and not candidate["posthold_drop"] and
                 candidate["posthold_endpoint_tracking_error_m"] <= COMPLETION_TOLERANCE_M)
    return {"selected_index": selected, "rejected": False,
            "unsafe": bool(candidate["posthold_unsafe"]),
            "drop": bool(candidate["posthold_drop"]), "completed": bool(completed),
            "forecast_unsafe": bool(candidate["forecast_unsafe"]),
            "forecast_drop": bool(candidate["forecast_drop"]),
            "forecast_completed": bool(forecast_completed),
            "additional_hold_risk": bool(candidate["additional_hold_risk"])}


def summarize_group(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    outcomes = [row["policy_outcomes"][policy] for row in rows]
    total = len(outcomes)
    selected_indices = [item["selected_index"] for item in outcomes if item["selected_index"] is not None]
    selected_candidates = [row["candidates"][outcome["selected_index"]]
                           for row, outcome in zip(rows, outcomes)
                           if outcome["selected_index"] is not None]
    return {
        "roots": total,
        "unsafe": wilson(sum(item["unsafe"] for item in outcomes), total),
        "drop": wilson(sum(item["drop"] for item in outcomes), total),
        "completion": wilson(sum(item["completed"] for item in outcomes), total),
        "reject": wilson(sum(item["rejected"] for item in outcomes), total),
        "forecast_secondary": {
            "unsafe": wilson(sum(item["forecast_unsafe"] for item in outcomes), total),
            "drop": wilson(sum(item["forecast_drop"] for item in outcomes), total),
            "completion": wilson(sum(item["forecast_completed"] for item in outcomes), total),
        },
        "additional_hold_risk": wilson(sum(item["additional_hold_risk"] for item in outcomes), total),
        "duration_counts": {str(duration): selected_indices.count(index)
                            for index, duration in enumerate(DURATIONS_S)},
        "sampled_servo_force_saturation": {
            "roots_with_saturated_frame_sample": sum(
                candidate["sampled_servo_force"]["saturated_frame_samples"] > 0
                for candidate in selected_candidates),
            "saturated_frame_samples": sum(
                candidate["sampled_servo_force"]["saturated_frame_samples"]
                for candidate in selected_candidates),
            "saturated_axis_samples": sum(
                candidate["sampled_servo_force"]["saturated_axis_samples"]
                for candidate in selected_candidates),
            "peak_absolute_actuator_force": max(
                (candidate["sampled_servo_force"]["peak_absolute_actuator_force"]
                 for candidate in selected_candidates), default=None),
            "sampling_note": "50 ms endpoint samples only; transient substep saturation may be missed",
        },
        "maximum_sampled_tracking_error_m": max(
            (candidate["maximum_sampled_tracking_error_m"] for candidate in selected_candidates),
            default=None),
    }


def paired_bootstrap(rows: list[dict[str, Any]], first: str, second: str,
                     metric: str, seed: int, np) -> dict[str, Any]:
    a = np.asarray([row["policy_outcomes"][first][metric] for row in rows], dtype=np.float64)
    b = np.asarray([row["policy_outcomes"][second][metric] for row in rows], dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
    samples = delta[indices].mean(axis=1)
    return {"first": first, "second": second, "metric": metric,
            "difference_first_minus_second": float(delta.mean()),
            "paired_root_bootstrap95": [float(np.quantile(samples, .025)),
                                        float(np.quantile(samples, .975))],
            "replicates": BOOTSTRAP_REPLICATES}


def evaluate(rows: list[dict[str, Any]], np) -> dict[str, Any]:
    for row in rows:
        row["policy_outcomes"] = {policy: policy_outcome(row, policy) for policy in POLICIES}
    aggregate = {policy: summarize_group(rows, policy) for policy in POLICIES}
    per_cell = {}
    for cell_index in range(27):
        cell = [row for row in rows if row["cell_index"] == cell_index]
        per_cell[str(cell_index)] = {
            "friction_bin": list(FRICTION_BINS[cell_index // 9]),
            "payload_mass_bin_kg": list(MASS_BINS_KG[(cell_index // 3) % 3]),
            "displacement_scale_bin": list(DISPLACEMENT_SCALE_BINS[cell_index % 3]),
            "policies": {policy: summarize_group(cell, policy) for policy in POLICIES},
        }
    comparisons = []
    comparison_index = 0
    for first, second in (("floor_0p015_shortest", "fixed_1p6"),
                          ("floor_0p015_shortest", "fixed_4p8")):
        for metric in ("unsafe", "drop", "completed", "rejected", "additional_hold_risk"):
            comparisons.append(paired_bootstrap(
                rows, first, second, metric, 20261020 + comparison_index, np))
            comparison_index += 1
    floor = aggregate["floor_0p015_shortest"]
    acceptance = {
        "predeclared": {"unsafe_count": 0, "drop_count": 0,
                        "minimum_completion_rate_all_roots": .95,
                        "outcome_window": "full_5p5_seconds"},
        "observed": {"unsafe_count": floor["unsafe"]["count"],
                     "drop_count": floor["drop"]["count"],
                     "completion_rate_all_roots": floor["completion"]["rate"]},
    }
    acceptance["pass"] = bool(
        acceptance["observed"]["unsafe_count"] == 0 and
        acceptance["observed"]["drop_count"] == 0 and
        acceptance["observed"]["completion_rate_all_roots"] >= .95)
    return {"aggregate": aggregate, "per_cell": per_cell,
            "paired_aggregate_comparisons": comparisons,
            "acceptance_gate": acceptance,
            "decision_inputs": ["actual candidate target acceleration",
                                "declared trusted profile mu floor 0.015",
                                "observed payload support at root"],
            "hidden_from_decisions": ["root friction", "payload mass", "future outcomes"],
            "mu_floor_semantics": "declared trusted profile configuration, not a friction measurement"}


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              check=True, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-evidence", type=Path)
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be within 1..8")
    if args.preflight and args.preflight_evidence:
        raise ValueError("preflight cannot consume preflight evidence")
    if not args.preflight and not args.preflight_evidence:
        raise ValueError("formal run requires frozen preflight evidence")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("--out must be absent or empty")
    protocol, protocol_sha = load_protocol(args.protocol)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.protocol, out / "frozen_protocol.json")
    started = time.time()
    import mujoco
    import numpy as np
    if not mujoco.__version__.startswith("3.14"):
        raise RuntimeError(f"requires MuJoCo 3.14.*, got {mujoco.__version__}")
    preflight = None
    script_sha = sha256(Path(__file__))
    if args.preflight_evidence:
        preflight = json.loads(args.preflight_evidence.read_text())
        required = {"status": "complete", "preflight": True,
                    "protocol_sha256": protocol_sha, "script_sha256": script_sha,
                    "root_count": 27, "root_base": PREFLIGHT_ROOT_BASE}
        if any(preflight.get(key) != value for key, value in required.items()):
            raise ValueError("preflight evidence does not bind frozen protocol and script")
        shutil.copy2(args.preflight_evidence, out / "preflight_evidence.json")
    root_base = PREFLIGHT_ROOT_BASE if args.preflight else FORMAL_ROOT_BASE
    roots_per_cell = PREFLIGHT_ROOTS_PER_CELL if args.preflight else FORMAL_ROOTS_PER_CELL
    tasks = [(cell, local, root_base + cell * 100 + local, args.preflight)
             for cell in range(27) for local in range(roots_per_cell)]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for complete, row in enumerate(pool.map(generate_root, tasks, chunksize=1), 1):
            rows.append(row)
            if complete % 27 == 0 or complete == len(tasks):
                print(json.dumps({"phase": "physics", "complete": complete,
                                  "total": len(tasks)}), flush=True)
    rows.sort(key=lambda row: row["seed"])
    metrics = evaluate(rows, np)
    states = {row["root_id"]: row.pop("integration_state") for row in rows}
    np.savez_compressed(out / "root_states.npz", **states)
    write_json(out / "per_root.json", rows)
    write_json(out / "metrics.json", metrics)
    gpu = gpu_name()
    manifest = {
        "schema": SCHEMA, "status": "complete", "preflight": args.preflight,
        "scope": "bounded 27-cell stratified MuJoCo simulator stress; not rectangular-domain qualification",
        "protocol_sha256": protocol_sha, "script_sha256": script_sha,
        "source_commit": args.source_commit, "root_base": root_base,
        "root_count": len(rows), "roots_per_cell": roots_per_cell,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "gpu_host": gpu,
                        "workers": args.workers},
        "sample_semantics": {"servo_force_and_tracking": "50 ms endpoint samples; not substep maxima",
                             "primary_outcome": "full 5.5 s candidate plus posthold window",
                             "forecast_secondary": "5.0 s candidate window",
                             "additional_hold": "new risk during 0.5 s after forecast"},
        "frozen_sources": protocol["frozen_sources"],
        "artifacts": {}, "elapsed_seconds": time.time() - started,
    }
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size,
                                                 "sha256": sha256(path)}
    write_json(out / "manifest.json", manifest)
    if args.preflight:
        evidence = {"status": "complete", "preflight": True,
                    "protocol_sha256": protocol_sha, "script_sha256": script_sha,
                    "root_count": len(rows), "root_base": root_base,
                    "metrics_sha256": sha256(out / "metrics.json"),
                    "manifest_sha256": sha256(out / "manifest.json")}
        write_json(out / "preflight_pass.json", evidence)
    print(json.dumps({"status": "complete", "preflight": args.preflight,
                      "roots": len(rows), "acceptance": metrics["acceptance_gate"],
                      "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
