#!/usr/bin/env python3
"""Independent long-horizon MuJoCo fallback study for the low-friction profile.

This study does not run or extend the trained W2 WorldGuard.  It defines a new
5 s physical evaluation profile with 1.6, 3.2 and 4.8 s sibling actions, plus a
0.5 s diagnostic hold.  The predeclared friction-floor rule uses only candidate
plan acceleration, an assumed supported-profile floor mu=0.015, and the final
observed payload support/relative position.  Root friction, mass and future
labels remain hidden from every policy decision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
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

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402
import train_mujoco_world as w2  # noqa: E402


PROTOCOL_SCHEMA = "sentinel-low-friction-fallback-protocol-v1"
RESULT_SCHEMA = "sentinel-low-friction-fallback-result-v1"
HORIZON = 100
DURATIONS = (1.6, 3.2, 4.8)
MU_FLOOR = 0.015
GRAVITY = 9.81
DISPLACEMENT = (0.35, 0.10, 0.08)
TASK_COMPLETION_ENDPOINT_TOLERANCE_M = 0.01


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    def scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               default=scalar, allow_nan=False) + "\n")


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes(); protocol = json.loads(raw)
    expected = {
        "schema": PROTOCOL_SCHEMA, "status": "frozen", "history_frames": w2.HISTORY,
        "forecast_frames": HORIZON, "frame_dt_s": w2.FRAME_DT,
        "forecast_s": 5.0, "posthold_s": 0.5, "durations_s": list(DURATIONS),
        "mu_floor": MU_FLOOR, "gravity_m_s2": GRAVITY,
        "displacement_xyz_m": list(DISPLACEMENT),
        "task_completion_endpoint_tolerance_m": TASK_COMPLETION_ENDPOINT_TOLERANCE_M,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol mismatch for {key}")
    if protocol.get("test_seed_start") != 620000 or protocol.get("test_roots") != 100:
        raise ValueError("formal test roots must be frozen at seeds 620000..620099")
    if protocol.get("preflight_seed_start") != 670000 or protocol.get("preflight_roots") != 5:
        raise ValueError("preflight roots must be frozen at seeds 670000..670004")
    return protocol, _sha_bytes(raw)


def _candidate_targets(start, duration: float, np):
    displacement = np.asarray(DISPLACEMENT, dtype=np.float64)
    rows = []
    for step in range(1, HORIZON + 1):
        u = min(step * w2.FRAME_DT / duration, 1.0)
        smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
        rows.append(np.asarray(start) + displacement * smooth)
    return np.asarray(rows, dtype=np.float32)


def _plan_bound(targets, start, np):
    positions = np.vstack((np.asarray(start), np.asarray(start), targets.astype(np.float64)))
    acceleration = np.diff(positions, n=2, axis=0) / (w2.FRAME_DT**2)
    horizontal = np.linalg.norm(acceleration[:, :2], axis=1)
    normal_acceleration = GRAVITY + acceleration[:, 2]
    capacity = MU_FLOOR * np.maximum(normal_acceleration, 0.0)
    margin = capacity - horizontal
    return {
        "compliant": bool(np.all((normal_acceleration > 0.0) & (margin >= 0.0))),
        "minimum_margin_m_s2": float(np.min(margin)),
        "peak_horizontal_acceleration_m_s2": float(np.max(horizontal)),
        "minimum_effective_normal_acceleration_m_s2": float(np.min(normal_acceleration)),
    }


def _generate_root(task):
    index, seed = task
    import mujoco
    import numpy as np

    rng = random.Random(seed)
    friction = math.exp(rng.uniform(math.log(0.015), math.log(0.05)))
    mass = rng.uniform(0.04, 0.16)
    perturb = (rng.uniform(-0.008, 0.008), rng.uniform(-0.008, 0.008))
    config = PhysicsConfig(timestep=w2.DT, friction=friction, payload_mass=mass, seed=seed)
    xml = model_xml(config)
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    payload_address = model.joint("payload_free").qposadr[0]
    data.qpos[payload_address:payload_address + 2] = perturb
    mujoco.mj_forward(model, data)
    base = np.asarray((0.0, 0.0, 0.45))
    for _ in range(round(0.3 / w2.DT)):
        data.ctrl[:] = (0.0, 0.0, 0.0); mujoco.mj_step(model, data)
    history, past_targets = [], []
    prior = base.copy()
    for frame in range(w2.HISTORY):
        if frame < 6:
            phase = math.pi * (frame + 1) / 6.0
            target = base + np.asarray((0.008 * math.sin(phase) ** 2,
                                        0.004 * math.sin(2 * phase),
                                        0.002 * math.sin(phase) ** 2))
        else:
            target = base.copy()
        w2._advance_frame(mujoco, model, data, prior, target); prior = target
        history.append(w2._observe(mujoco, model, data)[0]); past_targets.append(target)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    root_start = np.asarray(data.xpos[model.body("tray").id], dtype=np.float64)
    observed = np.asarray(history[-1])
    observed_r = float(np.linalg.norm(observed[6:8]))
    observed_supported = bool(observed[21] > 0.5)
    targets_all, truth_all = [], []
    risk_window, risk_posthold, drop_window, drop_posthold = [], [], [], []
    endpoint_forecast, endpoint_posthold, bounds, terminal = [], [], [], []
    actual_peak_acceleration, actual_target_acceleration_deviation = [], []
    action_hashes = []
    for duration in DURATIONS:
        branch = mujoco.MjData(model)
        mujoco.mj_setState(model, branch, state, spec); mujoco.mj_forward(model, branch)
        targets = _candidate_targets(root_start, duration, np)
        bounds.append(_plan_bound(targets, root_start, np))
        outputs, offsets, tray_positions = [], [], []
        dropped = False; prior_target = np.asarray(past_targets[-1], dtype=np.float64)
        for target in targets:
            w2._advance_frame(mujoco, model, branch, prior_target, target)
            prior_target = target
            _, output, _, floor = w2._observe(mujoco, model, branch)
            outputs.append(output); offsets.append(float(np.linalg.norm(output[:2])))
            tray_positions.append(np.asarray(branch.xpos[model.body("tray").id]).copy())
            dropped = dropped or floor or abs(output[0]) > 0.14 or abs(output[1]) > 0.12
        window_offset, window_drop = max(offsets), dropped
        tray_position = np.asarray(branch.xpos[model.body("tray").id])
        forecast_tracking = float(np.linalg.norm(tray_position - targets[-1]))
        target_positions = np.vstack((root_start, root_start, targets.astype(np.float64)))
        observed_positions = np.vstack((root_start, root_start, np.asarray(tray_positions)))
        target_acceleration = np.diff(target_positions, n=2, axis=0) / (w2.FRAME_DT**2)
        observed_acceleration = np.diff(observed_positions, n=2, axis=0) / (w2.FRAME_DT**2)
        actual_peak_acceleration.append(
            float(np.max(np.linalg.norm(observed_acceleration[:, :2], axis=1))))
        actual_target_acceleration_deviation.append(
            float(np.max(np.linalg.norm(
                observed_acceleration[:, :2] - target_acceleration[:, :2], axis=1))))
        for _ in range(10):
            w2._advance_frame(mujoco, model, branch, prior_target, prior_target)
            _, output, _, floor = w2._observe(mujoco, model, branch)
            offsets.append(float(np.linalg.norm(output[:2])))
            dropped = dropped or floor or abs(output[0]) > 0.14 or abs(output[1]) > 0.12
        tray_position = np.asarray(branch.xpos[model.body("tray").id])
        posthold_tracking = float(np.linalg.norm(tray_position - targets[-1]))
        targets_all.append(targets); truth_all.append(outputs)
        risk_window.append(window_offset > w2.RISK_LIMIT or window_drop)
        risk_posthold.append(max(offsets) > w2.RISK_LIMIT or dropped)
        drop_window.append(window_drop); drop_posthold.append(dropped)
        endpoint_forecast.append(forecast_tracking); endpoint_posthold.append(posthold_tracking)
        action_hashes.append(_sha_bytes(targets.tobytes()))
        terminal.append({"duration_s": duration, "forecast_maximum_xy_offset_m": window_offset,
                         "posthold_maximum_xy_offset_m": max(offsets),
                         "forecast_endpoint_tracking_error_m": forecast_tracking,
                         "posthold_endpoint_tracking_error_m": posthold_tracking,
                         "observed_peak_horizontal_tray_acceleration_m_s2":
                             actual_peak_acceleration[-1],
                         "maximum_observed_vs_target_horizontal_acceleration_deviation_m_s2":
                             actual_target_acceleration_deviation[-1],
                         "forecast_drop": window_drop, "posthold_drop": dropped})
    return {
        "index": index, "root_id": f"low-friction-fallback-{seed:06d}",
        "history": np.asarray(history, np.float32),
        "past_targets": np.asarray(past_targets, np.float32),
        "future_targets": np.asarray(targets_all, np.float32),
        "truth": np.asarray(truth_all, np.float32),
        "risk_window": np.asarray(risk_window, np.uint8),
        "risk_posthold": np.asarray(risk_posthold, np.uint8),
        "drop_window": np.asarray(drop_window, np.uint8),
        "drop_posthold": np.asarray(drop_posthold, np.uint8),
        "endpoint_forecast": np.asarray(endpoint_forecast, np.float32),
        "endpoint_posthold": np.asarray(endpoint_posthold, np.float32),
        "actual_peak_acceleration": np.asarray(actual_peak_acceleration, np.float32),
        "actual_target_acceleration_deviation": np.asarray(
            actual_target_acceleration_deviation, np.float32),
        "integration_state": state,
        "observed_r": np.float32(observed_r),
        "observed_supported": np.uint8(observed_supported),
        "metadata": {"seed": seed, "friction": friction, "payload_mass_kg": mass,
                     "initial_payload_xy_m": list(perturb), "mjcf_sha256": _sha_bytes(xml.encode()),
                     "integration_state_spec": int(spec),
                     "integration_state_sha256": _sha_bytes(state.tobytes()),
                     "all_siblings_restored_from_same_integration_state": True,
                     "action_sha256": action_hashes, "plan_bound": bounds,
                     "terminal_labels": terminal},
    }


def _generate(seed_start: int, roots: int, workers: int, np):
    tasks = [(index, seed_start + index) for index in range(roots)]; rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for complete, row in enumerate(pool.map(_generate_root, tasks, chunksize=1), 1):
            rows.append(row)
            if complete % 25 == 0 or complete == roots:
                print(json.dumps({"phase": "physics", "complete": complete,
                                  "total": roots}), flush=True)
    rows.sort(key=lambda row: row["index"])
    fields = ("history", "past_targets", "future_targets", "truth", "risk_window",
              "risk_posthold", "drop_window", "drop_posthold", "endpoint_forecast",
              "endpoint_posthold", "actual_peak_acceleration",
              "actual_target_acceleration_deviation", "integration_state", "observed_r",
              "observed_supported")
    arrays = {field: np.stack([row[field] for row in rows]) for field in fields}
    arrays["root_ids"] = np.asarray([row["root_id"] for row in rows])
    metadata = {row["root_id"]: row["metadata"] for row in rows}
    return arrays, metadata


def _ci(values, rng, np, replicates=2000, denominator=None):
    values = np.asarray(values, dtype=np.float64); count = len(values)
    indices = rng.integers(0, count, size=(replicates, count))
    if denominator is None:
        point = float(values.mean()); samples = values[indices].mean(1)
    else:
        denominator = np.asarray(denominator, dtype=np.float64)
        point = float(values.sum() / denominator.sum()) if denominator.sum() else None
        den = denominator[indices].sum(1); num = values[indices].sum(1)
        samples = np.divide(num, den, out=np.full(replicates, np.nan), where=den > 0)
        samples = samples[np.isfinite(samples)]
        if point is None or not len(samples):
            return {"value": point, "root_bootstrap_95_ci": None}
    return {"value": point, "root_bootstrap_95_ci":
            [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]}


def _policy(name, choice, selected, abort, arrays, rng, np):
    index = np.arange(len(selected)); unsafe = arrays["risk_window"][index, choice].astype(bool)
    posthold = arrays["risk_posthold"][index, choice].astype(bool)
    drop = arrays["drop_window"][index, choice].astype(bool)
    endpoint = arrays["endpoint_forecast"][index, choice]
    actual_acceleration = arrays["actual_peak_acceleration"][index, choice]
    acceleration_deviation = arrays["actual_target_acceleration_deviation"][index, choice]
    unsafe_selected = selected & unsafe; safe_selected = selected & ~unsafe
    completed = selected & ~unsafe & ~drop & (endpoint <= TASK_COMPLETION_ENDPOINT_TOLERANCE_M)
    durations = {str(duration): int(np.sum(selected & (choice == position)))
                 for position, duration in enumerate(DURATIONS)}
    return {
        "name": name, "roots": len(selected), "selected": int(selected.sum()),
        "rejected": int((~selected).sum()), "observed_risk_aborts": int(abort.sum()),
        "duration_counts": durations,
        "mean_selected_duration_s": (float(np.mean(np.asarray(DURATIONS)[choice[selected]]))
                                     if selected.any() else None),
        "unsafe_selected": int(unsafe_selected.sum()),
        "unsafe_rate_all_roots": _ci(unsafe_selected, rng, np),
        "unsafe_given_selected": _ci(unsafe_selected, rng, np, denominator=selected),
        "safe_rate_all_roots": _ci(safe_selected, rng, np),
        "task_completed": int(completed.sum()),
        "task_completion_rate_all_roots": _ci(completed, rng, np),
        "task_completion_definition": (
            "selected, no forecast risk/drop, and tray endpoint tracking error <= 0.01 m"
        ),
        "posthold_unsafe_selected": int(np.sum(selected & posthold)),
        "drop_selected": int(np.sum(selected & drop)),
        "endpoint_tracking_error_m": {
            "mean": float(endpoint[selected].mean()) if selected.any() else None,
            "p95": float(np.quantile(endpoint[selected], 0.95)) if selected.any() else None,
            "maximum": float(endpoint[selected].max()) if selected.any() else None,
        },
        "observed_peak_horizontal_tray_acceleration_m_s2": {
            "mean": float(actual_acceleration[selected].mean()) if selected.any() else None,
            "p95": float(np.quantile(actual_acceleration[selected], 0.95)) if selected.any() else None,
            "maximum": float(actual_acceleration[selected].max()) if selected.any() else None,
        },
        "observed_vs_target_horizontal_acceleration_deviation_m_s2": {
            "mean": float(acceleration_deviation[selected].mean()) if selected.any() else None,
            "p95": float(np.quantile(acceleration_deviation[selected], 0.95)) if selected.any() else None,
            "maximum": float(acceleration_deviation[selected].max()) if selected.any() else None,
        },
    }


def _evaluate(arrays, np):
    roots = len(arrays["root_ids"]); rng = np.random.default_rng(20261004)
    abort = (~arrays["observed_supported"].astype(bool)) | (arrays["observed_r"] > w2.RISK_LIMIT)
    fixed_selected = ~abort
    plan_bounds = []
    zero = np.zeros(3)
    for duration in DURATIONS:
        plan_bounds.append(_plan_bound(_candidate_targets(zero, duration, np), zero, np))
    compliant = np.asarray([row["compliant"] for row in plan_bounds], dtype=bool)
    if compliant.any():
        bound_index = int(np.argmax(compliant)); bound_selected = ~abort
    else:
        bound_index = 0; bound_selected = np.zeros(roots, dtype=bool)
    results = {
        "fixed_1p6": _policy("fixed_1p6", np.zeros(roots, dtype=int), fixed_selected,
                              abort, arrays, rng, np),
        "fixed_long_4p8": _policy("fixed_long_4p8", np.full(roots, 2, dtype=int),
                                   fixed_selected, abort, arrays, rng, np),
        "known_profile_target_acceleration_screening": _policy(
            "known_profile_target_acceleration_screening",
            np.full(roots, bound_index, dtype=int),
            bound_selected, abort, arrays, rng, np),
        "plan_bound": {"mu_floor": MU_FLOOR, "gravity_m_s2": GRAVITY,
                       "duration_diagnostics": dict(zip(map(str, DURATIONS), plan_bounds)),
                       "selected_duration_s": DURATIONS[bound_index] if compliant.any() else None,
                       "uses_root_friction_mass_or_future_label": False,
                       "scope_limit": (
                           "screens target acceleration only; servo tracking and contact-force error "
                           "are measured outcomes, not terms in the rule; no mathematical safety guarantee"
                       )},
        "candidate_label_counts": {
            str(duration): {"unsafe_forecast": int(arrays["risk_window"][:, index].sum()),
                            "unsafe_posthold": int(arrays["risk_posthold"][:, index].sum()),
                            "drops_forecast": int(arrays["drop_window"][:, index].sum())}
            for index, duration in enumerate(DURATIONS)},
    }
    results["paired_duration_cost"] = {
        "fixed_long_minus_fixed_1p6_s": 3.2,
        "fixed_long_over_fixed_1p6": 3.0,
        "screening_selected_minus_fixed_1p6_s": (
            results["known_profile_target_acceleration_screening"]["mean_selected_duration_s"] - 1.6
            if results["known_profile_target_acceleration_screening"]["mean_selected_duration_s"]
            is not None else None
        ),
    }
    return results


def _gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              check=True, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-evidence", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv); protocol, protocol_sha = _load_protocol(args.protocol)
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be within 1..8")
    if args.preflight and args.preflight_evidence:
        raise ValueError("preflight cannot consume preflight evidence")
    if not args.preflight and not args.preflight_evidence:
        raise ValueError("formal test requires frozen preflight evidence")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("--out must be empty")
    args.out.mkdir(parents=True, exist_ok=True); shutil.copyfile(args.protocol, args.out / "protocol.json")
    started = time.time()
    import mujoco
    import numpy as np

    if not mujoco.__version__.startswith("3.14"):
        raise RuntimeError(f"requires MuJoCo 3.14.*, got {mujoco.__version__}")
    source_hashes = {"train_mujoco_world.py": _sha_file(Path(w2.__file__)),
                     "src/sentinel_evc/physics.py": _sha_file(ROOT / "src/sentinel_evc/physics.py")}
    if source_hashes != protocol["frozen_sources"]:
        raise ValueError("physics source differs from frozen protocol")
    script_sha = _sha_file(Path(__file__))
    preflight_record = None
    if args.preflight_evidence:
        preflight_record = json.loads(args.preflight_evidence.read_text())
        required = {"status": "complete", "preflight": True, "protocol_sha256": protocol_sha,
                    "script_sha256": script_sha, "root_count": protocol["preflight_roots"]}
        if any(preflight_record.get(key) != value for key, value in required.items()):
            raise ValueError("preflight evidence does not match frozen code and protocol")
        shutil.copyfile(args.preflight_evidence, args.out / "preflight_evidence.json")
    seed_start = protocol["preflight_seed_start"] if args.preflight else protocol["test_seed_start"]
    roots = protocol["preflight_roots"] if args.preflight else protocol["test_roots"]
    arrays, metadata = _generate(seed_start, roots, args.workers, np)
    metrics = _evaluate(arrays, np)
    np.savez_compressed(args.out / "physics_results.npz", **arrays)
    _write_json(args.out / "root_metadata.json", metadata)
    _write_json(args.out / "metrics.json", {"schema": RESULT_SCHEMA, "preflight": args.preflight,
                                             "root_count": roots, "metrics": metrics})
    with (args.out / "summary.csv").open("w", newline="") as stream:
        fields = ("policy", "roots", "selected", "rejected", "observed_risk_aborts",
                  "unsafe_selected", "posthold_unsafe_selected", "drop_selected",
                  "task_completed", "task_completion_rate_all_roots",
                  "mean_selected_duration_s", "endpoint_tracking_mean_m",
                  "endpoint_tracking_p95_m", "endpoint_tracking_maximum_m",
                  "observed_peak_horizontal_acceleration_mean_m_s2",
                  "observed_vs_target_acceleration_deviation_mean_m_s2")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in ("fixed_1p6", "fixed_long_4p8",
                     "known_profile_target_acceleration_screening"):
            row = metrics[name]; endpoint = row["endpoint_tracking_error_m"]
            writer.writerow({"policy": name,
                             "roots": row["roots"], "selected": row["selected"],
                             "rejected": row["rejected"],
                             "observed_risk_aborts": row["observed_risk_aborts"],
                             "unsafe_selected": row["unsafe_selected"],
                             "posthold_unsafe_selected": row["posthold_unsafe_selected"],
                             "drop_selected": row["drop_selected"],
                             "task_completed": row["task_completed"],
                             "task_completion_rate_all_roots":
                                 row["task_completion_rate_all_roots"]["value"],
                             "mean_selected_duration_s": row["mean_selected_duration_s"],
                             "endpoint_tracking_mean_m": endpoint["mean"],
                             "endpoint_tracking_p95_m": endpoint["p95"],
                             "endpoint_tracking_maximum_m": endpoint["maximum"],
                             "observed_peak_horizontal_acceleration_mean_m_s2":
                                 row["observed_peak_horizontal_tray_acceleration_m_s2"]["mean"],
                             "observed_vs_target_acceleration_deviation_mean_m_s2":
                                 row["observed_vs_target_horizontal_acceleration_deviation_m_s2"]["mean"]})
    gpu = _gpu_name()
    if gpu != protocol["execution"]["required_gpu"]:
        raise RuntimeError(f"protocol requires actual {protocol['execution']['required_gpu']}, got {gpu}")
    manifest = {
        "schema": RESULT_SCHEMA, "status": "complete", "preflight": args.preflight,
        "scope": "new 5 s low-friction MuJoCo physical fallback profile; no learned-model or deployment claim",
        "protocol_sha256": protocol_sha, "script_sha256": script_sha,
        "source_commit": args.source_commit, "frozen_sources": source_hashes,
        "root_count": roots, "seed_start": seed_start,
        "decision_inputs": "plan acceleration plus mu floor and final observed support/relative XY only",
        "hidden_from_decision": ["root friction", "payload mass", "future labels"],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "gpu_host": gpu,
                        "workers": args.workers, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_threads": os.environ.get("MKL_NUM_THREADS")},
        "elapsed_seconds": time.time() - started, "artifacts": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size,
                                                  "sha256": _sha_file(path)}
    _write_json(args.out / "manifest.json", manifest)
    if args.preflight:
        evidence = {"status": "complete", "preflight": True, "protocol_sha256": protocol_sha,
                    "script_sha256": script_sha, "root_count": roots, "seed_start": seed_start,
                    "metrics_sha256": _sha_file(args.out / "metrics.json"),
                    "manifest_sha256": _sha_file(args.out / "manifest.json")}
        _write_json(args.out / "preflight_pass.json", evidence)
    print(json.dumps({"status": "complete", "preflight": args.preflight,
                      "roots": roots, "out": str(args.out), "elapsed_seconds":
                      manifest["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
