#!/usr/bin/env python3
"""Evaluate frozen W2 WorldGuard models on predeclared new MuJoCo scenarios.

The six scenarios use new, disjoint roots and preserve the trained four-action
family, 40 x 50 ms forecast window, and 0.5 s diagnostic hold.  Physical
parameters generate labels but never enter either learned model.  The
``displacement_shift`` scenario is outside the trained action contract: its
bare-model behavior is retained as a stress diagnostic while the product
contract policy rejects every root with ``MODEL_UNKNOWN``.

This is an offline simulation comparison.  It is not a hardware, functional
safety, or real-robot object-understanding claim.
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
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402
import train_mujoco_visual as visual  # noqa: E402
import train_mujoco_world as w2  # noqa: E402


PROTOCOL_SCHEMA = "sentinel-worldguard-scenario-protocol-v1"
RESULT_SCHEMA = "sentinel-worldguard-scenario-evaluation-v1"
SCENARIO_ORDER = (
    "nominal",
    "low_friction",
    "mass_low",
    "mass_high",
    "displacement_shift",
    "camera_shift",
)


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
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "frozen":
        raise ValueError("protocol must be a frozen scenario protocol")
    if protocol.get("scenario_order") != list(SCENARIO_ORDER):
        raise ValueError("protocol scenario order differs from the implemented frozen protocol")
    if protocol.get("durations_s") != list(w2.DURATIONS):
        raise ValueError("protocol must preserve the trained four durations")
    if protocol.get("history_frames") != w2.HISTORY or protocol.get("forecast_frames") != w2.HORIZON:
        raise ValueError("protocol history/forecast shape mismatch")
    if protocol.get("frame_dt_s") != w2.FRAME_DT or protocol.get("posthold_s") != 0.5:
        raise ValueError("protocol time windows mismatch")
    roots = int(protocol.get("roots_per_scenario", 0))
    if roots < 1:
        raise ValueError("roots_per_scenario must be positive")
    scenarios = protocol.get("scenarios", {})
    if set(scenarios) != set(SCENARIO_ORDER):
        raise ValueError("protocol must define exactly the six frozen scenarios")
    seeds = []
    for name in SCENARIO_ORDER:
        start = int(scenarios[name]["seed_start"])
        seeds.extend(range(start, start + roots))
    if len(seeds) != len(set(seeds)) or any(100000 <= seed <= 101998 for seed in seeds):
        raise ValueError("scenario seeds must be unique and disjoint from the W2 source split")
    return protocol, _sha_bytes(raw)


def _sample_physics(scenario: str, seed: int) -> tuple[float, float, tuple[float, float]]:
    rng = random.Random(seed)
    if scenario == "low_friction":
        friction = math.exp(rng.uniform(math.log(0.015), math.log(0.05)))
    else:
        friction = math.exp(rng.uniform(math.log(0.08), math.log(0.8)))
    if scenario == "mass_low":
        mass = rng.uniform(0.02, 0.035)
    elif scenario == "mass_high":
        mass = rng.uniform(0.17, 0.20)
    else:
        mass = rng.uniform(0.04, 0.16)
    perturb = (rng.uniform(-0.008, 0.008), rng.uniform(-0.008, 0.008))
    return friction, mass, perturb


def _candidate_targets(start, duration: float, displacement_scale: float, np):
    displacement = np.asarray((0.35, 0.10, 0.08)) * displacement_scale
    rows = []
    for step in range(1, w2.HORIZON + 1):
        u = min(step * w2.FRAME_DT / duration, 1.0)
        smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
        rows.append(start + displacement * smooth)
    return np.asarray(rows, dtype=np.float32)


def _generate_root(task):
    scenario, index, seed, displacement_scale = task
    import mujoco
    import numpy as np

    friction, mass, perturb = _sample_physics(scenario, seed)
    config = PhysicsConfig(timestep=w2.DT, friction=friction, payload_mass=mass, seed=seed)
    xml = model_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    payload_address = model.joint("payload_free").qposadr[0]
    data.qpos[payload_address:payload_address + 2] = perturb
    mujoco.mj_forward(model, data)
    base = np.asarray((0.0, 0.0, 0.45))
    for _ in range(round(0.3 / w2.DT)):
        data.ctrl[:] = (0.0, 0.0, 0.0)
        mujoco.mj_step(model, data)
    histories, past_targets = [], []
    prior = base.copy()
    for frame in range(w2.HISTORY):
        if frame < 6:
            phase = math.pi * (frame + 1) / 6.0
            target = base + np.asarray((0.008 * math.sin(phase) ** 2,
                                        0.004 * math.sin(2 * phase),
                                        0.002 * math.sin(phase) ** 2))
        else:
            target = base.copy()
        w2._advance_frame(mujoco, model, data, prior, target)
        prior = target
        histories.append(w2._observe(mujoco, model, data)[0])
        past_targets.append(target)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    state_hash = _sha_bytes(state.tobytes())
    root_start = np.asarray(data.xpos[model.body("tray").id], dtype=np.float64)
    initial_output = w2._observe(mujoco, model, data)[1]
    all_targets, all_truth = [], []
    risk_window, risk_posthold, drop_window, drop_posthold = [], [], [], []
    action_hashes, terminal = [], []
    for duration in w2.DURATIONS:
        branch = mujoco.MjData(model)
        mujoco.mj_setState(model, branch, state, spec)
        mujoco.mj_forward(model, branch)
        targets = _candidate_targets(root_start, duration, displacement_scale, np)
        outputs, offsets = [], []
        dropped = False
        prior_target = np.asarray(past_targets[-1], dtype=np.float64)
        for target in targets:
            w2._advance_frame(mujoco, model, branch, prior_target, target)
            prior_target = target
            _, output, _, floor = w2._observe(mujoco, model, branch)
            outputs.append(output)
            offsets.append(float(np.linalg.norm(output[:2])))
            dropped = dropped or floor or abs(output[0]) > 0.14 or abs(output[1]) > 0.12
        window_offset, window_drop = max(offsets), dropped
        for _ in range(10):
            w2._advance_frame(mujoco, model, branch, prior_target, prior_target)
            _, output, _, floor = w2._observe(mujoco, model, branch)
            offsets.append(float(np.linalg.norm(output[:2])))
            dropped = dropped or floor or abs(output[0]) > 0.14 or abs(output[1]) > 0.12
        posthold_offset = max(offsets)
        all_targets.append(targets); all_truth.append(outputs)
        risk_window.append(window_offset > w2.RISK_LIMIT or window_drop)
        risk_posthold.append(posthold_offset > w2.RISK_LIMIT or dropped)
        drop_window.append(window_drop); drop_posthold.append(dropped)
        action_hashes.append(_sha_bytes(targets.tobytes()))
        terminal.append({
            "duration_s": duration,
            "forecast_maximum_xy_offset_m": window_offset,
            "posthold_maximum_xy_offset_m": posthold_offset,
            "forecast_drop": window_drop,
            "posthold_drop": dropped,
        })
    return {
        "index": index,
        "root_id": f"scenario-{scenario}-{seed:06d}",
        "history": np.asarray(histories, dtype=np.float32),
        "past_targets": np.asarray(past_targets, dtype=np.float32),
        "future_targets": np.asarray(all_targets, dtype=np.float32),
        "truth": np.asarray(all_truth, dtype=np.float32),
        "initial_output": initial_output.astype(np.float32),
        "risk_window": np.asarray(risk_window, dtype=np.uint8),
        "risk_posthold": np.asarray(risk_posthold, dtype=np.uint8),
        "drop_window": np.asarray(drop_window, dtype=np.uint8),
        "drop_posthold": np.asarray(drop_posthold, dtype=np.uint8),
        "integration_state": state,
        "metadata": {
            "seed": seed,
            "friction": friction,
            "payload_mass_kg": mass,
            "initial_payload_xy_m": list(perturb),
            "displacement_scale": displacement_scale,
            "mjcf_sha256": _sha_bytes(xml.encode()),
            "integration_state_spec": int(spec),
            "integration_state_sha256": state_hash,
            "all_siblings_restored_from_same_integration_state": True,
            "action_sha256": action_hashes,
            "terminal_labels": terminal,
        },
    }


def _generate_scenario(name: str, spec: dict, roots: int, workers: int, np):
    scale = float(spec["displacement_scale"])
    start = int(spec["seed_start"])
    tasks = [(name, index, start + index, scale) for index in range(roots)]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for complete, row in enumerate(pool.map(_generate_root, tasks, chunksize=1), 1):
            rows.append(row)
            if complete % 25 == 0 or complete == roots:
                print(json.dumps({"phase": "generate", "scenario": name,
                                  "complete": complete, "total": roots}), flush=True)
    rows.sort(key=lambda row: row["index"])
    fields = ("history", "past_targets", "future_targets", "truth", "initial_output",
              "risk_window", "risk_posthold", "drop_window", "drop_posthold",
              "integration_state")
    split = {field: np.stack([row[field] for row in rows]) for field in fields}
    root_ids = np.asarray([row["root_id"] for row in rows])
    metadata = {row["root_id"]: row["metadata"] for row in rows}
    return split, root_ids, metadata


def _load_npz_models(run: Path, records: list[dict], Model, model_args, torch, np, device):
    models = []
    for record in records:
        path = run / record["checkpoint"]
        if _sha_file(path) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            state = {name: torch.from_numpy(archive[key])
                     for key, name in record["tensor_names"].items()}
        model = Model(*model_args).to(device)
        model.load_state_dict(state, strict=True); model.eval(); models.append(model)
    if len(models) != 3:
        raise ValueError("frozen comparison requires three-member ensembles")
    return models


def _frozen_calibration(metric: dict, prediction, truth, np):
    full = metric["full_state_joint_envelope"]
    xy = metric["risk_xy_joint_envelope"]
    full_scale = np.asarray(full["candidate_output_scales_dev"], dtype=np.float32)
    xy_scale = np.asarray(xy["candidate_xy_scales_dev"], dtype=np.float32)
    full_q = float(full["calibration"]["q"])
    xy_q = float(xy["calibration"]["q"])
    full_radius = full_q * full_scale[None, :, None, :]
    lower, upper = prediction - full_radius, prediction + full_radius
    covered = np.all((truth >= lower) & (truth <= upper), axis=(1, 2, 3))
    xy_radius = xy_q * xy_scale[None, :, None, :]
    xy_lower = prediction[..., :2] - xy_radius
    xy_upper = prediction[..., :2] + xy_radius
    xy_covered = np.all((truth[..., :2] >= xy_lower) &
                        (truth[..., :2] <= xy_upper), axis=(1, 2, 3))
    max_x = np.maximum(np.abs(xy_lower[..., 0]), np.abs(xy_upper[..., 0]))
    max_y = np.maximum(np.abs(xy_lower[..., 1]), np.abs(xy_upper[..., 1]))
    allowed = np.max(np.sqrt(max_x**2 + max_y**2), axis=2) <= w2.RISK_LIMIT
    return {
        "prediction": prediction.astype(np.float32),
        "lower": lower.astype(np.float32), "upper": upper.astype(np.float32),
        "covered": covered.astype(np.uint8),
        "risk_xy_lower": xy_lower.astype(np.float32),
        "risk_xy_upper": xy_upper.astype(np.float32),
        "risk_xy_covered": xy_covered.astype(np.uint8),
        "allowed": allowed.astype(np.uint8),
    }


def _ci(values, rng, np, replicates, ratio_denominator=None):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if ratio_denominator is None:
        point = float(np.mean(values))
        sampled = values[rng.integers(0, count, size=(replicates, count))].mean(1)
    else:
        denominator = np.asarray(ratio_denominator, dtype=np.float64)
        point = float(values.sum() / denominator.sum()) if denominator.sum() else None
        indices = rng.integers(0, count, size=(replicates, count))
        numerators = values[indices].sum(1); denominators = denominator[indices].sum(1)
        sampled = np.divide(numerators, denominators,
                            out=np.full(replicates, np.nan), where=denominators > 0)
        sampled = sampled[np.isfinite(sampled)]
        if point is None or not len(sampled):
            return {"value": point, "bootstrap_95_ci": None}
    return {"value": point,
            "bootstrap_95_ci": [float(np.quantile(sampled, 0.025)),
                                 float(np.quantile(sampled, 0.975))]}


def _policy_metrics(choice, selected, split, rng, np, replicates):
    roots = len(selected); indices = np.arange(roots)
    unsafe_window = split["risk_window"].astype(bool)[indices, choice]
    unsafe_posthold = split["risk_posthold"].astype(bool)[indices, choice]
    unsafe_selected = selected & unsafe_window
    unsafe_posthold_selected = selected & unsafe_posthold
    safe_selected = selected & ~unsafe_window
    durations = {str(duration): int(np.sum(selected & (choice == index)))
                 for index, duration in enumerate(w2.DURATIONS)}
    return {
        "roots": roots,
        "selected": int(selected.sum()), "rejected": int((~selected).sum()),
        "duration_counts": durations,
        "reject_rate": _ci(~selected, rng, np, replicates),
        "unsafe_selected": int(unsafe_selected.sum()),
        "unsafe_rate_all_roots": _ci(unsafe_selected, rng, np, replicates),
        "unsafe_given_selected": _ci(unsafe_selected, rng, np, replicates,
                                      ratio_denominator=selected),
        "safe_selected": int(safe_selected.sum()),
        "safe_rate_all_roots": _ci(safe_selected, rng, np, replicates),
        "posthold_unsafe_selected": int(unsafe_posthold_selected.sum()),
        "posthold_unsafe_rate_all_roots": _ci(unsafe_posthold_selected, rng, np, replicates),
    }


def _evaluate_model(name, prediction, arrays, split, rng, np, replicates):
    root_mae = np.mean(np.abs(prediction - split["truth"]), axis=(1, 2, 3))
    root_xy_mae = np.mean(np.abs(prediction[..., :2] - split["truth"][..., :2]),
                          axis=(1, 2, 3))
    allowed = arrays["allowed"].astype(bool)
    selected = np.any(allowed, axis=1)
    choice = np.argmax(allowed, axis=1)
    return {
        "name": name,
        "mae_15d": _ci(root_mae, rng, np, replicates),
        "mae_xy_m": _ci(root_xy_mae, rng, np, replicates),
        "full_state_joint_root_coverage": _ci(
            arrays["covered"].astype(bool), rng, np, replicates),
        "xy_joint_root_coverage": _ci(
            arrays["risk_xy_covered"].astype(bool), rng, np, replicates),
        "policy": _policy_metrics(choice, selected, split, rng, np, replicates),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--state-run", required=True, type=Path)
    parser.add_argument("--visual-run", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--render-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if min(args.workers, args.render_batch_size, args.bootstrap_replicates) < 1:
        raise ValueError("workers, render batch size and bootstrap replicates must be positive")
    protocol, protocol_sha = _load_protocol(args.protocol)
    roots = int(protocol["roots_per_scenario"])
    if args.preflight:
        roots = min(2, roots)
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("--out must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_bytes(args.protocol.read_bytes())
    started = time.time()
    try:
        import mujoco
        import numpy as np
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("requires existing mujoco, numpy, torch and torchvision") from exc
    if not mujoco.__version__.startswith("3.14"):
        raise RuntimeError(f"requires MuJoCo 3.14.*, got {mujoco.__version__}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)

    state_manifest_path = args.state_run / "manifest.json"
    state_metrics_path = args.state_run / "metrics.json"
    visual_manifest_path = args.visual_run / "manifest.json"
    visual_metrics_path = args.visual_run / "metrics.json"
    visual_cache_path = args.visual_run / "visual_cache.npz"
    state_manifest = json.loads(state_manifest_path.read_text())
    state_metrics_source = json.loads(state_metrics_path.read_text())["gru128_ensemble"]
    visual_manifest = json.loads(visual_manifest_path.read_text())
    visual_metrics_source = json.loads(visual_metrics_path.read_text())["visual_object_gru128"]
    if state_manifest.get("status") != "complete" or visual_manifest.get("status") != "complete":
        raise ValueError("source model runs must be complete")
    if _sha_file(visual_cache_path) != visual_manifest["visual"]["cache_sha256"]:
        raise ValueError("visual cache hash mismatch")
    frozen_sources = protocol["frozen_sources"]
    actual_sources = {
        "state_manifest_sha256": _sha_file(state_manifest_path),
        "state_metrics_sha256": _sha_file(state_metrics_path),
        "visual_manifest_sha256": _sha_file(visual_manifest_path),
        "visual_metrics_sha256": _sha_file(visual_metrics_path),
        "visual_cache_sha256": _sha_file(visual_cache_path),
        "encoder_weights_sha256": _sha_file(args.weights),
    }
    if actual_sources != frozen_sources:
        raise ValueError("source artifacts differ from the predeclared frozen protocol")
    with np.load(visual_cache_path, allow_pickle=False) as cache:
        pca = [(cache[f"pca_{camera}_mean"].copy(),
                cache[f"pca_{camera}_components"].copy(),
                cache[f"pca_{camera}_eigenvalues"].copy()) for camera in range(2)]

    state_models = _load_npz_models(
        args.state_run, state_manifest["members"], w2._model_class(torch),
        (state_manifest["normalizers"],), torch, np, device,
    )
    visual_models = _load_npz_models(
        args.visual_run, visual_manifest["training"]["visual_members"],
        visual._model_class(torch), (visual_manifest["normalizers"], True),
        torch, np, device,
    )
    encoder, encoder_sha = visual._encoder(args.weights, torch, torchvision, device)
    if encoder_sha != visual_manifest["visual"]["encoder_weights_sha256"]:
        raise ValueError("encoder weights differ from frozen visual source")

    splits, root_ids, root_metadata, render_metadata = {}, {}, {}, {}
    for scenario in SCENARIO_ORDER:
        scenario_spec = dict(protocol["scenarios"][scenario])
        if args.preflight:
            scenario_spec["seed_start"] += 50000
        split, identifiers, metadata = _generate_scenario(
            scenario, scenario_spec, roots, args.workers, np,
        )
        splits[scenario], root_ids[scenario], root_metadata[scenario] = split, identifiers, metadata
        raw_path = args.out / f"{scenario}-layer4.npy"
        render_args = SimpleNamespace(out=args.out, render_batch_size=args.render_batch_size)
        cameras = visual.SHIFTED_CAMERAS if scenario == "camera_shift" else visual.CAMERAS
        raw, render_metadata[scenario] = visual._render_raw(
            scenario, split, cameras, raw_path, encoder, render_args,
            mujoco, np, torch, torchvision, device,
        )
        split["visual"] = visual._project(raw, pca, np)
        del raw; raw_path.unlink()
    del encoder

    metrics, saved = {}, {}
    summary_rows = []
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        split = splits[scenario]
        state_prediction = w2._predict(state_models, split, torch, device, np)
        visual_prediction = visual._predict(visual_models, split, torch, device, np)
        state_arrays = _frozen_calibration(state_metrics_source, state_prediction,
                                           split["truth"], np)
        visual_arrays = _frozen_calibration(visual_metrics_source, visual_prediction,
                                            split["truth"], np)
        rng = np.random.default_rng(20261003 + scenario_index)
        scenario_metrics = {
            "scope": protocol["scenarios"][scenario]["scope"],
            "roots": roots,
            "geometry_only_fastest_0p6": _policy_metrics(
                np.zeros(roots, dtype=int), np.ones(roots, dtype=bool), split, rng, np,
                args.bootstrap_replicates),
            "fixed_1p6": _policy_metrics(
                np.full(roots, 3, dtype=int), np.ones(roots, dtype=bool), split, rng, np,
                args.bootstrap_replicates),
            "state_worldguard": _evaluate_model(
                "frozen_state_gru128", state_prediction, state_arrays, split, rng, np,
                args.bootstrap_replicates),
            "image_worldguard": _evaluate_model(
                "frozen_visual_gru128", visual_prediction, visual_arrays, split, rng, np,
                args.bootstrap_replicates),
        }
        if scenario == "displacement_shift":
            scenario_metrics["product_contract_gate"] = {
                "reason": "MODEL_UNKNOWN",
                "action_family_supported": False,
                **_policy_metrics(np.zeros(roots, dtype=int), np.zeros(roots, dtype=bool),
                                  split, rng, np, args.bootstrap_replicates),
            }
            scenario_metrics["comparison_status"] = (
                "cross-contract bare-model stress diagnostic; excluded from contract-supported aggregate"
            )
        else:
            scenario_metrics["comparison_status"] = "trained action-family comparison"
        metrics[scenario] = scenario_metrics
        saved[f"{scenario}_root_ids"] = root_ids[scenario]
        for field in ("history", "past_targets", "future_targets", "truth", "initial_output",
                      "risk_window", "risk_posthold", "drop_window", "drop_posthold",
                      "integration_state", "visual"):
            saved[f"{scenario}_{field}"] = split[field]
        for model_name, prediction, arrays in (
                ("state", state_prediction, state_arrays),
                ("visual", visual_prediction, visual_arrays)):
            saved[f"{scenario}_{model_name}_prediction"] = prediction
            for field in ("lower", "upper", "covered", "risk_xy_lower", "risk_xy_upper",
                          "risk_xy_covered", "allowed"):
                saved[f"{scenario}_{model_name}_{field}"] = arrays[field]
        for policy in ("geometry_only_fastest_0p6", "fixed_1p6"):
            row = scenario_metrics[policy]
            summary_rows.append({"scenario": scenario, "policy": policy,
                                 "selected": row["selected"], "rejected": row["rejected"],
                                 "unsafe_selected": row["unsafe_selected"],
                                 "unsafe_rate_all_roots": row["unsafe_rate_all_roots"]["value"],
                                 "safe_rate_all_roots": row["safe_rate_all_roots"]["value"]})
        for key in ("state_worldguard", "image_worldguard"):
            row = scenario_metrics[key]["policy"]
            summary_rows.append({"scenario": scenario, "policy": key,
                                 "selected": row["selected"], "rejected": row["rejected"],
                                 "unsafe_selected": row["unsafe_selected"],
                                 "unsafe_rate_all_roots": row["unsafe_rate_all_roots"]["value"],
                                 "safe_rate_all_roots": row["safe_rate_all_roots"]["value"]})

    np.savez_compressed(args.out / "scenario_predictions_and_labels.npz", **saved)
    _write_json(args.out / "metrics.json", {
        "schema": RESULT_SCHEMA,
        "scenarios": metrics,
        "aggregate_scope": {
            "training_distribution_nominal": ["nominal"],
            "physics_parameter_stress": ["low_friction", "mass_low", "mass_high"],
            "visual_profile_stress": ["camera_shift"],
            "cross_action_contract_diagnostic": ["displacement_shift"],
            "rejections_are_not_counted_as_success": True,
            "calibration": "frozen source dev scales and original alpha=0.05 calibration quantiles",
        },
    })
    _write_json(args.out / "root_metadata.json", root_metadata)
    with (args.out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    manifest = {
        "schema": RESULT_SCHEMA, "status": "complete", "preflight": args.preflight,
        "scope": "offline MuJoCo frozen-model scenario comparison; no product or safety certification claim",
        "protocol": {"path": str(args.protocol.resolve()), "sha256": protocol_sha,
                     "roots_per_scenario_executed": roots},
        "source_commit": args.source_commit,
        "script_sha256": _sha_file(Path(__file__)),
        "sources": actual_sources,
        "model_inputs": {
            "state": "22D observed history, past targets, future candidate, true current initial_output",
            "visual": "two-camera RGB history features, tray pose/velocity, past targets, future candidate",
            "hidden_label_parameters": ["friction", "payload_mass_kg", "initial_payload_xy_m"],
            "future_images": False,
        },
        "rendering": {
            "rgb_channel_order": "RGB uint8 from mujoco.Renderer",
            "encoder_input_channel_order": "RGB -> NCHW, ImageNet1K_V1 normalization",
            "scenario_profiles": render_metadata,
        },
        "timing": {"dt_s": w2.DT, "frame_dt_s": w2.FRAME_DT,
                   "forecast_frames": w2.HORIZON, "forecast_s": 2.0,
                   "posthold_s": 0.5, "durations_s": list(w2.DURATIONS)},
        "physics_pairing": "four sibling candidates restored from one complete mjSTATE_INTEGRATION root",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device)},
        "bootstrap": {"unit": "root", "replicates": args.bootstrap_replicates,
                      "seed_base": 20261003, "interval": "percentile 95%"},
        "elapsed_seconds": time.time() - started,
        "artifacts": {},
    }
    expected_gpu = protocol.get("execution", {}).get("required_gpu")
    if expected_gpu and manifest["environment"]["gpu"] != expected_gpu:
        raise RuntimeError(
            f"protocol requires actual {expected_gpu}, got {manifest['environment']['gpu']}"
        )
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size, "sha256": _sha_file(path),
            }
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "gpu": manifest["environment"]["gpu"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
