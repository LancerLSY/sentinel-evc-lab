#!/usr/bin/env python3
"""Generate and train the W2 MuJoCo tray world-model experiment.

This profile uses actual MuJoCo 3.14 contact dynamics but remains an offline
simulation experiment: it has no camera input, robot hardware, product lease,
ACK, prepare/commit, or online execution claim.  Evaluator friction, payload
mass and initial perturbation are retained as metadata and never enter the
network input.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402


DT = 0.002
FRAME_DT = 0.05
SUBSTEPS = 25
HISTORY = 8
HORIZON = 40
DURATIONS = (0.6, 0.9, 1.2, 1.6)
OUTPUT_DIM = 15
RISK_LIMIT = 0.06
MEMBER_SEEDS = (107, 211, 307)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> None:
    def numpy_scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               default=numpy_scalar, allow_nan=False) + "\n")


def _file_sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rotation6(matrix) -> list[float]:
    matrix = matrix.reshape(3, 3)
    return [float(matrix[row, column]) for column in (0, 1) for row in range(3)]


def _contacts(mujoco, model, data) -> tuple[bool, bool]:
    import numpy as np
    tray = model.geom("tray_surface").id
    payload = model.geom("payload_geom").id
    floor = model.geom("floor").id
    supported = floor_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom[0]), int(contact.geom[1])}
        if pair == {tray, payload}:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            supported = supported or float(force[0]) > 0.0
        floor_contact = floor_contact or pair == {floor, payload}
    return supported, floor_contact


def _observe(mujoco, model, data):
    import numpy as np
    tray_id, payload_id = model.body("tray").id, model.body("payload").id
    tray_pos = np.asarray(data.xpos[tray_id], dtype=np.float64)
    payload_pos = np.asarray(data.xpos[payload_id], dtype=np.float64)
    tray_vel = np.asarray(data.qvel[:3], dtype=np.float64)
    payload_vel = np.asarray(data.qvel[3:6], dtype=np.float64)
    angular = np.asarray(data.qvel[6:9], dtype=np.float64)
    relative_pos = payload_pos - tray_pos
    relative_vel = payload_vel - tray_vel
    rotation = np.asarray(_rotation6(np.asarray(data.xmat[payload_id])), dtype=np.float64)
    supported, floor_contact = _contacts(mujoco, model, data)
    observable = np.concatenate((tray_pos, tray_vel, relative_pos, relative_vel,
                                 rotation, angular, [float(supported)]))
    output = np.concatenate((relative_pos, relative_vel, rotation, angular))
    return observable, output, supported, floor_contact


def _advance_frame(mujoco, model, data, start, target) -> None:
    import numpy as np
    start, target = np.asarray(start), np.asarray(target)
    for substep in range(1, SUBSTEPS + 1):
        absolute = start + (target - start) * (substep / SUBSTEPS)
        data.ctrl[:] = (absolute[0], absolute[1], absolute[2] - 0.45)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def _candidate_targets(start, duration, np):
    rows = []
    displacement = np.asarray((0.35, 0.10, 0.08))
    for step in range(1, HORIZON + 1):
        u = min(step * FRAME_DT / duration, 1.0)
        smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u * u)
        rows.append(start + displacement * smooth)
    return np.asarray(rows, dtype=np.float32)


def _generate_root(task):
    index, seed, split = task
    import mujoco
    import numpy as np

    rng = random.Random(seed)
    friction = math.exp(rng.uniform(math.log(0.08), math.log(0.8)))
    mass = rng.uniform(0.04, 0.16)
    perturb = (rng.uniform(-0.008, 0.008), rng.uniform(-0.008, 0.008))
    config = PhysicsConfig(timestep=DT, friction=friction, payload_mass=mass, seed=seed)
    xml = model_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    payload_adr = model.joint("payload_free").qposadr[0]
    data.qpos[payload_adr:payload_adr + 2] = perturb
    mujoco.mj_forward(model, data)
    base = np.asarray((0.0, 0.0, 0.45))
    for _ in range(round(0.3 / DT)):
        data.ctrl[:] = (0.0, 0.0, 0.0)
        mujoco.mj_step(model, data)
    histories, past_targets = [], []
    prior = base.copy()
    for frame in range(HISTORY):
        if frame < 6:
            phase = math.pi * (frame + 1) / 6.0
            target = base + np.asarray((0.008 * math.sin(phase) ** 2,
                                        0.004 * math.sin(2 * phase),
                                        0.002 * math.sin(phase) ** 2))
        else:
            target = base.copy()
        _advance_frame(mujoco, model, data, prior, target)
        prior = target
        histories.append(_observe(mujoco, model, data)[0])
        past_targets.append(target)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    root_start = np.asarray(data.xpos[model.body("tray").id], dtype=np.float64)
    initial_output = _observe(mujoco, model, data)[1]
    all_targets, all_truth, terminal = [], [], []
    risk_window, risk_posthold, drop_window, drop_posthold = [], [], [], []
    action_hashes = []
    for duration in DURATIONS:
        branch = mujoco.MjData(model)
        mujoco.mj_setState(model, branch, state, spec)
        mujoco.mj_forward(model, branch)
        targets = _candidate_targets(root_start, duration, np)
        outputs, offsets = [], []
        dropped = False
        # The captured control target is the last probe target (base), while
        # candidate knots are anchored at the measured tray pose.  Keeping that
        # small tracking offset preserves the actual captured root semantics.
        prior_target = np.asarray(past_targets[-1], dtype=np.float64)
        for target in targets:
            _advance_frame(mujoco, model, branch, prior_target, target)
            prior_target = target
            _, output, _, floor = _observe(mujoco, model, branch)
            outputs.append(output)
            offsets.append(float(np.linalg.norm(output[:2])))
            dropped = dropped or floor or abs(output[0]) > 0.14 or abs(output[1]) > 0.12
        window_maximum_offset = max(offsets)
        window_dropped = dropped
        for _ in range(10):
            _advance_frame(mujoco, model, branch, prior_target, prior_target)
            _, hold_output, _, floor = _observe(mujoco, model, branch)
            offsets.append(float(np.linalg.norm(hold_output[:2])))
            dropped = dropped or floor or abs(hold_output[0]) > 0.14 or abs(hold_output[1]) > 0.12
        maximum_offset = max(offsets)
        branch_warnings = [int(warning.number) for warning in branch.warning]
        all_targets.append(targets)
        all_truth.append(outputs)
        risk_window.append(window_maximum_offset > RISK_LIMIT or window_dropped)
        risk_posthold.append(maximum_offset > RISK_LIMIT or dropped)
        drop_window.append(window_dropped)
        drop_posthold.append(dropped)
        terminal.append({"duration": duration,
                         "forecast_window_maximum_xy_offset_m": window_maximum_offset,
                         "posthold_maximum_xy_offset_m": maximum_offset,
                         "forecast_window_drop": window_dropped, "posthold_drop": dropped,
                         "supported_final": _observe(mujoco, model, branch)[2],
                         "warnings": branch_warnings})
        action_hashes.append(_sha_bytes(targets.tobytes()))
    warnings = [int(warning.number) for warning in data.warning]
    return {
        "index": index, "seed": seed, "split": split,
        "root_id": f"mujoco-root-{seed:08d}", "history": np.asarray(histories, np.float32),
        "past_targets": np.asarray(past_targets, np.float32),
        "future_targets": np.asarray(all_targets, np.float32),
        "truth": np.asarray(all_truth, np.float32), "initial_output": initial_output.astype(np.float32),
        "risk_window": np.asarray(risk_window, np.uint8),
        "risk_posthold": np.asarray(risk_posthold, np.uint8),
        "drop_window": np.asarray(drop_window, np.uint8),
        "drop_posthold": np.asarray(drop_posthold, np.uint8),
        "integration_state": state, "metadata": {
            "friction": friction, "payload_mass_kg": mass,
            "initial_payload_xy_m": list(perturb), "mjcf_sha256": _sha_bytes(xml.encode()),
            "integration_state_spec": int(spec),
            "integration_state_sha256": _sha_bytes(state.tobytes()),
            "action_sha256": action_hashes, "terminal_labels_after_0.5s_hold": terminal,
            "generation_warnings": warnings,
            "history_final_target_xyz": past_targets[-1].tolist(),
            "history_penultimate_target_xyz": past_targets[-2].tolist(),
            "history_final_tray_velocity_m_s": histories[-1][3:6].tolist(),
            "history_final_payload_relative_velocity_m_s": histories[-1][9:12].tolist(),
            "capture_actual_tray_vs_last_target_m": float(np.linalg.norm(root_start - past_targets[-1])),
        },
    }


def _generate_dataset(counts, root_seed: int, workers: int):
    tasks, index = [], 0
    for split, count in zip(("train", "dev", "cal", "test"), counts):
        for _ in range(count):
            tasks.append((index, root_seed + index, split))
            index += 1
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, result in enumerate(pool.map(_generate_root, tasks, chunksize=1), 1):
            results.append(result)
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({"phase": "generate", "complete": completed,
                                  "total": len(tasks)}), flush=True)
    return sorted(results, key=lambda row: row["index"])


def _pack(results, split, np):
    rows = [row for row in results if row["split"] == split]
    return {key: np.stack([row[key] for row in rows]) for key in
            ("history", "past_targets", "future_targets", "truth", "initial_output",
             "risk_window", "risk_posthold", "drop_window", "drop_posthold",
             "integration_state")}


def _load_dataset_run(directory, np):
    directory = directory.resolve()
    dataset_path, manifest_path = directory / "dataset.npz", directory / "dataset_manifest.json"
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("dataset run requires dataset.npz and dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mujoco", "").split(".")[:2] != ["3", "14"]:
        raise ValueError("reused dataset was not generated with MuJoCo 3.14")
    if manifest.get("dt") != DT or manifest.get("frame_dt") != FRAME_DT:
        raise ValueError("reused dataset timing does not match the frozen W2 profile")
    fields = ("history", "past_targets", "future_targets", "truth", "initial_output",
              "risk_window", "risk_posthold", "drop_window", "drop_posthold", "integration_state")
    splits, root_ids = {}, {}
    with np.load(dataset_path, allow_pickle=False) as archive:
        for split in ("train", "dev", "cal", "test"):
            expected = int(manifest["counts"][split])
            splits[split] = {field: archive[f"{split}_{field}"] for field in fields}
            root_ids[split] = archive[f"{split}_root_ids"]
            shapes = {
                "history": (expected, HISTORY, 22),
                "past_targets": (expected, HISTORY, 3),
                "future_targets": (expected, 4, HORIZON, 3),
                "truth": (expected, 4, HORIZON, OUTPUT_DIM),
                "initial_output": (expected, OUTPUT_DIM),
                "risk_window": (expected, 4), "risk_posthold": (expected, 4),
                "drop_window": (expected, 4), "drop_posthold": (expected, 4),
            }
            for field, shape in shapes.items():
                if splits[split][field].shape != shape:
                    raise ValueError(f"reused {split}_{field} shape mismatch")
            if len(root_ids[split]) != expected or any(value.dtype == object for value in splits[split].values()):
                raise ValueError(f"reused {split} arrays are unsafe or incomplete")
            if not all(np.isfinite(value).all() for key, value in splits[split].items()
                       if key not in ("risk_window", "risk_posthold", "drop_window", "drop_posthold")):
                raise ValueError(f"reused {split} contains non-finite numeric data")
    return splits, root_ids, manifest, {
        "dataset_sha256": _file_sha(dataset_path),
        "dataset_manifest_sha256": _file_sha(manifest_path),
    }


def _normalizers(train, np):
    history = np.concatenate((train["history"], train["past_targets"]), axis=2)
    return {
        "history_mean": history.mean((0, 1)).tolist(),
        "history_std": np.maximum(history.std((0, 1)), 1e-5).tolist(),
        "target_mean": train["future_targets"].mean((0, 1, 2)).tolist(),
        "target_std": np.maximum(train["future_targets"].std((0, 1, 2)), 1e-5).tolist(),
        "output_mean": train["truth"].mean((0, 1, 2)).tolist(),
        "output_std": np.maximum(train["truth"].std((0, 1, 2)), 1e-5).tolist(),
        "fit_split": "train",
    }


def _model_class(torch):
    nn = torch.nn

    class WorldGRU(nn.Module):
        def __init__(self, norms):
            super().__init__()
            self.encoder = nn.GRU(25, 128, batch_first=True)
            self.decoder = nn.GRUCell(18, 128)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, OUTPUT_DIM))
            for name, value in norms.items():
                if name != "fit_split":
                    self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

        def forward(self, history, past_targets, future_targets, initial_output,
                    use_history=True, zero_actions=False):
            if zero_actions:
                past_targets = torch.zeros_like(past_targets)
                future_targets = torch.zeros_like(future_targets)
            encoded = torch.cat((history, past_targets), -1)
            encoded = (encoded - self.history_mean) / self.history_std
            if use_history:
                _, hidden = self.encoder(encoded)
                hidden = hidden[0]
            else:
                hidden = torch.zeros((history.shape[0], 128), device=history.device)
            current = (initial_output - self.output_mean) / self.output_std
            outputs = []
            normalized_targets = (future_targets - self.target_mean) / self.target_std
            for step in range(HORIZON):
                hidden = self.decoder(torch.cat((current, normalized_targets[:, step]), -1), hidden)
                current = current + 0.5 * torch.tanh(self.head(hidden))
                outputs.append(current * self.output_std + self.output_mean)
            return torch.stack(outputs, 1)

    return WorldGRU


def _torch_arrays(split, torch, device):
    roots = split["history"].shape[0]
    expand = lambda array: torch.from_numpy(array).to(device)
    history = expand(split["history"])[:, None].expand(roots, 4, HISTORY, 22).reshape(-1, HISTORY, 22)
    past = expand(split["past_targets"])[:, None].expand(roots, 4, HISTORY, 3).reshape(-1, HISTORY, 3)
    initial = expand(split["initial_output"])[:, None].expand(roots, 4, OUTPUT_DIM).reshape(-1, OUTPUT_DIM)
    return history, past, expand(split["future_targets"]).reshape(-1, HORIZON, 3), initial, expand(split["truth"]).reshape(-1, HORIZON, OUTPUT_DIM)


def _predict(models, split, torch, device, np, mode="full"):
    history, past, future, initial, _ = _torch_arrays(split, torch, device)
    if mode == "action_shuffle":
        future = torch.roll(future, 7, 1)
    rows = []
    for model in models:
        model.eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(history), 256):
                parts.append(model(history[start:start + 256], past[start:start + 256],
                                   future[start:start + 256], initial[start:start + 256],
                                   use_history=mode != "no_history",
                                   zero_actions=mode == "no_action").cpu())
        rows.append(torch.cat(parts).numpy())
    roots = split["history"].shape[0]
    return np.mean(rows, axis=0).reshape(roots, 4, HORIZON, OUTPUT_DIM)


def _save_state(path, state, np):
    arrays, names = {}, {}
    for index, (name, tensor) in enumerate(sorted(state.items())):
        key = f"tensor_{index:03d}"
        arrays[key], names[key] = tensor.numpy(), name
    np.savez_compressed(path, **arrays)
    return names


def _train_member(seed, member, train, dev, norms, args, torch, np, device,
                  out, no_action=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = _model_class(torch)(norms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    history, past, future, initial, truth = _torch_arrays(train, torch, device)
    generator = torch.Generator().manual_seed(seed)
    best, best_step, best_state = math.inf, None, None
    prefix = "no-action-" if no_action else ""
    log_path = out / f"{prefix}member-{member}.jsonl"
    with log_path.open("w") as log:
        for step in range(1, args.steps + 1):
            ids = torch.randint(len(history), (args.batch_size,), generator=generator).to(device)
            model.train()
            prediction = model(history[ids], past[ids], future[ids], initial[ids], zero_actions=no_action)
            scale = model.output_std
            loss = torch.nn.functional.smooth_l1_loss(prediction / scale, truth[ids] / scale)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step % 25 == 0 or step == args.steps:
                predicted = _predict([model], dev, torch, device, np,
                                     mode="no_action" if no_action else "full")
                dev_mae = float(np.mean(np.abs(predicted - dev["truth"])))
                row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad),
                       "dev_mae_15d": dev_mae}
                log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"phase": "train", "member": member,
                                  "no_action": no_action, **row}), flush=True)
                if dev_mae < best:
                    best, best_step = dev_mae, step
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no dev-best checkpoint; calibration is forbidden")
    model.load_state_dict(best_state)
    checkpoint = out / f"{prefix}member-{member}.npz"
    names = _save_state(checkpoint, best_state, np)
    return model, {"seed": seed, "best_step": best_step, "best_dev_mae_15d": best,
                   "checkpoint": checkpoint.name, "checkpoint_sha256": _file_sha(checkpoint),
                   "log": log_path.name, "log_sha256": _file_sha(log_path),
                   "tensor_names": names, "independently_trained_no_action": no_action}


def _calibrated_metrics(name, predictor, dev, cal, test, np, alpha=0.05):
    dev_prediction = predictor(dev)
    full_scales = np.maximum(np.sqrt(np.mean((dev_prediction - dev["truth"])**2, axis=(0, 2))), 1e-5)
    xy_scales = np.maximum(
        np.sqrt(np.mean((dev_prediction[..., :2] - dev["truth"][..., :2])**2, axis=(0, 2))), 1e-5
    )
    cal_prediction = predictor(cal)
    full_scores = np.max(
        np.abs(cal_prediction - cal["truth"]) / full_scales[None, :, None, :], axis=(1, 2, 3)
    )
    xy_scores = np.max(
        np.abs(cal_prediction[..., :2] - cal["truth"][..., :2]) /
        xy_scales[None, :, None, :], axis=(1, 2, 3)
    )
    rank = math.ceil((len(full_scores) + 1) * (1 - alpha))
    full_q = float(np.sort(full_scores)[rank - 1]) if rank <= len(full_scores) else None
    xy_q = float(np.sort(xy_scores)[rank - 1]) if rank <= len(xy_scores) else None
    prediction = predictor(test)
    full_radius = np.inf if full_q is None else full_q * full_scales[None, :, None, :]
    lower, upper = prediction - full_radius, prediction + full_radius
    full_covered = np.all((test["truth"] >= lower) & (test["truth"] <= upper), axis=(1, 2, 3))
    xy_radius = np.inf if xy_q is None else xy_q * xy_scales[None, :, None, :]
    xy_lower, xy_upper = prediction[..., :2] - xy_radius, prediction[..., :2] + xy_radius
    xy_covered = np.all(
        (test["truth"][..., :2] >= xy_lower) & (test["truth"][..., :2] <= xy_upper),
        axis=(1, 2, 3),
    )
    max_x = np.maximum(np.abs(xy_lower[..., 0]), np.abs(xy_upper[..., 0]))
    max_y = np.maximum(np.abs(xy_lower[..., 1]), np.abs(xy_upper[..., 1]))
    allowed = np.max(np.sqrt(max_x**2 + max_y**2), axis=2) <= RISK_LIMIT
    unsafe_window = test["risk_window"].astype(bool)
    unsafe_posthold = test["risk_posthold"].astype(bool)
    learned_choice = np.argmax(allowed, axis=1)
    has_choice = np.any(allowed, axis=1)

    def policy(choice, selected):
        window = unsafe_window[np.arange(len(unsafe_window)), choice]
        posthold = unsafe_posthold[np.arange(len(unsafe_posthold)), choice]
        return {"selected_roots": int(np.sum(selected)),
                "rejected_roots": int(np.sum(~selected)),
                "forecast_window_2s_unsafe_selected": int(np.sum(window & selected)),
                "forecast_window_2s_false_allow_rate_all_roots": float(np.mean(window & selected)),
                "posthold_2p5s_unsafe_selected_diagnostic": int(np.sum(posthold & selected)),
                "posthold_2p5s_false_allow_rate_all_roots_diagnostic": float(np.mean(posthold & selected))}

    true_peak = np.max(np.linalg.norm(test["truth"][..., :2], axis=-1), axis=2)
    learned_regret = np.where(has_choice,
                              true_peak[np.arange(len(true_peak)), learned_choice] - np.min(true_peak, axis=1),
                              np.nan)
    return {
        "name": name, "dev_mae_15d": float(np.mean(np.abs(dev_prediction - dev["truth"]))),
        "test_mae_15d": float(np.mean(np.abs(prediction - test["truth"]))),
        "full_state_joint_envelope": {
            "candidate_output_scales_dev": full_scales.tolist(),
            "calibration": {"alpha": alpha, "roots": len(full_scores), "rank": rank,
                            "q": full_q,
                            "score": "root-max over 4 candidates x 40 times x 15 outputs"},
            "test_joint_root_coverage_2s": float(np.mean(full_covered)),
            "decision_use": "state forecast evaluation only; not used by the XY risk gate",
        },
        "risk_xy_joint_envelope": {
            "candidate_xy_scales_dev": xy_scales.tolist(),
            "calibration": {"alpha": alpha, "roots": len(xy_scores), "rank": rank,
                            "q": xy_q,
                            "score": "root-max over 4 candidates x 40 times x payload-relative XY only"},
            "test_joint_root_coverage_2s": float(np.mean(xy_covered)),
            "decision_use": "the only calibrated envelope used by the predeclared 0.06m radial XY gate",
            "scope_limit": "does not provide joint coverage for the other 13 outputs or the extra 0.5s hold",
        },
        "policies": {
            "geometry_earliest_0.6s": policy(np.zeros(len(true_peak), dtype=int), np.ones(len(true_peak), bool)),
            "fixed_slow_1.6s": policy(np.full(len(true_peak), 3), np.ones(len(true_peak), bool)),
            "learned_gate_earliest_allowed": policy(learned_choice, has_choice),
        },
        "learned_gate_allow_rate": float(np.mean(has_choice)),
        "learned_gate_mean_peak_offset_regret_given_allow": (
            float(np.nanmean(learned_regret)) if np.any(has_choice) else None
        ),
    }, {"prediction": prediction.astype(np.float32), "lower": lower.astype(np.float32),
               "upper": upper.astype(np.float32), "covered": full_covered.astype(np.uint8),
               "risk_xy_lower": xy_lower.astype(np.float32),
               "risk_xy_upper": xy_upper.astype(np.float32),
               "risk_xy_covered": xy_covered.astype(np.uint8),
               "allowed": allowed.astype(np.uint8)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-roots", type=int, default=1000)
    parser.add_argument("--dev-roots", type=int, default=200)
    parser.add_argument("--cal-roots", type=int, default=299)
    parser.add_argument("--test-roots", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root-seed", type=int, default=100000)
    parser.add_argument("--source-commit")
    parser.add_argument("--dataset-run", type=Path,
                        help="reuse a completed W2 dataset.npz + dataset_manifest.json")
    parser.add_argument("--train-no-action", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        args.steps = 25
        args.train_roots, args.dev_roots, args.cal_roots, args.test_roots = 8, 4, 19, 4
    counts = (args.train_roots, args.dev_roots, args.cal_roots, args.test_roots)
    if args.preflight and args.dataset_run:
        raise ValueError("--preflight and --dataset-run cannot be combined")
    if min(*counts, args.steps, args.batch_size, args.workers) <= 0:
        raise ValueError("counts, steps, batch size and workers must be positive")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import mujoco
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("training host requires mujoco==3.14.*, numpy and torch") from exc
    if not mujoco.__version__.startswith("3.14"):
        raise RuntimeError(f"this frozen profile requires MuJoCo 3.14, got {mujoco.__version__}")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    started = time.time()
    reused = None
    if args.dataset_run:
        splits, root_ids, dataset_manifest, reused = _load_dataset_run(args.dataset_run, np)
        counts = tuple(int(dataset_manifest["counts"][name]) for name in ("train", "dev", "cal", "test"))
        shutil.copyfile(args.dataset_run.resolve() / "dataset.npz", args.out / "dataset.npz")
        shutil.copyfile(args.dataset_run.resolve() / "dataset_manifest.json", args.out / "dataset_manifest.json")
    else:
        results = _generate_dataset(counts, args.root_seed, args.workers)
        splits = {name: _pack(results, name, np) for name in ("train", "dev", "cal", "test")}
        root_ids = {name: np.asarray([row["root_id"] for row in results if row["split"] == name])
                    for name in ("train", "dev", "cal", "test")}
        dataset_arrays = {}
        for split_name, split in splits.items():
            for key, value in split.items(): dataset_arrays[f"{split_name}_{key}"] = value
            dataset_arrays[f"{split_name}_root_ids"] = root_ids[split_name]
        np.savez_compressed(args.out / "dataset.npz", **dataset_arrays)
        metadata = {row["root_id"]: {"split": row["split"], "seed": row["seed"],
                                      **row["metadata"]} for row in results}
        _write_json(args.out / "dataset_manifest.json", {
        "profile": "W2 actual MuJoCo offline paired-branch dataset",
        "mujoco": mujoco.__version__, "dt": DT, "frame_dt": FRAME_DT,
        "substeps_per_frame": SUBSTEPS, "counts": dict(zip(("train", "dev", "cal", "test"), counts)),
        "observable_22d": "tray pos/vel 6; payload relative pos/vel 6; payload rot6d 6; angular vel 3; support 1",
        "output_15d": "payload relative pos/vel 6; rot6d 6; angular vel 3",
        "evaluator_metadata_not_network_input": ["friction", "payload_mass_kg", "initial_payload_xy_m"],
        "friction_range": [0.08, 0.8], "mass_range_kg": [0.04, 0.16],
        "risk_threshold_predeclared_xy_offset_m": RISK_LIMIT,
        "label_windows": {
            "forecast": "40 x 50ms = 2.0s; the only window covered by model envelopes",
            "posthold": "forecast plus 0.5s fixed-target hold; retained as a diagnostic outcome",
        },
        "capture_to_candidate_control": (
            "each branch starts interpolation from the captured last target (base); candidate knots are "
            "anchored at the measured tray pose, and the small tracking offset is retained per root"
        ),
            "root_metadata": metadata,
        })
    norms = _normalizers(splits["train"], np)
    models, members = [], []
    for member, seed in enumerate(MEMBER_SEEDS):
        model, record = _train_member(seed, member, splits["train"], splits["dev"], norms,
                                      args, torch, np, device, args.out)
        models.append(model); members.append(record)
    predictors = {
        "gru128_ensemble": lambda split: _predict(models, split, torch, device, np),
        "action_shuffled_inference": lambda split: _predict(models, split, torch, device, np, "action_shuffle"),
        "history_zeroed_inference": lambda split: _predict(models, split, torch, device, np, "no_history"),
        "persistence": lambda split: np.repeat(split["initial_output"][:, None, None, :], 4, 1).repeat(HORIZON, 2),
    }
    no_action_members = []
    if args.train_no_action:
        no_action_models = []
        for member, seed in enumerate(MEMBER_SEEDS):
            model, record = _train_member(seed, member, splits["train"], splits["dev"], norms,
                                          args, torch, np, device, args.out, no_action=True)
            no_action_models.append(model); no_action_members.append(record)
        predictors["independently_trained_no_action"] = lambda split: _predict(
            no_action_models, split, torch, device, np, "no_action")
    metrics, main_arrays = {}, None
    for name, predictor in predictors.items():
        metrics[name], arrays = _calibrated_metrics(name, predictor, splits["dev"],
                                                     splits["cal"], splits["test"], np)
        if name == "gru128_ensemble": main_arrays = arrays
    metrics["ablation_semantics"] = {
        "action_shuffled_inference": "inference intervention, not separately trained",
        "history_zeroed_inference": "inference intervention; current observed state retained",
        "independently_trained_no_action": bool(args.train_no_action),
    }
    _write_json(args.out / "metrics.json", metrics)
    np.savez_compressed(args.out / "test_predictions.npz", root_ids=root_ids["test"],
                        truth=splits["test"]["truth"],
                        risk_window=splits["test"]["risk_window"],
                        risk_posthold=splits["test"]["risk_posthold"],
                        drop_window=splits["test"]["drop_window"],
                        drop_posthold=splits["test"]["drop_posthold"], **main_arrays)
    source_commit = args.source_commit or _git_commit()
    manifest = {
        "schema": "sentinel-w2-mujoco-world-v1", "status": "complete",
        "scope": "offline actual MuJoCo dynamics; not visual, hardware, safety certification, or product execution loop",
        "offline_control_semantics": "25 direct MjData.ctrl substeps interpolate each 50ms target; no product ACK/prepare gap",
        "source_commit": source_commit, "script_sha256": _file_sha(Path(__file__)),
        "architecture": "22d observable state + 3d past targets; GRU128; autoregressive GRUCell future final XYZ targets; 15d state",
        "split_protocol": "all four sibling branches remain with one root; train/dev/cal/test disjoint",
        "calibration_protocol": (
            "two frozen objects with independent dev scales and the same 299-root cal split/rank285: "
            "full-state root-max 4x40x15 for 2s joint forecast coverage, and risk-specific root-max "
            "4x40x2 for the 0.06m XY gate; neither covers the extra 0.5s hold"
        ),
        "members": members, "no_action_members": no_action_members,
        "dataset_reuse": reused,
        "normalizers": norms, "risk_threshold_m": RISK_LIMIT,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "elapsed_seconds": time.time() - started, "artifacts": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file(): manifest["artifacts"][path.name] = {"bytes": path.stat().st_size,
                                                               "sha256": _file_sha(path)}
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "test": metrics["gru128_ensemble"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
