#!/usr/bin/env python3
"""Train the W0 numeric ResidualWorld GRU on a CUDA host.

This is an experiment entry point, not a runtime dependency.  It uses PyTorch and
NumPy only on the training host while the installable Sentinel core remains
standard-library plus cryptography.  Data comes from the shipped numeric plant:
hidden k/d/beta generate truth but are never exposed to the network.

Default protocol: 800/120/199/300 disjoint roots, four sibling candidates kept
together, three deterministic members, 1,200 updates/member, dev checkpoint
selection, then frozen dev scales and independent root-max conformal calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinel_evc.data import NumericDataset, make_numeric_dataset  # noqa: E402
from sentinel_evc.numeric_world import candidate_actions  # noqa: E402


DT = 0.05
HISTORY = 8
HORIZON = 40
CANDIDATES = 4
MEMBER_SEEDS = (17, 29, 43)
BASE_SCALES = (0.15, 0.6, 5.0)  # r, v, action; fixed units from the W0 design.
NOMINAL = (20.0, 1.75, 80.0)
RESIDUAL_ACCEL_LIMIT = 8.0


@dataclass(frozen=True)
class SplitArrays:
    name: str
    root_ids: "object"
    root_hashes: "object"
    history: "object"  # [root, 8, 3]
    actions: "object"  # [root, candidate, 40]
    truth_r: "object"  # [root, candidate, 40]
    truth_v: "object"  # [root, candidate, 40]

    @property
    def roots(self) -> int:
        return int(self.history.shape[0])


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _split_arrays(dataset: NumericDataset, name: str, np) -> SplitArrays:
    roots = dataset.split(name)
    histories, actions, truth_r, truth_v = [], [], [], []
    for root in roots:
        histories.append([[sample.r, sample.v, sample.a] for sample in root.history.samples])
        actions.append([candidate_actions(plan) for plan in root.candidates])
        truth_r.append([outcome.r for outcome in root.outcomes])
        truth_v.append([outcome.v for outcome in root.outcomes])
    return SplitArrays(
        name=name,
        root_ids=np.asarray([root.root_id for root in roots]),
        root_hashes=np.asarray([root.hash for root in roots]),
        history=np.asarray(histories, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        truth_r=np.asarray(truth_r, dtype=np.float32),
        truth_v=np.asarray(truth_v, dtype=np.float32),
    )


def _normalization(train: SplitArrays, np) -> dict[str, list[float]]:
    # First express all inputs in the fixed physical units, then estimate only the
    # centering and scale correction from training roots.  No dev/cal/test leakage.
    history = train.history / np.asarray(BASE_SCALES, dtype=np.float32)
    future_actions = train.actions.reshape(-1) / BASE_SCALES[2]
    mean = history.reshape(-1, 3).mean(axis=0)
    std = history.reshape(-1, 3).std(axis=0)
    mean[2] = future_actions.mean()
    std[2] = future_actions.std()
    std = np.maximum(std, 1e-4)
    return {
        "base_scales": list(BASE_SCALES),
        "mean_after_base_scaling": mean.astype(float).tolist(),
        "std_after_base_scaling": std.astype(float).tolist(),
        "fit_split": "train",
    }


def _device(requested: str, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _seed_everything(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _model_class(torch):
    nn = torch.nn

    class ResidualWorldGRU(nn.Module):
        def __init__(self, normalization: dict[str, list[float]]) -> None:
            super().__init__()
            self.encoder = nn.GRU(3, 48, batch_first=True)
            self.decoder = nn.GRUCell(3, 48)
            self.residual_head = nn.Sequential(nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))
            self.register_buffer("base", torch.tensor(normalization["base_scales"], dtype=torch.float32))
            self.register_buffer("mean", torch.tensor(normalization["mean_after_base_scaling"], dtype=torch.float32))
            self.register_buffer("std", torch.tensor(normalization["std_after_base_scaling"], dtype=torch.float32))

        def norm(self, values):
            return (values / self.base - self.mean) / self.std

        def forward(self, history, actions, use_history: bool = True):
            batch = history.shape[0]
            if use_history:
                _, hidden = self.encoder(self.norm(history))
                hidden = hidden[0]
            else:
                hidden = torch.zeros((batch, 48), dtype=history.dtype, device=history.device)
            r = history[:, -1, 0]
            v = history[:, -1, 1]
            rs, vs = [], []
            k, d, beta = NOMINAL
            for index in range(actions.shape[1]):
                action = actions[:, index]
                step_input = torch.stack((r, v, action), dim=-1)
                hidden = self.decoder(self.norm(step_input), hidden)
                residual_accel = RESIDUAL_ACCEL_LIMIT * torch.tanh(self.residual_head(hidden)[:, 0])
                nominal_accel = -action - k * r - d * v - beta * r.pow(3)
                v = v + DT * (nominal_accel + residual_accel)
                r = r + DT * v
                rs.append(r)
                vs.append(v)
            return torch.stack(rs, dim=1), torch.stack(vs, dim=1)

    return ResidualWorldGRU


def _tensor_batches(split: SplitArrays, torch, device):
    roots = split.roots
    history = torch.from_numpy(split.history).to(device)
    history = history[:, None].expand(roots, CANDIDATES, HISTORY, 3).reshape(-1, HISTORY, 3)
    return (
        history,
        torch.from_numpy(split.actions).to(device).reshape(-1, HORIZON),
        torch.from_numpy(split.truth_r).to(device).reshape(-1, HORIZON),
        torch.from_numpy(split.truth_v).to(device).reshape(-1, HORIZON),
    )


def _predict_model(model, split: SplitArrays, torch, device, batch_size: int = 512,
                   ablation: str = "full"):
    history, actions, _, _ = _tensor_batches(split, torch, device)
    if ablation == "no_action":
        history = history.clone()
        history[:, :, 2] = 0.0
        actions = torch.zeros_like(actions)
    elif ablation == "action_shuffle":
        actions = torch.roll(actions, shifts=7, dims=1)
    elif ablation not in ("full", "no_history"):
        raise ValueError(f"unknown ablation: {ablation}")
    rows_r, rows_v = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, history.shape[0], batch_size):
            pred_r, pred_v = model(
                history[start:start + batch_size], actions[start:start + batch_size],
                use_history=ablation != "no_history",
            )
            rows_r.append(pred_r.cpu())
            rows_v.append(pred_v.cpu())
    shape = (split.roots, CANDIDATES, HORIZON)
    return torch.cat(rows_r).numpy().reshape(shape), torch.cat(rows_v).numpy().reshape(shape)


def _physical_prediction(split: SplitArrays, np):
    r = np.repeat(split.history[:, -1, 0, None], CANDIDATES, axis=1)
    v = np.repeat(split.history[:, -1, 1, None], CANDIDATES, axis=1)
    rs, vs = [], []
    k, d, beta = NOMINAL
    for index in range(HORIZON):
        action = split.actions[:, :, index]
        v = v + DT * (-action - k * r - d * v - beta * r**3)
        r = r + DT * v
        rs.append(r.copy())
        vs.append(v.copy())
    return np.stack(rs, axis=2), np.stack(vs, axis=2)


def _persistence_prediction(split: SplitArrays, np):
    r = np.repeat(split.history[:, -1, 0, None, None], CANDIDATES, axis=1)
    r = np.repeat(r, HORIZON, axis=2)
    v = np.repeat(split.history[:, -1, 1, None, None], CANDIDATES, axis=1)
    v = np.repeat(v, HORIZON, axis=2)
    return r, v


def _constant_velocity_prediction(split: SplitArrays, np):
    v0 = split.history[:, -1, 1, None, None]
    times = DT * np.arange(1, HORIZON + 1, dtype=np.float32)[None, None, :]
    r = split.history[:, -1, 0, None, None] + v0 * times
    r = np.repeat(r, CANDIDATES, axis=1)
    v = np.repeat(v0, CANDIDATES, axis=1)
    v = np.repeat(v, HORIZON, axis=2)
    return r, v


def _physics_id_prediction(split: SplitArrays, ridge: float, np):
    """Fit k/d/beta from seven observable transitions, then roll each candidate.

    ``make_numeric_case`` records a sample after applying its attached action.
    Therefore sample ``i+1``'s action is the input for the observable transition
    from sample ``i`` to sample ``i+1``; the first recorded action has no preceding
    observed state and is deliberately excluded.  The nominal vector is only a
    ridge prior, never evaluator k/d/beta.
    """

    prior = np.asarray(NOMINAL, dtype=np.float64)
    parameter_scale = np.asarray((20.0, 2.0, 80.0), dtype=np.float64)
    predictions_r, predictions_v = [], []
    for root_index in range(split.roots):
        history = split.history[root_index].astype(np.float64)
        r_now, v_now = history[:-1, 0], history[:-1, 1]
        next_v = history[1:, 1]
        next_action = history[1:, 2]
        design = np.stack((r_now, v_now, r_now**3), axis=1)
        target = -(next_v - v_now) / DT - next_action
        scaled_design = design * parameter_scale[None, :]
        rhs = target - design @ prior
        system = scaled_design.T @ scaled_design + ridge * np.eye(3)
        correction = np.linalg.solve(system, scaled_design.T @ rhs)
        # Public plant-family support is a legitimate physical prior; the fit never
        # reads this root's evaluator parameters.
        k, d, beta = np.clip(
            prior + parameter_scale * correction,
            np.asarray((5.0, 0.5, 40.0)),
            np.asarray((35.0, 3.0, 120.0)),
        )
        root_rs, root_vs = [], []
        for candidate in range(CANDIDATES):
            r, v = float(history[-1, 0]), float(history[-1, 1])
            rs, vs = [], []
            for action in split.actions[root_index, candidate]:
                v = v + DT * (-float(action) - k * r - d * v - beta * r**3)
                r = r + DT * v
                rs.append(r)
                vs.append(v)
            root_rs.append(rs)
            root_vs.append(vs)
        predictions_r.append(root_rs)
        predictions_v.append(root_vs)
    return np.asarray(predictions_r, dtype=np.float32), np.asarray(predictions_v, dtype=np.float32)


def _select_physics_id_ridge(dev: SplitArrays, np):
    choices = (0.01, 0.1, 1.0, 10.0)
    rows = []
    for ridge in choices:
        pred_r, pred_v = _physics_id_prediction(dev, ridge, np)
        rows.append({"ridge": ridge, **_mae(pred_r, pred_v, dev, np)})
    best = min(rows, key=lambda row: row["r_mae"] + 0.2 * row["v_mae"])
    return float(best["ridge"]), rows


def _ensemble_prediction(models, split, torch, device, np, ablation="full"):
    predictions = [_predict_model(model, split, torch, device, ablation=ablation) for model in models]
    return (
        np.mean([item[0] for item in predictions], axis=0),
        np.mean([item[1] for item in predictions], axis=0),
    )


def _mae(pred_r, pred_v, split: SplitArrays, np) -> dict[str, float]:
    return {
        "r_mae": float(np.mean(np.abs(pred_r - split.truth_r))),
        "v_mae": float(np.mean(np.abs(pred_v - split.truth_v))),
    }


def _fit_scales(pred_r, split: SplitArrays, np):
    error = pred_r - split.truth_r
    return np.maximum(np.sqrt(np.mean(error**2, axis=(0, 2))), 1e-4)


def _calibrate(pred_r, split: SplitArrays, scales, alpha: float, np) -> dict[str, object]:
    normalized = np.abs(pred_r - split.truth_r) / scales[None, :, None]
    scores = np.max(normalized, axis=(1, 2))
    rank = int(math.ceil((split.roots + 1) * (1.0 - alpha)))
    q = float(np.sort(scores)[rank - 1]) if rank <= split.roots else None
    return {
        "alpha": alpha,
        "root_count": split.roots,
        "rank": rank,
        "q": q,
        "score": "root-max-over-four-candidates-and-40-times-absolute-r-error/dev-rmse-scale",
    }


def _bootstrap_mean_ci(values, np, seed: int = 20261002, draws: int = 1000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": None, "low": None, "high": None, "draws": draws}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "estimate": float(np.mean(values)),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "draws": draws,
        "resampling_unit": "root_with_all_four_sibling_candidates",
    }


def _evaluate(name: str, predictor: Callable[[SplitArrays], tuple[object, object]],
              dev: SplitArrays, cal: SplitArrays, test: SplitArrays, alpha: float, np,
              threshold: float = 0.12):
    dev_r, dev_v = predictor(dev)
    scales = _fit_scales(dev_r, dev, np)
    cal_r, _ = predictor(cal)
    calibration = _calibrate(cal_r, cal, scales, alpha, np)
    test_r, test_v = predictor(test)
    q = calibration["q"]
    if q is None:
        lower = np.full_like(test_r, -np.inf)
        upper = np.full_like(test_r, np.inf)
        covered = np.zeros(test.roots, dtype=bool)
    else:
        radius = q * scales[None, :, None]
        lower, upper = test_r - radius, test_r + radius
        covered = np.all((test.truth_r >= lower) & (test.truth_r <= upper), axis=(1, 2))
    branch_allowed = np.max(np.maximum(np.abs(lower), np.abs(upper)), axis=2) <= threshold
    truth_unsafe = np.max(np.abs(test.truth_r), axis=2) > threshold
    selected = np.argmax(branch_allowed, axis=1)
    has_allowed = np.any(branch_allowed, axis=1)
    selected_unsafe = truth_unsafe[np.arange(test.roots), selected]
    false_allow = has_allowed & selected_unsafe
    predicted_risk = np.max(np.abs(test_r), axis=2)
    truth_risk = np.max(np.abs(test.truth_r), axis=2)
    predicted_choice = np.argmin(predicted_risk, axis=1)
    oracle_choice = np.argmin(truth_risk, axis=1)
    ranking_regret = truth_risk[np.arange(test.roots), predicted_choice] - np.min(truth_risk, axis=1)
    allowed_regret = np.where(
        has_allowed,
        truth_risk[np.arange(test.roots), selected] - np.min(truth_risk, axis=1),
        np.nan,
    )
    r_mae_by_root = np.mean(np.abs(test_r - test.truth_r), axis=(1, 2))
    v_mae_by_root = np.mean(np.abs(test_v - test.truth_v), axis=(1, 2))
    result = {
        "name": name,
        "dev": _mae(dev_r, dev_v, dev, np),
        "test": _mae(test_r, test_v, test, np),
        "candidate_scales_from_dev": scales.astype(float).tolist(),
        "calibration": calibration,
        "test_joint_root_coverage": float(np.mean(covered)),
        "test_joint_root_covered": int(np.sum(covered)),
        "test_roots": test.roots,
        "decision_threshold_abs_r": threshold,
        "allowed_candidates": int(np.sum(branch_allowed)),
        "candidate_allow_rate": float(np.mean(branch_allowed)),
        "roots_with_an_allowed_candidate": int(np.sum(has_allowed)),
        "root_allow_rate": float(np.mean(has_allowed)),
        "root_false_allows": int(np.sum(false_allow)),
        "root_false_allow_rate_all_roots": float(np.mean(false_allow)),
        "root_false_allow_rate_given_allow": (
            float(np.sum(false_allow) / np.sum(has_allowed)) if np.any(has_allowed) else None
        ),
        "candidate_ranking": {
            "top1_matches_oracle": int(np.sum(predicted_choice == oracle_choice)),
            "top1_match_rate": float(np.mean(predicted_choice == oracle_choice)),
            "mean_abs_r_peak_regret": float(np.mean(ranking_regret)),
            "p95_abs_r_peak_regret": float(np.quantile(ranking_regret, 0.95)),
            "mean_permission_choice_regret_given_allow": (
                float(np.nanmean(allowed_regret)) if np.any(has_allowed) else None
            ),
        },
        "bootstrap_root_ci_95": {
            "r_mae": _bootstrap_mean_ci(r_mae_by_root, np, seed=20261003),
            "v_mae": _bootstrap_mean_ci(v_mae_by_root, np, seed=20261004),
            "joint_coverage": _bootstrap_mean_ci(covered.astype(float), np, seed=20261005),
            "root_allow_rate": _bootstrap_mean_ci(has_allowed.astype(float), np, seed=20261006),
            "root_false_allow_rate": _bootstrap_mean_ci(false_allow.astype(float), np, seed=20261007),
            "ranking_regret": _bootstrap_mean_ci(ranking_regret, np, seed=20261008),
        },
    }
    arrays = {
        "prediction_r": test_r.astype(np.float32),
        "truth_r": test.truth_r.astype(np.float32),
        "lower_r": lower.astype(np.float32),
        "upper_r": upper.astype(np.float32),
        "prediction_v": test_v.astype(np.float32),
        "truth_v": test.truth_v.astype(np.float32),
        "branch_allowed": branch_allowed.astype(np.uint8),
        "root_covered": covered.astype(np.uint8),
    }
    return result, arrays


def _inference_latency(models, split: SplitArrays, torch, device, np, repeats: int = 30):
    one = SplitArrays(
        split.name, split.root_ids[:1], split.root_hashes[:1], split.history[:1],
        split.actions[:1], split.truth_r[:1], split.truth_v[:1],
    )
    for _ in range(5):
        _ensemble_prediction(models, one, torch, device, np)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ensemble_prediction(models, one, torch, device, np)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "scope": "three-member ensemble, one root, four candidates, 40 autoregressive steps, host transfer included",
        "warmup": 5, "repeats": repeats,
        "p50_ms": float(np.quantile(timings, 0.50)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "samples_ms": timings,
    }


def _save_state(path: Path, state: dict[str, object], np) -> dict[str, str]:
    arrays, names = {}, {}
    for index, (name, tensor) in enumerate(sorted(state.items())):
        key = f"tensor_{index:03d}"
        arrays[key] = tensor.detach().cpu().numpy()
        names[key] = name
    np.savez_compressed(path, **arrays)
    return names


def _train_member(member_index: int, seed: int, train: SplitArrays, dev: SplitArrays,
                  normalization: dict[str, list[float]], args, torch, np, device, output: Path,
                  training_ablation: str = "full"):
    _seed_everything(seed, torch, np)
    model = _model_class(torch)(normalization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history, actions, truth_r, truth_v = _tensor_batches(train, torch, device)
    if training_ablation == "no_action":
        history = history.clone()
        history[:, :, 2] = 0.0
        actions = torch.zeros_like(actions)
    elif training_ablation != "full":
        raise ValueError(f"unknown training ablation: {training_ablation}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss, best_step, best_state = math.inf, 0, None
    prefix = "member" if training_ablation == "full" else f"{training_ablation}-member"
    log_path = output / f"{prefix}-{member_index}.jsonl"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            indices = torch.randint(0, history.shape[0], (args.batch_size,), generator=generator)
            indices = indices.to(device)
            model.train()
            pred_r, pred_v = model(history[indices], actions[indices])
            loss_r = torch.nn.functional.smooth_l1_loss(pred_r / BASE_SCALES[0], truth_r[indices] / BASE_SCALES[0])
            loss_v = torch.nn.functional.smooth_l1_loss(pred_v / BASE_SCALES[1], truth_v[indices] / BASE_SCALES[1])
            loss = loss_r + 0.2 * loss_v
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            if step % args.check_every == 0 or step == args.steps:
                dev_r, dev_v = _predict_model(model, dev, torch, device, ablation=training_ablation)
                dev_metrics = _mae(dev_r, dev_v, dev, np)
                dev_loss = dev_metrics["r_mae"] + 0.2 * dev_metrics["v_mae"]
                row = {
                    "step": step, "train_loss": float(loss.detach().cpu()),
                    "grad_norm_before_clip": float(grad_norm), **dev_metrics,
                    "elapsed_seconds": time.time() - started,
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps({"member": member_index, "training_ablation": training_ablation, **row}, sort_keys=True), flush=True)
                if dev_loss < best_loss:
                    best_loss, best_step = dev_loss, step
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no development checkpoint was evaluated")
    model.load_state_dict(best_state)
    checkpoint = output / f"{prefix}-{member_index}.npz"
    tensor_names = _save_state(checkpoint, best_state, np)
    return model, {
        "member": member_index, "seed": seed, "training_ablation": training_ablation,
        "best_step": best_step,
        "best_dev_objective": best_loss, "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint), "tensor_names": tensor_names,
        "log": log_path.name, "log_sha256": _sha256(log_path),
        "training_seconds": time.time() - started,
    }


def _environment(torch, np, device) -> dict[str, object]:
    cuda = None
    if torch.cuda.is_available():
        index = device.index if device.type == "cuda" and device.index is not None else 0
        cuda = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(index),
            "device_capability": list(torch.cuda.get_device_capability(index)),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
    return {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device), "cuda": cuda,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--train-roots", type=int, default=800)
    parser.add_argument("--dev-roots", type=int, default=120)
    parser.add_argument("--cal-roots", type=int, default=199)
    parser.add_argument("--test-roots", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root-seed", type=int, default=0)
    parser.add_argument("--source-commit", default=None, help="40-hex commit when running from a source archive")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true", help="25-step, small-root pipeline check")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.steps, args.train_roots, args.dev_roots, args.cal_roots, args.test_roots = 25, 16, 8, 19, 8
    if min(args.steps, args.train_roots, args.dev_roots, args.cal_roots, args.test_roots) <= 0:
        raise ValueError("steps and split root counts must be positive")
    if args.check_every <= 0 or args.batch_size <= 0 or not 0.0 < args.alpha < 1.0:
        raise ValueError("invalid optimization/calibration arguments")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("This experiment requires training-host packages numpy and torch") from exc

    device = _device(args.device, torch)
    source_commit = args.source_commit or _source_commit()
    if source_commit != "unknown" and (
        len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise ValueError("source commit must be a 40-hex Git commit")
    overall_started = time.time()
    dataset = make_numeric_dataset(
        args.train_roots, args.dev_roots, args.cal_roots, args.test_roots, args.root_seed,
    )
    splits = {name: _split_arrays(dataset, name, np) for name in ("train", "dev", "cal", "test")}
    normalization = _normalization(splits["train"], np)
    dataset_arrays = {}
    for name, split in splits.items():
        dataset_arrays.update({
            f"{name}_root_ids": split.root_ids,
            f"{name}_root_hashes": split.root_hashes,
            f"{name}_history": split.history,
            f"{name}_actions": split.actions,
            f"{name}_truth_r": split.truth_r,
            f"{name}_truth_v": split.truth_v,
        })
    np.savez_compressed(args.out / "dataset_arrays.npz", **dataset_arrays)
    split_manifest = {
        "dataset_hash": dataset.hash,
        "root_seed_ranges": {},
        "splits": {},
    }
    offset = args.root_seed
    for name in ("train", "dev", "cal", "test"):
        split = splits[name]
        split_manifest["root_seed_ranges"][name] = [offset, offset + split.roots - 1]
        offset += split.roots
        split_manifest["splits"][name] = {
            "count": split.roots, "root_ids": split.root_ids.tolist(),
            "root_hashes": split.root_hashes.tolist(),
        }
    _write_json(args.out / "dataset_manifest.json", split_manifest)

    models, members = [], []
    for index, seed in enumerate(MEMBER_SEEDS):
        model, record = _train_member(
            index, seed, splits["train"], splits["dev"], normalization,
            args, torch, np, device, args.out,
        )
        models.append(model)
        members.append(record)

    no_action_models, no_action_members = [], []
    for index, seed in enumerate(MEMBER_SEEDS):
        model, record = _train_member(
            index, seed, splits["train"], splits["dev"], normalization,
            args, torch, np, device, args.out, training_ablation="no_action",
        )
        no_action_models.append(model)
        no_action_members.append(record)

    ensemble = lambda split: _ensemble_prediction(models, split, torch, device, np)
    physics_id_ridge, physics_id_dev_rows = _select_physics_id_ridge(splits["dev"], np)
    predictors = {
        "gru_ensemble": ensemble,
        "fixed_nominal_physical_baseline": lambda split: _physical_prediction(split, np),
        "observable_history_physics_id": lambda split: _physics_id_prediction(split, physics_id_ridge, np),
        "persistence_baseline": lambda split: _persistence_prediction(split, np),
        "constant_velocity_baseline": lambda split: _constant_velocity_prediction(split, np),
        "no_action_trained_ensemble": lambda split: _ensemble_prediction(
            no_action_models, split, torch, device, np, "no_action"
        ),
        "full_model_action_zeroed_inference": lambda split: _ensemble_prediction(
            models, split, torch, device, np, "no_action"
        ),
        "full_model_action_shuffled_inference": lambda split: _ensemble_prediction(
            models, split, torch, device, np, "action_shuffle"
        ),
        "full_model_history_zeroed_inference": lambda split: _ensemble_prediction(
            models, split, torch, device, np, "no_history"
        ),
    }
    metrics = {}
    main_arrays = None
    for name, predictor in predictors.items():
        metrics[name], arrays = _evaluate(
            name, predictor, splits["dev"], splits["cal"], splits["test"], args.alpha, np,
        )
        if name == "gru_ensemble":
            main_arrays = arrays
    metrics["observable_history_physics_id"]["dev_ridge_selection"] = {
        "selected": physics_id_ridge,
        "candidates": physics_id_dev_rows,
        "selection_objective": "dev r_mae + 0.2 * dev v_mae",
        "history_action_alignment": "sample i+1 action drives observed transition i to i+1; first action excluded",
        "hidden_parameter_access": False,
        "parameter_support_prior": {
            "k": [5.0, 35.0], "d": [0.5, 3.0], "beta": [40.0, 120.0],
            "source": "published numeric plant family; per-root evaluator values are never read",
        },
    }
    metrics["ablation_semantics"] = {
        "no_action_trained_ensemble": "separately optimized members with past and future action channels zeroed",
        "full_model_action_zeroed_inference": "inference-only intervention; not a separately trained ablation",
        "full_model_action_shuffled_inference": "inference-only intervention; not a separately trained ablation",
        "full_model_history_zeroed_inference": "inference-only intervention; current observed r/v retained for integration",
    }
    metrics["gru_ensemble"]["inference_latency"] = _inference_latency(
        models, splits["test"], torch, device, np,
    )
    dev_best_r, dev_best_v = ensemble(splits["dev"])
    np.savez_compressed(
        args.out / "dev_best_predictions.npz",
        root_ids=splits["dev"].root_ids,
        root_hashes=splits["dev"].root_hashes,
        prediction_r=dev_best_r.astype(np.float32),
        prediction_v=dev_best_v.astype(np.float32),
        truth_r=splits["dev"].truth_r,
        truth_v=splits["dev"].truth_v,
    )
    prediction_path = args.out / "test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        root_ids=splits["test"].root_ids,
        root_hashes=splits["test"].root_hashes,
        candidate_names=np.asarray(["duration-0.6s", "duration-0.9s", "duration-1.2s", "duration-1.6s"]),
        **main_arrays,
    )
    _write_json(args.out / "metrics.json", metrics)
    manifest = {
        "schema": "sentinel-w0-numeric-gru-experiment-v1",
        "status": "complete",
        "claim_scope": "numeric W0 residual forecasting experiment; not 3D WorldGuard, robot safety, or certification",
        "source_commit": source_commit,
        "experiment_file_sha256": _sha256(Path(__file__)),
        "protocol": {
            "history": 8, "horizon": 40, "dt": DT, "candidate_count": 4,
            "architecture": "48d GRU history encoder + 48d GRUCell autoregressive rollout",
            "output": "tanh-bounded residual acceleration plus fixed nominal integration",
            "residual_acceleration_limit": RESIDUAL_ACCEL_LIMIT,
            "nominal_parameters": list(NOMINAL), "teacher_forcing": False,
            "known_integration_formula": "v_next=v+dt*(-a-20*r-1.75*v-80*r^3+learned_residual_accel); r_next=r+dt*v_next",
            "design_variant": (
                "v4 specifies bounded residual acceleration through known integration but does not require "
                "this fixed midpoint nominal reference; using (20,1.75,80) is an explicit W0 engineering "
                "variant and is separately compared with fixed and observable-history physics baselines"
            ),
            "member_seeds": list(MEMBER_SEEDS), "steps_per_member": args.steps,
            "trained_member_count": len(MEMBER_SEEDS) * 2,
            "batch_size": args.batch_size, "learning_rate": args.lr,
            "gradient_norm": args.grad_norm, "dev_check_every": args.check_every,
            "alpha": args.alpha, "decision_threshold_abs_r": 0.12,
            "npz_loading": "all arrays are numeric or fixed-width unicode and load with allow_pickle=False",
        },
        "normalization": normalization,
        "members": members,
        "no_action_trained_members": no_action_members,
        "environment": _environment(torch, np, device),
        "elapsed_seconds": time.time() - overall_started,
        "artifacts": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete", "out": str(args.out), "elapsed_seconds": manifest["elapsed_seconds"],
        "gru_test": metrics["gru_ensemble"]["test"],
        "joint_coverage": metrics["gru_ensemble"]["test_joint_root_coverage"],
        "root_false_allows": metrics["gru_ensemble"]["root_false_allows"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
