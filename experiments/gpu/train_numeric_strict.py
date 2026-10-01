#!/usr/bin/env python3
"""Train the v4-strict W0 numeric model against frozen W0 references.

Strict forward dynamics are exactly
``v_next=v+dt*(-a+8*tanh(head(h))); r_next=r+dt*v_next``.  No k, d or beta
value, estimate, midpoint or label enters this model.  Training batches sample
64 roots and retain all four sibling branches.  Because both architecture and
batching differ from the earlier W0 run, results are a combined-configuration
comparison and must not be reported as a single-factor causal ablation.
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import train_numeric_gru as w0  # noqa: E402
from sentinel_evc.data import NumericDataset, _make_root  # noqa: E402


EXPECTED_PARENT_SHA = "5927981711ba8b1794470d6228d1878331d7c15c82f7e55f531f8c0f19e09d42"
DT, HISTORY, HORIZON, CANDIDATES = 0.05, 8, 40, 4
SEEDS = (401, 503, 607)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _archive_split(archive, name, np):
    return w0.SplitArrays(
        name, archive[f"{name}_root_ids"], archive[f"{name}_root_hashes"],
        archive[f"{name}_history"].astype(np.float32),
        archive[f"{name}_actions"].astype(np.float32),
        archive[f"{name}_truth_r"].astype(np.float32),
        archive[f"{name}_truth_v"].astype(np.float32),
    )


def _fresh(name, start, count, np):
    roots = NumericDataset(tuple(_make_root(seed, name) for seed in range(start, start + count)))
    return w0._split_arrays(roots, name, np)


def _strict_class(torch):
    nn = torch.nn

    class StrictResidualWorld(nn.Module):
        def __init__(self, normalization):
            super().__init__()
            self.encoder = nn.GRU(3, 48, batch_first=True)
            self.decoder = nn.GRUCell(3, 48)
            self.head = nn.Sequential(nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))
            self.register_buffer("base", torch.tensor(normalization["base_scales"], dtype=torch.float32))
            self.register_buffer("mean", torch.tensor(normalization["mean_after_base_scaling"], dtype=torch.float32))
            self.register_buffer("std", torch.tensor(normalization["std_after_base_scaling"], dtype=torch.float32))

        def norm(self, values):
            return (values / self.base - self.mean) / self.std

        def forward(self, history, actions, no_action=False):
            if no_action:
                history = history.clone()
                history[:, :, 2] = 0.0
                actions = torch.zeros_like(actions)
            _, hidden = self.encoder(self.norm(history))
            hidden = hidden[0]
            r, v = history[:, -1, 0], history[:, -1, 1]
            rs, vs = [], []
            for step in range(actions.shape[1]):
                action = actions[:, step]
                hidden = self.decoder(self.norm(torch.stack((r, v, action), -1)), hidden)
                acceleration = -action + 8.0 * torch.tanh(self.head(hidden)[:, 0])
                v = v + DT * acceleration
                r = r + DT * v
                rs.append(r); vs.append(v)
            return torch.stack(rs, 1), torch.stack(vs, 1)

    return StrictResidualWorld


def _root_tensors(split, torch, device):
    return (torch.from_numpy(split.history).to(device),
            torch.from_numpy(split.actions).to(device),
            torch.from_numpy(split.truth_r).to(device),
            torch.from_numpy(split.truth_v).to(device))


def _strict_predict(models, split, torch, device, np, no_action=False):
    history, actions, _, _ = _root_tensors(split, torch, device)
    rows_r, rows_v = [], []
    for model in models:
        model.eval(); member_r, member_v = [], []
        with torch.inference_mode():
            for start in range(0, split.roots, 128):
                stop = start + 128
                batch_history = history[start:stop, None].expand(-1, CANDIDATES, -1, -1).reshape(-1, HISTORY, 3)
                batch_actions = actions[start:stop].reshape(-1, HORIZON)
                pred_r, pred_v = model(batch_history, batch_actions, no_action=no_action)
                member_r.append(pred_r.cpu()); member_v.append(pred_v.cpu())
        shape = (split.roots, CANDIDATES, HORIZON)
        rows_r.append(torch.cat(member_r).numpy().reshape(shape))
        rows_v.append(torch.cat(member_v).numpy().reshape(shape))
    return np.mean(rows_r, axis=0), np.mean(rows_v, axis=0)


def _save_state(path, state, np):
    arrays, names = {}, {}
    for index, (name, tensor) in enumerate(sorted(state.items())):
        key = f"tensor_{index:03d}"
        arrays[key], names[key] = tensor.numpy(), name
    np.savez_compressed(path, **arrays)
    return names


def _train_member(member, seed, train, dev, normalization, args, torch, np,
                  device, out, no_action=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = _strict_class(torch)(normalization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-5)
    history, actions, truth_r, truth_v = _root_tensors(train, torch, device)
    bag_rng = np.random.default_rng(seed)
    root_bag = bag_rng.integers(0, train.roots, train.roots, dtype=np.int64)
    sample_rng = np.random.default_rng(seed + 10000)
    best, best_step, best_state = math.inf, None, None
    prefix = "no-action-" if no_action else ""
    log_path = out / f"strict-{prefix}member-{member}.jsonl"
    started = time.time()
    with log_path.open("w") as log:
        for step in range(1, args.steps + 1):
            roots = root_bag[sample_rng.integers(0, len(root_bag), 64)]
            root_index = torch.from_numpy(roots).to(device)
            batch_history = history[root_index, None].expand(-1, CANDIDATES, -1, -1).reshape(-1, HISTORY, 3)
            batch_actions = actions[root_index].reshape(-1, HORIZON)
            batch_r = truth_r[root_index].reshape(-1, HORIZON)
            batch_v = truth_v[root_index].reshape(-1, HORIZON)
            model.train()
            pred_r, pred_v = model(batch_history, batch_actions, no_action=no_action)
            loss_r = torch.nn.functional.smooth_l1_loss(pred_r / 0.15, batch_r / 0.15)
            loss_v = torch.nn.functional.smooth_l1_loss(pred_v / 0.6, batch_v / 0.6)
            loss = loss_r + 0.2 * loss_v
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step % 25 == 0 or step == args.steps:
                dev_r, dev_v = _strict_predict([model], dev, torch, device, np, no_action)
                dev_metric = w0._mae(dev_r, dev_v, dev, np)
                objective = dev_metric["r_mae"] + 0.2 * dev_metric["v_mae"]
                row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad),
                       **dev_metric, "elapsed_seconds": time.time() - started}
                log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"phase": "strict-train", "member": member,
                                  "no_action": no_action, **row}), flush=True)
                if objective < best:
                    best, best_step = objective, step
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no dev-best checkpoint; calibration forbidden")
    model.load_state_dict(best_state)
    path = out / f"strict-{prefix}member-{member}.npz"
    tensor_names = _save_state(path, best_state, np)
    bag_path = out / f"strict-{prefix}member-{member}-root-bootstrap.npz"
    np.savez_compressed(bag_path, root_indices=root_bag,
                        root_ids=train.root_ids[root_bag], root_hashes=train.root_hashes[root_bag])
    return model, {"member": member, "seed": seed, "best_step": best_step,
                   "best_dev_objective": best, "checkpoint": path.name,
                   "checkpoint_sha256": _sha(path), "tensor_names": tensor_names,
                   "log": log_path.name, "log_sha256": _sha(log_path),
                   "root_bootstrap": bag_path.name, "root_bootstrap_sha256": _sha(bag_path),
                   "sampled_roots": train.roots, "siblings_per_sampled_root": 4,
                   "independently_trained_no_action": no_action}


def _load_w0_models(reference, manifest, torch, device, np):
    models, provenance = [], []
    for record in manifest["members"]:
        path = reference / record["checkpoint"]
        if _sha(path) != record["checkpoint_sha256"]:
            raise ValueError(f"reference checkpoint mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            state = {name: torch.from_numpy(archive[key].copy())
                     for key, name in record["tensor_names"].items()}
        model = w0._model_class(torch)(manifest["normalization"]).to(device)
        model.load_state_dict(state, strict=True); model.eval(); models.append(model)
        provenance.append({"name": path.name, "sha256": _sha(path)})
    return models, provenance


def _validate_unused_reference_weights(reference, records, np):
    files = []
    for record in records:
        path = reference / record["checkpoint"]
        if _sha(path) != record["checkpoint_sha256"]:
            raise ValueError(f"reference checkpoint mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(record["tensor_names"]):
                raise ValueError(f"reference checkpoint tensor set mismatch: {path.name}")
            if any(archive[key].dtype == object for key in archive.files):
                raise ValueError(f"unsafe object tensor in reference checkpoint: {path.name}")
        files.append({"name": path.name, "sha256": _sha(path), "used_for_prediction": False})
    return files


def _evaluate(name, predictor, dev, cal, test, np, alpha=.05):
    dev_r, _ = predictor(dev)
    scales = w0._fit_scales(dev_r, dev, np)
    cal_r, _ = predictor(cal)
    scores = np.max(np.abs(cal_r - cal.truth_r) / scales[None, :, None], axis=(1, 2))
    rank = math.ceil((cal.roots + 1) * (1 - alpha))
    q = float(np.sort(scores)[rank - 1]) if rank <= cal.roots else None
    test_r, test_v = predictor(test)
    radius = np.inf if q is None else q * scales[None, :, None]
    lower, upper = test_r - radius, test_r + radius
    covered = np.all((test.truth_r >= lower) & (test.truth_r <= upper), axis=(1, 2))
    allowed = np.max(np.maximum(np.abs(lower), np.abs(upper)), axis=2) <= 0.12
    selected = np.argmax(allowed, axis=1); has_allowed = np.any(allowed, axis=1)
    unsafe = np.max(np.abs(test.truth_r), axis=2) > 0.12
    false_allow = has_allowed & unsafe[np.arange(test.roots), selected]
    return {"name": name, "dev_scales": scales.tolist(),
            "calibration": {"alpha": alpha, "roots": cal.roots, "rank": rank, "q": q},
            "test": w0._mae(test_r, test_v, test, np),
            "joint_root_coverage": float(np.mean(covered)),
            "root_allow_rate": float(np.mean(has_allowed)),
            "root_false_allows": int(np.sum(false_allow)),
            "raw_negative": {"rejected_roots": int(np.sum(~has_allowed)),
                             "uncovered_roots": int(np.sum(~covered)),
                             "false_allow_roots": int(np.sum(false_allow))}}, {
                "prediction_r": test_r.astype(np.float32), "prediction_v": test_v.astype(np.float32),
                "truth_r": test.truth_r, "truth_v": test.truth_v,
                "lower_r": lower.astype(np.float32), "upper_r": upper.astype(np.float32),
                "covered": covered.astype(np.uint8), "allowed": allowed.astype(np.uint8),
                "has_allowed": has_allowed.astype(np.uint8),
                "false_allow": false_allow.astype(np.uint8),
                "calibration_scores": scores.astype(np.float32)}


def _latency(models, test, torch, device, np):
    one = w0.SplitArrays(test.name, test.root_ids[:1], test.root_hashes[:1], test.history[:1],
                         test.actions[:1], test.truth_r[:1], test.truth_v[:1])
    for _ in range(5): _strict_predict(models, one, torch, device, np)
    samples = []
    for _ in range(30):
        started = time.perf_counter(); _strict_predict(models, one, torch, device, np)
        if device.type == "cuda": torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return {"scope": "3-member ensemble, one root, four siblings, 40 steps, host transfer included",
            "p50_ms": float(np.quantile(samples, .5)), "p95_ms": float(np.quantile(samples, .95)),
            "samples_ms": samples}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--source-commit")
    parser.add_argument("--train-no-action", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); started = time.time()
    reference, out = args.reference_run.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("strict training requires numpy and torch") from exc
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    manifest_path, metrics_path = reference / "manifest.json", reference / "metrics.json"
    dataset_path, data_manifest_path = reference / "dataset_arrays.npz", reference / "dataset_manifest.json"
    manifest, old_metrics = json.loads(manifest_path.read_text()), json.loads(metrics_path.read_text())
    if _sha(HERE / "train_numeric_gru.py") != EXPECTED_PARENT_SHA:
        raise ValueError("local frozen W0 source hash changed")
    if manifest.get("experiment_file_sha256") != EXPECTED_PARENT_SHA:
        raise ValueError("reference run was not produced by the frozen W0 source")
    for path in (dataset_path, data_manifest_path, metrics_path):
        artifact = manifest["artifacts"][path.name]
        if _sha(path) != artifact["sha256"]:
            raise ValueError(f"reference artifact digest mismatch: {path.name}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        train, dev = _archive_split(archive, "train", np), _archive_split(archive, "dev", np)
    if train.roots != 800 or dev.roots != 120:
        raise ValueError("strict experiment requires frozen 800/120 train/dev roots")
    strict_models, members = [], []
    for member, seed in enumerate(SEEDS):
        model, record = _train_member(member, seed, train, dev, manifest["normalization"],
                                      args, torch, np, device, out)
        strict_models.append(model); members.append(record)
    no_action_models, no_action_members = [], []
    if args.train_no_action:
        for member, seed in enumerate(SEEDS):
            model, record = _train_member(member, seed, train, dev, manifest["normalization"],
                                          args, torch, np, device, out, no_action=True)
            no_action_models.append(model); no_action_members.append(record)
    old_models, old_weight_files = _load_w0_models(reference, manifest, torch, device, np)
    old_weight_files += _validate_unused_reference_weights(
        reference, manifest["no_action_trained_members"], np,
    )
    # Fresh calibration/test roots are first materialized only after all strict
    # dev-best weights are frozen; they cannot affect fitting or selection.
    cal, test = _fresh("cal", 800000, 199, np), _fresh("test", 900000, 300, np)
    ridge = old_metrics["observable_history_physics_id"]["dev_ridge_selection"]["selected"]
    predictors = {
        "strict_root_bootstrap": lambda split: _strict_predict(strict_models, split, torch, device, np),
        "frozen_engineering_variant": lambda split: w0._ensemble_prediction(old_models, split, torch, device, np),
        "observable_history_physics_id": lambda split: w0._physics_id_prediction(split, ridge, np),
    }
    if no_action_models:
        predictors["strict_independently_trained_no_action"] = lambda split: _strict_predict(
            no_action_models, split, torch, device, np, no_action=True)
    metrics, per_model = {}, {}
    for name, predictor in predictors.items():
        metrics[name], per_model[name] = _evaluate(name, predictor, dev, cal, test, np)
    metrics["strict_root_bootstrap"]["latency"] = _latency(strict_models, test, torch, device, np)
    metrics["interpretation"] = {
        "strict_difference": "architecture and root-level sibling-preserving batching changed together",
        "causal_claim": "none; this is not a single-factor ablation",
        "test_selection": "test roots were not read before weights, dev scales and calibration were frozen",
    }
    _write_json(out / "metrics.json", metrics)
    arrays = {}
    for name, values in per_model.items():
        for key, value in values.items(): arrays[f"{name}_{key}"] = value
    np.savez_compressed(out / "per_root_predictions.npz", test_root_ids=test.root_ids,
                        test_root_hashes=test.root_hashes, cal_root_ids=cal.root_ids,
                        cal_root_hashes=cal.root_hashes, **arrays)
    _write_json(out / "split_manifest.json", {
        "train": {"source": "frozen reference", "count": train.roots,
                  "root_ids": train.root_ids.tolist(), "root_hashes": train.root_hashes.tolist()},
        "dev": {"source": "frozen reference", "count": dev.roots,
                "root_ids": dev.root_ids.tolist(), "root_hashes": dev.root_hashes.tolist()},
        "cal": {"seed_range": [800000, 800198], "count": cal.roots,
                "root_ids": cal.root_ids.tolist(), "root_hashes": cal.root_hashes.tolist()},
        "test": {"seed_range": [900000, 900299], "count": test.roots,
                 "root_ids": test.root_ids.tolist(), "root_hashes": test.root_hashes.tolist()},
    })
    result = {
        "schema": "sentinel-w0-v4-strict-comparison-v1", "status": "complete",
        "strict_forward": "v_next=v+dt*(-a+8*tanh(head(h))); r_next=r+dt*v_next",
        "hidden_or_nominal_parameters_in_strict_forward": False, "teacher_forcing": False,
        "root_batch": "fixed 800-root bootstrap/member; each step samples 64 roots and all 4 siblings",
        "calibration": "fresh seeds 800000..800198, alpha=.05, n=199, rank190",
        "test": "fresh unread seeds 900000..900299",
        "combined_change_warning": "architecture and batching differ; no single-factor attribution",
        "members": members, "no_action_members": no_action_members,
        "provenance": {"reference_manifest_sha256": _sha(manifest_path),
                       "reference_dataset_sha256": _sha(dataset_path),
                       "reference_dataset_manifest_sha256": _sha(data_manifest_path),
                       "reference_weights": old_weight_files,
                       "frozen_parent_source_sha256": EXPECTED_PARENT_SHA,
                       "strict_source_sha256": _sha(Path(__file__)),
                       "source_commit": args.source_commit or _commit()},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "device": str(device)},
        "elapsed_seconds": time.time() - started, "artifacts": {},
    }
    for path in sorted(out.iterdir()):
        if path.is_file(): result["artifacts"][path.name] = {"bytes": path.stat().st_size,
                                                             "sha256": _sha(path)}
    _write_json(out / "manifest.json", result)
    print(json.dumps({"status": "complete", "out": str(out), "metrics": metrics},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
