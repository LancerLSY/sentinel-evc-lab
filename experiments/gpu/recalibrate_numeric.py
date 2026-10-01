#!/usr/bin/env python3
"""Recalibrate the frozen W0 ensemble at alpha=.05 on fresh roots.

The historical alpha=.10 run remains untouched.  This command loads its six
numeric NPZ checkpoints, frozen train-only normalization, frozen dev roots and
candidate scales; it never updates a model parameter or reads fresh test roots
while selecting any model, scale, rule or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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


def _split_from_archive(archive, name, np):
    return w0.SplitArrays(
        name=name,
        root_ids=archive[f"{name}_root_ids"],
        root_hashes=archive[f"{name}_root_hashes"],
        history=archive[f"{name}_history"].astype(np.float32),
        actions=archive[f"{name}_actions"].astype(np.float32),
        truth_r=archive[f"{name}_truth_r"].astype(np.float32),
        truth_v=archive[f"{name}_truth_v"].astype(np.float32),
    )


def _fresh_split(name, start, count, np):
    dataset = NumericDataset(tuple(_make_root(seed, name) for seed in range(start, start + count)))
    return w0._split_arrays(dataset, name, np)


def _load_members(run, records, normalization, torch, device, np):
    models, files = [], []
    for record in records:
        path = run / record["checkpoint"]
        if not path.is_file() or _sha(path) != record["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint digest mismatch: {path.name}")
        state = {}
        with np.load(path, allow_pickle=False) as archive:
            for key, name in record["tensor_names"].items():
                state[name] = torch.from_numpy(archive[key].copy())
        model = w0._model_class(torch)(normalization).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
        files.append({"name": path.name, "sha256": _sha(path)})
    return models, files


def _recalibrate(name, predictor, frozen_scales, dev, cal, test, np, alpha=.05):
    frozen_scales = np.asarray(frozen_scales, dtype=np.float64)
    dev_r, _ = predictor(dev)
    reproduced = w0._fit_scales(dev_r, dev, np)
    if not np.allclose(reproduced, frozen_scales, rtol=2e-5, atol=1e-8):
        raise ValueError(f"{name} frozen dev scales do not reproduce")
    cal_r, _ = predictor(cal)
    scores = np.max(np.abs(cal_r - cal.truth_r) / frozen_scales[None, :, None], axis=(1, 2))
    rank = math.ceil((cal.roots + 1) * (1.0 - alpha))
    if rank > cal.roots:
        raise ValueError("requested alpha has no finite conformal rank")
    q = float(np.sort(scores)[rank - 1])
    test_r, test_v = predictor(test)
    radius = q * frozen_scales[None, :, None]
    lower, upper = test_r - radius, test_r + radius
    covered = np.all((test.truth_r >= lower) & (test.truth_r <= upper), axis=(1, 2))
    branch_allowed = np.max(np.maximum(np.abs(lower), np.abs(upper)), axis=2) <= 0.12
    truth_unsafe = np.max(np.abs(test.truth_r), axis=2) > 0.12
    selected = np.argmax(branch_allowed, axis=1)
    has_allowed = np.any(branch_allowed, axis=1)
    false_allow = has_allowed & truth_unsafe[np.arange(test.roots), selected]
    r_root = np.mean(np.abs(test_r - test.truth_r), axis=(1, 2))
    return {
        "name": name,
        "frozen_candidate_scales": frozen_scales.tolist(),
        "fresh_calibration": {"alpha": alpha, "roots": cal.roots, "rank": rank, "q": q,
                              "score": "root-max-over-four-candidates-and-40-times-absolute-r-error/dev-scale"},
        "test": w0._mae(test_r, test_v, test, np),
        "test_joint_root_coverage": float(np.mean(covered)),
        "test_joint_root_covered": int(np.sum(covered)),
        "root_allow_rate": float(np.mean(has_allowed)),
        "root_false_allows": int(np.sum(false_allow)),
        "root_false_allow_rate_all_roots": float(np.mean(false_allow)),
        "bootstrap_root_ci_95": {
            "r_mae": w0._bootstrap_mean_ci(r_root, np, seed=20261020),
            "coverage": w0._bootstrap_mean_ci(covered.astype(float), np, seed=20261021),
            "allow_rate": w0._bootstrap_mean_ci(has_allowed.astype(float), np, seed=20261022),
            "false_allow_rate": w0._bootstrap_mean_ci(false_allow.astype(float), np, seed=20261023),
        },
    }, {"prediction_r": test_r.astype(np.float32), "prediction_v": test_v.astype(np.float32),
               "truth_r": test.truth_r, "truth_v": test.truth_v,
               "lower_r": lower.astype(np.float32), "upper_r": upper.astype(np.float32),
               "covered": covered.astype(np.uint8), "branch_allowed": branch_allowed.astype(np.uint8)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path,
                        help="completed W0 alpha=.10 training output")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    args.run, args.out = args.run.resolve(), args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("recalibration requires numpy and torch") from exc
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto": device_name = "cpu"
    device = torch.device(device_name)
    manifest_path = args.run / "manifest.json"
    metrics_path = args.run / "metrics.json"
    dataset_path = args.run / "dataset_arrays.npz"
    for path in (manifest_path, metrics_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    old_metrics = json.loads(metrics_path.read_text())
    if manifest.get("protocol", {}).get("alpha") != 0.1:
        raise ValueError("input is not the preserved alpha=.10 W0 run")
    artifact = manifest.get("artifacts", {}).get(dataset_path.name)
    if artifact and artifact.get("sha256") != _sha(dataset_path):
        raise ValueError("frozen W0 dataset digest mismatch")
    with np.load(dataset_path, allow_pickle=False) as archive:
        dev = _split_from_archive(archive, "dev", np)
    cal = _fresh_split("cal", 500000, 199, np)
    test = _fresh_split("test", 600000, 300, np)
    normalization = manifest["normalization"]
    main_models, main_files = _load_members(args.run, manifest["members"], normalization,
                                             torch, device, np)
    no_action_models, no_action_files = _load_members(
        args.run, manifest["no_action_trained_members"], normalization, torch, device, np,
    )
    if len(main_models) != 3 or len(no_action_models) != 3:
        raise ValueError("the frozen W0 run must contain three main and three no-action members")
    ridge = old_metrics["observable_history_physics_id"]["dev_ridge_selection"]["selected"]
    predictors = {
        "gru_ensemble": lambda split: w0._ensemble_prediction(main_models, split, torch, device, np),
        "fixed_nominal_physical_baseline": lambda split: w0._physical_prediction(split, np),
        "observable_history_physics_id": lambda split: w0._physics_id_prediction(split, ridge, np),
        "persistence_baseline": lambda split: w0._persistence_prediction(split, np),
        "constant_velocity_baseline": lambda split: w0._constant_velocity_prediction(split, np),
        "no_action_trained_ensemble": lambda split: w0._ensemble_prediction(
            no_action_models, split, torch, device, np, "no_action"),
        "full_model_action_zeroed_inference": lambda split: w0._ensemble_prediction(
            main_models, split, torch, device, np, "no_action"),
        "full_model_action_shuffled_inference": lambda split: w0._ensemble_prediction(
            main_models, split, torch, device, np, "action_shuffle"),
        "full_model_history_zeroed_inference": lambda split: w0._ensemble_prediction(
            main_models, split, torch, device, np, "no_history"),
    }
    recalibrated, main_arrays = {}, None
    for name, predictor in predictors.items():
        scales = old_metrics[name]["candidate_scales_from_dev"]
        recalibrated[name], arrays = _recalibrate(name, predictor, scales, dev, cal, test, np)
        if name == "gru_ensemble":
            main_arrays = arrays
            main_calibration_scores = (
                np.max(
                    np.abs(predictor(cal)[0] - cal.truth_r) /
                    np.asarray(scales)[None, :, None], axis=(1, 2)
                ).astype(np.float32)
            )
    _write_json(args.out / "metrics.json", recalibrated)
    np.savez_compressed(args.out / "calibration_scores.npz", root_ids=cal.root_ids,
                        root_hashes=cal.root_hashes, root_max_scores=main_calibration_scores)
    np.savez_compressed(args.out / "test_predictions.npz", root_ids=test.root_ids,
                        root_hashes=test.root_hashes, **main_arrays)
    split_manifest = {
        "dev": {"source": "frozen alpha=.10 run", "count": dev.roots,
                "root_ids": dev.root_ids.tolist(), "root_hashes": dev.root_hashes.tolist()},
        "cal": {"seed_range": [500000, 500198], "count": cal.roots,
                "root_ids": cal.root_ids.tolist(), "root_hashes": cal.root_hashes.tolist()},
        "test": {"seed_range": [600000, 600299], "count": test.roots,
                 "root_ids": test.root_ids.tolist(), "root_hashes": test.root_hashes.tolist()},
    }
    _write_json(args.out / "split_manifest.json", split_manifest)
    output = {
        "schema": "sentinel-w0-frozen-recalibration-v1", "status": "complete",
        "purpose": "design-fixed alpha=.05 recalibration; historical alpha=.10 artifacts retained",
        "selection": "weights, train normalization, dev roots, candidate scales and physics-ID ridge frozen",
        "fresh_calibration": {"alpha": 0.05, "roots": 199, "rank": 190,
                              "seed_range": [500000, 500198]},
        "fresh_test": {"roots": 300, "seed_range": [600000, 600299]},
        "input": {"manifest_sha256": _sha(manifest_path), "metrics_sha256": _sha(metrics_path),
                  "dataset_sha256": _sha(dataset_path), "checkpoints": main_files + no_action_files},
        "source": {"commit": _commit(), "frozen_training_script_sha256": _sha(HERE / "train_numeric_gru.py"),
                   "recalibration_script_sha256": _sha(Path(__file__))},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "device": str(device)},
        "artifacts": {}, "elapsed_seconds": time.time() - started,
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file():
            output["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    _write_json(args.out / "manifest.json", output)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "gru": recalibrated["gru_ensemble"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
