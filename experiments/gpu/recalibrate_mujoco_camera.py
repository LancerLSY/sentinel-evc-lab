#!/usr/bin/env python3
"""Engineering recalibration control for the existing W2 shifted-camera stress set.

This freezes the trained visual ensemble, train-only PCA, normalizers, labels,
and shifted test features from a completed run.  It renders only shifted-camera
dev/cal histories, refits residual scales and alpha=.05 quantiles on those
profile-matched splits, then evaluates the already sealed shifted test once.
The result is not a new independent test and is not model selection evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_mujoco_visual as visual  # noqa: E402
import train_mujoco_world as w2  # noqa: E402


def _sha(path: Path) -> str:
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
                               allow_nan=False, default=scalar) + "\n")


def _load_models(source: Path, source_manifest: dict, torch, np, device):
    Model = visual._model_class(torch)
    models = []
    records = source_manifest["training"]["visual_members"]
    if len(records) != 3:
        raise ValueError("source visual ensemble must have exactly three members")
    for record in records:
        checkpoint = source / record["checkpoint"]
        if _sha(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint.name}")
        with np.load(checkpoint, allow_pickle=False) as archive:
            state = {
                name: torch.from_numpy(archive[key])
                for key, name in record["tensor_names"].items()
            }
        model = Model(source_manifest["normalizers"], True).to(device)
        model.load_state_dict(state, strict=True); model.eval(); models.append(model)
    return models


def _render_profile(split_name, split, pca, encoder, args,
                    mujoco, np, torch, torchvision, device):
    raw_path = args.out / f"{split_name}-shift-layer4.npy"
    render_args = SimpleNamespace(out=args.out, render_batch_size=args.render_batch_size)
    raw, metadata = visual._render_raw(
        f"{split_name}_shifted", split, visual.SHIFTED_CAMERAS, raw_path,
        encoder, render_args, mujoco, np, torch, torchvision, device,
    )
    features = visual._project(raw, pca, np)
    del raw; raw_path.unlink()
    return features, metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("--out must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    import mujoco
    import numpy as np
    import torch
    import torchvision

    device = torch.device(args.device)
    source_manifest_path = args.source_run / "manifest.json"
    source_metrics_path = args.source_run / "metrics.json"
    cache_path = args.source_run / "visual_cache.npz"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_metrics = json.loads(source_metrics_path.read_text())
    if source_manifest.get("status") != "complete" or source_manifest.get("preflight"):
        raise ValueError("source must be the completed formal visual run")
    if _sha(cache_path) != source_manifest["visual"]["cache_sha256"]:
        raise ValueError("source visual cache hash mismatch")
    splits, root_ids, dataset_manifest, dataset_hashes = w2._load_dataset_run(
        args.dataset_run, np
    )
    if dataset_hashes != {
        key: source_manifest["dataset"][key]
        for key in ("dataset_sha256", "dataset_manifest_sha256")
    }:
        raise ValueError("dataset differs from the frozen source run")
    with np.load(cache_path, allow_pickle=False) as cache:
        for name in ("dev", "cal", "test"):
            if not np.array_equal(cache[f"{name}_root_ids"], root_ids[name]):
                raise ValueError(f"source visual cache root order mismatch: {name}")
        pca = [
            (cache[f"pca_{camera}_mean"], cache[f"pca_{camera}_components"],
             cache[f"pca_{camera}_eigenvalues"])
            for camera in range(2)
        ]
        sealed_test_features = cache["test_layout_shift_features"].copy()
    encoder, encoder_sha = visual._encoder(args.weights, torch, torchvision, device)
    if encoder_sha != source_manifest["visual"]["encoder_weights_sha256"]:
        raise ValueError("encoder differs from the frozen source run")
    rendering = {}
    for name in ("dev", "cal"):
        splits[name]["visual"], rendering[name] = _render_profile(
            name, splits[name], pca, encoder, args,
            mujoco, np, torch, torchvision, device,
        )
    del encoder
    splits["test"]["visual"] = sealed_test_features
    models = _load_models(args.source_run, source_manifest, torch, np, device)
    test_calls = 0

    def predictor(split):
        nonlocal test_calls
        if split is splits["test"]:
            test_calls += 1
            if test_calls > 1:
                raise RuntimeError("sealed shifted test may be evaluated only once")
        return visual._predict(models, split, torch, device, np)

    metric, arrays = w2._calibrated_metrics(
        "shifted_camera_profile_recalibrated_visual_ensemble",
        predictor, splits["dev"], splits["cal"], splits["test"], np, alpha=0.05,
    )
    metric["selection_denominators"] = visual._selection_report(
        arrays, splits["test"], np
    )
    metric["protocol"] = {
        "status": "engineering recalibration control on an existing stress set",
        "independent_new_test": False,
        "model_training_pca_normalizers_frozen": True,
        "shifted_dev_roots": 200,
        "shifted_cal_roots": 299,
        "sealed_shifted_test_roots": 500,
        "sealed_test_evaluations": test_calls,
        "test_used_for_parameters_scales_or_thresholds": False,
        "alpha": 0.05,
    }
    original = source_metrics["visual_camera_layout_shift_test_only"]
    metrics = {
        "recalibrated_shifted_profile": metric,
        "frozen_original_profile_calibration_on_same_shifted_test": original,
        "comparison_scope": (
            "same pre-existing shifted-camera stress test; compares profile-mismatched original "
            "calibration with profile-matched dev/cal recalibration; not a new generalization test"
        ),
    }
    _write_json(args.out / "metrics.json", metrics)
    np.savez_compressed(
        args.out / "sealed_test_recalibrated_predictions.npz",
        root_ids=root_ids["test"], truth=splits["test"]["truth"],
        risk_window=splits["test"]["risk_window"], risk_posthold=splits["test"]["risk_posthold"],
        **arrays,
    )
    manifest = {
        "schema": "sentinel-w2-shifted-camera-recalibration-control-v1",
        "status": "complete",
        "scope": (
            "engineering recalibration recovery on the existing shifted-camera stress set; "
            "not a new independent test and not model tuning"
        ),
        "script_sha256": _sha(Path(__file__)),
        "source_run": str(args.source_run.resolve()),
        "source_manifest_sha256": _sha(source_manifest_path),
        "source_metrics_sha256": _sha(source_metrics_path),
        "source_visual_cache_sha256": _sha(cache_path),
        "dataset": {**dataset_hashes, "counts": dataset_manifest["counts"]},
        "frozen": {
            "visual_checkpoint_sha256": [item["checkpoint_sha256"] for item in
                                           source_manifest["training"]["visual_members"]],
            "encoder_weights_sha256": encoder_sha,
            "pca_from_source_visual_cache": True,
            "normalizers_from_source_manifest": True,
            "shifted_test_features_from_sealed_source_cache": True,
        },
        "rendering": rendering,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device)},
        "elapsed_seconds": time.time() - started,
        "artifacts": {},
    }
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size, "sha256": _sha(path)
            }
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "test_mae_15d": metric["test_mae_15d"],
                      "selection": metric["selection_denominators"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
