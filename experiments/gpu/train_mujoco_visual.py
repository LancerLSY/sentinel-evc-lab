#!/usr/bin/env python3
"""W2 MuJoCo visual object-state shadow experiment.

This script reuses a completed W2 paired-branch dataset.  It never regenerates
or integrates trajectories.  Each history frame is reconstructed only for EGL
rendering from the recorded tray pose, payload-relative pose, and payload 6-D
rotation.  The causal visual model receives frozen image features, tray pose /
velocity, logged past targets, and a candidate future target.  It never reads
the recorded current payload state, ``initial_output``, or any future image.

The result is simulation evidence with exact object labels.  It is not evidence
of real-robot object understanding and is not a completed v4 product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402
import train_mujoco_world as w2  # noqa: E402


CAMERAS = {
    "overview": {
        "pos": [1.2, -1.5, 1.5],
        "xyaxes": [0.78, 0.62, 0.0, -0.34, 0.43, 0.84],
    },
    "front": {
        "pos": [0.0, -1.5, 0.9],
        "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.35, 0.94],
    },
}
SHIFTED_CAMERAS = {
    "overview_shift": {
        "pos": [1.35, -1.25, 1.25],
        "xyaxes": [0.68, 0.73, 0.0, -0.38, 0.35, 0.84],
    },
    "front_shift": {
        "pos": [0.30, -1.5, 0.9],
        "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.35, 0.94],
    },
}
IMAGE_SIZE = 256
PCA_DIM = 16
SLOTS = 4
FEATURE_DIM = 128
MEMBER_SEEDS = (107, 211, 307)
OFFICIAL_RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_value(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> None:
    def scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   allow_nan=False, default=scalar) + "\n"
    )


def _camera_xml(cameras: dict) -> str:
    rows = []
    for name, spec in cameras.items():
        pos = " ".join(str(value) for value in spec["pos"])
        axes = " ".join(str(value) for value in spec["xyaxes"])
        rows.append(f'<camera name="{name}" pos="{pos}" xyaxes="{axes}"/>')
    return "\n".join(rows)


def _render_xml(cameras: dict) -> str:
    xml = model_xml(PhysicsConfig())
    # Remove the owned default camera and inject the frozen experiment layout.
    start = xml.index('<camera name="overview"')
    stop = xml.index("/>", start) + 2
    return xml[:start] + _camera_xml(cameras) + xml[stop:]


def _rotation_matrix(rotation6, np):
    first = np.asarray(rotation6[:3], dtype=np.float64)
    second = np.asarray(rotation6[3:6], dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second -= first * float(np.dot(first, second))
    second /= max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def _set_recorded_frame(mujoco, model, data, history, np):
    tray = np.asarray(history[:3], dtype=np.float64)
    tray_velocity = np.asarray(history[3:6], dtype=np.float64)
    payload = tray + np.asarray(history[6:9], dtype=np.float64)
    payload_velocity = tray_velocity + np.asarray(history[9:12], dtype=np.float64)
    rotation = _rotation_matrix(history[12:18], np)
    angular = np.asarray(history[18:21], dtype=np.float64)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = tray - np.asarray((0.0, 0.0, 0.45))
    address = model.joint("payload_free").qposadr[0]
    data.qpos[address:address + 3] = payload
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    data.qpos[address + 3:address + 7] = quaternion
    data.qvel[:3] = tray_velocity
    data.qvel[3:6] = payload_velocity
    data.qvel[6:9] = angular
    mujoco.mj_forward(model, data)
    tray_actual = np.asarray(data.xpos[model.body("tray").id])
    payload_actual = np.asarray(data.xpos[model.body("payload").id])
    rotation_actual = np.asarray(w2._rotation6(data.xmat[model.body("payload").id]))
    return (
        float(np.max(np.abs(tray_actual - tray))),
        float(np.max(np.abs(payload_actual - payload))),
        float(np.max(np.abs(rotation_actual - history[12:18]))),
    )


def _encoder(weights: Path, torch, torchvision, device):
    weight_sha = _sha_file(weights)
    if weight_sha != OFFICIAL_RESNET18_SHA256:
        raise ValueError("ResNet18 weights do not match the full official SHA256")
    model = torchvision.models.resnet18(weights=None)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval(), weight_sha


def _encode_images(images, encoder, torch, device):
    value = torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float().div_(255.0)
    value = value[:, :, 16:240, 16:240]
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    with torch.inference_mode():
        layer4 = encoder((value - mean) / std)
        pooled = torch.nn.functional.adaptive_avg_pool2d(layer4, (2, 2))
    return pooled.permute(0, 2, 3, 1).reshape(-1, SLOTS, 512).cpu().numpy()


def _render_raw(split_name, split, cameras, raw_path, encoder, args,
                mujoco, np, torch, torchvision, device, keep_images=False):
    roots = split["history"].shape[0]
    xml = _render_xml(cameras)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    names = list(cameras)
    raw = np.lib.format.open_memmap(
        raw_path, "w+", dtype=np.float32, shape=(roots, w2.HISTORY, 2, SLOTS, 512)
    )
    images, locations = [], []
    reconstruction = [0.0, 0.0, 0.0]
    kept = 0

    def flush():
        if not images:
            return
        encoded = _encode_images(np.stack(images), encoder, torch, device)
        for location, feature in zip(locations, encoded):
            raw[location] = feature
        images.clear(); locations.clear()

    for root in range(roots):
        for frame in range(w2.HISTORY):
            errors = _set_recorded_frame(
                mujoco, model, data, split["history"][root, frame], np
            )
            reconstruction = [max(old, new) for old, new in zip(reconstruction, errors)]
            for camera_index, camera in enumerate(names):
                renderer.update_scene(data, camera=camera)
                image = renderer.render().copy()
                if keep_images and kept < 10 and frame == w2.HISTORY - 1:
                    torchvision.io.write_png(
                        torch.from_numpy(image).permute(2, 0, 1),
                        str(args.out / "raw_images" / f"{kept:02d}-{camera}.png"),
                    )
                    kept += 1
                images.append(image); locations.append((root, frame, camera_index))
                if len(images) >= args.render_batch_size:
                    flush()
        if (root + 1) % 100 == 0 or root + 1 == roots:
            print(json.dumps({"phase": "render", "split": split_name,
                              "roots": root + 1, "total": roots}), flush=True)
    flush(); raw.flush(); renderer.close()
    return raw, {
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "camera_layout_sha256": _sha_value(cameras),
        "camera_layout": cameras,
        "max_tray_position_reconstruction_error": reconstruction[0],
        "max_payload_position_reconstruction_error": reconstruction[1],
        "max_rotation6_reconstruction_error": reconstruction[2],
    }


def _fit_pca(raw, np, max_frames=4000):
    rng = np.random.default_rng(20261002)
    frames = raw.shape[0] * raw.shape[1]
    selected = np.arange(frames)
    if len(selected) > max_frames:
        selected = np.sort(rng.choice(selected, max_frames, replace=False))
    flat = np.asarray(raw).reshape(frames, 2, SLOTS, 512)
    pca = []
    for camera in range(2):
        samples = flat[selected, camera].reshape(-1, 512).astype(np.float64)
        mean = samples.mean(0)
        centered = samples - mean
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:PCA_DIM]
        pca.append((mean.astype(np.float32), eigenvectors[:, order].T.astype(np.float32),
                    eigenvalues[order].astype(np.float32)))
    return pca, selected


def _project(raw, pca, np):
    roots, history = raw.shape[:2]
    result = np.empty((roots, history, FEATURE_DIM), dtype=np.float32)
    for camera, (mean, components, _) in enumerate(pca):
        projected = (np.asarray(raw[:, :, camera]) - mean) @ components.T
        result[:, :, camera * 64:(camera + 1) * 64] = projected.reshape(roots, history, 64)
    return result


def _cache_features(splits, root_ids, encoder, encoder_sha, args,
                    mujoco, np, torch, torchvision, device):
    features, metadata = {}, {}
    train_raw_path = args.out / "train-layer4.npy"
    train_raw, metadata["train"] = _render_raw(
        "train", splits["train"], CAMERAS, train_raw_path, encoder, args,
        mujoco, np, torch, torchvision, device, keep_images=True,
    )
    pca, selected = _fit_pca(train_raw, np, args.pca_frames)
    features["train"] = _project(train_raw, pca, np)
    del train_raw; train_raw_path.unlink()
    for split_name in ("dev", "cal", "test"):
        path = args.out / f"{split_name}-layer4.npy"
        raw, metadata[split_name] = _render_raw(
            split_name, splits[split_name], CAMERAS, path, encoder, args,
            mujoco, np, torch, torchvision, device,
        )
        features[split_name] = _project(raw, pca, np)
        del raw; path.unlink()
    alternate_path = args.out / "test-layout-shift-layer4.npy"
    alternate, metadata["test_layout_shift"] = _render_raw(
        "test_layout_shift", splits["test"], SHIFTED_CAMERAS, alternate_path,
        encoder, args, mujoco, np, torch, torchvision, device,
    )
    alternate_features = _project(alternate, pca, np)
    del alternate; alternate_path.unlink()
    cache_arrays = {f"{name}_features": value for name, value in features.items()}
    cache_arrays["test_layout_shift_features"] = alternate_features
    for name, values in root_ids.items():
        cache_arrays[f"{name}_root_ids"] = values
    for camera, (mean, components, eigenvalues) in enumerate(pca):
        cache_arrays[f"pca_{camera}_mean"] = mean
        cache_arrays[f"pca_{camera}_components"] = components
        cache_arrays[f"pca_{camera}_eigenvalues"] = eigenvalues
    cache_arrays["pca_sampled_train_frames"] = selected
    cache_path = args.out / "visual_cache.npz"
    np.savez_compressed(cache_path, **cache_arrays)
    return features, alternate_features, {
        "cache": cache_path.name,
        "cache_sha256": _sha_file(cache_path),
        "encoder_weights_sha256": encoder_sha,
        "encoder": "torchvision ResNet18 ImageNet1K_V1 layer4",
        "preprocess": "256x256 render, center crop 224, ImageNet V1 mean/std",
        "spatial_features": "adaptive average 2x2; four fixed grid slots, not tracked objects",
        "pca": "per-camera shared-slot PCA 512->16 fit only on train roots",
        "pca_sampled_train_frames": int(len(selected)),
        "rendering": metadata,
        "raw_images": sorted(path.name for path in (args.out / "raw_images").glob("*.png")),
    }


def _normalizers(train, np):
    robot = np.concatenate((train["history"][..., :6], train["past_targets"]), -1)
    return {
        "robot_mean": robot.mean((0, 1)).tolist(),
        "robot_std": np.maximum(robot.std((0, 1)), 1e-5).tolist(),
        "visual_mean": train["visual"].mean((0, 1)).tolist(),
        "visual_std": np.maximum(train["visual"].std((0, 1)), 1e-5).tolist(),
        "target_mean": train["future_targets"].mean((0, 1, 2)).tolist(),
        "target_std": np.maximum(train["future_targets"].std((0, 1, 2)), 1e-5).tolist(),
        "output_mean": train["truth"].mean((0, 1, 2)).tolist(),
        "output_std": np.maximum(train["truth"].std((0, 1, 2)), 1e-5).tolist(),
        "fit_split": "train roots only",
    }


def _model_class(torch):
    nn = torch.nn

    class CausalObjectGRU(nn.Module):
        def __init__(self, norms, visual: bool):
            super().__init__()
            self.visual = visual
            self.encoder = nn.GRU(9 + (FEATURE_DIM if visual else 0), 128, batch_first=True)
            self.initial_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 15))
            self.decoder = nn.GRUCell(18, 128)
            self.delta_head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 15))
            for name, value in norms.items():
                if name != "fit_split":
                    self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

        def forward(self, robot, visual, future_targets, zero_actions=False):
            if zero_actions:
                robot = robot.clone(); robot[..., 6:9] = 0.0
                future_targets = torch.zeros_like(future_targets)
            encoded = [(robot - self.robot_mean) / self.robot_std]
            if self.visual:
                if visual is None:
                    raise ValueError("visual model requires historical visual features")
                encoded.append((visual - self.visual_mean) / self.visual_std)
            _, hidden = self.encoder(torch.cat(encoded, -1))
            hidden = hidden[0]
            current = self.initial_head(hidden)
            targets = (future_targets - self.target_mean) / self.target_std
            outputs = []
            for step in range(w2.HORIZON):
                hidden = self.decoder(torch.cat((current, targets[:, step]), -1), hidden)
                current = current + 0.5 * torch.tanh(self.delta_head(hidden))
                outputs.append(current * self.output_std + self.output_mean)
            return torch.stack(outputs, 1)

    return CausalObjectGRU


def _torch_arrays(split, torch, device):
    roots = split["history"].shape[0]
    robot = torch.from_numpy(
        __import__("numpy").concatenate((split["history"][..., :6], split["past_targets"]), -1)
    ).to(device)
    visual = torch.from_numpy(split["visual"]).to(device)
    robot = robot[:, None].expand(roots, 4, w2.HISTORY, 9).reshape(-1, w2.HISTORY, 9)
    visual = visual[:, None].expand(roots, 4, w2.HISTORY, FEATURE_DIM).reshape(
        -1, w2.HISTORY, FEATURE_DIM
    )
    future = torch.from_numpy(split["future_targets"]).to(device).reshape(-1, w2.HORIZON, 3)
    truth = torch.from_numpy(split["truth"]).to(device).reshape(-1, w2.HORIZON, 15)
    return robot, visual, future, truth


def _predict(models, split, torch, device, np, mode="full"):
    robot, visual, future, _ = _torch_arrays(split, torch, device)
    if mode == "action_shuffle":
        future = torch.roll(future, 7, 1)
    rows = []
    for model in models:
        model.eval(); parts = []
        with torch.inference_mode():
            for start in range(0, len(robot), 256):
                stop = min(start + 256, len(robot))
                parts.append(model(
                    robot[start:stop], visual[start:stop] if model.visual else None,
                    future[start:stop], zero_actions=mode == "no_action",
                ).cpu())
        rows.append(torch.cat(parts).numpy())
    roots = split["history"].shape[0]
    return np.mean(rows, 0).reshape(roots, 4, w2.HORIZON, 15)


def _save_state(path, state, np):
    arrays, names = {}, {}
    for index, (name, tensor) in enumerate(sorted(state.items())):
        key = f"tensor_{index:03d}"; arrays[key] = tensor.numpy(); names[key] = name
    np.savez_compressed(path, **arrays)
    return names


def _train_family(name, visual_input, zero_actions, train, dev, norms, args,
                  torch, np, device):
    models, records = [], []
    for member, seed in enumerate(MEMBER_SEEDS[:args.members]):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model = _model_class(torch)(norms, visual_input).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        robot, visual, future, truth = _torch_arrays(train, torch, device)
        generator = torch.Generator().manual_seed(seed + (100000 if zero_actions else 0))
        best, best_step, best_state = math.inf, None, None
        log_path = args.out / f"{name}-member-{member}.jsonl"
        with log_path.open("w") as log:
            for step in range(1, args.steps + 1):
                indices = torch.randint(len(robot), (args.batch_size,), generator=generator).to(device)
                model.train(); optimizer.zero_grad(set_to_none=True)
                prediction = model(robot[indices], visual[indices] if visual_input else None,
                                   future[indices], zero_actions=zero_actions)
                scale = model.output_std
                loss = torch.nn.functional.smooth_l1_loss(prediction / scale, truth[indices] / scale)
                if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    dev_prediction = _predict(
                        [model], dev, torch, device, np,
                        mode="no_action" if zero_actions else "full",
                    )
                    dev_mae = float(np.mean(np.abs(dev_prediction - dev["truth"])))
                    row = {"family": name, "member": member, "step": step,
                           "train_loss": float(loss.detach()), "grad_norm": float(grad),
                           "dev_mae_15d": dev_mae}
                    log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
                    if dev_mae < best:
                        best, best_step = dev_mae, step
                        best_state = {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval(); models.append(model)
        checkpoint = args.out / f"{name}-member-{member}.npz"
        names = _save_state(checkpoint, best_state, np)
        records.append({"member": member, "seed": seed, "best_step": best_step,
                        "best_dev_mae_15d": best, "checkpoint": checkpoint.name,
                        "checkpoint_sha256": _sha_file(checkpoint), "log": log_path.name,
                        "log_sha256": _sha_file(log_path), "tensor_names": names})
    return models, records


def _selection_report(arrays, test, np):
    allowed = arrays["allowed"].astype(bool)
    selected = np.any(allowed, 1)
    choice = np.argmax(allowed, 1)
    unsafe = test["risk_window"].astype(bool)[np.arange(len(selected)), choice]
    durations = {}
    for index, duration in enumerate(w2.DURATIONS):
        durations[str(duration)] = int(np.sum(selected & (choice == index)))
    selected_count = int(np.sum(selected)); rejected = int(np.sum(~selected))
    unsafe_selected = int(np.sum(selected & unsafe))
    return {
        "total_test_roots": int(len(selected)),
        "selected_roots_denominator": selected_count,
        "rejected_roots_denominator": rejected,
        "reject_rate": float(np.mean(~selected)),
        "selected_duration_counts": durations,
        "unsafe_selected_numerator": unsafe_selected,
        "unsafe_given_selected": (
            unsafe_selected / selected_count if selected_count else None
        ),
        "false_allow_rate_all_test_roots": float(np.mean(selected & unsafe)),
    }


def _label_sha(splits, np):
    digest = hashlib.sha256()
    for split in ("train", "dev", "cal", "test"):
        for field in ("truth", "risk_window", "risk_posthold", "drop_window", "drop_posthold"):
            digest.update(np.ascontiguousarray(splits[split][field]).tobytes())
    return digest.hexdigest()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--render-batch-size", type=int, default=64)
    parser.add_argument("--pca-frames", type=int, default=4000)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.preflight:
        args.steps = min(args.steps, 2); args.members = 1; args.pca_frames = 32
    if min(args.steps, args.members, args.batch_size, args.eval_every,
           args.render_batch_size, args.pca_frames) < 1:
        raise ValueError("numeric arguments must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("--out must be empty")
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "raw_images").mkdir()
    started = time.time()
    try:
        import mujoco
        import numpy as np
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError("requires existing mujoco, numpy, torch, torchvision") from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    splits, root_ids, dataset_manifest, dataset_hashes = w2._load_dataset_run(
        args.dataset_run, np
    )
    expected_counts = {"train": 1000, "dev": 200, "cal": 299, "test": 500}
    if dataset_manifest.get("counts") != expected_counts:
        raise ValueError("requires the frozen 1000/200/299/500 W2 dataset")
    if args.preflight:
        counts = {"train": 8, "dev": 4, "cal": 24, "test": 4}
        for name, count in counts.items():
            splits[name] = {key: value[:count] for key, value in splits[name].items()}
            root_ids[name] = root_ids[name][:count]
    labels_sha = _label_sha(splits, np)
    encoder, encoder_sha = _encoder(args.weights, torch, torchvision, device)
    features, alternate_features, visual_metadata = _cache_features(
        splits, root_ids, encoder, encoder_sha, args,
        mujoco, np, torch, torchvision, device,
    )
    del encoder
    for name in splits:
        splits[name]["visual"] = features[name]
    alternate_test = dict(splits["test"]); alternate_test["visual"] = alternate_features
    norms = _normalizers(splits["train"], np)
    visual_models, visual_members = _train_family(
        "visual", True, False, splits["train"], splits["dev"], norms,
        args, torch, np, device,
    )
    robot_models, robot_members = _train_family(
        "robot-only", False, False, splits["train"], splits["dev"], norms,
        args, torch, np, device,
    )
    no_action_models, no_action_members = _train_family(
        "visual-no-action", True, True, splits["train"], splits["dev"], norms,
        args, torch, np, device,
    )
    predictors = {
        "visual_object_gru128": lambda split: _predict(
            visual_models, split, torch, device, np),
        "robot_only_gru128": lambda split: _predict(
            robot_models, split, torch, device, np),
        "visual_action_shuffle_inference": lambda split: _predict(
            visual_models, split, torch, device, np, "action_shuffle"),
        "visual_independently_trained_no_action": lambda split: _predict(
            no_action_models, split, torch, device, np, "no_action"),
    }

    def shifted_predictor(split):
        evaluation = alternate_test if split is splits["test"] else split
        return _predict(visual_models, evaluation, torch, device, np)

    predictors["visual_camera_layout_shift_test_only"] = shifted_predictor
    metrics, prediction_arrays = {}, {}
    for name, predictor in predictors.items():
        metric, arrays = w2._calibrated_metrics(
            name, predictor, splits["dev"], splits["cal"], splits["test"], np, alpha=0.05
        )
        metric["selection_denominators"] = _selection_report(arrays, splits["test"], np)
        if name == "visual_camera_layout_shift_test_only":
            metric["protocol"] = (
                "same labels and original dev/cal calibration; shifted cameras only on original "
                "test roots; never used for tuning"
            )
        metrics[name] = metric
        prediction_arrays[name] = arrays
    oracle_metrics_path = args.dataset_run / "metrics.json"
    oracle_manifest_path = args.dataset_run / "manifest.json"
    if not oracle_metrics_path.is_file() or not oracle_manifest_path.is_file():
        raise FileNotFoundError("W2 oracle reference metrics/manifest are required")
    oracle = json.loads(oracle_metrics_path.read_text())["gru128_ensemble"]
    metrics["w2_true_object_observation_oracle_reference"] = {
        "information_scope": (
            "reference model reads true 22D history including payload state and true initial_output; "
            "it has strictly more information and is not a same-information baseline"
        ),
        "metrics": oracle,
        "metrics_sha256": _sha_file(oracle_metrics_path),
        "manifest_sha256": _sha_file(oracle_manifest_path),
    }
    metrics["interpretation"] = {
        "visual_claim": "MuJoCo-rendered object-state shadow only",
        "real_robot_object_understanding": False,
        "v4_product_complete": False,
        "initial_output_used_by_new_models": False,
        "future_images_used": False,
        "four_sibling_candidates_share_each_root": True,
        "negative_results_retained": list(predictors),
    }
    _write_json(args.out / "metrics.json", metrics)
    arrays = {
        "root_ids": root_ids["test"],
        "truth": splits["test"]["truth"],
        "risk_window": splits["test"]["risk_window"],
        "risk_posthold": splits["test"]["risk_posthold"],
    }
    for method, values in prediction_arrays.items():
        for key, value in values.items(): arrays[f"{method}_{key}"] = value
    np.savez_compressed(args.out / "test_predictions.npz", **arrays)
    manifest = {
        "schema": "sentinel-w2-mujoco-visual-object-shadow-v1",
        "status": "complete",
        "scope": (
            "offline MuJoCo visual object-state shadow; no real-robot object-understanding, "
            "safety certification, or v4 product completion claim"
        ),
        "source_commit": args.source_commit,
        "script_sha256": _sha_file(Path(__file__)),
        "dataset": {**dataset_hashes, "labels_sha256": labels_sha,
                    "counts": dataset_manifest["counts"], "trajectory_regeneration": False},
        "causal_inputs": (
            "8 historical frozen visual features + tray 6D pose/velocity + historical targets + "
            "candidate future targets; no payload truth/current initial_output/future image"
        ),
        "visual": visual_metadata,
        "normalizers": norms,
        "training": {"steps": args.steps, "batch_size": args.batch_size,
                     "hidden": 128, "members": args.members, "dev_selection": True,
                     "visual_members": visual_members, "robot_only_members": robot_members,
                     "visual_no_action_members": no_action_members},
        "calibration": (
            "independent alpha=0.05 root-max full 4x40x15 and XY 4x40x2 envelopes; "
            "299 calibration roots in formal run"
        ),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "mujoco": mujoco.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device)},
        "preflight": args.preflight,
        "elapsed_seconds": time.time() - started,
        "artifacts": {},
    }
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.relative_to(args.out).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _sha_file(path)
            }
    _write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out": str(args.out),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
