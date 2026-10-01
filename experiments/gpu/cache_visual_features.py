#!/usr/bin/env python3
"""Cache frozen two-camera SO100 visual features without network access.

The encoder is torchvision ResNet-18 with the official ImageNet1K_V1 state
dict loaded only from ``--weights``.  The script never asks torchvision to
download weights.  For each camera, layer4's 7x7 map is adaptive-average-
pooled to a fixed 2x2 grid, yielding four *spatial slots* of 512 dimensions.
These slots are fixed image regions, not tracked objects and not evidence of
object understanding.  A train-episode-only PCA maps each 512-D slot to 16-D;
two cameras therefore produce 2 x 4 x 16 = 128 cached dimensions per frame.

Video alignment is strict.  The v3 single AV1 file for each camera must decode
to 19,631 frames at 30 FPS.  Dataset row timestamps are combined with the
per-episode video ``from_timestamp`` metadata and matched to decoded PTS.  The
script refuses to fall back to blind frame offsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


TOTAL_EPISODES = 50
TOTAL_FRAMES = 19631
FPS = 30.0
SEED = 20261002
PCA_DIM = 16
SLOTS = 4
ENCODER_DIM = 512
OFFICIAL_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
OFFICIAL_DATASET_REVISION_PREFIX = "728583b5"
SPLIT_COUNTS = {"train": 30, "dev": 5, "cal": 10, "test": 5}
EXPECTED_DATASET = {
    "episodes": TOTAL_EPISODES,
    "frames": TOTAL_FRAMES,
    "fps": 30,
    "format_version": "v3.0",
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_value(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _verified_download_manifest(data_dir: Path) -> dict:
    path = data_dir / "download_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("official snapshot download_manifest.json is required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != "lerobot/svla_so100_pickplace":
        raise ValueError("download manifest repo_id is not lerobot/svla_so100_pickplace")
    revision = str(manifest.get("revision", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not revision.startswith(
        OFFICIAL_DATASET_REVISION_PREFIX
    ):
        raise ValueError("download manifest is not the fixed official 728583b5... snapshot")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("download manifest has no file inventory")
    seen = set()
    total_bytes = 0
    root = data_dir.resolve()
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("download manifest contains an invalid or duplicate path")
        seen.add(relative)
        file_path = (data_dir / relative).resolve()
        if not file_path.is_relative_to(root) or not file_path.is_file():
            raise ValueError(f"download manifest file is missing or escapes root: {relative}")
        expected_size = int(item.get("size", -1))
        expected_sha = str(item.get("sha256", "")).removeprefix("sha256:")
        actual_sha = _sha256_file(file_path).removeprefix("sha256:")
        if file_path.stat().st_size != expected_size or actual_sha != expected_sha:
            raise ValueError(f"download manifest integrity mismatch: {relative}")
        total_bytes += expected_size
    return {
        "path": path.relative_to(data_dir).as_posix(),
        "sha256": _sha256_file(path),
        "repo_id": manifest["repo_id"],
        "revision": revision,
        "official_metadata_sha256": manifest.get("official_metadata_sha256"),
        "validated_files": len(files),
        "validated_bytes": total_bytes,
    }


def _load_info(data_dir: Path) -> tuple[Path, dict]:
    for path in (data_dir / "meta" / "info.json", data_dir / "info.json"):
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("expected meta/info.json")


def _video_cameras(info: dict) -> list[str]:
    cameras = []
    for name, feature in info.get("features", {}).items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            cameras.append(name)
    cameras.sort()
    if len(cameras) != 2:
        raise ValueError(f"expected exactly two video cameras, metadata reports {cameras}")
    return cameras


def _validate_info(info: dict) -> None:
    def number(*keys: str):
        for key in keys:
            if key in info:
                return info[key]
        return None

    checks = (
        (number("total_episodes", "episodes"), TOTAL_EPISODES, "episodes"),
        (number("total_frames", "frames"), TOTAL_FRAMES, "frames"),
        (number("fps"), FPS, "fps"),
    )
    for actual, expected, label in checks:
        if actual is not None and float(actual) != float(expected):
            raise ValueError(f"expected {label}={expected}, metadata reports {actual}")
    version = number("codebase_version", "format_version", "dataset_format_version")
    if version is not None and str(version) not in ("v3.0", "3.0"):
        raise ValueError(f"expected LeRobot v3.0 metadata, reports {version}")


def _data_parquets(data_dir: Path) -> list[Path]:
    paths = [
        path for path in sorted(data_dir.rglob("*.parquet"))
        if "meta" not in path.relative_to(data_dir).parts
    ]
    if not paths:
        raise FileNotFoundError("no state/action parquet files found")
    return paths


def _column(names: Sequence[str], candidates: Sequence[str], required: bool = True):
    for name in candidates:
        if name in names:
            return name
    if required:
        raise ValueError(f"missing column; expected one of {list(candidates)}")
    return None


def _read_frame_index(data_dir: Path, pq, np) -> dict[str, object]:
    rows = []
    for path in _data_parquets(data_dir):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        global_name = _column(names, ("index",))
        episode_name = _column(names, ("episode_index", "episode_id"))
        frame_name = _column(names, ("frame_index",))
        timestamp_name = _column(names, ("timestamp",))
        columns = [global_name, episode_name, frame_name, timestamp_name]
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            values = batch.to_pydict()
            for offset in range(batch.num_rows):
                rows.append(
                    (
                        int(values[global_name][offset]),
                        int(values[episode_name][offset]),
                        int(values[frame_name][offset]),
                        float(values[timestamp_name][offset]),
                    )
                )
    rows.sort(key=lambda item: item[0])
    if len(rows) != TOTAL_FRAMES:
        raise ValueError(f"expected {TOTAL_FRAMES} parquet rows, found {len(rows)}")
    indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(TOTAL_FRAMES, dtype=np.int64)):
        raise ValueError("parquet global index must be exactly 0..19630")
    episode = np.asarray([row[1] for row in rows], dtype=np.int64)
    frame = np.asarray([row[2] for row in rows], dtype=np.int64)
    timestamp = np.asarray([row[3] for row in rows], dtype=np.float64)
    if not np.isfinite(timestamp).all():
        raise ValueError("parquet timestamps contain non-finite values")
    if len(np.unique(episode)) != TOTAL_EPISODES:
        raise ValueError("expected exactly 50 episode indices")
    for episode_id in sorted(np.unique(episode).astype(int).tolist()):
        mask = episode == episode_id
        local_frames = frame[mask]
        if not np.array_equal(local_frames, np.arange(len(local_frames), dtype=np.int64)):
            raise ValueError(f"episode {episode_id} frame_index is not contiguous from zero")
        if len(local_frames) > 1:
            local_time = timestamp[mask]
            if np.max(np.abs(np.diff(local_time) - 1.0 / FPS)) > 2e-3:
                raise ValueError(f"episode {episode_id} parquet timestamps are not 30 Hz")
    return {"index": indices, "episode": episode, "frame": frame, "timestamp": timestamp}


def _metadata_records(data_dir: Path, pq) -> list[dict]:
    meta = data_dir / "meta"
    records: list[dict] = []
    for path in sorted(meta.rglob("*.jsonl")) if meta.is_dir() else []:
        if "episode" not in path.as_posix().lower():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict) and "episode_index" in value:
                    records.append(value)
    for path in sorted(meta.rglob("*.parquet")) if meta.is_dir() else []:
        if "episode" not in path.as_posix().lower():
            continue
        table = pq.read_table(path)
        for value in table.to_pylist():
            if isinstance(value, dict) and "episode_index" in value:
                records.append(value)
    by_episode = {}
    for record in records:
        episode_id = int(record["episode_index"])
        # Prefer the episode record that actually carries video timing over a
        # same-index statistics record encountered elsewhere under meta/.
        timing_fields = sum(
            key.endswith(("from_timestamp", "to_timestamp"))
            for key in _flatten(record)
        )
        previous = by_episode.get(episode_id)
        previous_fields = (
            sum(
                key.endswith(("from_timestamp", "to_timestamp"))
                for key in _flatten(previous)
            )
            if previous is not None else -1
        )
        if timing_fields > previous_fields:
            by_episode[episode_id] = record
    if len(by_episode) != TOTAL_EPISODES:
        raise ValueError(
            f"expected episode video metadata for 50 episodes, found {len(by_episode)}"
        )
    return [by_episode[index] for index in sorted(by_episode)]


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
    else:
        result[prefix] = value
    return result


def _video_bounds(record: dict, camera: str) -> tuple[float, float]:
    videos = record.get("videos")
    if isinstance(videos, dict):
        direct = videos.get(camera)
        if isinstance(direct, dict):
            start = direct.get("from_timestamp")
            stop = direct.get("to_timestamp")
            if start is not None and stop is not None:
                return float(start), float(stop)
    flat = _flatten(record)
    starts = [
        value for key, value in flat.items()
        if camera in key and key.endswith("from_timestamp")
    ]
    stops = [
        value for key, value in flat.items()
        if camera in key and key.endswith("to_timestamp")
    ]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError(
            f"episode {record.get('episode_index')} lacks unambiguous video bounds for {camera}"
        )
    return float(starts[0]), float(stops[0])


def _video_file(data_dir: Path, camera: str) -> Path:
    video_paths = []
    for suffix in ("*.mp4", "*.mkv", "*.webm"):
        for path in data_dir.rglob(suffix):
            if "videos" in path.relative_to(data_dir).parts and camera in path.as_posix():
                video_paths.append(path)
    video_paths = sorted(set(video_paths))
    if len(video_paths) != 1:
        raise ValueError(
            f"v3 single-file profile requires exactly one video for {camera}, found {video_paths}"
        )
    return video_paths[0]


def _video_pts(path: Path, av, np) -> tuple[object, dict]:
    values = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"expected one video stream in {path}")
        stream = container.streams.video[0]
        codec = str(stream.codec_context.name).lower()
        if "av1" not in codec and "dav1d" not in codec:
            raise ValueError(f"expected AV1 video, decoded codec is {codec}")
        rate = float(stream.average_rate) if stream.average_rate is not None else math.nan
        if not math.isfinite(rate) or abs(rate - FPS) > 1e-6:
            raise ValueError(f"expected 30 FPS stream, reports {rate}")
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("video frame has no PTS")
            values.append(float(frame.pts * stream.time_base))
    pts = np.asarray(values, dtype=np.float64)
    if len(pts) != TOTAL_FRAMES:
        raise ValueError(f"expected {TOTAL_FRAMES} decoded frames, found {len(pts)}")
    if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("decoded video PTS must be finite and strictly increasing")
    if np.max(np.abs(np.diff(pts) - 1.0 / FPS)) > 2e-3:
        raise ValueError("decoded video PTS is not a continuous 30 FPS timeline")
    return pts, {"codec": codec, "average_rate": rate, "frames": len(pts)}


def _align_camera(camera: str, pts, frame_rows: dict, episode_records: list[dict], np):
    target = np.empty(TOTAL_FRAMES, dtype=np.float64)
    bounds = {}
    episodes = frame_rows["episode"]
    timestamps = frame_rows["timestamp"]
    by_episode = {int(record["episode_index"]): record for record in episode_records}
    for episode_id in sorted(np.unique(episodes).astype(int).tolist()):
        mask = episodes == episode_id
        local = timestamps[mask] - float(timestamps[mask][0])
        start, stop = _video_bounds(by_episode[episode_id], camera)
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError(f"invalid video timestamp bounds for episode {episode_id}/{camera}")
        target[mask] = start + local
        if target[mask][-1] > stop + (0.5 / FPS + 1e-4):
            raise ValueError(f"episode {episode_id}/{camera} rows exceed metadata to_timestamp")
        bounds[str(episode_id)] = {"from_timestamp": start, "to_timestamp": stop}
    positions = np.searchsorted(pts, target)
    positions = np.clip(positions, 0, len(pts) - 1)
    prior = np.maximum(positions - 1, 0)
    use_prior = np.abs(pts[prior] - target) < np.abs(pts[positions] - target)
    positions[use_prior] = prior[use_prior]
    error = np.abs(pts[positions] - target)
    tolerance = 0.5 / FPS + 1e-4
    if float(error.max()) > tolerance:
        raise ValueError(
            f"{camera} PTS alignment error {float(error.max())} exceeds {tolerance}"
        )
    if len(np.unique(positions)) != TOTAL_FRAMES:
        raise ValueError(f"{camera} timestamp mapping is not one-to-one")
    if not np.array_equal(np.sort(positions), np.arange(TOTAL_FRAMES)):
        raise ValueError(f"{camera} timestamp mapping does not cover the whole video")
    return positions.astype(np.int64), {
        "max_abs_pts_error_seconds": float(error.max()),
        "tolerance_seconds": tolerance,
        "one_to_one": True,
        "episode_video_bounds": bounds,
    }


def _load_encoder(weights_path: Path, weight_sha: str, torch, torchvision, device):
    if weight_sha.removeprefix("sha256:") != OFFICIAL_WEIGHT_SHA256:
        raise ValueError(
            "--weights does not match the full official resnet18-f37072fd.pth SHA256"
        )
    model = torchvision.models.resnet18(weights=None)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    encoder = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    return encoder


def _preprocess(images, torch, torchvision, device):
    # Exactly the ImageNet1K_V1 evaluation preprocessing: resize shorter side
    # to 256 with bilinear antialiasing, center crop 224, then normalize.
    functional = torchvision.transforms.functional
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    tensors = []
    for image in images:
        value = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        value = functional.resize(value, 256, interpolation=interpolation, antialias=True)
        value = functional.center_crop(value, [224, 224])
        value = functional.normalize(
            value,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        tensors.append(value)
    return torch.stack(tensors).to(device)


def _encode_video(path: Path, raw_path: Path, encoder, batch_size: int,
                  av, np, torch, torchvision, device):
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(TOTAL_FRAMES, SLOTS, ENCODER_DIM)
    )
    images, ordinals = [], []

    def flush():
        if not images:
            return
        batch = _preprocess(images, torch, torchvision, device)
        with torch.inference_mode():
            layer4 = encoder(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(layer4, (2, 2))
            slots = pooled.permute(0, 2, 3, 1).reshape(-1, SLOTS, ENCODER_DIM)
        raw[np.asarray(ordinals, dtype=np.int64)] = slots.cpu().numpy().astype(np.float32)
        images.clear()
        ordinals.clear()

    decoded = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(stream)):
            decoded = ordinal + 1
            images.append(frame.to_ndarray(format="rgb24"))
            ordinals.append(ordinal)
            if len(images) >= batch_size:
                flush()
        flush()
    if decoded != TOTAL_FRAMES:
        raise ValueError(f"feature pass decoded {decoded} frames, expected {TOTAL_FRAMES}")
    raw.flush()
    return raw


def _episode_split(episode_ids: Iterable[int]) -> dict[str, list[int]]:
    values = sorted(int(value) for value in episode_ids)
    random.Random(SEED).shuffle(values)
    result = {}
    offset = 0
    for name, count in SPLIT_COUNTS.items():
        result[name] = sorted(values[offset:offset + count])
        offset += count
    return result


def _fit_pca(raw, train_ordinals, max_frames: int, np):
    rng = np.random.default_rng(SEED)
    choices = np.asarray(train_ordinals, dtype=np.int64)
    if len(choices) > max_frames:
        choices = np.sort(rng.choice(choices, size=max_frames, replace=False))
    samples = np.asarray(raw[choices], dtype=np.float32).reshape(-1, ENCODER_DIM)
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = samples - mean
    covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1][:PCA_DIM]
    components = eigenvectors[:, order].T.astype(np.float32)
    explained = eigenvalues[order].astype(float)
    return mean, components, explained, choices


def _project(raw, mean, components, ordinal_for_global, np, chunk: int = 1024):
    result = np.empty((TOTAL_FRAMES, SLOTS * PCA_DIM), dtype=np.float32)
    for start in range(0, TOTAL_FRAMES, chunk):
        stop = min(start + chunk, TOTAL_FRAMES)
        encoded = np.asarray(raw[ordinal_for_global[start:stop]], dtype=np.float32)
        projected = (encoded - mean[None, None, :]) @ components.T
        result[start:stop] = projected.reshape(stop - start, SLOTS * PCA_DIM)
    return result


def _dataset_manifest(data_dir: Path) -> dict:
    info_path, _ = _load_info(data_dir)
    meta_root = data_dir / "meta"
    roots = [meta_root] if meta_root.is_dir() else [info_path.parent]
    meta_files = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path not in seen:
                seen.add(path)
                meta_files.append(
                    {
                        "path": path.relative_to(data_dir).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    data_files = [
        {
            "path": path.relative_to(data_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _data_parquets(data_dir)
    ]
    value = {
        "dataset": "lerobot/svla_so100_pickplace",
        "expected_profile": EXPECTED_DATASET,
        "metadata_files": meta_files,
        "data_parquet_inventory": data_files,
    }
    # Keep this content hash compatible with caches made before the source
    # snapshot manifest was promoted into provenance.  The separately hashed
    # and fully validated download manifest below binds the fixed Hub snapshot.
    value["dataset_manifest_hash"] = _sha256_value(value)
    value["download_manifest"] = _verified_download_manifest(data_dir)
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cache strict two-camera SO100 ResNet18/PCA features")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-frames", type=int, default=4000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not args.data_dir.is_dir() or not args.weights.is_file():
        raise FileNotFoundError("--data-dir and --weights must be local existing paths")
    if args.batch_size < 1 or args.pca_frames < 1:
        raise ValueError("batch-size and pca-frames must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("--out must be a new or empty directory")
    try:
        import av
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "this cache experiment requires av, numpy, pyarrow, torch, and torchvision"
        ) from exc
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"
    work.mkdir()
    started = time.time()

    info_path, info = _load_info(args.data_dir)
    _validate_info(info)
    cameras = _video_cameras(info)
    frame_rows = _read_frame_index(args.data_dir, pq, np)
    episode_records = _metadata_records(args.data_dir, pq)
    dataset_manifest = _dataset_manifest(args.data_dir)
    weight_sha = _sha256_file(args.weights)
    encoder = _load_encoder(args.weights, weight_sha, torch, torchvision, device)
    episode_split = _episode_split(np.unique(frame_rows["episode"]).tolist())
    split_hash = _sha256_value({"seed": SEED, "split": episode_split})
    train_episode_ids = episode_split["train"]
    train_global = np.flatnonzero(np.isin(frame_rows["episode"], train_episode_ids))

    camera_features = []
    camera_metadata = []
    for camera_number, camera in enumerate(cameras):
        video_path = _video_file(args.data_dir, camera)
        pts, stream_metadata = _video_pts(video_path, av, np)
        ordinal_for_global, alignment = _align_camera(
            camera, pts, frame_rows, episode_records, np
        )
        raw_path = work / f"camera_{camera_number}_layer4.npy"
        raw = _encode_video(
            video_path, raw_path, encoder, args.batch_size, av, np, torch, torchvision, device
        )
        train_ordinals = ordinal_for_global[train_global]
        mean, components, explained, sampled_ordinals = _fit_pca(
            raw, train_ordinals, args.pca_frames, np
        )
        pca_path = args.out / f"pca_camera_{camera_number}.npz"
        np.savez_compressed(
            pca_path,
            mean=mean,
            components=components,
            explained_variance=explained,
            sampled_video_ordinals=sampled_ordinals,
            train_episode_indices=np.asarray(train_episode_ids, dtype=np.int64),
        )
        camera_features.append(_project(raw, mean, components, ordinal_for_global, np))
        camera_metadata.append(
            {
                "camera_key": camera,
                "video_path": video_path.relative_to(args.data_dir).as_posix(),
                "video_sha256": _sha256_file(video_path),
                "stream": stream_metadata,
                "alignment": alignment,
                "pca_path": pca_path.name,
                "pca_sha256": _sha256_file(pca_path),
                "pca_fit_episode_indices": train_episode_ids,
                "pca_sampled_frames": int(len(sampled_ordinals)),
                "pca_fit_slots": int(len(sampled_ordinals) * SLOTS),
            }
        )
        del raw
        raw_path.unlink()

    features = np.concatenate(camera_features, axis=1).astype(np.float32)
    if features.shape != (TOTAL_FRAMES, 128) or not np.isfinite(features).all():
        raise ValueError(f"visual cache must be finite [19631,128], got {features.shape}")
    cache_path = args.out / "visual_features.npz"
    np.savez_compressed(
        cache_path,
        index=frame_rows["index"],
        episode_index=frame_rows["episode"],
        frame_index=frame_rows["frame"],
        timestamp=frame_rows["timestamp"],
        features=features,
        camera_keys=np.asarray(cameras),
        dataset_manifest_hash=np.asarray(dataset_manifest["dataset_manifest_hash"]),
        encoder_weights_sha256=np.asarray(weight_sha),
        split_seed=np.asarray(SEED, dtype=np.int64),
        split_hash=np.asarray(split_hash),
        pca_fit_split_hash=np.asarray(_sha256_value(train_episode_ids)),
        train_split_hash=np.asarray(_sha256_value(train_episode_ids)),
        cal_split_hash=np.asarray(_sha256_value(episode_split["cal"])),
        camera_pca_sha256=np.asarray(
            [camera["pca_sha256"] for camera in camera_metadata]
        ),
        frame_alignment_verified=np.asarray(True),
    )
    cache_sha = _sha256_file(cache_path)
    metadata = {
        "status": "completed",
        "scope": (
            "frozen ResNet18 ImageNet1K_V1 two-camera spatial-grid features; fixed 2x2 slots "
            "are not tracked objects and do not establish object understanding"
        ),
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_hash": dataset_manifest["dataset_manifest_hash"],
        "info_path": info_path.relative_to(args.data_dir).as_posix(),
        "encoder": {
            "architecture": "torchvision resnet18 layer4",
            "weights": "IMAGENET1K_V1 local resnet18-f37072fd.pth",
            "weights_sha256": weight_sha,
            "network_download": False,
            "preprocess": "resize-shorter-256 bilinear antialias, center-crop-224, ImageNet mean/std",
            "layer4_shape": [512, 7, 7],
            "pool": "adaptive_avg_pool2d(2,2)",
        },
        "pca": {
            "fit_split": "30 train episodes only",
            "split_seed": SEED,
            "train_episode_indices": train_episode_ids,
            "fit_split_hash": _sha256_value(train_episode_ids),
            "per_camera_shared_across_four_spatial_slots": True,
            "input_dim_per_slot": ENCODER_DIM,
            "output_dim_per_slot": PCA_DIM,
            "slots_per_camera": SLOTS,
            "total_output_dim": 128,
        },
        "episode_split": episode_split,
        "split_hash": split_hash,
        "train_split_hash": _sha256_value(train_episode_ids),
        "cal_split_hash": _sha256_value(episode_split["cal"]),
        "frame_alignment_verified": True,
        "cameras": camera_metadata,
        "cache_path": cache_path.name,
        "cache_sha256": cache_sha,
        "frames": TOTAL_FRAMES,
        "wall_seconds": time.time() - started,
        "limitations": [
            "no future image is an input",
            "no object pose/contact/collision label is created",
            "spatial slots are fixed grid regions, not tracked object slots",
            "features are suitable only as frozen historical context for a shadow predictor",
        ],
    }
    _write_json(args.out / "metadata.json", metadata)
    work.rmdir()
    print(json.dumps({"status": "completed", "cache": str(cache_path), "sha256": cache_sha}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
