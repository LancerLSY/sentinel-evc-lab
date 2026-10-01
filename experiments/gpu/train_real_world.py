#!/usr/bin/env python3
"""Train a real-data, action-conditioned joint-dynamics shadow model.

This experiment reads the local Hugging Face/LeRobot v3 parquet export of
``lerobot/svla_so100_pickplace`` directly with PyArrow.  It does not import
LeRobot, infer Cartesian poses, or create object-consequence labels.  The
six-dimensional ``observation.state`` and ``action`` arrays are kept in their
dataset-native logged representation; the source declares joint names but not
exact physical units.  An optional cache adds
frozen historical two-camera image-grid features; future images are never an
input and the fixed grid cells are not tracked objects.

The learned claim is narrow: given eight observed joint states/actions and a
forty-step final action sequence, predict the following forty joint states.
This is a real-data joint-dynamics *shadow* experiment, not the v4 visual
WorldGuard, not a collision predictor, and not evidence that an action is safe.

Default split protocol (seed 20261002): 30 train / 5 dev / 10 calibration /
5 test episodes.  Windows never cross episode boundaries.  Train-only state
and action scaling is used.  The action-conditioned model and the no-future-
action ablation are trained independently; persistence and action-target are
non-learned baselines.  Test is evaluated once after dev checkpoint selection
and episode-level calibration are frozen.

Example:

    python experiments/gpu/train_real_world.py \
      --data-dir /data/svla_so100_pickplace \
      --out runs/w1-real-20261002 \
      --steps 4000 --device cuda --source-commit <git-sha>

Experiment-only dependencies: numpy, pyarrow, torch.  They are intentionally
not runtime dependencies of the installable Sentinel package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


HISTORY = 8
HORIZON = 40
STATE_DIM = 6
ACTION_DIM = 6
DEFAULT_SEED = 20261002
SPLIT_COUNTS = {"train": 30, "dev": 5, "cal": 10, "test": 5}
EXPECTED_DATASET = {
    "episodes": 50,
    "frames": 19631,
    "fps": 30,
    "format_version": "v3.0",
}
OFFICIAL_DATASET_REVISION_PREFIX = "728583b5"
OFFICIAL_RESNET18_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


@dataclass(frozen=True)
class Episode:
    episode_index: int
    global_index: "object"  # np.ndarray [T]
    frame_index: "object"  # np.ndarray [T]
    state: "object"  # np.ndarray [T, 6]
    action: "object"  # np.ndarray [T, 6]


@dataclass(frozen=True)
class Windows:
    split: str
    episode_ids: "object"  # np.ndarray [N]
    history_index: "object"  # np.ndarray [N, 8], parquet global frame index
    history_state: "object"  # np.ndarray [N, 8, 6]
    history_action: "object"  # np.ndarray [N, 8, 6]
    future_action: "object"  # np.ndarray [N, 40, 6]
    future_state: "object"  # np.ndarray [N, 40, 6]
    history_visual: "object | None" = None  # optional np.ndarray [N, 8, 128]

    def __len__(self) -> int:
        return int(self.history_state.shape[0])


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


def _finite(array, np, label: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")


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
    candidates = (
        data_dir / "meta" / "info.json",
        data_dir / "info.json",
    )
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("expected meta/info.json in the local LeRobot dataset")


def _number(info: dict, *keys: str):
    for key in keys:
        if key in info:
            return info[key]
    return None


def _feature_shape(info: dict, name: str) -> list[int] | None:
    feature = info.get("features", {}).get(name)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    return list(shape) if isinstance(shape, (list, tuple)) else None


def _validate_dataset_identity(info: dict) -> None:
    episodes = _number(info, "total_episodes", "episodes")
    frames = _number(info, "total_frames", "frames")
    fps = _number(info, "fps")
    if episodes is not None and int(episodes) != EXPECTED_DATASET["episodes"]:
        raise ValueError(f"expected 50 episodes, metadata reports {episodes}")
    if frames is not None and int(frames) != EXPECTED_DATASET["frames"]:
        raise ValueError(f"expected 19631 frames, metadata reports {frames}")
    if fps is not None and float(fps) != float(EXPECTED_DATASET["fps"]):
        raise ValueError(f"expected 30 Hz, metadata reports {fps}")
    version = _number(info, "codebase_version", "format_version", "dataset_format_version")
    if version is not None and str(version) not in ("v3.0", "3.0"):
        raise ValueError(f"expected LeRobot v3.0 metadata, reports {version}")
    for name, expected in (("observation.state", STATE_DIM), ("action", ACTION_DIM)):
        shape = _feature_shape(info, name)
        if shape is not None and shape != [expected]:
            raise ValueError(f"expected {name} shape [{expected}], metadata reports {shape}")


def _local_dataset_manifest(data_dir: Path, info_path: Path) -> dict:
    """Build deterministic content and official-snapshot provenance.

    Metadata and state/action parquet form the cache-compatible dataset hash.
    Every file in download_manifest.json, including videos, is separately
    rehashed to bind the fixed official snapshot and reject damaged downloads.
    """

    metadata_files = []
    meta_root = data_dir / "meta"
    roots = [meta_root] if meta_root.is_dir() else [info_path.parent]
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path not in seen:
                seen.add(path)
                metadata_files.append(
                    {
                        "path": path.relative_to(data_dir).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    parquet_files = [
        {
            "path": path.relative_to(data_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(data_dir.rglob("*.parquet"))
        if "meta" not in path.relative_to(data_dir).parts
    ]
    if not parquet_files:
        raise FileNotFoundError("no data parquet shards found below --data-dir")
    manifest = {
        "dataset": "lerobot/svla_so100_pickplace",
        "expected_profile": EXPECTED_DATASET,
        "metadata_files": metadata_files,
        "data_parquet_inventory": parquet_files,
    }
    # Preserve the content hash used by already-running trusted cache jobs.
    # The separately hashed, fully validated download manifest binds the fixed
    # official Hub snapshot and is included in dataset_source.json.
    manifest["dataset_manifest_hash"] = _sha256_value(manifest)
    manifest["download_manifest"] = _verified_download_manifest(data_dir)
    return manifest


def _column_name(names: Sequence[str], candidates: Sequence[str], required: bool = True):
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise ValueError(f"missing parquet column; expected one of {list(candidates)}")
    return None


def _episode_from_filename(path: Path) -> int | None:
    match = re.search(r"episode[_-]?(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _vector(value: object, expected: int, label: str, np):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (expected,):
        raise ValueError(f"{label} must have shape [{expected}], got {array.shape}")
    _finite(array, np, label)
    return array


def _read_episodes(data_dir: Path, np, pq) -> dict[int, Episode]:
    rows: dict[int, list[tuple[int, int, object, object]]] = {}
    parquet_paths = [
        path for path in sorted(data_dir.rglob("*.parquet"))
        if "meta" not in path.relative_to(data_dir).parts
    ]
    global_ordinal = 0
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        state_name = _column_name(names, ("observation.state", "state"))
        action_name = _column_name(names, ("action",))
        episode_name = _column_name(names, ("episode_index", "episode_id"), required=False)
        frame_name = _column_name(names, ("frame_index",), required=False)
        global_name = _column_name(names, ("index",), required=False)
        columns = [state_name, action_name]
        if episode_name:
            columns.append(episode_name)
        if frame_name:
            columns.append(frame_name)
        if global_name:
            columns.append(global_name)
        inferred_episode = _episode_from_filename(path)
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            values = batch.to_pydict()
            count = len(values[state_name])
            for offset in range(count):
                if episode_name:
                    episode_index = int(values[episode_name][offset])
                elif inferred_episode is not None:
                    episode_index = inferred_episode
                else:
                    raise ValueError(f"cannot infer episode index for {path}")
                frame_index = (
                    int(values[frame_name][offset]) if frame_name else ordinal
                )
                state = _vector(values[state_name][offset], STATE_DIM, "state", np)
                action = _vector(values[action_name][offset], ACTION_DIM, "action", np)
                global_index = (
                    int(values[global_name][offset]) if global_name else global_ordinal
                )
                rows.setdefault(episode_index, []).append(
                    (frame_index, global_index, state, action)
                )
                ordinal += 1
                global_ordinal += 1

    episodes: dict[int, Episode] = {}
    for episode_index, episode_rows in rows.items():
        episode_rows.sort(key=lambda item: item[0])
        frame_index = np.asarray([item[0] for item in episode_rows], dtype=np.int64)
        if len(np.unique(frame_index)) != len(frame_index):
            raise ValueError(f"episode {episode_index} has duplicate frame indices")
        if len(frame_index) > 1 and not np.all(np.diff(frame_index) == 1):
            raise ValueError(f"episode {episode_index} has a frame gap; refusing cross-gap windows")
        episodes[episode_index] = Episode(
            episode_index=episode_index,
            global_index=np.asarray([item[1] for item in episode_rows], dtype=np.int64),
            frame_index=frame_index,
            state=np.stack([item[2] for item in episode_rows]).astype(np.float32),
            action=np.stack([item[3] for item in episode_rows]).astype(np.float32),
        )
    if len(episodes) != EXPECTED_DATASET["episodes"]:
        raise ValueError(f"expected 50 decoded episodes, found {len(episodes)}")
    decoded_frames = sum(len(episode.frame_index) for episode in episodes.values())
    if decoded_frames != EXPECTED_DATASET["frames"]:
        raise ValueError(f"expected 19631 decoded frames, found {decoded_frames}")
    global_indices = np.concatenate(
        [episode.global_index for episode in episodes.values()], axis=0
    )
    if not np.array_equal(
        np.sort(global_indices), np.arange(EXPECTED_DATASET["frames"], dtype=np.int64)
    ):
        raise ValueError("parquet global index must cover exactly 0..19630")
    return episodes


def _split_episodes(episode_ids: Iterable[int], seed: int) -> dict[str, list[int]]:
    shuffled = sorted(int(value) for value in episode_ids)
    random.Random(seed).shuffle(shuffled)
    result: dict[str, list[int]] = {}
    cursor = 0
    for name in ("train", "dev", "cal", "test"):
        count = SPLIT_COUNTS[name]
        result[name] = sorted(shuffled[cursor:cursor + count])
        cursor += count
    if cursor != len(shuffled) or len({value for ids in result.values() for value in ids}) != cursor:
        raise ValueError("episode split is incomplete or overlapping")
    return result


def _make_windows(
    name: str, episode_ids: Sequence[int], episodes: dict[int, Episode], stride: int, np
) -> Windows:
    history_index, history_state, history_action = [], [], []
    future_action, future_state, owners = [], [], []
    # Alignment: x[t-7:t+1], already logged u[t-8:t], final future u[t:t+40]
    # predict x[t+1:t+41].  We assume the recorded-flow alignment u[t] ->
    # x[t+1]; the source stores dataset-native teleoperation/processor target
    # values, not confirmed sent actuator commands.  No future state is input.
    for episode_id in episode_ids:
        episode = episodes[episode_id]
        needed = HISTORY + HORIZON
        for start in range(1, len(episode.frame_index) - needed + 1, stride):
            pivot = start + HISTORY - 1
            history_index.append(episode.global_index[start:start + HISTORY])
            history_state.append(episode.state[start:start + HISTORY])
            history_action.append(episode.action[start - 1:start + HISTORY - 1])
            future_action.append(episode.action[pivot:pivot + HORIZON])
            future_state.append(episode.state[pivot + 1:pivot + 1 + HORIZON])
            owners.append(episode_id)
    if not owners:
        raise ValueError(f"split {name} produced no windows")
    return Windows(
        split=name,
        episode_ids=np.asarray(owners, dtype=np.int64),
        history_index=np.stack(history_index).astype(np.int64),
        history_state=np.stack(history_state).astype(np.float32),
        history_action=np.stack(history_action).astype(np.float32),
        future_action=np.stack(future_action).astype(np.float32),
        future_state=np.stack(future_state).astype(np.float32),
    )


def _fit_normalization(train_ids: Sequence[int], episodes: dict[int, Episode], np) -> dict:
    states = np.concatenate([episodes[index].state for index in train_ids], axis=0)
    actions = np.concatenate([episodes[index].action for index in train_ids], axis=0)
    state_std = np.maximum(states.std(axis=0), 1e-6)
    action_std = np.maximum(actions.std(axis=0), 1e-6)
    return {
        "fit_split": "train",
        "representation": (
            "dataset-native logged 6D joint state and teleoperation/processor targets; "
            "exact physical units undeclared; train-only z-score used by this experiment"
        ),
        "state_mean": states.mean(axis=0).astype(float).tolist(),
        "state_std": state_std.astype(float).tolist(),
        "action_mean": actions.mean(axis=0).astype(float).tolist(),
        "action_std": action_std.astype(float).tolist(),
    }


def _load_visual_cache(path: Path, dataset_manifest_hash: str,
                       split_ids: dict[str, list[int]], seed: int, np):
    if not path.is_file():
        raise FileNotFoundError("--visual-cache must be an existing visual_features.npz")
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "index", "episode_index", "features", "dataset_manifest_hash",
            "encoder_weights_sha256", "split_seed", "split_hash",
            "train_split_hash", "cal_split_hash", "pca_fit_split_hash",
            "frame_alignment_verified",
        }
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"visual cache is missing fields: {missing}")
        index = np.asarray(cache["index"], dtype=np.int64)
        episode_index = np.asarray(cache["episode_index"], dtype=np.int64)
        features = np.asarray(cache["features"], dtype=np.float32)
        cached_manifest = str(cache["dataset_manifest_hash"].item())
        cached_seed = int(cache["split_seed"].item())
        cached_split_hash = str(cache["split_hash"].item())
        encoder_sha = str(cache["encoder_weights_sha256"].item())
        alignment_verified = bool(cache["frame_alignment_verified"].item())
        cached_train_hash = str(cache["train_split_hash"].item())
        cached_cal_hash = str(cache["cal_split_hash"].item())
        cached_pca_hash = str(cache["pca_fit_split_hash"].item())
    expected_split_hash = _sha256_value({"seed": seed, "split": split_ids})
    if cached_manifest != dataset_manifest_hash:
        raise ValueError("visual cache dataset manifest does not match --data-dir")
    if cached_seed != seed or cached_split_hash != expected_split_hash:
        raise ValueError("visual cache episode split does not match this training run")
    if not alignment_verified:
        raise ValueError("visual cache does not certify strict frame alignment")
    if cached_train_hash != _sha256_value(split_ids["train"]):
        raise ValueError("visual cache train-split hash does not match this run")
    if cached_pca_hash != cached_train_hash:
        raise ValueError("visual cache PCA was not fit on this run's train split")
    if cached_cal_hash != _sha256_value(split_ids["cal"]):
        raise ValueError("visual cache calibration-split hash does not match this run")
    if index.shape != (EXPECTED_DATASET["frames"],) or not np.array_equal(
        index, np.arange(EXPECTED_DATASET["frames"], dtype=np.int64)
    ):
        raise ValueError("visual cache index must be exactly 0..19630")
    if episode_index.shape != index.shape or features.shape != (len(index), 128):
        raise ValueError("visual cache must contain episode_index [19631] and features [19631,128]")
    _finite(features, np, "visual cache features")
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("visual cache metadata.json is required beside the NPZ")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_sha = _sha256_file(path)
    if metadata.get("frame_alignment_verified") is not True:
        raise ValueError("visual cache metadata does not certify strict frame alignment")
    if metadata.get("cache_sha256") != cache_sha:
        raise ValueError("visual cache SHA disagrees with metadata.json")
    if metadata.get("dataset_manifest_hash") != dataset_manifest_hash:
        raise ValueError("visual cache metadata dataset hash does not match --data-dir")
    if metadata.get("encoder", {}).get("weights_sha256") != encoder_sha:
        raise ValueError("visual cache encoder SHA disagrees with metadata.json")
    if encoder_sha.removeprefix("sha256:") != OFFICIAL_RESNET18_SHA256:
        raise ValueError("visual cache did not use the full verified official ResNet18 weights")
    if metadata.get("split_hash") != expected_split_hash:
        raise ValueError("visual cache split hash disagrees with metadata.json")
    train_mask = np.isin(episode_index, split_ids["train"])
    if int(train_mask.sum()) == 0:
        raise ValueError("visual cache contains no train-split frames")
    visual_std = np.maximum(features[train_mask].std(axis=0), 1e-6)
    provenance = {
        "path": str(path.resolve()),
        "sha256": cache_sha,
        "metadata_sha256": _sha256_file(metadata_path),
        "encoder_weights_sha256": encoder_sha,
        "dataset_manifest_hash": cached_manifest,
        "split_hash": cached_split_hash,
        "frame_alignment_verified": True,
        "feature_dim": 128,
        "fit_on_train_episodes_only": True,
        "spatial_semantics": "fixed 2x2 image-grid slots; not tracked objects",
        "pending_labels": ["object_pose", "contact", "collision", "safety"],
    }
    scaling = {
        "visual_mean": features[train_mask].mean(axis=0).astype(float).tolist(),
        "visual_std": visual_std.astype(float).tolist(),
        "visual_scaling_fit_split": "30 train episodes only",
    }
    return {int(key): value for key, value in zip(index.tolist(), features)}, scaling, provenance


def _attach_visual(windows: Windows, feature_by_index: dict[int, object], np) -> Windows:
    try:
        visual = np.stack(
            [feature_by_index[int(index)] for index in windows.history_index.reshape(-1)]
        ).reshape(len(windows), HISTORY, 128).astype(np.float32)
    except KeyError as exc:
        raise ValueError(f"visual cache lacks parquet global index {exc.args[0]}") from exc
    return Windows(
        split=windows.split,
        episode_ids=windows.episode_ids,
        history_index=windows.history_index,
        history_state=windows.history_state,
        history_action=windows.history_action,
        future_action=windows.future_action,
        future_state=windows.future_state,
        history_visual=visual,
    )


def _seed_everything(seed: int, torch, np) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _model_class(torch):
    nn = torch.nn

    class JointDynamicsGRU(nn.Module):
        def __init__(self, normalization: dict, hidden: int, use_future_action: bool,
                     visual_dim: int = 0):
            super().__init__()
            self.hidden = hidden
            self.use_future_action = use_future_action
            self.visual_dim = visual_dim
            self.encoder = nn.GRU(
                STATE_DIM + ACTION_DIM + visual_dim, hidden, batch_first=True
            )
            decoder_width = STATE_DIM + (ACTION_DIM if use_future_action else 0)
            self.decoder = nn.GRUCell(decoder_width, hidden)
            self.delta_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, STATE_DIM)
            )
            self.register_buffer("state_mean", torch.tensor(normalization["state_mean"], dtype=torch.float32))
            self.register_buffer("state_std", torch.tensor(normalization["state_std"], dtype=torch.float32))
            self.register_buffer("action_mean", torch.tensor(normalization["action_mean"], dtype=torch.float32))
            self.register_buffer("action_std", torch.tensor(normalization["action_std"], dtype=torch.float32))
            if visual_dim:
                self.register_buffer(
                    "visual_mean", torch.tensor(normalization["visual_mean"], dtype=torch.float32)
                )
                self.register_buffer(
                    "visual_std", torch.tensor(normalization["visual_std"], dtype=torch.float32)
                )

        def normalize_state(self, value):
            return (value - self.state_mean) / self.state_std

        def normalize_action(self, value):
            return (value - self.action_mean) / self.action_std

        def forward(self, history_state, history_action, future_action, history_visual=None):
            inputs = [
                self.normalize_state(history_state),
                self.normalize_action(history_action),
            ]
            if self.visual_dim:
                if history_visual is None or history_visual.shape[-1] != self.visual_dim:
                    raise ValueError("visual model requires correctly shaped history_visual")
                inputs.append((history_visual - self.visual_mean) / self.visual_std)
            elif history_visual is not None:
                raise ValueError("state-only model received history_visual")
            encoded = torch.cat(inputs, dim=-1)
            _, hidden = self.encoder(encoded)
            hidden = hidden[0]
            state = self.normalize_state(history_state[:, -1])
            predictions = []
            for index in range(future_action.shape[1]):
                if self.use_future_action:
                    decoder_input = torch.cat(
                        (state, self.normalize_action(future_action[:, index])), dim=-1
                    )
                else:
                    decoder_input = state
                hidden = self.decoder(decoder_input, hidden)
                state = state + self.delta_head(hidden)
                predictions.append(state)
            return torch.stack(predictions, dim=1)

        def denormalize_state(self, normalized):
            return normalized * self.state_std + self.state_mean

    return JointDynamicsGRU


def _batch(windows: Windows, indices, normalization: dict, torch, device):
    # The model owns normalization buffers.  Targets are normalized explicitly
    # for the loss using the identical train-only state statistics.
    history_state = torch.from_numpy(windows.history_state[indices]).to(device)
    history_action = torch.from_numpy(windows.history_action[indices]).to(device)
    future_action = torch.from_numpy(windows.future_action[indices]).to(device)
    history_visual = (
        torch.from_numpy(windows.history_visual[indices]).to(device)
        if windows.history_visual is not None else None
    )
    truth = torch.from_numpy(windows.future_state[indices]).to(device)
    mean = torch.tensor(normalization["state_mean"], device=device)
    std = torch.tensor(normalization["state_std"], device=device)
    truth_normalized = (truth - mean) / std
    return history_state, history_action, future_action, history_visual, truth_normalized


def _predict(model, windows: Windows, torch, device, batch_size: int, np):
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            indices = np.arange(start, stop)
            hs = torch.from_numpy(windows.history_state[indices]).to(device)
            ha = torch.from_numpy(windows.history_action[indices]).to(device)
            fa = torch.from_numpy(windows.future_action[indices]).to(device)
            hv = (
                torch.from_numpy(windows.history_visual[indices]).to(device)
                if windows.history_visual is not None else None
            )
            normalized = model(hs, ha, fa, hv)
            rows.append(model.denormalize_state(normalized).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _ensemble_predict(models, windows, torch, device, batch_size, np, future_action=None):
    if future_action is not None:
        windows = Windows(
            split=windows.split,
            episode_ids=windows.episode_ids,
            history_index=windows.history_index,
            history_state=windows.history_state,
            history_action=windows.history_action,
            future_action=future_action,
            future_state=windows.future_state,
            history_visual=windows.history_visual,
        )
    predictions = [_predict(model, windows, torch, device, batch_size, np) for model in models]
    return np.mean(predictions, axis=0)


def _dev_loss(model, dev: Windows, normalization: dict, torch, device, batch_size: int, np):
    prediction = _predict(model, dev, torch, device, batch_size, np)
    normalized_error = (
        prediction - dev.future_state
    ) / np.asarray(normalization["state_std"], dtype=np.float32)
    return float(np.mean(normalized_error**2))


def _copy_state_dict_cpu(model, torch):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_family(
    family: str,
    use_future_action: bool,
    members: int,
    hidden: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    train: Windows,
    dev: Windows,
    normalization: dict,
    base_seed: int,
    eval_every: int,
    torch,
    np,
    device,
    log_handle,
    use_visual: bool = False,
):
    Model = _model_class(torch)
    models = []
    summaries = []
    for member in range(members):
        member_seed = base_seed + member * 1009 + (0 if use_future_action else 100_000)
        _seed_everything(member_seed, torch, np)
        model = Model(normalization, hidden, use_future_action, 128 if use_visual else 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        rng = np.random.default_rng(member_seed)
        best_loss = math.inf
        best_state = None
        best_step = 0
        started = time.perf_counter()
        for step in range(1, steps + 1):
            indices = rng.integers(0, len(train), size=batch_size)
            history_state, history_action, future_action, history_visual, truth = _batch(
                train, indices, normalization, torch, device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_state, history_action, future_action, history_visual)
            loss = torch.mean((prediction - truth) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{family} member {member} produced non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % eval_every == 0 or step == steps:
                dev_loss = _dev_loss(model, dev, normalization, torch, device, batch_size, np)
                row = {
                    "family": family,
                    "member": member,
                    "seed": member_seed,
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "dev_normalized_mse": dev_loss,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                log_handle.flush()
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_step = step
                    best_state = _copy_state_dict_cpu(model, torch)
        if best_state is None:
            raise RuntimeError("dev selection did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        summaries.append(
            {
                "member": member,
                "seed": member_seed,
                "best_step": best_step,
                "best_dev_normalized_mse": best_loss,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    return models, summaries


def _save_weights(path: Path, model, np) -> None:
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez_compressed(path, **arrays)


def _persistence(windows: Windows, np):
    return np.repeat(windows.history_state[:, -1:, :], HORIZON, axis=1)


def _action_target(windows: Windows, np):
    # This is a deliberately simple native-value baseline.  It does not claim
    # logged processor targets equal observed state or were sent to actuators.
    return windows.future_action.copy()


def _metric_summary(prediction, windows: Windows, state_std, np) -> dict:
    error = np.abs(prediction - windows.future_state)
    normalized_error = error / np.asarray(state_std, dtype=np.float32)[None, None, :]
    return {
        "windows": len(windows),
        "mae_all": float(error.mean()),
        "mae_by_joint": error.mean(axis=(0, 1)).astype(float).tolist(),
        "mae_by_horizon": {
            str(horizon): float(error[:, horizon - 1].mean())
            for horizon in (1, 10, 20, 40)
        },
        "final_step_mae_by_joint": error[:, -1].mean(axis=0).astype(float).tolist(),
        "max_absolute_error": float(error.max()),
        "native_value_semantics": "dataset-native values; exact physical units undeclared",
        "train_zscore_absolute_error": {
            "mae_all": float(normalized_error.mean()),
            "mae_by_joint": normalized_error.mean(axis=(0, 1)).astype(float).tolist(),
            "mae_by_horizon": {
                str(horizon): float(normalized_error[:, horizon - 1].mean())
                for horizon in (1, 10, 20, 40)
            },
            "final_step_mae_by_joint": (
                normalized_error[:, -1].mean(axis=0).astype(float).tolist()
            ),
            "max_absolute_error": float(normalized_error.max()),
            "scale": "train-only observation.state standard deviation per joint",
        },
    }


def _fit_dev_scales(prediction, dev: Windows, np):
    error = prediction - dev.future_state
    return np.maximum(np.sqrt(np.mean(error**2, axis=(0, 1))), 1e-6)


def _calibrate_by_episode(prediction, cal: Windows, scales, alpha: float, np):
    episode_scores = []
    for episode_id in sorted(np.unique(cal.episode_ids).astype(int).tolist()):
        mask = cal.episode_ids == episode_id
        score = float(
            np.max(np.abs(prediction[mask] - cal.future_state[mask]) / scales[None, None, :])
        )
        episode_scores.append({"episode_index": episode_id, "score": score})
    count = len(episode_scores)
    rank = int(math.ceil((count + 1) * (1.0 - alpha)))
    q = (
        float(sorted(item["score"] for item in episode_scores)[rank - 1])
        if rank <= count else None
    )
    return {
        "alpha": alpha,
        "episode_count": count,
        "rank": rank,
        "q": q,
        "finite": q is not None,
        "score_definition": (
            "per-episode max over all windows, 40 future times, and 6 joints; "
            "absolute error divided by dev joint RMSE"
        ),
        "episode_scores": episode_scores,
        "alpha_0_05_note": (
            "With 10 calibration episodes alpha=0.05 has rank 11 and no finite threshold; "
            "this run does not claim 95% finite calibration."
        ),
    }


def _test_by_episode(prediction, test: Windows, scales, calibration: dict,
                     state_std, np):
    q = calibration["q"]
    rows = []
    for episode_id in sorted(np.unique(test.episode_ids).astype(int).tolist()):
        mask = test.episode_ids == episode_id
        metrics = _metric_summary(prediction[mask], Windows(
            split=test.split,
            episode_ids=test.episode_ids[mask],
            history_index=test.history_index[mask],
            history_state=test.history_state[mask],
            history_action=test.history_action[mask],
            future_action=test.future_action[mask],
            future_state=test.future_state[mask],
            history_visual=(
                test.history_visual[mask] if test.history_visual is not None else None
            ),
        ), state_std, np)
        if q is None:
            covered = None
            max_score = None
        else:
            normalized = np.abs(prediction[mask] - test.future_state[mask]) / scales[None, None, :]
            max_score = float(np.max(normalized))
            covered = bool(max_score <= q)
        rows.append(
            {
                "episode_index": episode_id,
                **metrics,
                "joint_trajectory_covered": covered,
                "max_normalized_score": max_score,
            }
        )
    covered_values = [row["joint_trajectory_covered"] for row in rows]
    finite_values = [value for value in covered_values if value is not None]
    return rows, {
        "covered_episodes": sum(bool(value) for value in finite_values) if finite_values else None,
        "test_episodes": len(rows),
        "joint_episode_coverage": (
            sum(bool(value) for value in finite_values) / len(finite_values)
            if finite_values else None
        ),
        "mean_full_width_by_joint": (
            (2.0 * q * scales).astype(float).tolist() if q is not None else None
        ),
    }


def _evaluate_method(name: str, dev_prediction, cal_prediction, test_prediction,
                     dev: Windows, cal: Windows, test: Windows, alpha: float,
                     state_std, np):
    scales = _fit_dev_scales(dev_prediction, dev, np)
    calibration = _calibrate_by_episode(cal_prediction, cal, scales, alpha, np)
    rows, coverage = _test_by_episode(
        test_prediction, test, scales, calibration, state_std, np
    )
    return {
        "method": name,
        "dev_joint_rmse_scales": scales.astype(float).tolist(),
        "calibration": calibration,
        "test_summary": {
            **_metric_summary(test_prediction, test, state_std, np), **coverage
        },
        "per_episode": rows,
    }


def _artifact_manifest(out: Path) -> dict:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("artifacts_manifest.json", "COMPLETE.json"):
            files.append(
                {
                    "path": path.relative_to(out).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    result = {"files": files}
    result["manifest_hash"] = _sha256_value(result)
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the SO100 real-data action-conditioned joint-dynamics shadow GRU"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--visual-cache", type=Path, default=None,
        help="optional visual_features.npz from cache_visual_features.py",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.steps < 1 or args.members < 1 or args.hidden < 1 or args.batch_size < 1:
        raise ValueError("steps, members, hidden, and batch-size must be positive")
    if args.stride < 1 or args.eval_every < 1:
        raise ValueError("stride and eval-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if not args.source_commit.strip():
        raise ValueError("source-commit must be non-empty")
    if not args.data_dir.is_dir():
        raise FileNotFoundError("--data-dir must be an existing directory")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("--out must be a new or empty directory")

    try:
        import numpy as np
        import pyarrow.parquet as pq
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "this experiment requires numpy, pyarrow, and torch on the training host"
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights").mkdir()

    started = time.time()
    info_path, info = _load_info(args.data_dir)
    _validate_dataset_identity(info)
    dataset_manifest = _local_dataset_manifest(args.data_dir, info_path)
    episodes = _read_episodes(args.data_dir, np, pq)
    split_ids = _split_episodes(episodes, args.seed)
    state_splits = {
        name: _make_windows(name, ids, episodes, args.stride, np)
        for name, ids in split_ids.items()
    }
    normalization = _fit_normalization(split_ids["train"], episodes, np)
    normalization["state_action_scaling_split_hash"] = _sha256_value(
        split_ids["train"]
    )
    visual_provenance = None
    visual_splits = None
    if args.visual_cache is not None:
        feature_by_index, visual_scaling, visual_provenance = _load_visual_cache(
            args.visual_cache,
            dataset_manifest["dataset_manifest_hash"],
            split_ids,
            args.seed,
            np,
        )
        normalization.update(visual_scaling)
        visual_splits = {
            name: _attach_visual(windows, feature_by_index, np)
            for name, windows in state_splits.items()
        }

    split_artifact = {
        "seed": args.seed,
        "counts": SPLIT_COUNTS,
        "episode_indices": split_ids,
        "split_hash": _sha256_value({"seed": args.seed, "split": split_ids}),
        "train_split_hash": _sha256_value(split_ids["train"]),
        "cal_split_hash": _sha256_value(split_ids["cal"]),
        "windows": {name: len(value) for name, value in state_splits.items()},
        "history": HISTORY,
        "future": HORIZON,
        "stride": args.stride,
        "window_alignment": (
            "history x[s:s+8], already logged u[s-1:s+7], final future u[t:t+40], "
            "targets x[t+1:t+41], t=s+7; assumes recorded-flow alignment u[t]->x[t+1]; "
            "never crosses episode"
        ),
    }
    _write_json(args.out / "dataset_source.json", dataset_manifest)
    _write_json(args.out / "split.json", split_artifact)
    _write_json(args.out / "normalization.json", normalization)

    device = _resolve_device(args.device, torch)
    config = {
        "experiment": (
            "W1 visual joint-dynamics shadow"
            if visual_splits is not None else "W1 state-only joint-dynamics shadow"
        ),
        "dataset": "lerobot/svla_so100_pickplace",
        "scope": (
            "predict future dataset-native logged 6D joint state from history and logged final "
            "teleoperation/processor targets; visual mode remains a joint-dynamics shadow, not "
            "a v4 object-consequence, Cartesian robotics, or safety model"
        ),
        "source_commit": args.source_commit,
        "script_sha256": _sha256_file(Path(__file__)),
        "dataset_manifest_hash": dataset_manifest["dataset_manifest_hash"],
        "download_manifest_sha256": dataset_manifest["download_manifest"]["sha256"],
        "dataset_snapshot_revision": dataset_manifest["download_manifest"]["revision"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": __import__("pyarrow").__version__,
        "device_requested": args.device,
        "device_resolved": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "history": HISTORY,
        "future": HORIZON,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "members": args.members,
        "hidden": args.hidden,
        "steps_per_member": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dev_selection_every_steps": args.eval_every,
        "alpha": args.alpha,
        "units_and_semantics": {
            "state": (
                "dataset-native logged 6D observation.state; joint names declared, exact "
                "physical units undeclared by the source snapshot"
            ),
            "action": (
                "dataset-native logged 6D teleoperation/processor target values; not confirmed "
                "as commands actually sent to actuators"
            ),
            "source_capture_caveat": (
                "recording flow stores action_values; no _sent_action field establishes actual "
                "actuator transmission"
            ),
            "temporal_alignment": "assumed recorded-flow relationship u[t] -> x[t+1]",
            "experiment_normalization": "train-only per-joint z-score",
            "cartesian_conversion": False,
            "images_used": visual_splits is not None,
            "visual_history_only": visual_splits is not None,
            "visual_feature_dim": 128 if visual_splits is not None else 0,
            "visual_spatial_slots_are_tracked_objects": False,
            "future_state_as_input": False,
            "future_image_as_input": False,
        },
        "visual_cache": visual_provenance,
    }
    _write_json(args.out / "training_config.json", config)

    with (args.out / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        action_models, action_training = _train_family(
            "state_action_conditioned" if visual_splits is not None else "action_conditioned",
            True,
            args.members,
            args.hidden,
            args.steps,
            args.batch_size,
            args.learning_rate,
            state_splits["train"],
            state_splits["dev"],
            normalization,
            args.seed,
            args.eval_every,
            torch,
            np,
            device,
            log_handle,
        )
        if visual_splits is not None:
            visual_models, visual_training = _train_family(
                "visual_state_action_conditioned",
                True,
                args.members,
                args.hidden,
                args.steps,
                args.batch_size,
                args.learning_rate,
                visual_splits["train"],
                visual_splits["dev"],
                normalization,
                args.seed,
                args.eval_every,
                torch,
                np,
                device,
                log_handle,
                use_visual=True,
            )
        else:
            visual_models, visual_training = None, None
        no_action_models, no_action_training = _train_family(
            "visual_no_future_action" if visual_splits is not None else "no_future_action",
            False,
            args.members,
            args.hidden,
            args.steps,
            args.batch_size,
            args.learning_rate,
            (visual_splits or state_splits)["train"],
            (visual_splits or state_splits)["dev"],
            normalization,
            args.seed,
            args.eval_every,
            torch,
            np,
            device,
            log_handle,
            use_visual=visual_splits is not None,
        )

    weight_artifacts = []
    model_families = [(
        "state_action" if visual_models is not None else "action", action_models
    )]
    if visual_models is not None:
        model_families.append(("visual_state_action", visual_models))
        model_families.append(("visual_no_future_action", no_action_models))
    else:
        model_families.append(("no_future_action", no_action_models))
    for family, models in model_families:
        for index, model in enumerate(models):
            weight_path = args.out / "weights" / f"{family}_member_{index:02d}.npz"
            _save_weights(weight_path, model, np)
            weight_artifacts.append(
                {
                    "family": family,
                    "member": index,
                    "path": weight_path.relative_to(args.out).as_posix(),
                    "sha256": _sha256_file(weight_path),
                }
            )
    _write_json(
        args.out / "model_metadata.json",
        {
            "architecture": (
                "GRU encoder + autoregressive GRUCell train-z-scored state-delta decoder"
            ),
            "action_conditioned_members": action_training,
            "visual_action_conditioned_members": visual_training,
            "independently_trained_no_future_action_members": no_action_training,
            "weight_format": "numeric-only compressed NPZ; load with allow_pickle=False",
            "weights": weight_artifacts,
            "selection": "minimum dev train-zscore MSE; test not read during selection",
            "visual_cache": visual_provenance,
            "object_pose_label_status": (
                "pending/not available; never used as a target"
                if visual_provenance is not None else None
            ),
        },
    )

    # Freeze all predictions before reading any test metric.  Test is not used to
    # tune checkpoints, scales, calibration, hyperparameters, or thresholds.
    prediction_sets: dict[str, dict[str, object]] = {}
    for method in ("persistence", "action_target"):
        function = _persistence if method == "persistence" else _action_target
        prediction_sets[method] = {
            name: function(state_splits[name], np) for name in ("dev", "cal", "test")
        }
    state_method = (
        "state_action_conditioned_gru"
        if visual_models is not None else "action_conditioned_gru"
    )
    learned_methods = [(state_method, action_models, state_splits)]
    if visual_models is not None:
        learned_methods.extend(
            [
                ("visual_state_action_conditioned_gru", visual_models, visual_splits),
                ("visual_trained_no_future_action_gru", no_action_models, visual_splits),
            ]
        )
    else:
        learned_methods.append(
            ("trained_no_future_action_gru", no_action_models, state_splits)
        )
    for method, models, method_splits in learned_methods:
        prediction_sets[method] = {
            name: _ensemble_predict(
                models, method_splits[name], torch, device, args.batch_size, np
            )
            for name in ("dev", "cal", "test")
        }

    shuffle_rng = np.random.default_rng(args.seed + 900_001)
    primary_models = visual_models if visual_models is not None else action_models
    primary_splits = visual_splits if visual_splits is not None else state_splits
    primary_name = (
        "visual_state_action_conditioned_gru"
        if visual_models is not None else "action_conditioned_gru"
    )
    source = primary_splits["test"].future_action
    permutation = shuffle_rng.permutation(len(source))
    # Freeze the normal action model's dev scale and calibration.  Only the
    # held-out test actions are mismatched.  Recalibrating on shuffled dev/cal
    # data would hide the causal degradation this ablation is intended to show.
    prediction_sets["test_time_shuffled_future_action"] = {
        "dev": prediction_sets[primary_name]["dev"],
        "cal": prediction_sets[primary_name]["cal"],
        "test": _ensemble_predict(
            primary_models,
            primary_splits["test"],
            torch,
            device,
            args.batch_size,
            np,
            future_action=source[permutation],
        ),
    }

    methods = []
    for name, values in prediction_sets.items():
        methods.append(
            _evaluate_method(
                name,
                values["dev"],
                values["cal"],
                values["test"],
                state_splits["dev"],
                state_splits["cal"],
                state_splits["test"],
                args.alpha,
                normalization["state_std"],
                np,
            )
        )

    test_results = {
        "test_evaluations": 1,
        "independent_unit": "episode",
        "methods": methods,
        "decision_metrics": {
            "allow_rate": None,
            "incident_rate": None,
            "false_allow_rate": None,
            "reason": (
                "This observational pick-place dataset has no counterfactual incident/safety label. "
                "Prediction error and joint trajectory coverage cannot generate an allow decision "
                "or accident-rate claim."
            ),
        },
        "limitations": [
            "real-data joint-dynamics shadow model only",
            (
                "frozen historical image-grid features are model inputs; they are not object tracks"
                if visual_splits is not None else
                "images are present in the source dataset but are not model inputs"
            ),
            "no object-state, contact, collision, Cartesian, force, or safety labels are invented",
            "logged teleoperation/processor targets are not confirmed sent actuator commands",
            "exact physical units are undeclared by the source snapshot",
            "future images are never model inputs",
            "episode-level split conformal coverage is marginal under exchangeability assumptions",
            "only five held-out test episodes; report numerator and denominator",
        ],
    }
    _write_json(args.out / "test_results.json", test_results)

    manifest = _artifact_manifest(args.out)
    _write_json(args.out / "artifacts_manifest.json", manifest)
    complete = {
        "status": "completed",
        "completed_unix_seconds": time.time(),
        "wall_seconds": time.time() - started,
        "artifact_manifest_hash": manifest["manifest_hash"],
        "dataset_manifest_hash": dataset_manifest["dataset_manifest_hash"],
        "scope": config["scope"],
    }
    _write_json(args.out / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
