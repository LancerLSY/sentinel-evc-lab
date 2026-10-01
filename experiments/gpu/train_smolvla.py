#!/usr/bin/env python3
"""Offline-only SmolVLA fine-tuning protocol for SVLA SO100 PickPlace.

This is an experiment-host entry point.  It is deliberately outside the
installable Sentinel package and never sends an action to a robot, enables
Weights & Biases, or uploads to the Hugging Face Hub.  It consumes predownloaded
local inputs only:

* LeRobot v0.6.1, which supports LeRobotDataset v3.0;
* a local snapshot of ``lerobot/svla_so100_pickplace``;
* local snapshots of ``lerobot/smolvla_base`` and its SmolVLM backbone.

The protocol makes a seeded 30/5/10/5 episode split, fits normalization only on
the 30 training episodes, picks a checkpoint only by the five development
episodes, then runs one held-out offline test pass.  Its ten shadow windows
preserve camera/state/action tensors and dataset-native action chunks with
hashes. Dataset metadata does not declare physical units. They are evidence for
offline predictions only, not robot
closed-loop behavior, safety, latency, or task success.

Upstream API basis (pinned to LeRobot v0.6.1):
https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_train.py
https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LEROBOT_VERSION = "0.6.1"
DATASET_REPO_ID = "lerobot/svla_so100_pickplace"
MODEL_REPO_ID = "lerobot/smolvla_base"
VLM_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SPLIT_SEED = 20261002
W1_SPLIT_HASH = "sha256:174c11723a1319508fd8f6282baa18a5600c8b8e7286d7918c9bfd67b91b1618"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"

# This entry point accepts only predownloaded local snapshots.  Set these before
# importing LeRobot/Transformers/HF Hub so an incomplete cache fails locally
# instead of quietly changing the experiment by downloading a moving revision.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@dataclass(frozen=True)
class EpisodeSplits:
    train: list[int]
    dev: list[int]
    cal: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train,
            "dev": self.dev,
            "cal": self.cal,
            "test": self.test,
        }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_or_verify_json(path: Path, value: object, description: str) -> None:
    """Persist immutable run metadata once, or prove an exact resume match."""
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Existing {description} is not valid JSON: {path}") from error
        if previous != value:
            raise RuntimeError(f"Existing {description} differs; refuse to mix experiment provenance: {path}")
        return
    _write_json(path, value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_value(value: object) -> str:
    return "sha256:" + _sha256_bytes(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_immutable_revision(value: str, name: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 40-character lowercase immutable commit SHA, not a tag: {value!r}")
    return value


def _manifest_value(manifest: dict[str, Any], keys: tuple[str, ...], description: str) -> Any:
    for key in keys:
        if key in manifest:
            return manifest[key]
    raise RuntimeError(f"download_manifest.json is missing {description}; accepted keys: {', '.join(keys)}")


def _manifest_files(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [{"path": path, **entry} if isinstance(entry, dict) else {"path": path, "sha256": entry} for path, entry in value.items()]
    if isinstance(value, list) and all(isinstance(entry, dict) for entry in value):
        return value
    raise RuntimeError("download_manifest.json files must be a list of objects or a path-keyed object")


def _verify_download_manifest(root: Path, supplied_path: Path | None, expected_repo: str, expected_revision: str, role: str) -> dict[str, Any]:
    """Verify the immutable local snapshot once before loading any model/data."""
    path = supplied_path or root / "download_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{role} needs a verified download_manifest.json: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid {role} download manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{role} download manifest must contain a JSON object")
    repo_id = _manifest_value(manifest, ("repo_id", "repo", "repository"), "repository id")
    revision = _manifest_value(manifest, ("revision", "commit", "commit_sha", "resolved_revision"), "immutable revision")
    if repo_id != expected_repo:
        raise RuntimeError(f"{role} manifest repo mismatch: {repo_id!r} != {expected_repo!r}")
    if revision != expected_revision:
        raise RuntimeError(f"{role} manifest revision mismatch: {revision!r} != {expected_revision!r}")
    records: list[dict[str, Any]] = []
    for entry in _manifest_files(_manifest_value(manifest, ("files",), "file list")):
        relative = _manifest_value(entry, ("path", "name", "filename", "relative_path", "rfilename"), "file path")
        expected_bytes = _manifest_value(entry, ("bytes", "size", "size_bytes"), "file bytes")
        expected_sha = _manifest_value(entry, ("sha256", "sha256sum"), "file SHA-256")
        if not isinstance(relative, str) or not isinstance(expected_bytes, int) or not isinstance(expected_sha, str):
            raise RuntimeError(f"{role} manifest has invalid file record: {entry!r}")
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents or candidate == root.resolve():
            raise RuntimeError(f"{role} manifest file path escapes snapshot root: {relative!r}")
        if not candidate.is_file():
            raise FileNotFoundError(f"{role} manifest file is absent: {relative}")
        actual_bytes = candidate.stat().st_size
        actual_sha = _sha256_file(candidate)
        normalized_sha = expected_sha.removeprefix("sha256:").lower()
        if len(normalized_sha) != 64 or any(char not in "0123456789abcdef" for char in normalized_sha):
            raise RuntimeError(f"{role} manifest has an invalid SHA-256 for {relative}")
        if actual_bytes != expected_bytes or actual_sha != normalized_sha:
            raise RuntimeError(f"{role} manifest integrity mismatch for {relative}")
        records.append({"path": relative, "bytes": actual_bytes, "sha256": actual_sha})
    if not records:
        raise RuntimeError(f"{role} manifest declares no files")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "manifest_sha256": _sha256_file(path),
        "files": records,
    }


def _source_commit() -> str:
    """Avoid a subprocess dependency; provenance still records an unknown checkout."""
    head = Path(__file__).resolve().parents[2] / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            return (head.parent / value.removeprefix("ref: ")).read_text(encoding="utf-8").strip()
        return value
    except OSError:
        return "unknown"


def _require_local_dir(path: Path, name: str, required: Iterable[str]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} must be a predownloaded local directory: {path}")
    missing = [filename for filename in required if not (path / filename).exists()]
    if missing:
        raise FileNotFoundError(f"{name} is incomplete at {path}; missing {missing}")


def _require_dependencies():
    """Import the exact upstream API surface only after giving a useful error."""
    if sys.version_info < (3, 12):
        raise RuntimeError("lerobot==0.6.1 requires Python 3.12 or newer")
    try:
        import importlib.metadata

        installed = importlib.metadata.version("lerobot")
        av_version = importlib.metadata.version("av")
    except Exception as error:  # pragma: no cover - executes only on experiment host
        raise RuntimeError(
            "Install the pinned experiment dependencies: `lerobot[smolvla]==0.6.1`, "
            "datasets>=4.8,<5, pyarrow>=21,<30, pandas>=2,<3, and av>=15,<16."
        ) from error
    if installed != LEROBOT_VERSION:
        raise RuntimeError(
            f"This protocol is pinned to lerobot=={LEROBOT_VERSION}, found {installed}. "
            "Do not run it against an unpinned main branch."
        )
    if not av_version.startswith("15."):
        raise RuntimeError(
            f"The pinned LeRobot dataset extra requires av>=15,<16 for the forced pyav backend; found av=={av_version}."
        )
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.collate import lerobot_collate_fn
    except ImportError as error:  # pragma: no cover - executes only on experiment host
        raise RuntimeError(
            "Missing experiment dependencies. Install `lerobot[smolvla]==0.6.1`, datasets, pyarrow, pandas, and av."
        ) from error
    return torch, PreTrainedConfig, LeRobotDataset, make_policy, make_pre_post_processors, lerobot_collate_fn


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cudnn keeps the split/checkpoint protocol auditable.  We do not
    # force all PyTorch kernels deterministic because that can reject valid VLM ops.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def _make_splits(total_episodes: int) -> EpisodeSplits:
    if total_episodes != 50:
        raise ValueError(
            f"Expected exactly 50 episodes for the fixed protocol, found {total_episodes}. "
            "Do not silently reinterpret the split."
        )
    episodes = list(range(total_episodes))
    random.Random(SPLIT_SEED).shuffle(episodes)
    return EpisodeSplits(
        train=sorted(episodes[:30]),
        dev=sorted(episodes[30:35]),
        cal=sorted(episodes[35:45]),
        test=sorted(episodes[45:50]),
    )


def _persist_or_verify_splits(output_dir: Path, splits: EpisodeSplits) -> None:
    path = output_dir / "episode_splits.json"
    split_ids = splits.as_dict()
    split_hash = _sha256_value({"seed": SPLIT_SEED, "split": split_ids})
    if split_hash != W1_SPLIT_HASH:
        raise RuntimeError(
            f"VLA split no longer matches W1/cache split contract: {split_hash} != {W1_SPLIT_HASH}"
        )
    payload = {
        "seed": SPLIT_SEED,
        "counts": {"train": 30, "dev": 5, "cal": 10, "test": 5},
        "episode_indices": split_ids,
        "split_hash": split_hash,
        "train_split_hash": _sha256_value(split_ids["train"]),
        "cal_split_hash": _sha256_value(split_ids["cal"]),
    }
    _write_or_verify_json(path, payload, "episode split manifest")


def _train_only_stats(dataset, torch) -> dict[str, dict[str, Any]]:
    """Compute exact mean/std/min/max from train episode parquet columns only.

    LeRobot's dataset-level metadata statistics cover all episodes.  Passing them to
    the processor would leak dev/calibration/test information, so this intentionally
    streams just two raw columns from the train-filtered dataset.
    """
    columns = dataset.select_columns(["observation.state", "action"])
    accumulators: dict[str, dict[str, Any]] = {}
    for key in ("observation.state", "action"):
        values0 = columns[0][key]
        dim = int(torch.as_tensor(values0).numel())
        accumulators[key] = {
            "count": 0,
            "sum": torch.zeros(dim, dtype=torch.float64),
            "sum_sq": torch.zeros(dim, dtype=torch.float64),
            "min": torch.full((dim,), float("inf"), dtype=torch.float64),
            "max": torch.full((dim,), float("-inf"), dtype=torch.float64),
        }
    block_size = 4096
    for start in range(0, len(columns), block_size):
        rows = columns[start : start + block_size]
        for key, values in rows.items():
            if key not in accumulators:
                continue
            # LeRobot v0.6.1 returns a Python list of per-row tensors for a
            # sliced selected column, which torch.as_tensor cannot convert.
            # Stack that documented runtime representation before reducing it.
            if isinstance(values, list):
                tensor = torch.stack([torch.as_tensor(value, dtype=torch.float64) for value in values])
            else:
                tensor = torch.as_tensor(values, dtype=torch.float64)
            tensor = tensor.reshape(-1, accumulators[key]["sum"].numel())
            item = accumulators[key]
            item["count"] += tensor.shape[0]
            item["sum"] += tensor.sum(dim=0)
            item["sum_sq"] += tensor.square().sum(dim=0)
            item["min"] = torch.minimum(item["min"], tensor.min(dim=0).values)
            item["max"] = torch.maximum(item["max"], tensor.max(dim=0).values)
    stats: dict[str, dict[str, Any]] = {}
    for key, item in accumulators.items():
        count = item["count"]
        if count < 2:
            raise RuntimeError(f"Not enough training values to fit stats for {key}")
        mean = item["sum"] / count
        # Population standard deviation matches the conventional dataset statistic.
        variance = torch.clamp(item["sum_sq"] / count - mean.square(), min=0.0)
        stats[key] = {
            "mean": mean.float(),
            "std": torch.sqrt(variance).clamp_min(1e-6).float(),
            "min": item["min"].float(),
            "max": item["max"].float(),
        }
    return stats


def _serializable_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    return {
        key: {stat: tensor.detach().cpu().tolist() for stat, tensor in values.items()}
        for key, values in stats.items()
    }


class _ResumableTrainBatchSampler:
    """A deterministic infinite-epoch order that resumes at an update boundary."""

    def __init__(self, size: int, batch_size: int, seed: int, start_step: int, stop_step: int, torch) -> None:
        if size < batch_size:
            raise ValueError(f"Training set has {size} frames, smaller than batch size {batch_size}")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step
        self.torch = torch
        self.batches_per_epoch = size // batch_size

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            epoch, batch_in_epoch = divmod(step, self.batches_per_epoch)
            generator = self.torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            order = self.torch.randperm(self.size, generator=generator).tolist()
            offset = batch_in_epoch * self.batch_size
            yield order[offset : offset + self.batch_size]

    def __len__(self) -> int:
        return self.stop_step - self.start_step


def _make_dataset(LeRobotDataset, args, episodes: list[int], fps: int, chunk_size: int):
    action_times = [index / fps for index in range(chunk_size)]
    return LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": action_times},
        revision=args.dataset_revision,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )


def _collate_for(dataset, lerobot_collate_fn):
    return lerobot_collate_fn if dataset.meta.has_language_columns else None


def _to_model_batch(batch: dict[str, Any], camera_keys: list[str], preprocessor, torch) -> dict[str, Any]:
    for key in camera_keys:
        if key not in batch:
            raise KeyError(f"Expected camera `{key}` in batch; available keys: {sorted(batch)}")
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32).div_(255.0)
    if "observation.state" not in batch or "action" not in batch or "task" not in batch:
        raise KeyError("SmolVLA requires two images, observation.state, action, and task in every training batch")
    return preprocessor(batch)


def _freeze_for_action_expert_and_state_projection(policy) -> list[str]:
    """Make the requested freeze boundary explicit and fail closed if it drifts."""
    allowed_prefixes = (
        "model.vlm_with_expert.lm_expert.",
        "model.state_proj.",
        "model.action_in_proj.",
        "model.action_out_proj.",
        "model.action_time_mlp_in.",
        "model.action_time_mlp_out.",
    )
    trainable: list[str] = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad = name.startswith(allowed_prefixes)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("No trainable action-expert/state-projection parameters found; upstream names changed")
    if any(name.startswith("model.vlm_with_expert.vlm.") for name in trainable):
        raise RuntimeError("Refusing to train VLM parameters in this frozen-VLM protocol")
    return trainable


def _verify_base_checkpoint_loaded(policy, model_path: Path, torch) -> None:
    """Fail if the local base file lacks, or was not copied into, core policy tensors."""
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - required by LeRobot itself
        raise RuntimeError("safetensors is required by lerobot==0.6.1") from error
    candidates = ("model.state_proj.weight", "model.action_out_proj.weight")
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for key in candidates:
            if key not in keys:
                raise RuntimeError(f"Local base checkpoint is missing required tensor {key}")
            loaded = policy.state_dict()[key].detach().cpu()
            if not torch.equal(loaded, checkpoint.get_tensor(key)):
                raise RuntimeError(f"Base checkpoint tensor {key} was not loaded into the policy")


def _validate_policy_contract(config, policy, dataset_meta, camera_keys: list[str]) -> None:
    """Check the v0.6.1 factory-derived shapes before any optimizer is created."""
    expected_state = tuple(dataset_meta.features["observation.state"]["shape"])
    expected_action = tuple(dataset_meta.features["action"]["shape"])
    if set(config.image_features) != set(camera_keys):
        raise RuntimeError(f"Factory did not derive the two dataset camera features: {config.image_features}")
    if config.robot_state_feature is None or tuple(config.robot_state_feature.shape) != expected_state:
        raise RuntimeError("Factory did not derive observation.state from the dataset metadata")
    if config.action_feature is None or tuple(config.action_feature.shape) != expected_action:
        raise RuntimeError("Factory did not derive action from the dataset metadata")
    if config.chunk_size <= 0 or config.num_steps <= 0:
        raise RuntimeError(f"Invalid SmolVLA inference configuration: chunk={config.chunk_size}, steps={config.num_steps}")
    if policy.config is not config or config.device != "cuda":
        raise RuntimeError("Policy/preprocessor device contract drifted from the CUDA-only experiment")


def _validate_preprocessed_batch(batch: dict[str, Any], config, camera_keys: list[str], torch) -> None:
    """Assert the upstream preprocessor emitted the tensors SmolVLA expects."""
    required = [*camera_keys, "observation.state", "action", "observation.language.tokens", "observation.language.attention_mask"]
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"LeRobot preprocessor omitted required SmolVLA inputs: {missing}")
    if batch["observation.state"].ndim != 2 or batch["observation.state"].shape[-1] != config.robot_state_feature.shape[0]:
        raise RuntimeError(f"Unexpected state shape after preprocessing: {tuple(batch['observation.state'].shape)}")
    if batch["action"].ndim != 3 or tuple(batch["action"].shape[1:]) != (config.chunk_size, config.action_feature.shape[0]):
        raise RuntimeError(f"Unexpected action-chunk shape after preprocessing: {tuple(batch['action'].shape)}")
    for key in camera_keys:
        if batch[key].ndim != 4 or batch[key].shape[1] != 3:
            raise RuntimeError(f"Unexpected image tensor shape for {key}: {tuple(batch[key].shape)}")
    tensor_keys = [key for key, value in batch.items() if isinstance(value, torch.Tensor)]
    wrong_device = [key for key in tensor_keys if batch[key].device.type != "cuda"]
    if wrong_device:
        raise RuntimeError(f"LeRobot preprocessor did not move tensors to CUDA: {wrong_device}")


def _build_policy(PreTrainedConfig, make_policy, make_pre_post_processors, args, dataset_meta, train_stats, torch):
    config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    if config.type != "smolvla":
        raise RuntimeError(f"Expected a SmolVLA checkpoint, received policy type {config.type!r}")
    # The base checkpoint's feature names need to be inferred from this dataset, not
    # copied from a previous robot's camera namespace.
    config.input_features = {}
    config.output_features = {}
    config.pretrained_path = Path(args.model_path)
    config.pretrained_revision = None
    config.device = args.device
    config.vlm_model_name = str(args.vlm_path)
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.load_vlm_weights = True
    config.use_cache = True
    policy = make_policy(config, ds_meta=dataset_meta)
    _validate_policy_contract(config, policy, dataset_meta, list(dataset_meta.camera_keys))
    _verify_base_checkpoint_loaded(policy, args.model_path, torch)
    trainable_names = _freeze_for_action_expert_and_state_projection(policy)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=train_stats)
    policy.to(torch.device(args.device))
    return policy, preprocessor, postprocessor, config, trainable_names


def _capture_rng(torch) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict[str, Any], torch) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _trainable_state(policy) -> dict[str, Any]:
    trainable_names = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in policy.state_dict().items() if name in trainable_names}


def _load_trainable_state(policy, state: dict[str, Any]) -> None:
    expected = set(policy.state_dict()) - set(state)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if set(missing) != expected or unexpected:
        raise RuntimeError(
            "Trainable checkpoint does not match the pinned base policy; "
            f"missing={len(missing)}, unexpected={unexpected}"
        )


def _checkpoint_dir(output_dir: Path, label: str) -> Path:
    return output_dir / "checkpoints" / label


def _save_checkpoint(
    output_dir: Path,
    label: str,
    policy,
    optimizer,
    step: int,
    best_dev_loss: float,
    stats_hash: str,
    run_identity: str,
    torch,
) -> Path:
    directory = _checkpoint_dir(output_dir, label)
    directory.mkdir(parents=True, exist_ok=True)
    trainable_state_path = directory / "trainable_state.pt"
    torch.save(_trainable_state(policy), trainable_state_path)
    trainable_state_sha256 = _sha256_file(trainable_state_path)
    torch.save(
        {
            "step": step,
            "best_dev_loss": best_dev_loss,
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(torch),
            "train_stats_sha256": stats_hash,
            "run_identity": run_identity,
        },
        directory / "training_state.pt",
    )
    _write_json(
        directory / "checkpoint.json",
        {
            "format": "smolvla-frozen-vlm-trainable-state-v1",
            "step": step,
            "best_dev_loss": best_dev_loss,
            "base_model": MODEL_REPO_ID,
            "base_model_revision": "recorded in run_manifest.json",
            "train_stats_sha256": stats_hash,
            "run_identity": run_identity,
            "trainable_state_sha256": trainable_state_sha256,
            "reload": "Rebuild the pinned base policy, then load trainable_state.pt before inference.",
        },
    )
    _write_json(
        output_dir / "latest_checkpoint.json",
        {"label": label, "step": step, "run_identity": run_identity, "trainable_state_sha256": trainable_state_sha256},
    )
    return directory


def _prune_periodic_checkpoints(output_dir: Path, keep: int) -> None:
    periodic = sorted(path for path in (output_dir / "checkpoints").glob("step_*") if path.is_dir())
    for path in periodic[:-keep]:
        shutil.rmtree(path)


def _load_checkpoint(path: Path, policy, optimizer, stats_hash: str, run_identity: str, torch) -> tuple[int, float]:
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state["train_stats_sha256"] != stats_hash:
        raise RuntimeError("Refusing to resume with different train-only normalization statistics")
    if state.get("run_identity") != run_identity:
        raise RuntimeError("Refusing to resume checkpoint with a different immutable run identity")
    checkpoint_path = path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    trainable_path = path / "trainable_state.pt"
    if checkpoint.get("run_identity") != run_identity or checkpoint.get("trainable_state_sha256") != _sha256_file(trainable_path):
        raise RuntimeError("Resume checkpoint provenance or trainable-state hash does not match")
    trainable = torch.load(trainable_path, map_location="cpu", weights_only=True)
    _load_trainable_state(policy, trainable)
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"], torch)
    return int(state["step"]), float(state["best_dev_loss"])


def _eval_loss(loader, policy, preprocessor, camera_keys, args, torch) -> float:
    policy.eval()
    total = 0.0
    batches = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_model_batch(raw_batch, camera_keys, preprocessor, torch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss, _ = policy.forward(batch)
            total += float(loss.detach().cpu())
            batches += 1
    if batches == 0:
        raise RuntimeError("Development split produced no batches")
    return total / batches


def _tensor_digest(tensor, torch) -> dict[str, Any]:
    copied = tensor.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(copied.dtype), "shape": list(copied.shape)}, sort_keys=True).encode("utf-8")
    return {
        "dtype": str(copied.dtype),
        "shape": list(copied.shape),
        "sha256": _sha256_bytes(header + copied.numpy().tobytes()),
    }


def _dataset_native_action(postprocessor, predicted):
    action = postprocessor(predicted)
    if not hasattr(action, "detach"):
        raise TypeError(f"Expected tensor action after LeRobot postprocessor, got {type(action)!r}")
    return action


def _test_base_and_finetuned(
    loader,
    base_policy,
    fine_policy,
    preprocessor,
    postprocessor,
    camera_keys,
    action_names: list[str],
    action_std,
    args,
    torch,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One fixed held-out pass that scores both checkpoints and retains ten windows."""
    base_policy.eval()
    fine_policy.eval()
    dimension = len(action_names)
    if action_std.numel() != dimension:
        raise RuntimeError("Train-only action standard deviation does not match dataset action names")
    total_abs = {name: torch.zeros(dimension, dtype=torch.float64) for name in ("base", "finetuned")}
    total_sq = {name: torch.zeros(dimension, dtype=torch.float64) for name in ("base", "finetuned")}
    norm_abs = {name: torch.zeros(dimension, dtype=torch.float64) for name in ("base", "finetuned")}
    norm_sq = {name: torch.zeros(dimension, dtype=torch.float64) for name in ("base", "finetuned")}
    total_count = torch.zeros(dimension, dtype=torch.int64)
    windows: list[dict[str, Any]] = []
    positions = {round(index * (len(loader.dataset) - 1) / max(args.shadow_windows - 1, 1)) for index in range(args.shadow_windows)}
    offset = 0
    with torch.inference_mode():
        for raw_batch in loader:
            raw_target = raw_batch["action"].detach().cpu().clone()
            raw_pad = raw_batch.get("action_is_pad")
            raw_images = {key: raw_batch[key].detach().cpu().clone() for key in camera_keys}
            raw_state = raw_batch["observation.state"].detach().cpu().clone()
            task_values = raw_batch["task"]
            task = [task_values] if isinstance(task_values, str) else list(task_values)
            prepared = _to_model_batch(raw_batch, camera_keys, preprocessor, torch)
            batch_size = raw_target.shape[0]
            generator = torch.Generator(device=args.device)
            generator.manual_seed(args.seed + 1_000_003 + offset)
            noise = torch.randn(
                (batch_size, fine_policy.config.chunk_size, fine_policy.config.max_action_dim),
                generator=generator,
                dtype=torch.float32,
                device=args.device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                base_pred = _dataset_native_action(postprocessor, base_policy.predict_action_chunk(prepared, noise=noise.clone()))
                fine_pred = _dataset_native_action(postprocessor, fine_policy.predict_action_chunk(prepared, noise=noise.clone()))
            base_pred = base_pred.detach().cpu().float()
            fine_pred = fine_pred.detach().cpu().float()
            target = raw_target.float()
            valid = torch.ones_like(target, dtype=torch.bool)
            if raw_pad is not None:
                valid &= ~raw_pad.detach().cpu().bool().unsqueeze(-1)
            for name, prediction in (("base", base_pred), ("finetuned", fine_pred)):
                error = prediction - target
                total_abs[name] += (error.abs() * valid).sum(dim=(0, 1), dtype=torch.float64)
                total_sq[name] += (error.square() * valid).sum(dim=(0, 1), dtype=torch.float64)
                normalized_error = error / action_std.reshape(1, 1, -1)
                norm_abs[name] += (normalized_error.abs() * valid).sum(dim=(0, 1), dtype=torch.float64)
                norm_sq[name] += (normalized_error.square() * valid).sum(dim=(0, 1), dtype=torch.float64)
            total_count += valid.sum(dim=(0, 1), dtype=torch.int64)
            for relative in range(batch_size):
                if offset + relative not in positions:
                    continue
                windows.append(
                    {
                        "test_dataset_position": offset + relative,
                        "episode_index": int(raw_batch["episode_index"][relative]),
                        "frame_index": int(raw_batch["frame_index"][relative]),
                        "task": task[relative],
                        "raw_images_uint8": {key: value[relative] for key, value in raw_images.items()},
                        "raw_state": raw_state[relative],
                        "dataset_native_action_chunk": target[relative],
                        "action_is_pad": None if raw_pad is None else raw_pad.detach().cpu()[relative],
                        "base_dataset_native_action_chunk": base_pred[relative],
                        "finetuned_dataset_native_action_chunk": fine_pred[relative],
                        "fixed_noise": noise.detach().cpu()[relative],
                    }
                )
            offset += batch_size
    if bool((total_count == 0).any()):
        raise RuntimeError("Held-out test has no non-padded action values")
    if len(windows) != args.shadow_windows:
        raise RuntimeError(f"Expected {args.shadow_windows} shadow windows, retained {len(windows)}")
    metrics: dict[str, Any] = {}
    for name in ("base", "finetuned"):
        raw_mae = total_abs[name] / total_count
        raw_rmse = (total_sq[name] / total_count).sqrt()
        scaled_mae = norm_abs[name] / total_count
        scaled_rmse = (norm_sq[name] / total_count).sqrt()
        metrics[name] = {
            "dataset_native_values_units": "undeclared_by_dataset_metadata",
            "dataset_native_per_joint": [
                {"name": action_names[index], "mae": float(raw_mae[index]), "rmse": float(raw_rmse[index])}
                for index in range(dimension)
            ],
            "dataset_native_overall": {"mae": float(total_abs[name].sum() / total_count.sum()), "rmse": float((total_sq[name].sum() / total_count.sum()).sqrt())},
            "train_action_std": {
                "fit_split": "train",
                "values": [float(value) for value in action_std],
                "units": "same dataset-native values; physical units undeclared",
            },
            "train_action_std_normalized_per_joint": [
                {"name": action_names[index], "mae": float(scaled_mae[index]), "rmse": float(scaled_rmse[index])}
                for index in range(dimension)
            ],
            "train_action_std_normalized_overall": {"mae": float(norm_abs[name].sum() / total_count.sum()), "rmse": float((norm_sq[name].sum() / total_count.sum()).sqrt())},
        }
    return metrics, windows


def _write_shadow_artifacts(
    output_dir: Path, windows: list[dict[str, Any]], run_identity: str, best_trainable_state_sha256: str, torch
) -> dict[str, Any]:
    tensor_path = output_dir / "shadow_windows.pt"
    if tensor_path.exists() or (output_dir / "shadow_windows_manifest.json").exists():
        raise RuntimeError("Shadow artifacts already exist; refuse to overwrite a completed held-out test pass")
    torch.save(
        {
            "schema": "smolvla-shadow-v2",
            "run_identity": run_identity,
            "best_trainable_state_sha256": best_trainable_state_sha256,
            "action_values": "dataset-native; physical units undeclared by metadata",
            "windows": windows,
        },
        tensor_path,
    )
    manifest_windows = []
    for window in windows:
        manifest_windows.append(
            {
                key: (_tensor_digest(value, torch) if hasattr(value, "detach") else value)
                for key, value in window.items()
                if key != "raw_images_uint8"
            }
            | {"raw_images_uint8": {key: _tensor_digest(value, torch) for key, value in window["raw_images_uint8"].items()}}
        )
    manifest = {
        "schema": "smolvla-shadow-manifest-v2",
        "run_identity": run_identity,
        "best_trainable_state_sha256": best_trainable_state_sha256,
        "claim_boundary": "Offline dataset-native action chunks only; physical units are undeclared by metadata. No robot was commanded.",
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": _sha256_file(tensor_path),
        "windows": manifest_windows,
    }
    _write_json(output_dir / "shadow_windows_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Predownloaded local v3.0 dataset directory.")
    parser.add_argument("--model-path", type=Path, required=True, help="Predownloaded local smolvla_base snapshot.")
    parser.add_argument("--vlm-path", type=Path, required=True, help="Predownloaded local SmolVLM2 backbone snapshot.")
    parser.add_argument("--dataset-manifest", type=Path, help="Verified dataset download_manifest.json (default: dataset root).")
    parser.add_argument("--model-manifest", type=Path, help="Verified base-model download_manifest.json (default: model root).")
    parser.add_argument("--vlm-manifest", type=Path, help="Verified VLM download_manifest.json (default: VLM root).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local run directory (keep outside version control).")
    parser.add_argument("--dataset-repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION, help="Immutable dataset commit required by this protocol.")
    parser.add_argument("--model-revision", required=True, help="Resolved immutable commit SHA of lerobot/smolvla_base.")
    parser.add_argument("--vlm-revision", required=True, help="Resolved immutable commit SHA of the SmolVLM backbone.")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--dev-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--infer-batch-size", type=int, default=1, help="Use 1 by default to fit both models on 24 GB.")
    parser.add_argument("--shadow-windows", type=int, default=10)
    parser.add_argument("--resume", type=Path, help="A checkpoint directory written by this script.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dataset_repo_id != DATASET_REPO_ID:
        raise ValueError(f"This fixed protocol accepts only {DATASET_REPO_ID}, not {args.dataset_repo_id}")
    if args.dataset_revision != DATASET_REVISION:
        raise ValueError(f"This fixed protocol requires --dataset-revision {DATASET_REVISION}")
    _require_immutable_revision(args.dataset_revision, "dataset revision")
    _require_immutable_revision(args.model_revision, "model revision")
    _require_immutable_revision(args.vlm_revision, "VLM revision")
    if args.steps <= 0 or args.batch_size <= 0 or args.dev_every <= 0 or args.save_every <= 0:
        raise ValueError("steps, batch size, dev interval, and save interval must be positive")
    if args.seed != SPLIT_SEED:
        raise ValueError(f"This registered split/training protocol requires --seed {SPLIT_SEED}")
    if args.infer_batch_size <= 0:
        raise ValueError("infer batch size must be positive")
    if args.shadow_windows != 10:
        raise ValueError("The registered shadow protocol requires exactly 10 windows")
    if args.device != "cuda":
        raise ValueError("This registered run is CUDA-only; use a separate smoke workflow for CPU")
    _require_local_dir(args.dataset_root, "dataset", ("meta/info.json", "data", "videos"))
    _require_local_dir(args.model_path, "SmolVLA base", ("config.json", "model.safetensors"))
    _require_local_dir(args.vlm_path, "SmolVLM backbone", ("config.json",))
    download_provenance = {
        "dataset": _verify_download_manifest(
            args.dataset_root, args.dataset_manifest, DATASET_REPO_ID, args.dataset_revision, "dataset"
        ),
        "base_model": _verify_download_manifest(
            args.model_path, args.model_manifest, MODEL_REPO_ID, args.model_revision, "base model"
        ),
        "vlm_backbone": _verify_download_manifest(
            args.vlm_path, args.vlm_manifest, VLM_REPO_ID, args.vlm_revision, "VLM backbone"
        ),
    }
    if args.output_dir.exists():
        nonempty = any(args.output_dir.iterdir())
        if nonempty and not args.resume:
            raise RuntimeError("Output directory is nonempty; use a checkpoint from this run with --resume or choose a new directory")
        if args.resume and not nonempty:
            raise RuntimeError("Cannot resume into an empty output directory")
    elif args.resume:
        raise RuntimeError("Cannot resume because the output directory does not exist")
    else:
        args.output_dir.mkdir(parents=True)
    if args.resume:
        resume_root = (args.output_dir / "checkpoints").resolve()
        resume_path = args.resume.resolve()
        if resume_root not in resume_path.parents or not resume_path.is_dir():
            raise RuntimeError("--resume must name an existing checkpoint under this output directory")
    torch, PreTrainedConfig, LeRobotDataset, make_policy, make_pre_post_processors, lerobot_collate_fn = _require_dependencies()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires a CUDA device with bfloat16 support")
    _seed_everything(args.seed, torch)

    # Metadata-only construction reads locally and makes the fixed split before any
    # visual frames are decoded.
    all_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    if str(all_dataset.meta.info.codebase_version) != "v3.0":
        raise RuntimeError(f"Expected dataset meta/info v3.0, found {all_dataset.meta.info.codebase_version!r}")
    splits = _make_splits(all_dataset.meta.total_episodes)
    _persist_or_verify_splits(args.output_dir, splits)

    # The 30 training episodes alone determine normalizer statistics.
    train_metadata_dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=splits.train,
        revision=args.dataset_revision,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    train_stats = _train_only_stats(train_metadata_dataset, torch)
    serialized_stats = _serializable_stats(train_stats)
    stats_hash = _sha256_bytes(_json_bytes(serialized_stats))
    train_stats_payload = {"fit_split": "train", "sha256": stats_hash, "stats": serialized_stats}
    train_stats_path = args.output_dir / "train_only_stats.json"
    _write_or_verify_json(train_stats_path, train_stats_payload, "train-only normalization statistics")

    # Build after loading checkpoint config so action horizon comes from the actual
    # upstream checkpoint rather than a duplicated local constant.
    config_probe = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    if config_probe.type != "smolvla" or config_probe.chunk_size <= 0:
        raise RuntimeError("Local base checkpoint does not expose a valid SmolVLA action chunk configuration")
    train_dataset = _make_dataset(LeRobotDataset, args, splits.train, int(all_dataset.meta.fps), config_probe.chunk_size)
    dev_dataset = _make_dataset(LeRobotDataset, args, splits.dev, int(all_dataset.meta.fps), config_probe.chunk_size)
    test_dataset = _make_dataset(LeRobotDataset, args, splits.test, int(all_dataset.meta.fps), config_probe.chunk_size)
    camera_keys = list(train_dataset.meta.camera_keys)
    if len(camera_keys) != 2:
        raise RuntimeError(f"Expected exactly two camera streams, found {camera_keys}")
    if "observation.state" not in train_dataset.features or "action" not in train_dataset.features:
        raise RuntimeError("Dataset lacks required observation.state/action features")
    action_metadata = train_dataset.meta.features["action"]
    action_names = list(action_metadata.get("names", []))
    if len(action_names) != 6 or tuple(action_metadata.get("shape", ())) != (6,):
        raise RuntimeError(f"Expected named 6D dataset action feature, found {action_metadata}")
    if action_metadata.get("dtype") != "float32":
        raise RuntimeError(f"Expected float32 dataset-native action values, found {action_metadata.get('dtype')!r}")

    identity_bindings = {
        "schema": "smolvla-run-identity-v1",
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "downloads": download_provenance,
        "episode_splits_sha256": _sha256_file(args.output_dir / "episode_splits.json"),
        "train_only_stats_sha256": _sha256_file(train_stats_path),
        "dataset_revision": args.dataset_revision,
        "model_revision": args.model_revision,
        "vlm_revision": args.vlm_revision,
    }
    run_identity = _sha256_value(identity_bindings)
    _write_or_verify_json(
        args.output_dir / "run_identity.json",
        {"schema": "smolvla-run-identity-v1", "run_identity": run_identity, "bindings": identity_bindings},
        "run identity",
    )

    run_manifest = {
        "schema": "smolvla-offline-run-v1",
        "claim_boundary": "Offline VLA action-chunk experiment. No robot closed-loop, safety, latency, or success claim.",
        "source_commit": _source_commit(),
        "run_identity": run_identity,
        "run_identity_bindings": "run_identity.json",
        "lerobot_version": LEROBOT_VERSION,
        "dataset": {"repo_id": args.dataset_repo_id, "requested_revision": args.dataset_revision, "resolved_revision": train_dataset.revision, "meta_codebase_version": train_dataset.meta.info.codebase_version, "fps": train_dataset.meta.fps, "camera_keys": camera_keys, "action_names": action_names, "action_values": "dataset-native float32; physical units undeclared by metadata"},
        "base_model": {"repo_id": MODEL_REPO_ID, "revision": args.model_revision},
        "vlm_backbone": {"repo_id": VLM_REPO_ID, "revision": args.vlm_revision},
        "splits": splits.as_dict(),
        "w1_split_hash": W1_SPLIT_HASH,
        "normalization": {"fit_split": "train", "stats_sha256": stats_hash},
        "download_provenance": download_provenance,
        "training": {"steps": args.steps, "batch_size": args.batch_size, "bf16": args.bf16, "gradient_clip": args.gradient_clip, "dev_every": args.dev_every, "save_every": args.save_every, "wandb": False, "hub_upload": False, "frozen": "VLM; trainable action expert, state projection, and action projections"},
        "calibration": {"split": "cal", "status": "W1 placeholder; untouched during this W0 behavior-cloning run", "episode_count": 10},
    }
    _write_or_verify_json(args.output_dir / "run_manifest.json", run_manifest, "run manifest")

    policy, preprocessor, _postprocessor, _config, trainable_names = _build_policy(
        PreTrainedConfig, make_policy, make_pre_post_processors, args, train_dataset.meta, train_stats, torch
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    start_step, best_dev_loss = 0, float("inf")
    if args.resume:
        start_step, best_dev_loss = _load_checkpoint(args.resume, policy, optimizer, stats_hash, run_identity, torch)
    if start_step >= args.steps:
        raise ValueError(f"Resume checkpoint is already at step {start_step}, not below requested {args.steps}")
    _write_or_verify_json(
        args.output_dir / "trainable_parameters.json",
        {"count": len(trainable_names), "names": trainable_names, "run_identity": run_identity},
        "trainable parameter manifest",
    )

    train_sampler = _ResumableTrainBatchSampler(len(train_dataset), args.batch_size, args.seed, start_step, args.steps, torch)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_for(train_dataset, lerobot_collate_fn),
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_for(dev_dataset, lerobot_collate_fn),
    )
    events_path = args.output_dir / "train_events.jsonl"
    last_dev_step = start_step
    policy.train()
    with events_path.open("a", encoding="utf-8") as events:
        for step, raw_batch in enumerate(train_loader, start=start_step + 1):
            started = time.perf_counter()
            batch = _to_model_batch(raw_batch, camera_keys, preprocessor, torch)
            if step == start_step + 1:
                _validate_preprocessed_batch(batch, policy.config, camera_keys, torch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss, loss_details = policy.forward(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            event: dict[str, Any] = {
                "event": "train",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "seconds": time.perf_counter() - started,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
            if loss_details:
                event["policy_metrics"] = loss_details
            if step % 25 == 0 or step == args.steps:
                events.write(json.dumps(event, sort_keys=True) + "\n")
                events.flush()
            if step % args.dev_every == 0 or step == args.steps:
                dev_loss = _eval_loss(dev_loader, policy, preprocessor, camera_keys, args, torch)
                last_dev_step = step
                dev_event = {"event": "dev", "step": step, "loss": dev_loss}
                events.write(json.dumps(dev_event, sort_keys=True) + "\n")
                events.flush()
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    _save_checkpoint(args.output_dir, "best", policy, optimizer, step, best_dev_loss, stats_hash, run_identity, torch)
                policy.train()
            if step % args.save_every == 0 or step == args.steps:
                _save_checkpoint(args.output_dir, f"step_{step:06d}", policy, optimizer, step, best_dev_loss, stats_hash, run_identity, torch)
                _prune_periodic_checkpoints(args.output_dir, args.keep_last_checkpoints)

    if last_dev_step != args.steps:
        # This only happens for a future CLI extension that changes loop behavior;
        # keep the test gate explicit rather than accepting an unselected checkpoint.
        dev_loss = _eval_loss(dev_loader, policy, preprocessor, camera_keys, args, torch)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            _save_checkpoint(args.output_dir, "best", policy, optimizer, args.steps, best_dev_loss, stats_hash, run_identity, torch)

    # Prove a fresh local reload before the one held-out test pass.  The base is
    # rebuilt from the immutable local snapshot; the finetuned state then overlays
    # only the declared trainable tensors.
    del policy
    torch.cuda.empty_cache()
    base_policy, test_preprocessor, test_postprocessor, _base_config, _ = _build_policy(
        PreTrainedConfig, make_policy, make_pre_post_processors, args, train_dataset.meta, train_stats, torch
    )
    fine_policy, _fine_preprocessor, _fine_postprocessor, _fine_config, _ = _build_policy(
        PreTrainedConfig, make_policy, make_pre_post_processors, args, train_dataset.meta, train_stats, torch
    )
    best_state_path = _checkpoint_dir(args.output_dir, "best") / "trainable_state.pt"
    best_checkpoint = json.loads((_checkpoint_dir(args.output_dir, "best") / "checkpoint.json").read_text(encoding="utf-8"))
    best_trainable_state_sha256 = _sha256_file(best_state_path)
    if best_checkpoint.get("run_identity") != run_identity or best_checkpoint.get("trainable_state_sha256") != best_trainable_state_sha256:
        raise RuntimeError("Best checkpoint provenance or trainable-state hash does not match this run")
    best_state = torch.load(best_state_path, map_location="cpu", weights_only=True)
    _load_trainable_state(fine_policy, best_state)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_for(test_dataset, lerobot_collate_fn),
    )
    metrics, shadow_windows = _test_base_and_finetuned(
        test_loader,
        base_policy,
        fine_policy,
        test_preprocessor,
        test_postprocessor,
        camera_keys,
        action_names,
        train_stats["action"]["std"].detach().cpu(),
        args,
        torch,
    )
    shadow_manifest = _write_shadow_artifacts(
        args.output_dir, shadow_windows, run_identity, best_trainable_state_sha256, torch
    )
    result = {
        "schema": "smolvla-offline-result-v1",
        "run_identity": run_identity,
        "best_trainable_state_sha256": best_trainable_state_sha256,
        "selection": {"split": "dev", "best_dev_loss": best_dev_loss, "checkpoint": "checkpoints/best"},
        "test": {"split": "test", "episode_count": 5, "single_pass": True, "metrics": metrics},
        "shadow": {"window_count": args.shadow_windows, "manifest_sha256": _sha256_bytes(_json_bytes(shadow_manifest))},
        "calibration": {"split": "cal", "episode_count": 10, "status": "Reserved for W1; not fitted or reported in this W0 run."},
        "claim_boundary": "Offline action-chunk reconstruction on held-out recorded windows. No robot was commanded.",
    }
    _write_or_verify_json(args.output_dir / "result.json", result, "result")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume from the latest checkpoint recorded in latest_checkpoint.json.", file=sys.stderr)
        raise SystemExit(130)
