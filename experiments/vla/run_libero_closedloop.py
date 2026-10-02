#!/usr/bin/env python3
"""Run the frozen SmolVLA LIBERO-Spatial closed-loop evaluation.

This runner deliberately uses LeRobot's official LIBERO environment, policy
factory, processors, and ``eval_one`` rollout loop.  The only wrappers are
observers: they record the raw predicted chunk, the selected normalized
action, the official postprocessor output, and the exact numpy action passed
to ``env.step``.  They never modify an action.

``--mode preflight`` is a one-task, one-initial-state compatibility check and
is excluded from final metrics.  ``--mode final`` refuses any task/episode
override and executes the frozen 10 x 10 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_identity(root: Path, scope: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{file_hash}\0{size}\n".encode())
        file_count += 1
        total_bytes += size
    return {
        "root": str(root.resolve()),
        "scope": scope,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return value


def _array_digest(value: Any) -> str:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


class JsonlSink:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class TraceRecorder:
    """Observe the native LeRobot action path without changing it."""

    def __init__(self, sink: JsonlSink, runtime_contract: dict[str, Any]):
        self.sink = sink
        self.runtime_contract = runtime_contract
        self.task_id = -1
        self.initial_state_index = -1
        self.episode_seed: int | None = None
        self.step = 0
        self.chunk_id = -1
        self.action_in_chunk = 0
        self.pending_postprocessed: Any | None = None
        self.max_passthrough_abs_diff = 0.0
        self.passthrough_comparisons = 0
        self.raw_chunks = 0
        self.actions = 0
        self.instruction = ""
        self.latest_input_context: dict[str, Any] | None = None

    def set_task(self, task_id: int, instruction: str) -> None:
        self.task_id = task_id
        self.instruction = instruction
        self.step = 0
        self.chunk_id = -1
        self.action_in_chunk = 0
        self.pending_postprocessed = None
        self.latest_input_context = None

    def plan_episode(self, initial_state_index: int, seed: int) -> None:
        self.initial_state_index = initial_state_index
        self.episode_seed = seed
        self.step = 0
        self.chunk_id = -1
        self.action_in_chunk = 0
        self.pending_postprocessed = None
        self.latest_input_context = None
        self.sink.write(
            {
                "event": "episode_plan",
                "task_id": self.task_id,
                "initial_state_index": initial_state_index,
                "seed": seed,
            }
        )

    def capture_input_context(self, observation: dict[str, Any]) -> None:
        state = observation.get("observation.state")
        if state is None or list(state.shape[1:]) != self.runtime_contract["state_shape"]:
            found = None if state is None else list(state.shape[1:])
            raise RuntimeError(
                f"runtime state shape mismatch: expected {self.runtime_contract['state_shape']}, found {found}"
            )
        camera_keys = sorted(
            key for key in observation if key.startswith("observation.images.")
        )
        if camera_keys != sorted(self.runtime_contract["camera_keys"]):
            raise RuntimeError(
                "runtime camera keys mismatch: "
                f"expected {sorted(self.runtime_contract['camera_keys'])}, found {camera_keys}"
            )
        cameras = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
            }
            for key, value in sorted(observation.items())
            if key.startswith("observation.images.")
        }
        self.latest_input_context = {
            "instruction": self.instruction,
            "state": None
            if state is None
            else {
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "sha256": _array_digest(state),
                "values": _jsonable(state),
            },
            "camera_frames": cameras,
            "camera_pixels_retained": False,
        }

    def on_reset(self, seeds: Any, actual_initial_state_index: int) -> None:
        if actual_initial_state_index != self.initial_state_index:
            raise RuntimeError(
                "LIBERO initial-state mismatch: "
                f"planned {self.initial_state_index}, reset used {actual_initial_state_index}"
            )
        seed_values = _jsonable(seeds)
        if not isinstance(seed_values, list):
            seed_values = [seed_values]
        if len(seed_values) != 1 or int(seed_values[0]) != self.episode_seed:
            raise RuntimeError(
                f"LIBERO seed mismatch: planned {self.episode_seed}, reset used {seed_values}"
            )
        self.step = 0
        self.chunk_id = -1
        self.action_in_chunk = 0
        self.pending_postprocessed = None
        self.sink.write(
            {
                "event": "episode_reset",
                "task_id": self.task_id,
                "initial_state_index": actual_initial_state_index,
                "seed": seed_values[0],
                "planned_seed": self.episode_seed,
                "hard_reset": True,
            }
        )

    def on_chunk(self, chunk: Any) -> None:
        self.chunk_id += 1
        self.action_in_chunk = 0
        self.raw_chunks += 1
        self.sink.write(
            {
                "event": "raw_policy_chunk",
                "task_id": self.task_id,
                "initial_state_index": self.initial_state_index,
                "step": self.step,
                "chunk_id": self.chunk_id,
                "shape": list(chunk.shape),
                "dtype": str(chunk.dtype),
                "sha256": _array_digest(chunk),
                "values": chunk,
                "input_context": self.latest_input_context,
            }
        )

    def on_selected(self, action: Any) -> None:
        self.sink.write(
            {
                "event": "selected_normalized_action",
                "task_id": self.task_id,
                "initial_state_index": self.initial_state_index,
                "step": self.step,
                "chunk_id": self.chunk_id,
                "action_index_in_chunk": self.action_in_chunk,
                "shape": list(action.shape),
                "sha256": _array_digest(action),
                "values": action,
            }
        )

    def on_postprocessed(self, action: Any) -> None:
        if hasattr(action, "detach"):
            self.pending_postprocessed = action.detach().to("cpu").numpy().copy()
        else:
            import numpy as np

            self.pending_postprocessed = np.asarray(action).copy()
        self.sink.write(
            {
                "event": "official_postprocessed_action",
                "task_id": self.task_id,
                "initial_state_index": self.initial_state_index,
                "step": self.step,
                "chunk_id": self.chunk_id,
                "action_index_in_chunk": self.action_in_chunk,
                "shape": list(action.shape),
                "sha256": _array_digest(action),
                "values": action,
            }
        )

    def on_env_step(self, action: Any) -> None:
        import numpy as np

        final_action = np.asarray(action)
        if list(final_action.shape[1:]) != self.runtime_contract["action_shape"]:
            raise RuntimeError(
                "runtime action shape mismatch: "
                f"expected {self.runtime_contract['action_shape']}, found {list(final_action.shape[1:])}"
            )
        if self.pending_postprocessed is None:
            raise RuntimeError("env.step observed before the official postprocessor output")
        if self.pending_postprocessed.shape != final_action.shape:
            raise RuntimeError(
                f"postprocessor/env.step shape mismatch: {self.pending_postprocessed.shape} != {final_action.shape}"
            )
        difference = float(np.max(np.abs(self.pending_postprocessed - final_action)))
        self.max_passthrough_abs_diff = max(self.max_passthrough_abs_diff, difference)
        self.passthrough_comparisons += 1
        if difference != 0.0:
            raise RuntimeError(f"shadow gate changed an action; max_abs_diff={difference}")
        self.sink.write(
            {
                "event": "env_step_action",
                "task_id": self.task_id,
                "initial_state_index": self.initial_state_index,
                "step": self.step,
                "chunk_id": self.chunk_id,
                "action_index_in_chunk": self.action_in_chunk,
                "shape": list(final_action.shape),
                "sha256": _array_digest(final_action),
                "values": final_action,
                "shadow_gate": {
                    "mode": "observe_only",
                    "decision": "PASSTHROUGH",
                    "intervention": False,
                },
                "postprocess_to_env_step_max_abs_diff": difference,
            }
        )
        self.pending_postprocessed = None
        self.actions += 1
        self.step += 1
        self.action_in_chunk += 1


class TracingProcessor:
    def __init__(self, delegate: Callable[[Any], Any], recorder: TraceRecorder):
        self.delegate = delegate
        self.recorder = recorder

    def __call__(self, action: Any) -> Any:
        output = self.delegate(action)
        self.recorder.on_postprocessed(output)
        return output


class TracingInputProcessor:
    def __init__(self, delegate: Callable[[Any], Any], recorder: TraceRecorder):
        self.delegate = delegate
        self.recorder = recorder

    def __call__(self, observation: dict[str, Any]) -> Any:
        self.recorder.capture_input_context(observation)
        return self.delegate(observation)


def _wrap_policy(policy: Any, recorder: TraceRecorder) -> None:
    original_get_chunk = policy._get_action_chunk
    original_select = policy.select_action

    def get_action_chunk(batch: Any, *args: Any, **kwargs: Any) -> Any:
        chunk = original_get_chunk(batch, *args, **kwargs)
        recorder.on_chunk(chunk)
        return chunk

    def select_action(batch: Any, **kwargs: Any) -> Any:
        action = original_select(batch, **kwargs)
        recorder.on_selected(action)
        return action

    policy._get_action_chunk = get_action_chunk
    policy.select_action = select_action


def _wrap_env(env: Any, recorder: TraceRecorder) -> None:
    original_reset = env.reset
    original_step = env.step

    def reset(*args: Any, **kwargs: Any) -> Any:
        seeds = kwargs.get("seed", args[0] if args else None)
        output = original_reset(*args, **kwargs)
        if not hasattr(env, "envs") or len(env.envs) != 1:
            raise RuntimeError("fixed-state audit requires one synchronous LIBERO environment")
        underlying = env.envs[0]
        actual_initial_state_index = underlying.init_state_id - underlying._reset_stride
        recorder.on_reset(seeds, actual_initial_state_index)
        return output

    def step(action: Any) -> Any:
        recorder.on_env_step(action)
        return original_step(action)

    env.reset = reset
    env.step = step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "final"), required=True)
    parser.add_argument("--preflight-task-id", type=int, default=0)
    parser.add_argument("--preflight-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_script_sha = protocol["implementation"]["script_sha256"]
    actual_script_sha = _sha256(Path(__file__).resolve())
    if expected_script_sha != actual_script_sha:
        raise RuntimeError(f"runner SHA mismatch: protocol={expected_script_sha}, actual={actual_script_sha}")
    if protocol["state"] != "frozen_before_final_run":
        raise RuntimeError("protocol must be frozen before any final run")
    if args.mode == "preflight" and not 0 <= args.preflight_task_id < 10:
        raise ValueError("preflight task id must be in [0, 9]")
    if args.mode == "preflight" and args.preflight_manifest is not None:
        raise ValueError("preflight mode cannot consume another preflight manifest")
    preflight_receipt: dict[str, Any] | None = None
    if args.mode == "final":
        if args.preflight_manifest is None or not args.preflight_manifest.is_file():
            raise FileNotFoundError("final mode requires --preflight-manifest from the frozen runner")
        preflight = json.loads(args.preflight_manifest.read_text(encoding="utf-8"))
        preflight_task = preflight.get("results", {}).get("tasks", {}).get("0", {})
        expected_preflight = protocol["preflight"]
        checks = {
            "mode": preflight.get("mode") == "preflight",
            "status": preflight.get("status") == "complete",
            "excluded": preflight.get("included_in_final_metrics") is False,
            "script": preflight.get("script_sha256") == actual_script_sha,
            "protocol": preflight.get("protocol_sha256") == _sha256(args.protocol),
            "task": list(preflight.get("results", {}).get("tasks", {})) == [
                str(expected_preflight["task_id"])
            ],
            "state": preflight_task.get("initial_state_indices")
            == [expected_preflight["initial_state_index"]],
            "seed": preflight_task.get("episode_seeds") == [expected_preflight["seed"]],
            "episodes": preflight_task.get("episodes") == expected_preflight["episodes"],
            "completed": preflight_task.get("completed_episodes")
            == expected_preflight["episodes"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"preflight evidence mismatch: {checks}")
        preflight_receipt = {
            "path": str(args.preflight_manifest.resolve()),
            "sha256": _sha256(args.preflight_manifest),
            "checks": checks,
        }

    for directory, label in ((args.checkpoint, "checkpoint"), (args.backbone, "backbone")):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {directory}")

    required_checkpoint_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    for name in required_checkpoint_files:
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(args.checkpoint / name)
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        if not (args.backbone / name).is_file():
            raise FileNotFoundError(args.backbone / name)
    checkpoint_hashes = {
        "config.json": _sha256(args.checkpoint / "config.json"),
        "model.safetensors": _sha256(args.checkpoint / "model.safetensors"),
        "policy_preprocessor.json": _sha256(args.checkpoint / "policy_preprocessor.json"),
        "policy_postprocessor.json": _sha256(args.checkpoint / "policy_postprocessor.json"),
        "policy_preprocessor_step_5_normalizer_processor.safetensors": _sha256(
            args.checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        ),
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": _sha256(
            args.checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        ),
    }
    for name, expected in protocol["checkpoint"]["sha256"].items():
        if checkpoint_hashes[name] != expected:
            raise RuntimeError(
                f"checkpoint digest mismatch for {name}: expected {expected}, found {checkpoint_hashes[name]}"
            )
    backbone_hashes = {
        "config.json": _sha256(args.backbone / "config.json"),
        "model.safetensors": _sha256(args.backbone / "model.safetensors"),
        "tokenizer.json": _sha256(args.backbone / "tokenizer.json"),
        "tokenizer_config.json": _sha256(args.backbone / "tokenizer_config.json"),
        "special_tokens_map.json": _sha256(args.backbone / "special_tokens_map.json"),
        "preprocessor_config.json": _sha256(args.backbone / "preprocessor_config.json"),
        "processor_config.json": _sha256(args.backbone / "processor_config.json"),
    }
    for name, expected in protocol["backbone"]["sha256"].items():
        if backbone_hashes[name] != expected:
            raise RuntimeError(
                f"backbone digest mismatch for {name}: expected {expected}, found {backbone_hashes[name]}"
            )

    installed_lerobot = importlib.metadata.version("lerobot")
    if installed_lerobot != protocol["software"]["lerobot"]:
        raise RuntimeError(
            f"LeRobot version mismatch: expected {protocol['software']['lerobot']}, found {installed_lerobot}"
        )
    if not platform.python_version().startswith(protocol["software"]["python"] + "."):
        raise RuntimeError(
            f"Python version mismatch: expected {protocol['software']['python']}.x, found {platform.python_version()}"
        )
    for package in ("hf-libero", "robosuite", "mujoco", "num2words"):
        installed = importlib.metadata.version(package)
        if installed != protocol["software"][package]:
            raise RuntimeError(
                f"{package} version mismatch: expected {protocol['software'][package]}, found {installed}"
            )

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sink = JsonlSink(args.output_dir / "action_trace.jsonl")
    recorder = TraceRecorder(sink, protocol["runtime_contract"])
    started = time.time()

    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import close_envs, eval_one
    from lerobot.utils.random_utils import set_seed

    frozen = protocol["evaluation"]
    seed = int(frozen["seed"] if args.mode == "final" else protocol["preflight"]["seed"])
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(frozen["allow_tf32"])

    asset_identity = _tree_identity(
        Path(get_libero_path("assets")), scope="mixed_site_packages_tree"
    )
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if asset_identity[key] != protocol["assets"]["resolved_tree"][key]:
            raise RuntimeError(
                f"resolved asset tree mismatch for {key}: expected "
                f"{protocol['assets']['resolved_tree'][key]}, found {asset_identity[key]}"
            )
    backbone_tree_identity = _tree_identity(
        args.backbone, scope="complete_local_snapshot_tree"
    )
    for key in ("file_count", "total_bytes", "tree_sha256"):
        if backbone_tree_identity[key] != protocol["backbone"]["tree_identity"][key]:
            raise RuntimeError(
                f"backbone tree mismatch for {key}: expected "
                f"{protocol['backbone']['tree_identity'][key]}, found {backbone_tree_identity[key]}"
            )

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_state_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    task_source_identity: dict[str, dict[str, str]] = {"init_states": {}, "bddl": {}}
    for task_id, task in enumerate(task_suite.tasks):
        init_path = init_state_root / task.init_states_file
        bddl_path = bddl_root / task.bddl_file
        task_source_identity["init_states"][str(task_id)] = _sha256(init_path)
        task_source_identity["bddl"][str(task_id)] = _sha256(bddl_path)
    if task_source_identity != protocol["assets"]["task_source_sha256"]:
        raise RuntimeError("LIBERO-Spatial init-state or BDDL source digest mismatch")

    selected_task_ids = list(range(10)) if args.mode == "final" else [args.preflight_task_id]
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=selected_task_ids,
        fps=int(frozen["fps"]),
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        max_parallel_tasks=1,
        observation_height=int(frozen["observation_height"]),
        observation_width=int(frozen["observation_width"]),
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    checkpoint_config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    if checkpoint_config["n_action_steps"] != frozen["n_action_steps"]:
        raise RuntimeError("checkpoint action horizon differs from the frozen protocol")
    policy_cfg.device = "cuda"
    policy_cfg.vlm_model_name = str(args.backbone.resolve())
    policy_cfg.pretrained_path = args.checkpoint.resolve()
    rename_map = dict(frozen["rename_map"])
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    _wrap_policy(policy, recorder)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.checkpoint.resolve()),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone.resolve())},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    tracing_postprocessor = TracingProcessor(postprocessor, recorder)
    tracing_preprocessor = TracingInputProcessor(preprocessor, recorder)

    task_results: dict[str, Any] = {}
    caught_error: Exception | None = None
    failure_receipt: dict[str, Any] | None = None
    try:
        for task_position, task_id in enumerate(selected_task_ids):
            env = envs["libero_spatial"][task_id]
            instruction = str(env.call("task_description")[0])
            recorder.set_task(task_id, instruction=instruction)
            _wrap_env(env, recorder)
            task_video_dir = args.output_dir / "videos" / f"task_{task_id:02d}"
            state_indices = (
                list(frozen["fixed_initial_state_indices"])
                if args.mode == "final"
                else [int(protocol["preflight"]["initial_state_index"])]
            )
            available_initial_states = len(env.envs[0]._init_states)
            if available_initial_states <= max(state_indices):
                raise RuntimeError(
                    f"task {task_id} has {available_initial_states} fixed states, "
                    f"cannot evaluate frozen index {max(state_indices)} without modulo reuse"
                )
            per_episode_success: list[bool] = []
            sum_rewards: list[float | None] = []
            max_rewards: list[float | None] = []
            video_paths: list[str] = []
            for initial_state_index in state_indices:
                episode_seed = (
                    seed + initial_state_index if args.mode == "final" else seed
                )
                try:
                    if not hasattr(env, "envs") or len(env.envs) != 1:
                        raise RuntimeError("fixed-state evaluation requires one synchronous environment")
                    env.envs[0].init_state_id = initial_state_index
                    if env.envs[0].init_state_id != initial_state_index:
                        raise RuntimeError("failed to set the frozen LIBERO initial-state index")
                    recorder.plan_episode(initial_state_index, episode_seed)
                    render_episode = (
                        args.mode == "final"
                        and task_id == frozen["video_selection"]["task_id"]
                        and initial_state_index
                        == frozen["video_selection"]["initial_state_index"]
                    )
                    result = eval_one(
                        env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=tracing_preprocessor,
                        postprocessor=tracing_postprocessor,
                        n_episodes=1,
                        max_episodes_rendered=1 if render_episode else 0,
                        videos_dir=task_video_dir if render_episode else None,
                        return_episode_data=False,
                        start_seed=episode_seed,
                    )
                    if len(result["successes"]) != 1:
                        raise RuntimeError("official eval_one returned an unexpected episode count")
                    per_episode_success.append(bool(result["successes"][0]))
                    sum_rewards.append(float(result["sum_rewards"][0]))
                    max_rewards.append(float(result["max_rewards"][0]))
                    video_paths.extend(str(path) for path in result["video_paths"])
                except Exception as error:
                    caught_error = error
                    failure_receipt = {
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "active_task_id": task_id,
                        "active_initial_state_index": initial_state_index,
                        "completed_episodes_before_failure": sum(
                            item["completed_episodes"] for item in task_results.values()
                        )
                        + len(per_episode_success),
                        "rule": "no retry or replacement; current and remaining frozen-grid episodes count as failure",
                    }
                    missing = len(state_indices) - len(per_episode_success)
                    per_episode_success.extend([False] * missing)
                    sum_rewards.extend([None] * missing)
                    max_rewards.extend([None] * missing)
                    break
            task_successes = sum(per_episode_success)
            completed_episodes = (
                len(state_indices)
                if caught_error is None
                else len(state_indices) - sum(value is None for value in sum_rewards)
            )
            task_results[str(task_id)] = {
                "execution_status": "complete" if caught_error is None else "partial_failure",
                "episodes": len(state_indices),
                "completed_episodes": completed_episodes,
                "successes": task_successes,
                "success_rate": task_successes / len(state_indices),
                "wilson_95": _wilson(task_successes, len(state_indices)),
                "initial_state_indices": state_indices,
                "episode_seeds": [
                    seed + index if args.mode == "final" else seed for index in state_indices
                ],
                "per_episode_success": per_episode_success,
                "sum_rewards": sum_rewards,
                "max_rewards": max_rewards,
                "video_paths": video_paths,
            }
            if caught_error is not None:
                for remaining_task_id in selected_task_ids[task_position + 1 :]:
                    remaining_states = list(frozen["fixed_initial_state_indices"])
                    task_results[str(remaining_task_id)] = {
                        "execution_status": "not_run_after_fatal_episode_failure",
                        "episodes": len(remaining_states),
                        "completed_episodes": 0,
                        "successes": 0,
                        "success_rate": 0.0,
                        "wilson_95": _wilson(0, len(remaining_states)),
                        "initial_state_indices": remaining_states,
                        "episode_seeds": [seed + index for index in remaining_states],
                        "per_episode_success": [False] * len(remaining_states),
                        "sum_rewards": [None] * len(remaining_states),
                        "max_rewards": [None] * len(remaining_states),
                        "video_paths": [],
                        "counted_as_failure_by_protocol": True,
                    }
                break
    except Exception as error:
        caught_error = error
        failure_receipt = {
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "rule": "fatal setup failure before an episode result",
        }
        for remaining_task_id in selected_task_ids:
            if str(remaining_task_id) in task_results:
                continue
            remaining_states = (
                list(frozen["fixed_initial_state_indices"])
                if args.mode == "final"
                else [int(protocol["preflight"]["initial_state_index"])]
            )
            task_results[str(remaining_task_id)] = {
                "execution_status": "failed_or_not_run",
                "episodes": len(remaining_states),
                "completed_episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "wilson_95": _wilson(0, len(remaining_states)),
                "initial_state_indices": remaining_states,
                "episode_seeds": [seed + index for index in remaining_states],
                "per_episode_success": [False] * len(remaining_states),
                "sum_rewards": [None] * len(remaining_states),
                "max_rewards": [None] * len(remaining_states),
                "video_paths": [],
                "counted_as_failure_by_protocol": True,
            }
    finally:
        close_envs(envs)
        sink.close()

    trace_path = args.output_dir / "action_trace.jsonl"
    trace_receipt = {
        "path": trace_path.name,
        "sha256": _sha256(trace_path),
        "bytes": trace_path.stat().st_size,
    }
    video_artifacts: list[dict[str, Any]] = []
    for task_result in task_results.values():
        for video_path_string in task_result["video_paths"]:
            video_path = Path(video_path_string)
            if not video_path.is_file():
                caught_error = FileNotFoundError(f"reported rollout video is missing: {video_path}")
                failure_receipt = {
                    "error_type": type(caught_error).__name__,
                    "message": str(caught_error),
                    "rule": "completed rollout metrics retained; missing required video fails the artifact run",
                }
                continue
            video_artifacts.append(
                {
                    "path": str(video_path.resolve()),
                    "sha256": _sha256(video_path),
                    "bytes": video_path.stat().st_size,
                }
            )

    total = sum(item["episodes"] for item in task_results.values())
    successes = sum(item["successes"] for item in task_results.values())
    elapsed = time.time() - started
    manifest = {
        "schema": "sentinel-libero-closedloop-v1",
        "status": "complete" if caught_error is None else "failed_with_frozen_grid_accounting",
        "mode": args.mode,
        "included_in_final_metrics": args.mode == "final",
        "protocol_sha256": _sha256(args.protocol),
        "script_sha256": actual_script_sha,
        "checkpoint": {
            "repo_id": protocol["checkpoint"]["repo_id"],
            "revision": protocol["checkpoint"]["revision"],
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hashes,
        },
        "backbone": {
            "repo_id": protocol["backbone"]["repo_id"],
            "revision": protocol["backbone"]["revision"],
            "path": str(args.backbone.resolve()),
            "sha256": backbone_hashes,
            "tree_identity": backbone_tree_identity,
        },
        "assets": {
            "declared_source": {
                "repo_id": protocol["assets"]["repo_id"],
                "repo_type": protocol["assets"]["repo_type"],
                "revision": protocol["assets"]["revision"],
            },
            "resolved_tree": asset_identity,
            "task_source_sha256": task_source_identity,
        },
        "preflight_evidence": preflight_receipt,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lerobot": installed_lerobot,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mujoco": importlib.metadata.version("mujoco"),
            "hf-libero": importlib.metadata.version("hf-libero"),
            "robosuite": importlib.metadata.version("robosuite"),
            "num2words": importlib.metadata.version("num2words"),
        },
        "effective_environment": asdict(env_cfg),
        "effective_policy": {
            "type": checkpoint_config["type"],
            "chunk_size": checkpoint_config["chunk_size"],
            "n_action_steps": checkpoint_config["n_action_steps"],
            "num_steps": checkpoint_config["num_steps"],
            "checkpoint_declared_state_shape": checkpoint_config["input_features"][
                "observation.state"
            ]["shape"],
            "runtime_backbone_path_substitution": True,
            "rename_map": rename_map,
            "runtime_contract": protocol["runtime_contract"],
        },
        "trace": {
            **trace_receipt,
            "raw_chunks": recorder.raw_chunks,
            "chunk_context": "instruction, unnormalized 8D state values, and per-camera frame hashes",
            "raw_camera_pixels_retained": False,
            "env_step_actions": recorder.actions,
            "passthrough_comparisons": recorder.passthrough_comparisons,
            "max_postprocess_to_env_step_abs_diff": recorder.max_passthrough_abs_diff,
            "gate_mode": "shadow_observe_only",
            "interventions": 0,
        },
        "video_artifacts": video_artifacts,
        "results": {
            "tasks": task_results,
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total,
            "wilson_95": _wilson(successes, total),
        },
        "timing": {"elapsed_seconds": elapsed, "seconds_per_episode": elapsed / total},
        "failure": failure_receipt,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["results"], indent=2, sort_keys=True))
    if caught_error is not None:
        raise caught_error
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
            failure_dir = Path(output_arg)
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "schema": "sentinel-libero-failure-v1",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "retained_as_result": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
