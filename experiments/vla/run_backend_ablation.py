#!/usr/bin/env python3
"""Run the frozen paired MuJoCo-version SmolVLA LIBERO compatibility study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_libero_closedloop import (
    JsonlSink,
    TraceRecorder,
    TracingInputProcessor,
    TracingProcessor,
    _array_digest,
    _jsonable,
    _sha256,
    _tree_identity,
    _wilson,
    _wrap_env,
    _wrap_policy,
)


class EpisodeSink(JsonlSink):
    """Write the full trace while retaining only per-episode counters in memory."""

    def __init__(self, path: Path, version_label: str):
        super().__init__(path)
        self.version_label = version_label
        self.episode_id: str | None = None
        self.raw_chunks = 0
        self.env_steps = 0
        self.first_chunk: dict[str, Any] | None = None
        self.reset_event: dict[str, int] | None = None

    def start_episode(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.raw_chunks = 0
        self.env_steps = 0
        self.first_chunk = None
        self.reset_event = None

    def write(self, payload: dict[str, Any]) -> None:
        enriched = _jsonable(
            {**payload, "backend_version_label": self.version_label, "episode_id": self.episode_id}
        )
        event = enriched.get("event")
        if event == "episode_reset":
            self.reset_event = {
                "actual_initial_state_index": int(enriched["initial_state_index"]),
                "actual_seed": int(enriched["seed"]),
            }
        elif event == "raw_policy_chunk":
            self.raw_chunks += 1
            if self.first_chunk is None:
                values = np.asarray(enriched["values"], dtype=np.float32)
                first_action = values[0, 0].copy()
                context = enriched["input_context"]
                self.first_chunk = {
                    "raw_chunk_sha256": enriched["sha256"],
                    "raw_chunk_shape": enriched["shape"],
                    "input_state_sha256": context["state"]["sha256"],
                    "input_camera_frames": context["camera_frames"],
                    "first_predicted_normalized_action": first_action.tolist(),
                    "first_predicted_normalized_action_sha256": _array_digest(first_action),
                }
        elif event == "env_step_action":
            self.env_steps += 1
        super().write(enriched)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path)
    parser.add_argument("--native-protocol", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--native-trace", type=Path)
    parser.add_argument("--reset-runner", type=Path)
    parser.add_argument("--reset-protocol", type=Path)
    parser.add_argument("--reset-current", type=Path)
    parser.add_argument("--reset-legacy", type=Path)
    parser.add_argument("--reset-comparison", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--libero-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal", "compare"), required=True)
    parser.add_argument("--version-label", choices=("current", "legacy"))
    parser.add_argument("--preflight-current", type=Path)
    parser.add_argument("--preflight-legacy", type=Path)
    parser.add_argument("--current-manifest", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    return parser.parse_args()


def _require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"output directory must be empty: {path}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _verify_bound_file(path: Path | None, expected: str, label: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing bound {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"bound {label} digest mismatch: expected {expected}, found {actual}")
    return _receipt(path)


def _verify_static_inputs(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    bound_paths = {
        "native_runner": args.native_runner,
        "native_protocol": args.native_protocol,
        "native_manifest": args.native_manifest,
        "native_trace": args.native_trace,
        "reset_runner": args.reset_runner,
        "reset_protocol": args.reset_protocol,
        "reset_current": args.reset_current,
        "reset_legacy": args.reset_legacy,
        "reset_comparison": args.reset_comparison,
    }
    receipts = {
        name: _verify_bound_file(path, protocol["source_bindings"][f"{name}_sha256"], name)
        for name, path in bound_paths.items()
    }
    native = _json(args.native_protocol)
    native_manifest = _json(args.native_manifest)
    if native_manifest.get("status") != "complete" or native_manifest["results"]["episodes"] != 100:
        raise RuntimeError("bound native evaluation is incomplete")
    reset_current = _json(args.reset_current)
    reset_legacy = _json(args.reset_legacy)
    reset_comparison = _json(args.reset_comparison)
    if reset_current.get("reset_count") != 20 or reset_legacy.get("reset_count") != 20:
        raise RuntimeError("bound reset audit does not contain both frozen 20-reset captures")
    if reset_comparison.get("status") != "complete" or reset_comparison.get("paired_reset_count") != 20:
        raise RuntimeError("bound reset-audit comparison is incomplete")

    checkpoint_hashes = {
        name: _sha256(args.checkpoint / name) for name in native["checkpoint"]["sha256"]
    }
    if checkpoint_hashes != native["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint digest mismatch")
    backbone_hashes = {name: _sha256(args.backbone / name) for name in native["backbone"]["sha256"]}
    if backbone_hashes != native["backbone"]["sha256"]:
        raise RuntimeError("backbone digest mismatch")
    backbone_tree = _tree_identity(args.backbone, "complete_local_snapshot_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if backbone_tree[key] != native["backbone"]["tree_identity"][key]:
            raise RuntimeError(f"backbone tree mismatch: {key}")
    return {
        "receipts": receipts,
        "native_protocol": native,
        "checkpoint_sha256": checkpoint_hashes,
        "backbone_sha256": backbone_hashes,
        "backbone_tree": backbone_tree,
    }


def _verify_preflight(
    path: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
    version_label: str,
    expected_artifacts: dict[str, Any],
) -> dict[str, Any]:
    manifest = _json(path)
    expected_top = {
        "schema": "sentinel-libero-backend-ablation-capture-v1",
        "status": "complete",
        "mode": "preflight",
        "included_in_formal_metrics": False,
        "version_label": version_label,
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "episodes_accounted": 1,
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"backend preflight {key} mismatch: {path}")
    if manifest["software"]["mujoco"] != protocol["versions"][version_label]:
        raise RuntimeError(f"backend preflight MuJoCo mismatch: {path}")
    for key, expected in expected_artifacts.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"backend preflight {key} binding mismatch: {path}")
    episodes = manifest.get("episodes", [])
    if len(episodes) != 1:
        raise RuntimeError(f"backend preflight episode count mismatch: {path}")
    episode = episodes[0]
    expected_episode = {
        "task_id": protocol["preflight"]["task_id"],
        "initial_state_index": protocol["preflight"]["initial_state_index"],
        "seed": protocol["preflight"]["seed"],
        "actual_initial_state_index": protocol["preflight"]["initial_state_index"],
        "actual_seed": protocol["preflight"]["seed"],
        "reset_observed": True,
        "execution_horizon": protocol["evaluation"]["execution_horizon"],
        "prediction_chunk_size": protocol["evaluation"]["prediction_chunk_size"],
        "crashed": False,
    }
    for key, expected in expected_episode.items():
        if episode.get(key) != expected:
            raise RuntimeError(f"backend preflight episode {key} mismatch: {path}")
    trace = manifest["trace"]
    trace_path = path.parent / trace["path"]
    if not trace_path.is_file() or trace_path.stat().st_size != trace["bytes"] or _sha256(trace_path) != trace["sha256"]:
        raise RuntimeError(f"backend preflight trace receipt mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "version_label": version_label,
        "mujoco": manifest["software"]["mujoco"],
        "task_id": episode["task_id"],
        "initial_state_index": episode["actual_initial_state_index"],
        "seed": episode["actual_seed"],
    }


def _episode_seed(protocol: dict[str, Any], mode: str, state_index: int) -> int:
    if mode == "preflight":
        return int(protocol["preflight"]["seed"])
    states = protocol["evaluation"]["fixed_initial_state_indices"]
    return int(protocol["evaluation"]["seed_base"] + state_index - states[0])


def _capture(args: argparse.Namespace, protocol: dict[str, Any], script_sha: str) -> None:
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    bound = _verify_static_inputs(args, protocol)
    native = bound["native_protocol"]
    expected_mujoco = protocol["versions"][args.version_label]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words")
    }
    for package in ("lerobot", "hf-libero", "robosuite", "num2words"):
        if versions[package] != native["software"][package]:
            raise RuntimeError(f"software mismatch: {package}")
    if versions["mujoco"] != expected_mujoco:
        raise RuntimeError(f"MuJoCo mismatch: expected {expected_mujoco}, found {versions['mujoco']}")
    if not platform.python_version().startswith(native["software"]["python"] + "."):
        raise RuntimeError("Python version mismatch")

    import mujoco
    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import close_envs, eval_one
    from lerobot.utils.random_utils import set_seed

    asset_tree = _tree_identity(Path(get_libero_path("assets")), "mixed_site_packages_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if asset_tree[key] != native["assets"]["resolved_tree"][key]:
            raise RuntimeError(f"asset tree mismatch: {key}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    sources = {"init_states": {}, "bddl": {}}
    for task_id, task in enumerate(suite.tasks):
        sources["init_states"][str(task_id)] = _sha256(init_root / task.init_states_file)
        sources["bddl"][str(task_id)] = _sha256(bddl_root / task.bddl_file)
    if sources != native["assets"]["task_source_sha256"]:
        raise RuntimeError("LIBERO task-source digest mismatch")

    preflight_evidence = None
    protocol_sha = _sha256(args.protocol)
    expected_preflight_artifacts = {
        "source_bindings": bound["receipts"],
        "model_artifacts": {
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "backbone_sha256": bound["backbone_sha256"],
            "backbone_tree": bound["backbone_tree"],
        },
        "asset_tree": asset_tree,
        "task_source_sha256": sources,
    }
    if args.mode == "formal":
        preflight_evidence = {
            "current": _verify_preflight(
                args.preflight_current,
                protocol,
                protocol_sha,
                script_sha,
                "current",
                expected_preflight_artifacts,
            ),
            "legacy": _verify_preflight(
                args.preflight_legacy,
                protocol,
                protocol_sha,
                script_sha,
                "legacy",
                expected_preflight_artifacts,
            ),
        }

    task_ids = (
        [protocol["preflight"]["task_id"]]
        if args.mode == "preflight"
        else protocol["evaluation"]["task_ids"]
    )
    states = (
        [protocol["preflight"]["initial_state_index"]]
        if args.mode == "preflight"
        else protocol["evaluation"]["fixed_initial_state_indices"]
    )
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=task_ids,
        fps=20,
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        max_parallel_tasks=1,
        observation_height=360,
        observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 50:
        raise RuntimeError("checkpoint must declare chunk_size=n_action_steps=50")
    policy_cfg.device = "cuda"
    policy_cfg.vlm_model_name = str(args.backbone.resolve())
    policy_cfg.pretrained_path = args.checkpoint.resolve()
    rename_map = native["evaluation"]["rename_map"]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    policy.config.n_action_steps = protocol["evaluation"]["execution_horizon"]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.checkpoint.resolve()),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone.resolve())},
        },
    )
    sink = EpisodeSink(args.output_dir / "action_trace.jsonl", args.version_label)
    recorder = TraceRecorder(sink, native["runtime_contract"])
    _wrap_policy(policy, recorder)
    tracing_preprocessor = TracingInputProcessor(preprocessor, recorder)
    tracing_postprocessor = TracingProcessor(postprocessor, recorder)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    episodes: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task_id in task_ids:
            env = envs["libero_spatial"][task_id]
            _wrap_env(env, recorder)
            recorder.set_task(task_id, str(env.call("task_description")[0]))
            if len(env.envs[0]._init_states) <= max(states):
                raise RuntimeError("insufficient fixed initial states")
            for state_index in states:
                seed = _episode_seed(protocol, args.mode, state_index)
                episode_id = f"task{task_id:02d}-state{state_index:02d}"
                sink.start_episode(episode_id)
                episode_started = time.time()
                episode: dict[str, Any] = {
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "initial_state_index": state_index,
                    "seed": seed,
                    "execution_horizon": protocol["evaluation"]["execution_horizon"],
                    "prediction_chunk_size": protocol["evaluation"]["prediction_chunk_size"],
                }
                try:
                    set_seed(seed)
                    policy.config.n_action_steps = protocol["evaluation"]["execution_horizon"]
                    policy.reset()
                    env.envs[0].init_state_id = state_index
                    recorder.plan_episode(state_index, seed)
                    result = eval_one(
                        env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=tracing_preprocessor,
                        postprocessor=tracing_postprocessor,
                        n_episodes=1,
                        max_episodes_rendered=0,
                        videos_dir=None,
                        return_episode_data=False,
                        start_seed=seed,
                    )
                    actual_index = int(env.envs[0].init_state_id - env.envs[0]._reset_stride)
                    actual_seed = int(env.envs[0].np_random_seed)
                    if actual_index != state_index or actual_seed != seed:
                        raise RuntimeError(
                            f"reset mismatch: state {actual_index}/{state_index}, seed {actual_seed}/{seed}"
                        )
                    if sink.first_chunk is None or sink.first_chunk["raw_chunk_shape"] != [1, 50, 7]:
                        raise RuntimeError("missing or invalid first 50-action prediction chunk")
                    episode.update(
                        {
                            "actual_initial_state_index": actual_index,
                            "actual_seed": actual_seed,
                            "reset_observed": True,
                            "success": bool(result["successes"][0]),
                            "sum_reward": float(result["sum_rewards"][0]),
                            "max_reward": float(result["max_rewards"][0]),
                            "env_steps": sink.env_steps,
                            "raw_chunks": sink.raw_chunks,
                            "first_chunk": sink.first_chunk,
                            "crashed": False,
                        }
                    )
                except Exception as error:
                    reset_event = sink.reset_event or {}
                    episode.update(
                        {
                            "actual_initial_state_index": reset_event.get(
                                "actual_initial_state_index"
                            ),
                            "actual_seed": reset_event.get("actual_seed"),
                            "reset_observed": sink.reset_event is not None,
                            "success": False,
                            "crashed": True,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                            "env_steps": sink.env_steps,
                            "raw_chunks": sink.raw_chunks,
                            "first_chunk": sink.first_chunk,
                        }
                    )
                episode["elapsed_seconds"] = time.time() - episode_started
                episodes.append(episode)
    finally:
        close_envs(envs)
        sink.close()

    task_metrics: dict[str, Any] = {}
    for task_id in task_ids:
        rows = [episode for episode in episodes if episode["task_id"] == task_id]
        successes = sum(bool(row["success"]) for row in rows)
        task_metrics[str(task_id)] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "wilson_95": _wilson(successes, len(rows)),
            "crashes": sum(bool(row["crashed"]) for row in rows),
            "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
        }
    trace_path = args.output_dir / "action_trace.jsonl"
    expected_count = 1 if args.mode == "preflight" else 20
    status = (
        "complete"
        if len(episodes) == expected_count
        and (args.mode == "formal" or not any(episode["crashed"] for episode in episodes))
        else "failed"
    )
    manifest = {
        "schema": "sentinel-libero-backend-ablation-capture-v1",
        "status": status,
        "mode": args.mode,
        "included_in_formal_metrics": args.mode == "formal",
        "version_label": args.version_label,
        "script_sha256": script_sha,
        "protocol_sha256": protocol_sha,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **versions,
            "mujoco_module": str(Path(mujoco.__file__).resolve()),
        },
        "source_bindings": bound["receipts"],
        "preflight_evidence": preflight_evidence,
        "effective_environment": asdict(env_cfg),
        "model_artifacts": {
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "backbone_sha256": bound["backbone_sha256"],
            "backbone_tree": bound["backbone_tree"],
        },
        "asset_tree": asset_tree,
        "task_source_sha256": sources,
        "episodes_accounted": len(episodes),
        "episodes": episodes,
        "metrics": task_metrics,
        "trace": {**_receipt(trace_path), "path": trace_path.name},
        "timing": {"elapsed_seconds": time.time() - started},
        "interpretation_limits": protocol["claims"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "complete":
        raise RuntimeError(f"backend capture accounted for {len(episodes)} of {expected_count} episodes")


def _paired_bootstrap(values: list[int], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _compare(args: argparse.Namespace, protocol: dict[str, Any], script_sha: str) -> None:
    current = _json(args.current_manifest)
    legacy = _json(args.legacy_manifest)
    protocol_sha = _sha256(args.protocol)
    expected_grid = {
        (task, state)
        for task in protocol["evaluation"]["task_ids"]
        for state in protocol["evaluation"]["fixed_initial_state_indices"]
    }
    records: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for label, path, manifest in (
        ("current", args.current_manifest, current),
        ("legacy", args.legacy_manifest, legacy),
    ):
        expected = {
            "schema": "sentinel-libero-backend-ablation-capture-v1",
            "status": "complete",
            "mode": "formal",
            "included_in_formal_metrics": True,
            "version_label": label,
            "script_sha256": script_sha,
            "protocol_sha256": protocol_sha,
            "episodes_accounted": 20,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"formal {label} {key} mismatch: {path}")
        if manifest["software"]["mujoco"] != protocol["versions"][label]:
            raise RuntimeError(f"formal {label} MuJoCo version mismatch")
        for name, expected_sha in protocol["source_bindings"].items():
            receipt_name = name.removesuffix("_sha256")
            if manifest["source_bindings"].get(receipt_name, {}).get("sha256") != expected_sha:
                raise RuntimeError(f"formal {label} source binding mismatch: {receipt_name}")
        evidence = manifest.get("preflight_evidence", {})
        if set(evidence) != {"current", "legacy"}:
            raise RuntimeError(f"formal {label} preflight evidence is incomplete")
        for preflight_label in ("current", "legacy"):
            receipt = evidence[preflight_label]
            if receipt.get("version_label") != preflight_label or receipt.get("mujoco") != protocol["versions"][preflight_label]:
                raise RuntimeError(f"formal {label} preflight receipt mismatch: {preflight_label}")
        trace_path = path.parent / manifest["trace"]["path"]
        if _sha256(trace_path) != manifest["trace"]["sha256"]:
            raise RuntimeError(f"formal {label} trace digest mismatch")
        mapped = {(row["task_id"], row["initial_state_index"]): row for row in manifest["episodes"]}
        if set(mapped) != expected_grid:
            raise RuntimeError(f"formal {label} grid mismatch")
        records[label] = mapped
    if current["preflight_evidence"] != legacy["preflight_evidence"]:
        raise RuntimeError("formal captures do not bind the same two preflight manifests")

    pairs: list[dict[str, Any]] = []
    for task_id, state_index in sorted(expected_grid):
        a = records["current"][(task_id, state_index)]
        b = records["legacy"][(task_id, state_index)]
        expected_seed = _episode_seed(protocol, "formal", state_index)
        for row in (a, b):
            if row["seed"] != expected_seed:
                raise RuntimeError("formal planned seed mismatch")
            if not row["crashed"] or row.get("reset_observed"):
                if row["actual_seed"] != expected_seed:
                    raise RuntimeError("formal observed reset seed mismatch")
                if row["actual_initial_state_index"] != state_index:
                    raise RuntimeError("formal observed reset state mismatch")
            elif row.get("actual_seed") is not None or row.get("actual_initial_state_index") is not None:
                raise RuntimeError("reset-unobserved crash retained stale reset evidence")
        first_a = a.get("first_chunk")
        first_b = b.get("first_chunk")
        state_equal = (
            None
            if first_a is None or first_b is None
            else first_a["input_state_sha256"] == first_b["input_state_sha256"]
        )
        camera_equal = (
            None
            if first_a is None or first_b is None
            else first_a["input_camera_frames"] == first_b["input_camera_frames"]
        )
        action_equal = (
            None
            if first_a is None or first_b is None
            else first_a["first_predicted_normalized_action_sha256"]
            == first_b["first_predicted_normalized_action_sha256"]
        )
        action_max_abs_diff = (
            None
            if first_a is None or first_b is None
            else float(
                np.max(
                    np.abs(
                        np.asarray(first_a["first_predicted_normalized_action"], dtype=np.float64)
                        - np.asarray(first_b["first_predicted_normalized_action"], dtype=np.float64)
                    )
                )
            )
        )
        pairs.append(
            {
                "task_id": task_id,
                "initial_state_index": state_index,
                "seed": expected_seed,
                "current_success": bool(a["success"]),
                "legacy_success": bool(b["success"]),
                "legacy_minus_current_success": int(b["success"]) - int(a["success"]),
                "current_crashed": bool(a["crashed"]),
                "legacy_crashed": bool(b["crashed"]),
                "current_reset_observed": bool(a.get("reset_observed")),
                "legacy_reset_observed": bool(b.get("reset_observed")),
                "first_input_state_hash_equal": state_equal,
                "first_input_camera_hashes_equal": camera_equal,
                "first_predicted_action_equal": action_equal,
                "first_predicted_action_max_abs_diff": action_max_abs_diff,
            }
        )

    metrics: dict[str, Any] = {}
    for task_id in protocol["evaluation"]["task_ids"]:
        rows = [row for row in pairs if row["task_id"] == task_id]
        differences = [row["legacy_minus_current_success"] for row in rows]
        current_successes = sum(row["current_success"] for row in rows)
        legacy_successes = sum(row["legacy_success"] for row in rows)
        metrics[str(task_id)] = {
            "pairs": len(rows),
            "current_successes": current_successes,
            "current_success_rate": current_successes / len(rows),
            "current_wilson_95": _wilson(current_successes, len(rows)),
            "legacy_successes": legacy_successes,
            "legacy_success_rate": legacy_successes / len(rows),
            "legacy_wilson_95": _wilson(legacy_successes, len(rows)),
            "paired_delta_legacy_minus_current": float(np.mean(differences)),
            "paired_bootstrap_95": _paired_bootstrap(
                differences,
                protocol["analysis"]["bootstrap_samples"],
                protocol["analysis"]["bootstrap_seed"] + task_id,
            ),
            "discordant_pairs": sum(row["current_success"] != row["legacy_success"] for row in rows),
            "reset_unobserved_crashes": sum(
                (row["current_crashed"] and not row["current_reset_observed"])
                + (row["legacy_crashed"] and not row["legacy_reset_observed"])
                for row in rows
            ),
            "first_input_state_different_pairs": sum(
                row["first_input_state_hash_equal"] is False for row in rows
            ),
            "first_input_camera_different_pairs": sum(
                row["first_input_camera_hashes_equal"] is False for row in rows
            ),
            "first_predicted_action_different_pairs": sum(
                row["first_predicted_action_equal"] is False for row in rows
            ),
            "missing_first_chunk_pairs": sum(
                row["first_predicted_action_equal"] is None for row in rows
            ),
            "max_first_predicted_action_abs_diff": max(
                (
                    row["first_predicted_action_max_abs_diff"]
                    for row in rows
                    if row["first_predicted_action_max_abs_diff"] is not None
                ),
                default=None,
            ),
        }
    pairs_path = args.output_dir / "paired_episodes.json"
    pairs_path.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = {
        "schema": "sentinel-libero-backend-ablation-comparison-v1",
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "current_manifest": _receipt(args.current_manifest),
        "legacy_manifest": _receipt(args.legacy_manifest),
        "paired_episode_count": len(pairs),
        "metrics": metrics,
        "paired_episodes": _receipt(pairs_path),
        "interpretation_limits": protocol["claims"],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    args.protocol = args.protocol.resolve()
    args.output_dir = args.output_dir.resolve()
    protocol = _json(args.protocol)
    if protocol.get("schema") != "sentinel-libero-backend-ablation-protocol-v1":
        raise RuntimeError("unexpected backend protocol schema")
    if protocol.get("state") != "frozen_before_formal_run":
        raise RuntimeError("backend protocol must be frozen before capture")
    script_sha = _sha256(Path(__file__).resolve())
    if protocol["implementation"]["script_sha256"] != script_sha:
        raise RuntimeError("backend runner digest differs from frozen protocol")
    _require_empty_output(args.output_dir)
    if args.mode == "compare":
        if args.current_manifest is None or args.legacy_manifest is None:
            raise RuntimeError("compare mode requires both formal manifests")
        _compare(args, protocol, script_sha)
    else:
        required = (
            args.version_label,
            args.libero_config,
            args.native_runner,
            args.native_protocol,
            args.native_manifest,
            args.native_trace,
            args.reset_runner,
            args.reset_protocol,
            args.reset_current,
            args.reset_legacy,
            args.reset_comparison,
            args.checkpoint,
            args.backbone,
        )
        if any(value is None for value in required):
            raise RuntimeError("capture mode requires every source, model, reset-audit, and environment input")
        if args.mode == "formal" and (args.preflight_current is None or args.preflight_legacy is None):
            raise RuntimeError("formal mode requires both version-specific preflight manifests")
        _capture(args, protocol, script_sha)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
