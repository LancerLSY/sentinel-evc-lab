#!/usr/bin/env python3
"""Run the prospective full-suite SmolVLA LIBERO confirmation lane."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_backend_ablation import EpisodeSink, _receipt, _verify_bound_file
from run_libero_closedloop import (
    TraceRecorder,
    TracingInputProcessor,
    TracingProcessor,
    _sha256,
    _tree_identity,
    _wilson,
    _wrap_env,
    _wrap_policy,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    for name in (
        "native_runner", "native_protocol", "native_manifest", "native_trace",
        "reset_comparison", "horizon_runner", "horizon_protocol", "horizon_manifest",
        "horizon_trace", "backend_runner", "backend_protocol", "backend_current",
        "backend_legacy", "backend_comparison",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--libero-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--preflight-manifest", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _verify_inputs(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    if Path(inspect.getsourcefile(EpisodeSink) or "").resolve() != args.backend_runner.resolve():
        raise RuntimeError("imported backend helper source differs from --backend-runner")
    if Path(inspect.getsourcefile(TraceRecorder) or "").resolve() != args.native_runner.resolve():
        raise RuntimeError("imported native trace source differs from --native-runner")
    paths = {
        name: getattr(args, name)
        for name in (
            "native_runner", "native_protocol", "native_manifest", "native_trace",
            "reset_comparison", "horizon_runner", "horizon_protocol", "horizon_manifest",
            "horizon_trace", "backend_runner", "backend_protocol", "backend_current",
            "backend_legacy", "backend_comparison",
        )
    }
    bindings = protocol["source_bindings"]
    if any(str(value).startswith("PENDING_") for value in bindings.values()):
        raise RuntimeError("confirmation protocol still has pending backend-result bindings")
    receipts = {
        name: _verify_bound_file(path, bindings[f"{name}_sha256"], name)
        for name, path in paths.items()
    }
    native = _load(args.native_protocol)
    native_manifest = _load(args.native_manifest)
    horizon_manifest = _load(args.horizon_manifest)
    backend_current = _load(args.backend_current)
    backend_legacy = _load(args.backend_legacy)
    backend_comparison = _load(args.backend_comparison)
    reset_comparison = _load(args.reset_comparison)
    if native_manifest.get("status") != "complete" or native_manifest["results"]["episodes"] != 100:
        raise RuntimeError("native 100-episode evidence is incomplete")
    if horizon_manifest.get("status") != "complete" or horizon_manifest.get("episodes_accounted") != 40:
        raise RuntimeError("10-action horizon evidence is incomplete")
    if reset_comparison.get("status") != "complete":
        raise RuntimeError("reset comparison is incomplete")
    for label, manifest in (("current", backend_current), ("legacy", backend_legacy)):
        if (
            manifest.get("schema") != "sentinel-libero-backend-ablation-capture-v1"
            or manifest.get("status") != "complete"
            or manifest.get("mode") != "formal"
            or manifest.get("version_label") != label
            or manifest.get("episodes_accounted") != 20
        ):
            raise RuntimeError(f"backend {label} formal result is incomplete")
    if (
        backend_comparison.get("schema") != "sentinel-libero-backend-ablation-comparison-v1"
        or backend_comparison.get("status") != "complete"
        or backend_comparison.get("paired_episode_count") != 20
    ):
        raise RuntimeError("backend comparison is incomplete")
    checkpoint_hashes = {name: _sha256(args.checkpoint / name) for name in native["checkpoint"]["sha256"]}
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
        "native": native, "receipts": receipts, "checkpoint_sha256": checkpoint_hashes,
        "backbone_sha256": backbone_hashes, "backbone_tree": backbone_tree,
    }


def _verify_preflight(
    path: Path, protocol: dict[str, Any], script_sha: str, protocol_sha: str,
    bound: dict[str, Any],
) -> dict[str, Any]:
    manifest = _load(path)
    expected = protocol["preflight"]
    checks = {
        "schema": manifest.get("schema") == "sentinel-libero-confirmation-v1",
        "status": manifest.get("status") == "complete",
        "mode": manifest.get("mode") == "preflight",
        "excluded": manifest.get("included_in_formal_metrics") is False,
        "script": manifest.get("script_sha256") == script_sha,
        "protocol": manifest.get("protocol_sha256") == protocol_sha,
        "count": manifest.get("episodes_accounted") == 1,
        "backend": manifest.get("software", {}).get("mujoco") == protocol["software"]["mujoco"],
        "sources": all(
            manifest.get("source_bindings", {}).get(name, {}).get("sha256") == receipt["sha256"]
            for name, receipt in bound["receipts"].items()
        ),
        "model": manifest.get("model_artifacts") == {
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "backbone_sha256": bound["backbone_sha256"],
            "backbone_tree": bound["backbone_tree"],
        },
    }
    rows = manifest.get("episodes", [])
    checks["episode"] = len(rows) == 1 and all(
        rows[0].get(key) == value for key, value in {
            "task_id": expected["task_id"], "initial_state_index": expected["initial_state_index"],
            "seed": expected["seed"], "actual_initial_state_index": expected["initial_state_index"],
            "actual_seed": expected["seed"], "reset_observed": True, "crashed": False,
        }.items()
    )
    trace = manifest.get("trace", {})
    trace_path = path.parent / trace.get("path", "")
    checks["trace"] = trace_path.is_file() and _sha256(trace_path) == trace.get("sha256")
    checks["outcomes"] = trace.get("env_step_outcomes") == trace.get("env_step_actions")
    if not all(checks.values()):
        raise RuntimeError(f"confirmation preflight mismatch: {checks}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "checks": checks}


def _wrap_step_outcome(env: Any, sink: EpisodeSink, recorder: TraceRecorder) -> None:
    original_step = env.step

    def step(action: Any) -> Any:
        result = original_step(action)
        observation, reward, terminated, truncated, info = result
        success = None
        success_source = None
        if "final_info" in info:
            final = info["final_info"]
            item = final if isinstance(final, dict) else final[0]
            if isinstance(item, dict) and "is_success" in item:
                value = item["is_success"]
                success = bool(np.asarray(value).reshape(-1)[0])
                success_source = "final_info"
        elif "is_success" in info:
            success = bool(np.asarray(info["is_success"]).reshape(-1)[0])
            success_source = "info"
        sink.write({"event": "env_step_outcome", "task_id": recorder.task_id,
            "initial_state_index": recorder.initial_state_index, "step": recorder.step - 1,
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
            "truncated": bool(np.asarray(truncated).reshape(-1)[0]), "is_success": success,
            "is_success_present": success is not None, "is_success_source": success_source})
        sink.outcome_events = getattr(sink, "outcome_events", 0) + 1
        return result

    env.step = step


def main() -> int:
    args = _args()
    protocol = _load(args.protocol)
    script_sha = _sha256(Path(__file__).resolve())
    if protocol["state"] != "frozen_before_formal_run":
        raise RuntimeError("confirmation protocol is not frozen")
    if protocol["implementation"]["script_sha256"] != script_sha:
        raise RuntimeError("confirmation runner SHA mismatch")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    bound = _verify_inputs(args, protocol)
    native = bound["native"]
    protocol_sha = _sha256(args.protocol)
    if args.mode == "formal":
        if args.preflight_manifest is None:
            raise FileNotFoundError("formal mode requires --preflight-manifest")
        preflight = _verify_preflight(args.preflight_manifest, protocol, script_sha, protocol_sha, bound)
    else:
        if args.preflight_manifest is not None:
            raise ValueError("preflight cannot consume a preflight manifest")
        preflight = None
    if importlib.metadata.version("mujoco") != protocol["software"]["mujoco"]:
        raise RuntimeError("confirmation requires isolated MuJoCo 3.3.7")
    for package in ("lerobot", "hf-libero", "robosuite", "num2words"):
        if importlib.metadata.version(package) != native["software"][package]:
            raise RuntimeError(f"software mismatch: {package}")
    if not platform.python_version().startswith(native["software"]["python"] + "."):
        raise RuntimeError("Python version mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)

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
        raise RuntimeError("task source digest mismatch")

    task_ids = [protocol["preflight"]["task_id"]] if args.mode == "preflight" else protocol["evaluation"]["task_ids"]
    states = [protocol["preflight"]["initial_state_index"]] if args.mode == "preflight" else protocol["evaluation"]["fixed_initial_state_indices"]
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=task_ids, fps=20, init_states=True,
        hard_reset=True, control_mode="relative", max_parallel_tasks=1,
        observation_height=360, observation_width=360)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 50:
        raise RuntimeError("checkpoint must declare chunk_size=n_action_steps=50")
    policy_cfg.device = "cuda"; policy_cfg.vlm_model_name = str(args.backbone.resolve()); policy_cfg.pretrained_path = args.checkpoint.resolve()
    rename_map = native["evaluation"]["rename_map"]
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval(); policy.config.n_action_steps = 10
    pre, post = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(args.checkpoint.resolve()),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": rename_map},
        "tokenizer_processor": {"tokenizer_name": str(args.backbone.resolve())}})
    sink = EpisodeSink(args.output_dir / "action_trace.jsonl", "legacy-confirmation")
    recorder = TraceRecorder(sink, native["runtime_contract"]); _wrap_policy(policy, recorder)
    tracing_pre = TracingInputProcessor(pre, recorder); tracing_post = TracingProcessor(post, recorder)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = True

    episodes: list[dict[str, Any]] = []; started = time.time(); fatal = None
    try:
        for task_id in task_ids:
            env = envs["libero_spatial"][task_id]; _wrap_env(env, recorder)
            _wrap_step_outcome(env, sink, recorder)
            recorder.set_task(task_id, str(env.call("task_description")[0]))
            if len(env.envs[0]._init_states) <= max(states):
                raise RuntimeError("insufficient fixed initial states")
            for state in states:
                seed = protocol["preflight"]["seed"] if args.mode == "preflight" else protocol["evaluation"]["seed_base"] + state - 30
                episode = {"task_id": task_id, "initial_state_index": state, "seed": seed,
                    "execution_horizon": 10, "prediction_chunk_size": 50}
                sink.start_episode(f"task{task_id:02d}-state{state:02d}"); episode_started = time.time()
                try:
                    set_seed(seed); policy.config.n_action_steps = 10; policy.reset(); env.envs[0].init_state_id = state
                    recorder.plan_episode(state, seed)
                    result = eval_one(env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                        preprocessor=tracing_pre, postprocessor=tracing_post, n_episodes=1,
                        max_episodes_rendered=0, videos_dir=None, return_episode_data=False, start_seed=seed)
                    reset = sink.reset_event or {}
                    if reset.get("actual_initial_state_index") != state or reset.get("actual_seed") != seed:
                        raise RuntimeError("observed reset state/seed mismatch")
                    if sink.first_chunk is None or sink.first_chunk["raw_chunk_shape"] != [1, 50, 7]:
                        raise RuntimeError("missing 50-action prediction chunk")
                    episode.update({"actual_initial_state_index": state, "actual_seed": seed,
                        "reset_observed": True, "success": bool(result["successes"][0]),
                        "sum_reward": float(result["sum_rewards"][0]), "max_reward": float(result["max_rewards"][0]),
                        "env_steps": sink.env_steps, "raw_chunks": sink.raw_chunks,
                        "first_chunk": sink.first_chunk, "crashed": False})
                except Exception as error:
                    reset = sink.reset_event or {}
                    episode.update({"actual_initial_state_index": reset.get("actual_initial_state_index"),
                        "actual_seed": reset.get("actual_seed"), "reset_observed": sink.reset_event is not None,
                        "success": False, "env_steps": sink.env_steps, "raw_chunks": sink.raw_chunks,
                        "first_chunk": sink.first_chunk, "crashed": True,
                        "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
                episode["elapsed_seconds"] = time.time() - episode_started; episodes.append(episode)
    except Exception as error:
        fatal = {"error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        close_envs(envs); sink.close()

    expected = [(task, state) for task in task_ids for state in states]
    present = {(row["task_id"], row["initial_state_index"]) for row in episodes}
    for task_id, state in expected:
        if (task_id, state) not in present:
            seed = protocol["preflight"]["seed"] if args.mode == "preflight" else protocol["evaluation"]["seed_base"] + state - 30
            episodes.append({"task_id": task_id, "initial_state_index": state, "seed": seed,
                "actual_initial_state_index": None, "actual_seed": None, "reset_observed": False,
                "execution_horizon": 10, "prediction_chunk_size": 50, "success": False,
                "crashed": True, "error_type": "NOT_RUN_AFTER_FATAL_FAILURE", "env_steps": 0,
                "raw_chunks": 0, "first_chunk": None, "elapsed_seconds": 0.0})
    episodes.sort(key=lambda row: (task_ids.index(row["task_id"]), states.index(row["initial_state_index"])))
    metrics = {}
    for task_id in task_ids:
        rows = [row for row in episodes if row["task_id"] == task_id]; successes = sum(row["success"] for row in rows)
        metrics[str(task_id)] = {"episodes": len(rows), "successes": successes,
            "success_rate": successes / len(rows), "wilson_95": _wilson(successes, len(rows)),
            "crashes": sum(row["crashed"] for row in rows)}
    successes = sum(row["success"] for row in episodes)
    metrics["overall"] = {"episodes": len(episodes), "successes": successes,
        "success_rate": successes / len(episodes), "wilson_95": _wilson(successes, len(episodes)),
        "crashes": sum(row["crashed"] for row in episodes),
        "env_steps": sum(row["env_steps"] for row in episodes),
        "raw_chunks": sum(row["raw_chunks"] for row in episodes)}
    trace_path = args.output_dir / "action_trace.jsonl"
    status = "complete" if len(episodes) == len(expected) and (args.mode == "formal" or not episodes[0]["crashed"]) else "failed"
    manifest = {"schema": "sentinel-libero-confirmation-v1", "status": status, "mode": args.mode,
        "included_in_formal_metrics": args.mode == "formal", "script_sha256": script_sha,
        "protocol_sha256": protocol_sha, "software": {"python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        **{p: importlib.metadata.version(p) for p in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words")}},
        "source_bindings": bound["receipts"], "preflight_evidence": preflight,
        "effective_environment": asdict(env_cfg), "model_artifacts": {"checkpoint_sha256": bound["checkpoint_sha256"],
        "backbone_sha256": bound["backbone_sha256"], "backbone_tree": bound["backbone_tree"]},
        "asset_tree": asset_tree, "task_source_sha256": sources, "episodes_accounted": len(episodes),
        "effective_policy": {"checkpoint_chunk_size": 50, "execution_horizon": 10,
        "runtime_contract": native["runtime_contract"]},
        "episodes": episodes, "metrics": metrics, "trace": {**_receipt(trace_path), "path": trace_path.name,
        "raw_chunks": recorder.raw_chunks, "env_step_actions": recorder.actions,
        "env_step_outcomes": getattr(sink, "outcome_events", 0),
        "passthrough_comparisons": recorder.passthrough_comparisons,
        "max_postprocess_to_env_step_abs_diff": recorder.max_passthrough_abs_diff,
        "interventions": 0},
        "timing": {"elapsed_seconds": time.time() - started}, "fatal_failure": fatal,
        "interpretation_limits": protocol["claims"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if status != "complete":
        raise RuntimeError("confirmation capture failed its fixed-grid accounting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
