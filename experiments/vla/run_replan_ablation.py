#!/usr/bin/env python3
"""Run the frozen paired SmolVLA execution-horizon ablation."""

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
    _jsonable,
    _sha256,
    _tree_identity,
    _wilson,
    _wrap_env,
    _wrap_policy,
)


class BranchSink(JsonlSink):
    def __init__(self, path: Path):
        super().__init__(path)
        self.branch: str | None = None
        self.pair_id: str | None = None
        self.events: list[dict[str, Any]] = []

    def write(self, payload: dict[str, Any]) -> None:
        enriched = dict(payload)
        enriched["branch"] = self.branch
        enriched["pair_id"] = self.pair_id
        self.events.append(_jsonable(enriched))
        super().write(enriched)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, required=True)
    parser.add_argument("--native-protocol", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--preflight-manifest", type=Path)
    return parser.parse_args()


def _verify_bound_inputs(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    bindings = protocol["native_bindings"]
    expected = {
        args.native_runner: bindings["runner_sha256"],
        args.native_protocol: bindings["protocol_sha256"],
        args.native_manifest: bindings["formal_manifest_sha256"],
        args.native_trace: bindings["formal_trace_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"bound native evidence mismatch: {path}")

    native_protocol = json.loads(args.native_protocol.read_text(encoding="utf-8"))
    native_manifest = json.loads(args.native_manifest.read_text(encoding="utf-8"))
    if native_manifest["status"] != "complete":
        raise RuntimeError("native formal manifest is not complete")
    if native_manifest["results"]["episodes"] != bindings["formal_episodes"]:
        raise RuntimeError("native formal episode count mismatch")
    if native_manifest["results"]["successes"] != bindings["formal_successes"]:
        raise RuntimeError("native formal success count mismatch")

    checkpoint_hashes = {
        name: _sha256(args.checkpoint / name)
        for name in native_protocol["checkpoint"]["sha256"]
    }
    if checkpoint_hashes != native_protocol["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint digest mismatch")
    backbone_hashes = {
        name: _sha256(args.backbone / name) for name in native_protocol["backbone"]["sha256"]
    }
    if backbone_hashes != native_protocol["backbone"]["sha256"]:
        raise RuntimeError("backbone digest mismatch")
    backbone_tree = _tree_identity(args.backbone, "complete_local_snapshot_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if backbone_tree[key] != native_protocol["backbone"]["tree_identity"][key]:
            raise RuntimeError(f"backbone tree mismatch: {key}")

    for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words"):
        if importlib.metadata.version(package) != native_protocol["software"][package]:
            raise RuntimeError(f"software version mismatch: {package}")
    if not platform.python_version().startswith(native_protocol["software"]["python"] + "."):
        raise RuntimeError("Python version mismatch")
    return {
        "native_protocol": native_protocol,
        "native_manifest": native_manifest,
        "checkpoint_sha256": checkpoint_hashes,
        "backbone_sha256": backbone_hashes,
        "backbone_tree": backbone_tree,
    }


def _paired_bootstrap(differences: list[int], samples: int, seed: int) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _first_ten_hash(raw_chunk: dict[str, Any]) -> str:
    values = np.asarray(raw_chunk["values"], dtype=np.float32)[:, :10, :]
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def main() -> int:
    args = _parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    actual_script_sha = _sha256(Path(__file__).resolve())
    if protocol["implementation"]["script_sha256"] != actual_script_sha:
        raise RuntimeError("replan runner SHA mismatch")
    if protocol["state"] != "frozen_before_formal_run":
        raise RuntimeError("replan protocol is not frozen")
    bound = _verify_bound_inputs(args, protocol)
    native_protocol = bound["native_protocol"]

    if args.mode == "formal":
        if args.preflight_manifest is None or not args.preflight_manifest.is_file():
            raise FileNotFoundError("formal mode requires --preflight-manifest")
        preflight = json.loads(args.preflight_manifest.read_text(encoding="utf-8"))
        checks = {
            "mode": preflight.get("mode") == "preflight",
            "status": preflight.get("status") == "complete",
            "excluded": preflight.get("included_in_formal_metrics") is False,
            "script": preflight.get("script_sha256") == actual_script_sha,
            "protocol": preflight.get("protocol_sha256") == _sha256(args.protocol),
            "task": preflight.get("pairs", [{}])[0].get("task_id")
            == protocol["preflight"]["task_id"],
            "state": preflight.get("pairs", [{}])[0].get("initial_state_index")
            == protocol["preflight"]["initial_state_index"],
            "seed": preflight.get("pairs", [{}])[0].get("seed") == protocol["preflight"]["seed"],
            "comparison": preflight.get("pairs", [{}])[0].get("comparison", {}).get("all_equal")
            is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"replan preflight mismatch: {checks}")
        preflight_receipt = {
            "path": str(args.preflight_manifest.resolve()),
            "sha256": _sha256(args.preflight_manifest),
            "checks": checks,
        }
    else:
        if args.preflight_manifest is not None:
            raise ValueError("only formal mode consumes a preflight manifest")
        preflight_receipt = None

    os.environ.setdefault("MUJOCO_GL", "egl")
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
        if asset_tree[key] != native_protocol["assets"]["resolved_tree"][key]:
            raise RuntimeError(f"asset tree mismatch: {key}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    sources = {"init_states": {}, "bddl": {}}
    for task_id, task in enumerate(suite.tasks):
        sources["init_states"][str(task_id)] = _sha256(init_root / task.init_states_file)
        sources["bddl"][str(task_id)] = _sha256(bddl_root / task.bddl_file)
    if sources != native_protocol["assets"]["task_source_sha256"]:
        raise RuntimeError("task source digest mismatch")

    tasks = (
        [protocol["preflight"]["task_id"]]
        if args.mode == "preflight"
        else protocol["evaluation"]["task_ids"]
    )
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=tasks,
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
        raise RuntimeError("official checkpoint does not declare chunk_size=n_action_steps=50")
    policy_cfg.device = "cuda"
    policy_cfg.vlm_model_name = str(args.backbone.resolve())
    policy_cfg.pretrained_path = args.checkpoint.resolve()
    rename_map = native_protocol["evaluation"]["rename_map"]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    sink = BranchSink(args.output_dir / "action_trace.jsonl")
    recorder = TraceRecorder(sink, native_protocol["runtime_contract"])
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
    tracing_preprocessor = TracingInputProcessor(preprocessor, recorder)
    tracing_postprocessor = TracingProcessor(postprocessor, recorder)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    pairs: list[dict[str, Any]] = []
    started = time.time()
    fatal_error: dict[str, Any] | None = None
    try:
        pair_ordinal = 0
        for task_id in tasks:
            env = envs["libero_spatial"][task_id]
            _wrap_env(env, recorder)
            recorder.set_task(task_id, str(env.call("task_description")[0]))
            states = (
                [protocol["preflight"]["initial_state_index"]]
                if args.mode == "preflight"
                else protocol["evaluation"]["fixed_initial_state_indices"]
            )
            if len(env.envs[0]._init_states) <= max(states):
                raise RuntimeError("insufficient fixed initial states")
            for state_index in states:
                seed = (
                    protocol["preflight"]["seed"]
                    if args.mode == "preflight"
                    else protocol["evaluation"]["seed_base"] + state_index - 10
                )
                order = ["execute_50", "execute_10"]
                if pair_ordinal % 2:
                    order.reverse()
                pair_id = f"task{task_id:02d}-state{state_index:02d}"
                pair = {
                    "pair_id": pair_id,
                    "task_id": task_id,
                    "initial_state_index": state_index,
                    "seed": seed,
                    "order": order,
                    "branches": {},
                }
                branch_raw: dict[str, dict[str, Any]] = {}
                for branch in order:
                    horizon = protocol["evaluation"]["branches"][branch]["execution_horizon"]
                    sink.branch = branch
                    sink.pair_id = pair_id
                    start_event = len(sink.events)
                    branch_started = time.time()
                    try:
                        set_seed(seed)
                        policy.config.n_action_steps = horizon
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
                        branch_events = sink.events[start_event:]
                        raw = [item for item in branch_events if item["event"] == "raw_policy_chunk"]
                        env_steps = sum(item["event"] == "env_step_action" for item in branch_events)
                        if not raw or any(item["shape"] != [1, 50, 7] for item in raw):
                            raise RuntimeError("raw prediction chunk shape mismatch")
                        branch_raw[branch] = raw[0]
                        pair["branches"][branch] = {
                            "execution_horizon": horizon,
                            "success": bool(result["successes"][0]),
                            "sum_reward": float(result["sum_rewards"][0]),
                            "max_reward": float(result["max_rewards"][0]),
                            "env_steps": env_steps,
                            "raw_chunks": len(raw),
                            "raw_chunks_per_step": len(raw) / env_steps,
                            "chunk_boundary_steps": [item["step"] for item in raw],
                            "elapsed_seconds": time.time() - branch_started,
                            "crashed": False,
                        }
                    except Exception as error:
                        pair["branches"][branch] = {
                            "execution_horizon": horizon,
                            "success": False,
                            "crashed": True,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "elapsed_seconds": time.time() - branch_started,
                        }
                if set(branch_raw) == {"execute_50", "execute_10"}:
                    left = branch_raw["execute_50"]
                    right = branch_raw["execute_10"]
                    state_equal = (
                        left["input_context"]["state"]["sha256"]
                        == right["input_context"]["state"]["sha256"]
                    )
                    camera_equal = left["input_context"]["camera_frames"] == right["input_context"][
                        "camera_frames"
                    ]
                    first_ten_equal = _first_ten_hash(left) == _first_ten_hash(right)
                    pair["comparison"] = {
                        "state_hash_equal": state_equal,
                        "camera_hashes_equal": camera_equal,
                        "first_10_predicted_actions_equal": first_ten_equal,
                        "first_10_sha256": _first_ten_hash(left),
                        "all_equal": state_equal and camera_equal and first_ten_equal,
                    }
                else:
                    pair["comparison"] = {"all_equal": False, "reason": "missing raw chunk"}
                pairs.append(pair)
                if not pair["comparison"]["all_equal"]:
                    raise RuntimeError(f"paired treatment isolation failed: {pair_id}")
                pair_ordinal += 1
    except Exception as error:
        fatal_error = {
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        close_envs(envs)
        sink.close()

    trace_path = args.output_dir / "action_trace.jsonl"
    expected_states = (
        [protocol["preflight"]["initial_state_index"]]
        if args.mode == "preflight"
        else protocol["evaluation"]["fixed_initial_state_indices"]
    )
    if fatal_error is not None:
        existing = {(pair["task_id"], pair["initial_state_index"]): pair for pair in pairs}
        for task_id in tasks:
            for state_index in expected_states:
                key = (task_id, state_index)
                seed = (
                    protocol["preflight"]["seed"]
                    if args.mode == "preflight"
                    else protocol["evaluation"]["seed_base"] + state_index - 10
                )
                pair = existing.get(key)
                if pair is None:
                    pair = {
                        "pair_id": f"task{task_id:02d}-state{state_index:02d}",
                        "task_id": task_id,
                        "initial_state_index": state_index,
                        "seed": seed,
                        "order": [],
                        "branches": {},
                        "comparison": {"all_equal": False, "reason": "not run after fatal failure"},
                    }
                    pairs.append(pair)
                for branch in ("execute_50", "execute_10"):
                    pair["branches"].setdefault(
                        branch,
                        {
                            "execution_horizon": protocol["evaluation"]["branches"][branch][
                                "execution_horizon"
                            ],
                            "success": False,
                            "crashed": True,
                            "error_type": "NOT_RUN_AFTER_FATAL_FAILURE",
                            "elapsed_seconds": 0.0,
                        },
                    )
        pairs.sort(key=lambda pair: (tasks.index(pair["task_id"]), pair["initial_state_index"]))
    task_metrics: dict[str, Any] = {}
    for task_id in tasks:
        task_pairs = [pair for pair in pairs if pair["task_id"] == task_id]
        metrics: dict[str, Any] = {"pairs": len(task_pairs), "branches": {}}
        for branch in ("execute_50", "execute_10"):
            outcomes = [int(pair["branches"][branch]["success"]) for pair in task_pairs]
            successes = sum(outcomes)
            observed_chunk_rates = [
                pair["branches"][branch]["raw_chunks_per_step"]
                for pair in task_pairs
                if "raw_chunks_per_step" in pair["branches"][branch]
            ]
            observed_times = [
                pair["branches"][branch]["elapsed_seconds"]
                for pair in task_pairs
                if not pair["branches"][branch].get("crashed", False)
            ]
            metrics["branches"][branch] = {
                "episodes": len(outcomes),
                "successes": successes,
                "success_rate": successes / len(outcomes) if outcomes else None,
                "wilson_95": _wilson(successes, len(outcomes)),
                "observed_noncrashed_episodes": len(observed_times),
                "mean_raw_chunks_per_step": (
                    float(np.mean(observed_chunk_rates)) if observed_chunk_rates else None
                ),
                "mean_elapsed_seconds": float(np.mean(observed_times)) if observed_times else None,
            }
        differences = [
            int(pair["branches"]["execute_10"]["success"])
            - int(pair["branches"]["execute_50"]["success"])
            for pair in task_pairs
        ]
        metrics["paired_delta_10_minus_50"] = sum(differences) / len(differences)
        metrics["paired_differences"] = differences
        metrics["paired_bootstrap_95"] = _paired_bootstrap(
            differences,
            protocol["analysis"]["bootstrap_samples"],
            protocol["analysis"]["bootstrap_seed"] + task_id,
        )
        task_metrics[str(task_id)] = metrics

    manifest = {
        "schema": "sentinel-libero-replan-ablation-v1",
        "mode": args.mode,
        "status": "complete" if fatal_error is None else "failed",
        "included_in_formal_metrics": args.mode == "formal",
        "script_sha256": actual_script_sha,
        "protocol_sha256": _sha256(args.protocol),
        "native_bindings": protocol["native_bindings"],
        "preflight_evidence": preflight_receipt,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **{
                package: importlib.metadata.version(package)
                for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words")
            },
        },
        "effective_environment": asdict(env_cfg),
        "artifact_receipts": {
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "backbone_sha256": bound["backbone_sha256"],
            "backbone_tree": bound["backbone_tree"],
            "asset_tree": asset_tree,
            "task_source_sha256": sources,
        },
        "pairs": pairs,
        "episodes_accounted": len(pairs) * 2,
        "metrics": task_metrics,
        "trace": {
            "path": trace_path.name,
            "sha256": _sha256(trace_path),
            "bytes": trace_path.stat().st_size,
            "raw_chunks": recorder.raw_chunks,
            "env_step_actions": recorder.actions,
            "passthrough_comparisons": recorder.passthrough_comparisons,
            "max_postprocess_to_env_step_abs_diff": recorder.max_passthrough_abs_diff,
            "interventions": 0,
        },
        "timing": {"elapsed_seconds": time.time() - started},
        "failure": fatal_error,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "metrics": task_metrics}, indent=2))
    return 0 if fatal_error is None else 1


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
