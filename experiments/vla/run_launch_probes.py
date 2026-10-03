#!/usr/bin/env python3
"""Run finite no-motion launch probes through official SmolVLA/LIBERO APIs.

``prepare`` freezes the protocol before outcomes. ``run`` resets each declared
LIBERO task, executes the official environment and policy preprocessors, real
SmolVLA inference and the official postprocessor, but never calls ``env.step``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import time
from typing import Any


TASKS = list(range(10))
STATE_INDEX = 46
RESET_SEED = 44046
INFERENCE_SEED = 240046
CASES = (
    "unchanged", "camera_swap", "state_index_swap", "action_order_swap",
    "action_sign", "gripper_inversion", "chunk_offset",
)
STATE_NAMES = [
    "eef_x", "eef_y", "eef_z",
    "eef_axis_angle_x", "eef_axis_angle_y", "eef_axis_angle_z",
    "gripper_left", "gripper_right",
]
STATE_UNITS = ["m", "m", "m", "rad", "rad", "rad", "m", "m"]
NORMALIZED_STATE_UNITS = [f"unitless(normalized_from_{unit})" for unit in STATE_UNITS]
UPSTREAM_CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
POLICY_CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
SCHEMA = "sentinel-vla-gpu-launch-probes-v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _save(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"non-finite JSON value: {value}")))


def _read_protocol(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise RuntimeError("protocol must be a regular file no larger than 1 MiB")
        chunks: list[bytes] = []
        length = 0
        while True:
            block = os.read(descriptor, min(65536, 1024 * 1024 + 1 - length))
            if not block:
                break
            chunks.append(block)
            length += len(block)
            if length > 1024 * 1024:
                raise RuntimeError("protocol grew beyond 1 MiB while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _protocol(script: Path) -> dict[str, Any]:
    repository = script.parents[2]
    native_protocol_path = repository / "experiments/vla/libero_protocol.json"
    native_protocol = json.loads(native_protocol_path.read_text(encoding="utf-8"))
    return {
        "schema": SCHEMA,
        "state": "frozen_before_outcomes",
        "script_sha256": _sha_file(script),
        "product_sources": {
            name: _sha_file(repository / name)
            for name in ("src/sentinel_evc/launch_gate.py",
                         "src/sentinel_evc/launch_diagnosis.py",
                         "src/sentinel_evc/native_gateway.py",
                         "src/sentinel_evc/contracts.py",
                         "src/sentinel_evc/events.py",
                         "src/sentinel_evc/evidence.py",
                         "src/sentinel_evc/scenario.py")
        },
        "native_protocol_sha256": _sha_file(native_protocol_path),
        "task_source_sha256": native_protocol["assets"]["task_source_sha256"],
        "checkpoint": {
            "repo_id": "lerobot/smolvla_libero",
            "revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
            "files": {
                "config.json": "sha256:5c9f3ba9f5f37ea7024c9501b0b20b1941f232989c2167853cb46d9071a70dd7",
                "model.safetensors": "sha256:9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8",
                "policy_preprocessor.json": "sha256:122ec5106602b1bf129f49690d05ab2f49748a0ac6119de55ea0677d4e90d248",
                "policy_postprocessor.json": "sha256:3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a",
                "policy_preprocessor_step_5_normalizer_processor.safetensors": "sha256:b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "sha256:b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
            },
        },
        "backbone": {
            "repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "files": {
                "config.json": "sha256:ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17",
                "model.safetensors": "sha256:b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
                "tokenizer.json": "sha256:5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
                "preprocessor_config.json": "sha256:149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374",
                "tokenizer_config.json": "sha256:dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913",
                "special_tokens_map.json": "sha256:2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60",
                "processor_config.json": "sha256:f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c",
            },
        },
        "runtime": {
            "software": {"lerobot": "0.6.1", "hf-libero": "0.1.4",
                         "robosuite": "1.4.0", "mujoco": "3.8.1",
                         "num2words": "0.5.14", "numpy": "2.2.6",
                         "torch": "2.8.0+cu128", "transformers": "5.5.4"},
            "tasks": TASKS, "initial_state_index": STATE_INDEX,
            "reset_seed": RESET_SEED, "inference_seed": INFERENCE_SEED,
            "camera_keys": ["observation.images.image", "observation.images.image2"],
            "state_shape": [1, 8], "action_shape": [1, 7],
            "cases": list(CASES), "env_step_calls": 0,
        },
        "acceptance": {
            "unchanged": "PASS with byte-identical fresh inference",
            "faults": "every changed binding, cursor or final action is BLOCK",
            "writer": "zero downstream software calls for every BLOCK",
        },
        "scope": {
            "measured": "actual reset, official preprocessors, GPU model inference and official postprocessor",
            "excluded": ["env.step", "task success", "physics", "robot motion", "physical stop"],
        },
    }


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("protocol output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = _protocol(Path(__file__).resolve())
    _save(args.output, value)
    print(json.dumps({"prepared": True, "protocol_sha256": _sha_file(args.output)}, indent=2))


def _verify_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    raw = _read_protocol(args.protocol)
    found = _strict_json(raw)
    expected = _protocol(Path(__file__).resolve())
    if found != expected:
        raise RuntimeError("protocol or runner changed after the pre-outcome freeze")
    for root, group in ((args.checkpoint, found["checkpoint"]),
                        (args.backbone, found["backbone"])):
        for name, digest in group["files"].items():
            if _sha_file(root / name) != digest:
                raise RuntimeError(f"pinned model file mismatch: {root / name}")
    for package, version in found["runtime"]["software"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"software mismatch: {package}")
    return found, raw


def _clone(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return copy.deepcopy(value)


def _cpu_array(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.asarray(value).copy()


def _input_identity(observation: dict[str, Any], instruction: str,
                    capture_tensor_identity: Any) -> str:
    state = _cpu_array(observation["observation.state"])
    value = {
        "instruction": instruction,
        "cameras": {key: capture_tensor_identity(observation[key])
                    for key in UPSTREAM_CAMERA_KEYS},
        "state_dtype": str(state.dtype), "state_shape": list(state.shape),
        "state_bytes": state.tobytes(order="C").hex(),
    }
    return _sha_bytes(_json_bytes(value))


def _upstream_record(observation: dict[str, Any], instruction: str,
                     capture_tensor_identity: Any) -> dict[str, Any]:
    state = _cpu_array(observation["observation.state"])
    return {
        "instruction_sha256": _sha_bytes(instruction.encode("utf-8")),
        "cameras": {key: capture_tensor_identity(observation[key])
                    for key in UPSTREAM_CAMERA_KEYS},
        "state": {"names": list(STATE_NAMES), "units": list(STATE_UNITS),
                  "dtype": str(state.dtype), "shape": list(state.shape),
                  "bytes_hex": state.tobytes(order="C").hex()},
    }


def _adapter_digest(script: Path, mode: str, model_files: dict[str, str]) -> str:
    return _sha_bytes(_json_bytes({"script": _sha_file(script), "mode": mode,
                                    "models": model_files}))


def _run_inference(policy: Any, preprocessor: Any, postprocessor: Any,
                   observation: dict[str, Any], seed: int) -> tuple[Any, dict[str, Any], int]:
    import torch
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    policy.reset()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        batch = preprocessor(observation)
        consumed = {key: _clone(batch[key])
                    for key in (*POLICY_CAMERA_KEYS, "observation.state")}
        selected = policy.select_action(batch)
        final = postprocessor(selected)
    torch.cuda.synchronize()
    return _cpu_array(final), consumed, time.perf_counter_ns() - started


def _probe(capture_probe: Any, *, probe_id: str, input_hash: str,
           consumed: dict[str, Any], action: Any, action_index: int = 0) -> dict[str, Any]:
    state = _cpu_array(consumed["observation.state"])
    if list(state.shape) != [1, 8]:
        raise RuntimeError("official environment state shape changed")
    final = _cpu_array(action)
    if list(final.shape) != [1, 7] or str(final.dtype) != "float32":
        raise RuntimeError("official postprocessed action contract changed")
    return capture_probe(
        probe_id=probe_id, input_hash=input_hash,
        cameras={key: consumed[key] for key in POLICY_CAMERA_KEYS},
        state_names=list(STATE_NAMES), state_units=list(NORMALIZED_STATE_UNITS),
        state_values=[float(item) for item in state[0]], chunk_id=0,
        action_index=action_index, action=final,
    )


def _pack(freeze_probe_pack: Any, suite_id: str, mode: str, probes: list[dict[str, Any]],
          provenance: dict[str, Any], model_files: dict[str, str]) -> bytes:
    return freeze_probe_pack({
        "schema": "sentinel-vla-probe-pack-v1", "suite_id": suite_id,
        "adapter_digest": _adapter_digest(Path(__file__).resolve(), mode, model_files),
        "provenance": {**provenance, "adapter_mode": mode}, "probes": probes,
    })


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("run output already exists")
    args.output.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    protocol, protocol_raw = _verify_protocol(args)
    (args.output / "protocol.json").write_bytes(protocol_raw)

    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from sentinel_evc.launch_gate import (
        QualifiedNativeWriter, build_qualification_capsule, capture_probe,
        capture_tensor_identity, freeze_probe_pack, qualify,
        reproduce_qualification_capsule,
    )
    from sentinel_evc.native_gateway import NativeGatewayDenied

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    task_sources = {
        "init_states": {str(index): _sha_file(init_root / task.init_states_file)[7:]
                        for index, task in enumerate(suite.tasks)},
        "bddl": {str(index): _sha_file(bddl_root / task.bddl_file)[7:]
                 for index, task in enumerate(suite.tasks)},
    }
    if task_sources != protocol["task_source_sha256"]:
        raise RuntimeError("LIBERO task or fixed-state sources differ from the frozen protocol")
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=TASKS, fps=20, init_states=True,
                        hard_reset=True, control_mode="relative", max_parallel_tasks=1,
                        observation_height=360, observation_width=360)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    policy_cfg.device = "cuda"
    policy_cfg.vlm_model_name = str(args.backbone.resolve())
    policy_cfg.pretrained_path = args.checkpoint.resolve()
    rename_map = {"observation.images.image": "observation.images.camera1",
                  "observation.images.image2": "observation.images.camera2"}
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(args.checkpoint.resolve()),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone.resolve())},
        })
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    model_files = {
        "checkpoint_model": _sha_file(args.checkpoint / "model.safetensors"),
        "backbone_model": _sha_file(args.backbone / "model.safetensors"),
    }
    provenance = {
        "kind": "fresh official SmolVLA/LIBERO no-motion inference",
        "checkpoint_revision": protocol["checkpoint"]["revision"],
        "backbone_revision": protocol["backbone"]["revision"],
        "model_files": model_files,
        "state_index": STATE_INDEX, "reset_seed": RESET_SEED,
        "env_step_calls": 0,
    }
    probes: dict[str, list[dict[str, Any]]] = {case: [] for case in CASES}
    timings: dict[str, list[int]] = {case: [] for case in ("reference", "unchanged", "camera_swap", "state_index_swap")}
    input_rows: list[dict[str, Any]] = []
    try:
        for task_id in TASKS:
            env = envs["libero_spatial"][task_id]
            underlying = env.envs[0]
            if len(underlying._init_states) <= STATE_INDEX:
                raise RuntimeError(f"task {task_id} lacks initial state {STATE_INDEX}")
            underlying.init_state_id = STATE_INDEX
            set_seed(RESET_SEED)
            observation, _ = env.reset(seed=[RESET_SEED], options={NEW_ROLLOUT_OPTION: True})
            actual_state = int(underlying.init_state_id - underlying._reset_stride)
            if actual_state != STATE_INDEX:
                raise RuntimeError(f"task {task_id} reset used state {actual_state}")
            processed = preprocess_observation(observation)
            instruction = str(env.call("task_description")[0])
            processed["task"] = [instruction]
            consumed = env_preprocessor(processed)
            input_hash = _input_identity(consumed, instruction, capture_tensor_identity)
            input_rows.append({"task_id": task_id, "input_hash": input_hash,
                               "upstream_observation": _upstream_record(
                                   consumed, instruction, capture_tensor_identity)})

            reference_action, reference_consumed, elapsed = _run_inference(
                policy, preprocessor, postprocessor, copy.deepcopy(consumed), INFERENCE_SEED + task_id)
            timings["reference"].append(elapsed)
            reference_probe = _probe(capture_probe, probe_id=f"task{task_id:02d}-state46",
                                     input_hash=input_hash, consumed=reference_consumed,
                                     action=reference_action)

            unchanged_action, unchanged_consumed, elapsed = _run_inference(
                policy, preprocessor, postprocessor, copy.deepcopy(consumed), INFERENCE_SEED + task_id)
            timings["unchanged"].append(elapsed)
            probes["unchanged"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=unchanged_consumed, action=unchanged_action))

            camera_consumed = {key: _clone(value) for key, value in consumed.items()}
            first, second = "observation.images.image", "observation.images.image2"
            camera_consumed[first], camera_consumed[second] = (
                camera_consumed[second], camera_consumed[first])
            camera_action, camera_policy_consumed, elapsed = _run_inference(
                policy, preprocessor, postprocessor, copy.deepcopy(camera_consumed),
                INFERENCE_SEED + task_id)
            timings["camera_swap"].append(elapsed)
            probes["camera_swap"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=camera_policy_consumed, action=camera_action))

            state_consumed = {key: _clone(value) for key, value in consumed.items()}
            state_consumed["observation.state"][:, [0, 1]] = state_consumed["observation.state"][:, [1, 0]]
            state_action, state_policy_consumed, elapsed = _run_inference(
                policy, preprocessor, postprocessor, copy.deepcopy(state_consumed),
                INFERENCE_SEED + task_id)
            timings["state_index_swap"].append(elapsed)
            probes["state_index_swap"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=state_policy_consumed, action=state_action))

            probes.setdefault("reference", []).append(reference_probe)
            base = reference_action.copy()
            changed = base.copy(); changed[:, [0, 1]] = changed[:, [1, 0]]
            probes["action_order_swap"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=reference_consumed, action=changed))
            changed = base.copy(); changed[:, 0] *= -1
            probes["action_sign"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=reference_consumed, action=changed))
            changed = base.copy(); changed[:, 6] *= -1
            probes["gripper_inversion"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=reference_consumed, action=changed))
            probes["chunk_offset"].append(_probe(
                capture_probe, probe_id=f"task{task_id:02d}-state46", input_hash=input_hash,
                consumed=reference_consumed, action=base, action_index=1))
    finally:
        for value in envs.values():
            if hasattr(value, "close"):
                value.close()

    suite_id = _sha_bytes(_json_bytes(input_rows))
    reference = _pack(freeze_probe_pack, suite_id, "reference", probes["reference"], provenance, model_files)
    packs_dir = args.output / "packs"
    capsules_dir = args.output / "capsules"
    packs_dir.mkdir(); capsules_dir.mkdir()
    (packs_dir / "reference.json").write_bytes(reference)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        candidate = _pack(freeze_probe_pack, suite_id, case, probes[case], provenance, model_files)
        (packs_dir / f"{case}.json").write_bytes(candidate)
        report = qualify(reference, candidate)
        expected = "PASS" if case == "unchanged" else "BLOCK"
        if report["verdict"] != expected:
            raise RuntimeError(f"{case}: expected {expected}, got {report['verdict']}")
        downstream: list[bytes] = []
        entered: list[bool] = []
        try:
            writer = QualifiedNativeWriter(
                reference, candidate, lambda raw=candidate: json.loads(raw)["adapter_digest"],
                lambda request: downstream.append(request) or "software-receipt")
            writer(b"no-motion-request", lambda: entered.append(True))
            writer_result = "DISPATCHED"
        except NativeGatewayDenied as exc:
            writer_result = exc.code
        capsule_root = capsules_dir / case
        capsule_report, receipt = build_qualification_capsule(
            reference, candidate, capsule_root, f"gpu-launch-{case.replace('_', '-')}")
        reproduction = reproduce_qualification_capsule(
            receipt["bundle_dir"], receipt["public_key"],
            f"gpu-launch-{case.replace('_', '-')}")
        if (report["sources"]["reference"]["pack_sha256"] != _sha_bytes(reference)
                or report["sources"]["candidate"]["pack_sha256"] != _sha_bytes(candidate)):
            raise RuntimeError(f"{case} report source hashes do not bind the emitted packs")
        manifest_bytes = (Path(receipt["bundle_dir"]) / "manifest.json").read_bytes()
        if capsule_report != report or reproduction["reproduction"] != "PASS":
            raise RuntimeError(f"{case} capsule reproduction failed")
        if expected == "BLOCK" and (downstream or entered):
            raise RuntimeError(f"{case} reached writer entry")
        if expected == "PASS" and (writer_result != "DISPATCHED"
                                    or len(entered) != 1 or len(downstream) != 1):
            raise RuntimeError(f"{case} did not complete the positive writer path")
        rows.append({"case": case, "verdict": report["verdict"],
                     "sources": report["sources"],
                     "capsule_manifest_sha256": _sha_bytes(manifest_bytes),
                     "first_issue": report["issues"][0] if report["issues"] else None,
                     "diagnosis": report.get("diagnosis"), "writer_result": writer_result,
                     "entry_acknowledgements": len(entered), "downstream_calls": len(downstream),
                     "capsule_reproduction": reproduction})

    result = {
        "schema": SCHEMA, "protocol_sha256": _sha_bytes(protocol_raw),
        "product_sources": protocol["product_sources"],
        "suite_id": suite_id, "probe_count": len(TASKS), "cases": rows,
        "upstream_inputs": input_rows,
        "acceptance": {
            "unchanged_pass": rows[0]["verdict"] == "PASS",
            "unchanged_entry_acknowledgements": rows[0]["entry_acknowledgements"],
            "unchanged_downstream_calls": rows[0]["downstream_calls"],
            "faults_blocked": sum(row["verdict"] == "BLOCK" for row in rows[1:]),
            "fault_count": len(rows) - 1,
            "blocked_downstream_calls": sum(row["downstream_calls"] for row in rows[1:]),
            "env_step_calls": 0,
        },
        "inference_ns": {
            key: {"count": len(values), "mean": round(sum(values) / len(values)),
                  "min": min(values), "max": max(values)}
            for key, values in timings.items()
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in protocol["runtime"]["software"]},
            "additional_packages": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "huggingface-hub", "torch")
            },
            "models": model_files,
            "task_sources": task_sources,
            "libero_paths": {key: str(get_libero_path(key))
                             for key in ("assets", "bddl_files", "init_states")},
        },
        "scope": protocol["scope"],
    }
    if result["acceptance"] != {"unchanged_pass": True,
                                 "unchanged_entry_acknowledgements": 1,
                                 "unchanged_downstream_calls": 1,
                                 "faults_blocked": 6,
                                 "fault_count": 6, "blocked_downstream_calls": 0,
                                 "env_step_calls": 0}:
        raise RuntimeError("GPU launch-probe acceptance failed")
    _save(args.output / "result.json", result)
    print(json.dumps({"complete": True, "probe_count": len(TASKS),
                      "faults_blocked": 6, "env_step_calls": 0,
                      "result": str(args.output / 'result.json')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--backbone", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
