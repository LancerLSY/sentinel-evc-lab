#!/usr/bin/env python3
"""Capture and compare frozen no-policy LIBERO reset states across MuJoCo versions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


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


def _array_digest(array: Any) -> str:
    import numpy as np

    value = np.asarray(array)
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(descriptor + b"\0" + value.tobytes(order="C")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _verify_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"output directory must be empty: {path}")


def _verify_version(package: str, expected: str) -> str:
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"{package} version mismatch: expected {expected}, found {actual}")
    return actual


def _artifact_receipt(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _body_id(model: Any, name: str) -> int:
    if hasattr(model, "body_name2id"):
        return int(model.body_name2id(name))
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise RuntimeError(f"body not found: {name}")
    return body_id


def _tilt_from_wxyz(quaternion: Any) -> float:
    import numpy as np

    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(float(np.clip(local_z_world_z, -1.0, 1.0))))


def _capture_pose(sim: Any, name: str) -> dict[str, Any]:
    import numpy as np

    body_id = _body_id(sim.model, name)
    position = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy()
    quaternion = np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).copy()
    return {
        "body_id": body_id,
        "body_name": name,
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "tilt_rad": _tilt_from_wxyz(quaternion),
    }


def _resolve_body_name(body_names: list[str], token: str) -> str:
    exact = f"{token}_main"
    if exact in body_names:
        return exact
    matches = [name for name in body_names if token in name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body containing {token!r}, found {matches}")
    return matches[0]


def _capture_one(
    env: Any,
    task_id: int,
    state_index: int,
    seed: int,
    output_dir: Path,
    body_tokens: dict[str, str],
) -> dict[str, Any]:
    import numpy as np

    if not hasattr(env, "envs") or len(env.envs) != 1:
        raise RuntimeError("audit requires exactly one synchronous environment")
    wrapper = env.envs[0]
    wrapper.init_state_id = state_index
    if int(wrapper.init_state_id) != state_index:
        raise RuntimeError("failed to assign the frozen initial-state index")

    observation, _ = env.reset(seed=[seed])
    actual_index = int(wrapper.init_state_id - wrapper._reset_stride)
    actual_seed = int(wrapper.np_random_seed)
    if actual_index != state_index:
        raise RuntimeError(f"reset index mismatch: requested {state_index}, used {actual_index}")
    if actual_seed != seed:
        raise RuntimeError(f"reset seed mismatch: requested {seed}, used {actual_seed}")

    pixels = observation.get("pixels")
    if not isinstance(pixels, dict) or sorted(pixels) != ["image", "image2"]:
        raise RuntimeError(f"unexpected raw camera keys: {sorted(pixels) if isinstance(pixels, dict) else pixels}")
    camera_agent = np.asarray(pixels["image"])
    camera_wrist = np.asarray(pixels["image2"])
    if camera_agent.shape != (1, 360, 360, 3) or camera_wrist.shape != (1, 360, 360, 3):
        raise RuntimeError(
            f"unexpected camera shapes: agent={camera_agent.shape}, wrist={camera_wrist.shape}"
        )
    camera_agent = camera_agent[0].copy()
    camera_wrist = camera_wrist[0].copy()
    if camera_agent.dtype != np.uint8 or camera_wrist.dtype != np.uint8:
        raise RuntimeError("raw reset cameras must be uint8 RGB")

    sim = wrapper._env.sim
    body_names = [str(name) for name in sim.model.body_names]
    bowl_name = _resolve_body_name(body_names, body_tokens["bowl"])
    ramekin_name = _resolve_body_name(body_names, body_tokens["ramekin"])
    bowl_pose = _capture_pose(sim, bowl_name)
    ramekin_pose = _capture_pose(sim, ramekin_name)
    relative = (
        np.asarray(bowl_pose["position_m"], dtype=np.float64)
        - np.asarray(ramekin_pose["position_m"], dtype=np.float64)
    )

    state_arrays = {
        "qpos": np.asarray(sim.data.qpos).copy(),
        "qvel": np.asarray(sim.data.qvel).copy(),
        "act": np.asarray(sim.data.act).copy(),
        "ctrl": np.asarray(sim.data.ctrl).copy(),
        "time": np.asarray([sim.data.time], dtype=np.float64),
    }
    stem = f"task_{task_id:02d}_state_{state_index:02d}"
    camera_path = output_dir / "cameras" / f"{stem}.npz"
    state_path = output_dir / "states" / f"{stem}.npz"
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(camera_path, agentview_image=camera_agent, robot0_eye_in_hand_image=camera_wrist)
    np.savez(state_path, **state_arrays)

    return {
        "task_id": task_id,
        "initial_state_index": state_index,
        "requested_seed": seed,
        "actual_initial_state_index": actual_index,
        "actual_seed": actual_seed,
        "pre_action": True,
        "policy_loaded": False,
        "actions_executed": 0,
        "camera_artifact": _artifact_receipt(camera_path, output_dir),
        "state_artifact": _artifact_receipt(state_path, output_dir),
        "camera_arrays": {
            "agentview_image": {
                "shape": list(camera_agent.shape),
                "dtype": str(camera_agent.dtype),
                "sha256": _array_digest(camera_agent),
            },
            "robot0_eye_in_hand_image": {
                "shape": list(camera_wrist.shape),
                "dtype": str(camera_wrist.dtype),
                "sha256": _array_digest(camera_wrist),
            },
        },
        "sim_state_arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_digest(value)}
            for key, value in state_arrays.items()
        },
        "objects": {
            "bowl": bowl_pose,
            "ramekin": ramekin_pose,
            "bowl_minus_ramekin_position_m": relative.tolist(),
        },
    }


def _verify_preflight(
    path: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
    version_label: str,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema") != "sentinel-libero-reset-audit-capture-v1":
        raise RuntimeError(f"invalid preflight schema: {path}")
    for key, expected in {
        "status": "complete",
        "mode": "preflight",
        "included_in_final_analysis": False,
        "version_label": version_label,
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "reset_count": 1,
    }.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"preflight {key} mismatch in {path}")
    if manifest["software"]["mujoco"] != protocol["versions"][version_label]:
        raise RuntimeError(f"preflight MuJoCo mismatch in {path}")
    if manifest.get("native_binding") != protocol["native_binding"]:
        raise RuntimeError(f"preflight native-protocol binding mismatch in {path}")
    if manifest.get("environment") != protocol["environment"]:
        raise RuntimeError(f"preflight environment mismatch in {path}")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if manifest["assets"]["resolved_tree"].get(key) != protocol["assets"]["resolved_tree"][key]:
            raise RuntimeError(f"preflight asset {key} mismatch in {path}")
    records = manifest.get("records", [])
    if len(records) != 1:
        raise RuntimeError(f"preflight is not the frozen task-5/state-47 capture: {path}")
    record = records[0]
    expected_reset = protocol["preflight"]
    expected_values = {
        "task_id": expected_reset["task_id"],
        "initial_state_index": expected_reset["initial_state_index"],
        "actual_initial_state_index": expected_reset["initial_state_index"],
        "requested_seed": expected_reset["seed"],
        "actual_seed": expected_reset["seed"],
        "pre_action": True,
        "policy_loaded": False,
        "actions_executed": 0,
    }
    for key, expected in expected_values.items():
        if record.get(key) != expected:
            raise RuntimeError(f"preflight record {key} mismatch in {path}")
    for artifact in manifest.get("output_artifacts", []):
        artifact_path = path.parent / artifact["path"]
        if not artifact_path.is_file() or artifact_path.stat().st_size != artifact["bytes"]:
            raise RuntimeError(f"preflight artifact missing or wrong size: {artifact_path}")
        if _sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"preflight artifact digest mismatch: {artifact_path}")
    if len(manifest.get("output_artifacts", [])) != 3:
        raise RuntimeError(f"preflight output artifact count mismatch in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "version_label": version_label,
        "mujoco": manifest["software"]["mujoco"],
        "task_id": record["task_id"],
        "initial_state_index": record["actual_initial_state_index"],
        "seed": record["actual_seed"],
    }


def _capture(args: argparse.Namespace, protocol: dict[str, Any], script_sha: str) -> None:
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")

    expected_version = protocol["versions"][args.version_label]
    software = protocol["software"]
    installed = {
        "python": platform.python_version(),
        "lerobot": _verify_version("lerobot", software["lerobot"]),
        "hf-libero": _verify_version("hf-libero", software["hf-libero"]),
        "robosuite": _verify_version("robosuite", software["robosuite"]),
        "mujoco": _verify_version("mujoco", expected_version),
    }
    if not installed["python"].startswith(software["python_prefix"]):
        raise RuntimeError(
            f"Python version mismatch: expected prefix {software['python_prefix']}, found {installed['python']}"
        )

    import mujoco
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    protocol_sha = _sha256(args.protocol)
    assets_root = Path(get_libero_path("assets"))
    asset_identity = _tree_identity(assets_root, "mixed_site_packages_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if asset_identity[key] != protocol["assets"]["resolved_tree"][key]:
            raise RuntimeError(f"asset tree mismatch for {key}")

    if _sha256(args.native_protocol) != protocol["native_binding"]["libero_protocol_sha256"]:
        raise RuntimeError("native LIBERO protocol digest mismatch")

    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    for task_id_string, task_binding in protocol["tasks"].items():
        task_id = int(task_id_string)
        task = task_suite.tasks[task_id]
        if task.language != task_binding["language"]:
            raise RuntimeError(f"task {task_id} language mismatch")
        if task.init_states_file != task_binding["init_file"] or task.bddl_file != task_binding["bddl_file"]:
            raise RuntimeError(f"task {task_id} source filename mismatch")
        if _sha256(init_root / task.init_states_file) != task_binding["init_sha256"]:
            raise RuntimeError(f"task {task_id} init-state digest mismatch")
        if _sha256(bddl_root / task.bddl_file) != task_binding["bddl_sha256"]:
            raise RuntimeError(f"task {task_id} BDDL digest mismatch")

    preflight_evidence: dict[str, Any] | None = None
    if args.mode == "preflight":
        task_ids = [int(protocol["preflight"]["task_id"])]
        state_indices = [int(protocol["preflight"]["initial_state_index"])]
    else:
        task_ids = [int(value) for value in protocol["final"]["task_ids"]]
        state_indices = [int(value) for value in protocol["final"]["initial_state_indices"]]
        current_preflight = _verify_preflight(
            args.preflight_current,
            protocol,
            protocol_sha,
            script_sha,
            "current",
        )
        legacy_preflight = _verify_preflight(
            args.preflight_legacy,
            protocol,
            protocol_sha,
            script_sha,
            "legacy",
        )
        preflight_evidence = {
            "current": current_preflight,
            "legacy": legacy_preflight,
        }

    env_cfg = LiberoEnv(
        task=protocol["environment"]["suite"],
        task_ids=task_ids,
        fps=protocol["environment"]["fps"],
        init_states=True,
        hard_reset=True,
        control_mode=protocol["environment"]["control_mode"],
        max_parallel_tasks=1,
        observation_height=protocol["environment"]["observation_height"],
        observation_width=protocol["environment"]["observation_width"],
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    records: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            env = envs[protocol["environment"]["suite"]][task_id]
            if len(env.envs[0]._init_states) <= max(state_indices):
                raise RuntimeError(f"task {task_id} lacks frozen state {max(state_indices)}")
            for state_index in state_indices:
                seed = int(protocol["preflight"]["seed"]) if args.mode == "preflight" else int(
                    protocol["final"]["seed_base"] + state_index
                )
                records.append(
                    _capture_one(
                        env,
                        task_id,
                        state_index,
                        seed,
                        args.output_dir,
                        protocol["tasks"][str(task_id)]["body_tokens"],
                    )
                )
    finally:
        for task_envs in envs.values():
            for env in task_envs.values():
                env.close()

    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    output_artifacts = [_artifact_receipt(path, args.output_dir) for path in sorted(args.output_dir.rglob("*")) if path.is_file()]
    manifest = {
        "schema": "sentinel-libero-reset-audit-capture-v1",
        "status": "complete",
        "mode": args.mode,
        "included_in_final_analysis": args.mode == "final",
        "version_label": args.version_label,
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "native_binding": protocol["native_binding"],
        "software": {
            **installed,
            "mujoco_module": str(Path(mujoco.__file__).resolve()),
            "platform": platform.platform(),
        },
        "environment": protocol["environment"],
        "assets": {"resolved_tree": asset_identity},
        "preflight_evidence": preflight_evidence,
        "reset_count": len(records),
        "records": records,
        "output_artifacts": output_artifacts,
        "scope": protocol["claims"]["scope"],
    }
    _write_json(args.output_dir / "manifest.json", manifest)


def _load_artifact_arrays(manifest_path: Path, record: dict[str, Any], key: str) -> dict[str, Any]:
    import numpy as np

    path = manifest_path.parent / record[key]["path"]
    if _sha256(path) != record[key]["sha256"]:
        raise RuntimeError(f"artifact digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _vector_metrics(left: Any, right: Any) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"comparison shape mismatch: {a.shape} != {b.shape}")
    delta = b - a
    return {
        "shape": list(a.shape),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "l2": float(np.linalg.norm(delta)),
        "exact_equal": bool(np.array_equal(a, b)),
    }


def _quat_angle(left: Any, right: Any) -> float:
    import numpy as np

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def _compare(args: argparse.Namespace, protocol: dict[str, Any], script_sha: str) -> None:
    import numpy as np

    current = _load_json(args.current_manifest)
    legacy = _load_json(args.legacy_manifest)
    protocol_sha = _sha256(args.protocol)
    for path, manifest, version_label in (
        (args.current_manifest, current, "current"),
        (args.legacy_manifest, legacy, "legacy"),
    ):
        if manifest.get("schema") != "sentinel-libero-reset-audit-capture-v1" or manifest.get("status") != "complete":
            raise RuntimeError(f"invalid capture manifest: {path}")
        if manifest.get("mode") != "final" or manifest.get("reset_count") != 20:
            raise RuntimeError(f"capture is not the frozen 20-reset final grid: {path}")
        if manifest.get("protocol_sha256") != protocol_sha or manifest.get("script_sha256") != script_sha:
            raise RuntimeError(f"capture source binding mismatch: {path}")
        if manifest["software"]["mujoco"] != protocol["versions"][version_label]:
            raise RuntimeError(f"capture MuJoCo version mismatch: {path}")

    current_records = {(r["task_id"], r["initial_state_index"]): r for r in current["records"]}
    legacy_records = {(r["task_id"], r["initial_state_index"]): r for r in legacy["records"]}
    expected = {(task_id, state) for task_id in protocol["final"]["task_ids"] for state in protocol["final"]["initial_state_indices"]}
    if set(current_records) != expected or set(legacy_records) != expected:
        raise RuntimeError("capture records do not equal the frozen task/state grid")

    paired: list[dict[str, Any]] = []
    for task_id, state_index in sorted(expected):
        current_record = current_records[(task_id, state_index)]
        legacy_record = legacy_records[(task_id, state_index)]
        for record in (current_record, legacy_record):
            if record["requested_seed"] != protocol["final"]["seed_base"] + state_index:
                raise RuntimeError("capture seed differs from frozen seed formula")
            if record["actual_seed"] != record["requested_seed"] or record["actual_initial_state_index"] != state_index:
                raise RuntimeError("capture reset assertion was not preserved")
        current_cameras = _load_artifact_arrays(args.current_manifest, current_record, "camera_artifact")
        legacy_cameras = _load_artifact_arrays(args.legacy_manifest, legacy_record, "camera_artifact")
        current_state = _load_artifact_arrays(args.current_manifest, current_record, "state_artifact")
        legacy_state = _load_artifact_arrays(args.legacy_manifest, legacy_record, "state_artifact")
        object_metrics: dict[str, Any] = {}
        for name in ("bowl", "ramekin"):
            a = current_record["objects"][name]
            b = legacy_record["objects"][name]
            object_metrics[name] = {
                "position": _vector_metrics(a["position_m"], b["position_m"]),
                "quaternion_angle_rad": _quat_angle(a["quaternion_wxyz"], b["quaternion_wxyz"]),
                "tilt_abs_diff_rad": abs(float(a["tilt_rad"]) - float(b["tilt_rad"])),
            }
        paired.append(
            {
                "task_id": task_id,
                "initial_state_index": state_index,
                "seed": current_record["actual_seed"],
                "cameras": {name: _vector_metrics(current_cameras[name], legacy_cameras[name]) for name in sorted(current_cameras)},
                "sim_state": {name: _vector_metrics(current_state[name], legacy_state[name]) for name in sorted(current_state)},
                "objects": object_metrics,
                "bowl_minus_ramekin_position": _vector_metrics(
                    current_record["objects"]["bowl_minus_ramekin_position_m"],
                    legacy_record["objects"]["bowl_minus_ramekin_position_m"],
                ),
            }
        )

    per_task: dict[str, Any] = {}
    for task_id in protocol["final"]["task_ids"]:
        rows = [row for row in paired if row["task_id"] == task_id]
        per_task[str(task_id)] = {
            "pairs": len(rows),
            "camera_changed_pairs": {
                camera: sum(not row["cameras"][camera]["exact_equal"] for row in rows)
                for camera in ("agentview_image", "robot0_eye_in_hand_image")
            },
            "max_bowl_position_delta_m": max(row["objects"]["bowl"]["position"]["l2"] for row in rows),
            "mean_bowl_position_delta_m": float(np.mean([row["objects"]["bowl"]["position"]["l2"] for row in rows])),
            "max_ramekin_position_delta_m": max(row["objects"]["ramekin"]["position"]["l2"] for row in rows),
            "max_relative_position_delta_m": max(row["bowl_minus_ramekin_position"]["l2"] for row in rows),
            "max_bowl_orientation_delta_rad": max(row["objects"]["bowl"]["quaternion_angle_rad"] for row in rows),
            "max_bowl_tilt_abs_diff_rad": max(row["objects"]["bowl"]["tilt_abs_diff_rad"] for row in rows),
        }

    paired_path = args.output_dir / "paired_records.json"
    _write_json(paired_path, paired)
    summary = {
        "schema": "sentinel-libero-reset-audit-comparison-v1",
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "current_manifest": {
            "path": str(args.current_manifest.resolve()),
            "sha256": _sha256(args.current_manifest),
        },
        "legacy_manifest": {
            "path": str(args.legacy_manifest.resolve()),
            "sha256": _sha256(args.legacy_manifest),
        },
        "paired_reset_count": len(paired),
        "per_task": per_task,
        "paired_records": _artifact_receipt(paired_path, args.output_dir),
        "interpretation_limits": protocol["claims"],
    }
    _write_json(args.output_dir / "comparison.json", summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "final", "compare"), required=True)
    parser.add_argument("--version-label", choices=("current", "legacy"))
    parser.add_argument("--libero-config", type=Path)
    parser.add_argument("--native-protocol", type=Path)
    parser.add_argument("--preflight-current", type=Path)
    parser.add_argument("--preflight-legacy", type=Path)
    parser.add_argument("--current-manifest", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.protocol = args.protocol.resolve()
    args.output_dir = args.output_dir.resolve()
    protocol = _load_json(args.protocol)
    if protocol.get("schema") != "sentinel-libero-reset-audit-protocol-v1" or protocol.get("state") != "frozen_before_final_run":
        raise RuntimeError("protocol must be the frozen reset-audit schema")
    script_sha = _sha256(Path(__file__).resolve())
    if script_sha != protocol["implementation"]["script_sha256"]:
        raise RuntimeError("runner digest differs from frozen protocol")
    _verify_empty_output(args.output_dir)

    if args.mode == "compare":
        if args.current_manifest is None or args.legacy_manifest is None:
            raise RuntimeError("compare mode requires both final capture manifests")
        _compare(args, protocol, script_sha)
    else:
        if args.version_label is None or args.libero_config is None or args.native_protocol is None:
            raise RuntimeError("capture modes require --version-label, --libero-config, and --native-protocol")
        if args.mode == "final" and (args.preflight_current is None or args.preflight_legacy is None):
            raise RuntimeError("final mode requires both frozen preflight manifests")
        _capture(args, protocol, script_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
