#!/usr/bin/env python3
"""Replay one frozen LIBERO failure from its recorded env.step actions.

This is a post-hoc diagnostic. It never loads or calls the policy and does not
add an episode to benchmark metrics. The source action trace, environment,
initial observation, every action, and generated artifacts are hash-bound in
the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_libero_closedloop import _array_digest, _jsonable, _sha256, _tree_identity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-protocol", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41021)
    return parser.parse_args()


def _load_source_actions(
    trace_path: Path, task_id: int, state_index: int
) -> tuple[list[np.ndarray], list[str], dict[str, Any]]:
    actions: list[np.ndarray] = []
    action_hashes: list[str] = []
    first_context: dict[str, Any] | None = None
    plan = reset = None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("task_id") != task_id or event.get("initial_state_index") != state_index:
            continue
        if event["event"] == "episode_plan":
            plan = event
        elif event["event"] == "episode_reset":
            reset = event
        elif event["event"] == "raw_policy_chunk" and event.get("step") == 0:
            first_context = event["input_context"]
        elif event["event"] == "env_step_action":
            action = np.asarray(event["values"], dtype=np.float32)
            if action.shape != (1, 7) or _array_digest(action) != event["sha256"]:
                raise RuntimeError(f"recorded action digest mismatch at step {event.get('step')}")
            if event.get("step") != len(actions):
                raise RuntimeError("recorded action steps are not contiguous")
            actions.append(action)
            action_hashes.append(event["sha256"])
    if plan is None or reset is None or first_context is None:
        raise RuntimeError("source trace is missing plan, reset, or first-chunk context")
    return actions, action_hashes, {"plan": plan, "reset": reset, "input": first_context}


def _body_names(sim: Any) -> list[str]:
    names = getattr(sim.model, "body_names", None)
    if names is not None:
        return [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    import mujoco

    model = getattr(sim.model, "_model", sim.model)
    return [
        name
        for body_id in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)) is not None
    ]


def _body_pose(sim: Any, name: str) -> dict[str, Any]:
    try:
        body_id = int(sim.model.body_name2id(name))
    except AttributeError:
        body_id = int(sim.model.body(name).id)
    return {
        "position": np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).tolist(),
    }


def _tracked_body_names(names: list[str]) -> dict[str, list[str]]:
    patterns = {
        "bowl": ("akita_black_bowl", "black_bowl"),
        "ramekin": ("ramekin",),
        "plate": ("plate",),
    }
    return {
        label: [name for name in names if any(token in name.lower() for token in tokens)]
        for label, tokens in patterns.items()
    }


def _pose_snapshot(underlying: Any, observation: dict[str, Any], tracked: dict[str, list[str]]) -> dict[str, Any]:
    sim = underlying._env.sim
    bodies = {
        label: {name: _body_pose(sim, name) for name in names}
        for label, names in tracked.items()
    }
    robot = observation["robot_state"]
    return {
        "bodies": bodies,
        "gripper": {
            "eef_position": np.asarray(robot["eef"]["pos"], dtype=np.float64).tolist(),
            "eef_quaternion_xyzw": np.asarray(robot["eef"]["quat"], dtype=np.float64).tolist(),
            "finger_qpos": np.asarray(robot["gripper"]["qpos"], dtype=np.float64).tolist(),
        },
    }


def _unbatch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _unbatch(item) for key, item in value.items()}
    return np.asarray(value)[0]


def _primary_position(snapshot: dict[str, Any], label: str) -> np.ndarray | None:
    poses = snapshot["bodies"].get(label, {})
    if not poses:
        return None
    return np.asarray(next(iter(poses.values()))["position"], dtype=np.float64)


def _annotated_frame(
    frame: np.ndarray,
    step: int,
    total_steps: int,
    reward: float,
    success: bool,
    snapshot: dict[str, Any],
    font_path: Path,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1280, 720), (12, 18, 28))
    camera = Image.fromarray(frame).resize((720, 720), Image.Resampling.LANCZOS)
    canvas.paste(camera, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(font_path), 32)
    body_font = ImageFont.truetype(str(font_path), 21)
    small_font = ImageFont.truetype(str(font_path), 18)
    x = 754
    draw.text((x, 30), "失败轨迹精确回放", font=title_font, fill=(246, 248, 252))
    draw.text((x, 74), "Exact Failure Replay", font=title_font, fill=(100, 190, 255))
    draw.line((x, 122, 1240, 122), fill=(64, 79, 99), width=2)
    lines = [
        "任务 / Task",
        "将黑碗从小烤碗上拿起并放到盘子上",
        "Pick up the black bowl on the ramekin",
        "and place it on the plate.",
        "",
        f"控制步 / Step       {step:03d} / {total_steps}",
        f"时间 / Time          {step / 20.0:5.2f} s",
        f"奖励 / Reward        {reward:.1f}",
        f"成功 / Success       {'YES' if success else 'NO'}",
    ]
    y = 146
    for line in lines:
        draw.text((x, y), line, font=body_font, fill=(225, 231, 239))
        y += 31 if line else 18

    bowl = _primary_position(snapshot, "bowl")
    ramekin = _primary_position(snapshot, "ramekin")
    plate = _primary_position(snapshot, "plate")
    gripper = np.asarray(snapshot["gripper"]["eef_position"], dtype=np.float64)
    draw.line((x, y + 4, 1240, y + 4), fill=(64, 79, 99), width=2)
    y += 24
    draw.text((x, y), "实测位姿 / Measured positions (m)", font=body_font, fill=(100, 190, 255))
    y += 34
    for label, value in (("Bowl", bowl), ("Ramekin", ramekin), ("Plate", plate), ("Gripper", gripper)):
        shown = "unavailable" if value is None else "[" + ", ".join(f"{v:+.3f}" for v in value) + "]"
        draw.text((x, y), f"{label:<9} {shown}", font=small_font, fill=(210, 218, 228))
        y += 27
    if bowl is not None:
        draw.text((x, y + 5), f"Bowl-gripper distance  {np.linalg.norm(bowl - gripper):.3f} m", font=small_font, fill=(210, 218, 228))
        y += 30
    if bowl is not None and plate is not None:
        draw.text((x, y + 5), f"Bowl-plate distance    {np.linalg.norm(bowl - plate):.3f} m", font=small_font, fill=(210, 218, 228))
        y += 30
    draw.rectangle((720, 650, 1280, 720), fill=(20, 31, 47))
    draw.text((x, 661), "原评估失败样本；不计入新基准回合", font=small_font, fill=(255, 199, 92))
    draw.text((x, 689), "Recorded failure; no new benchmark episode", font=small_font, fill=(255, 199, 92))
    return np.asarray(canvas)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> int:
    args = _parse_args()
    protocol = json.loads(args.native_protocol.read_text(encoding="utf-8"))
    native_manifest = json.loads(args.native_manifest.read_text(encoding="utf-8"))
    if _sha256(args.native_manifest) != "b5d8e982f50ee07452441a5f7070bf305faafc3465ae07b725574aa5324c7c4e":
        raise RuntimeError("native formal manifest mismatch")
    if _sha256(args.native_trace) != native_manifest["trace"]["sha256"]:
        raise RuntimeError("native action trace mismatch")
    if native_manifest["status"] != "complete" or native_manifest["results"]["tasks"][str(args.task_id)]["per_episode_success"][args.initial_state_index]:
        raise RuntimeError("selected source episode is not a completed recorded failure")
    if args.task_id != 5 or args.initial_state_index != 0 or args.seed != 41021:
        raise RuntimeError("this public diagnostic is frozen to task5/state0/seed41021")
    if not args.font.is_file():
        raise FileNotFoundError(args.font)
    if _sha256(args.checkpoint / "config.json") != protocol["checkpoint"]["sha256"]["config.json"]:
        raise RuntimeError("checkpoint config mismatch")
    actions, action_hashes, source = _load_source_actions(
        args.native_trace, args.task_id, args.initial_state_index
    )
    if len(actions) != 280 or source["plan"]["seed"] != args.seed or source["reset"]["seed"] != args.seed:
        raise RuntimeError("source failure does not have the frozen 280-step/seed contract")

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import torch
    from libero.libero import get_libero_path
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.random_utils import set_seed

    for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words"):
        if importlib.metadata.version(package) != protocol["software"][package]:
            raise RuntimeError(f"software mismatch: {package}")
    asset_tree = _tree_identity(Path(get_libero_path("assets")), "mixed_site_packages_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        if asset_tree[key] != protocol["assets"]["resolved_tree"][key]:
            raise RuntimeError(f"asset tree mismatch: {key}")

    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[5], fps=20, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360,
        observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][5]
    underlying = env.envs[0]
    if len(underlying._init_states) <= args.initial_state_index:
        raise RuntimeError("fixed initial state is unavailable")
    underlying.init_state_id = args.initial_state_index
    set_seed(args.seed)
    observation, _ = env.reset(seed=[args.seed], options={NEW_ROLLOUT_OPTION: True})
    if underlying.init_state_id - underlying._reset_stride != args.initial_state_index:
        raise RuntimeError("actual reset initial-state index mismatch")

    policy_cfg = SmolVLAConfig.from_pretrained(args.checkpoint, local_files_only=True)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    processed = preprocess_observation(observation)
    processed["task"] = list(env.call("task_description"))
    processed = env_preprocessor(processed)
    observed_input = {
        "state": {
            "shape": list(processed["observation.state"].shape),
            "dtype": str(processed["observation.state"].dtype),
            "sha256": _array_digest(processed["observation.state"]),
            "values": _jsonable(processed["observation.state"]),
        },
        "camera_frames": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_digest(value)}
            for key, value in sorted(processed.items()) if key.startswith("observation.images.")
        },
    }
    initial_checks = {
        "state": observed_input["state"]["sha256"] == source["input"]["state"]["sha256"],
        "cameras": observed_input["camera_frames"] == source["input"]["camera_frames"],
        "instruction": list(env.call("task_description"))[0] == source["input"]["instruction"],
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"initial observation mismatch: {initial_checks}")

    body_inventory = _body_names(underlying._env.sim)
    tracked = _tracked_body_names(body_inventory)
    if any(not names for names in tracked.values()):
        raise RuntimeError(f"required object bodies were not found: {tracked}")

    frames = [underlying.render().copy()]
    snapshots = [_pose_snapshot(underlying, _unbatch(observation), tracked)]
    step_records: list[dict[str, Any]] = []
    terminated_early = False
    for step, (action, expected_hash) in enumerate(zip(actions, action_hashes, strict=True)):
        if _array_digest(action) != expected_hash:
            raise RuntimeError(f"action changed before replay at step {step}")
        observation, reward, terminated, truncated, info = env.step(action)
        snapshot = _pose_snapshot(underlying, _unbatch(observation), tracked)
        frames.append(underlying.render().copy())
        snapshots.append(snapshot)
        record = {
            "step": step,
            "time_seconds": (step + 1) / 20.0,
            "action_sha256": expected_hash,
            "action": action[0].tolist(),
            "reward": float(reward[0]),
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
            "is_success": bool(info.get("is_success", [False])[0]),
            "pose_after_step": snapshot,
        }
        step_records.append(record)
        if record["terminated"] or record["truncated"]:
            terminated_early = True
            break
    env.close()
    if terminated_early or len(step_records) != 280:
        raise RuntimeError("replay terminated before the recorded 280-step failure")
    if any(item["is_success"] for item in step_records) or sum(item["reward"] for item in step_records) != 0:
        raise RuntimeError("replay outcome differs from the recorded zero-reward failure")

    raw_video = args.output_dir / "task5_state0_exact_rgb_20hz.mp4"
    _write_video(raw_video, frames, 20)
    annotated = [
        _annotated_frame(
            frame, index, 280, 0.0 if index == 0 else step_records[index - 1]["reward"],
            False if index == 0 else step_records[index - 1]["is_success"], snapshots[index], args.font,
        )
        for index, frame in enumerate(frames)
    ]
    annotated_video = args.output_dir / "task5_state0_failure_bilingual_20hz.mp4"
    _write_video(annotated_video, annotated, 20)

    trajectory = {
        "schema": "sentinel-libero-exact-action-replay-trajectory-v1",
        "selection": "outcome-conditioned recorded failure: native task5/state0",
        "benchmark_accounting": "post-hoc replay; excluded from all benchmark episode counts",
        "task_id": 5,
        "initial_state_index": 0,
        "seed": 41021,
        "instruction": source["input"]["instruction"],
        "tracked_body_names": tracked,
        "initial_pose": snapshots[0],
        "steps": step_records,
    }
    trajectory_path = args.output_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "sentinel-libero-exact-action-failure-replay-v1",
        "status": "complete",
        "selection": "outcome-conditioned recorded failure",
        "included_in_benchmark_metrics": False,
        "policy_inference": False,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "source": {
            "native_protocol_sha256": _sha256(args.native_protocol),
            "native_manifest_sha256": _sha256(args.native_manifest),
            "native_trace_sha256": _sha256(args.native_trace),
            "task_id": 5,
            "initial_state_index": 0,
            "seed": 41021,
            "recorded_actions": len(actions),
            "action_sequence_sha256": hashlib.sha256("\n".join(action_hashes).encode()).hexdigest(),
            "first_input_context": source["input"],
        },
        "initial_observation": {"checks": initial_checks, "observed": observed_input},
        "environment": asdict(env_cfg),
        "software": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            **{package: importlib.metadata.version(package) for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words")},
        },
        "asset_tree": asset_tree,
        "font": {"path": str(args.font.resolve()), "sha256": _sha256(args.font)},
        "body_inventory": body_inventory,
        "tracked_body_names": tracked,
        "outcome": {
            "steps": len(step_records), "sum_reward": 0.0, "max_reward": 0.0,
            "success": False, "terminated_early": False,
        },
        "artifacts": {
            "trajectory": {"path": trajectory_path.name, "sha256": _sha256(trajectory_path), "bytes": trajectory_path.stat().st_size},
            "raw_rgb_video": {"path": raw_video.name, "sha256": _sha256(raw_video), "bytes": raw_video.stat().st_size, "fps": 20, "frames": len(frames)},
            "bilingual_video": {"path": annotated_video.name, "sha256": _sha256(annotated_video), "bytes": annotated_video.stat().st_size, "fps": 20, "frames": len(annotated)},
        },
        "timing": {"elapsed_seconds": time.time() - started},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "outcome": manifest["outcome"], "artifacts": manifest["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
