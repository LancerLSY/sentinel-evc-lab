#!/usr/bin/env python3
"""Exact-action replay and bilingual film for the frozen backend discordant pair.

This post-hoc diagnostic never loads or calls the policy. Capture mode must be
run once in each isolated MuJoCo environment. Compose mode joins those native
RGB captures and labels any display-only hold after the shorter formal rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from run_libero_closedloop import _array_digest, _jsonable, _sha256, _tree_identity


TASK_ID = 5
STATE_INDEX = 21
SEED = 43022
FPS = 20
EXPECTED = {
    "current": {"mujoco": "3.8.1", "steps": 280, "success": False, "reward": 0.0},
    "legacy": {"mujoco": "3.3.7", "steps": 85, "success": True, "reward": 1.0},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "compose"), required=True)
    parser.add_argument("--backend-protocol", type=Path, required=True)
    parser.add_argument("--backend-runner", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--paired-episodes", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--current-trace", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-trace", type=Path, required=True)
    parser.add_argument("--native-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version-label", choices=("current", "legacy"))
    parser.add_argument("--libero-config", type=Path)
    parser.add_argument("--current-capture", type=Path)
    parser.add_argument("--legacy-capture", type=Path)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _require(not any(path.iterdir()), f"output directory must be empty: {path}")


def _episode(manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["episodes"]
        if row["task_id"] == TASK_ID and row["initial_state_index"] == STATE_INDEX
    ]
    _require(len(matches) == 1, "selected formal episode is missing or duplicated")
    return matches[0]


def _validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load(args.backend_protocol)
    native = _load(args.native_protocol)
    comparison = _load(args.comparison)
    pairs = _load(args.paired_episodes)
    manifests = {
        "current": _load(args.current_manifest),
        "legacy": _load(args.legacy_manifest),
    }
    traces = {"current": args.current_trace, "legacy": args.legacy_trace}
    _require(comparison["status"] == "complete", "backend comparison is incomplete")
    _require(comparison["protocol_sha256"] == _sha256(args.backend_protocol), "protocol receipt mismatch")
    _require(comparison["script_sha256"] == _sha256(args.backend_runner), "runner receipt mismatch")
    _require(comparison["paired_episodes"]["sha256"] == _sha256(args.paired_episodes), "pair receipt mismatch")
    _require(isinstance(pairs, list) and len(pairs) == 20, "expected the frozen 20 paired rows")
    task_pairs = sorted((row for row in pairs if row["task_id"] == TASK_ID), key=lambda row: row["initial_state_index"])
    discordant = [row for row in task_pairs if row["legacy_minus_current_success"] != 0]
    _require(discordant and discordant[0]["initial_state_index"] == STATE_INDEX, "state21 is not the first task5 discordant pair")
    selected_pair = discordant[0]
    _require(
        selected_pair["seed"] == SEED
        and not selected_pair["current_success"]
        and selected_pair["legacy_success"],
        "selected paired outcome changed",
    )
    for label in ("current", "legacy"):
        manifest = manifests[label]
        trace = traces[label]
        expected = EXPECTED[label]
        _require(manifest["status"] == "complete" and manifest["mode"] == "formal", f"{label} formal capture incomplete")
        _require(manifest["version_label"] == label, f"{label} version label mismatch")
        _require(manifest["software"]["mujoco"] == expected["mujoco"], f"{label} MuJoCo mismatch")
        _require(manifest["script_sha256"] == _sha256(args.backend_runner), f"{label} runner mismatch")
        _require(manifest["protocol_sha256"] == _sha256(args.backend_protocol), f"{label} protocol mismatch")
        _require(comparison[f"{label}_manifest"]["sha256"] == _sha256(getattr(args, f"{label}_manifest")), f"{label} manifest mismatch")
        _require(manifest["trace"]["sha256"] == _sha256(trace), f"{label} trace mismatch")
        row = _episode(manifest)
        _require(
            row["actual_initial_state_index"] == STATE_INDEX
            and row["actual_seed"] == SEED
            and row["env_steps"] == expected["steps"]
            and row["success"] is expected["success"]
            and row["sum_reward"] == expected["reward"]
            and not row["crashed"],
            f"{label} selected formal episode changed",
        )
    current = manifests["current"]
    legacy = manifests["legacy"]
    for key in ("source_bindings", "model_artifacts", "asset_tree", "task_source_sha256"):
        _require(current[key] == legacy[key], f"non-treatment binding differs: {key}")
    _require(
        current["source_bindings"]["native_protocol"]["sha256"] == _sha256(args.native_protocol),
        "native protocol binding mismatch",
    )
    checkpoint_hashes = current["model_artifacts"]["checkpoint_sha256"]
    for name, expected in checkpoint_hashes.items():
        _require(_sha256(args.checkpoint / name) == expected, f"checkpoint file mismatch: {name}")
    _require(args.font.is_file(), "font is missing")
    return {
        "protocol": protocol,
        "native": native,
        "comparison": comparison,
        "pairs": pairs,
        "manifests": manifests,
        "traces": traces,
        "selected_pair": selected_pair,
        "checkpoint_hashes": checkpoint_hashes,
    }


def _source_actions(trace: Path, version_label: str) -> dict[str, Any]:
    plan = reset = context = None
    env_actions: list[np.ndarray] = []
    action_hashes: list[str] = []
    postprocessed: dict[int, tuple[str, np.ndarray]] = {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("task_id") != TASK_ID or event.get("initial_state_index") != STATE_INDEX:
            continue
        _require(event.get("backend_version_label") == version_label, "trace version label mismatch")
        if event["event"] == "episode_plan":
            plan = event
        elif event["event"] == "episode_reset":
            reset = event
        elif event["event"] == "raw_policy_chunk" and event["step"] == 0:
            context = event["input_context"]
        elif event["event"] == "official_postprocessed_action":
            array = np.asarray(event["values"], dtype=np.float32)
            _require(_array_digest(array) == event["sha256"], "postprocessed action digest mismatch")
            postprocessed[int(event["step"])] = (event["sha256"], array)
        elif event["event"] == "env_step_action":
            step = len(env_actions)
            _require(event["step"] == step, "env actions are not contiguous")
            array = np.asarray(event["values"], dtype=np.float32)
            _require(array.shape == (1, 7) and _array_digest(array) == event["sha256"], "env action digest mismatch")
            _require(event["postprocess_to_env_step_max_abs_diff"] == 0.0, "formal observer recorded an action difference")
            env_actions.append(array)
            action_hashes.append(event["sha256"])
    _require(plan is not None and reset is not None and context is not None, "trace lacks plan, reset, or first input")
    _require(plan["seed"] == SEED and reset["seed"] == SEED, "trace seed mismatch")
    _require(len(postprocessed) == len(env_actions), "postprocessed action count mismatch")
    for step, action in enumerate(env_actions):
        digest, official = postprocessed[step]
        _require(digest == action_hashes[step] and np.array_equal(official, action), "formal postprocess/env action mismatch")
    return {
        "plan": plan,
        "reset": reset,
        "context": context,
        "actions": env_actions,
        "action_hashes": action_hashes,
    }


def _processed_camera_png(array: Any, path: Path) -> None:
    from PIL import Image

    pixels = np.asarray(array.detach().cpu(), dtype=np.float32)[0]
    pixels = np.moveaxis(pixels, 0, -1)
    pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, optimize=True)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def _capture(args: argparse.Namespace, bound: dict[str, Any]) -> None:
    _require(args.version_label in EXPECTED, "capture mode requires --version-label")
    _require(args.libero_config is not None and args.libero_config.is_dir(), "capture mode requires --libero-config")
    label = args.version_label
    expected = EXPECTED[label]
    source = _source_actions(bound["traces"][label], label)
    _require(len(source["actions"]) == expected["steps"], "source action count mismatch")
    _empty(args.output_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.utils.random_utils import set_seed

    versions = {
        package: importlib.metadata.version(package)
        for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words")
    }
    _require(versions["mujoco"] == expected["mujoco"], "active MuJoCo version mismatch")
    for package in ("lerobot", "hf-libero", "robosuite", "num2words"):
        _require(versions[package] == bound["native"]["software"][package], f"software mismatch: {package}")
    asset_tree = _tree_identity(Path(get_libero_path("assets")), "mixed_site_packages_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        _require(asset_tree[key] == bound["manifests"][label]["asset_tree"][key], f"asset tree mismatch: {key}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    task_sources = {
        "init_states": {str(i): _sha256(init_root / task.init_states_file) for i, task in enumerate(suite.tasks)},
        "bddl": {str(i): _sha256(bddl_root / task.bddl_file) for i, task in enumerate(suite.tasks)},
    }
    _require(task_sources == bound["manifests"][label]["task_source_sha256"], "task source mismatch")

    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[TASK_ID], fps=FPS, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360, observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][TASK_ID]
    underlying = env.envs[0]
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    started = time.time()
    try:
        underlying.init_state_id = STATE_INDEX
        set_seed(SEED)
        observation, _ = env.reset(seed=[SEED], options={NEW_ROLLOUT_OPTION: True})
        actual_state = int(underlying.init_state_id - underlying._reset_stride)
        actual_seed = int(underlying.np_random_seed)
        _require(actual_state == STATE_INDEX and actual_seed == SEED, "actual reset receipt mismatch")
        processed = preprocess_observation(observation)
        instruction = str(env.call("task_description")[0])
        processed["task"] = [instruction]
        processed = env_preprocessor(processed)
        state = processed["observation.state"]
        cameras = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_digest(value),
            }
            for key, value in sorted(processed.items())
            if key.startswith("observation.images.")
        }
        observed_context = {
            "instruction": instruction,
            "state": {"shape": list(state.shape), "dtype": str(state.dtype), "sha256": _array_digest(state), "values": _jsonable(state)},
            "camera_frames": cameras,
        }
        checks = {
            "instruction": instruction == source["context"]["instruction"],
            "processed_robot_state": observed_context["state"] == source["context"]["state"],
            "processed_camera_hashes": cameras == source["context"]["camera_frames"],
        }
        _require(all(checks.values()), f"first observation mismatch: {checks}")
        camera_artifacts = {}
        for index, key in enumerate(sorted(cameras), start=1):
            path = args.output_dir / f"first-camera-{index}.png"
            _processed_camera_png(processed[key], path)
            camera_artifacts[key] = _receipt(path)

        frames: list[np.ndarray] = []
        steps: list[dict[str, Any]] = []
        for step, (action, action_sha) in enumerate(zip(source["actions"], source["action_hashes"], strict=True)):
            _require(_array_digest(action) == action_sha, f"action changed before replay at step {step}")
            observation, reward, terminated, truncated, info = env.step(action)
            _require("is_success" in info, f"official is_success label missing at step {step}")
            success = bool(info["is_success"][0])
            record = {
                "step": step,
                "action_time_seconds": (step + 1) / FPS,
                "env_action_sha256": action_sha,
                "env_action": action[0].tolist(),
                "official_reward": float(reward[0]),
                "is_success": success,
                "is_success_present": True,
                "is_success_source": "LIBERO env.step info.is_success",
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
            }
            steps.append(record)
            frames.append(underlying.render().copy())
            if (record["terminated"] or record["truncated"]) and step + 1 < expected["steps"]:
                raise RuntimeError(f"replay terminated before recorded action count at step {step}")
    finally:
        env.close()

    sum_reward = float(sum(row["official_reward"] for row in steps))
    max_reward = float(max(row["official_reward"] for row in steps))
    success = any(row["is_success"] for row in steps)
    _require(
        len(steps) == expected["steps"]
        and all(row["is_success_present"] for row in steps)
        and success is expected["success"]
        and sum_reward == expected["reward"]
        and max_reward == expected["reward"],
        "replayed official outcome differs from the selected formal episode",
    )
    trajectory = {
        "schema": "sentinel-libero-backend-exact-action-trajectory-v1",
        "version_label": label,
        "mujoco": expected["mujoco"],
        "task_id": TASK_ID,
        "initial_state_index": STATE_INDEX,
        "seed": SEED,
        "steps": steps,
    }
    trajectory_path = args.output_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_video = args.output_dir / f"task5-state21-{label}-native-rgb.mp4"
    _write_video(raw_video, frames)
    formal_manifest_path = getattr(args, f"{label}_manifest")
    formal_trace_path = getattr(args, f"{label}_trace")
    manifest = {
        "schema": "sentinel-libero-backend-exact-action-capture-v1",
        "status": "complete",
        "included_in_benchmark_metrics": False,
        "policy_inference": False,
        "observer_interventions": 0,
        "version_label": label,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "selection": {
            "rule": "first state-ordered discordant task5 pair in the frozen paired rows",
            "task_id": TASK_ID,
            "initial_state_index": STATE_INDEX,
            "seed": SEED,
            "post_hoc": True,
        },
        "source": {
            "backend_protocol": _receipt(args.backend_protocol),
            "backend_runner": _receipt(args.backend_runner),
            "comparison": _receipt(args.comparison),
            "paired_episodes": _receipt(args.paired_episodes),
            "current_manifest": _receipt(args.current_manifest),
            "current_trace": _receipt(args.current_trace),
            "legacy_manifest": _receipt(args.legacy_manifest),
            "legacy_trace": _receipt(args.legacy_trace),
            "selected_formal_manifest": _receipt(formal_manifest_path),
            "selected_formal_trace": _receipt(formal_trace_path),
            "native_protocol": _receipt(args.native_protocol),
            "checkpoint_sha256": bound["checkpoint_hashes"],
            "formal_action_count": len(source["actions"]),
            "action_sequence_sha256": hashlib.sha256("\n".join(source["action_hashes"]).encode()).hexdigest(),
        },
        "actual_reset": {"initial_state_index": actual_state, "seed": actual_seed},
        "first_observation": {"checks": checks, "processed": observed_context, "camera_pngs": camera_artifacts},
        "environment": asdict(env_cfg),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, **versions},
        "asset_tree": asset_tree,
        "task_source_sha256": task_sources,
        "outcome": {
            "steps": len(steps), "sum_reward": sum_reward, "max_reward": max_reward,
            "success": success, "final_action_time_seconds": len(steps) / FPS,
        },
        "artifacts": {
            "trajectory": _receipt(trajectory_path),
            "native_rgb_video": {
                **_receipt(raw_video), "fps": FPS, "frames": len(frames),
                "duration_seconds": len(frames) / FPS,
                "frame_semantics": "post-action frames only; no extra initial frame",
                "source": "LIBERO Panda environment render after each recorded env.step action",
            },
        },
        "claim_boundary": [
            "Post-hoc exact-action replay; excluded from every formal denominator.",
            "The observer makes zero action changes and performs no policy inference.",
            "The complete MuJoCo version is the treatment; this replay does not isolate reset causation or physical accuracy.",
        ],
        "timing": {"elapsed_seconds": time.time() - started},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _font(path: Path, size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(path), size)


def _panel(draw: Any, x: int, title: str, subtitle: str, record: dict[str, Any], success: bool, held: bool, fonts: dict[str, Any]) -> None:
    draw.text((x, 142), title, font=fonts["heading"], fill="#f8fafc")
    draw.text((x, 184), subtitle, font=fonts["small"], fill="#94a3b8")
    status = "FORMAL: SUCCESS" if success else "FORMAL: FAILURE"
    color = "#16a34a" if success else "#dc2626"
    draw.rounded_rectangle((x, 902, x + 840, 970), radius=16, fill="#111827")
    draw.text((x + 22, 914), f"官方奖励 / Reward  {record['official_reward']:.1f}", font=fonts["body"], fill="#e2e8f0")
    draw.text((x + 330, 914), f"动作时间 / Action time  {record['action_time_seconds']:.2f}s", font=fonts["body"], fill="#e2e8f0")
    draw.text((x + 650, 908), "正式最终结果", font=fonts["tiny"], fill="#cbd5e1")
    draw.text((x + 650, 936), status, font=fonts["small_bold"], fill=color)
    if held:
        draw.rounded_rectangle((x + 70, 785, x + 770, 865), radius=16, fill="#7c2d12")
        draw.text((x + 118, 798), "成功后冻结最后一帧：无新增动作或运动", font=fonts["body"], fill="#fff7ed")
        draw.text((x + 127, 833), "Last frame held after success: no added action or motion", font=fonts["small"], fill="#fed7aa")


def _compose_frame(current: np.ndarray, legacy: np.ndarray, index: int, current_record: dict[str, Any], legacy_record: dict[str, Any], held: bool, font_path: Path) -> np.ndarray:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1920, 1080), "#07111f")
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": _font(font_path, 39), "heading": _font(font_path, 29), "body": _font(font_path, 21),
        "small": _font(font_path, 18), "small_bold": _font(font_path, 19), "tiny": _font(font_path, 16),
    }
    draw.text((60, 28), "同一正式样本的 MuJoCo 完整版本处理回放", font=fonts["title"], fill="#f8fafc")
    draw.text((60, 76), "Paired replay under complete MuJoCo version treatment", font=fonts["heading"], fill="#38bdf8")
    draw.text((1110, 42), "任务 / Task 5  ·  将黑碗从小烤碗上拿起并放到盘子上", font=fonts["small"], fill="#cbd5e1")
    draw.text((1110, 70), "Pick up the black bowl on the ramekin and place it on the plate", font=fonts["small"], fill="#94a3b8")
    left = Image.fromarray(current).resize((680, 680), Image.Resampling.LANCZOS)
    right = Image.fromarray(legacy).resize((680, 680), Image.Resampling.LANCZOS)
    canvas.paste(left, (140, 210)); canvas.paste(right, (1100, 210))
    _panel(draw, 60, "MuJoCo 3.8.1 · 当前栈 / current stack", f"正式动作 {index + 1:03d}/280 · recorded formal action", current_record, False, False, fonts)
    legacy_index = min(index + 1, EXPECTED["legacy"]["steps"])
    _panel(draw, 1020, "MuJoCo 3.3.7 · 兼容栈 / compatibility stack", f"正式动作 {legacy_index:03d}/85 · recorded formal action", legacy_record, True, held, fonts)
    draw.text((60, 992), "同一官方初态索引 21 与种子 43022；完整版本处理产生不同首次相机观测。", font=fonts["tiny"], fill="#cbd5e1")
    draw.text((60, 1018), "Same official init index and seed; the complete-version treatment yields different first camera observations.", font=fonts["tiny"], fill="#94a3b8")
    draw.text((60, 1044), "只回放已记录 env.step 动作 · 观察器零干预 · 事后诊断，不计入正式分母 / Exact recorded actions · zero intervention · post-hoc, excluded", font=fonts["tiny"], fill="#fbbf24")
    return np.asarray(canvas)


def _verified_capture(path: Path, label: str, script_sha: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = _load(path)
    _require(manifest["status"] == "complete" and manifest["version_label"] == label, f"{label} replay capture incomplete")
    _require(manifest["script_sha256"] == script_sha and not manifest["policy_inference"], f"{label} replay source mismatch")
    trajectory_path = path.parent / Path(manifest["artifacts"]["trajectory"]["path"]).name
    video_path = path.parent / Path(manifest["artifacts"]["native_rgb_video"]["path"]).name
    _require(_sha256(trajectory_path) == manifest["artifacts"]["trajectory"]["sha256"], f"{label} trajectory mismatch")
    _require(_sha256(video_path) == manifest["artifacts"]["native_rgb_video"]["sha256"], f"{label} video mismatch")
    trajectory = _load(trajectory_path)
    _require(len(trajectory["steps"]) == EXPECTED[label]["steps"], f"{label} trajectory step mismatch")
    return manifest, trajectory, video_path


def _compose(args: argparse.Namespace, bound: dict[str, Any]) -> None:
    _require(args.current_capture is not None and args.legacy_capture is not None, "compose mode requires both capture manifests")
    _empty(args.output_dir)
    script_sha = _sha256(Path(__file__).resolve())
    current_manifest, current_trajectory, current_video = _verified_capture(args.current_capture, "current", script_sha)
    legacy_manifest, legacy_trajectory, legacy_video = _verified_capture(args.legacy_capture, "legacy", script_sha)
    for key in (
        "backend_protocol", "backend_runner", "comparison", "paired_episodes",
        "current_manifest", "current_trace", "legacy_manifest", "legacy_trace",
        "native_protocol", "checkpoint_sha256",
    ):
        _require(current_manifest["source"][key] == legacy_manifest["source"][key], f"capture provenance differs: {key}")
    import imageio.v2 as imageio
    from PIL import Image

    legacy_frames = [np.asarray(frame) for frame in imageio.get_reader(legacy_video)]
    _require(len(legacy_frames) == EXPECTED["legacy"]["steps"], "legacy decoded frame count mismatch")
    output = args.output_dir / "backend-comparison-bilingual.mp4"
    poster = args.output_dir / "task5-state21-backend-paired-poster.png"
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8)
    count = 0
    last_composed = None
    try:
        for index, frame in enumerate(imageio.get_reader(current_video)):
            _require(index < EXPECTED["current"]["steps"], "current video has extra frames")
            held = index >= len(legacy_frames)
            legacy_index = min(index, len(legacy_frames) - 1)
            composed = _compose_frame(
                np.asarray(frame), legacy_frames[legacy_index], index,
                current_trajectory["steps"][index], legacy_trajectory["steps"][legacy_index], held, args.font,
            )
            writer.append_data(composed)
            last_composed = composed
            count += 1
    finally:
        writer.close()
    _require(count == EXPECTED["current"]["steps"] and last_composed is not None, "paired video frame count mismatch")
    Image.fromarray(last_composed).save(poster, optimize=True)
    manifest = {
        "schema": "sentinel-libero-backend-paired-replay-film-v1",
        "status": "complete",
        "included_in_benchmark_metrics": False,
        "policy_inference": False,
        "observer_interventions": 0,
        "script_sha256": script_sha,
        "selection": {
            "rule": "first state-ordered discordant task5 pair in frozen paired rows",
            "task_id": TASK_ID, "initial_state_index": STATE_INDEX, "seed": SEED,
            "current_formal_outcome": "failure", "legacy_formal_outcome": "success",
            "post_hoc": True,
        },
        "inputs": {
            "backend_protocol": _receipt(args.backend_protocol), "backend_runner": _receipt(args.backend_runner),
            "comparison": _receipt(args.comparison), "paired_episodes": _receipt(args.paired_episodes),
            "current_formal_manifest": _receipt(args.current_manifest), "current_formal_trace": _receipt(args.current_trace),
            "legacy_formal_manifest": _receipt(args.legacy_manifest), "legacy_formal_trace": _receipt(args.legacy_trace),
            "current_capture_manifest": _receipt(args.current_capture), "legacy_capture_manifest": _receipt(args.legacy_capture),
            "font": _receipt(args.font),
        },
        "display": {
            "fps": FPS, "frames": count, "resolution": [1920, 1080],
            "duration_seconds": count / FPS,
            "post_action_frames_only": True,
            "current_native_motion_frames": EXPECTED["current"]["steps"],
            "current_native_motion_seconds": EXPECTED["current"]["steps"] / FPS,
            "legacy_native_motion_frames": EXPECTED["legacy"]["steps"],
            "legacy_native_motion_seconds": EXPECTED["legacy"]["steps"] / FPS,
            "legacy_display_only_hold_frames": EXPECTED["current"]["steps"] - EXPECTED["legacy"]["steps"],
            "legacy_display_only_hold_seconds": (EXPECTED["current"]["steps"] - EXPECTED["legacy"]["steps"]) / FPS,
            "hold_rule": "repeat the final successful native RGB frame; no additional action, physics step, reward, or motion",
        },
        "outcomes": {"current": current_manifest["outcome"], "legacy": legacy_manifest["outcome"]},
        "artifacts": {"bilingual_paired_video": {**_receipt(output), "fps": FPS, "frames": count}, "poster": _receipt(poster)},
        "claim_boundary": [
            "The same official initial-state index and seed are used, while first camera observations differ under the complete MuJoCo version treatment.",
            "This paired replay does not isolate a reset mechanism and does not identify either MuJoCo version as physically more accurate.",
            "It replays formal env.step actions exactly with zero observer intervention and is excluded from all benchmark denominators.",
            "The shorter successful clip holds its final frame only for synchronized display; the hold adds no action, simulation, reward, or motion.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    bound = _validate_sources(args)
    if args.mode == "capture":
        _capture(args, bound)
    else:
        _compose(args, bound)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
