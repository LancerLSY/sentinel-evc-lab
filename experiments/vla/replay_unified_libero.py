#!/usr/bin/env python3
"""Replay the first declared unified LIBERO formal root from retained actions.

This is an excluded visualization/trace-equality check.  It loads no policy,
does not add an episode to experiment statistics, and aborts before writing a
video if any physics-substep Panda qpos differs from retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(
        str(array.dtype).encode() + b"\0" + _canonical(list(array.shape)) + b"\0" + array.tobytes()
    ).hexdigest()
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": "sha256:" + digest}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _step_with_substeps(underlying: Any, action: np.ndarray, qpos_indices: list[int]) -> tuple[Any, np.ndarray]:
    sim = underlying._env.sim
    data = getattr(sim.data, "_data", sim.data)
    original = sim.step
    had_instance_step = hasattr(sim, "__dict__") and "step" in sim.__dict__
    samples: list[np.ndarray] = []

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        samples.append(np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy())
        return result

    setattr(sim, "step", observed_step)
    try:
        output = underlying.step(np.asarray(action, dtype=np.float32))
    finally:
        if had_instance_step:
            setattr(sim, "step", original)
        else:
            delattr(sim, "step")
    if not samples:
        raise RuntimeError("replay action produced no physics substeps")
    return output, np.asarray(samples, dtype=np.float64)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 16
    width = max(draw.textlength(line, font=font) for line in lines) + 20
    height = line_height * len(lines) + 16
    draw.rectangle((8, 8, 8 + width, 8 + height), fill=(0, 0, 0, 178))
    for index, line in enumerate(lines):
        draw.text((18, 16 + index * line_height), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _episode_id(task_id: int, state_index: int) -> str:
    return f"formal-task{task_id:02d}-state{state_index:03d}-delta_evc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-rad", type=float, default=1e-8)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("output-dir must not exist")
    if not np.isfinite(args.tolerance_rad) or args.tolerance_rad <= 0:
        raise ValueError("tolerance-rad must be positive and finite")
    manifest_path = run_dir / "manifest.json"
    steps_path = run_dir / "steps.jsonl"
    episodes_path = run_dir / "per_episode.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    formal = config["formal"]
    if not formal["task_ids"] or not formal["initial_state_indices"]:
        raise RuntimeError("run has no declared formal root")
    task_id = int(formal["task_ids"][0])
    state_index = int(formal["initial_state_indices"][0])
    seed = int(formal["seed_base"]) + state_index - min(int(item) for item in formal["initial_state_indices"])
    branch = "delta_evc"
    episode_id = _episode_id(task_id, state_index)
    episode_rows = [row for row in _load_jsonl(episodes_path) if row.get("episode_id") == episode_id]
    if len(episode_rows) != 1:
        raise RuntimeError(f"expected one retained result for {episode_id}, found {len(episode_rows)}")
    episode = episode_rows[0]
    trace = [row for row in _load_jsonl(steps_path) if row.get("episode_id") == episode_id]
    actual_steps = sorted(
        (row for row in trace if row.get("event") == "actual_step"), key=lambda row: int(row["step"])
    )
    if [int(row["step"]) for row in actual_steps] != list(range(len(actual_steps))):
        raise RuntimeError("retained actual steps are not contiguous")
    denied = [row for row in trace if row.get("event") == "denied"]
    denied_reason = denied[-1].get("reason") if denied else None
    candidate_path = run_dir / episode["candidate_npz"]["path"]
    if _file_sha256(candidate_path) != episode["candidate_npz"]["sha256"]:
        raise RuntimeError("candidate NPZ digest does not match retained episode result")

    actions: list[np.ndarray] = []
    expected_by_cycle: dict[int, np.ndarray] = {}
    forecast_by_cycle: dict[int, np.ndarray] = {}
    cycle_offsets: dict[int, int] = {}
    with np.load(candidate_path, allow_pickle=False) as retained:
        for row in actual_steps:
            cycle, within = int(row["cycle"]), int(row["within_cycle"])
            action = np.asarray(retained[f"cycle_{cycle:02d}_final_native"][0, within], dtype=np.float32)
            identity = _array_identity(action[None, :])
            if identity["sha256"] != row["actual_env_step_hash"]:
                raise RuntimeError(f"retained final action differs from actual writer bytes at step {row['step']}")
            actions.append(action)
            if cycle not in expected_by_cycle:
                expected_by_cycle[cycle] = np.asarray(retained[f"cycle_{cycle:02d}_executed_qpos"], dtype=np.float64)
                forecast_by_cycle[cycle] = np.asarray(retained[f"cycle_{cycle:02d}_final_qpos"], dtype=np.float64)
                cycle_offsets[cycle] = 1

    os.environ.setdefault("MUJOCO_GL", "egl")
    started = time.time()
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.scripts.lerobot_eval import close_envs
    from lerobot.utils.random_utils import set_seed

    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[task_id], fps=20, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360, observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][task_id]
    underlying = env.envs[0]
    underlying.init_state_id = state_index
    set_seed(seed)
    env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    actual_state = int(underlying.init_state_id - underlying._reset_stride)
    if actual_state != state_index:
        raise RuntimeError(f"reset state mismatch: requested {state_index}, observed {actual_state}")
    qpos_indices = [int(item) for item in underlying._env.robots[0]._ref_joint_pos_indexes]
    instruction = str(env.call("task_description")[0])
    frames = [_annotate(underlying.render().copy(), [
        f"EXCLUDED REPLAY | task {task_id} state {state_index}",
        f"branch {branch} | step 0/{len(actions)}",
        instruction,
    ])]
    comparisons: list[dict[str, Any]] = []
    success = False
    replay_error: dict[str, Any] | None = None
    try:
        if actual_steps:
            first_cycle = int(actual_steps[0]["cycle"])
            live_qpos = np.asarray(
                underlying._env.sim.data.qpos[qpos_indices], dtype=np.float64
            )
            retained_initial = expected_by_cycle[first_cycle][0]
            initial_error = float(np.max(np.abs(live_qpos - retained_initial)))
            initial_forecast_error = float(
                np.max(np.abs(live_qpos - forecast_by_cycle[first_cycle][0]))
            )
            comparisons.append({
                "kind": "initial", "step": -1, "cycle": first_cycle,
                "physics_substeps": 0, "max_abs_qpos_error_rad": initial_error,
                "max_abs_forecast_qpos_error_rad": initial_forecast_error,
                "within_tolerance": max(initial_error, initial_forecast_error) <= args.tolerance_rad,
            })
            if max(initial_error, initial_forecast_error) > args.tolerance_rad:
                raise RuntimeError(
                    "initial Panda qpos differs from retained actual/forecast trace: "
                    f"actual={initial_error:.12g} forecast={initial_forecast_error:.12g} rad"
                )
        for step_index, (action, source) in enumerate(zip(actions, actual_steps, strict=True)):
            cycle = int(source["cycle"])
            output, observed = _step_with_substeps(underlying, action, qpos_indices)
            start = cycle_offsets[cycle]
            stop = start + len(observed)
            expected = expected_by_cycle[cycle][start:stop]
            forecast = forecast_by_cycle[cycle][start:stop]
            if len(expected) != len(observed) or len(forecast) != len(observed):
                raise RuntimeError(
                    f"substep count mismatch at step {step_index}: replay={len(observed)} "
                    f"retained={len(expected)} forecast={len(forecast)}"
                )
            max_error = float(np.max(np.abs(observed - expected)))
            forecast_error = float(np.max(np.abs(observed - forecast)))
            retained_forecast_error = float(np.max(np.abs(expected - forecast)))
            comparison = {
                "step": step_index, "cycle": cycle, "within_cycle": int(source["within_cycle"]),
                "physics_substeps": len(observed), "max_abs_qpos_error_rad": max_error,
                "max_abs_forecast_qpos_error_rad": forecast_error,
                "retained_actual_vs_forecast_max_abs_rad": retained_forecast_error,
                "within_tolerance": max(max_error, forecast_error, retained_forecast_error) <= args.tolerance_rad,
                "action_sha256": _array_identity(action[None, :])["sha256"],
            }
            comparisons.append(comparison)
            if max(max_error, forecast_error, retained_forecast_error) > args.tolerance_rad:
                raise RuntimeError(
                    f"physics replay diverged at step {step_index}: actual={max_error:.12g}, "
                    f"forecast={forecast_error:.12g}, retained={retained_forecast_error:.12g} rad"
                )
            cycle_offsets[cycle] = stop
            _, _, terminated, truncated, info = output
            if isinstance(info, dict):
                success = success or bool(np.asarray(info.get("is_success", False)).reshape(-1)[0])
            frame_lines = [
                f"EXCLUDED REPLAY | task {task_id} state {state_index}",
                f"branch {branch} | step {step_index + 1}/{len(actions)} | cycle {cycle}",
                f"substep qpos max error {max_error:.3e} rad",
                instruction,
            ]
            if step_index + 1 == len(actions) and denied_reason:
                frame_lines.append(f"terminal denial: {denied_reason}")
            frames.append(_annotate(underlying.render().copy(), frame_lines))
            if bool(terminated) or bool(truncated):
                if step_index + 1 != len(actions):
                    raise RuntimeError("replay environment terminated before retained action trace ended")
        if bool(episode.get("success")) != success:
            raise RuntimeError(
                f"replay success={success} differs from retained success={bool(episode.get('success'))}"
            )
    except Exception as error:
        replay_error = {"error_type": type(error).__name__, "message": str(error)}
    finally:
        close_envs(envs)

    output_dir.mkdir(parents=True)
    summary = {
        "schema": "sentinel-unified-libero-excluded-replay-v1",
        "status": "complete" if replay_error is None else "diverged",
        "included_in_experiment_statistics": False,
        "selection": "first declared formal task/state; fixed delta_evc branch; no outcome selection",
        "source": {
            "run_dir": str(run_dir), "manifest_sha256": _file_sha256(manifest_path),
            "steps_sha256": _file_sha256(steps_path), "episodes_sha256": _file_sha256(episodes_path),
            "candidate_npz_sha256": _file_sha256(candidate_path),
        },
        "episode": {"episode_id": episode_id, "task_id": task_id, "state_index": state_index,
                    "seed": seed, "branch": branch, "instruction": instruction},
        "retained_outcome": {"success": bool(episode.get("success")), "steps": int(episode.get("steps", 0)),
                             "denied_final_reason": denied_reason},
        "replay_outcome": {
            "success": success,
            "steps": sum(int(item.get("step", -1)) >= 0 for item in comparisons),
        },
        "trace_equality": {
            "tolerance_rad": args.tolerance_rad, "all_substeps_within_tolerance": replay_error is None,
            "max_abs_qpos_error_rad": max((item["max_abs_qpos_error_rad"] for item in comparisons), default=0.0),
            "max_abs_forecast_qpos_error_rad": max(
                (item["max_abs_forecast_qpos_error_rad"] for item in comparisons), default=0.0
            ),
            "comparisons": comparisons,
        },
        "claim_boundary": "direct MuJoCo physics replay of retained final native actions; no policy inference and no frozen-scene animation",
        "error": replay_error,
        "elapsed_seconds": time.time() - started,
    }
    if replay_error is None:
        video_path = output_dir / f"{episode_id}-excluded-replay.mp4"
        _write_video(video_path, frames, fps=20)
        summary["video"] = {"path": video_path.name, "sha256": _file_sha256(video_path),
                            "frames": len(frames), "fps": 20}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path),
                      "video": summary.get("video")}, indent=2))
    return 0 if replay_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
