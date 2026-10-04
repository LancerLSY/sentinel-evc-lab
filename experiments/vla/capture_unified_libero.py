#!/usr/bin/env python3
"""Capture one excluded live SmolVLA/LIBERO geometry+EVC episode.

The frozen experiment runner is imported unchanged. Frames are rendered only
after the NativeGateway writer has executed an authorized environment step;
forecast rollouts are never captured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_unified_libero as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="frozen experiment config to clone")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--state-index", type=int, default=40)
    parser.add_argument("--branch", choices=runner.BRANCHES, default="delta_evc")
    parser.add_argument("--seed", type=int, help="default: 2026100400 + state-index - 40")
    parser.add_argument("--deadline-timestamp", help="explicitly override the frozen config deadline")
    args = parser.parse_args()
    if args.task_id < 0 or args.state_index < 0:
        raise ValueError("task-id and state-index must be non-negative")
    seed = args.seed if args.seed is not None else 2026100400 + args.state_index - 40
    source_config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("output-dir must not exist")

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["output_dir"] = str(output_dir)
    if args.deadline_timestamp is not None:
        config["deadline_timestamp"] = args.deadline_timestamp
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    capture_config = output_dir.parent / f".{output_dir.name}-config.json"
    capture_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw_frames: list[np.ndarray] = []
    committed_writer_active = False
    original_submit = runner.NativeGateway.submit
    original_step = runner._step_with_substeps
    original_specs = runner._episode_specs
    original_branches = runner.BRANCHES
    original_argv = sys.argv[:]

    def capture_step(underlying: Any, action: Any, qpos_indices: list[int], geometry_profile: Any) -> Any:
        result = original_step(underlying, action, qpos_indices, geometry_profile)
        if committed_writer_active:
            raw_frames.append(np.asarray(underlying.render(), dtype=np.uint8).copy())
        return result

    def capture_submit(self: Any, permit: Any, request: Any, *positional: Any, **keywords: Any) -> Any:
        nonlocal committed_writer_active
        writer = keywords.get("writer")
        if writer is None:
            raise RuntimeError("NativeGateway.submit writer must be explicit for capture isolation")

        def committed_writer(exact_request: Any, entered: Any) -> Any:
            nonlocal committed_writer_active
            if committed_writer_active:
                raise RuntimeError("nested committed writer capture")
            committed_writer_active = True
            try:
                return writer(exact_request, entered)
            finally:
                committed_writer_active = False

        keywords["writer"] = committed_writer
        return original_submit(self, permit, request, *positional, **keywords)

    started = time.time()
    try:
        runner.BRANCHES = (args.branch,)
        runner._episode_specs = lambda unused: [{
            "phase": "formal", "task_id": args.task_id, "state_index": args.state_index, "seed": seed,
        }]
        runner._step_with_substeps = capture_step
        runner.NativeGateway.submit = capture_submit
        sys.argv = [str(Path(__file__)), "--config", str(capture_config)]
        return_code = runner.main()
    finally:
        sys.argv = original_argv
        runner.NativeGateway.submit = original_submit
        runner._step_with_substeps = original_step
        runner._episode_specs = original_specs
        runner.BRANCHES = original_branches

    episodes = _jsonl(output_dir / "per_episode.jsonl")
    if len(episodes) != 1:
        raise RuntimeError(f"expected one excluded live episode, found {len(episodes)}")
    episode = episodes[0]
    trace = _jsonl(output_dir / "steps.jsonl")
    actual = [row for row in trace if row.get("event") == "actual_step"]
    denied = [row for row in trace if row.get("event") == "denied"]
    recovery_cycles: dict[int, dict[str, Any]] = {}
    for row in trace:
        recovery = row.get("safe_prefix_recovery") if row.get("event") == "candidate" else None
        if isinstance(recovery, dict) and bool(recovery.get("prefix_success")):
            cycle = int(row["cycle"])
            recovery_cycles[cycle] = {
                "cycle": cycle,
                "prefix_length": int(recovery["prefix_length"]),
                "selected_execution_horizon": int(recovery["selected_execution_horizon"]),
                "policy": str(recovery["policy"]),
            }
    if len(raw_frames) != len(actual):
        raise RuntimeError(f"captured {len(raw_frames)} frames for {len(actual)} committed writes")
    raw_task = actual[0].get("info", {}).get("task") if actual else None
    if isinstance(raw_task, list) and raw_task:
        raw_task = raw_task[0]
    task_goal = str(raw_task).replace("_", " ").strip() if raw_task else f"LIBERO spatial task {args.task_id}"
    if task_goal:
        task_goal = task_goal[0].upper() + task_goal[1:]

    denied_reason = denied[-1].get("reason") if denied else None
    outcome = "success" if episode.get("success") else "stopped"
    if episode.get("crashed"):
        outcome = f"crashed: {episode.get('error_type')}"
    elif denied_reason:
        outcome = f"denied: {denied_reason}"
    frames = [
        _annotate(frame, [
            "EXCLUDED LIVE DEMO | geometry + EVC + SmolVLA",
            f"task {args.task_id} state {args.state_index} | {args.branch} | committed write {index + 1}/{len(raw_frames)}",
            f"outcome: {outcome}",
        ])
        for index, frame in enumerate(raw_frames)
    ]
    stem = f"task{args.task_id:02d}-state{args.state_index:03d}-{args.branch}"
    video_path = output_dir / f"{stem}-excluded-live.mp4"
    if frames:
        _write_video(video_path, frames)

    summary = {
        "schema": "sentinel-unified-libero-excluded-live-capture-v1",
        "included_in_experiment_statistics": False,
        "capture_boundary": "one frame rendered after each authorized NativeGateway writer environment step; forecast rollouts excluded",
        "episode": {"task_id": args.task_id, "state_index": args.state_index, "seed": seed,
                    "branch": args.branch, "task_goal": task_goal},
        "outcome": {
            "success": bool(episode.get("success")), "crashed": bool(episode.get("crashed")),
            "steps": int(episode.get("steps", 0)), "denied_final_reason": denied_reason,
        },
        "capture": {
            "frames": len(frames), "actual_writes": len(actual), "fps": 20,
            "actual_env_step_hashes": [row.get("actual_env_step_hash") for row in actual],
            "frame_telemetry": [{
                "frame": index, "step": int(row["step"]), "cycle": int(row["cycle"]),
                "within_cycle": int(row["within_cycle"]),
                "certified_prefix_length": (
                    recovery_cycles[int(row["cycle"])]["prefix_length"]
                    if int(row["cycle"]) in recovery_cycles else None
                ),
            } for index, row in enumerate(actual)],
            "video": ({"path": video_path.name, "sha256": _sha256(video_path)} if frames else None),
        },
        "safe_prefix_recovery": {
            "successful_recovery_count": len(recovery_cycles),
            "selected_horizons": [item["selected_execution_horizon"] for item in recovery_cycles.values()],
            "first_recovery_cycle": min(recovery_cycles) if recovery_cycles else None,
            "cycles": list(recovery_cycles.values()),
        },
        "sources": {
            "source_config": str(source_config), "source_config_sha256": _sha256(source_config),
            "capture_config_sha256": _sha256(capture_config),
            "frozen_runner_sha256": _sha256(Path(runner.__file__).resolve()),
            "capture_helper_sha256": _sha256(Path(__file__).resolve()),
            "run_manifest_sha256": _sha256(output_dir / "manifest.json"),
            "steps_sha256": _sha256(output_dir / "steps.jsonl"),
            "per_episode_sha256": _sha256(output_dir / "per_episode.jsonl"),
        },
        "geometry": {
            "joint_graph_dimension": 9 if bool(config.get("certify_moving_fingers", False)) else 7,
            "tracking_reserve_rad": config.get("geometry", {}).get("tracking_reserve_rad"),
            "tracking_reserve_slide_m": config.get("geometry", {}).get("tracking_reserve_slide_m"),
        },
        "deadline_timestamp": config.get("deadline_timestamp"),
        "runner_return_code": return_code,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "video": summary["capture"]["video"],
                      "outcome": summary["outcome"]}, indent=2))
    return 0 if return_code == 0 and not episode.get("crashed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
