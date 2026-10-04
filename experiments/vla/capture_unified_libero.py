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


TASK_ID = 0
STATE_INDEX = 40
SEED = 2026100400
BRANCH = "delta_evc"


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
    parser.add_argument("--config", type=Path, required=True, help="frozen A2 config to clone")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("output-dir must not exist")

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["output_dir"] = str(output_dir)
    config.pop("deadline_timestamp", None)
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
        runner.BRANCHES = (BRANCH,)
        runner._episode_specs = lambda unused: [{
            "phase": "formal", "task_id": TASK_ID, "state_index": STATE_INDEX, "seed": SEED,
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
    if len(raw_frames) != len(actual):
        raise RuntimeError(f"captured {len(raw_frames)} frames for {len(actual)} committed writes")

    denied_reason = denied[-1].get("reason") if denied else None
    outcome = "success" if episode.get("success") else "stopped"
    if episode.get("crashed"):
        outcome = f"crashed: {episode.get('error_type')}"
    elif denied_reason:
        outcome = f"denied: {denied_reason}"
    frames = [
        _annotate(frame, [
            "EXCLUDED LIVE DEMO | geometry + EVC + SmolVLA",
            f"task {TASK_ID} state {STATE_INDEX} | {BRANCH} | committed write {index + 1}/{len(raw_frames)}",
            f"outcome: {outcome}",
        ])
        for index, frame in enumerate(raw_frames)
    ]
    video_path = output_dir / "task00-state040-delta_evc-excluded-live.mp4"
    if frames:
        _write_video(video_path, frames)

    summary = {
        "schema": "sentinel-unified-libero-excluded-live-capture-v1",
        "included_in_experiment_statistics": False,
        "capture_boundary": "one frame rendered after each authorized NativeGateway writer environment step; forecast rollouts excluded",
        "episode": {"task_id": TASK_ID, "state_index": STATE_INDEX, "seed": SEED, "branch": BRANCH},
        "outcome": {
            "success": bool(episode.get("success")), "crashed": bool(episode.get("crashed")),
            "steps": int(episode.get("steps", 0)), "denied_final_reason": denied_reason,
        },
        "capture": {
            "frames": len(frames), "actual_writes": len(actual), "fps": 20,
            "actual_env_step_hashes": [row.get("actual_env_step_hash") for row in actual],
            "video": ({"path": video_path.name, "sha256": _sha256(video_path)} if frames else None),
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
