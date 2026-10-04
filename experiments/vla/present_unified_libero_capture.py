#!/usr/bin/env python3
"""Add an explanatory panel to an excluded live LIBERO capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TASK0_GOAL = "Pick up the black bowl between the plate and the ramekin and place it on the plate."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/local/share/fonts") / name,
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"DejaVu font not found: {name}")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _text_block(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, color: tuple[int, int, int],
    width: int, line_height: int,
) -> int:
    x, y = xy
    lines = _wrap(draw, text, font, width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += line_height
    return y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--playback-fps", type=float, default=5.0)
    args = parser.parse_args()
    if not np.isfinite(args.playback_fps) or args.playback_fps <= 0:
        raise ValueError("playback-fps must be positive and finite")
    input_video = args.input_video.resolve()
    summary_path = args.capture_summary.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("output-dir must not exist")
    output_dir.mkdir(parents=True)

    capture = json.loads(summary_path.read_text(encoding="utf-8"))
    input_hash = _sha256(input_video)
    if input_hash != capture["capture"]["video"]["sha256"]:
        raise RuntimeError("input video hash does not match capture summary")
    expected_frames = int(capture["capture"]["frames"])
    expected_writes = int(capture["capture"]["actual_writes"])
    if expected_frames != expected_writes:
        raise RuntimeError("live capture must contain exactly one frame per actual write")
    episode = capture["episode"]
    task_id = int(episode.get("task_id", 0))
    state_index = int(episode.get("state_index", 40))
    branch = str(episode.get("branch", "delta_evc"))
    task_goal = str(episode.get("task_goal") or TASK0_GOAL)
    if not task_goal.endswith("."):
        task_goal += "."
    graph_dimension = int(capture.get("geometry", {}).get("joint_graph_dimension", 7))
    pipeline = [
        "SmolVLA predicts a 50-action chunk",
        "Fixed overlap aggregation selects 10 actions",
        f"{graph_dimension}D Panda joint graph checks arm" + (" + gripper motion" if graph_dimension == 9 else " motion"),
        "EVC issues a one-use permit for exact action bytes",
        "LIBERO Panda controller executes the authorized write",
    ]
    success = bool(capture["outcome"].get("success"))
    outcome_label = f"Outcome: {'success' if success else 'stopped'} ({expected_writes} writes)"
    tracking_reserve = capture.get("geometry", {}).get("tracking_reserve_rad", 1e-5)
    frame_telemetry = capture["capture"].get("frame_telemetry")
    if frame_telemetry is None:
        frame_telemetry = [{} for _ in range(expected_frames)]
    if len(frame_telemetry) != expected_frames:
        raise RuntimeError("frame telemetry must have one entry per recorded writer frame")
    slowdown = 20.0 / args.playback_fps

    title_font = _font(29, bold=True)
    heading_font = _font(22, bold=True)
    body_font = _font(19)
    small_font = _font(16)
    reader = imageio.get_reader(input_video)
    stem = f"task{task_id:02d}-state{state_index:03d}-{branch}"
    output_video = output_dir / f"{stem}-excluded-live-explained.mp4"
    writer = imageio.get_writer(
        output_video, fps=args.playback_fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, quality=8
    )
    rendered = 0
    try:
        for index, raw in enumerate(reader):
            canvas = Image.new("RGB", (1280, 720), (15, 18, 24))
            live = Image.fromarray(np.asarray(raw, dtype=np.uint8)).convert("RGB").resize((720, 720), Image.Resampling.BICUBIC)
            canvas.paste(live, (0, 0))
            draw = ImageDraw.Draw(canvas)
            x, width = 755, 490
            y = 28
            draw.text((x, y), "Excluded live MuJoCo demo", font=title_font, fill=(245, 247, 250))
            y += 48
            draw.text((x, y), "Task goal", font=heading_font, fill=(104, 194, 255))
            y = _text_block(draw, (x, y + 31), task_goal, body_font, (235, 238, 242), width, 27) + 18
            draw.text((x, y), "Authorized pipeline", font=heading_font, fill=(104, 194, 255))
            y += 35
            for number, item in enumerate(pipeline, 1):
                y = _text_block(draw, (x, y), f"{number}. {item}", body_font, (235, 238, 242), width, 25) + 7
            y += 8
            prefix = frame_telemetry[index].get("certified_prefix_length")
            recovery_label = (
                f"A6 event: certified prefix {int(prefix)} -> execute, then re-observe"
                if prefix is not None else "A6 event: standard certified horizon"
            )
            draw.rounded_rectangle((x, y, 1245, y + 94), radius=10, fill=(28, 92, 60))
            draw.text((x + 16, y + 10), f"Committed write {index + 1} / {expected_writes}", font=heading_font, fill=(255, 255, 255))
            draw.text((x + 16, y + 39), outcome_label, font=small_font, fill=(224, 255, 235))
            draw.text((x + 16, y + 62), recovery_label, font=small_font, fill=(224, 255, 235))
            footer = (
                "Recorded after authorized environment writes only. "
                f"{args.playback_fps:g} fps playback / 20 Hz simulator; slowed {slowdown:g}x. "
                f"Simulation {expected_frames / 20:.1f} s; playback {expected_frames / args.playback_fps:.1f} s; "
                f"wall time {float(capture['elapsed_seconds']):.1f} s. "
                f"Geometry used a {float(tracking_reserve):.0e} rad tracking reserve. "
                "Excluded from experiment statistics."
            )
            _text_block(draw, (x, 626), footer, small_font, (166, 174, 187), width, 20)
            writer.append_data(np.asarray(canvas))
            rendered += 1
    finally:
        reader.close()
        writer.close()
    if rendered != expected_frames:
        raise RuntimeError(f"decoded {rendered} frames, expected {expected_frames}")

    metadata = {
        "schema": "sentinel-unified-libero-excluded-live-presentation-v1",
        "included_in_experiment_statistics": False,
        "visual_transform": "recorded live frames scaled to 720x720 at left; explanatory text panel added at right; no frames or motion synthesized",
        "playback": {"fps": args.playback_fps, "simulator_hz": 20, "slowdown_factor": slowdown,
                     "frames": rendered, "simulator_seconds": rendered / 20,
                     "playback_seconds": rendered / args.playback_fps,
                     "source_wall_clock_seconds": float(capture["elapsed_seconds"])},
        "episode": {"task_id": task_id, "state_index": state_index, "branch": branch},
        "task_goal": task_goal,
        "pipeline": pipeline,
        "joint_graph_dimension": graph_dimension,
        "tracking_reserve_rad": tracking_reserve,
        "outcome": capture["outcome"],
        "safe_prefix_recovery": capture.get("safe_prefix_recovery"),
        "source": {
            "input_video": str(input_video), "input_video_sha256": input_hash,
            "capture_summary": str(summary_path), "capture_summary_sha256": _sha256(summary_path),
            "capture_helper_sha256": capture["sources"]["capture_helper_sha256"],
            "frozen_runner_sha256": capture["sources"]["frozen_runner_sha256"],
        },
        "output": {"video": output_video.name, "video_sha256": _sha256(output_video)},
        "postprocessor_sha256": _sha256(Path(__file__).resolve()),
    }
    metadata_path = output_dir / "presentation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(output_video), "video_sha256": metadata["output"]["video_sha256"],
                      "metadata": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
