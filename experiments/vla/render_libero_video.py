#!/usr/bin/env python3
"""Wrap the predeclared native LIBERO rollout in a paper-ready explainer.

This script is intentionally a presentation-only transform.  It validates the
frozen protocol, formal manifest, action trace, and selected native video before
rendering.  Every output video frame contains exactly one decoded source frame;
no simulator or policy is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageDraw, ImageFont


CANVAS = (1920, 1080)
VIDEO_BOX = (34, 34, 1046, 1046)
PANEL = (1080, 34, 1886, 1046)
EXPECTED_TASK_ID = 0
EXPECTED_STATE_INDEX = 0
KNOWN_TRANSLATIONS = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate":
        "拿起盘子和小烤盅之间的黑色碗，并把它放到盘子上。",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, width: int) -> list[str]:
    """Pixel-wrap English on spaces and CJK text by character."""
    if not text:
        return []
    tokens = text.split(" ") if " " in text else list(text)
    separator = " " if " " in text else ""
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else current + separator + token
        if draw.textbbox((0, 0), candidate, font=face)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = token
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    face: ImageFont.ImageFont,
    fill: str,
    width: int,
    spacing: int = 8,
) -> int:
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag国", font=face)[3]
    for line in wrap(draw, text, face, width):
        draw.text((x, y), line, font=face, fill=fill)
        y += line_height + spacing
    return y


def rounded_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str) -> None:
    draw.rounded_rectangle(box, radius=22, fill=fill, outline="#334155", width=2)


def fit_frame(source: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS
    )
    return resized


def inspect_video(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, "source must contain exactly one video stream")
        require(len(container.streams.audio) == 0, "native rollout unexpectedly contains audio")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        require(rate is not None and float(rate) > 0, "source video has no usable frame rate")
        frames = 0
        first_pts = None
        last_pts = None
        for decoded in container.decode(stream):
            frames += 1
            first_pts = decoded.pts if first_pts is None else first_pts
            last_pts = decoded.pts
        require(frames > 0, "source video has no decodable frames")
        duration = frames / float(rate)
        return {
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "pixel_format": stream.codec_context.format.name,
            "average_rate": f"{rate.numerator}/{rate.denominator}",
            "fps": float(rate),
            "frames": frames,
            "duration_seconds_by_frame_count": duration,
            "first_pts": first_pts,
            "last_pts": last_pts,
            "time_base": str(stream.time_base),
            "audio_streams": 0,
        }


def find_video_artifact(manifest: dict[str, Any], source: Path) -> dict[str, Any]:
    source_hash = sha256(source)
    source_bytes = source.stat().st_size
    matches = [
        item
        for item in manifest.get("video_artifacts", [])
        if item.get("sha256") == source_hash and item.get("bytes") == source_bytes
    ]
    require(len(matches) == 1, "source does not uniquely match the formal manifest video artifact")
    return matches[0]


def formal_result(manifest: dict[str, Any]) -> tuple[bool, int]:
    task = manifest["results"]["tasks"][str(EXPECTED_TASK_ID)]
    states = task["initial_state_indices"]
    require(EXPECTED_STATE_INDEX in states, "formal result omits predeclared initial state 0")
    position = states.index(EXPECTED_STATE_INDEX)
    require(position < int(task["completed_episodes"]), "selected rollout was not completed")
    success = task["per_episode_success"][position]
    require(isinstance(success, bool), "selected rollout success label is not boolean")
    return success, int(task["episode_seeds"][position])


def validate_trace(trace_path: Path, manifest: dict[str, Any], seed: int) -> tuple[str, int]:
    receipt = manifest["trace"]
    require(sha256(trace_path) == receipt["sha256"], "action trace hash differs from formal manifest")
    require(trace_path.stat().st_size == receipt["bytes"], "action trace size differs from formal manifest")
    require(receipt["gate_mode"] == "shadow_observe_only", "formal run was not shadow-observe-only")
    require(receipt["interventions"] == 0, "formal trace reports a shadow intervention")
    require(receipt["max_postprocess_to_env_step_abs_diff"] == 0.0, "formal action path changed an action")

    plan_seen = False
    instruction: str | None = None
    selected_env_steps = 0
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("task_id") != EXPECTED_TASK_ID or event.get("initial_state_index") != EXPECTED_STATE_INDEX:
                continue
            if event.get("event") == "episode_plan":
                require(event.get("seed") == seed, "trace seed differs from formal manifest")
                plan_seen = True
            elif event.get("event") == "raw_policy_chunk" and instruction is None:
                instruction = event["input_context"]["instruction"]
            elif event.get("event") == "env_step_action":
                gate = event.get("shadow_gate", {})
                require(gate.get("mode") == "observe_only", "selected rollout gate mode changed")
                require(gate.get("intervention") is False, "selected rollout contains an intervention")
                require(event.get("postprocess_to_env_step_max_abs_diff") == 0.0, "selected rollout action changed")
                selected_env_steps += 1
    require(plan_seen, "selected rollout episode plan is absent from action trace")
    require(instruction is not None, "selected rollout instruction is absent from action trace")
    require(selected_env_steps > 0, "selected rollout has no env.step actions")
    return instruction, selected_env_steps


def validate_inputs(
    protocol_path: Path,
    manifest_path: Path,
    trace_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    require(protocol["state"] == "frozen_before_final_run", "protocol was not frozen before the final run")
    selection = protocol["evaluation"]["video_selection"]
    require(selection["task_id"] == EXPECTED_TASK_ID, "protocol selected a different video task")
    require(selection["initial_state_index"] == EXPECTED_STATE_INDEX, "protocol selected a different video state")
    require(manifest["protocol_sha256"] == sha256(protocol_path), "formal manifest uses a different protocol")
    require(manifest["mode"] == "final", "manifest is not a final evaluation")
    require(manifest["included_in_final_metrics"] is True, "manifest excludes this run from final metrics")
    require(manifest["status"] in {"complete", "failed_with_frozen_grid_accounting"}, "unexpected formal status")
    require(manifest["checkpoint"]["repo_id"] == "lerobot/smolvla_libero", "unexpected checkpoint")
    artifact = find_video_artifact(manifest, source_path)
    success, seed = formal_result(manifest)
    instruction, env_steps = validate_trace(trace_path, manifest, seed)
    simulation_fps = int(protocol["evaluation"]["fps"])
    require(simulation_fps > 0, "protocol simulation frame rate is invalid")
    return {
        "protocol": protocol,
        "manifest": manifest,
        "artifact": artifact,
        "success": success,
        "seed": seed,
        "instruction": instruction,
        "env_steps": env_steps,
        "simulation_fps": simulation_fps,
    }


def draw_frame(
    rgb: Image.Image,
    frame_index: int,
    frame_count: int,
    fps: float,
    instruction: str,
    success: bool,
    seed: int,
    native_size: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    image = Image.new("RGB", CANVAS, "#07111f")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(VIDEO_BOX, radius=28, fill="#020617", outline="#334155", width=2)
    fitted = fit_frame(rgb, (54, 54, 1026, 1026))
    fx = 54 + (972 - fitted.width) // 2
    fy = 54 + (972 - fitted.height) // 2
    image.paste(fitted, (fx, fy))
    draw.rounded_rectangle((72, 72, 500, 126), radius=14, fill="#07111f")
    draw.text((94, 83), "实际 LIBERO RGB · ACTUAL RGB", font=fonts["small_bold"], fill="#e2e8f0")

    rounded_card(draw, PANEL, "#0b1728")
    x, y, width = 1118, 66, 730
    draw.text((x, y), "原生闭环操作演示", font=fonts["title"], fill="#f8fafc")
    y += 54
    draw.text((x, y), "NATIVE CLOSED-LOOP ROLLOUT", font=fonts["label"], fill="#38bdf8")
    y += 52

    rounded_card(draw, (1108, y, 1858, y + 142), "#102137")
    draw.text((x, y + 18), "官方模型 / Official model", font=fonts["label"], fill="#94a3b8")
    draw.text((x, y + 52), "lerobot/smolvla_libero", font=fonts["body_bold"], fill="#f8fafc")
    draw.text((x, y + 91), "LIBERO-Spatial · Panda robot", font=fonts["body"], fill="#cbd5e1")
    y += 164

    draw.text((x, y), "任务目标 / TASK", font=fonts["label"], fill="#94a3b8")
    y += 34
    y = draw_wrapped(draw, (x, y), instruction, fonts["body"], "#f8fafc", width, 7)
    translation = KNOWN_TRANSLATIONS.get(instruction)
    if translation:
        y += 6
        y = draw_wrapped(draw, (x, y), translation, fonts["body"], "#a5f3fc", width, 7)
    y += 24

    rounded_card(draw, (1108, y, 1858, y + 138), "#102137")
    draw.text((x, y + 18), "预声明样本 / PREDECLARED CASE", font=fonts["label"], fill="#94a3b8")
    draw.text((x, y + 54), f"Task 0 · Initial state 0 · Seed {seed}", font=fonts["body_bold"], fill="#f8fafc")
    elapsed = frame_index / fps
    total = frame_count / fps
    draw.text((x, y + 94), f"仿真录像时间  {elapsed:05.1f}s / {total:05.1f}s", font=fonts["body"], fill="#cbd5e1")
    y += 160

    draw.text((x, y), "动作链 / ACTION PATH", font=fonts["label"], fill="#94a3b8")
    y += 38
    chain = [
        "1  输入 / INPUT · RGB + 8D state + instruction",
        "2  策略 / POLICY · official SmolVLA",
        "3  后处理 / POSTPROCESS · official scaler",
        "4  执行 / APPLY · LIBERO env.step",
    ]
    for item in chain:
        draw.rounded_rectangle((x, y, x + width, y + 35), radius=10, fill="#172b46")
        draw.text((x + 14, y + 5), item, font=fonts["chain"], fill="#e2e8f0")
        y += 39

    y += 2
    draw.text(
        (x, y),
        "影子记录器 / SHADOW LOGGER · 仅观察 · 0 干预 · 0 改写",
        font=fonts["chain"],
        fill="#86efac",
    )

    final_window = max(1, round(2.0 * fps))
    outcome_visible = frame_index >= frame_count - final_window
    outcome_fill = "#16a34a" if success else "#dc2626"
    outcome_text = "成功 / SUCCESS" if success else "失败 / FAILURE"
    status_text = outcome_text if outcome_visible else "结果由正式记录给出 / RECORDED OUTCOME"
    status_fill = outcome_fill if outcome_visible else "#334155"
    draw.rounded_rectangle((1108, 932, 1858, 996), radius=18, fill=status_fill)
    status_box = draw.textbbox((0, 0), status_text, font=fonts["body_bold"])
    draw.text((1483 - (status_box[2] - status_box[0]) // 2, 945), status_text, font=fonts["body_bold"], fill="#ffffff")

    progress = (frame_index + 1) / frame_count
    draw.rounded_rectangle((1108, 1012, 1858, 1024), radius=6, fill="#26374d")
    draw.rounded_rectangle((1108, 1012, 1108 + round(750 * progress), 1024), radius=6, fill="#38bdf8")
    draw.text((60, 1024), f"SOURCE {native_size[0]}×{native_size[1]}  ·  FRAME {frame_index + 1}/{frame_count}", font=fonts["tiny"], fill="#94a3b8")
    draw.text((1110, 1027), "仅排版、放大与字幕；未新增视觉证据。", font=fonts["tiny"], fill="#64748b")
    return image


def encode(
    source: Path,
    output: Path,
    poster: Path,
    metadata: dict[str, Any],
    font_path: Path,
) -> None:
    source_info = metadata["source_info"]
    simulation_fps = metadata["simulation_fps"]
    fps_fraction = Fraction(simulation_fps, 1)
    faces = {
        "title": font(font_path, 43),
        "label": font(font_path, 22),
        "body": font(font_path, 27),
        "body_bold": font(font_path, 29),
        "small": font(font_path, 22),
        "small_bold": font(font_path, 23),
        "chain": font(font_path, 20),
        "tiny": font(font_path, 15),
    }
    last_composed: Image.Image | None = None
    with av.open(str(source)) as input_container, av.open(str(output), "w") as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream("libx264", rate=fps_fraction)
        output_stream.width, output_stream.height = CANVAS
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "18", "preset": "medium"}
        for frame_index, decoded in enumerate(input_container.decode(input_stream)):
            composed = draw_frame(
                decoded.to_image().convert("RGB"),
                frame_index,
                source_info["frames"],
                simulation_fps,
                metadata["instruction"],
                metadata["success"],
                metadata["seed"],
                (source_info["width"], source_info["height"]),
                faces,
            )
            video_frame = av.VideoFrame.from_image(composed)
            for packet in output_stream.encode(video_frame):
                output_container.mux(packet)
            last_composed = composed
        for packet in output_stream.encode():
            output_container.mux(packet)
    require(last_composed is not None, "no source frame was rendered")
    last_composed.save(poster, format="PNG", optimize=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-video", type=Path, required=True)
    result.add_argument("--formal-manifest", type=Path, required=True)
    result.add_argument("--protocol", type=Path, required=True)
    result.add_argument("--action-trace", type=Path)
    result.add_argument("--font-path", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    for path in (args.source_video, args.formal_manifest, args.protocol, args.font_path):
        require(path.is_file(), f"required input is missing: {path}")
    manifest = load_json(args.formal_manifest)
    trace_path = args.action_trace or args.formal_manifest.parent / manifest["trace"]["path"]
    require(trace_path.is_file(), f"action trace is missing: {trace_path}")
    validated = validate_inputs(args.protocol, args.formal_manifest, trace_path, args.source_video)
    source_info = inspect_video(args.source_video)

    args.out.mkdir(parents=True, exist_ok=True)
    native_copy = args.out / "native-task00-state00.mp4"
    explained = args.out / "libero-task00-state00-explained.mp4"
    poster = args.out / "libero-task00-state00-poster.png"
    manifest_out = args.out / "video_manifest.json"
    explained_tmp = args.out / ".libero-task00-state00-explained.tmp.mp4"
    poster_tmp = args.out / ".libero-task00-state00-poster.tmp.png"

    shutil.copyfile(args.source_video, native_copy)
    require(sha256(native_copy) == sha256(args.source_video), "native copy is not byte-identical")
    render_metadata = {
        "source_info": source_info,
        "instruction": validated["instruction"],
        "success": validated["success"],
        "seed": validated["seed"],
        "simulation_fps": validated["simulation_fps"],
    }
    require(
        source_info["frames"] == validated["env_steps"],
        "native video frame count does not equal selected rollout env.step count",
    )
    encode(args.source_video, explained_tmp, poster_tmp, render_metadata, args.font_path)
    output_info = inspect_video(explained_tmp)
    require(output_info["width"] == CANVAS[0] and output_info["height"] == CANVAS[1], "output is not 1920x1080")
    require(output_info["frames"] == source_info["frames"], "output frame count differs from source")
    require(output_info["fps"] == validated["simulation_fps"], "output frame rate differs from protocol simulation rate")
    explained_tmp.replace(explained)
    poster_tmp.replace(poster)

    translation = KNOWN_TRANSLATIONS.get(validated["instruction"])
    receipt = {
        "schema": "sentinel-libero-video-wrapper-v1",
        "selection": {
            "rule": validated["protocol"]["evaluation"]["video_selection"]["rule"],
            "task_id": EXPECTED_TASK_ID,
            "initial_state_index": EXPECTED_STATE_INDEX,
            "episode_seed": validated["seed"],
            "outcome_source": "formal_manifest.results.tasks.0.per_episode_success at initial_state_index 0",
            "success": validated["success"],
        },
        "task": {
            "instruction_from_action_trace": validated["instruction"],
            "chinese_translation": translation,
            "translation_rule": "exact allowlist" if translation else "not rendered",
        },
        "provenance": {
            "checkpoint": validated["manifest"]["checkpoint"]["repo_id"],
            "checkpoint_revision": validated["manifest"]["checkpoint"]["revision"],
            "environment": "official LIBERO-Spatial simulated Panda robot",
            "formal_manifest": {"sha256": sha256(args.formal_manifest), "bytes": args.formal_manifest.stat().st_size},
            "protocol": {"sha256": sha256(args.protocol), "bytes": args.protocol.stat().st_size},
            "action_trace": {"sha256": sha256(trace_path), "bytes": trace_path.stat().st_size},
            "selected_env_step_actions": validated["env_steps"],
            "shadow_logger": {"mode": "observe_only", "interventions": 0, "action_changes": 0},
        },
        "native_video": {
            "formal_artifact_path": validated["artifact"]["path"],
            "source_sha256": sha256(args.source_video),
            "source_bytes": args.source_video.stat().st_size,
            "preserved_copy": native_copy.name,
            "preserved_copy_sha256": sha256(native_copy),
            **source_info,
        },
        "explained_video": {
            "path": explained.name,
            "sha256": sha256(explained),
            "bytes": explained.stat().st_size,
            **output_info,
            "playback_rate_basis": "one preserved native RGB frame per formal env.step at protocol evaluation fps",
        },
        "poster": {"path": poster.name, "sha256": sha256(poster), "bytes": poster.stat().st_size},
        "claim_boundary": [
            "The native frames are the predeclared official lerobot/smolvla_libero rollout in LIBERO with a Panda robot.",
            "The wrapper only upscales, lays out, and annotates those frames; it adds no visual evidence.",
            "The shadow logger observed the official action path and made no intervention or action change.",
            "This artifact is not an SO100 overlay, UR5e experiment, real-robot result, or safety guarantee.",
        ],
        "verification": {
            "native_copy_byte_identical": True,
            "full_source_decode": True,
            "full_output_decode": True,
            "frame_count_equal": True,
            "source_frames_equal_selected_env_step_actions": True,
            "output_frame_rate_matches_protocol_simulation_fps": True,
            "output_resolution": list(CANVAS),
        },
    }
    manifest_out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(explained), "poster": str(poster), "manifest": str(manifest_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
