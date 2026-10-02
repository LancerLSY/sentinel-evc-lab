#!/usr/bin/env python3
"""Render audited LIBERO companions from a completed signed native run.

This tool never invokes the policy.  It verifies a signed run bundle, resets the
same native environment, submits the retained official float32 requests, and
requires every action identity, observation hash, step outcome, and terminal
result to match before publishing video or pose assets.  High-resolution camera
renders are presentation views captured after each verified native transition;
the signed 360 px observations remain the trajectory audit authority.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from export_native_scene import NativeSceneExporter
from run_sentinel_libero import (
    _array_identity,
    _extract_observation,
    _extract_step_outcome,
    _file_sha256,
    _jsonable,
    _observation_hash,
    _observation_summary,
)


SCHEMA = "sentinel-native-companion-media-v1"
DEFAULT_TARGETS = ((0, 46, "predeclared"), (4, 46, "predeclared"),
                   (5, 46, "predeclared"), (0, 47, "posthoc_failure_diagnostic"))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _md5_bytes(value: bytes) -> str:
    return "md5:" + hashlib.md5(value).hexdigest()


def _load_font(path: Path | None, size: int) -> ImageFont.ImageFont:
    if path is not None:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _find_native_sim(environment: Any) -> Any:
    queue = [environment]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        sim = getattr(candidate, "sim", None)
        if sim is not None and callable(getattr(sim, "render", None)):
            return sim
        for name in ("_env", "env", "unwrapped"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
        envs = getattr(candidate, "envs", None)
        if envs:
            queue.extend(list(envs))
    raise RuntimeError("live native simulator render method was not found")


def _camera(sim: Any, name: str, size: int) -> np.ndarray:
    image = np.asarray(sim.render(camera_name=name, width=size, height=size, depth=False))
    if image.shape != (size, size, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected {name} render: shape={image.shape}, dtype={image.dtype}")
    return np.ascontiguousarray(image)


def _canvas(
    left: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    instruction: str,
    source_frame: dict[str, Any],
    episode_status: str,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    height, width = left.shape[:2]
    header = 156
    canvas = Image.new("RGB", (width * 2, height + header), (10, 15, 24))
    canvas.paste(Image.fromarray(left), (0, header))
    canvas.paste(Image.fromarray(right), (width, header))
    draw = ImageDraw.Draw(canvas)
    step = int(source_frame["step"])
    action = source_frame.get("action")
    if action is None:
        action_text = "reset"
        audit = "action=N/A observation=MATCH outcome=N/A"
    else:
        xyz = " ".join(f"{value:+.3f}" for value in action[:3])
        rotation = " ".join(f"{value:+.3f}" for value in action[3:6])
        action_text = f"Δxyz[{xyz}] Δrot[{rotation}] gripper[{action[6]:+.3f}]"
        audit = "action=MATCH observation=MATCH outcome=MATCH"
    permit = source_frame.get("permit_id")
    permit_label = str(permit)[:8] if permit else "none"
    cursors = source_frame["cursors"]
    outcome = source_frame.get("outcome") or {}
    outcome_text = (
        f"reward={outcome.get('reward', 'N/A')} terminated={outcome.get('terminated', 'N/A')} "
        f"truncated={outcome.get('truncated', 'N/A')} episode={episode_status}"
    )
    lines = [
        title,
        f"Instruction: {instruction}",
        f"Step {step:03d} / {step / 20.0:05.2f}s | {action_text}",
        f"Signed source: decision={source_frame['decision']} permit={permit_label} "
        f"cursors={cursors['submitted']}/{cursors['accepted']}/{cursors['observed']}",
        f"Audit: {audit} | {outcome_text}",
    ]
    y = 7
    for line in lines:
        draw.text((14, y), line, fill=(238, 244, 252), font=font)
        y += 24
    draw.text((14, y), f"agentview ({width} px presentation render)",
              fill=(238, 244, 252), font=font)
    draw.text((width + 14, y), f"eye-in-hand ({width} px presentation render)",
              fill=(238, 244, 252), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _import_and_verify(
    archive: Path, public_key: Path, temporary: Path, run_id: str
) -> tuple[Path, dict[str, Any]]:
    """Use the product's bounded importer before reading any archive member."""
    from sentinel_evc.native_runs import NativeRunStore

    store = NativeRunStore(temporary / "native-runs")
    imported = store.import_archive(archive.read_bytes(), run_id, public_key.read_bytes())
    return store.root / run_id / "bundle", {
        "verified": True,
        "scope": imported["verification"]["scope"],
        "trust": imported["verification"]["trust"],
    }


def _action(frame: dict[str, Any]) -> np.ndarray:
    values = np.asarray(frame["action"], dtype=np.float32).reshape(1, 7)
    if _array_identity(values)["sha256"] != frame["action_bytes_hash"]:
        raise RuntimeError(f"retained action identity mismatch at step {frame['step']}")
    return values


def _same_outcome(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return _canonical(_jsonable(actual)) == _canonical(_jsonable(expected))


def _write_gzip_json(path: Path, value: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True, allow_nan=False)
        handle.write("\n")


def _episode_result(result: dict[str, Any], task_id: int, state_index: int) -> dict[str, Any]:
    matches = [row for row in result["episodes"]
               if row["task_id"] == task_id and row["initial_state_index"] == state_index]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source result for task {task_id} state {state_index}")
    return matches[0]


def render(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.scripts.lerobot_eval import close_envs
    from lerobot.utils.random_utils import set_seed

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="sentinel-companion-") as temp_name:
        bundle, signature = _import_and_verify(
            args.bundle, args.public_key, Path(temp_name), args.run_id
        )
        result = json.loads((bundle / "result.json").read_text())
        replay = json.loads((bundle / "replay.json").read_text())
        config = json.loads((bundle / "config.json").read_text())
        if result["status"] != "complete" or result["mode"] != "active":
            raise RuntimeError("companion source must be a complete active run")

        source_episodes = {
            (row["task_id"], row["initial_state_index"]): row for row in replay["episodes"]
        }
        targets = [key for key in DEFAULT_TARGETS if (key[0], key[1]) in source_episodes]
        if len(targets) != len(DEFAULT_TARGETS):
            raise RuntimeError("source replay does not contain every frozen companion target")

        task_ids = [int(value) for value in config["task_ids"]]
        env_cfg = LiberoEnv(
            task="libero_spatial", task_ids=task_ids, fps=20, init_states=True,
            hard_reset=True, control_mode="relative", max_parallel_tasks=1,
            observation_height=360, observation_width=360,
        )
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
        font = _load_font(args.font, 18)
        assets: list[dict[str, Any]] = []
        scene_assets: dict[int, dict[str, Any]] = {}
        try:
            for task_id, state_index, purpose in targets:
                source = source_episodes[(task_id, state_index)]
                source_result = _episode_result(result, task_id, state_index)
                env = envs["libero_spatial"][task_id]
                env.envs[0].init_state_id = state_index
                set_seed(int(source_result["seed"]))
                reset = env.reset(seed=[int(source_result["seed"])], options={"new_rollout": True})
                observation = _extract_observation(reset)
                reset_hash = _observation_hash(_observation_summary(observation))
                if reset_hash != source["frames"][0]["observation_hash"]:
                    raise RuntimeError(f"reset observation mismatch for task {task_id} state {state_index}")

                instruction = str(env.call("task_description")[0])
                if instruction != source_result["instruction"]:
                    raise RuntimeError("live task instruction differs from signed result")
                exporter = NativeSceneExporter.from_environment(
                    env, task_id, task_suite="libero_spatial", task_name=instruction
                )
                expected_model = source["frames"][0]["modelId"]
                if exporter.model_id != expected_model:
                    raise RuntimeError("live compiled model identity differs from signed replay")
                if task_id not in scene_assets:
                    scene_path = args.output_dir / f"viewer-model-task{task_id}.json.gz"
                    _write_gzip_json(scene_path, exporter.export_scene())
                    scene_assets[task_id] = {
                        "path": scene_path.name, "bytes": scene_path.stat().st_size,
                        "sha256": _file_sha256(scene_path), "modelId": exporter.model_id,
                    }

                sim = _find_native_sim(env)
                pose_path = args.output_dir / f"poses-task{task_id:02d}-state{state_index:02d}.jsonl.gz"
                video_path = args.output_dir / f"companion-task{task_id:02d}-state{state_index:02d}.mp4"
                writer = imageio.get_writer(video_path, fps=20, codec="libx264", pixelformat="yuv420p",
                                             ffmpeg_log_level="error", macro_block_size=2)
                first_md5 = last_md5 = None
                compared_poses = 0
                try:
                    with gzip.open(pose_path, "wt", encoding="utf-8") as pose_file:
                        for index, source_frame in enumerate(source["frames"]):
                            if index:
                                action = _action(source_frame)
                                output = env.step(action)
                                observation = _extract_observation(output)
                                actual_hash = _observation_hash(_observation_summary(observation))
                                if actual_hash != source_frame["observation_hash"]:
                                    raise RuntimeError(f"observation mismatch at task {task_id} state {state_index} step {index}")
                                actual_outcome = _extract_step_outcome(output)
                                if not _same_outcome(actual_outcome, source_frame["outcome"]):
                                    raise RuntimeError(f"outcome mismatch at task {task_id} state {state_index} step {index}")
                                action_values = source_frame["action"]
                            else:
                                action_values = None

                            pose = exporter.capture_frame(index, step=index)
                            if source_frame.get("qpos"):
                                compared_poses += 1
                                if _canonical(pose["qpos"]) != _canonical(source_frame["qpos"]):
                                    raise RuntimeError(f"retained qpos mismatch at task {task_id} state {state_index} step {index}")
                            pose_file.write(json.dumps(pose, separators=(",", ":"), sort_keys=True,
                                                       ensure_ascii=False, allow_nan=False) + "\n")

                            left = _camera(sim, "agentview", args.render_size)
                            right = _camera(sim, "robot0_eye_in_hand", args.render_size)
                            terminal = index == len(source["frames"]) - 1
                            status = ("SUCCESS" if source_result["success"] else "FAILED") if terminal else "RUNNING"
                            title = (
                                f"Sentinel active replay · task{task_id}/state{state_index}"
                                if purpose == "predeclared"
                                else f"POSTHOC FAILURE DIAGNOSTIC · task{task_id}/state{state_index} · EXCLUDED"
                            )
                            frame = _canvas(
                                left, right, title=title, instruction=instruction,
                                source_frame=source_frame, episode_status=status, font=font,
                            )
                            frame_md5 = _md5_bytes(frame.tobytes())
                            first_md5 = first_md5 or frame_md5
                            last_md5 = frame_md5
                            writer.append_data(frame)
                finally:
                    writer.close()

                if int(source_result["env_steps"]) != len(source["frames"]) - 1:
                    raise RuntimeError("signed result step count differs from replay")
                assets.append({
                    "taskId": task_id, "initialStateIndex": state_index,
                    "purpose": purpose, "instruction": instruction,
                    "sourceEpisodeId": source["episode_id"],
                    "sourceSuccess": bool(source_result["success"]),
                    "frames": len(source["frames"]), "actionsVerified": len(source["frames"]) - 1,
                    "observationHashesVerified": len(source["frames"]),
                    "outcomesVerified": len(source["frames"]) - 1,
                    "retainedPosesCompared": compared_poses,
                    "mismatches": 0, "fps": 20,
                    "durationSeconds": len(source["frames"]) / 20.0,
                    "modelId": exporter.model_id,
                    "video": {"path": video_path.name, "bytes": video_path.stat().st_size,
                              "sha256": _file_sha256(video_path),
                              "firstFrameMd5": first_md5, "lastFrameMd5": last_md5,
                              "width": args.render_size * 2, "height": args.render_size + 156},
                    "poses": {"path": pose_path.name, "bytes": pose_path.stat().st_size,
                              "sha256": _file_sha256(pose_path),
                              "provenance": "fresh audited native replay; additional per-step poses are derived, not additional signed formal measurements"},
                })
        finally:
            close_envs(envs)

        manifest = {
            "schema": SCHEMA, "status": "complete",
            "source": {
                "runId": result["run_id"], "bundle": args.bundle.name,
                "bundleSha256": _file_sha256(args.bundle),
                "publicKeySha256": _file_sha256(args.public_key),
                "resultSha256": _file_sha256(bundle / "result.json"),
                "replaySha256": _file_sha256(bundle / "replay.json"),
                "signatureVerification": signature,
                "sourceCommit": args.run_source_commit,
            },
            "generator": {
                "path": Path(__file__).name,
                "sha256": _file_sha256(Path(__file__)),
                "sourceCommit": args.source_commit,
            },
            "rendering": {
                "trajectoryAuthority": "signed 360 px native observations",
                "presentationViews": ["agentview", "robot0_eye_in_hand"],
                "presentationRenderSize": [args.render_size, args.render_size],
                "presentationCapture": "same live simulator after each verified reset or env.step; no physics step added",
                "frameRate": 20, "interpolation": False, "policyInference": False,
            },
            "sceneAssets": [scene_assets[key] for key in sorted(scene_assets)],
            "media": assets,
            "claimLimits": [
                "companions replay retained official actions and are excluded from benchmark counts",
                "the posthoc failure diagnostic was selected only because it is the first failed active cell in frozen order",
                "presentation camera pixels are not substituted for the signed model-input observation hashes",
                "fault authorization results have no scene replay and are intentionally absent",
            ],
        }
        manifest_path = args.output_dir / "media-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                            ensure_ascii=False, allow_nan=False) + "\n")
        return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = render(_parse_args())
    print(json.dumps({"status": output["status"], "media": len(output["media"])}, sort_keys=True))
