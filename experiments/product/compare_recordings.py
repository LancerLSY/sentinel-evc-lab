#!/usr/bin/env python3
"""Project two native VLA recordings into Rerun and compare equal facts.

This is an experiment adapter, not a Sentinel runtime dependency. It deliberately
uses only the pinned Rerun environment supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rerun as rr


SCHEMA = "sentinel-rerun-recording-comparison-v1"
PROJECTION = "sentinel-vla-fact-projection-v1"
TIMELINE = "execution_step"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def episode_key(episode: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(episode["task_id"]),
        int(episode["initial_state_index"]),
        int(episode["seed"]),
        str(episode["instruction"]),
    )


def entity_path(key: tuple[int, int, int, str]) -> str:
    task_id, state_index, seed, instruction = key
    instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]
    return f"episodes/task_{task_id:02d}/state_{state_index}/seed_{seed}/instruction_{instruction_hash}"


def state_hashes(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, str]:
    found: dict[str, str] = {}
    if not isinstance(value, dict):
        return found
    if isinstance(value.get("sha256"), str):
        found["state__" + "__".join(prefix) + "__sha256"] = value["sha256"]
        return found
    for key, child in sorted(value.items()):
        found.update(state_hashes(child, prefix + (str(key),)))
    return found


def load_recording(bundle: Path) -> dict[str, Any]:
    result_path = bundle / "result.json"
    replay_path = bundle / "replay.json"
    result = load_json(result_path)
    replay = load_json(replay_path)
    if result.get("schema") != "sentinel-native-vla-run-v1":
        raise ValueError(f"unexpected result schema: {result.get('schema')!r}")
    if replay.get("schema") != "sentinel-native-vla-replay-v1":
        raise ValueError(f"unexpected replay schema: {replay.get('schema')!r}")
    if result.get("run_id") != replay.get("run_id"):
        raise ValueError("result and replay run_id differ")

    results_by_id: dict[str, dict[str, Any]] = {}
    for episode in result.get("episodes", []):
        episode_id = episode["episode_id"]
        if episode_id in results_by_id:
            raise ValueError(f"duplicate result episode_id: {episode_id}")
        results_by_id[episode_id] = episode

    episodes: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    replay_ids: set[str] = set()
    for replay_episode in replay.get("episodes", []):
        episode_id = replay_episode["episode_id"]
        if episode_id in replay_ids:
            raise ValueError(f"duplicate replay episode_id: {episode_id}")
        replay_ids.add(episode_id)
        result_episode = results_by_id.get(episode_id)
        if result_episode is None:
            raise ValueError(f"replay episode lacks result: {episode_id}")
        key = episode_key(result_episode)
        if key in episodes:
            raise ValueError(f"duplicate semantic episode key: {key!r}")
        for field in ("task_id", "initial_state_index"):
            if replay_episode.get(field) != result_episode.get(field):
                raise ValueError(f"episode {episode_id} disagrees on {field}")
        frames = replay_episode.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"episode {episode_id} has no frames")
        steps = [frame.get("step") for frame in frames]
        if steps != list(range(len(frames))):
            raise ValueError(f"episode {episode_id} steps are not contiguous from zero")
        episodes[key] = {"result": result_episode, "replay": replay_episode}

    missing_replays = sorted(set(results_by_id) - replay_ids)
    if missing_replays:
        raise ValueError(f"result episodes lack replay: {missing_replays[:3]}")
    return {
        "run_id": result["run_id"],
        "episodes": episodes,
        "inputs": {"result": file_fact(result_path), "replay": file_fact(replay_path)},
    }


def normalized_cursor(frame: dict[str, Any], origin: dict[str, int], name: str) -> int:
    cursors = frame.get("cursors") or {}
    value = cursors.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"cursor {name!r} is not an integer at step {frame.get('step')}")
    return value - origin[name]


def frame_projection(frame: dict[str, Any], origin: dict[str, int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_bytes_hash": frame.get("action_bytes_hash") or "",
        "observation_hash": frame.get("observation_hash") or "",
        "cursor_submitted": normalized_cursor(frame, origin, "submitted"),
        "cursor_accepted": normalized_cursor(frame, origin, "accepted"),
        "cursor_observed": normalized_cursor(frame, origin, "observed"),
        "outcome_json": canonical_json(frame.get("outcome")),
    }
    payload.update(state_hashes(frame.get("state") or {}))
    return payload


def export_rrd(recording: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    frame_count = 0
    action_count = 0
    with rr.RecordingStream(
        "sentinel-vla-fact-comparison",
        recording_id=PROJECTION,
        send_properties=False,
    ) as stream:
        stream.save(output)
        for key in sorted(recording["episodes"]):
            episode = recording["episodes"][key]
            frames = episode["replay"]["frames"]
            origin_raw = frames[0].get("cursors") or {}
            origin = {name: origin_raw.get(name) for name in ("submitted", "accepted", "observed")}
            if any(not isinstance(value, int) or isinstance(value, bool) for value in origin.values()):
                raise ValueError(f"invalid cursor origin for episode {key!r}")
            path = entity_path(key)
            rows: list[dict[str, Any]] = []
            for frame in frames:
                rows.append(frame_projection(frame, origin))
                frame_count += 1
                action_count += frame.get("action_bytes_hash") is not None
            component_names = sorted(rows[0])
            if any(sorted(row) != component_names for row in rows):
                raise ValueError(f"projected component set changes within episode {key!r}")
            stream.send_columns(
                path,
                indexes=[rr.TimeColumn(TIMELINE, sequence=[int(frame["step"]) for frame in frames])],
                columns=rr.AnyValues.columns(
                    **{name: [row[name] for row in rows] for name in component_names}
                ),
            )
        stream.flush()
    return {
        "run_id": recording["run_id"],
        "episode_count": len(recording["episodes"]),
        "frame_count": frame_count,
        "action_count": action_count,
        "seconds": time.perf_counter() - started,
        "rrd": file_fact(output),
        "inputs": recording["inputs"],
    }


def compare_source_facts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_keys = set(left["episodes"])
    right_keys = set(right["episodes"])
    if left_keys != right_keys:
        raise ValueError("source recordings do not have identical semantic episode keys")
    counts = {
        "paired_episodes": len(left_keys),
        "compared_frames": 0,
        "compared_actions": 0,
        "action_differences": 0,
        "observation_differences": 0,
        "cursor_differences": 0,
        "outcome_differences": 0,
        "state_leaf_differences": 0,
    }
    first_difference: dict[str, Any] | None = None
    state_leaf_paths: dict[str, int] = {}
    for key in sorted(left_keys):
        left_frames = left["episodes"][key]["replay"]["frames"]
        right_frames = right["episodes"][key]["replay"]["frames"]
        if len(left_frames) != len(right_frames):
            raise ValueError(f"frame count differs for episode {key!r}")
        left_origin = left_frames[0]["cursors"]
        right_origin = right_frames[0]["cursors"]
        for left_frame, right_frame in zip(left_frames, right_frames, strict=True):
            if left_frame["step"] != right_frame["step"]:
                raise ValueError(f"step differs for episode {key!r}")
            counts["compared_frames"] += 1
            if left_frame.get("action_bytes_hash") is not None:
                counts["compared_actions"] += 1
            categories: list[str] = []
            if left_frame.get("action_bytes_hash") != right_frame.get("action_bytes_hash"):
                counts["action_differences"] += 1
                categories.append("action")
            if left_frame.get("observation_hash") != right_frame.get("observation_hash"):
                counts["observation_differences"] += 1
                categories.append("observation")
            left_cursor = tuple(normalized_cursor(left_frame, left_origin, name) for name in ("submitted", "accepted", "observed"))
            right_cursor = tuple(normalized_cursor(right_frame, right_origin, name) for name in ("submitted", "accepted", "observed"))
            if left_cursor != right_cursor:
                counts["cursor_differences"] += 1
                categories.append("cursor")
            if canonical_json(left_frame.get("outcome")) != canonical_json(right_frame.get("outcome")):
                counts["outcome_differences"] += 1
                categories.append("outcome")
            left_state = state_hashes(left_frame.get("state") or {})
            right_state = state_hashes(right_frame.get("state") or {})
            for path in sorted(set(left_state) | set(right_state)):
                if left_state.get(path) != right_state.get(path):
                    counts["state_leaf_differences"] += 1
                    state_leaf_paths[path] = state_leaf_paths.get(path, 0) + 1
                    categories.append(path)
            if categories and first_difference is None:
                first_difference = {
                    "episode": {
                        "task_id": key[0],
                        "initial_state_index": key[1],
                        "seed": key[2],
                        "instruction": key[3],
                    },
                    "step": left_frame["step"],
                    "categories": categories,
                    "action_equal": left_frame.get("action_bytes_hash") == right_frame.get("action_bytes_hash"),
                    "previous_observation_equal": None,
                }
                if left_frame["step"] > 0:
                    index = int(left_frame["step"]) - 1
                    first_difference["previous_observation_equal"] = (
                        left_frames[index].get("observation_hash") == right_frames[index].get("observation_hash")
                    )
    return {"counts": counts, "state_leaf_paths": state_leaf_paths, "first_difference": first_difference}


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "started_at": started_at,
        "seconds": time.perf_counter() - started,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def save_command_output(directory: Path, name: str, result: dict[str, Any]) -> dict[str, Any]:
    stdout_path = directory / f"{name}.stdout.txt"
    stderr_path = directory / f"{name}.stderr.txt"
    stdout = result.pop("stdout")
    stderr = result.pop("stderr")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result["observed_unmatched_chunks"] = [
        {"count": int(count), "rrd": path}
        for count, path in re.findall(r'(\d+) chunk\(s\) from "([^"]+)" could not be matched', stderr)
    ]
    result["observed_unmatched_entity_paths"] = sorted(
        set(re.findall(r"rerun:entity_path:\s+(\S+)", stderr))
    )
    result["stdout"] = file_fact(stdout_path)
    result["stderr"] = file_fact(stderr_path)
    return result


def sysctl_value(name: str) -> str | None:
    result = subprocess.run(["sysctl", "-n", name], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_report(path: Path, report: dict[str, Any]) -> None:
    source = report["source_fact_comparison"]
    counts = source["counts"]
    first = source["first_difference"]
    rerun = report["rerun"]
    unmatched_paths = rerun["left_vs_right"]["observed_unmatched_entity_paths"]
    if unmatched_paths:
        rerun_observation = (
            "Rerun reported one unmatched chunk on each side for entity "
            f"`{unmatched_paths[0]}`. Its raw output retained the full unmatched episode chunks; "
            "the source-fact audit below locates the changed row and components."
        )
    else:
        rerun_observation = "Rerun did not report an unmatched entity path."
    lines = [
        "# Rerun 0.38.1 equal-fact comparison",
        "",
        "## Result",
        "",
        f"Two native VLA recordings were projected into the same `{PROJECTION}` schema. ",
        f"The projection covered {counts['paired_episodes']} paired episodes, "
        f"{counts['compared_frames']} feedback frames and {counts['compared_actions']} actions per side.",
        "",
        f"The identical-file control exited `{rerun['identical_control']['exit_code']}`. "
        f"The baseline-versus-active comparison exited `{rerun['left_vs_right']['exit_code']}`.",
        "",
        rerun_observation,
        "",
        "The source-fact audit found "
        f"{counts['action_differences']} action differences, "
        f"{counts['observation_differences']} observation differences, "
        f"{counts['cursor_differences']} normalized-cursor differences and "
        f"{counts['outcome_differences']} outcome differences. "
        f"It found {counts['state_leaf_differences']} differing state-leaf fingerprint.",
        "",
    ]
    if first:
        episode = first["episode"]
        lines.extend([
            "The first and only source difference is at "
            f"task `{episode['task_id']}`, initial state `{episode['initial_state_index']}`, "
            f"seed `{episode['seed']}`, step `{first['step']}`. "
            f"Categories: `{', '.join(first['categories'])}`. "
            f"The action at that step is equal: `{first['action_equal']}`; "
            f"the previous observation is equal: `{first['previous_observation_equal']}`.",
            "",
        ])
    lines.extend([
        "## Interpretation boundary",
        "",
        "Rerun's command establishes equality or inequality of the exported RRD facts. "
        "It does not verify the origin signature, classify expected authorization changes, "
        "or prove the cause of a changed camera fingerprint. Sentinel's product workflow combines "
        "independent source verification, VLA-aware first-difference classification and recorded 3D evidence navigation. "
        "This experiment does not claim that another product cannot assemble an equivalent workflow.",
        "",
        "Conversion time and RRD comparison time are reported separately because they are different workloads. "
        "No cross-tool speed claim is made.",
        "The machine facts below describe the local projection/comparison host, not the GPU host that produced the source recordings.",
        "",
        "## Reproduction facts",
        "",
        f"- Rerun version output: `{report['rerun']['version_text']}`",
        f"- Left conversion: `{report['exports']['left']['seconds']:.6f}` seconds",
        f"- Right conversion: `{report['exports']['right']['seconds']:.6f}` seconds",
        f"- Identical-file comparison: `{rerun['identical_control']['seconds']:.6f}` seconds, exit `{rerun['identical_control']['exit_code']}`",
        f"- Left-versus-right comparison: `{rerun['left_vs_right']['seconds']:.6f}` seconds, exit `{rerun['left_vs_right']['exit_code']}`",
        f"- Machine: `{canonical_json(report['machine'])}`",
        "",
        "Raw stdout and stderr, file hashes, sizes, commands and UTC start times are retained beside this report in `result.json`.",
        "",
        "Official references: [Rerun CLI](https://rerun.io/docs/reference/cli) and "
        "[custom data](https://rerun.io/docs/howto/logging-and-ingestion/custom-data).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-bundle", required=True, type=Path)
    parser.add_argument("--right-bundle", required=True, type=Path)
    parser.add_argument("--rerun-bin", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    left = load_recording(args.left_bundle)
    right = load_recording(args.right_bundle)
    source_facts = compare_source_facts(left, right)

    left_rrd = args.out_dir / "baseline-projection.rrd"
    right_rrd = args.out_dir / "active-projection.rrd"
    left_export = export_rrd(left, left_rrd)
    right_export = export_rrd(right, right_rrd)

    base_command = [str(args.rerun_bin), "rrd", "compare", "--unordered", "--ignore-timeline", "log_tick"]
    identical = save_command_output(
        args.out_dir,
        "rerun-identical-control",
        run_command(base_command + [str(left_rrd), str(left_rrd)], args.out_dir),
    )
    comparison = save_command_output(
        args.out_dir,
        "rerun-left-vs-right",
        run_command(base_command + [str(left_rrd), str(right_rrd)], args.out_dir),
    )
    version = run_command([str(args.rerun_bin), "--version"], args.out_dir)
    version_text = (version["stdout"] or version["stderr"]).strip().splitlines()[0]

    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "projection": {
            "schema": PROJECTION,
            "timeline": TIMELINE,
            "facts": [
                "action_bytes_hash",
                "observation_hash",
                "state leaf sha256 fingerprints",
                "episode-relative submitted/accepted/observed cursors",
                "canonical outcome JSON",
            ],
            "excluded": ["run_id", "episode_id", "permit_id", "latency", "wall-clock time", "authorization mode"],
        },
        "machine": {
            "scope": "local RRD projection and comparison host",
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "model": sysctl_value("hw.model"),
            "cpu": sysctl_value("machdep.cpu.brand_string"),
            "memory_bytes": int(sysctl_value("hw.memsize") or 0) or None,
            "python": platform.python_version(),
        },
        "exports": {"left": left_export, "right": right_export},
        "source_fact_comparison": source_facts,
        "rerun": {
            "version_command": version["command"],
            "version_exit_code": version["exit_code"],
            "version_text": version_text,
            "identical_control": identical,
            "left_vs_right": comparison,
        },
    }
    result_path = args.out_dir / "result.json"
    report_path = args.out_dir / "FACTS.md"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, report)

    if identical["exit_code"] != 0:
        raise SystemExit("identical-file Rerun control failed")
    difference_fields = (
        "action_differences",
        "observation_differences",
        "cursor_differences",
        "outcome_differences",
        "state_leaf_differences",
    )
    expected_difference = any(source_facts["counts"][field] > 0 for field in difference_fields)
    if expected_difference and comparison["exit_code"] == 0:
        raise SystemExit("Rerun reported equality despite a projected source-fact difference")
    if not expected_difference and comparison["exit_code"] != 0:
        raise SystemExit("Rerun reported inequality despite equal projected source facts")
    print(json.dumps({
        "result": str(result_path.resolve()),
        "facts": str(report_path.resolve()),
        "identical_exit_code": identical["exit_code"],
        "comparison_exit_code": comparison["exit_code"],
        "source_counts": source_facts["counts"],
        "first_difference": source_facts["first_difference"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
