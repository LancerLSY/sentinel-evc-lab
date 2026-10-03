"""Deterministic comparison of two independently verified native VLA runs."""
from __future__ import annotations

import json
import re


SCHEMA = "sentinel-native-vla-compare-v1"
_SOFTWARE_KEYS = ("mujoco", "robosuite", "lerobot")
_CATEGORIES = ("action", "observation", "cursor", "outcome", "authorization")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", value) is not None


def _without_paths(value):
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items()
                if key not in {"path", "protocol"} and not key.endswith("_path")}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _identity(result):
    software = result.get("software") if isinstance(result.get("software"), dict) else {}
    configuration = result.get("configuration") if isinstance(result.get("configuration"), dict) else {}
    checkpoint = result.get("checkpoint") if isinstance(result.get("checkpoint"), dict) else {}
    backbone = result.get("backbone") if isinstance(result.get("backbone"), dict) else {}
    return {
        "profile_hash": result.get("profile_hash"),
        "official_identity": _without_paths(result.get("official_identity")),
        "software": {key: software.get(key) for key in _SOFTWARE_KEYS},
        "execution_horizon": configuration.get("execution_horizon"),
        "checkpoint_config_hash": checkpoint.get("config_sha256"),
        "backbone_config_hash": backbone.get("config_sha256"),
    }


def _identity_differences(left, right):
    differences = []
    for field in sorted(set(left) | set(right)):
        if _canonical(left.get(field)) != _canonical(right.get(field)):
            differences.append({"field": field, "left": left.get(field), "right": right.get(field)})
    return differences


def _episode_key(episode):
    if any(episode.get(key) is None for key in ("task_id", "initial_state_index", "seed", "instruction")):
        raise ValueError("episode 缺少 task/state/seed/instruction 匹配字段。")
    return (
        episode.get("task_id"),
        episode.get("initial_state_index"),
        episode.get("seed"),
        episode.get("instruction"),
    )


def _key_dict(key):
    return dict(zip(("task_id", "initial_state_index", "seed", "instruction"), key))


def _episode_map(result):
    rows = {}
    for index, episode in enumerate(result.get("episodes", [])):
        if not isinstance(episode, dict):
            raise ValueError("原生运行包含不合法的 episode 结果。")
        key = _episode_key(episode)
        if key in rows:
            raise ValueError("原生运行包含重复的 task/state/seed/instruction 匹配键。")
        rows[key] = (index, episode)
    return rows


def _strict_frames(episode):
    frames = episode.get("frames") if isinstance(episode, dict) else None
    if not isinstance(frames, list):
        raise ValueError("回放 episode 缺少帧列表。")
    previous = None
    rows = {}
    for index, frame in enumerate(frames):
        step = frame.get("step") if isinstance(frame, dict) else None
        if type(step) is not int or (previous is not None and step <= previous):
            raise ValueError("回放 step 必须是唯一且严格递增的整数。")
        rows[step] = (index, frame)
        previous = step
    return rows


def _pose_index(frames, index, body_counts):
    def complete(frame):
        positions, rotations = frame.get("bodyWorldPosition"), frame.get("bodyWorldRotation")
        count = body_counts.get(frame.get("modelId"))
        return count is not None and isinstance(positions, list) and isinstance(rotations, list) and len(positions) == len(rotations) == count
    for candidate in range(index, -1, -1):
        if complete(frames[candidate]):
            return candidate
    for candidate in range(index + 1, len(frames)):
        if complete(frames[candidate]):
            return candidate
    return None


def _frame_ref(run_id, episode_index, frames, frame_index, body_counts):
    frame = frames[frame_index]
    pose_index = _pose_index(frames, frame_index, body_counts)
    return {
        "run_id": run_id,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "step": frame.get("step"),
        "pose_frame_index": pose_index,
        "pose_step": frames[pose_index].get("step") if pose_index is not None else None,
        "exact_pose": pose_index == frame_index,
    }


def _sha_leaves(value, prefix=""):
    leaves = {}
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str):
            leaves[prefix or "$"] = value["sha256"]
        else:
            for key in sorted(value):
                leaves.update(_sha_leaves(value[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(_sha_leaves(item, f"{prefix}[{index}]"))
    return leaves


def _state_leaf_differences(left, right):
    left_leaves, right_leaves = _sha_leaves(left), _sha_leaves(right)
    return [{"path": path, "left": left_leaves.get(path), "right": right_leaves.get(path)}
            for path in sorted(set(left_leaves) | set(right_leaves))
            if left_leaves.get(path) != right_leaves.get(path)][:128]


def _cursor_origin(frames):
    cursors = frames[0].get("cursors") if frames else None
    return {key: cursors.get(key) for key in ("submitted", "accepted", "observed")} if isinstance(cursors, dict) else {}


def _normalized_cursor(frame, origin):
    raw = frame.get("cursors")
    if not isinstance(raw, dict):
        return raw, None
    normalized = {}
    for key in ("submitted", "accepted", "observed"):
        value, base = raw.get(key), origin.get(key)
        normalized[key] = value - base if type(value) is int and type(base) is int else None
    return raw, normalized


def _permit_present(frame):
    return any(frame.get(key) is not None for key in ("permit", "permit_id")) or (
        isinstance(frame.get("authorization"), dict) and frame["authorization"].get("permit") is not None
    )


def _expected_authorization(left_mode, right_mode, left_frame, right_frame):
    if {left_mode, right_mode} != {"baseline", "active"}:
        return False
    baseline = left_frame if left_mode == "baseline" else right_frame
    active = left_frame if left_mode == "active" else right_frame
    return (baseline.get("decision") == "BASELINE_DIRECT" and active.get("decision") == "ACTIVE_AUTHORIZED"
            and baseline.get("reason") is None and active.get("reason") is None
            and not _permit_present(baseline) and _permit_present(active))


def _result_summary(episode):
    return {key: episode.get(key) for key in ("episode_id", "success", "crashed", "env_steps", "max_reward", "sum_reward")}


def _difference(category, step, details, left_ref, right_ref, previous_equal, action_equal):
    return {
        "category": category,
        "step": step,
        "details": details,
        "left_ref": left_ref,
        "right_ref": right_ref,
        "previous_observation_equal": previous_equal,
        "action_equal": action_equal,
    }


def _compare_episode(left_source, right_source, key, left_row, right_row):
    left_index, left_result = left_row
    right_index, right_result = right_row
    left_episode = left_source["replay"]["episodes"][left_index]
    right_episode = right_source["replay"]["episodes"][right_index]
    left_steps, right_steps = _strict_frames(left_episode), _strict_frames(right_episode)
    left_frames, right_frames = left_episode["frames"], right_episode["frames"]
    left_origin, right_origin = _cursor_origin(left_frames), _cursor_origin(right_frames)
    counts = {category: 0 for category in _CATEGORIES}
    counts.update({"compared_actions": 0, "expected_authorization": 0, "missing_left_steps": 0, "missing_right_steps": 0})
    first = {category: None for category in _CATEGORIES}
    previous_observation_equal = None
    missing_facts = []
    for side, result, frames, steps in (("left", left_result, left_frames, left_steps),
                                        ("right", right_result, right_frames, right_steps)):
        total = result.get("env_steps")
        if type(total) is not int or total < 0 or list(steps) != list(range(total + 1)):
            missing_facts.append({"side": side, "field": "complete_reset_and_execution_steps"})
        for field in ("success", "crashed"):
            if type(result.get(field)) is not bool:
                missing_facts.append({"side": side, "field": field})
        for frame in frames:
            step = frame["step"]
            required = ["observation_hash", "decision", "cursors"] + (["action", "action_bytes_hash", "outcome"] if step > 0 else [])
            for field in required:
                value = frame.get(field)
                absent = value is None
                if field == "cursors":
                    absent = not isinstance(value, dict) or any(type(value.get(k)) is not int for k in ("submitted", "accepted", "observed"))
                elif field in ("observation_hash", "action_bytes_hash"):
                    absent = not _valid_hash(value)
                elif field == "decision":
                    absent = not isinstance(value, str) or not value
                elif field == "outcome":
                    absent = not isinstance(value, dict)
                if absent:
                    missing_facts.append({"side": side, "step": step, "field": field})

    for step in sorted(set(left_steps) | set(right_steps)):
        left_item, right_item = left_steps.get(step), right_steps.get(step)
        if left_item is None or right_item is None:
            category = "observation"
            counts[category] += 1
            counts["missing_left_steps" if left_item is None else "missing_right_steps"] += 1
            present_index, present_frame = right_item if left_item is None else left_item
            left_ref = None if left_item is None else _frame_ref(left_source["result"]["run_id"], left_index, left_frames, left_item[0], left_source.get("model_body_counts", {}))
            right_ref = None if right_item is None else _frame_ref(right_source["result"]["run_id"], right_index, right_frames, right_item[0], right_source.get("model_body_counts", {}))
            diff = _difference(category, step, {"kind": "missing_step", "missing": "left" if left_item is None else "right"},
                               left_ref, right_ref, previous_observation_equal, None)
            first[category] = first[category] or diff
            previous_observation_equal = False
            continue
        left_frame_index, left_frame = left_item
        right_frame_index, right_frame = right_item
        left_ref = _frame_ref(left_source["result"]["run_id"], left_index, left_frames, left_frame_index, left_source.get("model_body_counts", {}))
        right_ref = _frame_ref(right_source["result"]["run_id"], right_index, right_frames, right_frame_index, right_source.get("model_body_counts", {}))
        left_action, right_action = left_frame.get("action_bytes_hash"), right_frame.get("action_bytes_hash")
        if left_action is not None or right_action is not None:
            counts["compared_actions"] += 1
        raw_action_equal = _canonical(left_frame.get("action")) == _canonical(right_frame.get("action"))
        action_equal = left_action == right_action and left_action is not None and raw_action_equal
        if left_action != right_action or not raw_action_equal:
            counts["action"] += 1
            first["action"] = first["action"] or _difference(
                "action", step, {"left_hash": left_action, "right_hash": right_action,
                                 "recorded_action_equal": raw_action_equal}, left_ref, right_ref,
                previous_observation_equal, False)

        left_observation, right_observation = left_frame.get("observation_hash"), right_frame.get("observation_hash")
        observation_equal = (left_observation == right_observation and left_observation is not None
                             and _canonical(left_frame.get("state")) == _canonical(right_frame.get("state")))
        if not observation_equal:
            counts["observation"] += 1
            first["observation"] = first["observation"] or _difference(
                "observation", step,
                {"left_hash": left_observation, "right_hash": right_observation,
                 "state_leaf_differences": _state_leaf_differences(left_frame.get("state"), right_frame.get("state"))},
                left_ref, right_ref, previous_observation_equal, action_equal)

        left_raw, left_cursor = _normalized_cursor(left_frame, left_origin)
        right_raw, right_cursor = _normalized_cursor(right_frame, right_origin)
        if left_cursor is None or right_cursor is None or left_cursor != right_cursor:
            counts["cursor"] += 1
            first["cursor"] = first["cursor"] or _difference(
                "cursor", step, {"left_raw": left_raw, "right_raw": right_raw,
                                 "left_normalized": left_cursor, "right_normalized": right_cursor},
                left_ref, right_ref, previous_observation_equal, action_equal)

        if _canonical(left_frame.get("outcome")) != _canonical(right_frame.get("outcome")):
            counts["outcome"] += 1
            first["outcome"] = first["outcome"] or _difference(
                "outcome", step, {"left": left_frame.get("outcome"), "right": right_frame.get("outcome")},
                left_ref, right_ref, previous_observation_equal, action_equal)

        authorization_equal = (left_frame.get("decision") == right_frame.get("decision")
                               and left_frame.get("reason") == right_frame.get("reason")
                               and _permit_present(left_frame) == _permit_present(right_frame))
        if not authorization_equal:
            details = {"left": {"decision": left_frame.get("decision"), "reason": left_frame.get("reason"),
                                 "permit_present": _permit_present(left_frame)},
                       "right": {"decision": right_frame.get("decision"), "reason": right_frame.get("reason"),
                                  "permit_present": _permit_present(right_frame)}}
            if _expected_authorization(left_source["result"].get("mode"), right_source["result"].get("mode"), left_frame, right_frame):
                counts["expected_authorization"] += 1
            else:
                counts["authorization"] += 1
                first["authorization"] = first["authorization"] or _difference(
                    "authorization", step, details, left_ref, right_ref, previous_observation_equal, action_equal)
        previous_observation_equal = observation_equal

    terminal_fields = ("success", "crashed", "env_steps", "max_reward", "sum_reward")
    left_terminal = {field: left_result.get(field) for field in terminal_fields}
    right_terminal = {field: right_result.get(field) for field in terminal_fields}
    if _canonical(left_terminal) != _canonical(right_terminal):
        counts["outcome"] += 1
        terminal_step = max(set(left_steps) | set(right_steps), default=0)
        first["outcome"] = first["outcome"] or _difference(
            "outcome", terminal_step, {"kind": "terminal_result", "left": left_terminal, "right": right_terminal},
            _frame_ref(left_source["result"]["run_id"], left_index, left_frames, len(left_frames)-1, left_source.get("model_body_counts", {})) if left_frames else None,
            _frame_ref(right_source["result"]["run_id"], right_index, right_frames, len(right_frames)-1, right_source.get("model_body_counts", {})) if right_frames else None,
            previous_observation_equal, None)
    candidates = [item for item in first.values() if item is not None]
    first_difference = min(candidates, key=lambda item: (item["step"], _CATEGORIES.index(item["category"]))) if candidates else None
    return {
        "key": _key_dict(key),
        "left_episode_index": left_index,
        "right_episode_index": right_index,
        "counts": counts,
        "missing_facts": missing_facts,
        "first_difference": first_difference,
        "first_by_category": first,
        "left": _result_summary(left_result),
        "right": _result_summary(right_result),
    }


def compare_verified_runs(left_source, right_source):
    left_result, right_result = left_source["result"], right_source["result"]
    left_rows, right_rows = _episode_map(left_result), _episode_map(right_result)
    common = sorted(set(left_rows) & set(right_rows), key=_canonical)
    unmatched_left = [_key_dict(key) for key in sorted(set(left_rows) - set(right_rows), key=_canonical)]
    unmatched_right = [_key_dict(key) for key in sorted(set(right_rows) - set(left_rows), key=_canonical)]
    identities = {"left": _identity(left_result), "right": _identity(right_result)}
    identity_differences = _identity_differences(identities["left"], identities["right"])
    missing_identity = []
    for side, identity in identities.items():
        for field, value in identity.items():
            missing = value is None or value == {} or value == ""
            if field.endswith("hash"):
                missing = not _valid_hash(value)
            elif field == "execution_horizon":
                missing = type(value) is not int or value <= 0
            elif field == "official_identity":
                missing = not isinstance(value, dict) or not value
            if missing:
                missing_identity.append({"side": side, "field": field})
        for name, version in identity["software"].items():
            if not isinstance(version, str) or not version:
                missing_identity.append({"side": side, "field": "software." + name})
    episodes = [_compare_episode(left_source, right_source, key, left_rows[key], right_rows[key]) for key in common]
    summary = {
        "paired_episodes": len(episodes),
        "compared_actions": sum(row["counts"]["compared_actions"] for row in episodes),
        "action_differences": sum(row["counts"]["action"] for row in episodes),
        "observation_differences": sum(row["counts"]["observation"] for row in episodes),
        "cursor_differences": sum(row["counts"]["cursor"] for row in episodes),
        "outcome_differences": sum(row["counts"]["outcome"] for row in episodes),
        "authorization_differences": sum(row["counts"]["authorization"] for row in episodes),
        "expected_authorization_differences": sum(row["counts"]["expected_authorization"] > 0 for row in episodes),
        "left_successes": sum(row.get("success") is True for row in left_result.get("episodes", [])),
        "right_successes": sum(row.get("success") is True for row in right_result.get("episodes", [])),
    }
    incomplete = any(row["missing_facts"] or row["counts"]["missing_left_steps"] or row["counts"]["missing_right_steps"] for row in episodes)
    compatible = not identity_differences and not missing_identity and not incomplete and not unmatched_left and not unmatched_right and bool(episodes)
    differences = sum(summary[key] for key in (
        "action_differences", "observation_differences", "cursor_differences",
        "outcome_differences", "authorization_differences"))
    summary["status"] = "INCOMPARABLE" if not compatible else ("DIFFERENCES" if differences else "EQUIVALENT")
    return {
        "schema": SCHEMA,
        "scope": "derived-from-two-independently-verified-records",
        "sources": {"left": left_source["source"], "right": right_source["source"]},
        "eligibility": {"compatible": compatible, "identity_differences": identity_differences,
                        "missing_identity": missing_identity, "incomplete_evidence": incomplete,
                        "unmatched_left": unmatched_left, "unmatched_right": unmatched_right},
        "summary": summary,
        "episodes": episodes,
    }
