#!/usr/bin/env python3
"""Exact-action replay of fixed failed/successful confirmation cases.

This post-hoc diagnostic never loads or calls the policy. It replays the
recorded official env.step actions for task4/state30 and task4/state32, checks
the original outcomes, and retains native RGB plus simulator measurements.
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

from replay_backend_comparison import _empty, _processed_camera_png, _receipt, _require
from replay_libero_failure import _body_names, _body_pose, _unbatch, _write_video
from run_libero_closedloop import _array_digest, _jsonable, _sha256, _tree_identity


TASK_ID = 4
FPS = 20
FORMAL_MANIFEST_SHA256 = "ec87b22d166dd167c3549d3e976bfe29d10c2de128aeef2218ceb67bdcd7efac"
FORMAL_TRACE_SHA256 = "a78454f9c5e138e056cd61d9243c7ac7fff1b6855df161d6e8903d1cb38792ab"
CASES = {
    "failure": {"state": 30, "seed": 44021, "steps": 280, "success": False, "reward": 0.0},
    "success": {"state": 32, "seed": 44023, "steps": 126, "success": True, "reward": 1.0},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-runner", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--libero-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_episode(manifest: dict[str, Any], state: int) -> dict[str, Any]:
    rows = [
        row for row in manifest["episodes"]
        if row["task_id"] == TASK_ID and row["initial_state_index"] == state
    ]
    _require(len(rows) == 1, f"formal task4/state{state} row missing or duplicated")
    return rows[0]


def _validate_sources(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load(args.confirmation_manifest)
    protocol = _load(args.confirmation_protocol)
    _require(_sha256(args.confirmation_manifest) == FORMAL_MANIFEST_SHA256, "formal manifest mismatch")
    _require(_sha256(args.confirmation_trace) == FORMAL_TRACE_SHA256, "formal trace mismatch")
    _require(manifest["trace"]["sha256"] == FORMAL_TRACE_SHA256, "formal trace receipt mismatch")
    _require(manifest["status"] == "complete" and manifest["mode"] == "formal", "formal run incomplete")
    _require(manifest["script_sha256"] == _sha256(args.confirmation_runner), "runner receipt mismatch")
    _require(manifest["protocol_sha256"] == _sha256(args.confirmation_protocol), "protocol receipt mismatch")
    _require(protocol["implementation"]["script_sha256"] == _sha256(args.confirmation_runner), "protocol runner mismatch")
    _require(args.font.is_file(), "font is missing")
    for name, expected in manifest["model_artifacts"]["checkpoint_sha256"].items():
        _require(_sha256(args.checkpoint / name) == expected, f"checkpoint mismatch: {name}")

    task_rows = sorted(
        (row for row in manifest["episodes"] if row["task_id"] == TASK_ID),
        key=lambda row: row["initial_state_index"],
    )
    failures = [row for row in task_rows if not row["success"]]
    first_failure = failures[0]
    later_successes = [
        row for row in task_rows
        if row["initial_state_index"] > first_failure["initial_state_index"] and row["success"]
    ]
    _require(first_failure["initial_state_index"] == CASES["failure"]["state"], "failure selection changed")
    _require(later_successes[0]["initial_state_index"] == CASES["success"]["state"], "success selection changed")
    for label, expected in CASES.items():
        row = _formal_episode(manifest, expected["state"])
        _require(
            row["actual_initial_state_index"] == expected["state"]
            and row["actual_seed"] == expected["seed"]
            and row["env_steps"] == expected["steps"]
            and row["success"] is expected["success"]
            and row["sum_reward"] == expected["reward"]
            and not row["crashed"],
            f"selected {label} outcome changed",
        )
    return {"manifest": manifest, "protocol": protocol}


def _source_cases(path: Path) -> dict[str, dict[str, Any]]:
    by_state = {spec["state"]: label for label, spec in CASES.items()}
    data = {
        label: {"plan": None, "reset": None, "context": None, "post": {}, "actions": [], "hashes": [], "outcomes": []}
        for label in CASES
    }
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("task_id") != TASK_ID or event.get("initial_state_index") not in by_state:
                continue
            label = by_state[event["initial_state_index"]]
            case = data[label]
            kind = event["event"]
            if kind == "episode_plan":
                case["plan"] = event
            elif kind == "episode_reset":
                case["reset"] = event
            elif kind == "raw_policy_chunk" and event["step"] == 0:
                case["context"] = event["input_context"]
            elif kind == "official_postprocessed_action":
                case["post"][event["step"]] = event
            elif kind == "env_step_action":
                action = np.asarray(event["values"], dtype=np.float32)
                _require(action.shape == (1, 7) and _array_digest(action) == event["sha256"], "action digest mismatch")
                _require(event["step"] == len(case["actions"]), "action steps are not contiguous")
                case["actions"].append(action)
                case["hashes"].append(event["sha256"])
            elif kind == "env_step_outcome":
                _require(event["step"] == len(case["outcomes"]), "outcome steps are not contiguous")
                case["outcomes"].append(event)

    for label, expected in CASES.items():
        case = data[label]
        _require(all(case[key] is not None for key in ("plan", "reset", "context")), f"{label} source context incomplete")
        _require(case["plan"]["seed"] == case["reset"]["seed"] == expected["seed"], f"{label} seed mismatch")
        _require(len(case["actions"]) == len(case["outcomes"]) == expected["steps"], f"{label} step count mismatch")
        _require(len(case["post"]) == expected["steps"], f"{label} postprocessed action count mismatch")
        for step, action in enumerate(case["actions"]):
            post = case["post"][step]
            official = np.asarray(post["values"], dtype=np.float32)
            _require(post["sha256"] == case["hashes"][step] and np.array_equal(action, official), "post/env action mismatch")
        last = case["outcomes"][-1]
        _require(
            all(row.get("is_success_present") is True and row.get("is_success_source") == "info" for row in case["outcomes"]),
            f"{label} official outcome provenance mismatch",
        )
        _require(
            bool(last["is_success"]) is expected["success"]
            and sum(float(row["reward"]) for row in case["outcomes"]) == expected["reward"],
            f"{label} recorded outcome mismatch",
        )
    return data


def _model_names(model: Any, kind: str) -> list[str]:
    names = getattr(model, f"{kind}_names", None)
    if names is not None:
        return [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    import mujoco

    raw = getattr(model, "_model", model)
    enum = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_JOINT
    count = raw.nbody if kind == "body" else raw.njnt
    return [mujoco.mj_id2name(raw, enum, index) or "" for index in range(count)]


def _body_id(model: Any, name: str) -> int:
    try:
        return int(model.body_name2id(name))
    except AttributeError:
        return int(model.body(name).id)


def _joint_id(model: Any, name: str) -> int:
    try:
        return int(model.joint_name2id(name))
    except AttributeError:
        return int(model.joint(name).id)


def _resolve_model(sim: Any) -> dict[str, Any]:
    bodies = _model_names(sim.model, "body")
    joints = _model_names(sim.model, "joint")

    def unique(label: str, predicate: Any) -> str:
        matches = [name for name in bodies if predicate(name.lower())]
        _require(len(matches) == 1, f"cannot uniquely resolve {label}: {matches}")
        return matches[0]

    bowl = unique("target black bowl", lambda name: name.endswith("akita_black_bowl_1_main"))
    plate = unique("plate", lambda name: name.endswith("plate_1_main"))
    top_joints = [name for name in joints if "drawer" in name.lower() and "top" in name.lower()]
    if len(top_joints) != 1:
        top_bodies = [name for name in bodies if "drawer" in name.lower() and "top" in name.lower()]
        derived: list[str] = []
        for name in top_bodies:
            body_id = _body_id(sim.model, name)
            count = int(sim.model.body_jntnum[body_id])
            start = int(sim.model.body_jntadr[body_id])
            derived.extend(joints[start : start + count])
        top_joints = sorted(set(name for name in derived if name))
    _require(len(top_joints) == 1, f"cannot uniquely resolve top-drawer joint: {top_joints}")
    return {"body_inventory": bodies, "joint_inventory": joints, "target_bowl": bowl, "plate": plate, "top_drawer_joint": top_joints[0]}


def _joint_position(sim: Any, name: str) -> float:
    joint_id = _joint_id(sim.model, name)
    address = int(sim.model.jnt_qposadr[joint_id])
    return float(sim.data.qpos[address])


def _descendant_ids(model: Any, root: int) -> set[int]:
    parent = np.asarray(model.body_parentid, dtype=np.int64)
    selected = {root}
    changed = True
    while changed:
        before = len(selected)
        selected.update(index for index, value in enumerate(parent) if int(value) in selected)
        changed = len(selected) != before
    return selected


def _target_contacts(sim: Any, target_body_name: str, body_names: list[str]) -> list[dict[str, Any]]:
    target_ids = _descendant_ids(sim.model, _body_id(sim.model, target_body_name))
    pairs: list[dict[str, Any]] = []
    for contact in sim.data.contact[: int(sim.data.ncon)]:
        body_a = int(sim.model.geom_bodyid[int(contact.geom1)])
        body_b = int(sim.model.geom_bodyid[int(contact.geom2)])
        if body_a not in target_ids and body_b not in target_ids:
            continue
        pairs.append({
            "body_a": body_names[body_a],
            "body_b": body_names[body_b],
            "distance": float(contact.dist),
        })
    return pairs


def _snapshot(underlying: Any, observation: dict[str, Any], resolved: dict[str, Any], initial: dict[str, float]) -> dict[str, Any]:
    sim = underlying._env.sim
    bowl = _body_pose(sim, resolved["target_bowl"])
    plate = _body_pose(sim, resolved["plate"])
    eef = np.asarray(observation["robot_state"]["eef"]["pos"], dtype=np.float64)
    bowl_pos = np.asarray(bowl["position"], dtype=np.float64)
    plate_pos = np.asarray(plate["position"], dtype=np.float64)
    drawer = _joint_position(sim, resolved["top_drawer_joint"])
    contacts = _target_contacts(sim, resolved["target_bowl"], resolved["body_inventory"])
    return {
        "target_bowl_pose": bowl,
        "plate_pose": plate,
        "observed_eef_position": eef.tolist(),
        "bowl_lift_from_reset_m": float(bowl_pos[2] - initial["bowl_z"]),
        "bowl_center_to_eef_m": float(np.linalg.norm(bowl_pos - eef)),
        "bowl_center_to_plate_center_m": float(np.linalg.norm(bowl_pos - plate_pos)),
        "top_drawer_joint_position": drawer,
        "top_drawer_displacement_from_reset": float(drawer - initial["drawer"]),
        "target_bowl_contact_count": len(contacts),
        "target_bowl_contact_pairs": contacts,
    }


def _annotated_pair(
    failure: np.ndarray,
    success: np.ndarray,
    frame_index: int,
    failure_record: dict[str, Any],
    success_record: dict[str, Any],
    held: bool,
    font_path: Path,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1600, 900), "#0b1220")
    left = Image.fromarray(failure).resize((720, 720), Image.Resampling.LANCZOS)
    right = Image.fromarray(success).resize((720, 720), Image.Resampling.LANCZOS)
    canvas.paste(left, (50, 92)); canvas.paste(right, (830, 92))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font_path), 34)
    body = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    draw.text((50, 24), "顶层抽屉任务：记录动作精确回放", font=title, fill="#f8fafc")
    draw.text((830, 24), "Top-drawer task: exact-action failure / success", font=title, fill="#7dd3fc")
    draw.text((50, 60), "Failure · task4/state30", font=body, fill="#fca5a5")
    draw.text((830, 60), "Success · task4/state32", font=body, fill="#86efac")

    def metrics(x: int, record: dict[str, Any]) -> None:
        measure = record["measurement"]
        line_one = f"step {record['step_display']}   reward {record['reward']:.1f}   success {record['is_success']}"
        line_two = (
            f"lift {measure['bowl_lift_from_reset_m']:+.3f} m   bowl-EEF {measure['bowl_center_to_eef_m']:.3f} m   "
            f"drawer Δ {measure['top_drawer_displacement_from_reset']:+.3f}   contacts {measure['target_bowl_contact_count']}"
        )
        draw.text((x, 818), line_one, font=small, fill="#e2e8f0")
        draw.text((x, 844), line_two, font=small, fill="#e2e8f0")

    metrics(50, failure_record); metrics(830, success_record)
    if held:
        draw.rectangle((830, 720, 1550, 812), fill="#7c2d12")
        draw.text((854, 738), "展示冻结 / DISPLAY HOLD", font=body, fill="#fff7ed")
        draw.text((854, 772), "无新增动作或仿真步 / no action or simulation step", font=small, fill="#fed7aa")
    draw.text(
        (50, 868),
        f"20 Hz frame {frame_index}; two different fixed initial states; post-hoc illustrations, not a paired treatment or causal contrast",
        font=small,
        fill="#94a3b8",
    )
    return np.asarray(canvas)


def main() -> int:
    args = _args()
    bound = _validate_sources(args)
    source = _source_cases(args.confirmation_trace)
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
    _require(versions["mujoco"] == "3.3.7", "replay requires isolated MuJoCo 3.3.7")
    for package in ("lerobot", "hf-libero", "robosuite", "num2words"):
        _require(versions[package] == bound["manifest"]["software"][package], f"software mismatch: {package}")
    asset_tree = _tree_identity(Path(get_libero_path("assets")), "mixed_site_packages_tree")
    for key in ("scope", "file_count", "total_bytes", "tree_sha256"):
        _require(asset_tree[key] == bound["manifest"]["asset_tree"][key], f"asset tree mismatch: {key}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    task_sources = {
        "init_states": {str(i): _sha256(init_root / task.init_states_file) for i, task in enumerate(suite.tasks)},
        "bddl": {str(i): _sha256(bddl_root / task.bddl_file) for i, task in enumerate(suite.tasks)},
    }
    _require(task_sources == bound["manifest"]["task_source_sha256"], "task source mismatch")

    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[TASK_ID], fps=FPS, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360, observation_width=360,
    )
    projected_environment = json.loads(json.dumps(bound["manifest"]["effective_environment"]))
    _require(projected_environment["task_ids"] == list(range(10)), "formal environment task grid mismatch")
    projected_environment["task_ids"] = [TASK_ID]
    _require(_jsonable(asdict(env_cfg)) == projected_environment, "projected effective environment mismatch")
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][TASK_ID]
    underlying = env.envs[0]
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    started = time.time()
    captures: dict[str, Any] = {}
    resolved: dict[str, Any] | None = None
    try:
        for label, expected in CASES.items():
            case = source[label]
            underlying.init_state_id = expected["state"]
            set_seed(expected["seed"])
            observation, _ = env.reset(seed=[expected["seed"]], options={NEW_ROLLOUT_OPTION: True})
            actual_state = int(underlying.init_state_id - underlying._reset_stride)
            actual_seed = int(underlying.np_random_seed)
            _require((actual_state, actual_seed) == (expected["state"], expected["seed"]), f"{label} reset mismatch")
            processed = preprocess_observation(observation)
            instruction = str(env.call("task_description")[0])
            processed["task"] = [instruction]
            processed = env_preprocessor(processed)
            state = processed["observation.state"]
            cameras = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_digest(value)}
                for key, value in sorted(processed.items()) if key.startswith("observation.images.")
            }
            observed = {
                "instruction": instruction,
                "state": {"shape": list(state.shape), "dtype": str(state.dtype), "sha256": _array_digest(state), "values": _jsonable(state)},
                "camera_frames": cameras,
            }
            checks = {
                "instruction": observed["instruction"] == case["context"]["instruction"],
                "processed_robot_state": observed["state"] == case["context"]["state"],
                "processed_camera_hashes": observed["camera_frames"] == case["context"]["camera_frames"],
            }
            _require(all(checks.values()), f"{label} first observation mismatch: {checks}")
            if resolved is None:
                resolved = _resolve_model(underlying._env.sim)
            else:
                _require(resolved == _resolve_model(underlying._env.sim), "model inventory changed between resets")
            initial_bowl = _body_pose(underlying._env.sim, resolved["target_bowl"])
            initial = {
                "bowl_z": float(initial_bowl["position"][2]),
                "drawer": _joint_position(underlying._env.sim, resolved["top_drawer_joint"]),
            }
            initial_measurement = _snapshot(underlying, _unbatch(observation), resolved, initial)
            frames = [underlying.render().copy()]
            records: list[dict[str, Any]] = []
            for step, (action, digest, official) in enumerate(zip(case["actions"], case["hashes"], case["outcomes"], strict=True)):
                _require(_array_digest(action) == digest, f"{label} action changed at step {step}")
                observation, reward, terminated, truncated, info = env.step(action)
                _require("is_success" in info, f"{label} official success label missing at step {step}")
                record = {
                    "step": step,
                    "step_display": step + 1,
                    "time_seconds": (step + 1) / FPS,
                    "action_sha256": digest,
                    "action": action[0].tolist(),
                    "reward": float(reward[0]),
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                    "is_success": bool(info["is_success"][0]),
                    "is_success_present": True,
                    "is_success_source": "LIBERO env.step info.is_success",
                    "measurement": _snapshot(underlying, _unbatch(observation), resolved, initial),
                }
                _require(
                    record["reward"] == float(official["reward"])
                    and record["terminated"] == bool(official["terminated"])
                    and record["truncated"] == bool(official["truncated"])
                    and record["is_success"] == bool(official["is_success"]),
                    f"{label} official outcome changed at step {step}",
                )
                records.append(record)
                frames.append(underlying.render().copy())
                if (record["terminated"] or record["truncated"]) and step + 1 < expected["steps"]:
                    raise RuntimeError(f"{label} terminated before the recorded action count")

            raw_video = args.output_dir / f"task4-state{expected['state']}-{label}-native-rgb-20hz.mp4"
            _write_video(raw_video, frames, FPS)
            camera_artifacts = {}
            for index, key in enumerate(sorted(cameras), start=1):
                path = args.output_dir / f"task4-state{expected['state']}-first-camera-{index}.png"
                _processed_camera_png(processed[key], path)
                camera_artifacts[key] = _receipt(path)
            captures[label] = {
                "expected": expected,
                "actual_reset": {"initial_state_index": actual_state, "seed": actual_seed},
                "first_observation": {"checks": checks, "processed": observed, "camera_pngs": camera_artifacts},
                "initial_measurement": initial_measurement,
                "records": records,
                "frames": frames,
                "raw_video": raw_video,
                "action_sequence_sha256": hashlib.sha256("\n".join(case["hashes"]).encode()).hexdigest(),
            }
    finally:
        env.close()

    trajectories = {}
    for label, capture in captures.items():
        path = args.output_dir / f"task4-state{capture['expected']['state']}-{label}-trajectory.json"
        path.write_text(json.dumps({
            "schema": "sentinel-libero-confirmation-case-trajectory-v1",
            "selection": label,
            "task_id": TASK_ID,
            "initial_state_index": capture["expected"]["state"],
            "seed": capture["expected"]["seed"],
            "instruction": capture["first_observation"]["processed"]["instruction"],
            "initial_measurement": capture["initial_measurement"],
            "steps": capture["records"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        trajectories[label] = path

    failure = captures["failure"]
    success = captures["success"]
    paired_frames = []
    for index, failure_frame in enumerate(failure["frames"]):
        success_index = min(index, len(success["frames"]) - 1)
        held = index >= len(success["frames"])
        failure_record = {
            "step_display": index,
            "reward": 0.0 if index == 0 else failure["records"][index - 1]["reward"],
            "is_success": False if index == 0 else failure["records"][index - 1]["is_success"],
            "measurement": failure["initial_measurement"] if index == 0 else failure["records"][index - 1]["measurement"],
        }
        success_record = {
            "step_display": success_index,
            "reward": 0.0 if success_index == 0 else success["records"][success_index - 1]["reward"],
            "is_success": False if success_index == 0 else success["records"][success_index - 1]["is_success"],
            "measurement": success["initial_measurement"] if success_index == 0 else success["records"][success_index - 1]["measurement"],
        }
        paired_frames.append(_annotated_pair(failure_frame, success["frames"][success_index], index, failure_record, success_record, held, args.font))
    paired_video = args.output_dir / "task4-state30-failure-vs-state32-success-bilingual-20hz.mp4"
    _write_video(paired_video, paired_frames, FPS)
    from PIL import Image

    poster = args.output_dir / "task4-drawer-failure-success-terminal-comparison.png"
    Image.fromarray(paired_frames[-1]).save(poster, optimize=True)

    manifest = {
        "schema": "sentinel-libero-confirmation-case-replay-v1",
        "status": "complete",
        "included_in_benchmark_metrics": False,
        "policy_inference": False,
        "observer_interventions": 0,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "selection": {
            "rule": "first task4 failure and first later task4 success in frozen state order",
            "post_hoc": True,
            "cases": CASES,
        },
        "source": {
            "design": _receipt(Path(__file__).with_name("CONFIRMATION_CASE_REPLAY_DESIGN.md")),
            "confirmation_runner": _receipt(args.confirmation_runner),
            "confirmation_protocol": _receipt(args.confirmation_protocol),
            "confirmation_manifest": _receipt(args.confirmation_manifest),
            "confirmation_trace": _receipt(args.confirmation_trace),
            "checkpoint_sha256": bound["manifest"]["model_artifacts"]["checkpoint_sha256"],
            "action_sequence_sha256": {label: capture["action_sequence_sha256"] for label, capture in captures.items()},
        },
        "environment": asdict(env_cfg),
        "environment_binding": {
            "formal_task_ids": bound["manifest"]["effective_environment"]["task_ids"],
            "replay_task_ids": [TASK_ID],
            "projection": "task_ids only; every other canonical effective-environment field is identical",
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, **versions},
        "asset_tree": asset_tree,
        "task_source_sha256": task_sources,
        "resolved_model": resolved,
        "cases": {
            label: {
                "actual_reset": capture["actual_reset"],
                "first_observation": capture["first_observation"],
                "outcome": {
                    "steps": len(capture["records"]),
                    "sum_reward": sum(row["reward"] for row in capture["records"]),
                    "success": any(row["is_success"] for row in capture["records"]),
                },
                "artifacts": {
                    "trajectory": _receipt(trajectories[label]),
                    "native_rgb_video": {
                        **_receipt(capture["raw_video"]),
                        "fps": FPS,
                        "frames": len(capture["frames"]),
                        "frame_semantics": "initial reset frame followed by one post-action native render per recorded action",
                        "source": "LIBERO Panda environment native render; simulated robot mesh and task geometry",
                    },
                },
            }
            for label, capture in captures.items()
        },
        "presentation": {
            "bilingual_video": {**_receipt(paired_video), "fps": FPS, "frames": len(paired_frames)},
            "terminal_comparison_still": _receipt(poster),
            "success_display_hold_frames": len(failure["frames"]) - len(success["frames"]),
            "hold_semantics": "comparison-only terminal display hold; no action or simulator step",
        },
        "claim_boundary": [
            "Outcome-conditioned post-hoc exact-action replay; excluded from every benchmark denominator.",
            "The two illustrations use different fixed initial states and are not a paired treatment or causal contrast.",
            "Distances, lift, drawer displacement, and simulator contact pairs are descriptive and do not identify causation.",
            "The replay performs no policy inference and changes no recorded action.",
        ],
        "timing": {"elapsed_seconds": time.time() - started},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cases": manifest["cases"], "presentation": manifest["presentation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
