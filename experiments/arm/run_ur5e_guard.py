#!/usr/bin/env python3
"""Paired UR5e guard experiment using the official MuJoCo Menagerie model.

The guard performs dense *discrete configuration checking* (<= 0.01 rad) and
a cloned-state MuJoCo rollout.  The former is not a continuous-collision proof.
Final outcomes are measured by a separate rollout and are never read by a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


UPSTREAM_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
SCENARIOS = (
    "clear",
    "late_suffix",
    "narrow_passage",
    "joint_limit",
    "tracking_disturbance",
    "environment_change",
)
METHODS = ("parent_only", "full", "incremental", "conservative")
DT = 0.002
FRAME_DT = 0.05
SUBSTEPS = 25
MAX_DISCRETE_STEP = 0.01
TRACKING_LIMIT_RAD = 0.12
JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def _native(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"cannot serialize {type(v).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_native) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_sha256(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_native).encode()
    return hashlib.sha256(packed).hexdigest()


def tree_hash(root: Path) -> tuple[str, dict[str, str]]:
    files = {}
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        files[str(p.relative_to(root))] = sha256(p)
    packed = "".join(f"{k}\0{v}\n" for k, v in files.items()).encode()
    return hashlib.sha256(packed).hexdigest(), files


def make_wrapper(asset_dir: Path, out: Path) -> Path:
    source = asset_dir / "ur5e.xml"
    if not source.is_file():
        raise FileNotFoundError(f"expected official model at {source}")
    # MuJoCo resolves include and its meshdir from the included file.
    xml = f'''<mujoco model="sentinel_ur5e_guard">
  <include file="{source.as_posix()}"/>
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".08 .10 .14 1" contype="1" conaffinity="1"/>
    <body name="guard_obstacle_body_a" mocap="true" pos="2 2 2"><geom name="guard_obstacle_a" type="sphere" size=".055" rgba=".95 .20 .16 .92" contype="1" conaffinity="1"/></body>
    <body name="guard_obstacle_body_b" mocap="true" pos="2 2 2"><geom name="guard_obstacle_b" type="sphere" size=".055" rgba="1 .55 .10 .88" contype="1" conaffinity="1"/></body>
    <camera name="guard_cam" pos="1.55 -1.55 1.25" xyaxes=".70 .71 0 -.32 .32 .89"/>
    <light pos="0 -1.2 2.2" dir="0 .35 -.94" diffuse=".9 .9 .9"/>
  </worldbody>
</mujoco>'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


@dataclass
class Layout:
    joints: np.ndarray
    qadr: np.ndarray
    obstacle: np.ndarray
    obstacle_mocap: np.ndarray
    robot_geoms: set[int]
    obstacle_geoms: set[int]
    site: int
    probe_geom: int
    home: np.ndarray
    low: np.ndarray
    high: np.ndarray


def layout(model: mujoco.MjModel) -> Layout:
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in JOINT_NAMES])
    if np.any(joints < 0):
        raise RuntimeError("official UR5e joint names changed")
    qadr = model.jnt_qposadr[joints].copy()
    obstacle = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_b"),
    ])
    obstacle_bodies = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_b"),
    ])
    obstacle_mocap = model.body_mocapid[obstacle_bodies].copy()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = set(range(model.ngeom)) - set(obstacle.tolist()) - {floor}
    collision_geoms = [g for g in robot_geoms if model.geom_group[g] == 3]
    probe_geom = collision_geoms[-1]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = model.key_qpos[key, qadr].copy()
    return Layout(joints, qadr, obstacle, obstacle_mocap, robot_geoms, set(obstacle.tolist()), site, probe_geom, home,
                  model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy())


def place_obstacles(data: mujoco.MjData, lay: Layout, positions: np.ndarray) -> None:
    data.mocap_pos[lay.obstacle_mocap] = positions


def integration_state(model: mujoco.MjModel, lay: Layout, q: np.ndarray,
                      obstacles: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    place_obstacles(data, lay, obstacles)
    data.qpos[lay.qadr] = q
    data.qvel[:] = 0
    data.ctrl[:] = np.clip(q, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state


def model_binding(model: mujoco.MjModel, lay: Layout, asset_sha: str,
                  wrapper_sha: str) -> dict[str, Any]:
    robot = sorted(lay.robot_geoms)
    return {
        "asset_tree_sha256": asset_sha,
        "wrapper_sha256": wrapper_sha,
        "model_shape": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                        "ngeom": model.ngeom, "nbody": model.nbody},
        "dt_seconds": DT, "control_frame_seconds": FRAME_DT,
        "substeps_per_frame": SUBSTEPS, "max_discrete_step_rad": MAX_DISCRETE_STEP,
        "joint_names": JOINT_NAMES, "joint_ranges": model.jnt_range[lay.joints].copy(),
        "actuator_ctrlrange": model.actuator_ctrlrange.copy(),
        "actuator_gain": model.actuator_gainprm[:, 0].copy(),
        "actuator_bias_position": model.actuator_biasprm[:, 1].copy(),
        "contact_config": {
            "robot_geom_ids": robot,
            "obstacle_geom_ids": sorted(lay.obstacle_geoms),
            "robot_contype": model.geom_contype[robot].copy(),
            "robot_conaffinity": model.geom_conaffinity[robot].copy(),
            "obstacle_size": model.geom_size[lay.obstacle].copy(),
        },
    }


def contacts(model: mujoco.MjModel, data: mujoco.MjData, lay: Layout) -> tuple[bool, bool]:
    obstacle, self_collision = False, False
    for c in data.contact[:data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        if (a in lay.obstacle_geoms and b in lay.robot_geoms) or (b in lay.obstacle_geoms and a in lay.robot_geoms):
            obstacle = True
        if a in lay.robot_geoms and b in lay.robot_geoms:
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            # Adjacent link collision primitives overlap by construction; MuJoCo's
            # welded/parent filters remove those. Remaining contacts are reported.
            if ba != bb:
                self_collision = True
    return obstacle, self_collision


def fk_probe(model: mujoco.MjModel, lay: Layout, q: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[lay.qadr] = q
    data.ctrl[:] = q
    mujoco.mj_forward(model, data)
    return data.geom_xpos[lay.probe_geom].copy()


def densify(plan: np.ndarray, start: int = 0) -> np.ndarray:
    chunks = [plan[start:start + 1]]
    for i in range(max(1, start + 1), len(plan)):
        prev = plan[i - 1]
        cur = plan[i]
        n = max(1, int(math.ceil(float(np.max(np.abs(cur - prev))) / MAX_DISCRETE_STEP)))
        chunks.append(np.linspace(prev, cur, n + 1, endpoint=True)[1:])
    return np.concatenate(chunks, axis=0)


def discrete_check(model: mujoco.MjModel, lay: Layout, plan: np.ndarray, obstacles: np.ndarray,
                   start: int = 0) -> dict[str, Any]:
    samples = densify(plan, start)
    data = mujoco.MjData(model)
    place_obstacles(data, lay, obstacles)
    obstacle = self_collision = False
    checked = 0
    if np.any(plan < lay.low - 1e-9) or np.any(plan > lay.high + 1e-9):
        return {"ok": False, "reason": "joint_limit", "samples": 0,
                "obstacle": False, "self_collision": False}
    for q in samples:
        checked += 1
        if np.any(q < lay.low - 1e-9) or np.any(q > lay.high + 1e-9):
            return {"ok": False, "reason": "joint_limit", "samples": checked,
                    "obstacle": obstacle, "self_collision": self_collision}
        data.qpos[lay.qadr] = q
        data.qvel[:] = 0
        data.ctrl[:] = np.clip(q, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(model, data)
        oc, sc = contacts(model, data, lay)
        obstacle |= oc
        self_collision |= sc
        if oc or sc:
            return {"ok": False, "reason": "collision", "samples": checked,
                    "obstacle": obstacle, "self_collision": self_collision}
    return {"ok": True, "reason": "ok", "samples": checked,
            "obstacle": obstacle, "self_collision": self_collision}


def rollout(model: mujoco.MjModel, lay: Layout, plan: np.ndarray, obstacles: np.ndarray,
            disturbance: float = 1.0,
            capture: bool = False) -> dict[str, Any]:
    data = mujoco.MjData(model)
    place_obstacles(data, lay, obstacles)
    data.qpos[lay.qadr] = plan[0]
    data.ctrl[:] = np.clip(plan[0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    original = model.actuator_gainprm[:, 0].copy()
    model.actuator_gainprm[:, 0] *= disturbance
    model.actuator_biasprm[:, 1] *= disturbance
    traj, sites, errors = [], [], []
    obstacle = self_collision = False
    t0 = time.perf_counter()
    try:
        for target in plan:
            data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
                oc, sc = contacts(model, data, lay)
                obstacle |= oc
                self_collision |= sc
            q = data.qpos[lay.qadr].copy()
            errors.append(float(np.sqrt(np.mean((q - target) ** 2))))
            if capture:
                traj.append(q)
                sites.append(data.site_xpos[lay.site].copy())
    finally:
        model.actuator_gainprm[:, 0] = original
        model.actuator_biasprm[:, 1] = -original
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {"ok": not (obstacle or self_collision or rmse > TRACKING_LIMIT_RAD),
            "obstacle": obstacle, "self_collision": self_collision, "tracking_rmse_rad": rmse,
            "wall_ms": (time.perf_counter() - t0) * 1000,
            "qpos": np.asarray(traj), "site": np.asarray(sites)}


def interpolate(a: np.ndarray, b: np.ndarray, frames: int = 41) -> np.ndarray:
    x = np.linspace(0, 1, frames)[:, None]
    s = x * x * (3 - 2 * x)
    return a + s * (b - a)


def make_case(model: mujoco.MjModel, lay: Layout, scenario: str, root: int) -> dict[str, Any]:
    rng = np.random.default_rng(710_000 + root)
    q0 = lay.home.copy()
    goal = q0 + rng.uniform([-.55, -.35, -.45, -.35, -.3, -.4], [.55, .35, .45, .35, .3, .4])
    parent = interpolate(q0, goal)
    final = parent.copy()
    split = 20
    transformed = scenario != "clear"
    disturbance = 1.0
    far = np.array([[2., 2., 2.], [2.2, 2., 2.]])
    parent_obs = far.copy()
    final_obs = far.copy()
    if scenario in ("late_suffix", "environment_change"):
        delta = np.array([rng.choice([-.48, .48]), rng.choice([-.34, .34]), .24, 0, 0, 0])
        ramp = np.linspace(0, 1, len(final) - split)[:, None]
        final[split:] += ramp * delta
    elif scenario == "narrow_passage":
        bump = np.sin(np.linspace(0, math.pi, len(final) - split))[:, None]
        final[split:] += bump * np.array([[rng.choice([-.38, .38]), .22, -.18, 0, 0, 0]])
    elif scenario == "joint_limit":
        final[split:, 2] = np.linspace(final[split, 2], lay.high[2] + .18, len(final) - split)
    elif scenario == "tracking_disturbance":
        final[split:] += np.linspace(0, 1, len(final) - split)[:, None] * np.array([[.75, -.55, .65, .6, -.5, .5]])
        disturbance = .16

    if scenario in ("late_suffix", "narrow_passage", "environment_change"):
        pxyz = np.array([fk_probe(model, lay, q) for q in parent])
        fxyz = np.array([fk_probe(model, lay, q) for q in final])
        sep = np.linalg.norm(fxyz - pxyz, axis=1)
        idx = int(split + np.argmax(sep[split:]))
        candidate = fxyz[idx]
        if scenario == "narrow_passage":
            tangent = fxyz[min(idx + 1, len(fxyz) - 1)] - fxyz[max(idx - 1, 0)]
            tangent /= np.linalg.norm(tangent) + 1e-12
            normal = np.cross(tangent, np.array([0., 0., 1.]))
            normal /= np.linalg.norm(normal) + 1e-12
            # Predeclared paired roots: even roots use a clear 0.32 m half-width;
            # odd roots place the same two-sphere corridor inside collision.
            corridor_half_width = .32 if root % 2 == 0 else .065
            final_obs = np.stack([candidate + corridor_half_width * normal,
                                  candidate - corridor_half_width * normal])
        else:
            final_obs[0] = candidate
        if scenario != "environment_change":
            parent_obs = final_obs.copy()
    return {"root": root, "scenario": scenario, "parent": parent, "final": final,
            "split": split, "parent_obstacles": parent_obs, "final_obstacles": final_obs,
            "disturbance": disturbance, "transformed": transformed}


def context_digest(obstacles: np.ndarray, disturbance: float) -> str:
    return hashlib.sha256(obstacles.tobytes() + np.float64(disturbance).tobytes()).hexdigest()


def validate(model: mujoco.MjModel, lay: Layout, plan: np.ndarray, obstacles: np.ndarray,
             disturbance: float, start: int = 0) -> dict[str, Any]:
    t0 = time.perf_counter()
    static = discrete_check(model, lay, plan, obstacles, start)
    dynamic = rollout(model, lay, plan, obstacles, disturbance, False) if static["ok"] else {
        "ok": False, "obstacle": static["obstacle"], "self_collision": static["self_collision"],
        "tracking_rmse_rad": float("nan"), "wall_ms": 0.0,
    }
    return {"allow": bool(static["ok"] and dynamic["ok"]), "static": static, "dynamic": dynamic,
            "wall_ms": (time.perf_counter() - t0) * 1000,
            "static_samples": static["samples"],
            "dynamic_rollouts": 1 if static["ok"] else 0}


def parent_record(model: mujoco.MjModel, lay: Layout, case: dict[str, Any],
                  parent_gate: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any] | None:
    if not parent_gate["allow"]:
        return None
    split = int(case["split"])
    parent_prefix = case["parent"][:split + 1]
    parent_state = integration_state(model, lay, case["parent"][0], case["parent_obstacles"])
    payload = {
        "kind": "bound_parent_verification_record",
        "parent_allowed": True,
        "reused_prefix_frames_inclusive": [0, split],
        "affected_final_frame_range_inclusive": [split, len(case["final"]) - 1],
        "parent_prefix_sha256": hashlib.sha256(parent_prefix.tobytes()).hexdigest(),
        "final_reused_prefix_sha256": hashlib.sha256(case["final"][:split + 1].tobytes()).hexdigest(),
        "parent_full_plan_sha256": hashlib.sha256(case["parent"].tobytes()).hexdigest(),
        "parent_context_sha256": context_digest(case["parent_obstacles"], 1.0),
        "initial_integration_state_sha256": hashlib.sha256(parent_state.tobytes()).hexdigest(),
        "integration_state_spec": "mjSTATE_INTEGRATION",
        "binding_sha256": object_sha256(binding),
        "binding": binding,
        "parent_gate_summary": {
            "static_ok": parent_gate["static"]["ok"],
            "static_samples": parent_gate["static_samples"],
            "dynamic_ok": parent_gate["dynamic"]["ok"],
            "dynamic_rollouts": parent_gate["dynamic_rollouts"],
        },
    }
    return {**payload, "record_sha256": object_sha256(payload)}


def check_parent_record(model: mujoco.MjModel, lay: Layout, case: dict[str, Any],
                        record: dict[str, Any] | None, binding: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if record is None:
        return False, ["parent_denied_or_record_missing"]
    saved_sha = record.get("record_sha256")
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    if saved_sha != object_sha256(payload):
        reasons.append("record_hash_mismatch")
    if not record.get("parent_allowed", False):
        reasons.append("parent_not_allowed")
    split = int(case["split"])
    parent_prefix = case["parent"][:split + 1]
    final_prefix = case["final"][:split + 1]
    if not np.array_equal(parent_prefix, final_prefix):
        reasons.append("reused_prefix_differs")
    if record.get("parent_prefix_sha256") != hashlib.sha256(final_prefix.tobytes()).hexdigest():
        reasons.append("prefix_hash_mismatch")
    if record.get("final_reused_prefix_sha256") != hashlib.sha256(final_prefix.tobytes()).hexdigest():
        reasons.append("final_prefix_hash_mismatch")
    if record.get("binding_sha256") != object_sha256(binding):
        reasons.append("model_or_validator_binding_changed")
    parent_context = context_digest(case["parent_obstacles"], 1.0)
    final_context = context_digest(case["final_obstacles"], case["disturbance"])
    if parent_context != final_context or record.get("parent_context_sha256") != final_context:
        reasons.append("context_changed")
    final_state = integration_state(model, lay, case["final"][0], case["final_obstacles"])
    if record.get("initial_integration_state_sha256") != hashlib.sha256(final_state.tobytes()).hexdigest():
        reasons.append("initial_integration_state_changed")
    return len(reasons) == 0, reasons


def run_case(model: mujoco.MjModel, lay: Layout, case: dict[str, Any],
             binding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent_digest = context_digest(case["parent_obstacles"], 1.0)
    final_digest = context_digest(case["final_obstacles"], case["disturbance"])
    parent_gate = validate(model, lay, case["parent"], case["parent_obstacles"], 1.0)
    full_gate = validate(model, lay, case["final"], case["final_obstacles"], case["disturbance"])
    record = parent_record(model, lay, case, parent_gate, binding)
    reuse_authorized, fallback_reasons = check_parent_record(model, lay, case, record, binding)
    fallback = not reuse_authorized
    if fallback:
        incremental_gate = validate(model, lay, case["final"], case["final_obstacles"], case["disturbance"])
        reused = 0
        incremental_allow = bool(incremental_gate["allow"])
    else:
        incremental_gate = validate(model, lay, case["final"], case["final_obstacles"], case["disturbance"], case["split"])
        reused = case["split"] + 1
        incremental_allow = bool(parent_gate["allow"] and record is not None and incremental_gate["allow"])
    # Independent evaluator: fresh data, same frozen final plan/context.
    static_truth = discrete_check(model, lay, case["final"], case["final_obstacles"])
    dynamic_truth = rollout(model, lay, case["final"], case["final_obstacles"], case["disturbance"], True)
    unsafe = not (static_truth["ok"] and dynamic_truth["ok"])
    decisions = {
        "parent_only": bool(parent_gate["allow"]),
        "full": bool(full_gate["allow"]),
        "incremental": incremental_allow,
        "conservative": not case["transformed"],
    }
    costs = {
        "parent_only": parent_gate["wall_ms"], "full": full_gate["wall_ms"],
        "incremental": parent_gate["wall_ms"] + incremental_gate["wall_ms"], "conservative": 0.0,
    }
    row = {
        "root": case["root"], "scenario": case["scenario"], "unsafe": unsafe,
        "unsafe_reason": static_truth["reason"] if not static_truth["ok"] else
                         ("dynamic_collision" if dynamic_truth["obstacle"] or dynamic_truth["self_collision"] else
                          "tracking" if not dynamic_truth["ok"] else "safe"),
        "discrete_truth": static_truth, "dynamic_truth": {k: v for k, v in dynamic_truth.items() if k not in ("qpos", "site")},
        "decisions": decisions, "gate_wall_ms": costs, "parent_gate": parent_gate,
        "full_gate": full_gate, "incremental_gate": incremental_gate,
        "incremental_fallback": fallback, "reused_parent_frames": reused,
        "binding_checked": True, "reuse_authorized": reuse_authorized,
        "fallback_reasons": fallback_reasons,
        "affected_final_frame_range_inclusive": [case["split"], len(case["final"]) - 1],
        "parent_verification_record": record,
        "parent_validation_wall_ms": parent_gate["wall_ms"],
        "incremental_suffix_or_fallback_wall_ms": incremental_gate["wall_ms"],
        "incremental_total_wall_ms": costs["incremental"],
        "context_changed": parent_digest != final_digest,
        "parent_context_sha256": parent_digest, "final_context_sha256": final_digest,
        "parent_obstacles": case["parent_obstacles"], "final_obstacles": case["final_obstacles"],
        "disturbance_gain_scale": case["disturbance"],
        "cost_accounting": {
            "parent_static_samples": parent_gate["static_samples"],
            "parent_dynamic_rollouts": parent_gate["dynamic_rollouts"],
            "incremental_static_samples": incremental_gate["static_samples"],
            "incremental_dynamic_rollouts": incremental_gate["dynamic_rollouts"],
            "matched_static_prefix_reuse": reuse_authorized,
            "full_fallback": fallback,
        },
    }
    arrays = {"parent_plan": case["parent"], "final_plan": case["final"],
              "executed_qpos": dynamic_truth["qpos"], "executed_site": dynamic_truth["site"],
              "parent_initial_integration_state": integration_state(
                  model, lay, case["parent"][0], case["parent_obstacles"]),
              "final_initial_integration_state": integration_state(
                  model, lay, case["final"][0], case["final_obstacles"])}
    return row, arrays


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe_n = sum(r["unsafe"] for r in rows)
    safe_n = len(rows) - unsafe_n
    methods = {}
    for m in METHODS:
        allow = [r["decisions"][m] for r in rows]
        false_allow = sum(a and r["unsafe"] for a, r in zip(allow, rows))
        false_reject = sum((not a) and (not r["unsafe"]) for a, r in zip(allow, rows))
        correct_allow = sum(a and not r["unsafe"] for a, r in zip(allow, rows))
        methods[m] = {
            "allow_rate": float(np.mean(allow)), "false_allow_count": false_allow,
            "false_allow_denominator_unsafe": unsafe_n,
            "false_allow_rate": false_allow / unsafe_n if unsafe_n else None,
            "false_reject_count": false_reject, "false_reject_denominator_safe": safe_n,
            "false_reject_rate": false_reject / safe_n if safe_n else None,
            "safe_allowed_count": correct_allow,
            "complete_validation_calls_per_root": 0 if m == "conservative" else (2 if m == "incremental" else 1),
            "gate_wall_ms_mean": float(np.mean([r["gate_wall_ms"][m] for r in rows])),
            "gate_wall_ms_p95": float(np.percentile([r["gate_wall_ms"][m] for r in rows], 95)),
        }
    by_scenario = {}
    for s in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == s]
        by_scenario[s] = {"roots": len(subset), "unsafe": sum(r["unsafe"] for r in subset),
                          **{f"{m}_allowed": sum(r["decisions"][m] for r in subset) for m in METHODS}}
    return {"roots": len(rows), "unsafe_roots": unsafe_n, "safe_roots": safe_n,
            "methods": methods, "by_scenario": by_scenario,
            "incremental_fallback_count": sum(r["incremental_fallback"] for r in rows),
            "definitions": {
                "false_allow": "gate allows final plan AND same-script reference replay labels root unsafe; denominator is unsafe roots",
                "false_reject": "gate rejects final plan AND same-script reference replay labels root safe; denominator is safe roots",
                "unsafe": f"joint violation, robot-obstacle/self collision, or dynamic tracking RMSE > {TRACKING_LIMIT_RAD} rad",
            }}


def render(model: mujoco.MjModel, lay: Layout, row: dict[str, Any], arrays: dict[str, np.ndarray], out: Path) -> dict[str, str]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import av
    from PIL import Image, ImageDraw, ImageFont
    renderer = mujoco.Renderer(model, 720, 1280)
    data = mujoco.MjData(model)
    place_obstacles(data, lay, np.asarray(row["final_obstacles"]))
    frames = []
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 25) if Path(font_path).is_file() else ImageFont.load_default()
    body_font = ImageFont.truetype(font_path, 17) if Path(font_path).is_file() else ImageFont.load_default()
    status_font = ImageFont.truetype(font_path, 19) if Path(font_path).is_file() else ImageFont.load_default()
    for i, q in enumerate(arrays["executed_qpos"]):
        data.qpos[lay.qadr] = q
        data.ctrl[:] = np.clip(arrays["final_plan"][min(i, len(arrays["final_plan"]) - 1)],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="guard_cam")
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((24, 20, 790, 146), radius=16, fill=(5, 11, 23, 225), outline=(78, 111, 158), width=2)
        draw.rectangle((24, 20, 33, 146), fill=(43, 171, 255))
        draw.text((54, 37), f"Sentinel EVC  ·  UR5e  ·  {row['scenario']}", fill="white", font=title_font)
        draw.text((55, 75), f"Root {row['root']}  |  Actual MuJoCo state  |  Obstacles in red/orange",
                  fill=(190, 218, 255), font=body_font)
        draw.text((55, 108), f"FULL GATE: {'ALLOW' if row['decisions']['full'] else 'REJECT'}    OUTCOME: {'UNSAFE' if row['unsafe'] else 'SAFE'}",
                  fill=(255, 190, 80) if row["unsafe"] else (100, 240, 170), font=status_font)
        frames.append(np.asarray(image))
    renderer.close()
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / "ur5e-validation.mp4"
    container = av.open(str(mp4), mode="w")
    stream = container.add_stream("libx264", rate=20)
    stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
    for arr in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    gif = out / "ur5e-validation.gif"
    thumbs = [Image.fromarray(x).resize((640, 360)) for x in frames[::2]]
    thumbs[0].save(gif, save_all=True, append_images=thumbs[1:], duration=100, loop=0, optimize=True)
    png = out / "ur5e-validation.png"
    Image.fromarray(frames[len(frames) // 2]).save(png)
    return {"mp4": str(mp4), "gif": str(gif), "png": str(png)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset-dir", type=Path, required=True, help="pinned universal_robots_ur5e directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots-per-scenario", type=int, default=30)
    p.add_argument("--preflight", action="store_true", help="run 2 roots/scenario")
    p.add_argument("--render", action="store_true")
    p.add_argument("--old-source", type=Path, help="full-v2 executed source to preserve in artifact")
    p.add_argument("--upstream-receipt", type=Path)
    args = p.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    roots_per = 2 if args.preflight else args.roots_per_scenario
    args.out.mkdir(parents=True, exist_ok=True)
    archived_wrapper = make_wrapper(args.asset_dir.resolve(), args.out / "model" / "guard_scene.xml")
    # An include inherits compiler path resolution from its containing file, so
    # load an identical temporary wrapper beside the pinned ur5e.xml.  Remove it
    # before hashing the untouched upstream asset tree.
    load_wrapper = args.asset_dir.resolve() / ".sentinel_guard_scene.xml"
    load_wrapper.write_text(archived_wrapper.read_text())
    try:
        model = mujoco.MjModel.from_xml_path(str(load_wrapper))
    finally:
        load_wrapper.unlink(missing_ok=True)
    lay = layout(model)
    asset_tree_sha, asset_files = tree_hash(args.asset_dir)
    binding = model_binding(model, lay, asset_tree_sha, sha256(archived_wrapper))
    source_dir = args.out / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), source_dir / "run_ur5e_guard.py")
    if args.old_source:
        shutil.copy2(args.old_source, source_dir / "full_v2_executed_run_ur5e_guard.py")
    if args.upstream_receipt:
        shutil.copy2(args.upstream_receipt, source_dir / "upstream_verification.json")
    rows, trace_map = [], {}
    start = time.time()
    for sidx, scenario in enumerate(SCENARIOS):
        for local in range(roots_per):
            root = sidx * 10_000 + local
            row, arrays = run_case(model, lay, make_case(model, lay, scenario, root), binding)
            rows.append(row)
            trace_map[root] = arrays
            print(json.dumps({"root": root, "scenario": scenario, "unsafe": row["unsafe"],
                              "decisions": row["decisions"]}), flush=True)
    summary = summarize(rows)
    manifest = {
        "schema": "sentinel-arm-ur5e-v3", "created_unix": time.time(),
        "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "upstream_repository": "google-deepmind/mujoco_menagerie",
        "upstream_commit": UPSTREAM_COMMIT, "upstream_license": "BSD-3-Clause",
        "upstream_verification": {
            "official_git_tree_api": "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/4d038b3feae26ec82b46a4d586379114012a8ac7?recursive=1",
            "receipt_sha256": sha256(args.upstream_receipt) if args.upstream_receipt else None,
            "source_archive_sha256": "db0e5007e26b06cdadf3974f1f7507a357dad9122176eba65ba33fcf7de6e1a0",
        },
        "asset_tree_sha256": asset_tree_sha, "asset_file_sha256": asset_files,
        "mujoco_version": mujoco.__version__, "python": platform.python_version(),
        "host": platform.node(), "gpu_claim": "none; CPU MuJoCo physics, EGL rendering",
        "imports": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "actual_available_gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip(),
        "scenario_order": SCENARIOS, "root_formula": "scenario_index*10000 + local_index",
        "roots_per_scenario": roots_per, "dt_seconds": DT, "control_frame_seconds": FRAME_DT,
        "substeps_per_frame": SUBSTEPS, "max_discrete_joint_step_rad": MAX_DISCRETE_STEP,
        "discrete_check_claim": "sampled configurations only; not a continuous collision certificate",
        "dynamic_check": "fresh MjData, position target control, 25 mj_step calls per 50 ms frame",
        "experiment_scope": "fixed constructed roots with adversarial obstacle placement from final kinematics; not a natural-scene distribution",
        "preregistered": False,
        "parent_record": "bound in-process verification record; not an externally signed certificate",
        "incremental_semantics": "static prefix reuse only; dynamic rollout always starts at frame 0; invalid record triggers full fallback",
        "reference_outcome_note": "gate-run reference outcome reuses gate implementations and is not independent; use verify_ur5e_roots.py outputs for review labels",
        "wall_seconds": time.time() - start, "gate_reference_summary": summary,
        "binding_sha256": object_sha256(binding), "binding": binding,
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "per_root.json", rows)
    trajectories_path = args.out / "trajectories.npz"
    np.savez_compressed(trajectories_path, **{
        f"root_{root}_{name}": arr for root, arrays in trace_map.items() for name, arr in arrays.items()
    })
    manifest["source_sha256"] = sha256(Path(__file__))
    manifest["raw_trajectories_sha256"] = sha256(trajectories_path)
    manifest["per_root_sha256"] = sha256(args.out / "per_root.json")
    manifest["wrapper_sha256"] = sha256(archived_wrapper)
    if args.old_source:
        manifest["full_v2_source_sha256"] = sha256(source_dir / "full_v2_executed_run_ur5e_guard.py")
    representative = next((r for r in rows if r["decisions"]["parent_only"] and r["unsafe"]), rows[0])
    if args.render:
        manifest["media"] = render(model, lay, representative, trace_map[representative["root"]], args.out / "media")
        for k, v in list(manifest["media"].items()):
            manifest["media"][k + "_sha256"] = sha256(Path(v))
        manifest["media_binding"] = {
            "asset_tree_sha256": asset_tree_sha,
            "scene": representative["scenario"],
            "root": representative["root"],
            "displayed_policy": "full",
            "displayed_decision": representative["decisions"]["full"],
            "displayed_outcome_unsafe": representative["unsafe"],
            "source_sha256": manifest["source_sha256"],
            "raw_trajectories_sha256": manifest["raw_trajectories_sha256"],
            "root_metrics": {"discrete_truth": representative["discrete_truth"],
                             "dynamic_truth": representative["dynamic_truth"]},
        }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
