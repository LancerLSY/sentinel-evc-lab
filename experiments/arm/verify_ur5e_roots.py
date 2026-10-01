#!/usr/bin/env python3
"""High-resolution review of fixed UR5e roots plus evidence-only rendering.

This module deliberately does not import run_ur5e_guard.  It constructs a new
MjModel/MjData, scans at <=0.005 rad, and replays at dt=0.001 with 50 steps per
50 ms target.  The source roots remain constructed adversarial cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
SCENARIOS = (
    "clear", "late_suffix", "narrow_passage", "joint_limit",
    "tracking_disturbance", "environment_change",
)
METHODS = ("parent_only", "full", "incremental", "conservative")
DT = .001
FRAME_DT = .05
SUBSTEPS = 50
MAX_STATIC_STEP = .005
TRACKING_LIMIT = .12
FPS = 20
WIDTH, HEIGHT, VIEW_WIDTH = 1280, 720, 960


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=native) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(root: Path) -> str:
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and not x.name.startswith(".sentinel_")):
        rows.append(f"{p.relative_to(root)}\0{sha256(p)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def make_model(asset_dir: Path, archive: Path) -> mujoco.MjModel:
    xml = f'''<mujoco model="sentinel_ur5e_highres_review">
  <include file="{(asset_dir / 'ur5e.xml').as_posix()}"/>
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="4096"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".055 .07 .105 1" contype="1" conaffinity="1"/>
    <body name="guard_obstacle_body_a" mocap="true" pos="2 2 2"><geom name="guard_obstacle_a" type="sphere" size=".055" rgba="1 .22 .12 .96" contype="1" conaffinity="1"/></body>
    <body name="guard_obstacle_body_b" mocap="true" pos="2 2 2"><geom name="guard_obstacle_b" type="sphere" size=".055" rgba="1 .58 .08 .94" contype="1" conaffinity="1"/></body>
    <light pos="-.3 -1.4 2.4" dir=".1 .35 -.93" diffuse="1 1 1"/>
    <light pos="1.2 .8 1.5" dir="-.7 -.3 -.65" diffuse=".45 .55 .7"/>
  </worldbody>
</mujoco>'''
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(xml)
    temporary = asset_dir / ".sentinel_highres_review.xml"
    temporary.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(temporary))
    finally:
        temporary.unlink(missing_ok=True)


def package_portable_model(asset_dir: Path, model_dir: Path, review_wrapper: Path) -> dict[str, Any]:
    package = model_dir / "portable_ur5e"
    shutil.copytree(asset_dir, package, dirs_exist_ok=True)
    portable = package / "sentinel_highres_review.xml"
    absolute = (asset_dir / "ur5e.xml").as_posix()
    portable.write_text(review_wrapper.read_text().replace(absolute, "ur5e.xml"))
    loaded = mujoco.MjModel.from_xml_path(str(portable))
    files = sorted(x for x in package.rglob("*") if x.is_file())
    official_files = sorted(x for x in asset_dir.rglob("*") if x.is_file())
    return {"directory": str(package), "wrapper": str(portable),
            "wrapper_sha256": sha256(portable), "tree_sha256": tree_sha(package),
            "total_file_count": len(files), "official_asset_file_count": len(official_files),
            "standalone_load_verified": True,
            "loaded_shape": {"nq": loaded.nq, "nv": loaded.nv, "nu": loaded.nu,
                             "ngeom": loaded.ngeom}}


def model_ids(model: mujoco.MjModel) -> dict[str, Any]:
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in JOINT_NAMES])
    obstacles = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_b"),
    ])
    obstacle_bodies = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_b"),
    ])
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = set(range(model.ngeom)) - set(obstacles.tolist()) - {floor}
    return {
        "joints": joints, "qadr": model.jnt_qposadr[joints],
        "low": model.jnt_range[joints, 0].copy(), "high": model.jnt_range[joints, 1].copy(),
        "obstacles": set(obstacles.tolist()), "mocap": model.body_mocapid[obstacle_bodies],
        "robot": robot,
        "site": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"),
    }


def set_scene(data: mujoco.MjData, ids: dict[str, Any], obstacles: np.ndarray) -> None:
    data.mocap_pos[ids["mocap"]] = obstacles


def collision_flags(model: mujoco.MjModel, data: mujoco.MjData,
                    ids: dict[str, Any]) -> tuple[bool, bool]:
    obstacle = self_collision = False
    for contact in data.contact[:data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if ((a in ids["obstacles"] and b in ids["robot"]) or
                (b in ids["obstacles"] and a in ids["robot"])):
            obstacle = True
        if a in ids["robot"] and b in ids["robot"]:
            if int(model.geom_bodyid[a]) != int(model.geom_bodyid[b]):
                self_collision = True
    return obstacle, self_collision


def dense_plan(plan: np.ndarray) -> np.ndarray:
    parts = [plan[:1]]
    for i in range(1, len(plan)):
        count = max(1, int(math.ceil(float(np.max(np.abs(plan[i] - plan[i - 1]))) / MAX_STATIC_STEP)))
        parts.append(np.linspace(plan[i - 1], plan[i], count + 1)[1:])
    return np.concatenate(parts)


def highres_static(model: mujoco.MjModel, ids: dict[str, Any], plan: np.ndarray,
                   obstacles: np.ndarray) -> dict[str, Any]:
    if np.any(plan < ids["low"] - 1e-9) or np.any(plan > ids["high"] + 1e-9):
        return {"ok": False, "reason": "joint_limit", "samples": 0,
                "obstacle": False, "self_collision": False}
    data = mujoco.MjData(model)
    set_scene(data, ids, obstacles)
    obs = self_collision = False
    checked = 0
    for q in dense_plan(plan):
        checked += 1
        data.qpos[ids["qadr"]] = q
        data.qvel[:] = 0
        data.ctrl[:] = np.clip(q, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(model, data)
        oc, sc = collision_flags(model, data, ids)
        obs |= oc
        self_collision |= sc
        if oc or sc:
            return {"ok": False, "reason": "collision", "samples": checked,
                    "obstacle": obs, "self_collision": self_collision}
    return {"ok": True, "reason": "ok", "samples": checked,
            "obstacle": obs, "self_collision": self_collision}


def highres_replay(model: mujoco.MjModel, ids: dict[str, Any], plan: np.ndarray,
                   obstacles: np.ndarray, disturbance: float) -> dict[str, Any]:
    data = mujoco.MjData(model)
    set_scene(data, ids, obstacles)
    data.qpos[ids["qadr"]] = plan[0]
    data.ctrl[:] = np.clip(plan[0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    gain = model.actuator_gainprm[:, 0].copy()
    bias = model.actuator_biasprm[:, 1].copy()
    model.actuator_gainprm[:, 0] *= disturbance
    model.actuator_biasprm[:, 1] *= disturbance
    qpos, site, errors = [], [], []
    obs = self_collision = False
    started = time.perf_counter()
    try:
        for target in plan:
            data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
                oc, sc = collision_flags(model, data, ids)
                obs |= oc
                self_collision |= sc
            actual = data.qpos[ids["qadr"]].copy()
            qpos.append(actual)
            site.append(data.site_xpos[ids["site"]].copy())
            errors.append(float(np.sqrt(np.mean((actual - target) ** 2))))
    finally:
        model.actuator_gainprm[:, 0] = gain
        model.actuator_biasprm[:, 1] = bias
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "ok": not (obs or self_collision or rmse > TRACKING_LIMIT),
        "obstacle": obs, "self_collision": self_collision,
        "tracking_rmse_rad": rmse, "wall_ms": (time.perf_counter() - started) * 1000,
        "qpos": np.asarray(qpos), "site": np.asarray(site),
    }


def policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = sum(x["review_unsafe"] for x in rows)
    safe = len(rows) - unsafe
    methods = {}
    for method in METHODS:
        allowed = sum(x["decisions"][method] for x in rows)
        false_allow = sum(x["decisions"][method] and x["review_unsafe"] for x in rows)
        false_reject = sum((not x["decisions"][method]) and (not x["review_unsafe"]) for x in rows)
        methods[method] = {
            "allow_count": allowed, "allow_denominator_all": len(rows),
            "false_allow_count": false_allow, "false_allow_denominator_unsafe": unsafe,
            "false_reject_count": false_reject, "false_reject_denominator_safe": safe,
        }
    groups = {}
    for name, predicate in (
        ("static_prefix_reuse", lambda x: x["reuse_authorized"]),
        ("full_fallback", lambda x: x["incremental_fallback"]),
    ):
        group = [x for x in rows if predicate(x)]
        groups[name] = {
            "roots": len(group),
            "mean_parent_ms": float(np.mean([x["parent_validation_wall_ms"] for x in group])) if group else None,
            "mean_suffix_or_fallback_ms": float(np.mean([x["incremental_suffix_or_fallback_wall_ms"] for x in group])) if group else None,
            "mean_total_ms": float(np.mean([x["incremental_total_wall_ms"] for x in group])) if group else None,
            "static_samples": int(sum(x["cost_accounting"]["parent_static_samples"] +
                                      x["cost_accounting"]["incremental_static_samples"] for x in group)),
            "dynamic_rollouts": int(sum(x["cost_accounting"]["parent_dynamic_rollouts"] +
                                        x["cost_accounting"]["incremental_dynamic_rollouts"] for x in group)),
        }
    return {
        "roots": len(rows), "review_unsafe_roots": unsafe, "review_safe_roots": safe,
        "unsafe_reason_counts": dict(Counter(x["review_reason"] for x in rows)),
        "methods": methods, "observed_gate_cost_by_incremental_path": groups,
        "fallback_reason_counts": dict(Counter(reason for x in rows for reason in x["fallback_reasons"])),
    }


def fonts() -> tuple[ImageFont.ImageFont, ...]:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if regular.is_file():
        return (ImageFont.truetype(str(bold if bold.is_file() else regular), 32),
                ImageFont.truetype(str(bold if bold.is_file() else regular), 24),
                ImageFont.truetype(str(regular), 18), ImageFont.truetype(str(regular), 15))
    default = ImageFont.load_default()
    return default, default, default, default


def base_canvas() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (5, 10, 20))


def label(text: str) -> str:
    return text.replace("_", " ").title()


def decision(value: bool) -> str:
    return "ALLOW" if value else "REJECT"


def title_card(title: str, subtitle: str, lines: list[str], badge: str) -> np.ndarray:
    title_f, _, body_f, small_f = fonts()
    image = base_canvas()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 14, HEIGHT), fill=(43, 177, 255))
    draw.rounded_rectangle((80, 92, 1196, 628), radius=28, fill=(10, 20, 37), outline=(58, 103, 151), width=2)
    draw.rounded_rectangle((112, 126, 430, 166), radius=14, fill=(18, 75, 112))
    draw.text((133, 137), badge, fill=(175, 230, 255), font=small_f)
    draw.text((112, 206), title, fill="white", font=title_f)
    draw.text((112, 259), subtitle, fill=(153, 197, 232), font=body_f)
    y = 340
    for line in lines:
        draw.ellipse((115, y + 5, 125, y + 15), fill=(43, 177, 255))
        draw.text((145, y), line, fill=(218, 230, 244), font=body_f)
        y += 52
    draw.text((112, 577), "Motion uses stored high-resolution replay qpos at 1×; title and hold frames are labeled.",
              fill=(113, 144, 175), font=small_f)
    return np.asarray(image)


def sidebar(rendered: np.ndarray, row: dict[str, Any], index: int, state_kind: str) -> np.ndarray:
    _, heading_f, body_f, small_f = fonts()
    canvas = base_canvas()
    canvas.paste(Image.fromarray(rendered), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((VIEW_WIDTH, 0, WIDTH, HEIGHT), fill=(7, 14, 27))
    draw.rectangle((VIEW_WIDTH, 0, VIEW_WIDTH + 8, HEIGHT), fill=(43, 177, 255))
    draw.text((990, 38), f"CASE {index:02d}/07", fill=(96, 196, 255), font=small_f)
    draw.text((990, 78), label(row["scenario"]), fill="white", font=heading_f)
    draw.text((990, 119), f"Root {row['root']}", fill=(158, 194, 225), font=body_f)
    draw.line((990, 164, 1248, 164), fill=(52, 77, 107), width=2)
    y = 198
    for name in ("parent_only", "full", "incremental"):
        value = decision(row["decisions"][name])
        draw.text((990, y), name.replace("_", " ").upper(), fill=(143, 171, 201), font=small_f)
        draw.text((990, y + 24), value, fill=(98, 231, 170) if value == "ALLOW" else (255, 181, 66), font=body_f)
        y += 76
    draw.line((990, 432, 1248, 432), fill=(52, 77, 107), width=2)
    outcome = "UNSAFE" if row["review_unsafe"] else "SAFE"
    draw.text((990, 465), "HIGH-RES REPLAY OUTCOME", fill=(143, 171, 201), font=small_f)
    draw.text((990, 493), outcome, fill=(255, 111, 91) if row["review_unsafe"] else (98, 231, 170), font=heading_f)
    draw.text((990, 535), f"Reason: {label(row['review_reason'])}", fill=(205, 220, 235), font=small_f)
    draw.rounded_rectangle((988, 616, 1254, 674), radius=12, fill=(13, 34, 55), outline=(40, 88, 126))
    draw.text((1004, 635), state_kind, fill=(168, 222, 255), font=small_f)
    return np.asarray(canvas)


def render_case(model: mujoco.MjModel, ids: dict[str, Any], row: dict[str, Any],
                qpos: np.ndarray, index: int) -> list[np.ndarray]:
    renderer = mujoco.Renderer(model, HEIGHT, VIEW_WIDTH)
    data = mujoco.MjData(model)
    set_scene(data, ids, np.asarray(row["final_obstacles"]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, .22, .48]
    camera.distance, camera.azimuth, camera.elevation = 1.72, 137, -20
    output = []
    for q in qpos:
        data.qpos[ids["qadr"]] = q
        data.ctrl[:] = np.clip(q, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        output.append(sidebar(renderer.render(), row, index, "STORED JOINT TRAJECTORY · 1×"))
    renderer.close()
    return output


def encode(path: Path, frames: list[np.ndarray]) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"crf": "21", "preset": "medium"}
    for image in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def choose_media(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for scenario in SCENARIOS:
        candidates = [x for x in rows if x["scenario"] == scenario]
        selected.append(candidates[0] if scenario == "clear" else next(x for x in candidates if x["review_unsafe"]))
    selected.append(next(x for x in rows if x["scenario"] == "narrow_passage" and not x["review_unsafe"]))
    return selected


def render_media(model: mujoco.MjModel, ids: dict[str, Any], rows: list[dict[str, Any]],
                 trajectories: Path, out: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    selected = choose_media(rows)
    with np.load(trajectories, allow_pickle=False) as arrays:
        cases = {r["root"]: render_case(model, ids, r, arrays[f"root_{r['root']}_qpos"], i)
                 for i, r in enumerate(selected, 1)}
    frames = []
    frames += [title_card("Sentinel EVC · reviewed UR5e evidence",
                          "Seven deterministic examples from the fixed 180 constructed roots",
                          ["Stored high-resolution replay joint trajectories",
                           "Selection uses review outcomes: first unsafe is outcome-conditioned",
                           "Six scenarios plus the first safe narrow-passage control"],
                          "DETERMINISTIC CASE SELECTION")] * 40
    for i, row in enumerate(selected, 1):
        rule = "first root" if row["scenario"] == "clear" else ("first safe review root" if i == 7 else "first unsafe review root")
        frames += [title_card(f"{i:02d} · {label(row['scenario'])}",
                              f"Root {row['root']} · deterministic outcome-conditioned rule: {rule}",
                              [f"Parent {decision(row['decisions']['parent_only'])} · Full {decision(row['decisions']['full'])}",
                               f"Incremental {decision(row['decisions']['incremental'])}",
                               f"High-res replay {'UNSAFE' if row['review_unsafe'] else 'SAFE'} · {label(row['review_reason'])}"],
                              "CASE INTRO · MOTION PAUSED")] * 24
        states = cases[row["root"]]
        frames += states
        hold = Image.fromarray(states[-1].copy())
        draw = ImageDraw.Draw(hold)
        _, _, _, small_f = fonts()
        draw.rounded_rectangle((28, 28, 255, 70), radius=10, fill=(8, 18, 32), outline=(79, 125, 165))
        draw.text((48, 41), "FINAL STATE HOLD", fill=(177, 220, 250), font=small_f)
        frames += [np.asarray(hold)] * 20
    frames += [title_card("Reviewed fixed-root summary",
                          f"{metrics['roots']} roots · {metrics['review_unsafe_roots']} unsafe · {metrics['review_safe_roots']} safe",
                          [f"Full false allow: {metrics['methods']['full']['false_allow_count']}/{metrics['review_unsafe_roots']}",
                           f"Incremental false allow: {metrics['methods']['incremental']['false_allow_count']}/{metrics['review_unsafe_roots']}",
                           "Constructed adversarial roots; no natural-scene distribution claim"],
                          "DETERMINISTIC CASE SELECTION")] * 40
    mp4 = out / "ur5e-validation-montage.mp4"
    encode(mp4, frames)
    late = next(x for x in selected if x["scenario"] == "late_suffix")
    gif_images = [Image.fromarray(x).resize((640, 360), Image.Resampling.LANCZOS) for x in cases[late["root"]][::2]]
    gif_images += [gif_images[-1]] * 8
    gif = out / "ur5e-validation.gif"
    gif_images[0].save(gif, save_all=True, append_images=gif_images[1:], duration=100, loop=0, optimize=True, disposal=2)
    with Image.open(gif) as encoded_gif:
        encoded_gif_frames = encoded_gif.n_frames
        encoded_gif_duration_ms = 0
        for frame_index in range(encoded_gif.n_frames):
            encoded_gif.seek(frame_index)
            encoded_gif_duration_ms += int(encoded_gif.info.get("duration", 0))
    png = out / "ur5e-validation.png"
    Image.fromarray(cases[late["root"]][-1]).save(png)
    media = {}
    for path in (mp4, gif, png):
        media[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    media[mp4.name].update({"width": WIDTH, "height": HEIGHT, "fps": FPS,
                            "frame_count": len(frames), "duration_seconds": len(frames) / FPS,
                            "stored_joint_state_frames": len(selected) * 41,
                            "title_frames": 40 + 40 + len(selected) * 24,
                            "hold_frames": len(selected) * 20})
    media[gif.name].update({"width": 640, "height": 360, "nominal_fps": 10,
                            "logical_frame_count": len(gif_images),
                            "encoded_frame_count": encoded_gif_frames,
                            "encoded_duration_seconds": encoded_gif_duration_ms / 1000,
                            "root": late["root"]})
    media[png.name].update({"width": WIDTH, "height": HEIGHT, "root": late["root"], "state_index": 40})
    return {"selection_rule": "clear first root; first review-unsafe root for other scenarios; then first review-safe narrow root (outcome-conditioned)",
            "selected": [{"segment": i, "root": r["root"], "scenario": r["scenario"],
                          "review_unsafe": r["review_unsafe"], "review_reason": r["review_reason"]}
                         for i, r in enumerate(selected, 1)], "media": media}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--media-out", type=Path, required=True)
    parser.add_argument("--upstream-receipt", type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    manifest_path, per_root_path = args.run / "manifest.json", args.run / "per_root.json"
    plan_path = args.run / "trajectories.npz"
    parent = json.loads(manifest_path.read_text())
    rows = json.loads(per_root_path.read_text())
    if sha256(plan_path) != parent["raw_trajectories_sha256"]:
        raise RuntimeError("plan artifact hash mismatch")
    asset_sha = tree_sha(args.asset_dir)
    if asset_sha != parent["asset_tree_sha256"]:
        raise RuntimeError("asset tree hash mismatch")
    model = make_model(args.asset_dir.resolve(), args.run / "review" / "highres_scene.xml")
    ids = model_ids(model)
    reviewed, arrays_out = [], {}
    with np.load(plan_path, allow_pickle=False) as plans:
        for row in rows:
            root = row["root"]
            plan = plans[f"root_{root}_final_plan"]
            obstacles = np.asarray(row["final_obstacles"], dtype=np.float64)
            disturbance = float(row["disturbance_gain_scale"])
            static = highres_static(model, ids, plan, obstacles)
            replay = highres_replay(model, ids, plan, obstacles, disturbance)
            unsafe = not (static["ok"] and replay["ok"])
            reason = static["reason"] if not static["ok"] else (
                "dynamic_collision" if replay["obstacle"] or replay["self_collision"] else
                "tracking" if not replay["ok"] else "safe")
            reviewed.append({**row, "review_unsafe": unsafe, "review_reason": reason,
                             "highres_static": static,
                             "highres_replay": {k: v for k, v in replay.items() if k not in ("qpos", "site")}})
            arrays_out[f"root_{root}_qpos"] = replay["qpos"]
            arrays_out[f"root_{root}_site"] = replay["site"]
            print(json.dumps({"root": root, "review_unsafe": unsafe, "reason": reason}), flush=True)
    review_dir = args.run / "review"
    reviewed_path = review_dir / "reviewed_per_root.json"
    trajectories = review_dir / "highres_replay_trajectories.npz"
    write_json(reviewed_path, reviewed)
    np.savez_compressed(trajectories, **arrays_out)
    metrics = policy_metrics(reviewed)
    write_json(review_dir / "reviewed_metrics.json", metrics)
    shutil.copy2(Path(__file__), args.run / "source" / "verify_ur5e_roots.py")
    if args.upstream_receipt:
        shutil.copy2(args.upstream_receipt, review_dir / "upstream_verification.json")
    media = render_media(model, ids, reviewed, trajectories, args.media_out, metrics)
    portable_model = package_portable_model(args.asset_dir.resolve(), args.media_out / "model",
                                            review_dir / "highres_scene.xml")
    previous = None
    if args.previous_run:
        old = json.loads((args.previous_run / "manifest.json").read_text())
        previous = {"manifest_sha256": sha256(args.previous_run / "manifest.json"),
                    "reported_summary": old.get("summary"),
                    "note": "separate earlier 180-root run; not recomputed or pooled with v3; legacy summary contains outdated independent-evaluator wording and is historical only"}
    verification = {
        "schema": "sentinel-arm-highres-review-v1", "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "source_sha256": sha256(Path(__file__)),
        "imports": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                    "numpy": np.__version__, "av": av.__version__},
        "input_hashes": {"run_manifest": sha256(manifest_path), "per_root": sha256(per_root_path),
                         "plans": sha256(plan_path), "asset_tree": asset_sha,
                         "review_wrapper": sha256(review_dir / "highres_scene.xml")},
        "output_hashes": {"reviewed_per_root": sha256(reviewed_path),
                          "highres_replay_trajectories": sha256(trajectories),
                          "reviewed_metrics": sha256(review_dir / "reviewed_metrics.json")},
        "resolution": {"static_max_joint_step_rad": MAX_STATIC_STEP, "dt_seconds": DT,
                       "control_frame_seconds": FRAME_DT, "substeps_per_frame": SUBSTEPS,
                       "difference_from_gate": "2x denser static sampling and 2x temporal physics resolution"},
        "implementation_independence": "does not import or call run_ur5e_guard discrete_check/rollout; uses a new MjModel and MjData",
        "scope": "same fixed 180 constructed roots with adversarial obstacle placement; not an untouched natural distribution",
        "asset_archive_receipt": {"declared_upstream_commit": parent["upstream_commit"],
                                  "tree_sha256": asset_sha,
                                  "verification_status": "independently verified by root integrator",
                                  "official_git_tree_api": "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/4d038b3feae26ec82b46a4d586379114012a8ac7?recursive=1",
                                  "receipt_sha256": sha256(args.upstream_receipt) if args.upstream_receipt else None,
                                  "source_archive_sha256": "db0e5007e26b06cdadf3974f1f7507a357dad9122176eba65ba33fcf7de6e1a0"},
        "metrics": metrics, "previous_run": previous,
        "media_v4": media, "portable_model": portable_model,
        "environment": {"actual_available_gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
            "physics": "CPU MuJoCo high-resolution replay", "render": "EGL"},
    }
    write_json(review_dir / "verification_manifest.json", verification)
    media_manifest = {"schema": "sentinel-arm-media-v4", "gate_recomputed": False,
                      "physics_during_media_render": False,
                      "state_source": "stored high-resolution replay executed joint trajectory",
                      "review_manifest_sha256": sha256(review_dir / "verification_manifest.json"),
                      "renderer_sha256": sha256(Path(__file__)),
                      "portable_model": portable_model, **media}
    write_json(args.media_out / "media_manifest.json", media_manifest)
    print(json.dumps({"metrics": metrics, "media": media}, indent=2), flush=True)


if __name__ == "__main__":
    main()
