#!/usr/bin/env python3
"""Render a clear paper/demo video from frozen UR5e MuJoCo evidence.

The left pane replays the independently reviewed final-plan trajectory.  When
the final gate rejected that plan, the left pane is explicitly labelled as a
counterfactual: it is evidence for *why* the plan was blocked, not motion that
Sentinel dispatched.  The right pane visualizes the gate decision from the
same initial state.  Safe controls execute the stored trajectory; rejected
plans hold because no candidate motion is dispatched.

This script does not rerun the experiment, recompute gate decisions, or claim
continuous collision certification.  It only renders stored states and frozen
review metadata with the full MuJoCo Menagerie UR5e meshes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
PANEL_WIDTH, PANEL_HEIGHT = 900, 675
FPS = 20
CONTROL_DT_SECONDS = 0.05
CARD_FRAMES = 38
INTRO_FRAMES = 54
HOLD_FRAMES = 18
OUTRO_FRAMES = 54
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass(frozen=True)
class CaseSpec:
    root: int
    scenario: str
    short_title: str
    short_title_zh: str
    task: str
    task_zh: str
    changed: str
    changed_zh: str
    expected_review_reason: str
    camera_lookat: tuple[float, float, float]
    camera_distance: float
    camera_azimuth: float
    camera_elevation: float = -20.0


# Explanatory examples are fixed by ID before rendering.  They are deliberately
# not presented as a statistical sample; aggregate claims come from all 180 roots.
CASES = (
    CaseSpec(
        root=0,
        scenario="clear",
        short_title="Safe control",
        short_title_zh="安全对照",
        task="Execute an unchanged collision-free joint trajectory.",
        task_zh="执行一条未经变换且评审为安全的关节轨迹。",
        changed="No context or suffix change.",
        changed_zh="场景上下文与轨迹后缀均未变化。",
        expected_review_reason="safe",
        camera_lookat=(0.0, 0.22, 0.46),
        camera_distance=1.62,
        camera_azimuth=137.0,
    ),
    CaseSpec(
        root=10000,
        scenario="late_suffix",
        short_title="Changed suffix",
        short_title_zh="后缀发生变化",
        task="Re-check a plan after its final motion segment changes.",
        task_zh="最终运动段变化后，重新检查完整计划。",
        changed="The transformed suffix intersects a guard obstacle.",
        changed_zh="变换后的后缀与防护障碍物相交。",
        expected_review_reason="collision",
        camera_lookat=(-0.08, 0.28, 0.48),
        camera_distance=1.58,
        camera_azimuth=132.0,
    ),
    CaseSpec(
        root=50000,
        scenario="environment_change",
        short_title="Scene changed",
        short_title_zh="场景发生变化",
        task="Reject reuse when the scene no longer matches the parent check.",
        task_zh="场景与父验证不再一致时，拒绝复用旧结论。",
        changed="A new obstacle invalidates the earlier context binding.",
        changed_zh="新增障碍物使原有上下文绑定失效。",
        expected_review_reason="collision",
        camera_lookat=(0.03, 0.28, 0.47),
        camera_distance=1.58,
        camera_azimuth=137.0,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(f"{path.relative_to(root)}\0{sha256(path)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _font(candidates: Iterable[Path], size: int) -> ImageFont.ImageFont:
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def font_pack(explicit_font: Path | None) -> tuple[dict[str, ImageFont.ImageFont], bool, str]:
    cjk_candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    cjk_bold_candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    latin_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    latin_bold_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    explicit = (explicit_font,) if explicit_font is not None else ()
    if explicit_font is not None and not explicit_font.is_file():
        raise RuntimeError(f"configured font does not exist: {explicit_font}")
    has_cjk = bool(explicit) or any(path.is_file() for path in cjk_candidates)
    regular = explicit + cjk_candidates + latin_candidates if has_cjk else latin_candidates
    # Some minimal GPU images ship only the regular CJK face.  Prefer a regular
    # CJK face over a bold Latin face that would render Chinese labels as boxes.
    bold = (explicit + cjk_bold_candidates + cjk_candidates + latin_bold_candidates
            if has_cjk else latin_bold_candidates)
    fonts = {
        "hero": _font(bold, 52),
        "title": _font(bold, 34),
        "heading": _font(bold, 26),
        "body": _font(regular, 22),
        "small": _font(regular, 18),
        "tiny": _font(regular, 15),
    }
    selected = next((str(path) for path in regular if path.is_file()), "Pillow default")
    return fonts, has_cjk, selected


def bilingual(en: str, zh: str, has_cjk: bool) -> str:
    return f"{en}  |  {zh}" if has_cjk else en


def reason_label(reason: str, has_cjk: bool) -> str:
    chinese = {
        "safe": "安全",
        "collision": "碰撞",
        "joint_limit": "关节限位",
        "tracking": "跟踪误差",
    }.get(reason, reason)
    return bilingual(reason.replace("_", " ").upper(), chinese, has_cjk)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def card_frame(title: str, subtitle: str, lines: list[str], fonts: dict[str, ImageFont.ImageFont],
               badge: str) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (5, 11, 22))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill=(35, 184, 255))
    draw.rounded_rectangle((135, 125, WIDTH - 135, HEIGHT - 125), radius=32,
                           fill=(11, 24, 43), outline=(49, 99, 142), width=3)
    draw.rounded_rectangle((180, 170, 535, 220), radius=16, fill=(16, 79, 119))
    draw.text((205, 183), badge, font=fonts["small"], fill=(183, 232, 255))
    draw.text((180, 284), title, font=fonts["hero"], fill=(244, 249, 255))
    draw.text((182, 360), subtitle, font=fonts["heading"], fill=(141, 197, 233))
    y = 475
    for line in lines:
        draw.ellipse((184, y + 8, 198, y + 22), fill=(35, 184, 255))
        draw.text((222, y), line, font=fonts["body"], fill=(215, 230, 243))
        y += 76
    draw.text((182, 865), "Frozen MuJoCo evidence replay · stored joint states · 1× motion",
              font=fonts["small"], fill=(102, 139, 170))
    return image


def model_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, set[int]]:
    joints = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
    ])
    qpos_addresses = model.jnt_qposadr[joints]
    body_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guard_obstacle_body_b"),
    ])
    geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "guard_obstacle_b"),
    }
    return qpos_addresses, model.body_mocapid[body_ids], geom_ids


def set_state(model: mujoco.MjModel, data: mujoco.MjData, qpos_addresses: np.ndarray,
              mocap_ids: np.ndarray, obstacles: list[list[float]], qpos: np.ndarray) -> None:
    data.qpos[qpos_addresses] = qpos
    data.ctrl[:] = np.clip(qpos, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.mocap_pos[mocap_ids] = np.asarray(obstacles, dtype=np.float64)
    mujoco.mj_forward(model, data)


def obstacle_contact(data: mujoco.MjData, obstacle_geom_ids: set[int]) -> bool:
    return any(
        int(data.contact[i].geom1) in obstacle_geom_ids
        or int(data.contact[i].geom2) in obstacle_geom_ids
        for i in range(data.ncon)
    )


def first_contact_index(model: mujoco.MjModel, qpos_addresses: np.ndarray, mocap_ids: np.ndarray,
                        obstacle_geom_ids: set[int], obstacles: list[list[float]],
                        trajectory: np.ndarray) -> int | None:
    data = mujoco.MjData(model)
    for index, qpos in enumerate(trajectory):
        set_state(model, data, qpos_addresses, mocap_ids, obstacles, qpos)
        if obstacle_contact(data, obstacle_geom_ids):
            return index
    return None


def render_state(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
                 qpos_addresses: np.ndarray, mocap_ids: np.ndarray, row: dict[str, Any],
                 qpos: np.ndarray, spec: CaseSpec) -> Image.Image:
    set_state(model, data, qpos_addresses, mocap_ids, row["final_obstacles"], qpos)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(spec.camera_lookat)
    camera.distance = spec.camera_distance
    camera.azimuth = spec.camera_azimuth
    camera.elevation = spec.camera_elevation
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render())


def draw_pane(draw: ImageDraw.ImageDraw, origin: tuple[int, int], title: str, subtitle: str,
              color: tuple[int, int, int], fonts: dict[str, ImageFont.ImageFont]) -> None:
    x, y = origin
    draw.rounded_rectangle((x - 3, y - 3, x + PANEL_WIDTH + 3, y + PANEL_HEIGHT + 3),
                           radius=12, outline=color, width=4)
    draw.rounded_rectangle((x + 18, y + 18, x + PANEL_WIDTH - 18, y + 89), radius=12,
                           fill=(7, 15, 27), outline=color, width=2)
    draw.text((x + 38, y + 29), title, font=fonts["heading"], fill=color)
    draw.text((x + 39, y + 61), subtitle, font=fonts["tiny"], fill=(196, 213, 228))


def evidence_frame(left: Image.Image, right: Image.Image, spec: CaseSpec, row: dict[str, Any],
                   frame_index: int, contact_index: int | None,
                   fonts: dict[str, ImageFont.ImageFont], has_cjk: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (5, 11, 22))
    image.paste(left, (30, 154))
    image.paste(right, (990, 154))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 124), fill=(8, 18, 33))
    draw.rectangle((0, 122, WIDTH, 128), fill=(35, 184, 255))
    case_title = bilingual(spec.short_title, spec.short_title_zh, has_cjk)
    draw.text((32, 25), f"{case_title} · Root {spec.root}", font=fonts["title"],
              fill=(241, 248, 255))
    phase = (bilingual("SHARED PREFIX", "共同前缀", has_cjk) if frame_index < 20
             else bilingual("TRANSFORMED SUFFIX", "变换后缀", has_cjk))
    draw.rounded_rectangle((1390, 27, 1888, 94), radius=18, fill=(15, 46, 69),
                           outline=(55, 151, 201), width=2)
    centered(draw, (1639, 44), phase, fonts["small"], (174, 225, 251))

    contact_now = contact_index is not None and frame_index >= contact_index
    left_color = (255, 91, 76) if row["review_unsafe"] and contact_now else (255, 178, 57)
    if not row["review_unsafe"]:
        left_color = (80, 224, 157)
    right_color = (80, 224, 157) if row["decisions"]["full"] else (73, 188, 255)
    left_title = (bilingual("COUNTERFACTUAL: parent-only branch", "反事实：仅复用父验证", has_cjk)
                  if row["review_unsafe"] else bilingual("PROPOSED PLAN", "候选计划", has_cjk))
    left_sub = (bilingual("Reviewed trajectory; not dispatched", "审阅轨迹；未下发", has_cjk)
                if row["review_unsafe"] else bilingual("Stored safe-control trajectory", "存储的安全对照轨迹", has_cjk))
    right_title = (bilingual("SENTINEL: EXECUTE", "Sentinel：执行", has_cjk)
                   if row["decisions"]["full"] else bilingual("SENTINEL: REJECT / HOLD", "Sentinel：拒绝并保持", has_cjk))
    right_sub = (bilingual("Final gate allows this exact plan", "最终门禁允许此精确计划", has_cjk)
                 if row["decisions"]["full"] else bilingual("No candidate motion is dispatched", "候选动作未下发", has_cjk))
    draw_pane(draw, (30, 154), left_title, left_sub, left_color, fonts)
    draw_pane(draw, (990, 154), right_title, right_sub, right_color, fonts)

    if row["review_unsafe"] and contact_now:
        draw.rounded_rectangle((300, 720, 660, 798), radius=18, fill=(77, 15, 18),
                               outline=(255, 91, 76), width=3)
        centered(draw, (480, 736), bilingual("RISK OBSERVED", "已观测到风险", has_cjk),
                 fonts["heading"], (255, 138, 126))

    time_seconds = frame_index * CONTROL_DT_SECONDS
    gate_ms = float(row["gate_wall_ms"]["full"])
    reason = reason_label(row["review_reason"], has_cjk)
    draw.rounded_rectangle((30, 855, 1890, 1048), radius=20, fill=(10, 23, 40),
                           outline=(38, 69, 96), width=2)
    task = bilingual(spec.task, spec.task_zh, has_cjk)
    changed = bilingual(spec.changed, spec.changed_zh, has_cjk)
    draw.text((57, 880), f"TASK / 任务  {task}", font=fonts["small"], fill=(222, 235, 247))
    draw.text((57, 916), f"CHANGE / 变化  {changed}", font=fonts["small"], fill=(170, 203, 227))
    decision = "ALLOW + EXECUTE" if row["decisions"]["full"] else "REJECT + HOLD"
    decision_zh = "允许并执行" if row["decisions"]["full"] else "拒绝并保持"
    draw.text((57, 958), f"FINAL GATE / 最终门禁  {decision} / {decision_zh}   ·   REVIEW / 评审  {reason}   ·   {gate_ms:.1f} ms",
              font=fonts["small"], fill=right_color)
    bar_x0, bar_x1, bar_y = 1450, 1858, 987
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=(65, 91, 115), width=8)
    progress = frame_index / 40.0
    draw.line((bar_x0, bar_y, bar_x0 + int((bar_x1 - bar_x0) * progress), bar_y),
              fill=(35, 184, 255), width=8)
    draw.ellipse((bar_x0 + int((bar_x1 - bar_x0) * progress) - 8, bar_y - 8,
                  bar_x0 + int((bar_x1 - bar_x0) * progress) + 8, bar_y + 8),
                 fill=(223, 243, 255))
    draw.text((1450, 1011), f"t = {time_seconds:0.2f} s / 2.00 s", font=fonts["tiny"],
              fill=(154, 184, 207))
    return image


class AvWriter:
    def __init__(self, path: Path) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV with the libx264 encoder is required") from exc
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=FPS)
        self.stream.width = WIDTH
        self.stream.height = HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}

    def write(self, frame: Image.Image) -> None:
        video_frame = self.av.VideoFrame.from_ndarray(
            np.asarray(frame.convert("RGB"), dtype=np.uint8), format="rgb24"
        )
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def repeat(writer: AvWriter, frame: Image.Image, count: int) -> int:
    for _ in range(count):
        writer.write(frame)
    return count


def load_and_verify(run: Path, portable_model: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, np.ndarray], dict[str, str]]:
    review_dir = run / "review"
    run_manifest_path = run / "manifest.json"
    review_manifest_path = review_dir / "verification_manifest.json"
    rows_path = review_dir / "reviewed_per_root.json"
    trajectories_path = review_dir / "highres_replay_trajectories.npz"
    run_manifest = json.loads(run_manifest_path.read_text())
    review_manifest = json.loads(review_manifest_path.read_text())
    rows = json.loads(rows_path.read_text())
    expected = review_manifest["output_hashes"]
    checks = {
        "reviewed_per_root_sha256": sha256(rows_path),
        "highres_replay_trajectories_sha256": sha256(trajectories_path),
        "portable_model_tree_sha256": tree_hash(portable_model),
    }
    if checks["reviewed_per_root_sha256"] != expected["reviewed_per_root"]:
        raise RuntimeError("reviewed_per_root.json differs from the frozen review manifest")
    if checks["highres_replay_trajectories_sha256"] != expected["highres_replay_trajectories"]:
        raise RuntimeError("high-resolution trajectories differ from the frozen review manifest")
    if checks["portable_model_tree_sha256"] != review_manifest["portable_model"]["tree_sha256"]:
        raise RuntimeError("portable UR5e model tree differs from the frozen review manifest")
    by_root = {int(row["root"]): row for row in rows}
    selected_rows = []
    for spec in CASES:
        row = by_root.get(spec.root)
        if row is None or row["scenario"] != spec.scenario:
            raise RuntimeError(f"fixed case root {spec.root} is missing or has the wrong scenario")
        if row["review_reason"] != spec.expected_review_reason:
            raise RuntimeError(f"fixed case root {spec.root} review label changed")
        selected_rows.append(row)
    with np.load(trajectories_path, allow_pickle=False) as arrays:
        trajectories = {}
        for spec in CASES:
            key = f"root_{spec.root}_qpos"
            trajectory = np.array(arrays[key], dtype=np.float64, copy=True)
            if trajectory.shape != (41, 6):
                raise RuntimeError(f"{key} has unexpected shape {trajectory.shape}")
            trajectories[spec.root] = trajectory
    checks.update({
        "run_manifest_sha256": sha256(run_manifest_path),
        "review_manifest_sha256": sha256(review_manifest_path),
        "run_source_sha256": run_manifest["source_sha256"],
        "review_source_sha256": review_manifest["source_sha256"],
    })
    return review_manifest, selected_rows, trajectories, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="frozen full-v3 run containing review outputs")
    parser.add_argument("--portable-model", type=Path, required=True,
                        help="verified portable_ur5e directory from media-v4")
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory; must not contain output files")
    parser.add_argument("--font-path", type=Path,
                        help="optional CJK-capable OTF/TTC font for bilingual labels")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    occupied = [args.out / name for name in ("sentinel-ur5e-paper.mp4", "sentinel-ur5e-paper-poster.png", "video_manifest.json")]
    if any(path.exists() for path in occupied):
        raise RuntimeError("refusing to overwrite existing paper-video outputs")

    fonts, has_cjk, font_path = font_pack(args.font_path.resolve() if args.font_path else None)
    review_manifest, rows, trajectories, input_hashes = load_and_verify(
        args.run.resolve(), args.portable_model.resolve()
    )
    model_path = args.portable_model.resolve() / "sentinel_highres_review.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos_addresses, mocap_ids, obstacle_geom_ids = model_ids(model)
    renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
    data = mujoco.MjData(model)
    contact_indices = {
        spec.root: first_contact_index(
            model, qpos_addresses, mocap_ids, obstacle_geom_ids,
            row["final_obstacles"], trajectories[spec.root],
        )
        for spec, row in zip(CASES, rows)
    }

    output_video = args.out / "sentinel-ur5e-paper.mp4"
    writer = AvWriter(output_video)
    frame_count = 0
    poster: Image.Image | None = None
    try:
        intro = card_frame(
            bilingual("Why final actions must be checked", "为什么最终动作必须重新检查", has_cjk),
            "Sentinel EVC · UR5e full-mesh MuJoCo evidence",
            [
                bilingual("Left: what the changed plan would do", "左：变换后计划会发生什么", has_cjk),
                bilingual("Right: the motion Sentinel actually permits", "右：Sentinel 实际允许的动作", has_cjk),
                bilingual("Three fixed IDs from the frozen 180-root review", "来自冻结 180 根评审的三个固定编号", has_cjk),
            ],
            fonts,
            "EVIDENCE REPLAY · MUJOCO SIMULATION",
        )
        frame_count += repeat(writer, intro, INTRO_FRAMES)

        for chapter, (spec, row) in enumerate(zip(CASES, rows), 1):
            allow = bool(row["decisions"]["full"])
            gate_action = "ALLOW + EXECUTE" if allow else "REJECT + HOLD"
            case_card = card_frame(
                f"{chapter:02d} · {bilingual(spec.short_title, spec.short_title_zh, has_cjk)}",
                f"Root {spec.root} · final gate: {gate_action}",
                [
                    bilingual(f"Task: {spec.task}", f"任务：{spec.task_zh}", has_cjk),
                    bilingual(f"Change: {spec.changed}", f"变化：{spec.changed_zh}", has_cjk),
                    bilingual(
                        f"Independent review: {row['review_reason'].replace('_', ' ')}",
                        f"独立评审：{reason_label(row['review_reason'], True).split('  |  ')[-1]}",
                        has_cjk,
                    ),
                ],
                fonts,
                "CASE INTRO · MOTION PAUSED",
            )
            frame_count += repeat(writer, case_card, CARD_FRAMES)
            trajectory = trajectories[spec.root]
            for index, proposed_qpos in enumerate(trajectory):
                sentinel_qpos = proposed_qpos if allow else trajectory[0]
                left = render_state(renderer, model, data, qpos_addresses, mocap_ids,
                                    row, proposed_qpos, spec)
                right = render_state(renderer, model, data, qpos_addresses, mocap_ids,
                                     row, sentinel_qpos, spec)
                composed = evidence_frame(left, right, spec, row, index,
                                          contact_indices[spec.root], fonts, has_cjk)
                writer.write(composed)
                frame_count += 1
                if spec.root == 10000:
                    target_index = contact_indices[spec.root] if contact_indices[spec.root] is not None else 32
                    if index == target_index:
                        poster = composed.copy()
            hold_left = render_state(renderer, model, data, qpos_addresses, mocap_ids,
                                     row, trajectory[-1], spec)
            hold_right = render_state(renderer, model, data, qpos_addresses, mocap_ids,
                                      row, trajectory[-1] if allow else trajectory[0], spec)
            hold = evidence_frame(hold_left, hold_right, spec, row, 40,
                                  contact_indices[spec.root], fonts, has_cjk)
            frame_count += repeat(writer, hold, HOLD_FRAMES)

        metrics = review_manifest["metrics"]
        methods = metrics["methods"]
        outro = card_frame(
            bilingual("What the frozen review establishes", "冻结评审说明了什么", has_cjk),
            "180 constructed roots · independent high-resolution replay",
            [
                bilingual(
                    f"Parent-only: {methods['parent_only']['false_allow_count']}/{methods['parent_only']['false_allow_denominator_unsafe']} false allows",
                    f"仅复用父验证：{methods['parent_only']['false_allow_count']}/{methods['parent_only']['false_allow_denominator_unsafe']} 次错误放行",
                    has_cjk,
                ),
                bilingual(
                    f"Full final gate: {methods['full']['false_allow_count']}/{methods['full']['false_allow_denominator_unsafe']} observed false allows",
                    f"完整最终门禁：{methods['full']['false_allow_count']}/{methods['full']['false_allow_denominator_unsafe']} 次观测到错误放行",
                    has_cjk,
                ),
                bilingual(
                    "Scope: constructed MuJoCo cases; not real hardware or a continuous collision proof",
                    "范围：构造的 MuJoCo 案例；不是真机验证，也不是连续碰撞证明",
                    has_cjk,
                ),
            ],
            fonts,
            "AGGREGATE RESULT · ALL ROOTS",
        )
        frame_count += repeat(writer, outro, OUTRO_FRAMES)
    finally:
        renderer.close()
        writer.close()

    if poster is None:
        raise RuntimeError("poster frame was not produced")
    output_poster = args.out / "sentinel-ur5e-paper-poster.png"
    poster.save(output_poster)
    renderer_sha = sha256(Path(__file__))
    selected = []
    for chapter, (spec, row) in enumerate(zip(CASES, rows), 1):
        selected.append({
            "chapter": chapter,
            "root": spec.root,
            "scenario": spec.scenario,
            "selection_rule": "fixed explanatory ID declared in renderer source and storyboard",
            "review_reason": row["review_reason"],
            "review_unsafe": row["review_unsafe"],
            "parent_only_allow": row["decisions"]["parent_only"],
            "full_final_gate_allow": row["decisions"]["full"],
            "contact_frame_from_render_model": contact_indices[spec.root],
            "full_gate_wall_ms": row["gate_wall_ms"]["full"],
        })
    manifest = {
        "schema": "sentinel-ur5e-paper-video-v1",
        "render_only": True,
        "physics_rerun": False,
        "gate_recomputed": False,
        "resolution": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "frame_count": frame_count,
        "duration_seconds": frame_count / FPS,
        "language": "English + Simplified Chinese" if has_cjk else "English (CJK font unavailable)",
        "font": font_path,
        "source_semantics": {
            "left": "stored independently reviewed high-resolution final-plan qpos; rejected branches are counterfactual and were not dispatched",
            "right_allowed": "same stored qpos for a plan the frozen full gate allowed",
            "right_rejected": "hold at the same initial qpos because the frozen full gate rejected the candidate before dispatch",
            "obstacles": "final_obstacles from reviewed_per_root.json",
            "motion_timing": "41 stored states at 0.05 s per state interval (1x)",
        },
        "selection": {
            "purpose": "fixed explanatory cases, not a statistical sample",
            "root_ids": [spec.root for spec in CASES],
            "cases": selected,
        },
        "inputs": input_hashes,
        "outputs": {
            output_video.name: {"sha256": sha256(output_video), "bytes": output_video.stat().st_size},
            output_poster.name: {"sha256": sha256(output_poster), "bytes": output_poster.stat().st_size},
        },
        "renderer_sha256": renderer_sha,
        "environment": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "render_backend": "EGL expected on headless GPU host",
            "video_encoder": "PyAV libx264",
            "pyav": __import__("av").__version__,
            "gpu": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True,
            ).stdout.strip(),
        },
        "claim_boundary": (
            "MuJoCo simulation evidence from 180 constructed roots. Rejected-branch motion is a labelled "
            "counterfactual. Static sampled checks do not establish continuous collision freedom. This is not "
            "real-hardware validation, functional-safety certification, or SmolVLA-controlled UR5e motion."
        ),
    }
    write_json(args.out / "video_manifest.json", manifest)
    print(json.dumps({"video": str(output_video), "poster": str(output_poster),
                      "manifest": str(args.out / 'video_manifest.json'),
                      "duration_seconds": manifest["duration_seconds"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
