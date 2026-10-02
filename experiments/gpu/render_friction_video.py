#!/usr/bin/env python3
"""Render a frozen low-friction MuJoCo failure/fallback comparison.

This is a render-only replay of stored qpos.  It does not rerun physics or use
evaluator-only friction, mass, or future labels as decision inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_evc.physics import PhysicsConfig, model_xml  # noqa: E402


WIDTH, HEIGHT = 1920, 1080
PANEL_WIDTH, PANEL_HEIGHT = 900, 625
FPS = 20
FRAME_DT = 0.05
RISK_LIMIT = 0.06
ROOT_ID = "low_friction-711000"
FAST_INDEX = 3
FALLBACK_INDEX = 5
INTRO_FRAMES = 38
HOLD_FRAMES = 18
OUTRO_FRAMES = 38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _font(candidates: Iterable[Path], size: int) -> ImageFont.ImageFont:
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def font_pack(explicit: Path | None) -> tuple[dict[str, ImageFont.ImageFont], bool, str]:
    if explicit is not None and not explicit.is_file():
        raise RuntimeError(f"font does not exist: {explicit}")
    cjk = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    latin = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    bold = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    prefix = (explicit,) if explicit else ()
    has_cjk = bool(prefix) or any(path.is_file() for path in cjk)
    regular_candidates = prefix + cjk + latin if has_cjk else latin
    bold_candidates = prefix + bold + cjk + latin
    fonts = {
        "hero": _font(bold_candidates, 48),
        "title": _font(bold_candidates, 34),
        "heading": _font(bold_candidates, 25),
        "body": _font(regular_candidates, 21),
        "small": _font(regular_candidates, 17),
        "tiny": _font(regular_candidates, 14),
    }
    selected = next((str(path) for path in regular_candidates if path.is_file()), "Pillow default")
    return fonts, has_cjk, selected


def bi(en: str, zh: str, enabled: bool) -> str:
    return f"{en}  |  {zh}" if enabled else en


def card(title: str, subtitle: str, lines: list[str], fonts: dict[str, ImageFont.ImageFont]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (5, 11, 22))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill=(61, 197, 143))
    draw.rounded_rectangle((135, 125, WIDTH - 135, HEIGHT - 125), radius=32,
                           fill=(11, 24, 43), outline=(48, 103, 135), width=3)
    draw.rounded_rectangle((180, 170, 580, 220), radius=16, fill=(19, 82, 79))
    draw.text((205, 183), "FROZEN MUJOCO EVIDENCE REPLAY", font=fonts["small"], fill=(184, 244, 221))
    draw.text((180, 284), title, font=fonts["hero"], fill=(244, 249, 255))
    draw.text((182, 360), subtitle, font=fonts["heading"], fill=(152, 205, 232))
    y = 476
    for line in lines:
        draw.ellipse((184, y + 7, 198, y + 21), fill=(61, 197, 143))
        draw.text((222, y), line, font=fonts["body"], fill=(215, 230, 243))
        y += 76
    draw.text((182, 865), "Stored qpos only · no simulation rerun · no robot arm or VLA attribution",
              font=fonts["small"], fill=(103, 143, 173))
    return image


class VideoWriter:
    def __init__(self, path: Path) -> None:
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=FPS)
        self.stream.width = WIDTH
        self.stream.height = HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}

    def write(self, image: Image.Image) -> None:
        frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB"), np.uint8), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def repeat(writer: VideoWriter, image: Image.Image, count: int) -> int:
    for _ in range(count):
        writer.write(image)
    return count


def reconstruct(model: mujoco.MjModel, trajectories: np.ndarray) -> dict[str, np.ndarray]:
    tray_id = model.body("tray").id
    payload_id = model.body("payload").id
    floor_id = model.geom("floor").id
    payload_geom_id = model.geom("payload_geom").id
    xy, dropped, tray_xyz, payload_xyz = [], [], [], []
    data = mujoco.MjData(model)
    for branch in trajectories:
        branch_xy, branch_drop, branch_tray, branch_payload = [], [], [], []
        for qpos in branch:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            tray = np.asarray(data.xpos[tray_id], dtype=np.float64).copy()
            payload = np.asarray(data.xpos[payload_id], dtype=np.float64).copy()
            relative = payload - tray
            floor_contact = any(
                {int(data.contact[i].geom[0]), int(data.contact[i].geom[1])} == {floor_id, payload_geom_id}
                for i in range(data.ncon)
            )
            branch_xy.append(float(np.linalg.norm(relative[:2])))
            branch_drop.append(bool(floor_contact or abs(relative[0]) > .14 or abs(relative[1]) > .12))
            branch_tray.append(tray)
            branch_payload.append(payload)
        xy.append(branch_xy)
        dropped.append(branch_drop)
        tray_xyz.append(branch_tray)
        payload_xyz.append(branch_payload)
    return {
        "xy": np.asarray(xy),
        "dropped": np.asarray(dropped),
        "tray_xyz": np.asarray(tray_xyz),
        "payload_xyz": np.asarray(payload_xyz),
    }


def load_inputs(run: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    manifest_path = run / "manifest.json"
    protocol_path = run / "protocol.json"
    per_root_path = run / "per_root.json"
    results_path = run / "results.npz"
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    rows = json.loads(per_root_path.read_text())
    if manifest["status"] != "complete" or protocol["status"] != "frozen":
        raise RuntimeError("intervention evidence is not complete and frozen")
    if sha256(results_path) != manifest["output_hashes"]["results.npz"]:
        raise RuntimeError("results.npz hash mismatch")
    if sha256(per_root_path) != manifest["output_hashes"]["per_root.json"]:
        raise RuntimeError("per_root.json hash mismatch")
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise RuntimeError("protocol hash mismatch")
    row = next((item for item in rows if item["root_id"] == ROOT_ID), None)
    if row is None:
        raise RuntimeError(f"fixed root {ROOT_ID} is absent")
    if row["choices"]["fixed_1p6_unbound"] != FAST_INDEX or row["choices"]["fixed_4p8_unbound"] != FALLBACK_INDEX:
        raise RuntimeError("fixed comparison choices changed")
    outcomes = row["outcomes_by_duration"]
    if not (outcomes[FAST_INDEX]["unsafe"] and outcomes[FAST_INDEX]["drop"]):
        raise RuntimeError("fixed fast branch is no longer the declared failure")
    if outcomes[FALLBACK_INDEX]["unsafe"] or outcomes[FALLBACK_INDEX]["drop"]:
        raise RuntimeError("fixed fallback branch is no longer the declared safe completion")
    with np.load(results_path, allow_pickle=False) as arrays:
        trajectories = np.asarray(arrays["low_friction_example_qpos"], dtype=np.float64)
        stored = {key: np.asarray(arrays[key]) for key in (
            "low_friction_truth", "low_friction_unsafe", "low_friction_drop",
            "low_friction_endpoint", "low_friction_integration_state",
        )}
    if trajectories.shape != (6, 110, 10):
        raise RuntimeError(f"unexpected stored qpos shape: {trajectories.shape}")
    hashes = {
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "per_root_sha256": sha256(per_root_path),
        "results_npz_sha256": sha256(results_path),
    }
    return manifest, row, trajectories, stored, hashes


def render_panel(renderer: mujoco.Renderer, model: mujoco.MjModel, qpos: np.ndarray) -> Image.Image:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray((0.19, 0.055, 0.48))
    camera.distance = 1.05
    camera.azimuth = 135
    camera.elevation = -23
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render())


def chart(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int, height: int,
          fast_xy: np.ndarray, safe_xy: np.ndarray, frame_index: int,
          fonts: dict[str, ImageFont.ImageFont], cjk: bool) -> None:
    y_max = max(.16, float(max(fast_xy.max(), safe_xy.max())) * 1.08)
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=14,
                           fill=(7, 17, 31), outline=(42, 72, 99), width=2)
    plot_x0, plot_x1 = x0 + 64, x0 + width - 24
    plot_y0, plot_y1 = y0 + 24, y0 + height - 35
    threshold_y = plot_y1 - int((RISK_LIMIT / y_max) * (plot_y1 - plot_y0))
    draw.line((plot_x0, threshold_y, plot_x1, threshold_y), fill=(255, 187, 69), width=2)
    draw.text((plot_x0 + 8, threshold_y - 23), bi("risk threshold 0.06 m", "风险阈值 0.06 米", cjk),
              font=fonts["tiny"], fill=(255, 205, 115))
    def points(values: np.ndarray) -> list[tuple[int, int]]:
        return [
            (plot_x0 + int(i / 109 * (plot_x1 - plot_x0)),
             plot_y1 - int(min(float(value), y_max) / y_max * (plot_y1 - plot_y0)))
            for i, value in enumerate(values[:frame_index + 1])
        ]
    fast_points, safe_points = points(fast_xy), points(safe_xy)
    if len(fast_points) > 1:
        draw.line(fast_points, fill=(255, 92, 78), width=4)
        draw.line(safe_points, fill=(75, 224, 154), width=4)
    cursor_x = plot_x0 + int(frame_index / 109 * (plot_x1 - plot_x0))
    draw.line((cursor_x, plot_y0, cursor_x, plot_y1), fill=(175, 205, 227), width=2)
    draw.text((plot_x0, plot_y1 + 8), "0.0 s", font=fonts["tiny"], fill=(134, 163, 188))
    draw.text((plot_x1 - 45, plot_y1 + 8), "5.5 s", font=fonts["tiny"], fill=(134, 163, 188))
    draw.text((x0 + 10, plot_y0), "XY\n(m)", font=fonts["tiny"], fill=(134, 163, 188))


def compose(left: Image.Image, right: Image.Image, frame_index: int,
            xy: np.ndarray, dropped: np.ndarray, row: dict[str, Any],
            fonts: dict[str, ImageFont.ImageFont], cjk: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (5, 11, 22))
    image.paste(left, (30, 142))
    image.paste(right, (990, 142))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 112), fill=(8, 18, 33))
    draw.rectangle((0, 108, WIDTH, 114), fill=(61, 197, 143))
    draw.text((32, 19), bi("Payload transport under low friction", "低摩擦条件下的载荷运输", cjk),
              font=fonts["title"], fill=(241, 248, 255))
    draw.text((32, 69), bi("Same initial state · actual stored MuJoCo qpos", "相同初始状态 · 实际存储的 MuJoCo qpos", cjk),
              font=fonts["small"], fill=(143, 184, 212))
    labels = (
        (30, (255, 91, 76), bi("FAST PLAN · 1.6 s", "快速方案 · 1.6 秒", cjk),
         bi("FAILURE: payload slips / drops", "失败：载荷滑移并掉落", cjk)),
        (990, (75, 224, 154), bi("FALLBACK · 4.8 s", "回退方案 · 4.8 秒", cjk),
         bi("COMPLETES: 3× commanded duration", "完成：指令时长为 3 倍", cjk)),
    )
    for x, color, title, subtitle in labels:
        draw.rounded_rectangle((x - 3, 139, x + PANEL_WIDTH + 3, 770), radius=12, outline=color, width=4)
        draw.rounded_rectangle((x + 18, 160, x + PANEL_WIDTH - 18, 230), radius=12,
                               fill=(7, 15, 27), outline=color, width=2)
        draw.text((x + 38, 171), title, font=fonts["heading"], fill=color)
        draw.text((x + 39, 204), subtitle, font=fonts["tiny"], fill=(201, 218, 231))
    fast_drop = bool(dropped[FAST_INDEX, frame_index])
    fast_risk = xy[FAST_INDEX, frame_index] > RISK_LIMIT or fast_drop
    if fast_risk:
        wording = bi("DROPPED CUBE", "方块已掉落", cjk) if fast_drop else bi("RISK LIMIT CROSSED", "超过风险阈值", cjk)
        draw.rounded_rectangle((304, 672, 655, 741), radius=16, fill=(78, 14, 18), outline=(255, 91, 76), width=3)
        draw.text((329, 690), wording, font=fonts["heading"], fill=(255, 143, 132))
    safe = xy[FALLBACK_INDEX, frame_index] <= RISK_LIMIT and not dropped[FALLBACK_INDEX, frame_index]
    if frame_index == 109 and safe:
        draw.rounded_rectangle((1264, 672, 1617, 741), radius=16, fill=(10, 64, 47), outline=(75, 224, 154), width=3)
        draw.text((1287, 690), bi("SAFE COMPLETION", "稳定完成", cjk),
                  font=fonts["heading"], fill=(137, 246, 199))
    chart(draw, 30, 795, 1860, 245, xy[FAST_INDEX], xy[FALLBACK_INDEX], frame_index, fonts, cjk)
    elapsed = frame_index * FRAME_DT
    exact_mu = row["evaluator_only"]["friction"]
    mass = row["evaluator_only"]["mass_kg"]
    draw.text((1120, 115), f"t = {elapsed:.2f} / 5.45 s", font=fonts["small"], fill=(177, 208, 230))
    draw.text((1400, 115), bi(
        f"Eval-only mu={exact_mu:.4f}, m={mass:.3f}kg",
        "评估标注", cjk),
        font=fonts["tiny"], fill=(255, 198, 109))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = ("friction-failure-vs-fallback.mp4", "friction-failure-vs-fallback-poster.png", "video_manifest.json")
    if any((args.out / name).exists() for name in names):
        raise RuntimeError("refusing to overwrite friction-video outputs")

    fonts, cjk, font_path = font_pack(args.font_path.resolve() if args.font_path else None)
    manifest, row, trajectories, stored, input_hashes = load_inputs(args.run.resolve())
    physics_source = ROOT / "src" / "sentinel_evc" / "physics.py"
    expected_source = json.loads((args.run / "protocol.json").read_text())["frozen_sources"]["src/sentinel_evc/physics.py"]
    if sha256(physics_source) != expected_source:
        raise RuntimeError("physics source differs from the frozen protocol")
    config = PhysicsConfig(
        timestep=.002,
        friction=float(row["evaluator_only"]["friction"]),
        payload_mass=float(row["evaluator_only"]["mass_kg"]),
        seed=int(row["evaluator_only"]["seed"]),
    )
    xml = model_xml(config)
    physics_model = mujoco.MjModel.from_xml_string(xml)
    visual_clause = '<visual><global offwidth="640" offheight="480"/></visual>'
    render_clause = f'<visual><global offwidth="{PANEL_WIDTH}" offheight="{PANEL_HEIGHT}"/></visual>'
    if xml.count(visual_clause) != 1:
        raise RuntimeError("unexpected visualization clause in frozen physics model XML")
    render_xml = xml.replace(visual_clause, render_clause)
    model = mujoco.MjModel.from_xml_string(render_xml)
    if physics_model.nq != trajectories.shape[-1] or model.nq != physics_model.nq:
        raise RuntimeError(
            f"model nq mismatch: physics={physics_model.nq}, render={model.nq}, stored={trajectories.shape[-1]}"
        )
    reconstructed = reconstruct(physics_model, trajectories)
    relative_truth = stored["low_friction_truth"][0, :, :, :3]
    max_truth_error = float(np.max(np.abs(
        reconstructed["payload_xyz"][:, :40] - reconstructed["tray_xyz"][:, :40] - relative_truth
    )))
    if max_truth_error > 2e-6:
        raise RuntimeError(f"stored qpos does not reconstruct frozen relative-position truth: {max_truth_error}")
    calculated_unsafe = (reconstructed["xy"].max(axis=1) > RISK_LIMIT) | reconstructed["dropped"].any(axis=1)
    if not np.array_equal(calculated_unsafe, stored["low_friction_unsafe"][0]):
        raise RuntimeError("qpos-derived risk labels disagree with frozen labels")
    if not np.array_equal(reconstructed["dropped"].any(axis=1), stored["low_friction_drop"][0]):
        raise RuntimeError("qpos-derived drop labels disagree with frozen labels")

    renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
    output_video = args.out / names[0]
    writer = VideoWriter(output_video)
    frame_count = 0
    poster: Image.Image | None = None
    drop_indices = np.flatnonzero(reconstructed["dropped"][FAST_INDEX])
    risk_indices = np.flatnonzero(reconstructed["xy"][FAST_INDEX] > RISK_LIMIT)
    poster_index = int(drop_indices[0] if len(drop_indices) else risk_indices[0])
    try:
        intro = card(
            bi("Why the obvious fast plan fails", "为什么明显更快的方案会失败", cjk),
            bi("Task: transport the payload without slip", "任务：运输载荷且不得滑移", cjk),
            [
                bi("Same frozen root and initial state", "相同的冻结根与初始状态", cjk),
                bi("1.6 s fixed plan: unsafe in 100/100 low-friction roots", "1.6 秒固定方案：100/100 个低摩擦根不安全", cjk),
                bi("4.8 s fallback: 100/100 completed; 3× duration tradeoff", "4.8 秒回退：100/100 完成；代价是 3 倍时长", cjk),
            ], fonts,
        )
        frame_count += repeat(writer, intro, INTRO_FRAMES)
        for frame_index in range(110):
            left = render_panel(renderer, model, trajectories[FAST_INDEX, frame_index])
            right = render_panel(renderer, model, trajectories[FALLBACK_INDEX, frame_index])
            frame = compose(left, right, frame_index, reconstructed["xy"], reconstructed["dropped"], row, fonts, cjk)
            writer.write(frame)
            frame_count += 1
            if frame_index == poster_index:
                poster = frame.copy()
        final = compose(
            render_panel(renderer, model, trajectories[FAST_INDEX, -1]),
            render_panel(renderer, model, trajectories[FALLBACK_INDEX, -1]),
            109, reconstructed["xy"], reconstructed["dropped"], row, fonts, cjk,
        )
        frame_count += repeat(writer, final, HOLD_FRAMES)
        outro = card(
            bi("Failure mechanism and intervention", "失败机制与干预结果", cjk),
            bi("Low friction is outside the original training support", "低摩擦超出原始训练支持范围", cjk),
            [
                bi("Exact friction and mass are evaluator-only labels", "精确摩擦系数和质量仅用于评估", cjk),
                bi("Decision uses declared mu floor 0.015, not hidden truth", "决策使用声明下限 0.015，不读取隐藏真值", cjk),
                bi("Observed result is scoped to 100 constructed MuJoCo roots", "观测结果仅适用于 100 个构造的 MuJoCo 根", cjk),
            ], fonts,
        )
        frame_count += repeat(writer, outro, OUTRO_FRAMES)
    finally:
        renderer.close()
        writer.close()
    if poster is None:
        raise RuntimeError("poster frame was not captured")
    output_poster = args.out / names[1]
    poster.save(output_poster)

    metrics = json.loads((args.run / "metrics.json").read_text())["low_friction"]["policies"]
    output_manifest = {
        "schema": "sentinel-friction-video-v1",
        "render_only": True,
        "physics_rerun": False,
        "root_selection": "first frozen low_friction test root by protocol seed order",
        "root_id": ROOT_ID,
        "branches": {"fast_index": FAST_INDEX, "fast_duration_s": 1.6,
                     "fallback_index": FALLBACK_INDEX, "fallback_duration_s": 4.8},
        "task": "transport payload without slip",
        "resolution": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "frame_count": frame_count,
        "duration_seconds": frame_count / FPS,
        "stored_motion_frames": 110,
        "stored_motion_seconds": 5.5,
        "input_hashes": input_hashes,
        "renderer_sha256": sha256(Path(__file__)),
        "model": {"nq": physics_model.nq, "nv": physics_model.nv, "nbody": physics_model.nbody,
                  "physics_xml_sha256": sha256_bytes(xml.encode()),
                  "render_xml_sha256": sha256_bytes(render_xml.encode()),
                  "render_xml_change": "visual.global offscreen framebuffer only; physics XML otherwise byte-identical",
                  "physics_source_sha256": sha256(physics_source)},
        "replay_verification": {
            "max_relative_position_vs_truth_error": max_truth_error,
            "qpos_derived_unsafe_matches_frozen_labels": True,
            "qpos_derived_drop_matches_frozen_labels": True,
            "fast_first_risk_frame": int(risk_indices[0]),
            "fast_first_drop_frame": int(drop_indices[0]),
            "poster_frame": poster_index,
        },
        "aggregate_context": {
            "fixed_1p6_unbound": metrics["fixed_1p6_unbound"],
            "fixed_4p8_unbound": metrics["fixed_4p8_unbound"],
        },
        "evaluator_only_annotations": row["evaluator_only"],
        "decision_input": row["router_input"],
        "outputs": {
            output_video.name: {"sha256": sha256(output_video), "bytes": output_video.stat().st_size},
            output_poster.name: {"sha256": sha256(output_poster), "bytes": output_poster.stat().st_size},
        },
        "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                        "numpy": np.__version__, "pyav": av.__version__, "encoder": "libx264",
                        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                              capture_output=True, text=True).stdout.strip()},
        "claim_boundary": (
            "Offline MuJoCo tray/payload replay from a frozen constructed root. Exact friction and mass are "
            "evaluator-only annotations, not decision inputs. No robot arm, VLA, hardware, or general safety claim."
        ),
    }
    write_json(args.out / names[2], output_manifest)
    print(json.dumps({"video": str(output_video), "poster": str(output_poster),
                      "manifest": str(args.out / names[2]), "replay": output_manifest["replay_verification"]}, indent=2))


if __name__ == "__main__":
    main()
