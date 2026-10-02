#!/usr/bin/env python3
"""Export the pinned UR5e visual meshes and recorded replay poses for WebGL.

This performs only mesh preparation and MuJoCo forward kinematics on saved qpos.
It does not integrate dynamics, rerun a gate, or create collision labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_MODEL = "runs/arm-20261002/media-v4/model/portable_ur5e/sentinel_highres_review.xml"
DEFAULT_NPZ = "runs/arm-20261002/full-v3/review/highres_replay_trajectories.npz"
DEFAULT_REVIEWS = "runs/arm-20261002/full-v3/review/reviewed_per_root.json"
DEFAULT_MEDIA = "runs/arm-20261002/media-v4/media_manifest.json"
ROOT_IDS = (10000, 20000)
BODY_NAMES = (
    "base", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
)
JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    rows = []
    for item in sorted(x for x in path.rglob("*") if x.is_file() and not x.name.startswith(".sentinel_")):
        rows.append(f"{item.relative_to(path)}\0{file_sha256(item)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Missing model object: {name}")
    return value


def quat_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def export(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    model_path = resolve(repo_root, args.model)
    npz_path = resolve(repo_root, args.npz)
    review_path = resolve(repo_root, args.reviews)
    media_path = resolve(repo_root, args.media)
    model_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    body_ids = np.array([object_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_NAMES])
    joint_ids = np.array([object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
    qpos_addresses = model.jnt_qposadr[joint_ids]
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}

    meshes = []
    source_meshes = []
    source_vertex_count = 0
    viewer_vertex_count = 0
    triangle_count = 0
    degenerate_removed = 0
    for geom_id in range(model.ngeom):
        is_visual_mesh = (
            int(model.geom_group[geom_id]) == 2
            and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        )
        body_id = int(model.geom_bodyid[geom_id])
        if not is_visual_mesh or body_id not in body_to_index:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_address:vertex_address + vertex_count], dtype=np.float64
        )
        faces = np.asarray(
            model.mesh_face[face_address:face_address + face_count], dtype=np.int64
        )
        rotation = quat_matrix(model.geom_quat[geom_id])
        vertices = vertices @ rotation.T + np.asarray(model.geom_pos[geom_id], dtype=np.float64)

        unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces]
        keep = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        degenerate_removed += int(np.count_nonzero(~keep))
        faces = faces[keep]
        normals = np.zeros_like(unique_vertices)
        face_normals = np.cross(
            unique_vertices[faces[:, 1]] - unique_vertices[faces[:, 0]],
            unique_vertices[faces[:, 2]] - unique_vertices[faces[:, 0]],
        )
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        normals[lengths > 0] /= lengths[lengths > 0, None]

        material_id = int(model.geom_matid[geom_id])
        rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
        source_path = model_dir / "assets" / f"{mesh_name}.obj"
        source_meshes.append({
            "path": f"assets/{mesh_name}.obj",
            "bytes": source_path.stat().st_size,
            "sha256": file_sha256(source_path),
        })
        meshes.append({
            "name": mesh_name,
            "bodyIndex": body_to_index[body_id],
            "positions": np.round(unique_vertices, 6).reshape(-1).tolist(),
            "normals": np.round(normals, 5).reshape(-1).tolist(),
            "indices": faces.astype(np.int32).reshape(-1).tolist(),
            "color": np.round(np.asarray(rgba[:3]), 6).tolist(),
            "sourceVertexCount": vertex_count,
            "viewerVertexCount": len(unique_vertices),
            "triangleCount": len(faces),
        })
        source_vertex_count += vertex_count
        viewer_vertex_count += len(unique_vertices)
        triangle_count += len(faces)

    arrays = np.load(npz_path, allow_pickle=False)
    reviews = {row["root"]: row for row in json.loads(review_path.read_text())}
    selections = {row["root"]: row for row in json.loads(media_path.read_text())["selected"]}
    cases = []
    for root_id in ROOT_IDS:
        row = reviews[root_id]
        frames = []
        qpos = arrays[f"root_{root_id}_qpos"]
        for time_index, values in enumerate(qpos):
            data.qpos[qpos_addresses] = values
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            frames.append({
                "timeIndex": time_index,
                "timeSeconds": round((time_index + 1) * 0.05, 8),
                "qposRad": np.round(values, 8).tolist(),
                "bodyWorldPosition": np.round(data.xpos[body_ids], 8).tolist(),
                "bodyWorldRotation": np.round(data.xmat[body_ids], 8).reshape(len(body_ids), 9).tolist(),
            })
        cases.append({
            "rootId": root_id,
            "scenario": row["scenario"],
            "mediaSelection": selections[root_id],
            "obstacles": [
                {"name": "guard_obstacle_a", "type": "sphere", "radiusM": 0.055, "centerWorldM": row["final_obstacles"][0]},
                {"name": "guard_obstacle_b", "type": "sphere", "radiusM": 0.055, "centerWorldM": row["final_obstacles"][1]},
            ],
            "recordedOutcome": {
                "reviewUnsafe": row["review_unsafe"],
                "reviewReason": row["review_reason"],
                "gateDecisionsAllow": row["decisions"],
                "aggregateReplayObstacleCollision": row["highres_replay"]["obstacle"],
                "aggregateReplaySelfCollision": row["highres_replay"]["self_collision"],
                "perFrameCollisionLabelsAvailable": False,
            },
            "frames": frames,
        })

    asset = {
        "schema": "sentinel-ur5e-viewer-model-v2",
        "units": {"position": "meter", "jointAngle": "radian", "time": "second"},
        "bodyPositionOrder": list(BODY_NAMES),
        "meshCoordinates": "body-local; each visual geom local pose is preapplied",
        "meshes": meshes,
        "cases": cases,
        "provenance": {
            "exporter": {
                "name": Path(__file__).name,
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "portableModel": {
                "path": args.model,
                "sha256": file_sha256(model_path),
                "treeSha256": tree_sha256(model_dir),
                "mujocoVersion": mujoco.__version__,
            },
            "recordedQpos": {
                "path": args.npz,
                "sha256": file_sha256(npz_path),
                "keys": [f"root_{root_id}_qpos" for root_id in ROOT_IDS],
            },
            "reviewLabels": {"path": args.reviews, "sha256": file_sha256(review_path)},
            "mediaSelection": {"path": args.media, "sha256": file_sha256(media_path)},
            "license": {
                "spdx": "BSD-3-Clause",
                "path": "LICENSE",
                "sha256": file_sha256(model_dir / "LICENSE"),
            },
            "sourceMeshes": source_meshes,
            "geometryPreparation": {
                "allVisualMeshesIncluded": True,
                "allNondegenerateTrianglesRetained": True,
                "sourceCompiledVertexCount": source_vertex_count,
                "viewerExactPositionVertexCount": viewer_vertex_count,
                "triangleCount": triangle_count,
                "degenerateTrianglesRemoved": degenerate_removed,
                "positionWelding": "vertices with exactly equal compiled body-local XYZ are shared",
                "positionsRoundedDecimals": 6,
                "normals": "area-weighted smooth normals derived from retained triangles after exact-position welding",
                "normalsRoundedDecimals": 5,
            },
            "posePreparation": "Recorded qpos forwarded with mj_forward through the exact portable MJModel; no dynamics integration or gate evaluation.",
            "labelLimitations": [
                "The source review stores aggregate whole-trajectory collision flags, not collision timestamps or per-frame collision labels.",
                "No per-frame collision labels were inferred, recomputed, or fabricated.",
                "The two cases were selected after review from fixed constructed adversarial roots.",
            ],
        },
    }

    asset_path = output_dir / "ur5e-viewer-model.json"
    asset_path.write_text(json.dumps(asset, separators=(",", ":"), sort_keys=True) + "\n")
    license_path = output_dir / "LICENSE-UR5E-BSD-3-Clause.txt"
    shutil.copy2(model_dir / "LICENSE", license_path)
    manifest = {
        "schema": "sentinel-ur5e-viewer-assets-v2",
        "files": {
            asset_path.name: {"bytes": asset_path.stat().st_size, "sha256": file_sha256(asset_path)},
            license_path.name: {"bytes": license_path.stat().st_size, "sha256": file_sha256(license_path)},
            Path(__file__).name: {
                "bytes": Path(__file__).stat().st_size,
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Root containing the repo-relative source paths")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for viewer JSON, license and manifest")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Portable review MJCF path relative to --repo-root")
    parser.add_argument("--npz", default=DEFAULT_NPZ, help="Recorded qpos NPZ path relative to --repo-root")
    parser.add_argument("--reviews", default=DEFAULT_REVIEWS, help="Reviewed labels JSON path relative to --repo-root")
    parser.add_argument("--media", default=DEFAULT_MEDIA, help="Media selection manifest path relative to --repo-root")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
