#!/usr/bin/env python3
"""Export a compiled LIBERO MuJoCo scene and same-process body poses for WebGL.

The exporter reads geometry from the live ``MjModel`` and poses from its paired
``MjData``.  It never substitutes proxy robot links for visual meshes and it
does not advance physics.  A runner may keep one :class:`NativeSceneExporter`
beside its environment and call :meth:`capture_frame` after each real step.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np


SCENE_SCHEMA = "sentinel-native-mujoco-scene-v1"
FRAME_SCHEMA = "sentinel-native-mujoco-frame-v1"


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed_{kind.name.lower()}_{index}"


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _transform(
    positions: np.ndarray,
    normals: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = _quat_matrix(quaternion)
    return (
        np.asarray(positions, dtype=np.float64) @ rotation.T + np.asarray(position, dtype=np.float64),
        np.asarray(normals, dtype=np.float64) @ rotation.T,
    )


def _flat_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return triangle-expanded positions with exact face normals."""
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 0.0, 1.0)
    positions = triangles.reshape(-1, 3)
    expanded_normals = np.repeat(normals, 3, axis=0)
    indices = np.arange(len(positions), dtype=np.int32).reshape(-1, 3)
    return positions, expanded_normals, indices


def _smooth_indexed(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 1e-15] /= lengths[lengths > 1e-15, None]
    normals[lengths <= 1e-15] = (0.0, 0.0, 1.0)
    return vertices, normals, faces


def _box(size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = np.asarray(size[:3], dtype=np.float64)
    vertices = np.array(
        [[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
         [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.int32,
    )
    return _flat_faces(vertices, faces)


def _uv_sphere(radii: np.ndarray, latitude: int = 16, longitude: int = 24):
    rx, ry, rz = np.asarray(radii[:3], dtype=np.float64)
    vertices = [[0.0, 0.0, rz]]
    for row in range(1, latitude):
        phi = np.pi * row / latitude
        for column in range(longitude):
            theta = 2.0 * np.pi * column / longitude
            vertices.append([rx * np.sin(phi) * np.cos(theta), ry * np.sin(phi) * np.sin(theta), rz * np.cos(phi)])
    vertices.append([0.0, 0.0, -rz])
    south = len(vertices) - 1
    faces: list[list[int]] = []
    for column in range(longitude):
        faces.append([0, 1 + column, 1 + (column + 1) % longitude])
    for row in range(latitude - 2):
        start = 1 + row * longitude
        next_start = start + longitude
        for column in range(longitude):
            nxt = (column + 1) % longitude
            faces.extend([[start + column, next_start + column, next_start + nxt],
                          [start + column, next_start + nxt, start + nxt]])
    last = 1 + (latitude - 2) * longitude
    for column in range(longitude):
        faces.append([last + column, south, last + (column + 1) % longitude])
    return _smooth_indexed(np.asarray(vertices), np.asarray(faces))


def _cylinder(radius: float, half_height: float, segments: int = 32):
    vertices: list[list[float]] = []
    for z in (-half_height, half_height):
        vertices.extend([[radius * np.cos(2 * np.pi * i / segments),
                          radius * np.sin(2 * np.pi * i / segments), z] for i in range(segments)])
    vertices.extend([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]])
    bottom, top = 2 * segments, 2 * segments + 1
    faces: list[list[int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([[i, j, segments + j], [i, segments + j, segments + i],
                      [bottom, j, i], [top, segments + i, segments + j]])
    return _smooth_indexed(np.asarray(vertices), np.asarray(faces))


def _capsule(radius: float, half_height: float, segments: int = 32, hemi_rows: int = 8):
    rings: list[tuple[float, float]] = []
    for row in range(hemi_rows + 1):
        angle = -np.pi / 2 + row * (np.pi / 2) / hemi_rows
        rings.append((radius * np.cos(angle), -half_height + radius * np.sin(angle)))
    for row in range(1, hemi_rows + 1):
        angle = row * (np.pi / 2) / hemi_rows
        rings.append((radius * np.cos(angle), half_height + radius * np.sin(angle)))
    vertices = [[ring_radius * np.cos(2 * np.pi * i / segments),
                 ring_radius * np.sin(2 * np.pi * i / segments), z]
                for ring_radius, z in rings for i in range(segments)]
    faces: list[list[int]] = []
    for row in range(len(rings) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = row * segments + i, row * segments + j
            c, d = (row + 1) * segments + i, (row + 1) * segments + j
            faces.extend([[a, c, d], [a, d, b]])
    return _smooth_indexed(np.asarray(vertices), np.asarray(faces))


def _plane(size: np.ndarray, fallback_extent: float):
    x = float(size[0]) if float(size[0]) > 0 else fallback_extent
    y = float(size[1]) if float(size[1]) > 0 else fallback_extent
    vertices = np.array([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return _flat_faces(vertices, faces)


def _compiled_mesh(model: mujoco.MjModel, mesh_id: int):
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_address = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vertex_address:vertex_address + vertex_count], dtype=np.float64)
    faces = np.asarray(model.mesh_face[face_address:face_address + face_count], dtype=np.int32)
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise RuntimeError(f"compiled mesh {mesh_id} has out-of-range face indices")

    # MuJoCo's compiler already applies mesh_scale to mesh_vert and places its
    # centering transform in the referencing geom pose.  Applying mesh_scale,
    # mesh_pos or mesh_quat again here would double-transform the live model.
    # Normals are recomputed after the referencing geom's local transform and
    # exact-position welding.  Positions and the complete source tessellation
    # remain unchanged; only shading normals at asset seams are regenerated.
    return vertices, np.zeros_like(vertices), faces


def _weld_exact_positions(
    positions: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    unique, inverse = np.unique(np.asarray(positions, dtype=np.float64), axis=0, return_inverse=True)
    remapped = inverse[np.asarray(faces, dtype=np.int64)]
    keep = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    removed = int(np.count_nonzero(~keep))
    vertices, normals, indices = _smooth_indexed(unique, remapped[keep])
    return vertices, normals, indices, removed


def _geom_geometry(model: mujoco.MjModel, geom_id: int, plane_extent: float):
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        return _compiled_mesh(model, int(model.geom_dataid[geom_id]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return _box(size)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return _uv_sphere(np.repeat(size[0], 3))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return _uv_sphere(size)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return _cylinder(float(size[0]), float(size[1]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return _capsule(float(size[0]), float(size[1]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return _plane(size, plane_extent)
    raise ValueError(f"unsupported compiled geom type {geom_type}")


def unwrap_model_data(environment: Any) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Find the paired raw MuJoCo model/data without creating or stepping an env."""
    queue = [environment]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        sim = getattr(candidate, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
        for attribute in ("_env", "env", "unwrapped"):
            nested = getattr(candidate, attribute, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
        envs = getattr(candidate, "envs", None)
        if envs:
            queue.extend(list(envs))
    raise RuntimeError("could not find a live MuJoCo sim.model/sim.data pair")


def _end_effector_reference(environment: Any, model: mujoco.MjModel) -> dict[str, Any] | None:
    """Resolve the robot-declared EEF site to its actual compiled parent body."""
    queue = [environment]
    visited: set[int] = set()
    site_id: int | None = None
    source = None
    while queue and site_id is None:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        robots = getattr(candidate, "robots", None)
        if robots:
            declared = getattr(robots[0], "eef_site_id", None)
            if declared is not None:
                if isinstance(declared, dict):
                    declared = next(iter(declared.values()))
                if isinstance(declared, (list, tuple, np.ndarray)):
                    declared = np.asarray(declared).reshape(-1)[0]
                site_id = int(declared)
                source = "environment.robots[0].eef_site_id"
                break
        for attribute in ("_env", "env", "unwrapped"):
            nested = getattr(candidate, attribute, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
        envs = getattr(candidate, "envs", None)
        if envs:
            queue.extend(list(envs))

    if site_id is None:
        named = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if named >= 0:
            site_id = int(named)
            source = "compiled_site_name:gripper0_grip_site"
    if site_id is None or not 0 <= site_id < model.nsite:
        return None
    body_id = int(model.site_bodyid[site_id])
    return {
        "source": source,
        "siteIndex": site_id,
        "siteName": _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
        "bodyIndex": body_id,
        "bodyName": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "siteLocalPosition": np.asarray(model.site_pos[site_id], dtype=np.float64).tolist(),
        "siteLocalRotation": _quat_matrix(model.site_quat[site_id]).reshape(-1).tolist(),
        "viewerBodyReferencePoint": "body_origin",
    }


def _model_id(model: mujoco.MjModel, task_suite: str, task_id: int) -> str:
    digest = hashlib.sha256(f"{task_suite}\0{task_id}\0{model.nbody}\0{model.ngeom}\0{model.nmesh}\n".encode())
    arrays = (
        model.body_parentid, model.geom_type, model.geom_bodyid, model.geom_group,
        model.geom_dataid, model.geom_matid, model.geom_pos, model.geom_quat,
        model.geom_size, model.geom_rgba, model.mesh_vertadr, model.mesh_vertnum,
        model.mesh_faceadr, model.mesh_facenum, model.mesh_vert, model.mesh_face,
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    for body_id in range(model.nbody):
        digest.update((_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) + "\0").encode())
    for geom_id in range(model.ngeom):
        digest.update((_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) + "\0").encode())
    return digest.hexdigest()


def _rgba(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        return np.asarray(model.mat_rgba[material_id], dtype=np.float64)
    return np.asarray(model.geom_rgba[geom_id], dtype=np.float64)


@dataclass
class NativeSceneExporter:
    model: mujoco.MjModel
    data: mujoco.MjData
    task_id: int
    task_suite: str = "libero_spatial"
    task_name: str | None = None
    geom_groups: tuple[int, ...] = (1,)
    include_transparent: bool = False
    plane_extent: float = 2.0
    end_effector_reference: dict[str, Any] | None = None
    _cached_model_id: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_environment(cls, environment: Any, task_id: int, **kwargs: Any) -> "NativeSceneExporter":
        model, data = unwrap_model_data(environment)
        kwargs.setdefault("end_effector_reference", _end_effector_reference(environment, model))
        return cls(model=model, data=data, task_id=task_id, **kwargs)

    @property
    def model_id(self) -> str:
        if self._cached_model_id is None:
            self._cached_model_id = _model_id(self.model, self.task_suite, self.task_id)
        return self._cached_model_id

    def export_scene(self) -> dict[str, Any]:
        model = self.model
        selected_groups = {int(value) for value in self.geom_groups}
        meshes: list[dict[str, Any]] = []
        skipped_transparent: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        included_by_group: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        source_vertex_count = 0
        viewer_vertex_count = 0
        source_triangle_count = 0
        viewer_triangle_count = 0
        degenerate_triangles_removed = 0
        for geom_id in range(model.ngeom):
            group = int(model.geom_group[geom_id])
            if group not in selected_groups:
                continue
            rgba = _rgba(model, geom_id)
            geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if rgba[3] <= 0 and not self.include_transparent:
                skipped_transparent.append({"geomIndex": geom_id, "geomName": geom_name, "group": group})
                continue
            geom_type_id = int(model.geom_type[geom_id])
            geom_type = mujoco.mjtGeom(geom_type_id).name.removeprefix("mjGEOM_").lower()
            try:
                positions, normals, indices = _geom_geometry(model, geom_id, self.plane_extent)
            except ValueError:
                unsupported.append({"geomIndex": geom_id, "geomName": geom_name, "geomType": geom_type})
                continue
            positions, normals = _transform(
                positions, normals, model.geom_pos[geom_id], model.geom_quat[geom_id]
            )
            source_vertices = int(len(positions))
            source_triangles = int(len(indices))
            removed = 0
            if geom_type == "mesh":
                positions, normals, indices, removed = _weld_exact_positions(positions, indices)
            body_id = int(model.geom_bodyid[geom_id])
            mesh_id = int(model.geom_dataid[geom_id]) if geom_type_id == int(mujoco.mjtGeom.mjGEOM_MESH) else None
            material_id = int(model.geom_matid[geom_id])
            meshes.append({
                "name": geom_name,
                "bodyIndex": body_id,
                "positions": positions.astype(np.float32).reshape(-1).tolist(),
                "normals": normals.astype(np.float32).reshape(-1).tolist(),
                "indices": indices.astype(np.int32).reshape(-1).tolist(),
                "color": rgba[:3].astype(np.float32).tolist(),
                "opacity": float(rgba[3]),
                "geomIndex": geom_id,
                "geomType": geom_type,
                "geomGroup": group,
                "geomLocalPosition": np.asarray(model.geom_pos[geom_id], dtype=np.float64).tolist(),
                "geomLocalRotation": _quat_matrix(model.geom_quat[geom_id]).reshape(-1).tolist(),
                "materialId": material_id,
                "textureIds": np.asarray(model.mat_texid[material_id], dtype=np.int32).reshape(-1).tolist()
                if material_id >= 0 else [],
                "meshId": mesh_id,
                "meshName": _name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id is not None else None,
                "sourceVertexCount": source_vertices,
                "viewerVertexCount": int(len(positions)),
                "sourceTriangleCount": source_triangles,
                "triangleCount": int(len(indices)),
                "degenerateTrianglesRemoved": removed,
                "infinitePlaneRenderedFinite": geom_type == "plane",
            })
            included_by_group[str(group)] = included_by_group.get(str(group), 0) + 1
            type_counts[geom_type] = type_counts.get(geom_type, 0) + 1
            source_vertex_count += source_vertices
            viewer_vertex_count += int(len(positions))
            source_triangle_count += source_triangles
            viewer_triangle_count += int(len(indices))
            degenerate_triangles_removed += removed

        bodies = [{
            "bodyIndex": body_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "parentBodyIndex": int(model.body_parentid[body_id]),
        } for body_id in range(model.nbody)]
        return {
            "schema": SCENE_SCHEMA,
            "modelId": self.model_id,
            "task": {"suite": self.task_suite, "taskId": self.task_id, "name": self.task_name},
            "endEffectorBodyIndex": (
                self.end_effector_reference["bodyIndex"] if self.end_effector_reference else None
            ),
            "endEffectorSiteIndex": (
                self.end_effector_reference["siteIndex"] if self.end_effector_reference else None
            ),
            "units": {"position": "meter", "rotation": "row-major 3x3 matrix"},
            "meshCoordinates": "body-local; exact compiled geom local pose is preapplied",
            "bodies": bodies,
            "meshes": meshes,
            "provenance": {
                "source": "live compiled MuJoCo MjModel",
                "mujocoVersion": mujoco.__version__,
                "compiledModel": {
                    "nbody": int(model.nbody), "ngeom": int(model.ngeom), "nmesh": int(model.nmesh),
                    "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
                },
                "endEffectorReference": self.end_effector_reference,
                "geometrySelection": {
                    "requestedGroups": sorted(selected_groups),
                    "includedByGroup": included_by_group,
                    "includedTypeCounts": type_counts,
                    "includeTransparent": self.include_transparent,
                    "skippedTransparent": skipped_transparent,
                    "unsupported": unsupported,
                    "visualGeometryPreferred": selected_groups == {1},
                    "compiledMeshTransform": "mesh_vert is already scale-compiled; geom_pos/geom_quat are applied once",
                    "positionPreparation": "exactly equal body-local XYZ values are welded; no spatial tolerance and no decimation",
                    "normalPreparation": "area-weighted viewer normals are recomputed after exact-position welding",
                    "sourceCompiledVertexCount": source_vertex_count,
                    "viewerExactPositionVertexCount": viewer_vertex_count,
                    "sourceTriangleCount": source_triangle_count,
                    "viewerTriangleCount": viewer_triangle_count,
                    "degenerateTrianglesRemoved": degenerate_triangles_removed,
                    "allNondegenerateSourceTrianglesRetained": (
                        viewer_triangle_count + degenerate_triangles_removed == source_triangle_count
                    ),
                    "collisionProxySubstitution": False,
                    "infinitePlaneRepresentation": f"finite {self.plane_extent:g} m half-extent only when compiled plane size is zero",
                },
            },
        }

    def capture_frame(self, frame_index: int, *, step: int | None = None) -> dict[str, Any]:
        model, data = self.model, self.data
        eef_site_id = (
            int(self.end_effector_reference["siteIndex"])
            if self.end_effector_reference is not None else None
        )
        contacts = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            contacts.append({
                "contactIndex": contact_index,
                "geom1Index": first,
                "geom1Name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "geom2Index": second,
                "geom2Name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "distance": float(contact.dist),
                "position": np.asarray(contact.pos, dtype=np.float64).tolist(),
            })
        return {
            "schema": FRAME_SCHEMA,
            "modelId": self.model_id,
            "task": {"suite": self.task_suite, "taskId": self.task_id},
            "frameIndex": int(frame_index),
            "step": int(step) if step is not None else None,
            "simulationTime": float(data.time),
            "bodyWorldPosition": np.asarray(data.xpos, dtype=np.float64).tolist(),
            "bodyWorldRotation": np.asarray(data.xmat, dtype=np.float64).reshape(model.nbody, 9).tolist(),
            "endEffectorBodyIndex": (
                int(self.end_effector_reference["bodyIndex"])
                if self.end_effector_reference is not None else None
            ),
            "endEffectorSiteIndex": eef_site_id,
            "endEffectorSiteWorldPosition": (
                np.asarray(data.site_xpos[eef_site_id], dtype=np.float64).tolist()
                if eef_site_id is not None else None
            ),
            "endEffectorSiteWorldRotation": (
                np.asarray(data.site_xmat[eef_site_id], dtype=np.float64).reshape(9).tolist()
                if eef_site_id is not None else None
            ),
            "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
            "contacts": contacts,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    values = {"mujoco": mujoco.__version__}
    for package in ("lerobot", "hf-libero", "robosuite", "numpy"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "unavailable"
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--geom-group", type=int, action="append", dest="geom_groups")
    parser.add_argument("--include-transparent", action="store_true")
    parser.add_argument("--plane-extent", type=float, default=2.0)
    parser.add_argument("--gzip", action="store_true", help="Write scene.json.gz instead of scene.json")
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    from lerobot.envs import make_env
    from lerobot.envs.configs import LiberoEnv

    if args.task_suite != "libero_spatial":
        raise ValueError("this entrypoint currently supports the pinned libero_spatial suite")
    groups = tuple(args.geom_groups if args.geom_groups is not None else [1])
    config = LiberoEnv(
        task=args.task_suite, task_ids=[args.task_id], fps=20, init_states=True,
        hard_reset=True, control_mode="relative", max_parallel_tasks=1,
        observation_height=360, observation_width=360,
    )
    envs = make_env(config, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs[args.task_suite][args.task_id]
    underlying = env.envs[0]
    try:
        underlying.init_state_id = args.initial_state_index
        env.reset(seed=[args.seed], options={"new_rollout": True})
        task_name = str(env.call("task_description")[0])
        exporter = NativeSceneExporter.from_environment(
            env, args.task_id, task_suite=args.task_suite, task_name=task_name,
            geom_groups=groups, include_transparent=args.include_transparent,
            plane_extent=args.plane_extent,
        )
        scene_name = "scene.json.gz" if args.gzip else "scene.json"
        scene_path = args.output_dir / scene_name
        frame_path = args.output_dir / "frame-000000.json"
        _write_json(scene_path, exporter.export_scene())
        _write_json(frame_path, exporter.capture_frame(0, step=0))
        manifest = {
            "schema": "sentinel-native-mujoco-export-v1",
            "status": "complete",
            "modelId": exporter.model_id,
            "task": {"suite": args.task_suite, "taskId": args.task_id, "name": task_name,
                     "initialStateIndex": args.initial_state_index, "seed": args.seed},
            "software": _versions(),
            "files": {
                scene_name: {"bytes": scene_path.stat().st_size, "sha256": _sha256(scene_path)},
                frame_path.name: {"bytes": frame_path.stat().st_size, "sha256": _sha256(frame_path)},
            },
        }
        _write_json(args.output_dir / "manifest.json", manifest)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    _main(_parse_args())
