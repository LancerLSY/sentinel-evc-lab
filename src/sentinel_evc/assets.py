"""Bounded, local-only 3D asset import, preview, and optional MuJoCo inspection."""
from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .contracts import canonical_json


MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_VERTICES = 200_000
MAX_TRIANGLES = 300_000
MAX_XML_ELEMENTS = 2_000
MAX_MODEL_GEOMS = 1_000
MAX_MODEL_JOINTS = 128
MAX_MODEL_ACTUATORS = 128
MAX_DIMENSION = 1_000.0
MAX_POSITION = 10_000.0
ASSET_ID = re.compile(r"^asset-[0-9a-f]{32}$")
FORMATS = frozenset({"obj", "stl", "mjcf", "urdf"})
_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_FORBIDDEN_XML = re.compile(br"<!\s*(?:DOCTYPE|ENTITY)|<\?", re.IGNORECASE)


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("symlink storage refused")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finite(values, *, count=None, label="vector") -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if (count is not None and len(result) != count) or any(not math.isfinite(value) for value in result):
        raise ValueError(f"invalid {label}")
    return result


def _bounded_magnitude(values: list[float], label: str, limit: float) -> list[float]:
    if any(abs(value) > limit for value in values):
        raise ValueError(f"{label} exceeds magnitude limit")
    return values


def _bounds(vertices: list[list[float]]) -> dict:
    if not vertices:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": [min(vertex[axis] for vertex in vertices) for axis in range(3)],
        "max": [max(vertex[axis] for vertex in vertices) for axis in range(3)],
    }


def _mesh_geometry(vertices: list[list[float]], triangles: list[list[int]], source: str) -> dict:
    if not vertices or not triangles:
        raise ValueError(f"{source.upper()} contains no triangular geometry")
    if len(vertices) > MAX_VERTICES or len(triangles) > MAX_TRIANGLES:
        raise ValueError("mesh exceeds geometry limits")
    return {
        "schema": "sentinel-geometry-v1",
        "kind": "mesh",
        "source": source,
        "vertices": vertices,
        "triangles": triangles,
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "bounds": _bounds(vertices),
        "physics_authorized": False,
    }


def _parse_obj(data: bytes) -> dict:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("OBJ must be UTF-8 text") from exc
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(f"OBJ vertex on line {line_number} must have three coordinates")
            vertices.append(_finite(fields[1:], count=3, label="OBJ vertex"))
            if len(vertices) > MAX_VERTICES:
                raise ValueError("OBJ exceeds vertex limit")
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"OBJ face on line {line_number} has fewer than three vertices")
            face = []
            for token in fields[1:]:
                head = token.split("/", 1)[0]
                try:
                    index = int(head)
                except ValueError as exc:
                    raise ValueError(f"invalid OBJ face on line {line_number}") from exc
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"OBJ face on line {line_number} references a missing vertex")
                face.append(index)
            for offset in range(1, len(face) - 1):
                triangles.append([face[0], face[offset], face[offset + 1]])
                if len(triangles) > MAX_TRIANGLES:
                    raise ValueError("OBJ exceeds triangle limit")
    return _mesh_geometry(vertices, triangles, "obj")


def _parse_stl(data: bytes) -> dict:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count <= MAX_TRIANGLES and len(data) == 84 + count * 50:
            for index in range(count):
                offset = 84 + index * 50 + 12
                triangle = []
                for vertex_index in range(3):
                    vertex = _finite(
                        struct.unpack_from("<fff", data, offset + vertex_index * 12),
                        count=3,
                        label="binary STL vertex",
                    )
                    triangle.append(len(vertices))
                    vertices.append(vertex)
                triangles.append(triangle)
            return _mesh_geometry(vertices, triangles, "stl-binary")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("STL is neither bounded binary STL nor ASCII STL") from exc
    for raw in text.splitlines():
        fields = raw.strip().split()
        if fields and fields[0].lower() == "vertex":
            if len(fields) != 4:
                raise ValueError("ASCII STL vertex must have three coordinates")
            vertices.append(_finite(fields[1:], count=3, label="ASCII STL vertex"))
            if len(vertices) > MAX_TRIANGLES * 3:
                raise ValueError("STL exceeds triangle limit")
    if len(vertices) % 3:
        raise ValueError("ASCII STL has an incomplete triangle")
    triangles = [[index, index + 1, index + 2] for index in range(0, len(vertices), 3)]
    return _mesh_geometry(vertices, triangles, "stl-ascii")


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].lower()


def _parse_safe_xml(data: bytes, format_name: str) -> ET.Element:
    if _FORBIDDEN_XML.search(data) or b"\x00" in data:
        raise ValueError("XML declarations, DTDs, and entities are not accepted")
    try:
        root = ET.fromstring(data)
    except (ET.ParseError, UnicodeDecodeError) as exc:
        raise ValueError("invalid XML model") from exc
    expected = "mujoco" if format_name == "mjcf" else "robot"
    if _local_name(root.tag) != expected:
        raise ValueError(f"{format_name.upper()} root must be <{expected}>")
    elements = list(root.iter())
    if len(elements) > MAX_XML_ELEMENTS:
        raise ValueError("XML model exceeds element limit")
    forbidden_elements = {
        "include", "plugin", "extension", "attach", "replicate", "composite", "flexcomp",
        "flex", "mesh", "hfield", "skin", "texture", "size",
    }
    forbidden_attributes = {"file", "filename", "meshdir", "texturedir", "assetdir"}
    allocation_attributes = {"memory", "nstack", "njmax", "nconmax"}
    for element in elements:
        tag = _local_name(element.tag)
        if tag in forbidden_elements:
            raise ValueError("external, expandable, and resource-allocation model features are not supported")
        for key, value in element.attrib.items():
            attribute = _local_name(key)
            lowered = value.strip().lower()
            if attribute in forbidden_attributes and lowered:
                raise ValueError("external model assets are not supported")
            if attribute in allocation_attributes or attribute.startswith("nuser_"):
                raise ValueError("explicit model resource allocation is not supported")
            if "://" in lowered or lowered.startswith(("file:", "package:")):
                raise ValueError("external model references are not supported")
            if attribute in {"size", "radius", "length"}:
                _positive(_finite(value.split(), label=f"XML {attribute}"), f"XML {attribute}")
            elif attribute in {"pos", "xyz", "fromto"}:
                _bounded_magnitude(
                    _finite(value.split(), label=f"XML {attribute}"),
                    f"XML {attribute}", MAX_POSITION,
                )
        if tag == "option":
            for attribute in ("iterations", "ls_iterations", "noslip_iterations", "ccd_iterations"):
                if attribute in element.attrib:
                    try:
                        value = int(element.attrib[attribute])
                    except ValueError as exc:
                        raise ValueError("invalid solver iteration limit") from exc
                    if not 0 <= value <= 1_000:
                        raise ValueError("solver iteration limit exceeds inspection bounds")
    if format_name == "mjcf":
        geom_count = sum(_local_name(element.tag) == "geom" for element in elements)
        joint_count = sum(_local_name(element.tag) in {"joint", "freejoint"} for element in elements)
        actuator_count = sum(
            len(list(element)) for element in elements if _local_name(element.tag) == "actuator"
        )
    else:
        geom_count = sum(_local_name(element.tag) in {"box", "sphere", "cylinder"} for element in elements)
        joint_count = sum(_local_name(element.tag) == "joint" for element in elements)
        actuator_count = 0
    if geom_count > MAX_MODEL_GEOMS or joint_count > MAX_MODEL_JOINTS or actuator_count > MAX_MODEL_ACTUATORS:
        raise ValueError("model structure exceeds inspection limits")
    return root


def _identity_rotation() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _rpy_rotation(values) -> list[float]:
    roll, pitch, yaw = _finite(values, count=3, label="URDF origin rotation")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ]


def _positive(values: list[float], label: str) -> list[float]:
    if any(value <= 0.0 or value > MAX_DIMENSION for value in values):
        raise ValueError(f"{label} must be positive and within dimension limits")
    return values


def _xml_geometry(root: ET.Element, format_name: str) -> dict:
    primitives = []
    unsupported = []
    for element in root.iter():
        tag = _local_name(element.tag)
        if format_name == "mjcf" and tag == "geom":
            kind = element.attrib.get("type", "sphere").lower()
            size = _finite(element.attrib.get("size", "").split(), label="MJCF geom size")
            position = _bounded_magnitude(
                _finite(element.attrib.get("pos", "0 0 0").split(), count=3, label="MJCF geom position"),
                "MJCF geom position", MAX_POSITION,
            )
            if kind == "box" and len(size) >= 3:
                dimensions = _positive([2 * value for value in size[:3]], "MJCF box size")
            elif kind == "sphere" and len(size) >= 1:
                dimensions = _positive([size[0]], "MJCF sphere size")
            elif kind in {"cylinder", "capsule"} and len(size) >= 2:
                dimensions = _positive([size[0], 2 * size[1]], "MJCF cylinder size")
            else:
                unsupported.append(kind)
                continue
            primitives.append({"type": kind, "dimensions": dimensions, "position": position,
                               "rotation": _identity_rotation(), "frame": "declared_local"})
        elif format_name == "urdf" and tag in {"visual", "collision"}:
            origin = next((child for child in element if _local_name(child.tag) == "origin"), None)
            position = _bounded_magnitude(
                _finite((origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split(),
                        count=3, label="URDF origin"),
                "URDF origin", MAX_POSITION,
            )
            rotation = _rpy_rotation((origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split())
            geometry = next((child for child in element if _local_name(child.tag) == "geometry"), None)
            shape = next(iter(geometry), None) if geometry is not None else None
            if shape is None:
                continue
            kind = _local_name(shape.tag)
            if kind == "box":
                dimensions = _positive(_finite(shape.attrib.get("size", "").split(), count=3, label="URDF box size"), "URDF box size")
            elif kind == "sphere":
                dimensions = _positive(_finite([shape.attrib.get("radius")], count=1, label="URDF sphere radius"), "URDF sphere radius")
            elif kind == "cylinder":
                dimensions = _positive(_finite([shape.attrib.get("radius"), shape.attrib.get("length")], count=2,
                                                label="URDF cylinder"), "URDF cylinder")
            else:
                unsupported.append(kind)
                continue
            primitives.append({"type": kind, "dimensions": dimensions, "position": position,
                               "rotation": rotation, "frame": "declared_link_local",
                               "role": tag})
    if len(primitives) > MAX_TRIANGLES:
        raise ValueError("model exceeds primitive limit")
    return {
        "schema": "sentinel-geometry-v1",
        "kind": "primitives",
        "source": format_name,
        "primitives": primitives,
        "primitive_count": len(primitives),
        "unsupported_types": sorted(set(unsupported)),
        "transform_scope": "declared local transforms; run MuJoCo check for compiled world transforms",
        "physics_authorized": False,
    }


def _compiled_geometry(mujoco, model, data) -> dict:
    supported = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    }
    primitives = []
    unsupported = []
    for index in range(int(model.ngeom)):
        kind_id = int(model.geom_type[index])
        kind = supported.get(kind_id)
        if kind is None:
            unsupported.append(str(kind_id))
            continue
        size = _bounded_magnitude(
            _finite(model.geom_size[index], count=3, label="compiled geom size"),
            "compiled geom size", MAX_DIMENSION,
        )
        if kind == "box":
            dimensions = [2 * value for value in size]
        elif kind == "sphere":
            dimensions = [size[0]]
        else:
            dimensions = [size[0], 2 * size[1]]
        primitives.append({
            "type": kind,
            "dimensions": dimensions,
            "position": _bounded_magnitude(
                _finite(data.geom_xpos[index], count=3, label="compiled geom position"),
                "compiled geom position", MAX_POSITION,
            ),
            "rotation": _finite(data.geom_xmat[index], count=9, label="compiled geom rotation"),
            "frame": "compiled_world",
        })
    return {
        "schema": "sentinel-geometry-v1",
        "kind": "primitives",
        "source": "mujoco-compiled",
        "primitives": primitives,
        "primitive_count": len(primitives),
        "unsupported_types": sorted(set(unsupported)),
        "transform_scope": "compiled MuJoCo world transforms after bounded settle",
        "physics_authorized": False,
    }


class AssetStore:
    """Atomic local store for bounded single-file model assets."""

    def __init__(self, root):
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise ValueError("symlink asset root refused")
        self.root = requested.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _directory(self, asset_id: str) -> Path:
        if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id):
            raise KeyError("asset not found")
        path = self.root / asset_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise KeyError("asset not found")
        return path

    def import_asset(self, name, format, *, content=None, content_base64=None) -> dict:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("name must be 1..128 printable characters")
        if not isinstance(format, str) or format.lower() not in FORMATS:
            raise ValueError("format must be obj, stl, mjcf, or urdf")
        format_name = format.lower()
        if (content is None) == (content_base64 is None):
            raise ValueError("provide exactly one of content or content_base64")
        if content_base64 is not None:
            if not isinstance(content_base64, str) or len(content_base64) > MAX_ASSET_BYTES * 2:
                raise ValueError("invalid base64 asset")
            try:
                raw = base64.b64decode(content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("invalid base64 asset") from exc
        else:
            if not isinstance(content, (bytes, bytearray, memoryview)):
                raise TypeError("content must be bytes")
            raw = bytes(content)
        if not raw or len(raw) > MAX_ASSET_BYTES:
            raise ValueError("asset must be non-empty and no larger than 8 MiB")

        if format_name == "obj":
            geometry = _parse_obj(raw)
        elif format_name == "stl":
            geometry = _parse_stl(raw)
        else:
            xml_root = _parse_safe_xml(raw, format_name)
            geometry = _xml_geometry(xml_root, format_name)

        asset_id = "asset-" + uuid.uuid4().hex
        geometry_count = geometry.get("triangle_count", geometry.get("primitive_count", 0))
        record = {
            "id": asset_id,
            "name": name,
            "format": format_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "geometry_kind": geometry["kind"],
            "geometry_count": geometry_count,
            "preview_available": geometry_count > 0,
            "physics_check": "not_run" if format_name in {"mjcf", "urdf"} else "not_applicable",
            "physics_authorized": False,
            "capabilities": {
                "preview": True,
                "mujoco_check": format_name in {"mjcf", "urdf"},
                "fixture_authorization": False,
                "training": False,
            },
        }
        with self._lock:
            temporary = self.root / ("." + asset_id + "." + uuid.uuid4().hex + ".tmp")
            temporary.mkdir()
            try:
                _atomic_write(temporary / ("source." + format_name), raw)
                _atomic_write(temporary / "geometry.json", canonical_json(geometry))
                _atomic_write(temporary / "asset.json", canonical_json(record))
                os.replace(temporary, self._directory(asset_id))
            except Exception:
                import shutil
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return dict(record)

    def get(self, asset_id) -> dict:
        with self._lock:
            try:
                record = json.loads((self._directory(asset_id) / "asset.json").read_text("utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise KeyError("asset not found") from exc
            if record.get("id") != asset_id or record.get("format") not in FORMATS:
                raise KeyError("asset not found")
            return record

    def list(self) -> list[dict]:
        with self._lock:
            records = []
            paths = [path for path in self.root.iterdir()
                     if path.is_dir() and not path.is_symlink() and ASSET_ID.fullmatch(path.name)]
            for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[:200]:
                if path.is_dir() and not path.is_symlink() and ASSET_ID.fullmatch(path.name):
                    try:
                        records.append(self.get(path.name))
                    except KeyError:
                        continue
            return records

    def geometry(self, asset_id) -> dict:
        with self._lock:
            self.get(asset_id)
            try:
                value = json.loads((self._directory(asset_id) / "geometry.json").read_text("utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise KeyError("asset geometry not found") from exc
            if not isinstance(value, dict) or value.get("schema") != "sentinel-geometry-v1":
                raise KeyError("asset geometry not found")
            return value

    def _save_check(self, record: dict, report: dict, geometry: dict | None = None) -> dict:
        directory = self._directory(record["id"])
        _atomic_write(directory / "check.json", canonical_json(report))
        if geometry is not None:
            _atomic_write(directory / "geometry.json", canonical_json(geometry))
            record["geometry_kind"] = geometry["kind"]
            record["geometry_count"] = geometry.get("triangle_count", geometry.get("primitive_count", 0))
            record["preview_available"] = record["geometry_count"] > 0
        record["physics_check"] = report["status"]
        record["last_check"] = report
        _atomic_write(directory / "asset.json", canonical_json(record))
        return report

    def check(self, asset_id) -> dict:
        with self._lock:
            record = self.get(asset_id)
            if record["format"] not in {"mjcf", "urdf"}:
                return self._save_check(record, {
                    "asset_id": asset_id,
                    "status": "not_applicable",
                    "scope": "mesh syntax and preview only",
                    "physics_authorized": False,
                })
            if importlib.util.find_spec("mujoco") is None:
                return self._save_check(record, {
                    "asset_id": asset_id,
                    "status": "unavailable",
                    "error": {"code": "MUJOCO_NOT_INSTALLED", "message": "Install sentinel-evc-lab[physics] to run the model check."},
                    "physics_authorized": False,
                })
            source_path = self._directory(asset_id) / ("source." + record["format"])
            raw = source_path.read_bytes()
            if "sha256:" + hashlib.sha256(raw).hexdigest() != record.get("source_sha256"):
                raise ValueError("stored source differs from its import digest")
            _parse_safe_xml(raw, record["format"])
            try:
                import mujoco
                model = mujoco.MjModel.from_xml_string(raw.decode("utf-8"))
                if (int(model.nbody) > MAX_XML_ELEMENTS or int(model.ngeom) > MAX_MODEL_GEOMS
                        or int(model.njnt) > MAX_MODEL_JOINTS or int(model.nu) > MAX_MODEL_ACTUATORS):
                    raise ValueError("compiled model exceeds inspection limits")
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                for _ in range(5):
                    mujoco.mj_step(model, data)
                state = list(data.qpos) + list(data.qvel)
                if any(not math.isfinite(float(value)) for value in state):
                    raise ValueError("compiled model produced non-finite state")
                warning_stats = getattr(data, "warning", ())
                if any(int(warning.number) > 0 for warning in warning_stats):
                    raise ValueError("compiled model emitted MuJoCo warnings during settle")
                geometry = _compiled_geometry(mujoco, model, data)
                report = {
                    "asset_id": asset_id,
                    "status": "passed",
                    "scope": "MuJoCo compile, model inspection, and five-step settle",
                    "engine": getattr(mujoco, "__version__", "unknown"),
                    "counts": {"bodies": int(model.nbody), "geoms": int(model.ngeom),
                               "joints": int(model.njnt), "actuators": int(model.nu)},
                    "settle_steps": 5,
                    "geometry_updated": True,
                    "physics_authorized": False,
                    "authorization_note": "Inspection does not authorize this model for Executor dispatch or training.",
                }
            except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
                report = {
                    "asset_id": asset_id,
                    "status": "failed",
                    "error": {"code": "MODEL_CHECK_FAILED", "message": str(exc)[:512]},
                    "physics_authorized": False,
                }
                geometry = None
            return self._save_check(record, report, geometry)
