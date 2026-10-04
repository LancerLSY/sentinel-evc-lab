#!/usr/bin/env python3
"""Native MuJoCo geometry checks for the LIBERO Panda experiment.

The native seven-dimensional policy action is relative Cartesian control.  It
is bound to a result as opaque bytes and is never interpreted as joint angles.
The trajectory accepted here is a separate forecast in the seven actual Panda
hinge coordinates and, when configured, controlled finger slide coordinates,
beginning at the live simulator configuration.

Continuous certificates use MuJoCo's compiled collision geometries together
with a conservative joint-motion Lipschitz bound.  ``dense_review`` is a point
sampling diagnostic and deliberately does not issue a continuous certificate.
This experiment is not a functional-safety or physical-stop system.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTINUOUS_SCOPE = "continuous_joint_linear_frozen_scene_lipschitz"
_MAX_EXTRA_SEPARATOR_AXES = 2048


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("panda geometry checking requires the optional mujoco package") from exc
    return mujoco


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _array_bytes(value: Any) -> bytes:
    array = np.ascontiguousarray(value)
    return (str(array.dtype).encode() + b"\0" + repr(array.shape).encode() + b"\0" + array.tobytes())


def _digest_arrays(tag: str, arrays: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256(tag.encode() + b"\0")
    for name, value in arrays:
        digest.update(name.encode() + b"\0")
        digest.update(_array_bytes(value))
    return digest.hexdigest()


def _name(model: Any, object_type: Any, index: int) -> str:
    value = _mujoco().mj_id2name(model, object_type, int(index))
    return value if value is not None else f"unnamed-{int(index)}"


def _pair(a: int, b: int) -> tuple[int, int]:
    return (int(a), int(b)) if a < b else (int(b), int(a))


@dataclass(frozen=True)
class AllowedContact:
    """An exact unordered geom pair allowed by the task.

    ``max_penetration=None`` ignores the pair.  A finite value still rejects a
    contact deeper than that many metres.
    """

    geom_a: int
    geom_b: int
    label: str
    max_penetration: float | None = None

    def __post_init__(self) -> None:
        if self.geom_a == self.geom_b or self.geom_a < 0 or self.geom_b < 0:
            raise ValueError("allowed contact needs two distinct non-negative geom ids")
        if not self.label:
            raise ValueError("allowed contact label is required")
        if self.max_penetration is not None and (
            not math.isfinite(self.max_penetration) or self.max_penetration < 0
        ):
            raise ValueError("max_penetration must be finite and non-negative")

    @property
    def pair(self) -> tuple[int, int]:
        return _pair(self.geom_a, self.geom_b)


@dataclass(frozen=True)
class GeometryConfig:
    max_joint_step_rad: float = 0.01
    max_subdivision_depth: int = 12
    distance_query_max: float = 10.0
    collision_tolerance: float = 0.0
    continuous: bool = True
    separator_min_pad_m: float = 1e-9
    tracking_reserve_rad: float = 0.0
    max_slide_step_m: float = 0.001
    tracking_reserve_slide_m: float = 0.0
    mesh_separator_axes: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_joint_step_rad) or self.max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive and finite")
        if not math.isfinite(self.max_slide_step_m) or self.max_slide_step_m <= 0:
            raise ValueError("max_slide_step_m must be positive and finite")
        if not isinstance(self.max_subdivision_depth, int) or self.max_subdivision_depth < 0:
            raise ValueError("max_subdivision_depth must be a non-negative integer")
        if not math.isfinite(self.distance_query_max) or self.distance_query_max <= 0:
            raise ValueError("distance_query_max must be positive and finite")
        if not math.isfinite(self.collision_tolerance):
            raise ValueError("collision_tolerance must be finite")
        if not math.isfinite(self.separator_min_pad_m) or self.separator_min_pad_m < 1e-9:
            raise ValueError("separator_min_pad_m must be finite and at least 1e-9 metres")
        if not math.isfinite(self.tracking_reserve_rad) or self.tracking_reserve_rad < 0:
            raise ValueError("tracking_reserve_rad must be finite and non-negative")
        if not math.isfinite(self.tracking_reserve_slide_m) or self.tracking_reserve_slide_m < 0:
            raise ValueError("tracking_reserve_slide_m must be finite and non-negative")
        if not isinstance(self.mesh_separator_axes, bool):
            raise ValueError("mesh_separator_axes must be a bool")

    @property
    def digest(self) -> str:
        return _json_sha(asdict(self))


@dataclass(frozen=True)
class PairSpec:
    geom_a: int
    geom_b: int
    kind: str
    threshold: float
    sensitivity: tuple[float, ...]
    allowed_label: str | None = None

    @property
    def pair(self) -> tuple[int, int]:
        return _pair(self.geom_a, self.geom_b)

    @property
    def key(self) -> str:
        a, b = self.pair
        return f"{a}:{b}"


@dataclass(frozen=True)
class SceneSnapshot:
    digest: str
    state_digest: str
    model_digest: str
    external_sha256: str
    simulation_time: float


@dataclass(frozen=True)
class CollisionRecord:
    contact_index: int
    geom_a: int
    geom_b: int
    geom_a_name: str
    geom_b_name: str
    body_a: int
    body_b: int
    distance: float
    pair_kind: str
    allowed_label: str | None


@dataclass(frozen=True)
class GeometryCertificate:
    digest: str
    action_sha256: str
    action_nbytes: int
    trajectory_sha256: str
    trajectory_dtype: str
    trajectory_shape: tuple[int, int]
    trajectory: tuple[tuple[float, ...], ...]
    coordinate_units: tuple[str, ...]
    scene_digest: str
    state_digest: str
    model_digest: str
    config_digest: str
    profile_digest: str
    scope: str
    pair_margins: tuple[tuple[str, float], ...]
    min_margin: float
    min_joint_limit_margin_rad: float
    min_slide_limit_margin_m: float
    parent_digest: str | None


@dataclass(frozen=True)
class GeometryDecision:
    allowed: bool
    status: str
    method: str
    scope: str
    min_margin: float | None
    min_joint_limit_margin_rad: float | None
    unknown_pairs: tuple[str, ...]
    violations: tuple[CollisionRecord, ...]
    work: Mapping[str, int]
    latency_ns: int
    certificate: GeometryCertificate | None
    reason: str
    min_slide_limit_margin_m: float | None = None


@dataclass(frozen=True)
class ExecutionTrackingReport:
    certificate_digest: str
    executed_trajectory_sha256: str
    executed_dtype: str
    executed_shape: tuple[int, ...]
    aligned: bool
    rmse_rad: float | None
    max_abs_error_rad: float | None
    final_max_abs_error_rad: float | None
    per_joint_max_abs_error_rad: tuple[float, ...]
    note: str
    rmse_slide_m: float | None = None
    max_abs_error_slide_m: float | None = None
    final_max_abs_error_slide_m: float | None = None
    per_slide_max_abs_error_m: tuple[float, ...] = ()


def _model_digest(model: Any) -> str:
    """Fingerprint geometry, hierarchy, joints and collision filtering in MjModel."""
    fields = (
        "body_parentid", "body_pos", "body_quat", "body_weldid", "body_jntadr", "body_jntnum",
        "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
        "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity", "geom_group",
        "geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_aabb",
        "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_faceadr", "mesh_facenum", "mesh_face",
        "hfield_adr", "hfield_nrow", "hfield_ncol", "hfield_size", "hfield_data",
        "pair_geom1", "pair_geom2", "exclude_signature",
    )
    arrays = []
    for field in fields:
        if hasattr(model, field):
            arrays.append((field, getattr(model, field)))
    if hasattr(model, "names"):
        arrays.append(("names", np.frombuffer(bytes(model.names), dtype=np.uint8)))
    shape = np.array(
        [model.nq, model.nv, model.nu, model.nbody, model.njnt, model.ngeom, model.nmesh],
        dtype=np.int64,
    )
    arrays.insert(0, ("shape", shape))
    return _digest_arrays("panda-mjmodel-v1", arrays)


def _geom_radius(model: Any, geom_id: int) -> float:
    mj = _mujoco()
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if kind == int(mj.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
        return float(size[0] + size[1])
    if kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
        return float(math.hypot(size[0], size[1]))
    if kind in (int(mj.mjtGeom.mjGEOM_BOX), int(mj.mjtGeom.mjGEOM_ELLIPSOID)):
        return float(np.linalg.norm(size[:3]))
    if kind == int(mj.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        if count <= 0:
            raise ValueError(f"mesh geom {geom_id} has no compiled vertices")
        return float(np.max(np.linalg.norm(np.asarray(model.mesh_vert[start:start + count]), axis=1)))
    if kind == int(mj.mjtGeom.mjGEOM_HFIELD):
        return float(np.linalg.norm(size[:3]))
    if kind == int(mj.mjtGeom.mjGEOM_PLANE):
        return math.inf
    # MuJoCo's zero-radius POINT geom and any future bounded primitive.
    if np.all(np.isfinite(size)):
        return float(np.linalg.norm(size[:3]))
    raise ValueError(f"cannot bound geom {geom_id} type {kind}")


def _is_descendant(model: Any, body: int, ancestor: int) -> bool:
    current = int(body)
    while current > 0:
        if current == ancestor:
            return True
        current = int(model.body_parentid[current])
    return ancestor == 0 and current == 0


def _body_translation_extent(model: Any, body: int) -> float:
    """Bound joint-created translation of a body frame from its XML body_pos."""
    mj = _mujoco()
    start = int(model.body_jntadr[body])
    count = int(model.body_jntnum[body])
    extent = 0.0
    for joint_id in range(start, start + count):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mj.mjtJoint.mjJNT_SLIDE):
            if hasattr(model, "jnt_limited") and not bool(model.jnt_limited[joint_id]):
                raise ValueError(f"cannot globally bound unbounded slide joint {joint_id}")
            extent += float(np.max(np.abs(model.jnt_range[joint_id])))
        elif joint_type == int(mj.mjtJoint.mjJNT_FREE):
            raise ValueError(f"cannot globally bound free joint {joint_id} in the arm subtree")
    return extent


def _motion_sensitivity(model: Any, geom_id: int, joint_ids: Sequence[int]) -> tuple[float, ...]:
    mj = _mujoco()
    geom_body = int(model.geom_bodyid[geom_id])
    radius = _geom_radius(model, geom_id)
    values: list[float] = []
    for joint_id in joint_ids:
        joint_body = int(model.jnt_bodyid[joint_id])
        if not _is_descendant(model, geom_body, joint_body):
            values.append(0.0)
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mj.mjtJoint.mjJNT_SLIDE):
            values.append(1.0)
            continue
        if joint_type != int(mj.mjtJoint.mjJNT_HINGE):
            raise ValueError("controlled geometry coordinates must be scalar hinge/slide joints")
        joint_pos = np.asarray(model.jnt_pos[joint_id], dtype=np.float64)
        geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        if geom_body == joint_body:
            bound = (
                float(np.linalg.norm(geom_pos - joint_pos)) + radius
                + _body_translation_extent(model, geom_body)
            )
        else:
            bound = float(np.linalg.norm(joint_pos)) + radius + float(np.linalg.norm(geom_pos))
            current = geom_body
            while current != joint_body:
                bound += (
                    float(np.linalg.norm(model.body_pos[current]))
                    + _body_translation_extent(model, current)
                )
                current = int(model.body_parentid[current])
        values.append(bound)
    return tuple(values)


def _mask_enabled(model: Any, a: int, b: int) -> bool:
    return bool(
        (int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
        or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))
    )


def _excluded_body_pair(model: Any, a: int, b: int) -> bool:
    """Apply compiled ``<exclude>`` signatures to generated geom pairs."""
    if int(model.nexclude) == 0:
        return False
    body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
    weld_a, weld_b = int(model.body_weldid[body_a]), int(model.body_weldid[body_b])
    signatures = set(int(value) for value in np.asarray(model.exclude_signature).reshape(-1))
    candidates = {
        (body_a << 16) + body_b, (body_b << 16) + body_a,
        (weld_a << 16) + weld_b, (weld_b << 16) + weld_a,
    }
    return not signatures.isdisjoint(candidates)


def _infer_joint_ids(model: Any, qpos_indices: Sequence[int]) -> tuple[int, ...]:
    found = []
    for qpos_index in qpos_indices:
        matches = np.flatnonzero(np.asarray(model.jnt_qposadr) == int(qpos_index))
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely infer joint for qpos index {qpos_index}")
        found.append(int(matches[0]))
    return tuple(found)


@dataclass(frozen=True)
class PandaGeometryProfile:
    joint_qpos_indices: tuple[int, ...]
    joint_ids: tuple[int, ...]
    controlled_extra_qpos_indices: tuple[int, ...]
    controlled_extra_joint_ids: tuple[int, ...]
    moving_qpos_indices: tuple[int, ...]
    moving_joint_ids: tuple[int, ...]
    coordinate_units: tuple[str, ...]
    arm_body_ids: tuple[int, ...]
    arm_geom_ids: tuple[int, ...]
    scene_geom_ids: tuple[int, ...]
    checked_pairs: tuple[tuple[int, int], ...]
    pair_specs: tuple[PairSpec, ...]
    allowed_contacts: tuple[AllowedContact, ...]
    ignored_geom_ids: tuple[int, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    model_digest: str
    config: GeometryConfig
    digest: str

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        joint_qpos_indices: Sequence[int],
        joint_ids: Sequence[int] | None = None,
        controlled_extra_qpos_indices: Sequence[int] = (),
        controlled_extra_joint_ids: Sequence[int] | None = None,
        arm_body_ids: Sequence[int] | None = None,
        allowed_contacts: Sequence[AllowedContact] = (),
        ignored_geom_ids: Sequence[int] = (),
        config: GeometryConfig = GeometryConfig(),
    ) -> "PandaGeometryProfile":
        qpos = tuple(int(v) for v in joint_qpos_indices)
        if len(qpos) != 7 or len(set(qpos)) != 7 or any(v < 0 or v >= model.nq for v in qpos):
            raise ValueError("joint_qpos_indices must identify exactly seven distinct native Panda coordinates")
        joints = tuple(int(v) for v in (joint_ids if joint_ids is not None else _infer_joint_ids(model, qpos)))
        if len(joints) != 7 or len(set(joints)) != 7:
            raise ValueError("joint_ids must contain exactly seven distinct Panda joints")
        expected_qpos = tuple(int(model.jnt_qposadr[j]) for j in joints)
        if expected_qpos != qpos:
            raise ValueError("joint_ids and joint_qpos_indices do not describe the same ordered joints")
        extra_qpos = tuple(int(v) for v in controlled_extra_qpos_indices)
        if len(set(extra_qpos)) != len(extra_qpos) or any(v < 0 or v >= model.nq for v in extra_qpos):
            raise ValueError("controlled_extra_qpos_indices must be distinct valid scalar coordinates")
        if set(qpos) & set(extra_qpos):
            raise ValueError("controlled extra coordinates must not duplicate the seven Panda arm coordinates")
        if len(extra_qpos) not in (0, 2):
            raise ValueError("configure either both Panda finger slide coordinates or neither")
        extra_joints = tuple(int(v) for v in (
            controlled_extra_joint_ids
            if controlled_extra_joint_ids is not None
            else _infer_joint_ids(model, extra_qpos)
        ))
        if len(extra_joints) != len(extra_qpos) or len(set(extra_joints)) != len(extra_joints):
            raise ValueError("controlled_extra_joint_ids must match the distinct extra coordinates")
        if tuple(int(model.jnt_qposadr[j]) for j in extra_joints) != extra_qpos:
            raise ValueError("controlled extra joint ids and qpos indices do not match")
        mj = _mujoco()
        if any(int(model.jnt_type[j]) != int(mj.mjtJoint.mjJNT_HINGE) for j in joints):
            raise ValueError("the seven Panda arm coordinates must be hinge joints in radians")
        if any(int(model.jnt_type[j]) != int(mj.mjtJoint.mjJNT_SLIDE) for j in extra_joints):
            raise ValueError("controlled extra coordinates must be slide joints in metres")
        moving_qpos = qpos + extra_qpos
        moving_joints = joints + extra_joints
        coordinate_units = ("rad",) * len(joints) + ("m",) * len(extra_joints)
        seeds = tuple(int(v) for v in (arm_body_ids or tuple(int(model.jnt_bodyid[j]) for j in joints)))
        if not seeds or any(v <= 0 or v >= model.nbody for v in seeds):
            raise ValueError("arm_body_ids must reference non-world bodies")
        arm_bodies = tuple(
            b for b in range(1, model.nbody) if any(_is_descendant(model, b, seed) for seed in seeds)
        )
        if any(int(model.jnt_bodyid[j]) not in arm_bodies for j in extra_joints):
            raise ValueError("controlled extra joints must belong to the Panda arm body subtree")
        ignored = tuple(sorted(set(int(v) for v in ignored_geom_ids)))
        if any(v < 0 or v >= model.ngeom for v in ignored):
            raise ValueError("ignored geom id out of range")
        arm = tuple(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies and g not in ignored)
        scene = tuple(g for g in range(model.ngeom) if g not in set(arm) and g not in ignored)
        if not arm:
            raise ValueError("no arm geoms discovered below the Panda joints")
        allowed = tuple(allowed_contacts)
        allowed_by_pair = {rule.pair: rule for rule in allowed}
        if len(allowed_by_pair) != len(allowed):
            raise ValueError("duplicate allowed-contact pair")
        if any(max(rule.pair) >= model.ngeom for rule in allowed):
            raise ValueError("allowed-contact geom id out of range")

        specs: list[PairSpec] = []
        added_pairs: set[tuple[int, int]] = set()
        sensitivity = {g: _motion_sensitivity(model, g, moving_joints) for g in arm}

        def add(a: int, b: int, kind: str) -> None:
            pair = _pair(a, b)
            if pair in added_pairs:
                return
            rule = allowed_by_pair.get(pair)
            if rule is not None and rule.max_penetration is None:
                added_pairs.add(pair)
                return
            threshold = -float(rule.max_penetration) if rule is not None else config.collision_tolerance
            sens = tuple(
                sensitivity.get(a, (0.0,) * len(moving_joints))[i]
                + sensitivity.get(b, (0.0,) * len(moving_joints))[i]
                for i in range(len(moving_joints))
            )
            specs.append(PairSpec(a, b, kind, threshold, sens, rule.label if rule else None))
            added_pairs.add(pair)

        arm_set, scene_set, ignored_set = set(arm), set(scene), set(ignored)
        # Explicit <pair> entries are collision candidates even when contype /
        # conaffinity masks would reject the generated pair.
        for index in range(int(model.npair)):
            a, b = int(model.pair_geom1[index]), int(model.pair_geom2[index])
            if a in ignored_set or b in ignored_set:
                continue
            if (a in arm_set and b in scene_set) or (b in arm_set and a in scene_set):
                add(a, b, "arm_scene")
            elif a in arm_set and b in arm_set:
                add(a, b, "self")

        for a in arm:
            for b in scene:
                if _mask_enabled(model, a, b) and not _excluded_body_pair(model, a, b):
                    add(a, b, "arm_scene")
        mj = _mujoco()
        try:
            filter_parent = not bool(
                int(model.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_FILTERPARENT)
            )
        except (AttributeError, TypeError):
            filter_parent = True
        for a, b in itertools.combinations(arm, 2):
            body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            weld_a, weld_b = int(model.body_weldid[body_a]), int(model.body_weldid[body_b])
            if (
                weld_a == weld_b
                or not _mask_enabled(model, a, b)
                or _excluded_body_pair(model, a, b)
            ):
                continue
            # MuJoCo normally filters adjacent welded bodies.  Omitting them also
            # avoids treating intentionally overlapping link-end primitives as a
            # self collision; live contacts and distance checks share this policy.
            parent_weld_a = int(model.body_weldid[int(model.body_parentid[weld_a])]) if weld_a else -1
            parent_weld_b = int(model.body_weldid[int(model.body_parentid[weld_b])]) if weld_b else -1
            if filter_parent and (parent_weld_a == weld_b or parent_weld_b == weld_a):
                continue
            add(a, b, "self")
        specs.sort(key=lambda spec: spec.pair)
        if not specs:
            raise ValueError("profile discovered no collision-enabled arm/scene or self geom pairs")
        if any(not bool(model.jnt_limited[j]) for j in moving_joints):
            raise ValueError("every controlled moving joint must have finite limits")
        low = tuple(float(model.jnt_range[j, 0]) for j in moving_joints)
        high = tuple(float(model.jnt_range[j, 1]) for j in moving_joints)
        model_sha = _model_digest(model)
        payload = {
            "joint_qpos_indices": qpos, "joint_ids": joints,
            "controlled_extra_qpos_indices": extra_qpos,
            "controlled_extra_joint_ids": extra_joints,
            "moving_qpos_indices": moving_qpos, "moving_joint_ids": moving_joints,
            "coordinate_units": coordinate_units, "arm_body_ids": arm_bodies,
            "arm_geom_ids": arm, "scene_geom_ids": scene, "ignored_geom_ids": ignored,
            "pair_specs": [asdict(spec) for spec in specs],
            "allowed_contacts": [asdict(rule) for rule in allowed],
            "joint_lower": low, "joint_upper": high,
            "model_digest": model_sha, "config_digest": config.digest,
        }
        return cls(
            qpos, joints, extra_qpos, extra_joints, moving_qpos, moving_joints,
            coordinate_units, arm_bodies, arm, scene, tuple(spec.pair for spec in specs),
            tuple(specs), allowed, ignored, low, high, model_sha, config, _json_sha(payload),
        )


def _live_digests(model: Any, data: Any, profile: PandaGeometryProfile, external_sha: str) -> tuple[str, str]:
    moving_qpos = set(profile.moving_qpos_indices)
    other_qpos = np.asarray([data.qpos[i] for i in range(model.nq) if i not in moving_qpos], dtype=np.float64)
    scene_arrays: list[tuple[str, Any]] = [
        ("other_qpos", other_qpos), ("mocap_pos", data.mocap_pos), ("mocap_quat", data.mocap_quat),
    ]
    if hasattr(data, "eq_active"):
        scene_arrays.append(("eq_active", data.eq_active))
    scene_base = _digest_arrays("panda-scene-v1", scene_arrays)
    scene_digest = _json_sha({"model": profile.model_digest, "scene": scene_base, "external": external_sha})
    state_arrays = [
        ("qpos", data.qpos), ("qvel", data.qvel), ("act", data.act), ("ctrl", data.ctrl),
        ("mocap_pos", data.mocap_pos), ("mocap_quat", data.mocap_quat),
        ("time", np.asarray([data.time], dtype=np.float64)),
    ]
    state_digest = _digest_arrays("panda-live-state-v1", state_arrays)
    return scene_digest, state_digest


def capture_scene_snapshot(
    model: Any,
    data: Any,
    profile: PandaGeometryProfile,
    *,
    external_snapshot: bytes = b"",
) -> SceneSnapshot:
    if not isinstance(external_snapshot, bytes):
        raise TypeError("external_snapshot must be exact bytes")
    if _model_digest(model) != profile.model_digest:
        raise ValueError("model changed after Panda geometry profile construction")
    external_sha = _sha(external_snapshot)
    scene_digest, state_digest = _live_digests(model, data, profile, external_sha)
    return SceneSnapshot(scene_digest, state_digest, profile.model_digest, external_sha, float(data.time))


def _copy_data(model: Any, source: Any) -> Any:
    mj = _mujoco()
    target = mj.MjData(model)
    copied = False
    if hasattr(mj, "mj_copyData"):
        try:
            mj.mj_copyData(target, model, source)
            copied = True
        except TypeError:
            copied = False
    if not copied:
        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
            if hasattr(source, field) and hasattr(target, field):
                getattr(target, field)[...] = getattr(source, field)
        if hasattr(source, "eq_active") and hasattr(target, "eq_active"):
            target.eq_active[...] = source.eq_active
        target.time = source.time
        mj.mj_forward(model, target)
    return target


def current_unwanted_collisions(
    model: Any, data: Any, profile: PandaGeometryProfile
) -> tuple[CollisionRecord, ...]:
    """Return live contacts that violate the profile's exact pair policy."""
    mj = _mujoco()
    spec_by_pair = {spec.pair: spec for spec in profile.pair_specs}
    records = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = _pair(int(contact.geom1), int(contact.geom2))
        spec = spec_by_pair.get(pair)
        if spec is None or float(contact.dist) > spec.threshold:
            continue
        a, b = pair
        records.append(CollisionRecord(
            index, a, b,
            _name(model, mj.mjtObj.mjOBJ_GEOM, a), _name(model, mj.mjtObj.mjOBJ_GEOM, b),
            int(model.geom_bodyid[a]), int(model.geom_bodyid[b]), float(contact.dist),
            spec.kind, spec.allowed_label,
        ))
    return tuple(records)


class _UnknownDistance(RuntimeError):
    pass


def _support_interval(model: Any, data: Any, geom_id: int, direction: np.ndarray) -> tuple[float, float]:
    mj = _mujoco()
    kind = int(model.geom_type[geom_id])
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    projected_center = float(np.dot(center, direction))
    if kind == int(mj.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
        projected = (vertices @ rotation.T + center) @ direction
        return float(np.min(projected)), float(np.max(projected))
    if kind == int(mj.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
    elif kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
        radius = float(size[0] + size[1] * abs(np.dot(direction, rotation[:, 2])))
    elif kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
        axial = abs(float(np.dot(direction, rotation[:, 2])))
        radius = float(size[1] * axial + size[0] * math.sqrt(max(0.0, 1.0 - axial * axial)))
    elif kind == int(mj.mjtGeom.mjGEOM_BOX):
        radius = float(np.dot(size[:3], np.abs(rotation.T @ direction)))
    elif kind == int(mj.mjtGeom.mjGEOM_ELLIPSOID):
        radius = float(np.linalg.norm(size[:3] * (rotation.T @ direction)))
    else:
        # Conservative compiled AABB support for hfields and future bounded
        # primitives. Infinite planes deliberately remain unsupported here.
        if kind == int(mj.mjtGeom.mjGEOM_PLANE):
            raise _UnknownDistance(f"{geom_id}:plane_separator_unsupported")
        local_center, local_half = model.geom_aabb[geom_id, :3], model.geom_aabb[geom_id, 3:]
        center = center + rotation @ local_center
        radius = float(np.dot(np.abs(rotation) @ local_half, np.abs(direction)))
        projected_center = float(np.dot(center, direction))
    return projected_center - radius, projected_center + radius


def _separating_lower_bound(
    model: Any, data: Any, geom_a: int, geom_b: int, witness: np.ndarray, minimum_pad: float,
    mesh_separator_axes: bool = False,
) -> float:
    rotations = [
        np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        for geom_id in (geom_a, geom_b)
    ]
    directions = [np.eye(3)[i] for i in range(3)]
    directions.extend(rotation[:, i] for rotation in rotations for i in range(3))
    directions.append(np.asarray(data.geom_xpos[geom_b]) - np.asarray(data.geom_xpos[geom_a]))
    if np.all(np.isfinite(witness)):
        directions.append(np.asarray(witness[3:]) - np.asarray(witness[:3]))
    best = 0.0
    projection_scale = 1.0
    seen: set[tuple[float, float, float]] = set()
    extra_axes = 0

    def consume(raw_directions: Iterable[np.ndarray], *, extra: bool = False) -> None:
        nonlocal best, projection_scale, extra_axes
        for raw in raw_directions:
            if extra and extra_axes >= _MAX_EXTRA_SEPARATOR_AXES:
                break
            norm = float(np.linalg.norm(raw))
            if not math.isfinite(norm) or norm <= 1e-15:
                continue
            direction = np.asarray(raw, dtype=np.float64) / norm
            key = tuple(float(round(value, 12)) for value in direction)
            if key in seen:
                continue
            seen.add(key)
            if extra:
                extra_axes += 1
            first = _support_interval(model, data, geom_a, direction)
            second = _support_interval(model, data, geom_b, direction)
            if not all(math.isfinite(value) for value in (*first, *second)):
                raise _UnknownDistance("nonfinite_support_projection")
            projection_scale = max(projection_scale, *(abs(value) for value in (*first, *second)))
            best = max(best, first[0] - second[1], second[0] - first[1])

    def lower_bound() -> float:
        pad = minimum_pad + 64 * np.finfo(np.float64).eps * projection_scale
        return max(0.0, best - pad)

    consume(directions)
    if mesh_separator_axes and lower_bound() == 0.0:
        mj = _mujoco()

        def mesh_batches(geom_id: int, other_rotation: np.ndarray, edges: bool):
            if int(model.geom_type[geom_id]) != int(mj.mjtGeom.mjGEOM_MESH):
                return
            mesh_id = int(model.geom_dataid[geom_id])
            vert_start = int(model.mesh_vertadr[mesh_id])
            vert_count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vert_start:vert_start + vert_count], dtype=np.float64,
            )
            faces = np.asarray(
                model.mesh_face[face_start:face_start + face_count], dtype=np.int64,
            ).reshape(-1, 3)
            if faces.size and not np.all((faces >= 0) & (faces < vert_count)):
                if np.all((faces >= vert_start) & (faces < vert_start + vert_count)):
                    faces = faces - vert_start
                else:
                    raise _UnknownDistance(f"{geom_id}:mesh_face_index_out_of_range")
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            other_axes = other_rotation.T
            for start in range(0, len(faces), 256):
                triangles = vertices[faces[start:start + 256]]
                local_edges = np.stack(
                    (triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 1],
                     triangles[:, 0] - triangles[:, 2]),
                    axis=1,
                )
                world_edges = local_edges @ rotation.T
                if not edges:
                    yield np.cross(world_edges[:, 0], world_edges[:, 1])
                else:
                    crosses = np.cross(
                        world_edges[:, :, None, :], other_axes[None, None, :, :],
                    )
                    yield crosses.reshape(-1, 3)

        def consume_mesh_phase(edges: bool) -> None:
            for geom_id, other_rotation in ((geom_a, rotations[1]), (geom_b, rotations[0])):
                for batch in mesh_batches(geom_id, other_rotation, edges) or ():
                    consume(batch, extra=True)
                    if lower_bound() > 0.0 or extra_axes >= _MAX_EXTRA_SEPARATOR_AXES:
                        return

        consume_mesh_phase(False)
        if lower_bound() == 0.0 and extra_axes < _MAX_EXTRA_SEPARATOR_AXES:
            consume_mesh_phase(True)
    # Include a scene-scale roundoff budget and a declared metre floor.
    return lower_bound()


def _distance(
    model: Any, data: Any, spec: PairSpec, maximum: float,
    contacts_by_pair: dict[tuple[int, int], list[float]] | None = None,
    separator_min_pad_m: float = 1e-9,
    mesh_separator_axes: bool = False,
) -> float:
    mj = _mujoco()
    nearest = np.full(6, np.nan, dtype=np.float64)
    try:
        value = float(mj.mj_geomDistance(model, data, spec.geom_a, spec.geom_b, maximum, nearest))
    except Exception as exc:
        raise _UnknownDistance(f"{spec.key}:{type(exc).__name__}") from exc
    if not math.isfinite(value):
        raise _UnknownDistance(f"{spec.key}:nonfinite")
    if contacts_by_pair is None:
        contacts_by_pair = {}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            contacts_by_pair.setdefault(_pair(int(contact.geom1), int(contact.geom2)), []).append(float(contact.dist))
    pair_contacts = contacts_by_pair.get(spec.pair, [])
    if any(not math.isfinite(distance) for distance in pair_contacts):
        raise _UnknownDistance(f"{spec.key}:nonfinite_contact")
    if pair_contacts:
        value = min(value, *pair_contacts)
    if value == 0.0:
        if pair_contacts:
            return value
        try:
            lower = _separating_lower_bound(
                model, data, spec.geom_a, spec.geom_b, nearest, separator_min_pad_m,
                mesh_separator_axes,
            )
        except _UnknownDistance:
            lower = 0.0
        if lower > 0.0:
            return lower
        raise _UnknownDistance(f"{spec.key}:native_zero_without_contact_no_separator")
    return value


def _collision_record(model: Any, spec: PairSpec, distance: float) -> CollisionRecord:
    mj = _mujoco()
    a, b = spec.pair
    return CollisionRecord(
        -1, a, b, _name(model, mj.mjtObj.mjOBJ_GEOM, a),
        _name(model, mj.mjtObj.mjOBJ_GEOM, b), int(model.geom_bodyid[a]),
        int(model.geom_bodyid[b]), distance, spec.kind, spec.allowed_label,
    )


def _set_q(model: Any, data: Any, profile: PandaGeometryProfile, q: np.ndarray) -> None:
    data.qpos[list(profile.moving_qpos_indices)] = q
    _mujoco().mj_forward(model, data)


def _tracking_reserves(profile: PandaGeometryProfile) -> np.ndarray:
    reserves = []
    for unit in profile.coordinate_units:
        if unit == "rad":
            reserves.append(profile.config.tracking_reserve_rad)
        elif unit == "m":
            reserves.append(profile.config.tracking_reserve_slide_m)
        else:
            raise ValueError(f"unsupported controlled-coordinate unit: {unit!r}")
    return np.asarray(reserves, dtype=np.float64)


def _point_margins(
    model: Any, data: Any, profile: PandaGeometryProfile, q: np.ndarray, work: dict[str, int]
) -> tuple[dict[str, float], list[CollisionRecord], list[str]]:
    _set_q(model, data, profile, q)
    work["configurations"] += 1
    margins, violations, unknown = {}, [], []
    contacts_by_pair: dict[tuple[int, int], list[float]] = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        contacts_by_pair.setdefault(_pair(int(contact.geom1), int(contact.geom2)), []).append(float(contact.dist))
    for spec in profile.pair_specs:
        try:
            distance = _distance(
                model, data, spec, profile.config.distance_query_max,
                contacts_by_pair, profile.config.separator_min_pad_m,
                profile.config.mesh_separator_axes,
            )
            work["distance_queries"] += 1
        except _UnknownDistance as exc:
            unknown.append(str(exc))
            continue
        physical_margin = distance - spec.threshold
        reserve = float(np.dot(np.asarray(spec.sensitivity), _tracking_reserves(profile)))
        margin = physical_margin - reserve
        margins[spec.key] = margin
        if physical_margin <= 0:
            violations.append(_collision_record(model, spec, distance))
        elif margin <= 0:
            unknown.append(f"{spec.key}:tracking_reserve_exhausted")
    return margins, violations, unknown


def _validate_trajectory(
    model: Any, data: Any, profile: PandaGeometryProfile, qpos_samples: Any,
    snapshot: SceneSnapshot, action_bytes: bytes,
) -> tuple[np.ndarray, str, str, float, float]:
    if not isinstance(action_bytes, bytes):
        raise TypeError("action_bytes must be exact native action bytes")
    q_original = np.asarray(qpos_samples)
    width = len(profile.moving_qpos_indices)
    if q_original.ndim != 2 or q_original.shape[1] != width or q_original.shape[0] < 1:
        raise ValueError(
            f"qpos_samples must have shape (N, {width}) in profile moving-coordinate order"
        )
    if q_original.dtype.kind != "f" or not np.all(np.isfinite(q_original)):
        raise ValueError("qpos_samples must be a finite floating-point array")
    q = np.asarray(q_original, dtype=np.float64)
    live_scene, live_state = _live_digests(model, data, profile, snapshot.external_sha256)
    if _model_digest(model) != profile.model_digest or snapshot.model_digest != profile.model_digest:
        raise ValueError("model/profile/snapshot identity mismatch")
    if live_scene != snapshot.digest or live_state != snapshot.state_digest:
        raise ValueError("live simulator scene/state no longer matches the supplied snapshot")
    live_q = np.asarray(data.qpos[list(profile.moving_qpos_indices)], dtype=np.float64)
    if not np.array_equal(q[0], live_q):
        raise ValueError("qpos_samples[0] must exactly equal the live controlled moving coordinates")
    low, high = np.asarray(profile.joint_lower), np.asarray(profile.joint_upper)
    per_coordinate = np.minimum(np.min(q - low, axis=0), np.min(high - q, axis=0))
    hinge_margin = float(np.min(per_coordinate[:7]))
    slide_margin = float(np.min(per_coordinate[7:])) if len(per_coordinate) > 7 else math.inf
    return q, _sha(_array_bytes(q_original)), live_state, hinge_margin, slide_margin


def _certificate(
    *, profile: PandaGeometryProfile, q: np.ndarray, trajectory_sha: str,
    trajectory_dtype: str, action_bytes: bytes, snapshot: SceneSnapshot,
    state_digest: str, pair_margins: Mapping[str, float], hinge_limit_margin: float,
    slide_limit_margin: float,
    parent_digest: str | None,
) -> GeometryCertificate:
    trajectory = tuple(tuple(float(value) for value in row) for row in q)
    margins = tuple(sorted((key, float(value)) for key, value in pair_margins.items()))
    payload = {
        "action_sha256": _sha(action_bytes), "action_nbytes": len(action_bytes),
        "trajectory_sha256": trajectory_sha, "trajectory_dtype": trajectory_dtype,
        "trajectory_shape": tuple(q.shape), "trajectory": trajectory,
        "coordinate_units": profile.coordinate_units,
        "scene_digest": snapshot.digest, "state_digest": state_digest,
        "model_digest": profile.model_digest, "config_digest": profile.config.digest,
        "profile_digest": profile.digest, "scope": CONTINUOUS_SCOPE,
        "pair_margins": margins,
        "min_margin": min(pair_margins.values(), default=math.inf),
        "min_joint_limit_margin_rad": hinge_limit_margin,
        "min_slide_limit_margin_m": slide_limit_margin, "parent_digest": parent_digest,
    }
    return GeometryCertificate(digest=_json_sha(payload), **payload)


def _full_continuous(
    model: Any, source_data: Any, profile: PandaGeometryProfile, q: np.ndarray,
    work: dict[str, int],
) -> tuple[str, dict[str, float], tuple[CollisionRecord, ...], tuple[str, ...]]:
    data = _copy_data(model, source_data)
    best = {spec.key: math.inf for spec in profile.pair_specs}
    violations: list[CollisionRecord] = []
    unknown: list[str] = []

    def check_segment(left: np.ndarray, right: np.ndarray, depth: int) -> bool:
        work["segments"] += 1
        midpoint = (left + right) * 0.5
        margins, bad, missing = _point_margins(model, data, profile, midpoint, work)
        violations.extend(bad)
        unknown.extend(missing)
        if bad or missing:
            for key, value in margins.items():
                best[key] = min(best[key], value)
            return False
        half_delta = np.abs(right - left) * 0.5
        lower_bounds: dict[str, float] = {}
        unresolved: list[str] = []
        for spec in profile.pair_specs:
            lower = margins[spec.key] - float(np.dot(np.asarray(spec.sensitivity), half_delta))
            lower_bounds[spec.key] = lower
            if lower <= 0:
                unresolved.append(spec.key)
        if not unresolved:
            for key, lower in lower_bounds.items():
                best[key] = min(best[key], lower)
            return True
        if depth >= profile.config.max_subdivision_depth:
            unknown.extend(f"{key}:subdivision_limit:depth={depth}" for key in unresolved)
            return False
        work["subdivisions"] += 1
        return check_segment(left, midpoint, depth + 1) and check_segment(midpoint, right, depth + 1)

    def check_window(left_index: int, right_index: int) -> bool:
        if right_index - left_index == 1:
            return check_segment(q[left_index], q[right_index], 0)
        work["windows"] += 1
        window = q[left_index:right_index + 1]
        low, high = np.min(window, axis=0), np.max(window, axis=0)
        margins, bad, missing = _point_margins(model, data, profile, (low + high) * 0.5, work)
        half_span = (high - low) * 0.5
        lower_bounds = {
            spec.key: margins.get(spec.key, -math.inf)
            - float(np.dot(np.asarray(spec.sensitivity), half_span))
            for spec in profile.pair_specs
        }
        if not bad and not missing and all(value > 0 for value in lower_bounds.values()):
            for key, lower in lower_bounds.items():
                best[key] = min(best[key], lower)
            return True
        work["window_splits"] += 1
        middle = (left_index + right_index) // 2
        return check_window(left_index, middle) and check_window(middle, right_index)

    if len(q) == 1:
        margins, bad, missing = _point_margins(model, data, profile, q[0], work)
        best.update(margins)
        violations.extend(bad)
        unknown.extend(missing)
    else:
        check_window(0, len(q) - 1)
    if violations:
        return "collision", best, tuple(violations), tuple(sorted(set(unknown)))
    if unknown:
        return "unknown", best, (), tuple(sorted(set(unknown)))
    return "certified", best, (), ()


def _incremental(
    profile: PandaGeometryProfile, q: np.ndarray, parent: GeometryCertificate,
    snapshot: SceneSnapshot, state_digest: str, work: dict[str, int],
) -> tuple[str | None, dict[str, float] | None]:
    required = (
        parent.scope == CONTINUOUS_SCOPE
        and parent.model_digest == profile.model_digest
        and parent.config_digest == profile.config.digest
        and parent.profile_digest == profile.digest
        and parent.scene_digest == snapshot.digest
        and parent.state_digest == state_digest
    )
    if not required:
        return None, None
    parent_q = np.asarray(parent.trajectory, dtype=np.float64)
    work["incremental_comparisons"] += 1
    if parent_q.shape != q.shape:
        return None, None
    if np.array_equal(parent_q, q):
        return "incremental_exact", dict(parent.pair_margins)
    if any(rule.max_penetration is not None for rule in profile.allowed_contacts):
        return None, None
    deviation = np.max(np.abs(q - parent_q), axis=0)
    parent_margins = dict(parent.pair_margins)
    inherited: dict[str, float] = {}
    for spec in profile.pair_specs:
        margin = parent_margins.get(spec.key)
        if margin is None:
            return None, None
        child_margin = float(margin - np.dot(np.asarray(spec.sensitivity), deviation))
        if child_margin <= 0:
            return None, None
        inherited[spec.key] = child_margin
    return "incremental_clearance", inherited


def certify_trajectory(
    model: Any,
    data: Any,
    profile: PandaGeometryProfile,
    qpos_samples: Any,
    *,
    action_bytes: bytes,
    scene_snapshot: SceneSnapshot,
    parent: GeometryCertificate | None = None,
) -> GeometryDecision:
    """Check a forecast and optionally reuse a sound parent certificate."""
    started = time.perf_counter_ns()
    work = {"configurations": 0, "distance_queries": 0, "segments": 0,
            "subdivisions": 0, "windows": 0, "window_splits": 0,
            "incremental_comparisons": 0, "full_fallbacks": 0}
    try:
        q_original = np.asarray(qpos_samples)
        q, trajectory_sha, state_digest, hinge_nominal, slide_nominal = _validate_trajectory(
            model, data, profile, q_original, scene_snapshot, action_bytes,
        )
    except (TypeError, ValueError) as exc:
        return GeometryDecision(False, "invalid", "full", "none", None, None, (), (), work,
                                time.perf_counter_ns() - started, None, str(exc))
    if hinge_nominal < 0 or slide_nominal < 0:
        return GeometryDecision(False, "collision", "full", CONTINUOUS_SCOPE, None,
                                hinge_nominal, (), (), work, time.perf_counter_ns() - started,
                                None, "joint_limit", slide_nominal)
    hinge_limit = hinge_nominal - profile.config.tracking_reserve_rad
    slide_limit = slide_nominal - profile.config.tracking_reserve_slide_m
    if hinge_limit < 0 or slide_limit < 0:
        exhausted = tuple(
            label for value, label in (
                (hinge_limit, "hinge_joint_limit:tracking_reserve_exhausted"),
                (slide_limit, "slide_joint_limit:tracking_reserve_exhausted"),
            ) if value < 0
        )
        return GeometryDecision(False, "unknown", "full", CONTINUOUS_SCOPE, None,
                                hinge_limit, exhausted, (), work,
                                time.perf_counter_ns() - started, None,
                                "tracking reserve reaches joint limit", slide_limit)

    if not profile.config.continuous:
        return dense_review(model, data, profile, q_original, action_bytes=action_bytes,
                            scene_snapshot=scene_snapshot,
                            max_joint_step_rad=profile.config.max_joint_step_rad,
                            max_slide_step_m=profile.config.max_slide_step_m)

    method, margins = (None, None)
    if parent is not None:
        method, margins = _incremental(profile, q, parent, scene_snapshot, state_digest, work)
    if method is None:
        method = "fallback_full" if parent is not None else "full"
        if parent is not None:
            work["full_fallbacks"] += 1
        status, margins, violations, unknown = _full_continuous(model, data, profile, q, work)
        if status != "certified":
            return GeometryDecision(False, status, method, CONTINUOUS_SCOPE,
                                    min(margins.values(), default=None), hinge_limit, unknown,
                                    violations, work, time.perf_counter_ns() - started, None,
                                    "collision" if status == "collision" else "continuous clearance not proven",
                                    slide_limit)
    assert margins is not None
    cert = _certificate(
        profile=profile, q=q, trajectory_sha=trajectory_sha,
        trajectory_dtype=str(q_original.dtype), action_bytes=action_bytes,
        snapshot=scene_snapshot, state_digest=state_digest, pair_margins=margins,
        hinge_limit_margin=hinge_limit, slide_limit_margin=slide_limit,
        parent_digest=parent.digest if parent is not None else None,
    )
    return GeometryDecision(True, "certified", method, CONTINUOUS_SCOPE, cert.min_margin,
                            hinge_limit, (), (), work, time.perf_counter_ns() - started, cert, "ok",
                            slide_limit)


def dense_review(
    model: Any,
    data: Any,
    profile: PandaGeometryProfile,
    qpos_samples: Any,
    *,
    action_bytes: bytes,
    scene_snapshot: SceneSnapshot,
    max_joint_step_rad: float = 0.0025,
    max_slide_step_m: float | None = None,
) -> GeometryDecision:
    """Finite-resolution diagnostic; never returns a continuous certificate."""
    started = time.perf_counter_ns()
    work = {"configurations": 0, "distance_queries": 0, "segments": 0,
            "subdivisions": 0, "windows": 0, "window_splits": 0,
            "incremental_comparisons": 0, "full_fallbacks": 0}
    slide_step = profile.config.max_slide_step_m if max_slide_step_m is None else max_slide_step_m
    scope = (
        "finite_resolution_joint_linear_frozen_scene_"
        f"max_joint_step_rad={max_joint_step_rad:.17g}_max_slide_step_m={slide_step:.17g}"
    )
    if (not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0
            or not math.isfinite(slide_step) or slide_step <= 0):
        return GeometryDecision(False, "invalid", "dense_review", scope, None, None, (), (), work,
                                time.perf_counter_ns() - started, None, "invalid review step")
    try:
        original = np.asarray(qpos_samples)
        q, _, _, hinge_nominal, slide_nominal = _validate_trajectory(
            model, data, profile, original, scene_snapshot, action_bytes,
        )
    except (TypeError, ValueError) as exc:
        return GeometryDecision(False, "invalid", "dense_review", scope, None, None, (), (), work,
                                time.perf_counter_ns() - started, None, str(exc))
    if hinge_nominal < 0 or slide_nominal < 0:
        return GeometryDecision(False, "collision", "dense_review", scope, None, hinge_nominal,
                                (), (), work, time.perf_counter_ns() - started, None, "joint_limit",
                                slide_nominal)
    hinge_limit = hinge_nominal - profile.config.tracking_reserve_rad
    slide_limit = slide_nominal - profile.config.tracking_reserve_slide_m
    if hinge_limit < 0 or slide_limit < 0:
        exhausted = tuple(
            label for value, label in (
                (hinge_limit, "hinge_joint_limit:tracking_reserve_exhausted"),
                (slide_limit, "slide_joint_limit:tracking_reserve_exhausted"),
            ) if value < 0
        )
        return GeometryDecision(False, "unknown", "dense_review", scope, None, hinge_limit,
                                exhausted, (), work,
                                time.perf_counter_ns() - started, None,
                                "tracking reserve reaches joint limit", slide_limit)
    points = [q[0]]
    step_limits = np.asarray(
        [max_joint_step_rad] * 7 + [slide_step] * len(profile.controlled_extra_joint_ids),
        dtype=np.float64,
    )
    for index in range(1, len(q)):
        delta = q[index] - q[index - 1]
        count = max(1, int(math.ceil(float(np.max(np.abs(delta) / step_limits)))))
        points.extend(q[index - 1] + delta * (step / count) for step in range(1, count + 1))
    review_data = _copy_data(model, data)
    best = {spec.key: math.inf for spec in profile.pair_specs}
    violations: list[CollisionRecord] = []
    unknown: list[str] = []
    for point in points:
        margins, bad, missing = _point_margins(model, review_data, profile, point, work)
        for key, value in margins.items():
            best[key] = min(best[key], value)
        violations.extend(bad)
        unknown.extend(missing)
        if bad or missing:
            break
    status = "collision" if violations else "unknown" if unknown else "finite_clear"
    return GeometryDecision(status == "finite_clear", status, "dense_review", scope,
                            min(best.values(), default=None), hinge_limit,
                            tuple(sorted(set(unknown))), tuple(violations), work,
                            time.perf_counter_ns() - started, None,
                            "point samples only; not a continuous certificate", slide_limit)


def compare_execution(
    certificate: GeometryCertificate, executed_qpos_samples: Any
) -> ExecutionTrackingReport:
    """Bind and compare an executed controlled-coordinate trace to its forecast.

    This reports tracking error only. It does not retroactively certify an
    execution or replace checking live unwanted contacts at every env step.
    """
    executed_original = np.asarray(executed_qpos_samples)
    executed_sha = _sha(_array_bytes(executed_original))
    shape = tuple(int(value) for value in executed_original.shape)
    units = certificate.coordinate_units
    expected_width = certificate.trajectory_shape[1]
    if len(units) != expected_width or units[:7] != ("rad",) * 7 or any(
        unit != "m" for unit in units[7:]
    ):
        return ExecutionTrackingReport(
            certificate.digest, executed_sha, str(executed_original.dtype), shape, False,
            None, None, None, (), "certificate has an invalid controlled-coordinate unit layout",
        )
    if (
        executed_original.dtype.kind != "f"
        or executed_original.ndim != 2
        or executed_original.shape[1] != expected_width
        or not np.all(np.isfinite(executed_original))
    ):
        return ExecutionTrackingReport(
            certificate.digest, executed_sha, str(executed_original.dtype), shape, False,
            None, None, None, (),
            f"executed trace must be a finite floating (N, {expected_width}) array",
        )
    forecast = np.asarray(certificate.trajectory, dtype=np.float64)
    executed = np.asarray(executed_original, dtype=np.float64)
    if executed.shape != forecast.shape:
        return ExecutionTrackingReport(
            certificate.digest, executed_sha, str(executed_original.dtype), shape, False,
            None, None, None, (), "executed trace and certified forecast shapes differ",
        )
    error = executed - forecast
    hinge = error[:, [i for i, unit in enumerate(units) if unit == "rad"]]
    slide = error[:, [i for i, unit in enumerate(units) if unit == "m"]]
    per_joint = tuple(float(value) for value in np.max(np.abs(hinge), axis=0))
    per_slide = tuple(float(value) for value in np.max(np.abs(slide), axis=0))
    return ExecutionTrackingReport(
        certificate.digest, executed_sha, str(executed_original.dtype), shape, True,
        float(np.sqrt(np.mean(np.square(hinge)))), float(np.max(np.abs(hinge))),
        float(np.max(np.abs(hinge[-1]))), per_joint,
        "tracking comparison only; inspect live contacts and dynamics separately",
        float(np.sqrt(np.mean(np.square(slide)))) if slide.size else None,
        float(np.max(np.abs(slide))) if slide.size else None,
        float(np.max(np.abs(slide[-1]))) if slide.size else None,
        per_slide,
    )


__all__ = [
    "CONTINUOUS_SCOPE", "AllowedContact", "CollisionRecord", "ExecutionTrackingReport",
    "GeometryCertificate", "GeometryConfig",
    "GeometryDecision", "PandaGeometryProfile", "SceneSnapshot",
    "capture_scene_snapshot", "certify_trajectory", "compare_execution",
    "current_unwanted_collisions", "dense_review",
]
