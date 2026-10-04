"""Prospective fixed single-step Cartesian recovery candidates for LIBERO.

This module only proposes, forecasts, and fully certifies candidates. It never
authorizes or writes an action and does not inspect task reward or goal state.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from panda_geometry import certify_trajectory


NATIVE_WIDTH = 7
MOVING_WIDTH = 9
PHYSICS_SUBSTEPS = 25
_DEFAULT_LOWER = (-1.0,) * NATIVE_WIDTH
_DEFAULT_UPPER = (1.0,) * NATIVE_WIDTH


@dataclass(frozen=True)
class RecoveryAttempt:
    name: str
    native_values: tuple[tuple[tuple[float, ...], ...], ...]
    native_shape: tuple[int, ...]
    native_dtype: str
    action_bytes_sha256: str
    action_bytes_hex: str
    l2_to_original_native: float
    configuration: dict[str, Any]
    forecast_restore: Any
    forecast_qpos_shape: tuple[int, ...]
    forecast_qpos_sha256: str
    decision: Any
    geometry_certified: bool
    contact_veto_enabled: bool
    contact_vetoed: bool
    unwanted_contact_substeps: int
    unwanted_contact_records: int
    unwanted_contact_evidence: tuple[dict[str, Any], ...]
    certified: bool
    forecast_latency_ns: int
    certification_latency_ns: int
    total_latency_ns: int


@dataclass(frozen=True)
class CartesianRecoveryResult:
    native: np.ndarray | None
    qpos: np.ndarray | None
    restore: Any | None
    decision: Any | None
    selected_name: str | None
    attempts: tuple[RecoveryAttempt, ...]
    exclusions: tuple[dict[str, Any], ...]
    total_latency_ns: int


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = (
        str(contiguous.dtype).encode()
        + b"\0"
        + str(tuple(contiguous.shape)).encode()
        + b"\0"
        + contiguous.tobytes(order="C")
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _native_action(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size != NATIVE_WIDTH:
        raise ValueError(f"original_native must contain exactly seven values, found shape {array.shape}")
    array = np.ascontiguousarray(array.reshape(1, 1, NATIVE_WIDTH), dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("original_native must be finite")
    return array


def _candidate(
    original: np.ndarray, *, translation_scale: float, translation_mask: tuple[float, float, float],
) -> np.ndarray:
    candidate = original.copy()
    mask = np.asarray(translation_mask, dtype=np.float32).reshape(1, 1, 3)
    candidate[..., :3] = original[..., :3] * np.float32(translation_scale) * mask
    candidate[..., 3:6] = np.float32(0.0)
    candidate[..., 6] = original[..., 6]
    return np.ascontiguousarray(candidate, dtype=np.float32)


def _contact_evidence(restore: Any, *, required: bool) -> tuple[int, int, tuple[dict[str, Any], ...]]:
    if not isinstance(restore, Mapping) or "substep_unwanted_contacts" not in restore:
        if required:
            raise RuntimeError("contact veto requires forecast restore metadata with substep_unwanted_contacts")
        return 0, 0, ()
    contacts = restore["substep_unwanted_contacts"]
    if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
        raise RuntimeError("forecast substep_unwanted_contacts must be a sequence")
    if len(contacts) != PHYSICS_SUBSTEPS:
        raise RuntimeError(
            f"single-step forecast contact evidence must contain {PHYSICS_SUBSTEPS} physics substeps, found {len(contacts)}"
        )
    evidence: list[dict[str, Any]] = []
    records = 0
    for index, row in enumerate(contacts):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise RuntimeError(f"forecast contact evidence at physics substep {index} must be a sequence")
        if row:
            copied = tuple(row)
            records += len(copied)
            evidence.append({"physics_substep": index, "contacts": copied})
    return len(evidence), records, tuple(evidence)


def recover_cartesian_motion(
    original_native: Any,
    *,
    forecast: Callable[[np.ndarray], tuple[Any, Any]],
    model: Any,
    data: Any,
    profile: Any,
    scene_snapshot: Any,
    native_lower: Any = _DEFAULT_LOWER,
    native_upper: Any = _DEFAULT_UPPER,
    veto_unwanted_contacts: bool = False,
) -> CartesianRecoveryResult:
    """Return the closest fixed certified repair, or an empty result.

    Candidate proximity is only a deterministic ordering rule. It is not a
    reward, task-progress, or safety score. Any forecast/restore exception is
    deliberately propagated to the caller.
    """
    started = time.monotonic_ns()
    original = _native_action(original_native)
    lower = np.asarray(native_lower, dtype=np.float32).reshape(1, 1, NATIVE_WIDTH)
    upper = np.asarray(native_upper, dtype=np.float32).reshape(1, 1, NATIVE_WIDTH)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("native bounds must be finite ordered seven-component arrays")
    if np.any(original < lower) or np.any(original > upper):
        raise ValueError("original_native is outside the unchanged native bounds")

    definitions = (
        ("xyz_keep_rot_zero", 1.0, (1.0, 1.0, 1.0)),
        ("xyz_half_rot_zero", 0.5, (1.0, 1.0, 1.0)),
        ("x_only_rot_zero", 1.0, (1.0, 0.0, 0.0)),
        ("y_only_rot_zero", 1.0, (0.0, 1.0, 0.0)),
        ("z_only_rot_zero", 1.0, (0.0, 0.0, 1.0)),
        ("xyz_retreat_half_rot_zero", -0.5, (1.0, 1.0, 1.0)),
    )
    generated: list[tuple[int, str, np.ndarray, dict[str, Any], float]] = []
    exclusions: list[dict[str, Any]] = []
    seen: dict[bytes, str] = {}
    original_bytes = original.tobytes(order="C")
    for order, (name, scale, mask) in enumerate(definitions):
        candidate = _candidate(original, translation_scale=scale, translation_mask=mask)
        configuration = {
            "translation_scale": scale,
            "translation_mask_xyz": list(mask),
            "rotation_transform": "set_axis_angle_xyz_to_zero",
            "gripper_transform": "preserve_original_exactly",
            "native_bounds_transform": "none",
        }
        action_bytes = candidate.tobytes(order="C")
        if action_bytes == original_bytes:
            exclusions.append({"name": name, "reason": "exact_duplicate_of_original", "configuration": configuration})
            continue
        if np.count_nonzero(candidate[..., :6]) == 0:
            exclusions.append({"name": name, "reason": "zero_motion_hold_excluded", "configuration": configuration})
            continue
        if action_bytes in seen:
            exclusions.append({"name": name, "reason": "exact_duplicate_of_generated_candidate",
                               "duplicate_of": seen[action_bytes], "configuration": configuration})
            continue
        if np.any(candidate < lower) or np.any(candidate > upper):
            raise RuntimeError(f"fixed recovery candidate {name} exceeded unchanged native bounds")
        seen[action_bytes] = name
        distance = float(np.linalg.norm(candidate.astype(np.float64) - original.astype(np.float64)))
        generated.append((order, name, candidate, configuration, distance))

    generated.sort(key=lambda item: (item[4], item[0]))
    attempts: list[RecoveryAttempt] = []
    for _, name, candidate, configuration, distance in generated:
        attempt_started = time.monotonic_ns()
        action_bytes = candidate.tobytes(order="C")
        forecast_started = time.monotonic_ns()
        qpos_value, restore = forecast(candidate.copy())
        forecast_latency = time.monotonic_ns() - forecast_started
        qpos = np.ascontiguousarray(np.asarray(qpos_value))
        expected_shape = (PHYSICS_SUBSTEPS + 1, MOVING_WIDTH)
        if qpos.shape != expected_shape or not np.issubdtype(qpos.dtype, np.floating) or not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"recovery forecast must return finite floating qpos shape {expected_shape}, found {qpos.shape}")
        certification_started = time.monotonic_ns()
        decision = certify_trajectory(
            model, data, profile, qpos, action_bytes=action_bytes, scene_snapshot=scene_snapshot,
        )
        certification_latency = time.monotonic_ns() - certification_started
        contact_substeps, contact_records, contact_evidence = (
            _contact_evidence(restore, required=True) if veto_unwanted_contacts else (0, 0, ())
        )
        geometry_certified = bool(getattr(decision, "allowed", False))
        contact_vetoed = bool(veto_unwanted_contacts and contact_substeps)
        certified = bool(geometry_certified and not contact_vetoed)
        attempt = RecoveryAttempt(
            name=name,
            native_values=tuple(tuple(tuple(float(value) for value in action) for action in chunk) for chunk in candidate),
            native_shape=tuple(candidate.shape),
            native_dtype=str(candidate.dtype),
            action_bytes_sha256="sha256:" + hashlib.sha256(action_bytes).hexdigest(),
            action_bytes_hex=action_bytes.hex(),
            l2_to_original_native=distance,
            configuration=configuration,
            forecast_restore=restore,
            forecast_qpos_shape=tuple(qpos.shape),
            forecast_qpos_sha256=_array_sha256(qpos),
            decision=decision,
            geometry_certified=geometry_certified,
            contact_veto_enabled=bool(veto_unwanted_contacts),
            contact_vetoed=contact_vetoed,
            unwanted_contact_substeps=contact_substeps,
            unwanted_contact_records=contact_records,
            unwanted_contact_evidence=contact_evidence,
            certified=certified,
            forecast_latency_ns=forecast_latency,
            certification_latency_ns=certification_latency,
            total_latency_ns=time.monotonic_ns() - attempt_started,
        )
        attempts.append(attempt)
        if certified:
            return CartesianRecoveryResult(
                native=candidate.copy(), qpos=qpos.copy(), restore=restore, decision=decision,
                selected_name=name, attempts=tuple(attempts), exclusions=tuple(exclusions),
                total_latency_ns=time.monotonic_ns() - started,
            )
    return CartesianRecoveryResult(
        native=None, qpos=None, restore=None, decision=None, selected_name=None,
        attempts=tuple(attempts), exclusions=tuple(exclusions), total_latency_ns=time.monotonic_ns() - started,
    )


__all__ = ["CartesianRecoveryResult", "RecoveryAttempt", "recover_cartesian_motion"]
