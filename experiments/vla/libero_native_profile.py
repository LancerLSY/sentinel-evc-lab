"""Frozen support declaration for the native LIBERO/Panda action request.

The limits are the native relative-control request envelope accepted by this
product experiment.  They are not robot joint, collision, force or human
protection limits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinel_evc.native_gateway import NativeActionProfile


DEFAULT_PROFILE = {
    "schema_id": "libero-panda-relative-7d-v1",
    "component_names": [
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "gripper",
    ],
    "lower": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "semantics": (
        "Official LeRobot LIBERO postprocessor output for Panda relative control: "
        "3 translation deltas, 3 axis-angle rotation deltas, and one gripper request. "
        "Bounds are the declared native request envelope, not physical safety limits."
    ),
    "request_shape": [1, 7],
    "request_dtypes": ["float32", "float64"],
    "unsupported_checks": [
        "collision_clearance",
        "self_collision",
        "joint_limits_after_controller_mapping",
        "dynamics",
        "contact_force",
        "physical_stop",
        "worldguard_prediction",
    ],
}


def load_profile(path: Path | None = None) -> NativeActionProfile:
    payload: dict[str, Any] = DEFAULT_PROFILE
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("native profile JSON must be an object")
        allowed = {
            "schema_id",
            "component_names",
            "lower",
            "upper",
            "semantics",
            "request_shape",
            "request_dtypes",
            "unsupported_checks",
        }
        unknown = sorted(set(loaded) - allowed)
        if unknown:
            raise ValueError(f"unknown native profile fields: {unknown}")
        payload = {**DEFAULT_PROFILE, **loaded}
    return NativeActionProfile(
        schema_id=payload["schema_id"],
        component_names=tuple(payload["component_names"]),
        lower=tuple(payload["lower"]),
        upper=tuple(payload["upper"]),
        semantics=payload["semantics"],
        request_shape=tuple(payload["request_shape"]),
        request_dtypes=tuple(payload["request_dtypes"]),
        unsupported_checks=tuple(payload["unsupported_checks"]),
    )


__all__ = ["DEFAULT_PROFILE", "load_profile"]
