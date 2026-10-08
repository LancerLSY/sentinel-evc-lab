#!/usr/bin/env python3
"""Run the preregistered SmolVLA/LIBERO Panda geometry + EVC experiment.

This is an experimental simulator runner.  Geometry certificates cover the
declared MuJoCo model/profile and predicted simulator rollout only; they are
not continuous physical-safety or functional-safety claims.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentinel_evc.events import EventLog
from sentinel_evc.native_gateway import NativeContext, NativeGateway, NativeGatewayDenied, NativeSnapshot

from libero_native_profile import load_profile
from panda_geometry import (
    AllowedContact,
    GeometryConfig,
    PandaGeometryProfile,
    capture_scene_snapshot,
    certify_trajectory,
    current_unwanted_collisions,
    dense_review,
    compare_execution,
)


BRANCHES = ("parent_only", "full_final", "delta_evc")
TRANSFORM = {
    "name": "fixed_overlap_temporal_aggregation_v1",
    "status": "optional_preregistered_deployment_transform",
    "newest_weight": 0.5,
    "existing_unexecuted_weight": 0.5,
    "alignment": "new[0:10] with previous[10:20] after ten executed actions",
    "claim_boundary": "not the default SmolVLA runtime temporal aggregation",
}


def _selected_transform(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("_product_profile") and config.get("_product_aggregation") == "native":
        return {
            "name": "native_first_ten_v1", "status": "product_default",
            "operation": "execute the official postprocessed first ten actions without overlap averaging",
            "claim_boundary": "preserves the selected native action chunk; geometry checks, permits and recovery remain active",
        }
    return TRANSFORM


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, collections.deque)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        identity = _file_digest(path)
        size = path.stat().st_size
        digest.update(f"{path.relative_to(root).as_posix()}\0{identity}\0{size}\n".encode())
        count += 1
        total += size
    return {"files": count, "bytes": total, "sha256": "sha256:" + digest.hexdigest()}


def _array(value: Any, *, dtype: Any | None = None) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _array_identity(value: Any) -> dict[str, Any]:
    array = _array(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _digest_bytes(str(array.dtype).encode() + b"\0" + _canonical(list(array.shape)) + b"\0" + array.tobytes()),
    }


def _observation_hash(observation: Any) -> str:
    def identities(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): identities(item) for key, item in sorted(value.items())}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return _array_identity(value)

    return _digest_bytes(_canonical(identities(observation)))


def _batch_native(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {key: _batch_native(item) for key, item in value.items()}
    return np.expand_dims(value, 0)


def _percentiles_ns(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))] / 1e6 if ordered else None
    return {"count": len(ordered), "p50_ms": pick(0.50), "p95_ms": pick(0.95)}


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self.handle.write(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


class ProductSessionStopped(RuntimeError):
    """A managed simulator stop observed immediately before writer entry."""


def _deadline(config: dict[str, Any]) -> float | None:
    raw = config.get("deadline_timestamp")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def _safe_leaf(value: Any) -> bool:
    import numpy as np

    if isinstance(value, (str, int, float, bool, type(None), np.ndarray, np.generic)):
        return True
    if isinstance(value, (list, tuple, collections.deque)):
        return all(_safe_leaf(item) for item in value)
    if isinstance(value, dict):
        return all(_safe_leaf(key) and _safe_leaf(item) for key, item in value.items())
    return False


def _stateful_objects(underlying: Any) -> list[Any]:
    """Discover mutable Python state below one native LIBERO environment.

    The live forecast advances controller goals, observable buffers and wrapper
    counters in addition to MjData.  Traverse those package-owned objects and
    restore their scalar/array/container leaves in-place; structural object
    references themselves are never replaced.
    """
    values: list[Any] = [underlying]
    result: list[Any] = []
    seen: set[int] = set()

    def children(value: Any) -> list[Any]:
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, (list, tuple, collections.deque)):
            return list(value)
        return [value]

    while values:
        value = values.pop()
        if value is None or id(value) in seen:
            continue
        if isinstance(value, (dict, list, tuple, collections.deque)):
            seen.add(id(value))
            values.extend(children(value))
            continue
        if not hasattr(value, "__dict__"):
            continue
        module = type(value).__module__
        if value is not underlying and not module.startswith(("lerobot", "libero", "robosuite", "gymnasium")):
            continue
        seen.add(id(value))
        result.append(value)
        for child in value.__dict__.values():
            if _safe_leaf(child):
                continue
            values.extend(children(child))
    return result


def _rollout_state_inventory(underlying: Any, objects: list[Any]) -> dict[str, Any]:
    """Require real wrapper and controller coverage before forecasting."""
    covered = {id(obj) for obj in objects}
    native = underlying._env
    robots = getattr(native, "robots", ())
    controllers = [robot.controller for robot in robots]
    if not controllers or id(underlying) not in covered or id(native) not in covered:
        raise RuntimeError("rollout snapshot did not reach the native environment and robot controllers")
    if any(id(controller) not in covered for controller in controllers):
        raise RuntimeError("rollout snapshot omitted a live robot controller")
    for controller in controllers:
        for name in ("interpolator_pos", "interpolator_ori"):
            interpolator = getattr(controller, name, None)
            if interpolator is not None and id(interpolator) not in covered:
                raise RuntimeError(f"rollout snapshot omitted controller {name}")
    observable_count = sum(type(obj).__module__.startswith("robosuite.utils.observables") for obj in objects)
    if not observable_count:
        raise RuntimeError("rollout snapshot omitted native observable buffers")
    entries = [{
        "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "leaf_keys": sorted(key for key, value in obj.__dict__.items() if _safe_leaf(value)),
    } for obj in objects]
    return {
        "object_count": len(entries), "controller_count": len(controllers),
        "observable_count": observable_count, "objects": entries,
    }


def _mujoco_copy_data_api() -> tuple[Any, Any]:
    """Load the pinned MuJoCo 3.3.7 mj_copyData C API exactly once."""
    import ctypes
    import mujoco

    cached = getattr(_mujoco_copy_data_api, "_cached", None)
    if cached is not None:
        return cached
    if getattr(mujoco, "__version__", None) != "3.3.7":
        raise RuntimeError(f"full MjData restore requires pinned mujoco==3.3.7, found {mujoco.__version__!r}")
    libraries = sorted(Path(mujoco.__file__).resolve().parent.glob("libmujoco.so*"))
    if len(libraries) != 1:
        raise RuntimeError(f"expected exactly one bundled libmujoco.so, found {[str(item) for item in libraries]}")
    library = ctypes.CDLL(str(libraries[0]))
    function = library.mj_copyData
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_void_p
    cached = (library, function)
    setattr(_mujoco_copy_data_api, "_cached", cached)
    return cached


def _copy_mujoco_data(target: Any, model: Any, source: Any) -> None:
    import ctypes

    addresses = [getattr(item, "_address", None) for item in (target, model, source)]
    if any(not isinstance(value, int) or value <= 0 for value in addresses):
        raise RuntimeError(f"MuJoCo model/data objects lack valid native addresses: {addresses}")
    _, function = _mujoco_copy_data_api()
    returned = function(*(ctypes.c_void_p(value) for value in addresses))
    if int(returned or 0) != addresses[0]:
        raise RuntimeError("mj_copyData did not return the destination MjData address")


def _native_data_digest(data: Any) -> str:
    import numpy as np

    arrays: dict[str, Any] = {}
    for name in (
        "qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
        "mocap_pos", "mocap_quat", "userdata", "qfrc_constraint", "qfrc_bias", "qM",
    ):
        if hasattr(data, name):
            arrays[name] = _array_identity(np.asarray(getattr(data, name)))
    nefc = int(data.nefc)
    ncon = int(data.ncon)
    for name in ("efc_force", "efc_state"):
        if hasattr(data, name):
            arrays[name] = _array_identity(np.asarray(getattr(data, name))[:nefc])
    contacts: dict[str, Any] = {}
    for name in ("geom", "dist", "pos", "frame", "dim", "efc_address"):
        if hasattr(data.contact, name):
            contacts[name] = _array_identity(np.asarray(getattr(data.contact, name))[:ncon])
    return _digest_bytes(_canonical({
        "time": float(data.time), "ncon": ncon, "nefc": nefc, "arrays": arrays, "contacts": contacts,
    }))


def _capture_rollout_state(underlying: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    import mujoco

    sim = underlying._env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, integration, state_spec)
    python_state: list[tuple[Any, dict[str, Any]]] = []
    original_leaves: list[dict[str, Any]] = []
    object_rngs: list[tuple[Any, Any]] = []
    objects = _stateful_objects(underlying)
    inventory = _rollout_state_inventory(underlying, objects)
    for obj in objects:
        leaves = {key: copy.deepcopy(value) for key, value in obj.__dict__.items() if _safe_leaf(value)}
        python_state.append((obj, leaves))
        original_leaves.append({key: obj.__dict__[key] for key in leaves})
        for value in obj.__dict__.values():
            if hasattr(value, "bit_generator") and hasattr(value.bit_generator, "state"):
                object_rngs.append((value, copy.deepcopy(value.bit_generator.state)))
    return {
        "integration": integration,
        "native_data": copy.copy(data),
        "native_data_digest": _native_data_digest(data),
        "objects": python_state,
        "original_leaves": original_leaves,
        "state_inventory": inventory,
        "object_rngs": object_rngs,
        "random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.random.get_rng_state().clone(),
        "cuda_random": [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def _restore_rollout_state(underlying: Any, state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    sim = underlying._env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    _copy_mujoco_data(data, model, state["native_data"])
    def restore_leaf(current: Any, saved: Any) -> Any:
        if isinstance(current, np.ndarray) and isinstance(saved, np.ndarray) and current.shape == saved.shape:
            current[...] = saved
            return current
        if isinstance(current, list) and isinstance(saved, list):
            current[:] = copy.deepcopy(saved)
            return current
        if isinstance(current, dict) and isinstance(saved, dict):
            current.clear(); current.update(copy.deepcopy(saved))
            return current
        if isinstance(current, collections.deque) and isinstance(saved, collections.deque):
            current.clear(); current.extend(copy.deepcopy(saved))
            return current
        return copy.deepcopy(saved)

    for (obj, leaves), references in zip(state["objects"], state["original_leaves"]):
        for key, value in leaves.items():
            setattr(obj, key, restore_leaf(references[key], value))
    for generator, generator_state in state["object_rngs"]:
        generator.bit_generator.state = copy.deepcopy(generator_state)
    random.setstate(state["random"])
    np.random.set_state(state["numpy_random"])
    torch.random.set_rng_state(state["torch_random"])
    if state["cuda_random"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_random"])


def _rollout_fingerprint(underlying: Any, state: dict[str, Any] | None = None) -> str:
    import numpy as np

    if state is None:
        state = _capture_rollout_state(underlying)

    leaves: list[Any] = []
    for obj, attrs in state["objects"]:
        leaves.append({
            "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "leaves": {key: _snapshot_value_identity(value) for key, value in sorted(attrs.items())},
        })
    rng = {
        "python": state["random"],
        "numpy": state["numpy_random"],
        "torch": _array_identity(state["torch_random"]),
        "cuda": [_array_identity(item) for item in (state["cuda_random"] or [])],
        "objects": [_digest_bytes(_canonical(saved)) for _, saved in state.get("object_rngs", [])],
    }
    return _digest_bytes(
        state["integration"].tobytes()
        + str(state["native_data_digest"]).encode()
        + _canonical(leaves)
        + _canonical(state["state_inventory"])
        + _canonical(rng)
    )


def _snapshot_value_identity(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)):
        return _array_identity(value)
    if isinstance(value, dict):
        return {str(key): _snapshot_value_identity(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, collections.deque)):
        return [_snapshot_value_identity(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"float_hex": value.hex()}
    return value


def _sim_handles(underlying: Any) -> tuple[Any, Any, Any]:
    sim = underlying._env.sim
    return sim, getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def _step_with_substeps(underlying: Any, action: Any, qpos_indices: list[int], geometry_profile: Any) -> tuple[Any, list[Any], list[Any]]:
    """Run one control action while observing every robosuite physics substep."""
    import numpy as np

    sim, model, data = _sim_handles(underlying)
    original = sim.step
    had_instance_step = hasattr(sim, "__dict__") and "step" in sim.__dict__
    qpos: list[Any] = []
    contacts: list[Any] = []

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        qpos.append(np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy())
        contacts.append(_jsonable(current_unwanted_collisions(model, data, geometry_profile)))
        return result

    try:
        setattr(sim, "step", observed_step)
    except Exception as exc:
        raise RuntimeError("native sim.step cannot be instrumented for physics-substep evidence") from exc
    try:
        output = underlying.step(action)
    finally:
        if had_instance_step:
            setattr(sim, "step", original)
        else:
            delattr(sim, "step")
    if not qpos:
        raise RuntimeError("native action produced no observed physics substeps")
    return output, qpos, contacts


def _joint_indices(underlying: Any) -> tuple[list[int], list[int] | None, list[int] | None]:
    robot = underlying._env.robots[0]
    qpos = [int(item) for item in robot._ref_joint_pos_indexes]
    if len(qpos) != 7:
        raise RuntimeError(f"native Panda must expose seven arm qpos indices, found {qpos}")
    joint_ids = [int(item) for item in getattr(robot, "_ref_joint_indexes", ())] or None
    return qpos, joint_ids, None


def _native_actions(
    raw_actions: Any, postprocessor: Callable[[Any], Any], env_postprocessor: Callable[[Any], Any], device: str
) -> Any:
    import numpy as np
    import torch

    raw = _array(raw_actions)
    if raw.shape != (1, 10, 7):
        raise RuntimeError(f"candidate must have shape (1,10,7), found {raw.shape}")
    values = []
    for index in range(10):
        selected = torch.as_tensor(raw[:, index, :], device=device)
        postprocessed = postprocessor(selected)
        transition = env_postprocessor({"action": postprocessed})
        actual = _array(transition["action"], dtype=np.float32)
        if actual.shape != (1, 7) or not np.all(np.isfinite(actual)):
            raise RuntimeError(f"postprocessed native action {index} is invalid: {actual.shape}")
        values.append(actual)
    return np.stack(values, axis=1)


def _clip_native_to_profile(actions: Any, profile: Any, enabled: bool) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    original = _array(actions, dtype=np.float32)
    if not enabled:
        return original.copy(), {
            "enabled": False, "changed_components": 0, "changed_actions": 0,
            "transform": "none",
        }
    lower = np.asarray(profile.lower, dtype=np.float32).reshape(1, 1, -1)
    upper = np.asarray(profile.upper, dtype=np.float32).reshape(1, 1, -1)
    if original.shape[-1] != lower.shape[-1]:
        raise RuntimeError("native profile bounds do not match postprocessed action width")
    clipped = np.clip(original, lower, upper)
    changed = clipped != original
    return np.ascontiguousarray(clipped), {
        "enabled": True,
        "changed_components": int(np.count_nonzero(changed)),
        "changed_actions": int(np.count_nonzero(np.any(changed, axis=-1))),
        "per_component_changed": np.count_nonzero(changed, axis=(0, 1)).astype(int).tolist(),
        "lower": lower.reshape(-1).tolist(), "upper": upper.reshape(-1).tolist(),
        "transform": "componentwise_np_clip_to_declared_native_profile",
        "controller_equivalence_basis": (
            "robosuite base_controller.scale_action clips request input to [-1,1]; "
            "PandaGripper.format_action applies sign then clips gripper state"
        ),
        "claim_boundary": "optional explicit pre-gate bounding transform; native profile bounds are unchanged",
    }


def _forecast(underlying: Any, actions: Any, qpos_indices: list[int], geometry_profile: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    snapshot = _capture_rollout_state(underlying)
    before = _rollout_fingerprint(underlying, snapshot)
    before_native_data = str(snapshot["native_data_digest"])
    _, _, data = _sim_handles(underlying)
    qpos = [np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()]
    action_end_indices: list[int] = []
    substep_contacts: list[Any] = []
    error: BaseException | None = None
    try:
        for action in actions[0]:
            _, step_qpos, step_contacts = _step_with_substeps(
                underlying, np.asarray(action, dtype=np.float32), qpos_indices, geometry_profile
            )
            qpos.extend(step_qpos)
            substep_contacts.extend(step_contacts)
            action_end_indices.append(len(qpos) - 1)
    except BaseException as caught:
        error = caught
    finally:
        _restore_rollout_state(underlying, snapshot)
    after_state = _capture_rollout_state(underlying)
    after = _rollout_fingerprint(underlying, after_state)
    after_native_data = str(after_state["native_data_digest"])
    if after != before:
        differences = []
        for (obj, old), (_, new) in zip(snapshot["objects"], after_state["objects"]):
            for key in sorted(set(old) | set(new)):
                if key not in old or key not in new or _canonical(_snapshot_value_identity(old.get(key))) != _canonical(_snapshot_value_identity(new.get(key))):
                    differences.append(f"{type(obj).__module__}.{type(obj).__qualname__}.{key}")
        raise RuntimeError(f"forecast state restoration mismatch: {before} != {after}; changed leaves={differences[:20]}")
    if after_native_data != before_native_data:
        raise RuntimeError(f"forecast native MjData restoration mismatch: {before_native_data} != {after_native_data}")
    if error is not None:
        raise RuntimeError("reversible native rollout failed") from error
    return np.asarray(qpos, dtype=np.float64), {
        "before": before, "after": after, "match": after == before,
        "native_data_before": before_native_data, "native_data_after": after_native_data,
        "native_data_match": after_native_data == before_native_data,
        "state_inventory": snapshot["state_inventory"],
        "action_end_indices": action_end_indices, "physics_substeps": len(qpos) - 1,
        "substep_unwanted_contacts": substep_contacts,
    }


def _certificate_digest(decision: Any) -> str:
    certificate = getattr(decision, "certificate", None)
    if certificate is None:
        return "sha256:" + "0" * 64
    return str(certificate.digest)


def _decision_record(decision: Any) -> dict[str, Any]:
    payload = _jsonable(decision)
    if not isinstance(payload, dict):
        payload = {"repr": payload}
    payload["certificate_digest"] = _certificate_digest(decision)
    return payload


def _context_dependencies(base: str, scene: Any, geometry_profile: Any, certificate_digest: str) -> str:
    return _digest_bytes(_canonical({
        "base_dependencies": base,
        "scene_hash": str(scene.digest),
        "geometry_profile_hash": str(geometry_profile.digest),
        "geometry_certificate_hash": certificate_digest,
    }))


def _episode_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = dict(config["preflight"])
    formal = config["formal"]
    tasks = [int(item) for item in formal["task_ids"]]
    states = [int(item) for item in formal["initial_state_indices"]]
    if any(state < 0 for state in states):
        raise ValueError("formal initial_state_indices must be non-negative")
    specs = [{"phase": "preflight", "task_id": int(preflight["task_id"]), "state_index": int(preflight["initial_state_index"]), "seed": int(preflight["seed"])}]
    seed_base = int(formal["seed_base"])
    first_state = min(states) if states else 0
    for task in tasks:
        for state in states:
            specs.append({"phase": "formal", "task_id": task, "state_index": state, "seed": seed_base + state - first_state})
    return specs


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"checkpoint", "backbone", "output_dir", "preflight", "formal"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    branches = tuple(config.get("branches", BRANCHES))
    if branches != BRANCHES:
        raise ValueError(f"branches must be exactly {BRANCHES}")
    if int(config.get("prediction_chunk_size", 50)) != 50 or int(config.get("execution_horizon", 10)) != 10:
        raise ValueError("the frozen experiment requires predict 50 / execute 10")
    if int(config.get("max_episode_steps", 280)) != 280:
        raise ValueError("max_episode_steps must remain 280")
    if not isinstance(config.get("native_clip_to_profile", False), bool):
        raise ValueError("native_clip_to_profile must be boolean")
    _episode_specs(config)
    return config


def _allowed_contacts(config: dict[str, Any], model: Any, task_id: int) -> tuple[AllowedContact, ...]:
    import mujoco

    resolved = [AllowedContact(**item) for item in config.get("geometry", {}).get("allowed_contacts", [])]
    for rule in config.get("geometry", {}).get("allowed_contact_rules", []):
        if rule.get("task_ids") is not None and task_id not in [int(item) for item in rule["task_ids"]]:
            continue
        ids = []
        for key in ("geom_a_name", "geom_b_name"):
            geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(rule[key])))
            if geom_id < 0:
                raise ValueError(f"allowed contact geom name was not found: {rule[key]}")
            ids.append(geom_id)
        resolved.append(AllowedContact(ids[0], ids[1], str(rule["label"]), rule.get("max_penetration")))
    return tuple(resolved)


def _geometry_config(config: dict[str, Any]) -> GeometryConfig:
    values = {key: value for key, value in config.get("geometry", {}).items()
              if key not in {"allowed_contacts", "allowed_contact_rules"}}
    return GeometryConfig(**values)


def _model_and_source_identity(config_path: Path, checkpoint: Path, backbone: Path) -> dict[str, Any]:
    sources = {
        "runner": _file_digest(Path(__file__)),
        "panda_geometry": _file_digest(Path(__file__).with_name("panda_geometry.py")),
        "native_gateway": _file_digest(SRC_ROOT / "sentinel_evc" / "native_gateway.py"),
        "native_profile": _file_digest(Path(__file__).with_name("libero_native_profile.py")),
        "config": _file_digest(config_path),
    }
    recovery_source = Path(__file__).with_name("safe_prefix_recovery.py")
    if recovery_source.exists():
        sources["safe_prefix_recovery"] = _file_digest(recovery_source)
    motion_source = Path(__file__).with_name("cartesian_recovery.py")
    if motion_source.exists():
        sources["cartesian_recovery"] = _file_digest(motion_source)
    return {"sources": sources, "checkpoint": _tree_digest(checkpoint), "backbone": _tree_digest(backbone)}


def _run_episode(
    *, env: Any, policy: Any, env_preprocessor: Any, env_postprocessor: Any, preprocessor: Any,
    postprocessor: Any, gateway_profile: Any, config: dict[str, Any], dependency_hash: str,
    spec: dict[str, Any], branch: str, output_dir: Path, step_trace: JsonlWriter,
    result_trace: JsonlWriter, deadline: float | None, set_seed: Callable[[int], None],
    preprocess_observation: Callable[[Any], Any], rollout_option: Any, product_io: Any | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    task_id, state_index, seed = spec["task_id"], spec["state_index"], spec["seed"]
    episode_id = f"{spec['phase']}-task{task_id:02d}-state{state_index:03d}-{branch}"
    started = time.monotonic_ns()
    record: dict[str, Any] = {**spec, "branch": branch, "episode_id": episode_id, "success": False, "crashed": False, "steps": 0}
    arrays: dict[str, Any] = {}
    cycle_times: dict[str, list[int]] = collections.defaultdict(list)
    valid_parent_final_collision = valid_parent_final_unproven = 0
    full_incremental_disagreement = finite_resolution_disagreement = 0
    recovered_full_incremental_disagreement = 0
    unwanted: list[dict[str, Any]] = []
    denied = 0
    raw_preclip_components = final_preclip_components = 0
    env.envs[0].init_state_id = state_index
    set_seed(seed)
    policy.reset()
    observation, _ = env.reset(seed=[seed], options={rollout_option: True})
    underlying = env.envs[0]
    actual_state = int(underlying.init_state_id - underlying._reset_stride)
    if actual_state != state_index:
        raise RuntimeError(f"reset state mismatch: requested {state_index}, observed {actual_state}")
    _, model, data = _sim_handles(underlying)
    qpos_indices, joint_ids, body_ids = _joint_indices(underlying)
    finger_qpos: list[int] = []
    if config.get("certify_moving_fingers", False):
        import mujoco

        finger_joints = underlying._env.robots[0].gripper.joints
        finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in finger_joints]
        if len(finger_ids) != 2 or any(joint_id < 0 for joint_id in finger_ids):
            raise RuntimeError("native Panda gripper must expose both named slide joints")
        finger_qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in finger_ids]
    geometry_profile = PandaGeometryProfile.from_model(
        model, joint_qpos_indices=qpos_indices, joint_ids=joint_ids,
        controlled_extra_qpos_indices=finger_qpos,
        arm_body_ids=body_ids, allowed_contacts=_allowed_contacts(config, model, task_id), config=_geometry_config(config),
    )
    qpos_indices = list(geometry_profile.moving_qpos_indices)
    record["geometry_profile"] = {
        "digest": str(geometry_profile.digest),
        "allowed_contacts_resolved": _jsonable(geometry_profile.allowed_contacts),
        "arm_geom_ids": list(geometry_profile.arm_geom_ids),
        "scene_geom_ids": list(geometry_profile.scene_geom_ids),
        "moving_qpos_indices": qpos_indices,
        "coordinate_units": list(geometry_profile.coordinate_units),
        "scope_limit": (
            "arm and both moving fingers against the scene frozen at each chunk root; carried-object future motion is not certified"
            if finger_qpos else
            "arm-link compiled collision proxies against the frozen scene; moving fingers and carried objects are not jointly certified"
        ),
    }
    external_scene_base = {"task_id": task_id, "state_index": state_index, "seed": seed,
                           "instruction": str(env.call("task_description")[0])}
    scene = capture_scene_snapshot(model, data, geometry_profile, external_snapshot=_canonical(external_scene_base))
    run_id = f"{episode_id}-{uuid.uuid4().hex[:8]}"
    event_log = EventLog(run_id, maxlen=100_000, schema_version="product-v1")
    gateway = NativeGateway(
        gateway_profile, event_log,
        lease_ttl_ns=int(float(config.get("lease_ttl_ms", 1000.0)) * 1e6),
        max_feedback_age_ns=int(float(config.get("max_feedback_age_ms", 2000.0)) * 1e6),
    )
    if product_io is not None:
        product_io.bind_scene(
            env, task_id=task_id, task_name=str(external_scene_base["instruction"]),
            state_index=state_index, seed=seed,
        )
    previous_raw = None
    previous_horizon = 10
    recovery_enabled = bool(config.get("recover_safe_prefix", False)) and branch in {"full_final", "delta_evc"}
    motion_repair_count = 0
    total_reward = 0.0
    max_reward = -math.inf
    done = False
    episode_step_limit = int(config.get("_preflight_step_limit", 1)) if spec["phase"] == "preflight" else int(
        config.get("_formal_step_limit", 280)
    )

    def cooperative_stop(stage: str) -> bool:
        if product_io is None or not product_io.stop_requested(stage):
            return False
        gateway.revoke("operator requested cooperative product session stop")
        record.update({"stopped": True, "stop_reason": "operator_requested", "stop_stage": stage})
        return True

    try:
        for cycle in range(280 if recovery_enabled else 28):
            if done or record["steps"] >= episode_step_limit or _deadline_reached(deadline):
                break
            if cooperative_stop("before_policy_cycle"):
                break
            cycle_start = time.monotonic_ns()
            policy_start = time.monotonic_ns()
            processed = preprocess_observation(observation)
            processed["task"] = list(env.call("task_description"))
            processed = env_preprocessor(processed)
            batch = preprocessor(processed)
            chunk_policy_input_hash = _observation_hash(batch)
            inference_start = time.monotonic_ns()
            with torch.inference_mode():
                raw_chunk = _array(policy._get_action_chunk(batch))
            inference_ns = time.monotonic_ns() - inference_start
            preprocess_ns = inference_start - policy_start
            cycle_times["preprocess"].append(preprocess_ns)
            cycle_times["inference"].append(inference_ns)
            if raw_chunk.shape != (1, 50, 7):
                raise RuntimeError(f"SmolVLA prediction must be (1,50,7), found {raw_chunk.shape}")
            transform_start = time.monotonic_ns()
            raw_candidate = raw_chunk[:, :10, :].copy()
            selected_transform = _selected_transform(config)
            if selected_transform["name"] == "native_first_ten_v1" or previous_raw is None:
                aggregate = raw_candidate.copy()
            else:
                aggregate = 0.5 * raw_candidate + 0.5 * previous_raw[:, previous_horizon:previous_horizon + 10, :]
            changed = float(np.linalg.norm(aggregate - raw_candidate))
            raw_native_preclip = _native_actions(
                raw_candidate, postprocessor, env_postprocessor, str(config.get("device", "cuda"))
            )
            final_native_preclip = _native_actions(
                aggregate, postprocessor, env_postprocessor, str(config.get("device", "cuda"))
            )
            clip_enabled = bool(config.get("native_clip_to_profile", False))
            raw_native, raw_clip = _clip_native_to_profile(raw_native_preclip, gateway_profile, clip_enabled)
            final_native, final_clip = _clip_native_to_profile(final_native_preclip, gateway_profile, clip_enabled)
            raw_preclip_components += int(raw_clip["changed_components"])
            final_preclip_components += int(final_clip["changed_components"])
            transform_ns = time.monotonic_ns() - transform_start
            cycle_times["transform_and_postprocess"].append(transform_ns)
            execution_state = _capture_rollout_state(underlying)
            execution_state_fingerprint = _rollout_fingerprint(underlying, execution_state)
            execution_native_data_digest = str(execution_state["native_data_digest"])
            external_scene = _canonical({
                **external_scene_base,
                "integration_native_data_controller_rng_fingerprint": execution_state_fingerprint,
                "native_data_digest": execution_native_data_digest,
            })
            scene = capture_scene_snapshot(model, data, geometry_profile, external_snapshot=external_scene)
            raw_forecast_start = time.monotonic_ns()
            raw_qpos, raw_restore = _forecast(underlying, raw_native, qpos_indices, geometry_profile)
            raw_forecast_ns = time.monotonic_ns() - raw_forecast_start
            if changed == 0.0:
                final_qpos, final_restore, final_forecast_ns = raw_qpos.copy(), dict(raw_restore), 0
            else:
                final_forecast_start = time.monotonic_ns()
                final_qpos, final_restore = _forecast(underlying, final_native, qpos_indices, geometry_profile)
                final_forecast_ns = time.monotonic_ns() - final_forecast_start
            if raw_restore["before"] != execution_state_fingerprint or raw_restore["after"] != execution_state_fingerprint:
                raise RuntimeError("raw forecast did not preserve the bound integration/controller/RNG state")
            if final_restore["before"] != execution_state_fingerprint or final_restore["after"] != execution_state_fingerprint:
                raise RuntimeError("final forecast did not preserve the bound integration/controller/RNG state")
            cycle_times["raw_forecast"].append(raw_forecast_ns)
            cycle_times["final_forecast"].append(final_forecast_ns)
            action_bytes_raw = raw_native.tobytes(order="C")
            action_bytes_final = final_native.tobytes(order="C")
            parent_start = time.monotonic_ns()
            parent = certify_trajectory(model, data, geometry_profile, raw_qpos, action_bytes=action_bytes_raw, scene_snapshot=scene)
            parent_ns = time.monotonic_ns() - parent_start
            full_start = time.monotonic_ns()
            full = certify_trajectory(model, data, geometry_profile, final_qpos, action_bytes=action_bytes_final, scene_snapshot=scene)
            full_ns = time.monotonic_ns() - full_start
            incremental_start = time.monotonic_ns()
            incremental = certify_trajectory(
                model, data, geometry_profile, final_qpos, action_bytes=action_bytes_final,
                scene_snapshot=scene, parent=getattr(parent, "certificate", None),
            )
            incremental_ns = time.monotonic_ns() - incremental_start
            review_start = time.monotonic_ns()
            review = dense_review(
                model, data, geometry_profile, final_qpos, action_bytes=action_bytes_final,
                scene_snapshot=scene,
                max_joint_step_rad=float(config.get("dense_review_max_joint_step_rad", 0.002)),
            )
            dense_review_ns = time.monotonic_ns() - review_start
            cycle_times["parent_geometry"].append(parent_ns)
            cycle_times["full_geometry"].append(full_ns)
            cycle_times["incremental_geometry"].append(incremental_ns)
            cycle_times["dense_review_counterfactual"].append(dense_review_ns)
            if bool(getattr(parent, "allowed", False)) and not bool(getattr(full, "allowed", False)):
                if getattr(full, "status", None) == "collision":
                    valid_parent_final_collision += 1
                else:
                    valid_parent_final_unproven += 1
            if bool(getattr(full, "allowed", False)) != bool(getattr(incremental, "allowed", False)):
                full_incremental_disagreement += 1
            review_disagrees = bool(getattr(full, "allowed", False)) != bool(getattr(review, "allowed", False))
            recovery_record = None
            recovery_ns = 0
            forecast_contact_veto = bool(config.get("forecast_contact_veto", False))
            original_forecast_contact_veto = forecast_contact_veto and any(
                final_restore["substep_unwanted_contacts"]
            )
            final_branch_decision = full if branch == "full_final" else incremental
            if recovery_enabled and (final_branch_decision.status in {"collision", "unknown"} or original_forecast_contact_veto):
                from safe_prefix_recovery import recover_safe_prefix

                recovery = recover_safe_prefix(
                    model, data, geometry_profile, final_native, final_qpos,
                    final_restore["action_end_indices"], scene,
                    forecast_substep_unwanted_contacts=final_restore["substep_unwanted_contacts"],
                    veto_unwanted_contacts=forecast_contact_veto,
                )
                recovery_ns = recovery.total_latency_ns
                recovery_record = {
                    "success": recovery.success, "prefix_length": recovery.prefix_length,
                    "attempts": _jsonable(recovery.attempts), "total_work": dict(recovery.total_work),
                    "total_latency_ns": recovery_ns, "original_full": _decision_record(full),
                    "original_parent": _decision_record(parent),
                    "original_incremental": _decision_record(incremental),
                    "original_dense_review": _decision_record(review),
                    "policy": "unchanged_prefix_5_2_1_then_reobserve_replan",
                    "original_forecast_contact_veto": original_forecast_contact_veto,
                }
                chosen_success = recovery.success
                chosen_n = recovery.prefix_length
                chosen_native, chosen_qpos, chosen_decision = recovery.native_actions, recovery.qpos_samples, recovery.decision
                if not recovery.success and config.get("recover_cartesian_motion", False):
                    repair_budget = int(config.get("max_motion_repairs", 32))
                    if motion_repair_count < repair_budget:
                        from cartesian_recovery import recover_cartesian_motion

                        motion = recover_cartesian_motion(
                            final_native[:, :1, :],
                            forecast=lambda actions: _forecast(underlying, actions, qpos_indices, geometry_profile),
                            model=model, data=data, profile=geometry_profile, scene_snapshot=scene,
                            native_lower=gateway_profile.lower, native_upper=gateway_profile.upper,
                            veto_unwanted_contacts=forecast_contact_veto,
                        )
                        recovery_ns += motion.total_latency_ns
                        recovery_record["cartesian_motion"] = {
                            "selected_name": motion.selected_name, "attempts": _jsonable(motion.attempts),
                            "exclusions": _jsonable(motion.exclusions), "total_latency_ns": motion.total_latency_ns,
                        }
                        if motion.native is not None:
                            chosen_success, chosen_n = True, 1
                            chosen_native, chosen_qpos, chosen_decision = motion.native, motion.qpos, motion.decision
                            final_restore = dict(motion.restore)
                            if final_restore["before"] != execution_state_fingerprint or final_restore["after"] != execution_state_fingerprint:
                                raise RuntimeError("Cartesian repair forecast did not preserve its bound controller state")
                            motion_repair_count += 1
                    else:
                        recovery_record["cartesian_motion"] = {"reason": "motion_repair_budget_exhausted", "budget": repair_budget}
                recovery_record["prefix_success"] = recovery.success
                recovery_record["success"] = chosen_success
                recovery_record["prefix_length"] = chosen_n
                recovery_record["selected_execution_horizon"] = chosen_n
                recovery_record["total_latency_ns"] = recovery_ns
                cycle_times["safe_prefix_recovery"].append(recovery_ns)
                if chosen_success:
                    n = chosen_n
                    arrays[f"cycle_{cycle:02d}_proposal_raw_native"] = raw_native.copy()
                    arrays[f"cycle_{cycle:02d}_proposal_final_native"] = final_native.copy()
                    arrays[f"cycle_{cycle:02d}_proposal_raw_qpos"] = raw_qpos.copy()
                    arrays[f"cycle_{cycle:02d}_proposal_final_qpos"] = final_qpos.copy()
                    raw_end = raw_restore["action_end_indices"][n - 1]
                    raw_native = np.ascontiguousarray(raw_native[:, :n, :])
                    raw_qpos = raw_qpos[:raw_end + 1].copy()
                    final_native = chosen_native
                    final_qpos = chosen_qpos
                    for restored, retained in ((raw_restore, raw_qpos), (final_restore, final_qpos)):
                        restored["action_end_indices"] = restored["action_end_indices"][:n]
                        restored["physics_substeps"] = len(retained) - 1
                        restored["substep_unwanted_contacts"] = restored["substep_unwanted_contacts"][:len(retained) - 1]
                    action_bytes_raw = raw_native.tobytes(order="C")
                    action_bytes_final = final_native.tobytes(order="C")
                    prefix_parent_start = time.monotonic_ns()
                    parent = certify_trajectory(model, data, geometry_profile, raw_qpos, action_bytes=action_bytes_raw, scene_snapshot=scene)
                    parent_ns += time.monotonic_ns() - prefix_parent_start
                    full = chosen_decision
                    prefix_incremental_start = time.monotonic_ns()
                    incremental = certify_trajectory(
                        model, data, geometry_profile, final_qpos, action_bytes=action_bytes_final,
                        scene_snapshot=scene, parent=getattr(parent, "certificate", None),
                    )
                    incremental_ns += time.monotonic_ns() - prefix_incremental_start
                    if bool(full.allowed) != bool(incremental.allowed):
                        recovered_full_incremental_disagreement += 1
            selected = parent if branch == "parent_only" else full if branch == "full_final" else incremental
            selected_forecast_contact_veto = forecast_contact_veto and any(
                final_restore["substep_unwanted_contacts"]
            )
            selected_certificate = getattr(selected, "certificate", None)
            expected_certificate_action_sha = hashlib.sha256(
                action_bytes_raw if branch == "parent_only" else action_bytes_final
            ).hexdigest()
            certificate_action_matches_selected_candidate = bool(
                selected_certificate is not None
                and selected_certificate.action_sha256 == expected_certificate_action_sha
            )
            if bool(getattr(selected, "allowed", False)) and not certificate_action_matches_selected_candidate:
                raise RuntimeError("selected geometry certificate is not bound to the selected complete candidate bytes")
            certificate_matches_dispatched_candidate = bool(
                selected_certificate is not None
                and selected_certificate.action_sha256 == hashlib.sha256(action_bytes_final).hexdigest()
            )
            active_geometry_ns = parent_ns if branch == "parent_only" else full_ns + recovery_ns if branch == "full_final" else parent_ns + incremental_ns + recovery_ns
            effective_final_forecast_ns = raw_forecast_ns if changed == 0.0 else final_forecast_ns
            active_forecast_ns = (
                raw_forecast_ns + final_forecast_ns if branch in {"parent_only", "delta_evc"}
                else effective_final_forecast_ns
            )
            active_decision_ns = preprocess_ns + inference_ns + transform_ns + active_forecast_ns + active_geometry_ns
            cycle_times["active_decision_path"].append(active_decision_ns)
            if review_disagrees:
                finite_resolution_disagreement += 1
            arrays[f"cycle_{cycle:02d}_raw_chunk"] = raw_chunk
            arrays[f"cycle_{cycle:02d}_aggregate"] = aggregate
            arrays[f"cycle_{cycle:02d}_raw_native_preclip"] = raw_native_preclip
            arrays[f"cycle_{cycle:02d}_final_native_preclip"] = final_native_preclip
            arrays[f"cycle_{cycle:02d}_raw_native"] = raw_native
            arrays[f"cycle_{cycle:02d}_final_native"] = final_native
            arrays[f"cycle_{cycle:02d}_raw_qpos"] = raw_qpos
            arrays[f"cycle_{cycle:02d}_final_qpos"] = final_qpos
            step_trace.write({
                "event": "candidate", "episode_id": episode_id, "cycle": cycle,
                "raw_chunk": _array_identity(raw_chunk), "raw_candidate": _array_identity(raw_candidate),
                "aggregate": _array_identity(aggregate), "raw_to_aggregate_l2": changed,
                "transform": selected_transform, "raw_forecast": _array_identity(raw_qpos),
                "previous_consumed_horizon": previous_horizon,
                "selected_execution_horizon": int(final_native.shape[1]),
                "forecast_contact_veto_enabled": forecast_contact_veto,
                "original_forecast_contact_veto": original_forecast_contact_veto,
                "selected_forecast_contact_veto": selected_forecast_contact_veto,
                "safe_prefix_recovery": recovery_record,
                "dense_review_scope": "original_ten_action_proposal",
                "original_proposal_comparisons": "valid_parent_final and full_incremental_disagreements use original ten-action proposals",
                "native_transform_pipeline": [
                    "official_lerobot_postprocessor",
                    "optional_componentwise_native_profile_clip" if clip_enabled else "no_optional_native_clip",
                    "reversible_native_simulator_forecast",
                    "optional_certified_cartesian_motion_repair" if config.get("recover_cartesian_motion", False) else "no_cartesian_motion_repair",
                    "geometry_certificate",
                    "native_gateway_permit_and_exact_writer",
                ],
                "raw_native_preclip": _array_identity(raw_native_preclip),
                "final_native_preclip": _array_identity(final_native_preclip),
                "raw_native_preclip_values": _jsonable(raw_native_preclip),
                "final_native_preclip_values": _jsonable(final_native_preclip),
                "raw_native_final": _array_identity(raw_native),
                "final_native_final": _array_identity(final_native),
                "raw_native_final_values": _jsonable(raw_native),
                "final_native_final_values": _jsonable(final_native),
                "raw_native_clip": raw_clip, "final_native_clip": final_clip,
                "aggregate_forecast": _array_identity(final_qpos), "parent": _decision_record(parent),
                "raw_forecast_restore": raw_restore, "aggregate_forecast_restore": final_restore,
                "full": _decision_record(full), "incremental": _decision_record(incremental),
                "dense_independent_review": _decision_record(review), "selected_branch_decision": branch,
                "direct_parent_legacy_disposition": bool(getattr(parent, "allowed", False)),
                "chunk_policy_input_hash": chunk_policy_input_hash,
                "certificate_action_matches_selected_candidate": certificate_action_matches_selected_candidate,
                "certificate_matches_dispatched_candidate": certificate_matches_dispatched_candidate,
                "parent_only_deliberate_legacy_mismatch": branch == "parent_only" and changed != 0.0,
                "integration_controller_rng_fingerprint": execution_state_fingerprint,
                "native_data_digest": execution_native_data_digest,
                "timing_ns": {
                    "preprocess": preprocess_ns, "inference": inference_ns,
                    "transform_and_postprocess": transform_ns,
                    "raw_forecast": raw_forecast_ns, "final_forecast": final_forecast_ns,
                    "parent_geometry": parent_ns, "full_geometry": full_ns,
                    "incremental_geometry": incremental_ns, "dense_review_counterfactual": dense_review_ns,
                    "safe_prefix_recovery": recovery_ns,
                    "active_decision_path": active_decision_ns,
                    "note": "component-sum estimate only; active path excludes counterfactual methods and dense review; full_cycle is measured and includes all diagnostics plus actual writes",
                },
            })
            permitted = bool(getattr(selected, "allowed", False)) and not selected_forecast_contact_veto
            if product_io is not None:
                product_io.publish_candidate(
                    episode_id=episode_id, cycle=cycle, step=record["steps"],
                    action=_jsonable(final_native), qpos=_jsonable(final_qpos),
                    decision={**_decision_record(selected), "allowed": permitted},
                    reason=("forecast_unwanted_contact" if selected_forecast_contact_veto else getattr(selected, "status", None)),
                )
            executed_qpos = [np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()]
            for within in range(int(final_native.shape[1])):
                if done or record["steps"] >= episode_step_limit or _deadline_reached(deadline):
                    break
                if cooperative_stop("before_dispatch"):
                    done = True
                    break
                requested = final_native[:, within, :].copy()
                dispatch = requested
                disposition = "candidate"
                selected_for_step = selected
                expected_step_qpos = final_qpos[final_restore["action_end_indices"][within]]
                fallback_restore = None
                if not permitted:
                    denied += 1
                    disposition = "terminal_denial_no_env_step"
                    done = True
                    step_trace.write({"event": "denied", "episode_id": episode_id, "step": record["steps"],
                                      "reason": "forecast_unwanted_contact" if selected_forecast_contact_veto else getattr(selected, "status", "not_certified"),
                                      "fallback": "none", "env_step_called": False})
                    break
                elif branch in {"full_final", "delta_evc"} and not certificate_matches_dispatched_candidate:
                    raise RuntimeError("final-candidate branch lost its complete-candidate certificate binding")
                observation_before = observation
                admission_processed = preprocess_observation(observation_before)
                admission_processed["task"] = list(env.call("task_description"))
                admission_processed = env_preprocessor(admission_processed)
                admission_batch = preprocessor(admission_processed)
                admission_input_hash = _observation_hash(admission_batch)
                snapshot = NativeSnapshot(
                    feedback_id=f"{episode_id}:step:{record['steps']}", observation_hash=_observation_hash(observation_before),
                    step=record["steps"], capture_mono_ns=time.monotonic_ns(),
                )
                cert_digest = _certificate_digest(selected_for_step)
                context = NativeContext(
                    robot_id="libero-panda", boot_id=f"process-{os.getpid()}", episode_id=episode_id,
                    scene_id=f"libero-{task_id}-{state_index}-{str(scene.digest)[7:23]}",
                    controller_id="libero-relative-controller", task_phase="episode-running",
                    queue_rev=record["steps"],
                    dependencies_hash=_context_dependencies(dependency_hash, scene, geometry_profile, cert_digest),
                    chunk_id=cycle, action_index_in_chunk=within,
                    policy_input_hash=admission_input_hash, policy_input_step=record["steps"],
                )
                gateway.observe_feedback(snapshot, context)
                authorization_start = time.monotonic_ns()
                permit = gateway.authorize(dispatch, snapshot=snapshot, context=context)
                cycle_times["authorize"].append(time.monotonic_ns() - authorization_start)
                if cooperative_stop("before_gateway_submit"):
                    done = True
                    step_trace.write({
                        "event": "stopped", "episode_id": episode_id, "step": record["steps"],
                        "reason": "operator_requested", "env_step_called": False,
                    })
                    break
                entered_ns: int | None = None
                writer_request_identity: dict[str, Any] | None = None
                writer_substep_qpos: list[Any] = []
                writer_substep_contacts: list[Any] = []

                def writer(exact_request: Any, entered: Callable[[], None]) -> Any:
                    nonlocal entered_ns, writer_request_identity, writer_substep_qpos, writer_substep_contacts
                    writer_request_identity = _array_identity(exact_request)
                    if writer_request_identity["sha256"] != permit.request_bytes_hash:
                        raise NativeGatewayDenied("ACTION_REPLACED", "writer received bytes different from permit")
                    if product_io is not None and product_io.stop_requested("writer_entry"):
                        record.update({"stopped": True, "stop_reason": "operator_requested", "stop_stage": "writer_entry"})
                        raise ProductSessionStopped("operator requested stop before native writer entry")
                    entered()
                    # The acknowledgement is the writer-entry boundary. A stop
                    # already requested there must still prevent the env call.
                    if product_io is not None and product_io.stop_requested("writer_entry"):
                        record.update({"stopped": True, "stop_reason": "operator_requested", "stop_stage": "writer_entry"})
                        raise ProductSessionStopped("operator requested stop at native writer entry")
                    entered_ns = time.monotonic_ns()
                    native_output, writer_substep_qpos, writer_substep_contacts = _step_with_substeps(
                        underlying, _array(exact_request)[0], qpos_indices, geometry_profile
                    )
                    observation_value, reward_value, terminated_value, truncated_value, info_value = native_output
                    return (
                        _batch_native(observation_value),
                        np.asarray([reward_value]), np.asarray([terminated_value]),
                        np.asarray([truncated_value]),
                        _batch_native(info_value),
                    )

                submit_start = time.monotonic_ns()
                output = gateway.submit(permit, dispatch, snapshot=snapshot, context=context, writer=writer)
                submit_end = time.monotonic_ns()
                cycle_times["gateway_to_writer"].append((entered_ns or submit_end) - submit_start)
                cycle_times["env_step"].append(submit_end - (entered_ns or submit_start))
                observation, reward, terminated, truncated, info = output[:5]
                if writer_request_identity is None:
                    raise RuntimeError("gateway writer returned without recording its exact request")
                reward_value = float(_array(reward).reshape(-1)[0])
                total_reward += reward_value
                max_reward = max(max_reward, reward_value)
                record["steps"] += 1
                collisions = current_unwanted_collisions(model, data, geometry_profile)
                observed_qpos = np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()
                executed_qpos.extend(writer_substep_qpos)
                arrays[f"cycle_{cycle:02d}_executed_qpos"] = np.asarray(
                    executed_qpos, dtype=np.float64
                ).copy()
                expected_start = 1 if within == 0 else final_restore["action_end_indices"][within - 1] + 1
                expected_end = final_restore["action_end_indices"][within] + 1
                expected_substeps = final_qpos[expected_start:expected_end]
                actual_substeps = np.asarray(writer_substep_qpos, dtype=np.float64)
                if len(actual_substeps) != len(expected_substeps):
                    raise RuntimeError(
                        f"physics substep count changed after restored forecast: {len(actual_substeps)} != {len(expected_substeps)}"
                    )
                tracking_difference = np.abs(actual_substeps - expected_substeps)
                tracking_error = float(np.max(tracking_difference[:, :7]))
                slide_tracking_error = float(np.max(tracking_difference[:, 7:])) if finger_qpos else 0.0
                tracking_tolerance = float(config.get("forecast_tracking_tolerance_rad", 1e-8))
                slide_tracking_tolerance = float(config.get("forecast_tracking_tolerance_slide_m", 0.0))
                collision_rows = [_jsonable(item) for item in collisions]
                for substep_index, substep_rows in enumerate(writer_substep_contacts):
                    unwanted.extend({"step": record["steps"], "physics_substep": substep_index, **item}
                                    for item in substep_rows)
                actual_action_hash = _array_identity(dispatch)["sha256"]
                step_trace.write({
                    "event": "actual_step", "episode_id": episode_id, "step": record["steps"] - 1,
                    "cycle": cycle, "within_cycle": within, "disposition": disposition,
                    "certificate_hash": cert_digest, "permit_id": permit.permit_id,
                    "authorized_action_hash": actual_action_hash,
                    "permit_request_bytes_hash": permit.request_bytes_hash,
                    "actual_env_step_hash": writer_request_identity["sha256"],
                    "writer_hash_matches_permit": writer_request_identity["sha256"] == permit.request_bytes_hash,
                    "gateway_context_digest": context.digest, "scene_hash": str(scene.digest),
                    "geometry_profile_hash": str(geometry_profile.digest), "reward": reward_value,
                    "chunk_policy_input_hash": chunk_policy_input_hash,
                    "admission_observation_input_hash": admission_input_hash,
                    "forecast_tracking_max_abs_rad": tracking_error,
                    "forecast_tracking_tolerance_rad": tracking_tolerance,
                    "forecast_tracking_max_abs_slide_m": slide_tracking_error,
                    "forecast_tracking_tolerance_slide_m": slide_tracking_tolerance,
                    "fallback_forecast_restore": fallback_restore,
                    "actual_physics_substeps": len(writer_substep_qpos),
                    "actual_substep_qpos": _array_identity(np.asarray(writer_substep_qpos)),
                    "actual_substep_unwanted_contacts": writer_substep_contacts,
                    "terminated": _jsonable(terminated), "truncated": _jsonable(truncated),
                    "info": _jsonable(info), "unwanted_collisions": collision_rows,
                    "full_cycle_ns": time.monotonic_ns() - cycle_start,
                })
                if tracking_error > tracking_tolerance or slide_tracking_error > slide_tracking_tolerance:
                    gateway.revoke("forecast tracking drift invalidated geometry certificate")
                    raise RuntimeError(
                        f"actual Panda qpos drifted from restored rollout by {tracking_error:.12g} rad / {slide_tracking_error:.12g} m"
                    )
                done = bool(_array(terminated).reshape(-1)[0]) or bool(_array(truncated).reshape(-1)[0])
                step_success = bool(_array(info.get("is_success", [False])).reshape(-1)[0]) if isinstance(info, dict) else False
                if step_success:
                    record["success"] = True
                    done = True
                if product_io is not None:
                    product_io.publish_actual(
                        step=record["steps"] - 1, cycle=cycle, action=_jsonable(dispatch), reward=reward_value,
                        decision={
                            "allowed": True, "status": getattr(selected_for_step, "status", None),
                            "certificate_hash": cert_digest, "permit_id": permit.permit_id,
                            "writer_hash_matches_permit": writer_request_identity["sha256"] == permit.request_bytes_hash,
                        },
                        permit_id=permit.permit_id, success=record["success"],
                    )
            previous_raw = raw_chunk.copy()
            previous_horizon = int(final_native.shape[1])
            arrays[f"cycle_{cycle:02d}_executed_qpos"] = np.asarray(executed_qpos, dtype=np.float64)
            tracking_report = None
            if selected_certificate is not None and len(executed_qpos) == len(final_qpos):
                tracking_report = _jsonable(compare_execution(selected_certificate, np.asarray(executed_qpos)))
            step_trace.write({
                "event": "cycle_complete", "episode_id": episode_id, "cycle": cycle,
                "executed_qpos": _array_identity(np.asarray(executed_qpos)),
                "geometry_execution_tracking": tracking_report,
                "parent_only_tracking_note": (
                    "certificate covers raw parent while actual dispatch follows aggregate"
                    if branch == "parent_only" and changed != 0.0 else None
                ),
            })
            cycle_times["full_cycle"].append(time.monotonic_ns() - cycle_start)
        record.update({
            "sum_reward": total_reward, "max_reward": None if max_reward == -math.inf else max_reward,
            "unwanted_collisions": len(unwanted), "unwanted_collision_records": unwanted,
            "recovered_full_incremental_disagreements": recovered_full_incremental_disagreement,
            "motion_repair_candidates_selected": motion_repair_count,
            "valid_parent_final_collision": valid_parent_final_collision,
            "valid_parent_final_unproven": valid_parent_final_unproven,
            "full_incremental_disagreements": full_incremental_disagreement,
            "finite_resolution_disagreements": finite_resolution_disagreement, "denied_candidates": denied,
            "fallback_steps": 0, "deadline_reached": _deadline_reached(deadline),
            "native_clip_to_profile": bool(config.get("native_clip_to_profile", False)),
            "raw_candidate_preclip_components_changed": raw_preclip_components,
            "final_candidate_preclip_components_changed": final_preclip_components,
            "timing": {key: _percentiles_ns(values) for key, values in cycle_times.items()},
        })
        if spec["phase"] == "preflight" and record["steps"] != episode_step_limit:
            raise RuntimeError(
                f"preflight executed {record['steps']} actual steps, expected {episode_step_limit}"
            )
    except ProductSessionStopped:
        gateway.revoke("operator requested cooperative product session stop before writer entry")
        record.update({
            "sum_reward": total_reward, "max_reward": None if max_reward == -math.inf else max_reward,
            "unwanted_collisions": len(unwanted), "unwanted_collision_records": unwanted,
            "denied_candidates": denied, "fallback_steps": 0,
            "deadline_reached": _deadline_reached(deadline),
        })
    except Exception as error:
        record.update({"crashed": True, "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        record["elapsed_seconds"] = (time.monotonic_ns() - started) / 1e9
        npz = output_dir / "candidates" / f"{episode_id}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays)
        record["candidate_npz"] = {"path": str(npz.relative_to(output_dir)), "sha256": _file_digest(npz), "arrays": len(arrays)}
        event_path = output_dir / "events" / f"{episode_id}.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(event_log.to_jsonl(), encoding="utf-8")
        record["gateway_events"] = {"path": str(event_path.relative_to(output_dir)), "sha256": _file_digest(event_path)}
        result_trace.write(record)
    return record


def main() -> int:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--config", type=Path, required=True)
    args.add_argument("--preflight-only", action="store_true", help="run only the excluded one-step API/state proof")
    args.add_argument("--preflight-steps", type=int, default=1)
    args.add_argument("--product-session", action="store_true", help="run one managed delta_evc product session")
    args.add_argument("--output-dir", type=Path, help="fresh product output directory")
    args.add_argument("--stop-file", type=Path, help="cooperative product stop request file")
    parsed = args.parse_args()
    product_io = None
    product_config_identity = None
    if parsed.product_session:
        if parsed.preflight_only:
            raise ValueError("--product-session cannot be combined with --preflight-only")
        if parsed.output_dir is None or parsed.stop_file is None:
            raise ValueError("--product-session requires --output-dir and --stop-file")
        from product_session import ProductSessionIO, load_product_config

        config, product_config_identity = load_product_config(parsed.config.resolve())
        config["output_dir"] = str(parsed.output_dir.expanduser().resolve())
        branches = tuple(config.get("branches", BRANCHES))
        if branches != BRANCHES:
            raise ValueError(f"product protocol branches must be exactly {BRANCHES}")
    else:
        config = _load_config(parsed.config.resolve())
    if parsed.preflight_steps <= 0 or parsed.preflight_steps > 280:
        raise ValueError("preflight-steps must be in [1, 280]")
    config["_preflight_step_limit"] = parsed.preflight_steps if parsed.preflight_only else int(
        config.get("preflight", {}).get("max_episode_steps", 1)
    )
    if not 1 <= config["_preflight_step_limit"] <= 280:
        raise ValueError("preflight max_episode_steps must be in [1, 280]")
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("configured output_dir must not exist")
    output_dir.mkdir(parents=True)
    if parsed.product_session:
        product_io = ProductSessionIO(
            output_dir, parsed.stop_file.expanduser().resolve(), product_config_identity
        )
    step_trace = JsonlWriter(output_dir / "steps.jsonl")
    result_trace = JsonlWriter(output_dir / "per_episode.jsonl")
    started = time.time()
    deadline = _deadline(config)
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    backbone = Path(config["backbone"]).expanduser().resolve()
    if not checkpoint.is_dir() or not backbone.is_dir():
        raise FileNotFoundError("checkpoint and backbone directories must exist")
    if product_io is not None and product_io.stop_requested("before_source_and_model_identity"):
        step_trace.close()
        result_trace.close()
        product_io.finalize(status="stopped", result={"success": False, "stopped": True, "steps": 0})
        return 0
    identity = _model_and_source_identity(parsed.config.resolve(), checkpoint, backbone)
    dependency_hash = _digest_bytes(_canonical(identity))
    os.environ.setdefault("MUJOCO_GL", str(config.get("mujoco_gl", "egl")))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if product_io is not None:
        product_io.progress("preparing", "loading_dependencies")

    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.envs import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import close_envs
    from lerobot.utils.random_utils import set_seed

    torch.set_num_threads(int(config.get("torch_threads", 4)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if product_io is not None and product_io.stop_requested("before_environment_load"):
        step_trace.close()
        result_trace.close()
        product_io.finalize(status="stopped", result={"success": False, "stopped": True, "steps": 0})
        return 0
    specs = _episode_specs(config)
    if parsed.preflight_only:
        specs = specs[:1]
    elif parsed.product_session:
        specs = specs[1:2]
    task_ids = sorted({item["task_id"] for item in specs})
    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=task_ids, fps=20, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360, observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    if product_io is not None:
        product_io.progress("preparing", "loading_policy")
        if product_io.stop_requested("before_policy_load"):
            close_envs(envs)
            step_trace.close()
            result_trace.close()
            product_io.finalize(status="stopped", result={"success": False, "stopped": True, "steps": 0})
            return 0
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 50:
        raise RuntimeError("official checkpoint must declare chunk_size=n_action_steps=50")
    policy_cfg.device = str(config.get("device", "cuda"))
    policy_cfg.vlm_model_name = str(backbone)
    policy_cfg.pretrained_path = checkpoint
    if hasattr(policy_cfg, "load_vlm_weights"):
        policy_cfg.load_vlm_weights = False
    rename_map = config.get("rename_map", {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    })
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    from safetensors import safe_open
    checkpoint_keys: set[str] = set()
    model_tensor_files = sorted(checkpoint.glob("model*.safetensors"))
    if not model_tensor_files:
        raise FileNotFoundError("checkpoint has no model*.safetensors tensors")
    for tensor_file in model_tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            checkpoint_keys.update(handle.keys())
    loaded_keys = set(policy.state_dict().keys())
    missing_keys = sorted(loaded_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - loaded_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"checkpoint/model state key mismatch: missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
    checkpoint_key_validation = {
        "model_tensor_files": [path.name for path in model_tensor_files],
        "checkpoint_keys": len(checkpoint_keys), "loaded_model_keys": len(loaded_keys),
        "missing_keys": 0, "unexpected_keys": 0,
    }
    if parsed.product_session and _model_and_source_identity(parsed.config.resolve(), checkpoint, backbone) != identity:
        close_envs(envs)
        raise RuntimeError("model or source files changed while the product policy was loading")
    policy.config.n_action_steps = 10
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(backbone)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    gateway_profile = load_profile(Path(config["native_profile"]).resolve() if config.get("native_profile") else None)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    results: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    preflight_passed = True
    branch_names = ("delta_evc",) if parsed.product_session else BRANCHES
    try:
        for spec in specs:
            for branch in branch_names:
                if product_io is not None and product_io.stop_requested("before_episode_load"):
                    unfinished.append({**spec, "branch": branch, "reason": "operator_requested"})
                    continue
                if _deadline_reached(deadline):
                    unfinished.append({**spec, "branch": branch, "reason": "deadline_reached"})
                    continue
                if spec["phase"] == "formal" and not preflight_passed:
                    unfinished.append({**spec, "branch": branch, "reason": "preflight_failed"})
                    continue
                env = envs["libero_spatial"][spec["task_id"]]
                available = len(env.envs[0]._init_states)
                if spec["state_index"] >= available:
                    row = {**spec, "branch": branch, "success": False, "crashed": True, "steps": 0,
                           "error_type": "INITIAL_STATE_UNAVAILABLE", "message": f"only {available} states available"}
                    result_trace.write(row)
                    results.append(row)
                    if spec["phase"] == "preflight":
                        preflight_passed = False
                    continue
                try:
                    row = _run_episode(
                        env=env, policy=policy, env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor, preprocessor=preprocessor,
                        postprocessor=postprocessor, gateway_profile=gateway_profile, config=config,
                        dependency_hash=dependency_hash, spec=spec, branch=branch, output_dir=output_dir,
                        step_trace=step_trace, result_trace=result_trace, deadline=deadline, set_seed=set_seed,
                        preprocess_observation=preprocess_observation, rollout_option=NEW_ROLLOUT_OPTION,
                        product_io=product_io,
                    )
                except Exception as error:
                    row = {**spec, "branch": branch, "success": False, "crashed": True, "steps": 0,
                           "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                    result_trace.write(row)
                results.append(row)
                if spec["phase"] == "preflight" and row.get("crashed"):
                    preflight_passed = False
    finally:
        close_envs(envs)
        step_trace.close()
        result_trace.close()

    formal = [row for row in results if row["phase"] == "formal"]
    expected_formal = sum(item["phase"] == "formal" for item in specs) * len(branch_names)
    metrics: dict[str, Any] = {}
    for branch in branch_names:
        rows = [row for row in formal if row["branch"] == branch]
        metrics[branch] = {
            "episodes": len(rows), "successes": sum(bool(row.get("success")) for row in rows),
            "crashes": sum(bool(row.get("crashed")) for row in rows),
            "steps": sum(int(row.get("steps", 0)) for row in rows),
            "unwanted_collisions": sum(int(row.get("unwanted_collisions", 0)) for row in rows),
            "valid_parent_final_collision": sum(int(row.get("valid_parent_final_collision", 0)) for row in rows),
            "valid_parent_final_unproven": sum(int(row.get("valid_parent_final_unproven", 0)) for row in rows),
            "finite_resolution_disagreements": sum(int(row.get("finite_resolution_disagreements", 0)) for row in rows),
            "full_incremental_disagreements": sum(int(row.get("full_incremental_disagreements", 0)) for row in rows),
            "raw_candidate_preclip_components_changed": sum(
                int(row.get("raw_candidate_preclip_components_changed", 0)) for row in rows
            ),
            "final_candidate_preclip_components_changed": sum(
                int(row.get("final_candidate_preclip_components_changed", 0)) for row in rows
            ),
        }
    identity_valid = not parsed.product_session or (
        _model_and_source_identity(parsed.config.resolve(), checkpoint, backbone) == identity
    )
    manifest = {
        "schema": "sentinel-unified-smolvla-libero-geometry-evc-product-v1" if parsed.product_session else "sentinel-unified-smolvla-libero-geometry-evc-v1",
        "status": "failed" if not identity_valid else "stopped" if parsed.product_session and (
            any(bool(row.get("stopped")) for row in results)
            or any(item.get("reason") == "operator_requested" for item in unfinished)
        ) else "rejected" if parsed.product_session and results and (
            not any(bool(row.get("crashed")) for row in results)
            and any(int(row.get("denied_candidates", 0)) > 0 for row in results)
        ) else "complete" if (
            len(results) == len(specs) * len(branch_names)
            and len(formal) == expected_formal
            and not unfinished
            and not any(bool(row.get("crashed")) for row in results)
        ) else "deadline_stopped" if unfinished else "failed",
        "claim_boundary": "MuJoCo/LIBERO experimental certificates; no continuous physical-safety or functional-safety claim",
        "actual_policy": "lerobot/smolvla_libero native Panda checkpoint",
        "prediction_chunk_size": 50, "execution_horizon": 10, "max_episode_steps": 280,
        "execution_mode": "product_session" if parsed.product_session else "preflight_only" if parsed.preflight_only else "preflight_then_formal",
        "preflight_step_limit": config["_preflight_step_limit"],
        "formal_step_limit": int(config.get("_formal_step_limit", 280)),
        "temporal_aggregation": _selected_transform(config), "preflight_excluded_from_formal": True,
        "native_profile_clip": {
            "enabled": bool(config.get("native_clip_to_profile", False)),
            "position": "after official LeRobot postprocessor; before forecast, geometry, authorization, and writer",
            "profile_bounds_unchanged": True,
        },
        "config": config, "identity": identity, "dependencies_hash": dependency_hash,
        "product_config_identity": product_config_identity,
        "product_model_source_identity_unchanged": identity_valid if parsed.product_session else None,
        "checkpoint_key_validation": checkpoint_key_validation,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     **{name: importlib.metadata.version(name) for name in ("lerobot", "hf-libero", "robosuite", "mujoco")}},
        "libero_assets": _tree_digest(Path(get_libero_path("assets"))),
        "task_names": {str(task): str(suite.tasks[task].name) for task in task_ids},
        "results": results, "metrics": metrics, "unfinished_cases": unfinished,
        "timing": {"started_unix": started, "finished_unix": time.time(), "elapsed_seconds": time.time() - started,
                   "deadline_timestamp": config.get("deadline_timestamp")},
        "artifacts": {
            "steps": {"path": "steps.jsonl", "sha256": _file_digest(output_dir / "steps.jsonl")},
            "per_episode": {"path": "per_episode.jsonl", "sha256": _file_digest(output_dir / "per_episode.jsonl")},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if product_io is not None:
        result = results[0] if results else {"success": False, "stopped": manifest["status"] == "stopped", "steps": 0}
        product_io.finalize(status=manifest["status"], result=result)
    print(json.dumps({"status": manifest["status"], "formal_episodes": len(formal), "unfinished": len(unfinished), "output": str(output_dir)}, indent=2))
    return 0 if manifest["status"] in {"complete", "stopped", "rejected"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
