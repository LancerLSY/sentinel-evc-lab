#!/usr/bin/env python3
"""Forensic comparison of LIBERO rollout state around restored native actions.

This diagnostic loads no policy and never writes into the source experiment.
It replays retained native actions, records every MuJoCo contact, restores the
runner snapshot, and reports leaf values, identities, aliases, RNG, and native
integration state before rollout, after rollout, and after restore.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import run_unified_libero as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _value_digest(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return runner._array_identity(item)
        if isinstance(item, np.generic):
            return {"numpy_scalar": str(item.dtype), "value": item.item()}
        if isinstance(item, dict):
            return {repr(key): encode(child) for key, child in sorted(item.items(), key=lambda pair: repr(pair[0]))}
        if isinstance(item, (list, tuple, collections.deque)):
            return [encode(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite_float": repr(item)}
        return item

    return runner._digest_bytes(runner._canonical(encode(value)))


def _leaf_inventory(underlying: Any) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    aliases: dict[int, list[str]] = collections.defaultdict(list)

    def visit(value: Any, path: str) -> None:
        metadata: dict[str, Any] = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
        if isinstance(value, np.ndarray):
            metadata.update({"object_id": id(value), "shape": list(value.shape), "dtype": str(value.dtype),
                             "digest": runner._array_identity(value)["sha256"]})
            aliases[id(value)].append(path)
        elif isinstance(value, np.generic):
            metadata.update({"dtype": str(value.dtype), "digest": _value_digest(value)})
        elif isinstance(value, dict):
            metadata.update({"object_id": id(value), "length": len(value), "digest": _value_digest(value)})
            aliases[id(value)].append(path)
            for key, child in sorted(value.items(), key=lambda pair: repr(pair[0])):
                visit(child, f"{path}[{repr(key)}]")
        elif isinstance(value, (list, tuple, collections.deque)):
            metadata.update({"object_id": id(value), "length": len(value), "digest": _value_digest(value)})
            if isinstance(value, (list, collections.deque)):
                aliases[id(value)].append(path)
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        else:
            metadata.update({"digest": _value_digest(value), "repr": repr(value)})
        nodes[path] = metadata

    objects: dict[str, Any] = {}
    for obj in runner._stateful_objects(underlying):
        object_key = f"{type(obj).__module__}.{type(obj).__qualname__}@{id(obj)}"
        safe_keys = sorted(key for key, value in obj.__dict__.items() if runner._safe_leaf(value))
        objects[object_key] = {
            "object_id": id(obj), "type": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "all_attribute_keys": sorted(obj.__dict__), "safe_attribute_keys": safe_keys,
        }
        for key in safe_keys:
            visit(obj.__dict__[key], f"{object_key}.{key}")
    alias_groups = sorted(
        (sorted(paths) for paths in aliases.values() if len(paths) > 1), key=lambda paths: paths[0]
    )
    return {"objects": objects, "nodes": nodes, "alias_groups": alias_groups}


def _comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_objects, after_objects = before["objects"], after["objects"]
    before_nodes, after_nodes = before["nodes"], after["nodes"]
    common_nodes = sorted(set(before_nodes) & set(after_nodes))
    value_changes: list[dict[str, Any]] = []
    identity_changes: list[dict[str, Any]] = []
    for path in common_nodes:
        left, right = before_nodes[path], after_nodes[path]
        comparable = ("type", "shape", "dtype", "length", "digest", "repr")
        changed = {key: {"before": left.get(key), "after": right.get(key)}
                   for key in comparable if left.get(key) != right.get(key)}
        if changed:
            value_changes.append({"path": path, "changes": changed})
        if left.get("object_id") != right.get("object_id") and (
            "object_id" in left or "object_id" in right
        ):
            identity_changes.append({"path": path, "before": left.get("object_id"), "after": right.get("object_id")})
    common_objects = sorted(set(before_objects) & set(after_objects))
    object_key_changes = []
    for key in common_objects:
        left, right = before_objects[key], after_objects[key]
        if left["all_attribute_keys"] != right["all_attribute_keys"] or left["safe_attribute_keys"] != right["safe_attribute_keys"]:
            object_key_changes.append({"object": key, "before_all": left["all_attribute_keys"],
                                       "after_all": right["all_attribute_keys"],
                                       "before_safe": left["safe_attribute_keys"],
                                       "after_safe": right["safe_attribute_keys"]})
    return {
        "missing_object_ids": sorted(set(before_objects) - set(after_objects)),
        "new_object_ids": sorted(set(after_objects) - set(before_objects)),
        "object_attribute_key_changes": object_key_changes,
        "missing_leaf_paths": sorted(set(before_nodes) - set(after_nodes)),
        "new_leaf_paths": sorted(set(after_nodes) - set(before_nodes)),
        "value_or_type_changes": value_changes,
        "container_or_array_identity_changes": identity_changes,
        "alias_groups_before": before["alias_groups"],
        "alias_groups_after": after["alias_groups"],
        "alias_groups_match": before["alias_groups"] == after["alias_groups"],
        "exact_leaf_values_match": not value_changes and set(before_nodes) == set(after_nodes),
    }


def _snapshot_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": None,
        "integration": runner._array_identity(state["integration"]),
        "native_data_digest": state["native_data_digest"],
        "state_inventory": state["state_inventory"],
        "rng": {
            "python": runner._digest_bytes(runner._canonical(state["random"])),
            "numpy": _value_digest(state["numpy_random"]),
            "torch": runner._array_identity(state["torch_random"]),
            "cuda": [runner._array_identity(item) for item in (state["cuda_random"] or [])],
            "objects": [runner._digest_bytes(runner._canonical(saved)) for _, saved in state["object_rngs"]],
        },
    }


def _raw_contacts(model: Any, data: Any, allowed: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    import mujoco

    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        pair = tuple(sorted((geom1, geom2)))
        rows.append({
            "contact_index": index, "geom1": geom1, "geom2": geom2,
            "geom1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "dist": float(contact.dist), "pos": np.asarray(contact.pos).tolist(),
            "allowed_label": allowed.get(pair),
        })
    return rows


def _case(
    *, env: Any, config: dict[str, Any], run_dir: Path, task_id: int, state_index: int, seed: int,
    branch: str, rollout_option: Any, set_seed: Any,
) -> dict[str, Any]:
    underlying = env.envs[0]
    underlying.init_state_id = state_index
    set_seed(seed)
    env.reset(seed=[seed], options={rollout_option: True})
    actual_state = int(underlying.init_state_id - underlying._reset_stride)
    if actual_state != state_index:
        raise RuntimeError(f"reset state mismatch: requested {state_index}, observed {actual_state}")
    _, model, data = runner._sim_handles(underlying)
    qpos_indices, _, _ = runner._joint_indices(underlying)
    allowed_contacts = runner._allowed_contacts(config, model, task_id)
    allowed = {tuple(sorted((int(item.geom_a), int(item.geom_b)))): str(item.label) for item in allowed_contacts}

    episode_id = f"formal-task{task_id:02d}-state{state_index:03d}-{branch}"
    rows = [row for row in _jsonl(run_dir / "per_episode.jsonl") if row.get("episode_id") == episode_id]
    if len(rows) != 1:
        raise RuntimeError(f"expected one A3 result for {episode_id}, found {len(rows)}")
    npz_path = run_dir / rows[0]["candidate_npz"]["path"]
    with np.load(npz_path, allow_pickle=False) as retained:
        candidates = sorted(key for key in retained.files if key.endswith("_final_native"))
        if not candidates:
            raise RuntimeError(f"{npz_path} has no retained final native action chunk")
        action_chunks = [(key, np.asarray(retained[key], dtype=np.float32)) for key in candidates[:4]]
    if any(actions.shape != (1, 10, 7) for _, actions in action_chunks):
        raise RuntimeError(f"retained actions have unexpected shapes {[item.shape for _, item in action_chunks]}")

    contact_phase = "forecast"
    contact_cycle = 0
    all_contacts: list[dict[str, Any]] = []
    original_contacts = runner.current_unwanted_collisions

    def observe_contacts(model_arg: Any, data_arg: Any, unused_profile: Any) -> tuple[Any, ...]:
        rows_now = _raw_contacts(model_arg, data_arg, allowed)
        all_contacts.append({"cycle": contact_cycle, "phase": contact_phase,
                             "physics_substep": len(all_contacts), "contacts": rows_now})
        return tuple(rows_now)

    runner.current_unwanted_collisions = observe_contacts
    cycles: list[dict[str, Any]] = []
    try:
        for contact_cycle, (candidate_key, actions) in enumerate(action_chunks):
            before_state = runner._capture_rollout_state(underlying)
            before_fingerprint = runner._rollout_fingerprint(underlying, before_state)
            before_leaves = _leaf_inventory(underlying)
            rollout_qpos = [np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()]
            contact_phase = "forecast"
            forecast_contact_start = len(all_contacts)
            for action in actions[0]:
                _, samples, _ = runner._step_with_substeps(underlying, action, qpos_indices, None)
                rollout_qpos.extend(samples)
            forecast_contacts = all_contacts[forecast_contact_start:]
            mutated_state = runner._capture_rollout_state(underlying)
            mutated_fingerprint = runner._rollout_fingerprint(underlying, mutated_state)
            mutated_leaves = _leaf_inventory(underlying)
            runner._restore_rollout_state(underlying, before_state)
            restored_state = runner._capture_rollout_state(underlying)
            restored_fingerprint = runner._rollout_fingerprint(underlying, restored_state)
            restored_leaves = _leaf_inventory(underlying)

            before_summary = _snapshot_summary(before_state)
            mutated_summary = _snapshot_summary(mutated_state)
            restored_summary = _snapshot_summary(restored_state)
            before_summary["fingerprint"] = before_fingerprint
            mutated_summary["fingerprint"] = mutated_fingerprint
            restored_summary["fingerprint"] = restored_fingerprint
            restored_comparison = _comparison(before_leaves, restored_leaves)
            restored_comparison["top_level_identity_changes"] = [
                item for item in restored_comparison["container_or_array_identity_changes"]
                if "[" not in item["path"].split("@", 1)[-1]
            ]
            cycles.append({
                "cycle": contact_cycle, "candidate_key": candidate_key,
                "action_chunk": runner._array_identity(actions),
                "forecast_qpos": runner._array_identity(np.asarray(rollout_qpos)),
                "snapshots": {"before": before_summary, "mutated": mutated_summary, "restored": restored_summary},
                "comparisons": {
                    "before_vs_mutated": _comparison(before_leaves, mutated_leaves),
                    "before_vs_restored": restored_comparison,
                    "fingerprint_restored": before_fingerprint == restored_fingerprint,
                    "integration_restored": np.array_equal(before_state["integration"], restored_state["integration"]),
                    "native_data_restored": before_state["native_data_digest"] == restored_state["native_data_digest"],
                    "rng_restored": before_summary["rng"] == restored_summary["rng"],
                },
            })
            contact_phase = "actual_advance"
            actual_contact_start = len(all_contacts)
            actual_qpos = [np.asarray(data.qpos[qpos_indices], dtype=np.float64).copy()]
            for action in actions[0]:
                _, samples, _ = runner._step_with_substeps(underlying, action, qpos_indices, None)
                actual_qpos.extend(samples)
            actual_contacts = all_contacts[actual_contact_start:]
            forecast_array = np.asarray(rollout_qpos, dtype=np.float64)
            actual_array = np.asarray(actual_qpos, dtype=np.float64)
            forecast_contact_signatures = [
                runner._digest_bytes(runner._canonical(item["contacts"])) for item in forecast_contacts
            ]
            actual_contact_signatures = [
                runner._digest_bytes(runner._canonical(item["contacts"])) for item in actual_contacts
            ]
            first_contact_difference = next(
                (index for index, (left, right) in enumerate(zip(forecast_contact_signatures, actual_contact_signatures))
                 if left != right),
                None,
            )
            if first_contact_difference is None and len(forecast_contact_signatures) != len(actual_contact_signatures):
                first_contact_difference = min(len(forecast_contact_signatures), len(actual_contact_signatures))
            cycles[-1]["actual_qpos"] = runner._array_identity(actual_array)
            cycles[-1]["forecast_actual_shape_match"] = forecast_array.shape == actual_array.shape
            cycles[-1]["forecast_actual_max_abs_rad"] = (
                float(np.max(np.abs(forecast_array - actual_array)))
                if forecast_array.shape == actual_array.shape else None
            )
            cycles[-1]["contact_sequence"] = {
                "forecast_substeps": len(forecast_contact_signatures),
                "actual_substeps": len(actual_contact_signatures),
                "forecast_sha256": runner._digest_bytes(runner._canonical(forecast_contact_signatures)),
                "actual_sha256": runner._digest_bytes(runner._canonical(actual_contact_signatures)),
                "exact_match": forecast_contact_signatures == actual_contact_signatures,
                "first_differing_physics_substep": first_contact_difference,
            }
    finally:
        runner.current_unwanted_collisions = original_contacts
    return {
        "episode_id": episode_id, "task_id": task_id, "state_index": state_index, "seed": seed, "branch": branch,
        "source_npz": str(npz_path), "source_npz_sha256": _sha256(npz_path),
        "cycles": cycles,
        "physics_substeps": len(all_contacts), "all_contacts_by_substep": all_contacts,
        "contact_summary": {
            "substeps_with_contacts": sum(bool(item["contacts"]) for item in all_contacts),
            "contact_records": sum(len(item["contacts"]) for item in all_contacts),
            "allowed_contact_records": sum(
                contact["allowed_label"] is not None for item in all_contacts for contact in item["contacts"]
            ),
        },
        "all_cycle_fingerprints_restored": all(item["comparisons"]["fingerprint_restored"] for item in cycles),
        "all_cycle_leaf_values_restored": all(
            item["comparisons"]["before_vs_restored"]["exact_leaf_values_match"] for item in cycles
        ),
        "all_cycle_forecasts_exact_actual": all(
            item["forecast_actual_shape_match"] and item["forecast_actual_max_abs_rad"] == 0.0
            and item["contact_sequence"]["exact_match"] for item in cycles
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument("--states", type=int, nargs="+", default=[40, 41])
    parser.add_argument("--branch", default="delta_evc")
    args = parser.parse_args()
    config_path, run_dir, output = args.config.resolve(), args.run_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("output must not exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION
    from lerobot.scripts.lerobot_eval import close_envs
    from lerobot.utils.random_utils import set_seed

    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[args.task_id], fps=20, init_states=True, hard_reset=True,
        control_mode="relative", max_parallel_tasks=1, observation_height=360, observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][args.task_id]
    formal = config["formal"]
    first_state = min(int(item) for item in formal["initial_state_indices"])
    seed_base = int(formal["seed_base"])
    report = {
        "schema": "sentinel-unified-libero-rollout-state-forensic-v1",
        "diagnostic_only": True, "policy_loaded": False,
        "sources": {"config": str(config_path), "config_sha256": _sha256(config_path),
                    "run_dir": str(run_dir), "runner_sha256": _sha256(Path(runner.__file__).resolve()),
                    "diagnostic_sha256": _sha256(Path(__file__).resolve())},
        "cases": [],
    }
    try:
        for state in args.states:
            report["cases"].append(_case(
                env=env, config=config, run_dir=run_dir, task_id=args.task_id, state_index=state,
                seed=seed_base + state - first_state, branch=args.branch,
                rollout_option=NEW_ROLLOUT_OPTION, set_seed=set_seed,
            ))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        close_envs(envs)
    print(json.dumps({
        "output": str(output), "cases": [{
            "episode_id": case["episode_id"], "fingerprint_restored": case["all_cycle_fingerprints_restored"],
            "leaf_values_restored": case["all_cycle_leaf_values_restored"],
            "forecasts_exact_actual": case["all_cycle_forecasts_exact_actual"],
            "cycles": [{"cycle": item["cycle"],
                         "alias_match": item["comparisons"]["before_vs_restored"]["alias_groups_match"],
                         "top_level_identity_changes": len(item["comparisons"]["before_vs_restored"]["top_level_identity_changes"]),
                         "forecast_actual_max_abs_rad": item["forecast_actual_max_abs_rad"],
                         "contacts_exact": item["contact_sequence"]["exact_match"],
                         "first_differing_contact_substep": item["contact_sequence"]["first_differing_physics_substep"]}
                        for item in case["cycles"]],
            "contacts": case["contact_summary"],
        } for case in report["cases"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
