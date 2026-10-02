#!/usr/bin/env python3
"""Run the official SmolVLA/LIBERO path through Sentinel's native gateway.

Heavy inference and simulator imports stay in this experiment entry point.
The package core remains standard-library plus ``cryptography``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
OFFICIAL_IDENTITY_PROTOCOL_SHA256 = "431ea1bea335c071f723b1ef1898002b0965b3dde8de4903322874bc1d7ae66b"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentinel_evc.evidence import build_bundle, verify_bundle
from sentinel_evc.events import EventLog
from sentinel_evc.native_gateway import (
    NativeContext,
    NativeGateway,
    NativeGatewayDenied,
    NativePermit,
    NativeSnapshot,
)

from libero_native_profile import load_profile


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_digest = _file_sha256(path).removeprefix("sha256:")
        size = path.stat().st_size
        digest.update(f"{relative}\0{file_digest}\0{size}\n".encode())
        count += 1
        total_bytes += size
    return {
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _array_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(_canonical(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": "sha256:" + digest.hexdigest(),
    }


def _observation_summary(observation: Any) -> dict[str, Any]:
    if isinstance(observation, tuple) and observation:
        observation = observation[0]
    if not isinstance(observation, dict):
        return {"value": _array_identity(observation)}

    def summarize(key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(child): summarize(str(child), item) for child, item in sorted(value.items())}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            identity = _array_identity(value)
            if "state" in key.lower() and math.prod(identity["shape"]) <= 256:
                identity["values"] = _jsonable(value)
            return identity
        except Exception:
            return {"type": type(value).__name__, "canonical": _jsonable(value)}

    summary: dict[str, Any] = {}
    for key, value in sorted(observation.items()):
        summary[str(key)] = summarize(str(key), value)
    return summary


def _observation_hash(summary: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(summary)).hexdigest()


def _action_values(value: Any) -> list[float]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    array = np.asarray(value)
    if array.shape == (1, 7):
        array = array[0]
    if array.shape != (7,):
        raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", f"expected (1, 7) or (7,), found {array.shape}")
    return [float(item) for item in array]


def _extract_observation(output: Any) -> Any:
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def _extract_step_outcome(output: Any) -> dict[str, Any]:
    if not isinstance(output, tuple) or len(output) < 5:
        return {"return_type": type(output).__name__}
    _, reward, terminated, truncated, info = output[:5]
    info_value = _jsonable(info)
    return {
        "reward": _jsonable(reward),
        "terminated": _jsonable(terminated),
        "truncated": _jsonable(truncated),
        "info": info_value,
    }


def _validate_native_transition(output: Any) -> None:
    import numpy as np

    if not isinstance(output, tuple) or len(output) < 5:
        raise RuntimeError("native env.step did not return an observation/reward/termination transition")
    observation, reward, terminated, truncated, info = output[:5]
    if not isinstance(observation, dict) or not observation:
        raise RuntimeError("native env.step returned an empty or unsupported observation")
    reward_array = np.asarray(reward)
    if reward_array.dtype.kind not in "fiu" or not np.all(np.isfinite(reward_array)):
        raise RuntimeError("native env.step returned a non-finite reward")
    for value, name in ((terminated, "terminated"), (truncated, "truncated")):
        array = np.asarray(value)
        if array.dtype.kind not in "biu":
            raise RuntimeError(f"native env.step returned invalid {name}")
    if not isinstance(info, (dict, list, tuple)):
        raise RuntimeError("native env.step returned unsupported info")


def _native_observation_schema(value: Any) -> Any:
    import numpy as np
    if isinstance(value, dict):
        if not value:
            raise RuntimeError("native observation contains an empty object")
        return {str(key): _native_observation_schema(item) for key, item in sorted(value.items())}
    if isinstance(value, str):
        return {"type": "str"}
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or (array.dtype.kind == "f" and not np.all(np.isfinite(array))):
        raise RuntimeError("native observation contains unsupported or non-finite data")
    return {"dtype": str(array.dtype), "shape": list(array.shape)}


class JsonlTrace:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, **payload: Any) -> None:
        self._handle.write(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class NativeRunRecorder:
    def __init__(
        self,
        *,
        mode: str,
        gateway: NativeGateway,
        log: EventLog,
        trace: JsonlTrace,
        fault_names: list[str],
        record_pose_stride: int,
        render_episode_keys: set[tuple[int, int]],
        dependencies_hash: str,
    ) -> None:
        self.mode = mode
        self.gateway = gateway
        self.log = log
        self.trace = trace
        self.fault_names = fault_names
        self.record_pose_stride = record_pose_stride
        self.render_episode_keys = render_episode_keys
        self.dependencies_hash = dependencies_hash
        self.faults_done: set[str] = set()
        self.fault_results: list[dict[str, Any]] = []
        self.env: Any = None
        self.scene_exporter: Any = None
        self.original_step: Callable[[Any], Any] | None = None
        self.context: NativeContext | None = None
        self.snapshot: NativeSnapshot | None = None
        self.pending_action: Any = None
        self.pending_values: list[float] | None = None
        self.pending_bytes_hash: str | None = None
        self.pending_shape: tuple[int, ...] | None = None
        self.pending_dtype: str | None = None
        self.pending_permit: NativePermit | None = None
        self.pending_shadow_reason: str | None = None
        self.pending_authorization_ns: int | None = None
        self.pending_feedback_age_ns: int | None = None
        self.current_episode: dict[str, Any] | None = None
        self.replay_episode: dict[str, Any] | None = None
        self.episodes: list[dict[str, Any]] = []
        self.replay_episodes: list[dict[str, Any]] = []
        self.task_id = -1
        self.state_index = -1
        self.step = 0
        self.raw_chunks = 0
        self.chunk_id = -1
        self.action_index_in_chunk = 0
        self.submitted = 0
        self.accepted = 0
        self.observed = 0
        self.shadow_comparisons = 0
        self.shadow_rejections = 0
        self.shadow_max_abs_diff = 0.0
        self.authorization_latency_ns: list[int] = []
        self.admission_latency_ns: list[int] = []
        self.env_step_latency_ns: list[int] = []
        self.active_submit_total_ns: list[int] = []
        self.feedback_age_at_authorize_ns: list[int] = []
        self.inference_latency_ns: list[int] = []
        self.action_component_min = [math.inf] * 7
        self.action_component_max = [-math.inf] * 7
        self.postprocessed_actions = 0
        self.latest_policy_input_hash: str | None = None
        self.latest_policy_input_mono_ns: int | None = None
        self.latest_policy_input_step = -1

    def set_env(self, env: Any, task_id: int, scene_exporter: Any = None) -> None:
        self.env = env
        self.task_id = task_id
        self.scene_exporter = scene_exporter

    def _pose_frame(self) -> dict[str, Any]:
        if self.scene_exporter is not None:
            native = self.scene_exporter.capture_frame(self.step, step=self.step)
            return {
                "modelId": native["modelId"],
                "bodyWorldPosition": native["bodyWorldPosition"],
                "bodyWorldRotation": native["bodyWorldRotation"],
                "qpos": native["qpos"],
                "qvel": native["qvel"],
                "contacts": native["contacts"],
            }
        return {
            "modelId": None,
            "bodyWorldPosition": [],
            "bodyWorldRotation": [],
            "qpos": None,
            "qvel": None,
            "contacts": None,
        }

    def plan_episode(self, state_index: int, seed: int, instruction: str) -> None:
        episode_id = f"task{self.task_id:02d}-state{state_index:02d}-{uuid.uuid4().hex[:8]}"
        self.state_index = state_index
        self.step = 0
        self.faults_done = set()
        self.chunk_id = -1
        self.action_index_in_chunk = 0
        self.current_episode = {
            "episode_id": episode_id,
            "task_id": self.task_id,
            "initial_state_index": state_index,
            "seed": seed,
            "instruction": instruction,
            "success": False,
            "crashed": False,
            "env_steps": 0,
            "fault_attempts": [],
        }
        self.replay_episode = {
            "episode_id": episode_id,
            "task_id": self.task_id,
            "initial_state_index": state_index,
            "model_asset": f"viewer-model-task{self.task_id}.json"
            if (self.task_id, state_index) in self.render_episode_keys
            else None,
            "frames": [],
        }
        self.context = NativeContext(
            robot_id="libero-panda",
            boot_id=f"process-{platform.node()}-{id(self)}",
            episode_id=episode_id,
            scene_id=f"libero-spatial-task-{self.task_id}-init-{state_index}-seed-{seed}",
            controller_id="libero-relative-controller",
            task_phase="episode-running",
            queue_rev=0,
            dependencies_hash=self.dependencies_hash,
        )
        self.snapshot = None
        self.pending_action = None
        self.pending_permit = None
        self.latest_policy_input_hash = None
        self.latest_policy_input_mono_ns = None
        self.latest_policy_input_step = -1
        self.pending_shadow_reason = None
        self.trace.write(event="episode_plan", **self.current_episode)

    def finish_episode(self, result: dict[str, Any] | None, error: Exception | None) -> None:
        assert self.current_episode is not None and self.replay_episode is not None
        if error is None and result is not None:
            self.current_episode.update(
                success=bool(result["successes"][0]),
                sum_reward=float(result["sum_rewards"][0]),
                max_reward=float(result["max_rewards"][0]),
            )
        elif error is not None:
            self.current_episode.update(
                crashed=True,
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        self.current_episode["env_steps"] = self.step
        self.current_episode["fault_attempts"] = [
            row for row in self.fault_results if row["episode_id"] == self.current_episode["episode_id"]
        ]
        self.episodes.append(self.current_episode)
        self.replay_episodes.append(self.replay_episode)
        self.log.append(
            "OUTCOME",
            episode_id=self.current_episode["episode_id"],
            decision="COMPLETE" if not self.current_episode["crashed"] else "FAILED",
            task_success=self.current_episode["success"],
            env_steps=self.step,
        )

    def capture_policy_input(self, observation: Any) -> None:
        if not isinstance(observation, dict):
            raise RuntimeError("official preprocessed policy input must be a mapping")
        state = observation.get("observation.state")
        state_identity = _array_identity(state) if state is not None else None
        if state_identity is None or tuple(state_identity["shape"][-1:]) != (8,):
            raise RuntimeError(f"official policy state must end in 8 values, found {state_identity}")
        for camera_key in ("observation.images.image", "observation.images.image2"):
            camera = observation.get(camera_key)
            identity = _array_identity(camera) if camera is not None else None
            if identity is None or len(identity["shape"]) < 3:
                raise RuntimeError(f"official policy camera is missing or malformed: {camera_key}")
        summary = _observation_summary(observation)
        self.latest_policy_input_hash = _observation_hash(summary)
        self.latest_policy_input_mono_ns = time.monotonic_ns()
        self.latest_policy_input_step = self.step
        self.trace.write(
            event="policy_input",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            observation_hash=self.latest_policy_input_hash,
            observation=summary,
            mono_ns=self.latest_policy_input_mono_ns,
        )

    def on_raw_chunk(self, chunk: Any) -> None:
        chunk_mono_ns = time.monotonic_ns()
        if self.latest_policy_input_mono_ns is not None:
            self.inference_latency_ns.append(chunk_mono_ns - self.latest_policy_input_mono_ns)
        self.raw_chunks += 1
        self.chunk_id += 1
        self.action_index_in_chunk = 0
        self.trace.write(
            event="raw_policy_chunk",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            chunk_id=self.chunk_id,
            identity=_array_identity(chunk),
            values=_jsonable(chunk),
            policy_input_hash=self.latest_policy_input_hash,
            mono_ns=chunk_mono_ns,
            policy_input_to_chunk_ns=(
                chunk_mono_ns - self.latest_policy_input_mono_ns
                if self.latest_policy_input_mono_ns is not None
                else None
            ),
        )

    def on_selected(self, action: Any) -> None:
        self.trace.write(
            event="selected_normalized_action",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            chunk_id=self.chunk_id,
            action_index_in_chunk=self.action_index_in_chunk,
            identity=_array_identity(action),
            values=_jsonable(action),
        )

    def on_postprocessed(self, action: Any) -> None:
        self.pending_action = action
        self.pending_values = _action_values(action)
        self.postprocessed_actions += 1
        for index, value in enumerate(self.pending_values):
            self.action_component_min[index] = min(self.action_component_min[index], value)
            self.action_component_max[index] = max(self.action_component_max[index], value)
        identity = _array_identity(action)
        self.pending_bytes_hash = identity["sha256"]
        self.pending_shape = tuple(identity["shape"])
        self.pending_dtype = identity["dtype"]
        self.pending_permit = None
        self.pending_shadow_reason = None
        self.pending_authorization_ns = None
        self.pending_feedback_age_ns = None
        postprocess_mono_ns = time.monotonic_ns()
        self.log.append(
            "PROPOSAL",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            chunk_id=self.chunk_id,
            action_index_in_chunk=self.action_index_in_chunk,
            source="official_smolvla_postprocessor",
            raw_chunk_recorded=True,
        )
        self.log.append(
            "TRANSFORM",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            transform="official_lerobot_postprocessor",
            chunk_id=self.chunk_id,
            action_index_in_chunk=self.action_index_in_chunk,
            final_request_hash=self.pending_bytes_hash,
            final_action=self.pending_values,
        )
        self.trace.write(
            event="official_postprocessed_action",
            episode_id=self.context.episode_id if self.context else None,
            step=self.step,
            chunk_id=self.chunk_id,
            action_index_in_chunk=self.action_index_in_chunk,
            identity=_array_identity(action),
            values=self.pending_values,
            mono_ns=postprocess_mono_ns,
            policy_input_to_postprocess_ns=(
                postprocess_mono_ns - self.latest_policy_input_mono_ns
                if self.latest_policy_input_mono_ns is not None
                else None
            ),
        )
        if self.mode in {"shadow", "active", "fault"}:
            if self.snapshot is None or self.context is None:
                raise NativeGatewayDenied("NO_FEEDBACK", "postprocessor ran without actual reset/step feedback")
            self.context = NativeContext(
                **{
                    **self.context.summary(),
                    "chunk_id": self.chunk_id,
                    "action_index_in_chunk": self.action_index_in_chunk,
                    "policy_input_hash": self.latest_policy_input_hash or "sha256:" + "0" * 64,
                    "policy_input_step": self.latest_policy_input_step,
                }
            )
            self.gateway.refresh_context(self.context, snapshot_hash=self.snapshot.digest)
            started = time.monotonic_ns()
            self.pending_feedback_age_ns = started - self.snapshot.capture_mono_ns
            self.feedback_age_at_authorize_ns.append(self.pending_feedback_age_ns)
            try:
                self.pending_permit = self.gateway.authorize(
                    self.pending_action,
                    snapshot=self.snapshot,
                    context=self.context,
                    now_ns=started,
                )
            except NativeGatewayDenied as error:
                if self.mode != "shadow":
                    raise
                self.pending_shadow_reason = error.code
                self.shadow_rejections += 1
                self.trace.write(
                    event="shadow_decision",
                    episode_id=self.context.episode_id,
                    step=self.step,
                    decision="WOULD_REJECT",
                    reason=error.code,
                    intervention=False,
                )
            finally:
                self.pending_authorization_ns = time.monotonic_ns() - started
                self.authorization_latency_ns.append(self.pending_authorization_ns)

    def on_reset(self, output: Any, actual_state_index: int, capture_mono_ns: int) -> None:
        if actual_state_index != self.state_index:
            raise RuntimeError(f"reset state mismatch: planned {self.state_index}, observed {actual_state_index}")
        observation = _extract_observation(output)
        self.observation_schema = _native_observation_schema(observation)
        summary = _observation_summary(observation)
        assert self.context is not None
        self.snapshot = NativeSnapshot(
            feedback_id=f"{self.context.episode_id}:reset",
            observation_hash=_observation_hash(summary),
            step=0,
            capture_mono_ns=capture_mono_ns,
        )
        self.gateway.observe_feedback(self.snapshot, self.context)
        pose = self._pose_frame()
        assert self.replay_episode is not None
        self.replay_episode["frames"].append(
            {
                "step": 0,
                "timeSeconds": 0.0,
                **pose,
                "action": None,
                "decision": "RESET_OBSERVED",
                "reason": None,
                "cursors": {"submitted": self.submitted, "accepted": self.accepted, "observed": self.observed},
                "observation_hash": self.snapshot.observation_hash,
                "state": summary,
            }
        )

    def _fault_attempt(self, name: str, permit: NativePermit, action: Any) -> None:
        assert self.snapshot is not None and self.context is not None and self.pending_values is not None
        callback_calls = 0

        def forbidden_writer(_: Any, __: Callable[[], None]) -> None:
            nonlocal callback_calls
            callback_calls += 1
            raise RuntimeError("fault attempt reached env.step")

        code = "NOT_BLOCKED"
        try:
            if name == "action_replacement":
                import numpy as np

                altered = np.asarray(_jsonable(action), dtype=np.asarray(_jsonable(action)).dtype).copy()
                altered.reshape(-1)[0] += 0.001 if altered.reshape(-1)[0] <= 0.998 else -0.001
                self.gateway.submit(
                    permit,
                    altered,
                    snapshot=self.snapshot,
                    context=self.context,
                    writer=forbidden_writer,
                )
            elif name == "expired_permit":
                self.gateway.submit(
                    permit,
                    action,
                    snapshot=self.snapshot,
                    context=self.context,
                    writer=forbidden_writer,
                    now_ns=permit.deadline_mono_ns + 1,
                )
            elif name == "context_changed":
                changed = NativeContext(**{**self.context.summary(), "queue_rev": self.context.queue_rev + 1})
                self.gateway.submit(
                    permit,
                    action,
                    snapshot=self.snapshot,
                    context=changed,
                    writer=forbidden_writer,
                )
            elif name == "revoked_generation":
                self.gateway.revoke("fault injection: revoke after authorize")
                self.gateway.submit(
                    permit,
                    action,
                    snapshot=self.snapshot,
                    context=self.context,
                    writer=forbidden_writer,
                )
            elif name == "old_feedback":
                old = NativeSnapshot(
                    feedback_id=self.snapshot.feedback_id + ":old",
                    observation_hash=self.snapshot.observation_hash,
                    step=self.snapshot.step,
                    capture_mono_ns=max(0, self.snapshot.capture_mono_ns - 1),
                )
                self.gateway.submit(
                    permit,
                    action,
                    snapshot=old,
                    context=self.context,
                    writer=forbidden_writer,
                )
            else:
                raise ValueError(f"unknown fault: {name}")
        except NativeGatewayDenied as error:
            code = error.code
        expected_codes = {
            "action_replacement": "ACTION_REPLACED",
            "expired_permit": "LEASE_EXPIRED",
            "context_changed": "CONTEXT_CHANGED",
            "revoked_generation": "GENERATION_REVOKED",
            "old_feedback": "FEEDBACK_CHANGED",
        }
        row = {
            "episode_id": self.context.episode_id,
            "step": self.step,
            "fault": name,
            "blocked": code == expected_codes[name] and callback_calls == 0,
            "expected_reason": expected_codes[name],
            "reason": code,
            "env_step_calls": callback_calls,
            "counts_as_task_success": False,
        }
        self.fault_results.append(row)
        self.trace.write(event="fault_attempt", **row)
        if not row["blocked"]:
            raise RuntimeError(f"fault injection was not blocked before env.step: {row}")

    def _run_pending_faults(self, action: Any) -> None:
        assert self.pending_permit is not None and self.pending_values is not None
        for name in self.fault_names:
            if name == "lease_replay" or name in self.faults_done:
                continue
            permit = self.pending_permit
            self._fault_attempt(name, permit, action)
            self.faults_done.add(name)
            self.gateway.revoke(f"fault attempt complete: discard {name} permit")
            assert self.snapshot is not None and self.context is not None
            self.pending_permit = self.gateway.authorize(
                self.pending_action,
                snapshot=self.snapshot,
                context=self.context,
            )

    def env_step(self, action: Any) -> Any:
        import numpy as np

        assert self.original_step is not None
        if self.pending_action is None or self.pending_values is None or self.pending_bytes_hash is None:
            raise RuntimeError("env.step reached without official postprocessed action")
        actual_identity = _array_identity(action)
        expected = np.asarray(_jsonable(self.pending_action))
        actual = np.asarray(_jsonable(action))
        if expected.shape != actual.shape:
            raise RuntimeError("official postprocessor/env.step shape changed")
        difference = float(np.max(np.abs(expected - actual)))
        self.shadow_comparisons += 1
        self.shadow_max_abs_diff = max(self.shadow_max_abs_diff, difference)
        if difference != 0.0 or actual_identity["sha256"] != self.pending_bytes_hash:
            raise NativeGatewayDenied("ACTION_REPLACED", "env.step request differs from official postprocessor bytes")

        writer_enter_ns: int | None = None
        env_started_ns: int | None = None
        env_step_ns: int | None = None
        env_return_ns: int | None = None

        def validated_writer(request: Any, entered: Callable[[], None] = lambda: None) -> Any:
            nonlocal writer_enter_ns, env_started_ns, env_step_ns, env_return_ns
            writer_enter_ns = time.monotonic_ns()
            entered()
            self.submitted += 1
            env_started = time.monotonic_ns()
            env_started_ns = env_started
            output = self.original_step(request)
            env_return_ns = time.monotonic_ns()
            env_step_ns = env_return_ns - env_started
            self.env_step_latency_ns.append(env_step_ns)
            _validate_native_transition(output)
            if _native_observation_schema(_extract_observation(output)) != self.observation_schema:
                raise RuntimeError("native step observation schema differs from actual reset observation")
            return output

        decision = "BASELINE_DIRECT"
        permit_used: NativePermit | None = None
        if self.mode == "baseline":
            output = validated_writer(action)
            self.accepted += 1
        elif self.mode == "shadow":
            decision = "SHADOW_WOULD_REJECT_PASSTHROUGH" if self.pending_shadow_reason else "SHADOW_PASSTHROUGH"
            output = validated_writer(action)
            self.accepted += 1
        else:
            if self.mode == "fault":
                self._run_pending_faults(action)
            assert self.pending_permit is not None and self.snapshot is not None and self.context is not None
            decision = "ACTIVE_AUTHORIZED"
            permit_used = self.pending_permit
            started = time.monotonic_ns()
            try:
                output = self.gateway.submit(
                    permit_used,
                    action,
                    snapshot=self.snapshot,
                    context=self.context,
                    writer=validated_writer,
                )
            except Exception:
                raise
            else:
                self.accepted += 1
            finally:
                finished = time.monotonic_ns()
                self.active_submit_total_ns.append(finished - started)
                if env_started_ns is not None:
                    self.admission_latency_ns.append(env_started_ns - started)

        self.step += 1
        observation = _extract_observation(output)
        summary = _observation_summary(observation)
        assert self.context is not None
        self.context = NativeContext(**{**self.context.summary(), "queue_rev": self.step})
        assert env_return_ns is not None
        capture = env_return_ns
        self.snapshot = NativeSnapshot(
            feedback_id=f"{self.context.episode_id}:step:{self.step}",
            observation_hash=_observation_hash(summary),
            step=self.step,
            capture_mono_ns=capture,
        )
        self.gateway.observe_feedback(self.snapshot, self.context)
        self.observed += 1
        retain_pose = (
            (self.task_id, self.state_index) in self.render_episode_keys
            or self.step % self.record_pose_stride == 0
        )
        pose = self._pose_frame() if retain_pose else {
            "modelId": self.scene_exporter.model_id if self.scene_exporter is not None else None,
            "bodyWorldPosition": [],
            "bodyWorldRotation": [],
            "qpos": None,
            "qvel": None,
            "contacts": None,
        }
        outcome = _extract_step_outcome(output)
        frame = {
            "step": self.step,
            "timeSeconds": self.step / 20.0,
            **pose,
            "action": self.pending_values,
            "action_bytes_hash": self.pending_bytes_hash,
            "decision": decision,
            "reason": self.pending_shadow_reason,
            "permit_id": permit_used.permit_id if permit_used else None,
            "cursors": {"submitted": self.submitted, "accepted": self.accepted, "observed": self.observed},
            "observation_hash": self.snapshot.observation_hash,
            "state": summary,
            "outcome": outcome,
            "latency_ms": {
                "feedback_age_at_authorize": (
                    self.pending_feedback_age_ns / 1e6 if self.pending_feedback_age_ns is not None else None
                ),
                "gateway_authorize": (
                    self.pending_authorization_ns / 1e6 if self.pending_authorization_ns is not None else None
                ),
                "gateway_submit_admission_before_writer": (
                    (env_started_ns - started) / 1e6
                    if self.mode in {"active", "fault"} and env_started_ns is not None
                    else None
                ),
                "native_env_step": env_step_ns / 1e6 if env_step_ns is not None else None,
            },
        }
        assert self.replay_episode is not None
        self.replay_episode["frames"].append(frame)
        self.trace.write(event="env_step_outcome", episode_id=self.context.episode_id, **frame)

        if self.mode == "fault" and "lease_replay" in self.fault_names and "lease_replay" not in self.faults_done:
            assert permit_used is not None
            callback_calls = 0

            def forbidden_writer(_: Any, __: Callable[[], None]) -> None:
                nonlocal callback_calls
                callback_calls += 1

            code = "NOT_BLOCKED"
            try:
                self.gateway.submit(
                    permit_used,
                    action,
                    snapshot=NativeSnapshot(
                        feedback_id=frame["observation_hash"],
                        observation_hash=frame["observation_hash"],
                        step=max(0, self.step - 1),
                        capture_mono_ns=permit_used.issued_mono_ns,
                    ),
                    context=NativeContext(**{**self.context.summary(), "queue_rev": max(0, self.step - 1)}),
                    writer=forbidden_writer,
                )
            except NativeGatewayDenied as error:
                code = error.code
            row = {
                "episode_id": self.context.episode_id,
                "step": self.step - 1,
                "fault": "lease_replay",
                "blocked": code == "LEASE_REPLAY" and callback_calls == 0,
                "reason": code,
                "env_step_calls": callback_calls,
                "counts_as_task_success": False,
            }
            self.fault_results.append(row)
            self.trace.write(event="fault_attempt", **row)
            self.faults_done.add("lease_replay")
            if not row["blocked"]:
                raise RuntimeError(f"lease replay was not blocked: {row}")

        self.pending_action = None
        self.pending_values = None
        self.pending_bytes_hash = None
        self.pending_shape = None
        self.pending_dtype = None
        self.pending_permit = None
        self.pending_shadow_reason = None
        self.pending_authorization_ns = None
        self.pending_feedback_age_ns = None
        self.action_index_in_chunk += 1
        return output


class InputProcessor:
    def __init__(self, delegate: Callable[[Any], Any], recorder: NativeRunRecorder):
        self.delegate = delegate
        self.recorder = recorder

    def __call__(self, observation: Any) -> Any:
        self.recorder.capture_policy_input(observation)
        return self.delegate(observation)


class OutputProcessor:
    def __init__(self, delegate: Callable[[Any], Any], recorder: NativeRunRecorder):
        self.delegate = delegate
        self.recorder = recorder

    def __call__(self, action: Any) -> Any:
        result = self.delegate(action)
        self.recorder.on_postprocessed(result)
        return result


def _wrap_policy(policy: Any, recorder: NativeRunRecorder) -> None:
    original_chunk = policy._get_action_chunk
    original_select = policy.select_action

    def get_chunk(batch: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_chunk(batch, *args, **kwargs)
        recorder.on_raw_chunk(result)
        return result

    def select(batch: Any, **kwargs: Any) -> Any:
        result = original_select(batch, **kwargs)
        recorder.on_selected(result)
        return result

    policy._get_action_chunk = get_chunk
    policy.select_action = select


def _wrap_env(
    env: Any,
    recorder: NativeRunRecorder,
    bind_live_scene: Callable[[], None] | None = None,
) -> None:
    original_reset = env.reset
    recorder.original_step = env.step

    def reset(*args: Any, **kwargs: Any) -> Any:
        output = original_reset(*args, **kwargs)
        capture_mono_ns = time.monotonic_ns()
        if bind_live_scene is not None:
            bind_live_scene()
        underlying = env.envs[0]
        actual = int(underlying.init_state_id - underlying._reset_stride)
        recorder.on_reset(output, actual, capture_mono_ns)
        return output

    env.reset = reset
    env.step = recorder.env_step


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    required = {"mode", "checkpoint", "backbone", "task_ids", "initial_state_indices"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if config["mode"] not in {"baseline", "shadow", "active", "fault"}:
        raise ValueError("mode must be baseline, shadow, active, or fault")
    if not config["task_ids"] or not config["initial_state_indices"]:
        raise ValueError("task_ids and initial_state_indices must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9 for value in config["task_ids"]):
        raise ValueError("task_ids must be integers in [0, 9]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in config["initial_state_indices"]):
        raise ValueError("initial_state_indices must be non-negative integers")
    supported_faults = {
        "action_replacement",
        "lease_replay",
        "expired_permit",
        "old_feedback",
        "context_changed",
        "revoked_generation",
    }
    unknown_faults = sorted(set(config.get("faults", [])) - supported_faults)
    if unknown_faults:
        raise ValueError(f"unknown faults: {unknown_faults}")
    if config["mode"] != "fault" and config.get("faults"):
        raise ValueError("faults are allowed only in fault mode")
    if config.get("expected_mujoco") not in {"3.3.7", "3.8.1"}:
        raise ValueError("expected_mujoco must explicitly select preregistered backend 3.3.7 or 3.8.1")
    config.setdefault("identity_protocol", str(REPO_ROOT / "experiments" / "vla" / "libero_protocol.json"))
    return config


def _verify_model_identity(checkpoint: Path, backbone: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "sentinel-libero-protocol-v1" or protocol.get("state") != "frozen_before_final_run":
        raise RuntimeError("identity protocol is not the frozen official LIBERO protocol")
    receipts: dict[str, Any] = {"checkpoint": {}, "backbone": {}}
    for label, root in (("checkpoint", checkpoint), ("backbone", backbone)):
        expected = protocol[label]["sha256"]
        for name, digest in expected.items():
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"required official {label} file is missing: {path}")
            actual = _file_sha256(path).removeprefix("sha256:")
            if actual != digest:
                raise RuntimeError(f"official {label} identity mismatch: {name}")
            receipts[label][name] = actual
    backbone_tree = _tree_identity(backbone)
    expected_tree = protocol["backbone"]["tree_identity"]
    for key in ("file_count", "total_bytes", "tree_sha256"):
        if backbone_tree[key] != expected_tree[key]:
            raise RuntimeError(f"official backbone tree mismatch: {key}")
    receipts["backbone_tree"] = backbone_tree
    return receipts


def _metrics(recorder: NativeRunRecorder, elapsed: float) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in recorder.episodes)

    def latency_summary(values: list[int]) -> dict[str, Any]:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "mean": (sum(ordered) / len(ordered) / 1e6) if ordered else None,
            "p50": (ordered[min(len(ordered) - 1, math.ceil(0.50 * len(ordered)) - 1)] / 1e6) if ordered else None,
            "p95": (ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)] / 1e6) if ordered else None,
            "p99": (ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)] / 1e6) if ordered else None,
        }

    return {
        "episodes": len(recorder.episodes),
        "task_successes": successes,
        "task_success_rate": successes / len(recorder.episodes) if recorder.episodes else 0.0,
        "crashes": sum(bool(row["crashed"]) for row in recorder.episodes),
        "env_steps": recorder.step if len(recorder.episodes) == 1 else sum(row["env_steps"] for row in recorder.episodes),
        "raw_policy_chunks": recorder.raw_chunks,
        "submitted": recorder.submitted,
        "accepted": recorder.accepted,
        "observed": recorder.observed,
        "postprocessor_to_env_exact_comparisons": recorder.shadow_comparisons,
        "postprocessor_to_env_max_abs_diff": recorder.shadow_max_abs_diff,
        "shadow_would_reject": recorder.shadow_rejections,
        "postprocessed_action_profile": {
            "count": recorder.postprocessed_actions,
            "component_min": recorder.action_component_min if recorder.postprocessed_actions else None,
            "component_max": recorder.action_component_max if recorder.postprocessed_actions else None,
        },
        "fault_attempts": len(recorder.fault_results),
        "fault_attempts_blocked": sum(bool(row["blocked"]) for row in recorder.fault_results),
        "latency_ms": {
            "policy_inference_to_raw_chunk": latency_summary(recorder.inference_latency_ns),
            "feedback_age_at_authorize": latency_summary(recorder.feedback_age_at_authorize_ns),
            "gateway_authorize": latency_summary(recorder.authorization_latency_ns),
            "gateway_submit_admission_before_writer": latency_summary(recorder.admission_latency_ns),
            "native_env_step": latency_summary(recorder.env_step_latency_ns),
            "active_submit_including_env_step": latency_summary(recorder.active_submit_total_ns),
        },
        "elapsed_seconds": elapsed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError("output-dir must not already exist")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    run_id = str(config.get("run_id") or f"native-vla-{int(started)}-{uuid.uuid4().hex[:8]}")
    checkpoint = Path(config["checkpoint"]).resolve()
    backbone = Path(config["backbone"]).resolve()
    if not checkpoint.is_dir() or not backbone.is_dir():
        raise FileNotFoundError("checkpoint and backbone directories must exist")
    identity_protocol_path = Path(config["identity_protocol"]).resolve()
    if _file_sha256(identity_protocol_path).removeprefix("sha256:") != OFFICIAL_IDENTITY_PROTOCOL_SHA256:
        raise RuntimeError("identity protocol does not match the pinned official source")
    identity_protocol = json.loads(identity_protocol_path.read_text(encoding="utf-8"))
    official_identity = _verify_model_identity(checkpoint, backbone, identity_protocol)
    dependency_ids: dict[str, str] = {}
    for prefix, root, names in (
        (
            "checkpoint",
            checkpoint,
            (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ),
        ),
        (
            "backbone",
            backbone,
            (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json",
            ),
        ),
    ):
        for name in names:
            path = root / name
            if path.is_file():
                dependency_ids[f"{prefix}/{name}"] = _file_sha256(path)
    dependency_ids["run/config"] = _file_sha256(args.config)
    dependency_ids["run/identity-protocol"] = _file_sha256(identity_protocol_path)
    if config.get("profile"):
        dependency_ids["run/profile"] = _file_sha256(Path(config["profile"]))
    dependencies_hash = "sha256:" + hashlib.sha256(_canonical(dependency_ids)).hexdigest()
    log = EventLog(run_id, maxlen=int(config.get("event_log_maxlen", 200_000)), schema_version="product-v1")
    profile_path = Path(config["profile"]) if config.get("profile") else None
    profile = load_profile(profile_path)
    gateway = NativeGateway(
        profile,
        log,
        lease_ttl_ns=int(float(config.get("lease_ttl_ms", 250.0)) * 1e6),
        max_feedback_age_ns=int(float(config.get("max_feedback_age_ms", 500.0)) * 1e6),
    )
    trace = JsonlTrace(args.output_dir / "native_trace.jsonl")
    render_episode_keys = {
        (int(item[0]), int(item[1]))
        for item in config.get(
            "render_episode_keys",
            [[int(config["task_ids"][0]), int(config["initial_state_indices"][0])]],
        )
    }
    record_pose_stride = int(config.get("record_pose_stride", 10))
    if record_pose_stride <= 0:
        raise ValueError("record_pose_stride must be positive")
    recorder = NativeRunRecorder(
        mode=config["mode"],
        gateway=gateway,
        log=log,
        trace=trace,
        fault_names=list(config.get("faults", [])),
        record_pose_stride=record_pose_stride,
        render_episode_keys=render_episode_keys,
        dependencies_hash=dependencies_hash,
    )
    fatal: dict[str, Any] | None = None
    envs: Any = None
    video_paths: list[tuple[str, Path]] = []

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import close_envs, eval_one
    from lerobot.utils.random_utils import set_seed
    from export_native_scene import NativeSceneExporter

    expected_software = identity_protocol["software"]
    if importlib.metadata.version("mujoco") != config["expected_mujoco"]:
        raise RuntimeError("installed MuJoCo does not match the preregistered expected_mujoco backend")
    for package in ("lerobot", "hf-libero", "robosuite", "num2words"):
        if importlib.metadata.version(package) != expected_software[package]:
            raise RuntimeError(f"official software identity mismatch: {package}")
    if not platform.python_version().startswith(expected_software["python"] + "."):
        raise RuntimeError("official Python identity mismatch")

    task_ids = [int(value) for value in config["task_ids"]]
    state_indices = [int(value) for value in config["initial_state_indices"]]
    seed_base = int(config.get("seed_base", 44021))
    execution_horizon = int(config.get("execution_horizon", 10))
    render_episodes = int(config.get("render_episodes", 1))
    torch_threads = int(config.get("torch_threads", 4))
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    env_cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=task_ids,
        fps=20,
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        max_parallel_tasks=1,
        observation_height=360,
        observation_width=360,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 50:
        raise RuntimeError("official checkpoint must declare chunk_size=n_action_steps=50")
    policy_cfg.device = "cuda"
    policy_cfg.vlm_model_name = str(backbone)
    policy_cfg.pretrained_path = checkpoint
    rename_map = config.get(
        "rename_map",
        {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        },
    )
    if rename_map != identity_protocol["evaluation"]["rename_map"]:
        raise RuntimeError("rename_map differs from the frozen official identity protocol")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    policy.config.n_action_steps = execution_horizon
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(backbone)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    asset_identity = _tree_identity(Path(get_libero_path("assets")))
    expected_asset_identity = identity_protocol["assets"]["resolved_tree"]
    for key in ("file_count", "total_bytes", "tree_sha256"):
        if asset_identity[key] != expected_asset_identity[key]:
            raise RuntimeError(f"official LIBERO asset tree mismatch: {key}")
    dependency_ids["libero/assets-tree"] = asset_identity["tree_sha256"]
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_root = Path(get_libero_path("init_states")) / "libero_spatial"
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_spatial"
    for task_id in task_ids:
        task = suite.tasks[task_id]
        dependency_ids[f"libero/task-{task_id}/init-states"] = _file_sha256(init_root / task.init_states_file)
        dependency_ids[f"libero/task-{task_id}/bddl"] = _file_sha256(bddl_root / task.bddl_file)
    for task_id, task in enumerate(suite.tasks):
        actual_init = _file_sha256(init_root / task.init_states_file).removeprefix("sha256:")
        actual_bddl = _file_sha256(bddl_root / task.bddl_file).removeprefix("sha256:")
        if actual_init != identity_protocol["assets"]["task_source_sha256"]["init_states"][str(task_id)]:
            raise RuntimeError(f"official init-state identity mismatch: task {task_id}")
        if actual_bddl != identity_protocol["assets"]["task_source_sha256"]["bddl"][str(task_id)]:
            raise RuntimeError(f"official BDDL identity mismatch: task {task_id}")
    dependency_ids.update(
        {
            "source/native_gateway.py": _file_sha256(REPO_ROOT / "src" / "sentinel_evc" / "native_gateway.py"),
            "source/libero_native_profile.py": _file_sha256(Path(__file__).with_name("libero_native_profile.py")),
            "source/run_sentinel_libero.py": _file_sha256(Path(__file__)),
            "source/export_native_scene.py": _file_sha256(Path(__file__).with_name("export_native_scene.py")),
            "runtime/env-config": "sha256:" + hashlib.sha256(_canonical(_jsonable(vars(env_cfg)))).hexdigest(),
            "runtime/rename-map": "sha256:" + hashlib.sha256(_canonical(rename_map)).hexdigest(),
        }
    )
    for package in ("mujoco", "lerobot", "hf-libero", "robosuite", "torch"):
        version = torch.__version__ if package == "torch" else importlib.metadata.version(package)
        dependency_ids[f"runtime/{package}"] = "sha256:" + hashlib.sha256(version.encode()).hexdigest()
    dependencies_hash = "sha256:" + hashlib.sha256(_canonical(dependency_ids)).hexdigest()
    recorder.dependencies_hash = dependencies_hash
    _wrap_policy(policy, recorder)
    traced_preprocessor = InputProcessor(preprocessor, recorder)
    traced_postprocessor = OutputProcessor(postprocessor, recorder)
    rendered = 0
    model_index: dict[str, Any] = {
        "schema": "sentinel-native-mujoco-model-index-v1",
        "models": [],
    }
    try:
        for task_id in task_ids:
            env = envs["libero_spatial"][task_id]
            available_initial_states = len(env.envs[0]._init_states)
            if available_initial_states <= max(state_indices):
                raise RuntimeError(
                    f"task {task_id} has {available_initial_states} initial states; "
                    f"requested index {max(state_indices)}"
                )
            instruction = str(env.call("task_description")[0])
            task_has_demo = any(key[0] == task_id for key in render_episode_keys)
            recorder.set_env(env, task_id, None)
            task_model_id: str | None = None

            def bind_live_scene() -> None:
                nonlocal task_model_id
                if not task_has_demo:
                    recorder.scene_exporter = None
                    return
                recorder.scene_exporter = NativeSceneExporter.from_environment(
                    env,
                    task_id,
                    task_suite="libero_spatial",
                    task_name=instruction,
                )
                live_model_id = recorder.scene_exporter.model_id
                if task_model_id is not None and live_model_id != task_model_id:
                    raise RuntimeError("hard reset changed the task model identity; one asset cannot represent both models")
                task_model_id = live_model_id

            _wrap_env(env, recorder, bind_live_scene)
            for state_index in state_indices:
                seed = seed_base + state_index - state_indices[0]
                set_seed(seed)
                policy.config.n_action_steps = execution_horizon
                policy.reset()
                env.envs[0].init_state_id = state_index
                recorder.plan_episode(state_index, seed, instruction)
                episode_result: dict[str, Any] | None = None
                episode_error: Exception | None = None
                try:
                    render = (task_id, state_index) in render_episode_keys and rendered < render_episodes
                    video_dir = args.output_dir / "videos" if render else None
                    episode_result = eval_one(
                        env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=traced_preprocessor,
                        postprocessor=traced_postprocessor,
                        n_episodes=1,
                        max_episodes_rendered=1 if render else 0,
                        videos_dir=video_dir,
                        return_episode_data=False,
                        start_seed=seed,
                    )
                    if render:
                        rendered += 1
                        generated = [Path(path) for path in episode_result.get("video_paths", [])]
                        if generated:
                            asset_name = f"episode-task{task_id:02d}-state{state_index:02d}.mp4"
                            video_paths.append((asset_name, generated[0]))
                            assert recorder.current_episode is not None and recorder.replay_episode is not None
                            recorder.current_episode["video_asset"] = asset_name
                            recorder.replay_episode["video_asset"] = asset_name
                except Exception as error:
                    episode_error = error
                recorder.finish_episode(episode_result, episode_error)
                if episode_error is not None:
                    raise episode_error
            if task_has_demo:
                scene_exporter = recorder.scene_exporter
                if scene_exporter is None:
                    raise RuntimeError("render task completed without a live native scene exporter")
                scene_asset = f"viewer-model-task{task_id}.json"
                (args.output_dir / scene_asset).write_text(
                    json.dumps(scene_exporter.export_scene(), separators=(",", ":"), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                model_index["models"].append(
                    {
                        "task_id": task_id,
                        "asset": scene_asset,
                        "modelId": scene_exporter.model_id,
                        "body_order": "scene.bodies sorted by bodyIndex; frame pose arrays use the same indices",
                    }
                )
        if model_index["models"]:
            (args.output_dir / "viewer-model.json").write_text(
                json.dumps(model_index, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as error:
        fatal = {"error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if envs is not None:
            close_envs(envs)
        trace.close()

    elapsed = time.time() - started
    metrics = _metrics(recorder, elapsed)
    faults_complete = all(
        set(recorder.fault_names).issubset({attempt["fault"] for attempt in episode.get("fault_attempts", [])})
        for episode in recorder.episodes
    )
    status = "complete" if fatal is None and (config["mode"] != "fault" or faults_complete) else "failed"
    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for package in ("lerobot", "hf-libero", "robosuite", "mujoco", "num2words"):
        software[package] = importlib.metadata.version(package)
    result = {
        "schema": "sentinel-native-vla-run-v1",
        "run_id": run_id,
        "mode": config["mode"],
        "status": status,
        "profile": profile.summary(),
        "profile_hash": profile.digest,
        "episodes": recorder.episodes,
        "metrics": metrics,
        "fault_attempts": recorder.fault_results,
        "physical_validation": {
            "status": "unsupported",
            "reason": "native gateway validates request schema, finite values and declared component bounds only",
        },
        "worldguard": {
            "status": "unsupported",
            "reason": "no WorldGuard model is bound to this LIBERO native action profile",
        },
        "software": software,
        "dependencies": {
            "hash": dependencies_hash,
            "files": dependency_ids,
            "libero_assets": asset_identity,
        },
        "official_identity": {
            "protocol": str(identity_protocol_path),
            "protocol_sha256": _file_sha256(identity_protocol_path),
            "model": official_identity,
            "expected_mujoco": config["expected_mujoco"],
        },
        "checkpoint": {"path": str(checkpoint), "config_sha256": dependency_ids.get("checkpoint/config.json")},
        "backbone": {"path": str(backbone), "config_sha256": dependency_ids.get("backbone/config.json")},
        "configuration": {
            "task_ids": task_ids,
            "initial_state_indices": state_indices,
            "execution_horizon": execution_horizon,
            "seed_base": seed_base,
            "lease_ttl_ms": float(config.get("lease_ttl_ms", 250.0)),
            "max_feedback_age_ms": float(config.get("max_feedback_age_ms", 500.0)),
            "torch_threads": torch.get_num_threads(),
            "record_pose_stride": record_pose_stride,
            "render_episode_keys": [list(item) for item in sorted(render_episode_keys)],
        },
        "fatal_failure": fatal,
        "claim_limits": [
            "official SmolVLA checkpoint inference and native LIBERO/Panda env.step were used when status is complete",
            "active/fault modes make the gateway the only env.step writer",
            "shadow mode observes exact postprocessor-to-env bytes and does not establish intervention efficacy",
            "request authorization is not collision, dynamics, physical-stop or functional-safety validation",
            "task success is the official LIBERO result; fault attempts are excluded from task success",
        ],
    }
    replay = {"schema": "sentinel-native-vla-replay-v1", "run_id": run_id, "episodes": recorder.replay_episodes}
    artifacts: dict[str, bytes] = {
        "result.json": json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n",
        "replay.json": json.dumps(replay, separators=(",", ":"), sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n",
        "native_trace.jsonl": (args.output_dir / "native_trace.jsonl").read_bytes(),
        "config.json": args.config.read_bytes(),
    }
    for asset_name, video_path in video_paths:
        if video_path.is_file():
            artifacts[asset_name] = video_path.read_bytes()
    for scene_asset in sorted(args.output_dir.glob("viewer-model*.json")):
        artifacts[scene_asset.name] = scene_asset.read_bytes()
    bundle_receipt = build_bundle(log, str(args.output_dir), artifacts=artifacts)
    verified, verification_detail = verify_bundle(
        bundle_receipt["bundle_dir"], bundle_receipt["public_key"], run_id
    )
    receipt = {
        "schema": "sentinel-native-vla-receipt-v1",
        "run_id": run_id,
        "status": status,
        "bundle": bundle_receipt,
        "verified": verified,
        "verification_detail": verification_detail,
        "result_sha256": _file_sha256(Path(bundle_receipt["bundle_dir"]) / "result.json"),
        "replay_sha256": _file_sha256(Path(bundle_receipt["bundle_dir"]) / "replay.json"),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    archive = args.output_dir / "sentinel-native-vla-bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for item in sorted((args.output_dir / "bundle").iterdir()):
            handle.write(item, f"bundle/{item.name}")
        handle.write(args.output_dir / "anchors" / "demo.public", "anchors/demo.public")
    print(json.dumps({"status": status, "run_id": run_id, "metrics": metrics, "verified": verified}, indent=2))
    return 0 if status == "complete" and verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
