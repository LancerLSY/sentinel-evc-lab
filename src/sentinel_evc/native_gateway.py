"""Native action-request authorization for external robot environments.

This module deliberately does not reinterpret a native 7-D action as the
numeric three-dimensional :class:`~sentinel_evc.contracts.Plan`.  It binds an
opaque, explicitly profiled environment request to fresh feedback, execution
context, step and revocation generation, then permits exactly one submission.

The gateway checks schema, finite values and declared component bounds only.
It does not provide a collision, dynamics, WorldGuard, physical-stop or
functional-safety certificate.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .contracts import canonical_json, sha256_hex
from .events import EventLog


class NativeGatewayDenied(RuntimeError):
    """A native request was refused before the environment writer ran."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _bounded_text(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty string of at most {maximum} characters")
    return value


def _digest(value: Any, name: str) -> str:
    value = _bounded_text(value, name, 71)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 digest") from exc
    return value


def _flatten_scalars(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_scalars(item))
        return result
    return [value]


def _native_request_identity(action: Any) -> tuple[tuple[float, ...], tuple[int, ...], str, str]:
    """Read a tensor/array without importing its heavy implementation.

    The digest matches the runner's C-order identity and is recomputed both at
    authorization and immediately before writer admission.
    """
    array = action
    if hasattr(array, "detach"):
        array = array.detach().to("cpu").numpy()
    if not hasattr(array, "shape") or not hasattr(array, "dtype") or not hasattr(array, "tolist"):
        raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", "native request must expose shape/dtype/tolist")
    shape = tuple(int(value) for value in array.shape)
    dtype = str(array.dtype)
    try:
        raw = array.tobytes(order="C")
    except TypeError:
        raw = array.tobytes()
    if not isinstance(raw, bytes):
        raw = bytes(raw)
    digest = hashlib.sha256()
    digest.update(dtype.encode())
    digest.update(b"\0")
    digest.update(canonical_json(list(shape)))
    digest.update(b"\0")
    digest.update(raw)
    values = tuple(_flatten_scalars(array.tolist()))
    return values, shape, dtype, "sha256:" + digest.hexdigest()


def _owned_native_request(action: Any) -> Any:
    """Copy caller-owned array storage before admission and writer dispatch.

    Array implementations remain in the policy environment. Their C-order
    copy must own its storage; a caller can then change the original array
    without changing the request whose identity the gateway checks.
    """
    array = action
    if hasattr(array, "detach"):
        array = array.detach().to("cpu").numpy()
    if not hasattr(array, "copy"):
        raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", "native request must support an independent copy")
    try:
        owned = array.copy(order="C")
    except TypeError:
        owned = array.copy()
    if owned is array:
        raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", "native request copy reused caller storage")
    return owned


@dataclass(frozen=True)
class NativeActionProfile:
    """Declared support boundary for a native environment request."""

    schema_id: str
    component_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    semantics: str
    request_shape: tuple[int, ...] = (1, 7)
    request_dtypes: tuple[str, ...] = ("float32", "float64")
    unsupported_checks: tuple[str, ...] = (
        "collision_clearance",
        "dynamics",
        "physical_stop",
        "worldguard_prediction",
    )

    def __post_init__(self) -> None:
        _bounded_text(self.schema_id, "schema_id")
        _bounded_text(self.semantics, "semantics", 1024)
        names = tuple(self.component_names)
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        unsupported = tuple(self.unsupported_checks)
        request_shape = tuple(self.request_shape)
        request_dtypes = tuple(self.request_dtypes)
        if not names or len(names) > 64 or len(names) != len(lower) or len(names) != len(upper):
            raise ValueError("profile component names and bounds must have equal non-zero length")
        for index, (name, lo, hi) in enumerate(zip(names, lower, upper)):
            _bounded_text(name, f"component_names[{index}]", 64)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError(f"invalid bounds for component {name}")
        for index, name in enumerate(unsupported):
            _bounded_text(name, f"unsupported_checks[{index}]", 128)
        if not request_shape or len(request_shape) > 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in request_shape
        ):
            raise ValueError("request_shape must contain positive integer dimensions")
        if math.prod(request_shape) != len(names):
            raise ValueError("request_shape element count must equal component count")
        if not request_dtypes or len(request_dtypes) > 16:
            raise ValueError("request_dtypes must be non-empty")
        for index, name in enumerate(request_dtypes):
            _bounded_text(name, f"request_dtypes[{index}]", 32)
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "unsupported_checks", unsupported)
        object.__setattr__(self, "request_shape", request_shape)
        object.__setattr__(self, "request_dtypes", request_dtypes)

    @property
    def digest(self) -> str:
        return sha256_hex(self.summary())

    def summary(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "component_names": list(self.component_names),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "semantics": self.semantics,
            "request_shape": list(self.request_shape),
            "request_dtypes": list(self.request_dtypes),
            "supported_checks": ["shape", "finite", "declared_component_bounds"],
            "unsupported_checks": list(self.unsupported_checks),
        }

    def validate(self, action: Any) -> tuple[float, ...]:
        if not isinstance(action, (list, tuple)) or len(action) != len(self.component_names):
            raise NativeGatewayDenied(
                "UNSUPPORTED_ACTION_SCHEMA",
                f"{self.schema_id} requires exactly {len(self.component_names)} scalar components",
            )
        values: list[float] = []
        for index, (raw, lo, hi) in enumerate(zip(action, self.lower, self.upper)):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", f"component {index} is not numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise NativeGatewayDenied("NONFINITE_ACTION", f"component {index} is not finite")
            if value < lo or value > hi:
                raise NativeGatewayDenied(
                    "ACTION_OUT_OF_PROFILE",
                    f"{self.component_names[index]}={value:.12g} outside [{lo:.12g}, {hi:.12g}]",
                )
            values.append(value)
        return tuple(values)


@dataclass(frozen=True)
class NativeSnapshot:
    feedback_id: str
    observation_hash: str
    step: int
    capture_mono_ns: int

    def __post_init__(self) -> None:
        _bounded_text(self.feedback_id, "feedback_id")
        _digest(self.observation_hash, "observation_hash")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if isinstance(self.capture_mono_ns, bool) or not isinstance(self.capture_mono_ns, int) or self.capture_mono_ns < 0:
            raise ValueError("capture_mono_ns must be a non-negative integer")

    @property
    def digest(self) -> str:
        return sha256_hex(self.summary())

    def summary(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "observation_hash": self.observation_hash,
            "step": self.step,
            "capture_mono_ns": self.capture_mono_ns,
        }


@dataclass(frozen=True)
class NativeContext:
    robot_id: str
    boot_id: str
    episode_id: str
    scene_id: str
    controller_id: str
    task_phase: str
    queue_rev: int
    dependencies_hash: str = "sha256:" + "0" * 64
    chunk_id: int = -1
    action_index_in_chunk: int = -1
    policy_input_hash: str = "sha256:" + "0" * 64
    policy_input_step: int = -1

    def __post_init__(self) -> None:
        for name in ("robot_id", "boot_id", "episode_id", "scene_id", "controller_id", "task_phase"):
            _bounded_text(getattr(self, name), name)
        if isinstance(self.queue_rev, bool) or not isinstance(self.queue_rev, int) or self.queue_rev < 0:
            raise ValueError("queue_rev must be a non-negative integer")
        _digest(self.dependencies_hash, "dependencies_hash")
        for name in ("chunk_id", "action_index_in_chunk", "policy_input_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(f"{name} must be an integer >= -1")
        _digest(self.policy_input_hash, "policy_input_hash")

    @property
    def digest(self) -> str:
        return sha256_hex(self.summary())

    def summary(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "boot_id": self.boot_id,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "controller_id": self.controller_id,
            "task_phase": self.task_phase,
            "queue_rev": self.queue_rev,
            "dependencies_hash": self.dependencies_hash,
            "chunk_id": self.chunk_id,
            "action_index_in_chunk": self.action_index_in_chunk,
            "policy_input_hash": self.policy_input_hash,
            "policy_input_step": self.policy_input_step,
        }


@dataclass(frozen=True)
class NativePermit:
    permit_id: str
    request_hash: str
    request_bytes_hash: str
    profile_hash: str
    snapshot_hash: str
    context_hash: str
    step: int
    generation: int
    issued_mono_ns: int
    deadline_mono_ns: int
    authenticator: str

    def signed_fields(self) -> dict[str, Any]:
        return {
            "permit_id": self.permit_id,
            "request_hash": self.request_hash,
            "request_bytes_hash": self.request_bytes_hash,
            "profile_hash": self.profile_hash,
            "snapshot_hash": self.snapshot_hash,
            "context_hash": self.context_hash,
            "step": self.step,
            "generation": self.generation,
            "issued_mono_ns": self.issued_mono_ns,
            "deadline_mono_ns": self.deadline_mono_ns,
        }


class NativeGateway:
    """Single-process, single-writer native request gateway."""

    def __init__(
        self,
        profile: NativeActionProfile,
        event_log: EventLog,
        *,
        lease_ttl_ns: int = 250_000_000,
        max_feedback_age_ns: int = 500_000_000,
        secret: bytes | None = None,
    ) -> None:
        if not isinstance(profile, NativeActionProfile):
            raise TypeError("profile must be NativeActionProfile")
        if not isinstance(event_log, EventLog) or event_log.schema_version != "product-v1":
            raise TypeError("native gateway requires a product-v1 EventLog")
        for value, name in ((lease_ttl_ns, "lease_ttl_ns"), (max_feedback_age_ns, "max_feedback_age_ns")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.profile = profile
        self.log = event_log
        self.lease_ttl_ns = lease_ttl_ns
        self.max_feedback_age_ns = max_feedback_age_ns
        self._secret = secret or secrets.token_bytes(32)
        self._generation = 0
        self._snapshot: NativeSnapshot | None = None
        self._context: NativeContext | None = None
        self._permits: dict[str, NativePermit] = {}
        self._consumed: set[str] = set()
        self._committed_snapshots: set[str] = set()
        self._lock = threading.RLock()
        self._writer_condition = threading.Condition(self._lock)
        self._writer_state = "idle"

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def snapshot(self) -> NativeSnapshot | None:
        with self._lock:
            return self._snapshot

    @property
    def context(self) -> NativeContext | None:
        with self._lock:
            return self._context

    def _deny(self, code: str, detail: str, **payload: Any) -> None:
        self.log.append("OUTCOME", decision="REJECTED", code=code, detail=detail, **payload)
        raise NativeGatewayDenied(code, detail)

    def observe_feedback(self, snapshot: NativeSnapshot, context: NativeContext) -> None:
        """Install actual reset/step feedback as the only authorizable snapshot."""
        if not isinstance(snapshot, NativeSnapshot) or not isinstance(context, NativeContext):
            raise TypeError("snapshot/context types are required")
        with self._lock:
            episode_changed = self._context is not None and context.episode_id != self._context.episode_id
            if self._snapshot is not None and snapshot.capture_mono_ns <= self._snapshot.capture_mono_ns:
                self._deny("OLD_FEEDBACK", "feedback timestamp was repeated or moved backwards")
            if self._snapshot is None or episode_changed:
                if snapshot.step != 0 or context.queue_rev != 0:
                    self._deny("OLD_FEEDBACK", "new episode feedback must start at step/queue revision zero")
            elif snapshot.step != self._snapshot.step + 1:
                self._deny("OLD_FEEDBACK", "same-episode feedback must advance exactly one step")
            if context.queue_rev != snapshot.step:
                self._deny("CONTEXT_CHANGED", "queue revision must equal the observed environment step")
            if episode_changed:
                self._generation += 1
                self._permits.clear()
                self._consumed.clear()
                self._committed_snapshots.clear()
            self._snapshot = snapshot
            self._context = context
            self.log.append(
                "OBSERVED",
                snapshot=snapshot.summary(),
                snapshot_hash=snapshot.digest,
                context=context.summary(),
                context_hash=context.digest,
                generation=self._generation,
                observation_source="actual_env_reset_or_step_return",
            )

    def revoke(self, reason: str, *, now_ns: int | None = None) -> int:
        _bounded_text(reason, "reason", 512)
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._writer_condition:
            while self._writer_state == "admitting":
                self._writer_condition.wait()
            self._generation += 1
            self.log.append("REVOKE", reason=reason, generation=self._generation, mono_ns=now)
            return self._generation

    def refresh_context(self, context: NativeContext, *, snapshot_hash: str) -> None:
        """Bind a chunk/action cursor without pretending that feedback changed."""
        snapshot_hash = _digest(snapshot_hash, "snapshot_hash")
        with self._lock:
            if self._snapshot is None or self._context is None or self._snapshot.digest != snapshot_hash:
                self._deny("FEEDBACK_CHANGED", "context refresh did not bind the current feedback")
            previous = self._context.summary()
            current = context.summary()
            for key in previous:
                if key not in {"chunk_id", "action_index_in_chunk", "policy_input_hash", "policy_input_step"} and previous[key] != current[key]:
                    self._deny("CONTEXT_CHANGED", f"context refresh changed stable field {key}")
            self._context = context

    def _authenticator(self, fields: dict[str, Any]) -> str:
        return "hmac-sha256:" + hmac.new(self._secret, canonical_json(fields), hashlib.sha256).hexdigest()

    def authorize(
        self,
        action: Any,
        *,
        snapshot: NativeSnapshot,
        context: NativeContext,
        now_ns: int | None = None,
    ) -> NativePermit:
        """Authorize one exact postprocessed environment request."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        raw_values, request_shape, request_dtype, request_bytes_hash = _native_request_identity(action)
        if request_shape != self.profile.request_shape or request_dtype not in self.profile.request_dtypes:
            self._deny(
                "UNSUPPORTED_ACTION_SCHEMA",
                f"request array {request_shape}/{request_dtype} is outside the declared profile",
            )
        try:
            values = self.profile.validate(raw_values)
        except NativeGatewayDenied as exc:
            self._deny(exc.code, exc.detail, request_bytes_hash=request_bytes_hash)
        with self._lock:
            if self.log.has_gap:
                self._deny("EVIDENCE_GAP", "event log has a visible gap")
            if self._snapshot is None or self._context is None:
                self._deny("NO_FEEDBACK", "no actual environment feedback is installed")
            if snapshot != self._snapshot:
                self._deny("OLD_FEEDBACK", "authorization did not use the current feedback snapshot")
            if context != self._context:
                self._deny("CONTEXT_CHANGED", "authorization did not use the current execution context")
            if context.policy_input_step != snapshot.step or context.policy_input_hash == "sha256:" + "0" * 64 or context.chunk_id < 0 or context.action_index_in_chunk < 0:
                self._deny("POLICY_INPUT_CHANGED", "authorization requires the current step's actual policy input")
            if snapshot.step < 0 or now < snapshot.capture_mono_ns:
                self._deny("INVALID_CLOCK", "feedback timestamp is ahead of the authorization clock")
            if now - snapshot.capture_mono_ns > self.max_feedback_age_ns:
                self._deny("STALE_FEEDBACK", "actual environment feedback exceeded the configured age")
            request = {
                "schema_id": self.profile.schema_id,
                "values": list(values),
                "request_bytes_hash": request_bytes_hash,
                "request_shape": list(request_shape),
                "request_dtype": request_dtype,
            }
            request_hash = sha256_hex(request)
            self.log.append(
                "CERTIFICATE",
                status="SUPPORTED_PROFILE_REQUEST",
                request_hash=request_hash,
                request_bytes_hash=request_bytes_hash,
                profile=self.profile.summary(),
                profile_hash=self.profile.digest,
                limitation="schema/finite/component-bound check only",
            )
            issued = now
            fields = {
                "permit_id": secrets.token_hex(16),
                "request_hash": request_hash,
                "request_bytes_hash": request_bytes_hash,
                "profile_hash": self.profile.digest,
                "snapshot_hash": snapshot.digest,
                "context_hash": context.digest,
                "step": snapshot.step,
                "generation": self._generation,
                "issued_mono_ns": issued,
                "deadline_mono_ns": issued + self.lease_ttl_ns,
            }
            permit = NativePermit(**fields, authenticator=self._authenticator(fields))
            self._permits[permit.permit_id] = permit
            self.log.append(
                "PREPARE",
                permit_id=permit.permit_id,
                request_hash=permit.request_hash,
                request_bytes_hash=permit.request_bytes_hash,
                snapshot_hash=permit.snapshot_hash,
                context_hash=permit.context_hash,
                step=permit.step,
                generation=permit.generation,
                issued_mono_ns=permit.issued_mono_ns,
                deadline_mono_ns=permit.deadline_mono_ns,
            )
            self.log.append("LEASE", permit_id=permit.permit_id, state="ISSUED", one_time=True)
            return permit

    def submit(
        self,
        permit: NativePermit,
        action: Any,
        *,
        snapshot: NativeSnapshot,
        context: NativeContext,
        writer: Callable[[Any, Callable[[], None]], Any],
        now_ns: int | None = None,
    ) -> Any:
        """Consume a permit and invoke the sole environment writer once."""
        clock_origin = time.monotonic_ns()
        now = clock_origin if now_ns is None else now_ns
        owned_action = _owned_native_request(action)
        raw_values, request_shape, request_dtype, request_bytes_hash = _native_request_identity(owned_action)
        if request_shape != self.profile.request_shape or request_dtype not in self.profile.request_dtypes:
            self._deny(
                "UNSUPPORTED_ACTION_SCHEMA",
                f"request array {request_shape}/{request_dtype} is outside the declared profile",
                permit_id=getattr(permit, "permit_id", None),
            )
        try:
            values = self.profile.validate(raw_values)
        except NativeGatewayDenied as exc:
            self._deny(exc.code, exc.detail, permit_id=getattr(permit, "permit_id", None))
        with self._lock:
            if not isinstance(permit, NativePermit):
                self._deny("INVALID_PERMIT", "permit type is invalid")
            registered = self._permits.get(permit.permit_id)
            if registered is None:
                if permit.permit_id in self._consumed:
                    self._deny("LEASE_REPLAY", "permit was already consumed", permit_id=permit.permit_id)
                self._deny("INVALID_PERMIT", "permit is not registered", permit_id=permit.permit_id)
            if registered != permit or not hmac.compare_digest(permit.authenticator, self._authenticator(permit.signed_fields())):
                self._deny("INVALID_PERMIT", "permit content or authenticator changed", permit_id=permit.permit_id)
            if now > permit.deadline_mono_ns:
                self._permits.pop(permit.permit_id, None)
                self._deny("LEASE_EXPIRED", "permit deadline passed", permit_id=permit.permit_id)
            if permit.generation != self._generation:
                self._permits.pop(permit.permit_id, None)
                self._deny("GENERATION_REVOKED", "permit generation is no longer current", permit_id=permit.permit_id)
            if self._snapshot != snapshot or permit.snapshot_hash != snapshot.digest:
                self._deny("FEEDBACK_CHANGED", "feedback changed after authorization", permit_id=permit.permit_id)
            if self._context != context or permit.context_hash != context.digest:
                self._deny("CONTEXT_CHANGED", "execution context changed after authorization", permit_id=permit.permit_id)
            if now < snapshot.capture_mono_ns or now - snapshot.capture_mono_ns > self.max_feedback_age_ns:
                self._deny("STALE_FEEDBACK", "feedback is too old at submission", permit_id=permit.permit_id)
            request_hash = sha256_hex({
                "schema_id": self.profile.schema_id,
                "values": list(values),
                "request_bytes_hash": request_bytes_hash,
                "request_shape": list(request_shape),
                "request_dtype": request_dtype,
            })
            if request_hash != permit.request_hash or request_bytes_hash != permit.request_bytes_hash:
                self._deny("ACTION_REPLACED", "submitted request differs from the authorized request", permit_id=permit.permit_id)
            if snapshot.digest in self._committed_snapshots:
                self._deny("FEEDBACK_REUSED", "current feedback already authorized one native submission", permit_id=permit.permit_id)
            if self._writer_state != "idle":
                self._deny("WRITER_REENTRY", "native environment writer is already active", permit_id=permit.permit_id)
            self._permits.pop(permit.permit_id, None)
            self._consumed.add(permit.permit_id)
            self._committed_snapshots.add(snapshot.digest)
            self._writer_state = "admitting"
            self.log.append(
                "COMMIT",
                permit_id=permit.permit_id,
                request_hash=request_hash,
                request_bytes_hash=request_bytes_hash,
                step=permit.step,
                generation=permit.generation,
                mono_ns=now,
            )
            self.log.append(
                "DISPATCH",
                permit_id=permit.permit_id,
                request_hash=request_hash,
                request_bytes_hash=request_bytes_hash,
                step=permit.step,
                writer="NativeGateway.submit",
            )
        def entered() -> None:
            with self._writer_condition:
                if self._writer_state != "admitting":
                    raise RuntimeError("native writer entered outside admission")
                entry_now = now + time.monotonic_ns() - clock_origin
                if permit.generation != self._generation:
                    self._deny("GENERATION_REVOKED", "permit generation changed before writer entry", permit_id=permit.permit_id)
                if self._snapshot != snapshot or permit.snapshot_hash != snapshot.digest:
                    self._deny("FEEDBACK_CHANGED", "feedback changed before writer entry", permit_id=permit.permit_id)
                if self._context != context or permit.context_hash != context.digest:
                    self._deny("CONTEXT_CHANGED", "execution context changed before writer entry", permit_id=permit.permit_id)
                if entry_now > permit.deadline_mono_ns:
                    self._deny("LEASE_EXPIRED", "permit deadline passed before writer entry", permit_id=permit.permit_id)
                if entry_now < snapshot.capture_mono_ns or entry_now - snapshot.capture_mono_ns > self.max_feedback_age_ns:
                    self._deny("STALE_FEEDBACK", "feedback became too old before writer entry", permit_id=permit.permit_id)
                self._writer_state = "entered"
                self._writer_condition.notify_all()

        try:
            result = writer(owned_action, entered)
            with self._lock:
                if self._writer_state == "admitting":
                    raise RuntimeError("native writer did not acknowledge entry")
        except Exception as exc:
            self.log.append(
                "CONTROLLER_ACK",
                permit_id=permit.permit_id,
                accepted=False,
                status="writer_raised",
                error_type=type(exc).__name__,
            )
            raise
        else:
            self.log.append(
                "CONTROLLER_ACK",
                permit_id=permit.permit_id,
                accepted=True,
                status="env_step_returned",
            )
            return result
        finally:
            with self._writer_condition:
                self._writer_state = "idle"
                self._writer_condition.notify_all()


__all__ = [
    "NativeActionProfile",
    "NativeContext",
    "NativeGateway",
    "NativeGatewayDenied",
    "NativePermit",
    "NativeSnapshot",
]
