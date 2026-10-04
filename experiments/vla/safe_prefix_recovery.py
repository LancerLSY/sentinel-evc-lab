#!/usr/bin/env python3
"""Certified unchanged-prefix recovery for a denied 10-action forecast."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Support both direct experiment execution and package imports.
    from .panda_geometry import (
        GeometryCertificate,
        GeometryDecision,
        PandaGeometryProfile,
        SceneSnapshot,
        certify_trajectory,
    )
except ImportError:  # pragma: no cover - direct script import on the GPU host
    from panda_geometry import (
        GeometryCertificate,
        GeometryDecision,
        PandaGeometryProfile,
        SceneSnapshot,
        certify_trajectory,
    )


@dataclass(frozen=True)
class SafePrefixAttempt:
    prefix_length: int
    qpos_sample_count: int
    action_sha256: str
    action_nbytes: int
    geometry_allowed: bool
    contact_veto_enabled: bool
    contact_vetoed: bool
    contact_substeps_checked: int
    unwanted_contact_substeps: int
    unwanted_contact_records: int
    unwanted_contact_evidence: tuple[dict[str, Any], ...]
    allowed: bool
    status: str
    method: str
    reason: str
    min_margin: float | None
    unknown_pairs: tuple[str, ...]
    violations: tuple[tuple[int, int, float], ...]
    work: Mapping[str, int]
    latency_ns: int
    certificate_digest: str | None


@dataclass(frozen=True)
class SafePrefixRecoveryResult:
    success: bool
    prefix_length: int
    native_actions: np.ndarray | None
    qpos_samples: np.ndarray | None
    action_end_indices: tuple[int, ...]
    certificate: GeometryCertificate | None
    decision: GeometryDecision | None
    attempts: tuple[SafePrefixAttempt, ...]
    total_work: Mapping[str, int]
    total_latency_ns: int
    reason: str


def _attempt_record(
    prefix_length: int,
    qpos_count: int,
    action_bytes: bytes,
    decision: GeometryDecision,
    *,
    contact_veto_enabled: bool,
    contact_substeps_checked: int,
    unwanted_contact_evidence: tuple[dict[str, Any], ...],
) -> SafePrefixAttempt:
    unwanted_contact_records = sum(len(item["contacts"]) for item in unwanted_contact_evidence)
    contact_vetoed = bool(contact_veto_enabled and unwanted_contact_evidence)
    return SafePrefixAttempt(
        prefix_length=prefix_length,
        qpos_sample_count=qpos_count,
        action_sha256=hashlib.sha256(action_bytes).hexdigest(),
        action_nbytes=len(action_bytes),
        geometry_allowed=decision.allowed,
        contact_veto_enabled=contact_veto_enabled,
        contact_vetoed=contact_vetoed,
        contact_substeps_checked=contact_substeps_checked,
        unwanted_contact_substeps=len(unwanted_contact_evidence),
        unwanted_contact_records=unwanted_contact_records,
        unwanted_contact_evidence=unwanted_contact_evidence,
        allowed=bool(decision.allowed and not contact_vetoed),
        status=decision.status,
        method=decision.method,
        reason="known unwanted forecast contact veto" if contact_vetoed else decision.reason,
        min_margin=decision.min_margin,
        unknown_pairs=tuple(decision.unknown_pairs),
        violations=tuple(
            (record.geom_a, record.geom_b, record.distance)
            for record in decision.violations
        ),
        work=dict(decision.work),
        latency_ns=decision.latency_ns,
        certificate_digest=(
            decision.certificate.digest if decision.certificate is not None else None
        ),
    )


def _sum_work(attempts: Sequence[SafePrefixAttempt]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for attempt in attempts:
        for key, value in attempt.work.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def recover_safe_prefix(
    model: Any,
    data: Any,
    profile: PandaGeometryProfile,
    final_native: Any,
    final_qpos: Any,
    final_action_end_indices: Sequence[int],
    scene_snapshot: SceneSnapshot,
    *,
    prefix_lengths: Sequence[int] = (5, 2, 1),
    forecast_substep_unwanted_contacts: Sequence[Sequence[Any]] | None = None,
    veto_unwanted_contacts: bool = False,
) -> SafePrefixRecoveryResult:
    """Return the longest certified member of a fixed unchanged-prefix set.

    This function only slices the supplied candidate. It does not synthesize an
    action, change collision policy, widen tolerances, or dispatch movement.
    """
    started = time.perf_counter_ns()
    native = np.asarray(final_native)
    qpos = np.asarray(final_qpos)
    ends = tuple(int(value) for value in final_action_end_indices)
    prefixes = tuple(int(value) for value in prefix_lengths)
    width = len(profile.moving_qpos_indices)

    if width != 9 or profile.coordinate_units != ("rad",) * 7 + ("m",) * 2:
        raise ValueError("A6 prefix recovery requires the seven-hinge plus two-slide profile")
    if native.shape != (1, 10, 7) or native.dtype.kind != "f" or not np.all(np.isfinite(native)):
        raise ValueError("final_native must be a finite floating array with shape (1, 10, 7)")
    if qpos.shape != (251, 9) or qpos.dtype.kind != "f" or not np.all(np.isfinite(qpos)):
        raise ValueError("final_qpos must be a finite floating array with shape (251, 9)")
    if len(ends) != 10 or any(value < 1 for value in ends):
        raise ValueError("final_action_end_indices must contain ten positive sample indices")
    if any(right <= left for left, right in zip(ends, ends[1:])) or ends[-1] != len(qpos) - 1:
        raise ValueError("action end indices must be strictly increasing and end at final_qpos[-1]")
    if prefixes != (5, 2, 1):
        raise ValueError("the bounded recovery set is exactly the descending prefixes (5, 2, 1)")
    contacts: tuple[tuple[Any, ...], ...] = ()
    if veto_unwanted_contacts:
        if forecast_substep_unwanted_contacts is None:
            raise ValueError("contact veto requires full forecast_substep_unwanted_contacts evidence")
        if len(forecast_substep_unwanted_contacts) != len(qpos) - 1:
            raise ValueError(
                "forecast_substep_unwanted_contacts must contain one entry per forecast physics substep"
            )
        normalized: list[tuple[Any, ...]] = []
        for index, row in enumerate(forecast_substep_unwanted_contacts):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise ValueError(f"forecast contact evidence at physics substep {index} must be a sequence")
            normalized.append(tuple(row))
        contacts = tuple(normalized)

    attempts: list[SafePrefixAttempt] = []
    last_decision: GeometryDecision | None = None
    for prefix_length in prefixes:
        end = ends[prefix_length - 1]
        prefix_native = np.ascontiguousarray(native[:, :prefix_length, :])
        prefix_qpos = np.ascontiguousarray(qpos[:end + 1])
        prefix_ends = ends[:prefix_length]
        action_bytes = prefix_native.tobytes(order="C")
        decision = certify_trajectory(
            model,
            data,
            profile,
            prefix_qpos,
            action_bytes=action_bytes,
            scene_snapshot=scene_snapshot,
        )
        last_decision = decision
        prefix_contact_evidence = tuple(
            {"physics_substep": index, "contacts": row}
            for index, row in enumerate(contacts[:end]) if row
        )
        attempt = _attempt_record(
            prefix_length, len(prefix_qpos), action_bytes, decision,
            contact_veto_enabled=bool(veto_unwanted_contacts),
            contact_substeps_checked=end if veto_unwanted_contacts else 0,
            unwanted_contact_evidence=prefix_contact_evidence,
        )
        attempts.append(attempt)
        if attempt.allowed and decision.status == "certified" and decision.certificate is not None:
            return SafePrefixRecoveryResult(
                success=True,
                prefix_length=prefix_length,
                native_actions=prefix_native,
                qpos_samples=prefix_qpos,
                action_end_indices=prefix_ends,
                certificate=decision.certificate,
                decision=decision,
                attempts=tuple(attempts),
                total_work=_sum_work(attempts),
                total_latency_ns=time.perf_counter_ns() - started,
                reason="certified unchanged prefix",
            )

    return SafePrefixRecoveryResult(
        success=False,
        prefix_length=0,
        native_actions=None,
        qpos_samples=None,
        action_end_indices=(),
        certificate=None,
        decision=last_decision,
        attempts=tuple(attempts),
        total_work=_sum_work(attempts),
        total_latency_ns=time.perf_counter_ns() - started,
        reason=(
            "no configured unchanged prefix certified without known unwanted forecast contact"
            if veto_unwanted_contacts else "no configured unchanged prefix certified"
        ),
    )


__all__ = [
    "SafePrefixAttempt",
    "SafePrefixRecoveryResult",
    "recover_safe_prefix",
]
