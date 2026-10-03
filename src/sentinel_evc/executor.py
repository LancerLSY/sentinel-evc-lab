"""Single-writer executor with observation-bound, revocable dispatch."""

from __future__ import annotations

import math
import hashlib
import struct
import threading
from typing import Optional

from .authority import MAX_OBS_AGE_NS, START_TUBE, Authority
from .contracts import Context, ErrorCode, Lease, Plan, Rejection, Snapshot
from .events import EventLog


class ExecutorState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FAULT = "FAULT"


_OPEN_PHASES = frozenset({"place", "release", "handoff", "supported_release"})
_MAX_MONOTONIC_NS = (1 << 63) - 1


class Executor:
    """The only object allowed to call the controller write interface."""

    def __init__(self, authority: Authority, controller, events: EventLog):
        self._authority = authority
        self._controller = controller
        self._events = events
        self._lock = threading.RLock()
        self._state = ExecutorState.IDLE
        self._generation = authority.generation
        self._plan: Optional[Plan] = None
        self._lease: Optional[Lease] = None
        self._context: Optional[Context] = None
        self._pending: list[int] = []
        self._dispatched = 0
        self._observed_at_commit = 0
        self._last_snapshot_capture = -1
        self._last_obs_id: Optional[str] = None
        self._next_dispatch_ns = 0
        self._revoked_after_capture = -1
        self._revoked_at_ns = -1
        self._pre_revoke_context: Optional[Context] = None
        self._last_context: Optional[Context] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @staticmethod
    def _check_context(live: Context, permitted: Context, generation: int) -> None:
        if live.epoch != permitted.epoch or live.epoch != generation:
            raise Rejection(ErrorCode.STALE_GENERATION, "执行代次变化")
        for field_name in (
            "robot",
            "boot",
            "scene_id",
            "controller",
            "task_phase",
            "queue_rev",
            "committed_prefix_hash",
            "scene_hash",
        ):
            if getattr(live, field_name) != getattr(permitted, field_name):
                raise Rejection(ErrorCode.CONTEXT_CHANGED, f"{field_name} 变化")

    @staticmethod
    def _check_snapshot_age(snapshot: Snapshot, now_ns: int) -> None:
        if not snapshot.valid:
            raise Rejection(ErrorCode.STATE_STALE, "快照无效")
        age = now_ns - snapshot.capture_mono_ns
        if age < 0 or age > MAX_OBS_AGE_NS:
            raise Rejection(ErrorCode.STATE_STALE, f"观测年龄 {age}ns")

    def _matches_controller_feedback(self, snapshot: Snapshot, now_ns: int) -> bool:
        """Bind the supplied snapshot to one fresh controller feedback sample.

        ``now_ns`` is only the comparison clock.  It cannot make an old controller
        sample fresh, so the controller-provided capture time must match the
        snapshot and independently satisfy the observation-age bound.
        """
        try:
            feedback = self._controller.read_feedback(now_ns)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not isinstance(feedback, dict):
            return False

        captured = feedback.get("capture_mono_ns")
        if (
            isinstance(captured, bool)
            or not isinstance(captured, int)
            or captured < 0
            or captured > _MAX_MONOTONIC_NS
            or captured != snapshot.capture_mono_ns
        ):
            return False
        age = now_ns - captured
        if age < 0 or age > MAX_OBS_AGE_NS:
            return False

        if "valid" in feedback and feedback["valid"] is not True:
            return False
        position = feedback.get("position")
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in position
        ):
            return False
        return math.dist(snapshot.position, position) <= 1e-12

    def commit(
        self,
        lease: Lease,
        plan: Plan,
        snapshot: Snapshot,
        live_context: Context,
        now_ns: int,
    ) -> None:
        """Consume a permit only after a complete, current commit-time recheck."""
        with self._lock:
            if self._state == ExecutorState.FAULT:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "故障状态下不接纳新许可")
            if self._state == ExecutorState.RUNNING:
                raise Rejection(ErrorCode.CONTROLLER_FULL, "已有前缀正在执行")
            if self._events.has_gap:
                raise Rejection(ErrorCode.EVIDENCE_GAP, "事件链已有 LOG_GAP")
            if not self._events.can_append_without_gap():
                self._events.note_gap("COMMIT event would overflow bounded buffer")
                raise Rejection(ErrorCode.EVIDENCE_GAP, "提交事件无法完整记录")
            if not self._authority.verify_mac(lease):
                raise Rejection(ErrorCode.LEASE_UNKNOWN, "许可签名不匹配")
            if lease.plan_hash != plan.hash:
                raise Rejection(ErrorCode.CERTIFICATE_MISS, "许可未绑定这个最终动作")
            exact = self._authority.exact_binding(lease.lease_id)
            if exact is not None and exact != plan.exact_hash:
                raise Rejection(ErrorCode.CERTIFICATE_MISS, "最终动作与许可绑定的字节不一致")
            self._authority.validate_available(lease)
            self._authority.validate_runtime(lease, plan.hash, now_ns)
            self._check_context(live_context, lease.context, self._generation)
            self._check_snapshot_age(snapshot, now_ns)
            if math.dist(snapshot.position, plan.knots[0]) > START_TUBE:
                raise Rejection(ErrorCode.TRACKING_TUBE, "提交时起点已出管")
            if not self._matches_controller_feedback(snapshot, now_ns):
                raise Rejection(ErrorCode.STATE_STALE, "快照不是控制器当前实际反馈")

            self._authority.consume(lease)
            self._plan = plan
            self._lease = lease
            self._context = live_context
            self._last_context = live_context
            self._pending = list(range(lease.prefix_len))
            self._dispatched = 0
            self._observed_at_commit = self._controller.cursors()["observed"]
            self._last_snapshot_capture = snapshot.capture_mono_ns
            self._last_obs_id = snapshot.obs_id
            self._next_dispatch_ns = now_ns
            self._state = ExecutorState.RUNNING
            self._events.append(
                "COMMIT",
                plan_hash=plan.hash,
                plan_exact=plan.exact_hash,
                permit_payload=lease.payload(),
                lease_id=lease.lease_id,
                cert_id=lease.cert_id,
                prediction_hash=lease.prediction_hash,
                prefix_len=lease.prefix_len,
                obs_id=snapshot.obs_id,
                generation=self._generation,
            )
            if self._events.has_gap:
                self._pending.clear()
                self._state = ExecutorState.FAULT
                self._generation = self._authority.advance_generation()
                self._lease = None
                self._plan = None
                self._controller.cancel()
                raise Rejection(ErrorCode.EVIDENCE_GAP, "COMMIT 触发 LOG_GAP")

    def _enter_runtime_fault(self, rejection: Rejection) -> None:
        self._generation = self._authority.advance_generation()
        self._revoked_after_capture = self._last_snapshot_capture
        self._pre_revoke_context = self._context or self._last_context
        self._pending.clear()
        lease_id = self._lease.lease_id if self._lease else None
        self._lease = None
        self._plan = None
        self._state = ExecutorState.FAULT
        self._events.append(
            "BACKUP",
            reason=rejection.code,
            detail=rejection.detail,
            lease_id=lease_id,
            new_generation=self._generation,
        )
        self._controller.cancel()

    def tick(
        self,
        now_ns: int,
        snapshot: Optional[Snapshot] = None,
        live_context: Optional[Context] = None,
    ) -> bool:
        """Advance one controller cycle and dispatch at most one new action."""
        with self._lock:
            if self._state != ExecutorState.RUNNING:
                self._controller.tick()
                return False

            if not self._pending:
                self._controller.tick()
                if self._controller.is_drained:
                    self._state = ExecutorState.IDLE
                    self._lease = None
                    self._plan = None
                    self._context = None
                return False

            if self._controller.free_slots() <= 0 or now_ns < self._next_dispatch_ns:
                self._controller.tick()
                return False

            try:
                if snapshot is None or live_context is None:
                    raise Rejection(ErrorCode.STATE_STALE, "新派发缺少 Snapshot 或 Context")
                assert self._lease is not None and self._plan is not None and self._context is not None
                self._authority.validate_runtime(self._lease, self._plan.hash, now_ns)
                self._check_context(live_context, self._lease.context, self._generation)
                self._check_snapshot_age(snapshot, now_ns)
                if snapshot.capture_mono_ns <= self._last_snapshot_capture or snapshot.obs_id == self._last_obs_id:
                    raise Rejection(ErrorCode.STATE_STALE, "观测未更新")
                if not self._matches_controller_feedback(snapshot, now_ns):
                    raise Rejection(ErrorCode.STATE_STALE, "快照不是控制器当前实际反馈")
                observed_since_commit = max(
                    0, self._controller.cursors()["observed"] - self._observed_at_commit
                )
                expected_index = min(observed_since_commit, self._lease.prefix_len)
                if math.dist(snapshot.position, self._plan.knots[expected_index]) > START_TUBE:
                    raise Rejection(ErrorCode.TRACKING_TUBE, "实际反馈偏离已观测计划节点")

                # v2 completion budget: this step and every remaining approved step must be able to
                # finish (one dt each, the controller's nominal step) before the lease deadline
                if now_ns + len(self._pending) * int(self._plan.dt * 1e9) > self._lease.deadline_mono_ns:
                    raise Rejection(ErrorCode.LEASE_EXPIRED, "剩余已批准步骤无法在许可期限内完成")
                step = self._pending[0]
                grip = dict(self._plan.gripper_events).get(step)
                if grip == "open" and (
                    snapshot.supported is not True or live_context.task_phase not in _OPEN_PHASES
                ):
                    raise Rejection(
                        ErrorCode.TRACKING_TUBE,
                        "open 要求 supported=True 且处于允许释放阶段",
                    )
                if not self._events.can_append_without_gap():
                    self._events.note_gap("DISPATCH event would overflow bounded buffer")
                    raise Rejection(ErrorCode.EVIDENCE_GAP, "派发事件无法完整记录")
                action = self._plan.knots[step + 1]
                ok = self._authority.admit_submission(
                    self._lease, self._plan.hash, now_ns,
                    lambda: self._controller.submit(action, self._generation, grip),
                )
                if ok:
                    self._pending.pop(0)
                    self._dispatched += 1
                    self._last_snapshot_capture = snapshot.capture_mono_ns
                    self._last_obs_id = snapshot.obs_id
                    self._next_dispatch_ns = now_ns + int(self._plan.dt * 1e9)
                    self._events.append(
                        "DISPATCH",
                        plan_hash=self._plan.hash,
                        plan_exact=self._plan.exact_hash,
                        action_float64_le=struct.pack("<3d", *action).hex(),
                        action_bytes_hash="sha256:" + hashlib.sha256(struct.pack("<3d", *action)).hexdigest(),
                        lease_id=self._lease.lease_id,
                        step=step,
                        gripper_event=grip,
                        obs_id=snapshot.obs_id,
                        generation=self._generation,
                    )
                    if self._events.has_gap:
                        raise Rejection(ErrorCode.EVIDENCE_GAP, "DISPATCH 触发 LOG_GAP")
                self._controller.tick()
                return ok
            except Rejection as exc:
                self._enter_runtime_fault(exc)
                raise

    def revoke(self, reason: str, now_ns: Optional[int] = None) -> None:
        """Invalidate permits and request asynchronous controller cancellation.

        ``now_ns`` (v2) records the revocation instant on the executor's monotonic
        clock; recovery then requires feedback captured strictly after it.
        """
        with self._lock:
            self._generation = self._authority.advance_generation()
            self._revoked_after_capture = self._last_snapshot_capture
            if now_ns is not None:
                self._revoked_at_ns = now_ns
            self._pre_revoke_context = self._context or self._last_context
            self._pending.clear()
            self._state = ExecutorState.FAULT
            lease_id = self._lease.lease_id if self._lease else None
            self._lease = None
            self._plan = None
            self._events.append(
                "REVOKE",
                reason=reason,
                lease_id=lease_id,
                new_generation=self._generation,
            )
            if self._events.supports("CANCEL_REQUEST"):
                self._events.append(
                    "CANCEL_REQUEST", reason=reason, generation=self._generation
                )
        self._controller.cancel()

    def poll_cancel(self) -> Optional[bool]:
        acked = self._controller.cancel_acked
        if acked is not None:
            if acked and self._events.supports("CANCEL_ACCEPTED"):
                self._events.append("CANCEL_ACCEPTED", generation=self._generation)
            self._events.append("CANCEL_ACK", acked=bool(acked), confirmed=bool(acked))
        return acked

    def try_recover(
        self,
        operator_approved: bool,
        snapshot: Optional[Snapshot] = None,
        live_context: Optional[Context] = None,
        now_ns: Optional[int] = None,
    ) -> None:
        """Recover to IDLE only after confirmed cancel, drain and fresh feedback."""
        with self._lock:
            if self._state != ExecutorState.FAULT:
                return
            if self._controller.cancel_acked is not True:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "取消未确认，不恢复")
            if not self._controller.is_drained:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "控制器前缀尚未排空")
            if not operator_approved:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "缺少人工批准")
            if snapshot is None or live_context is None or now_ns is None:
                raise Rejection(ErrorCode.STATE_STALE, "恢复缺少撤销后的新观测或上下文")
            self._check_snapshot_age(snapshot, now_ns)
            if snapshot.capture_mono_ns <= max(self._revoked_after_capture, self._revoked_at_ns):
                raise Rejection(ErrorCode.STATE_STALE, "恢复观测不是撤销后新采样")
            if not self._matches_controller_feedback(snapshot, now_ns):
                raise Rejection(ErrorCode.STATE_STALE, "恢复快照不是控制器当前实际反馈")
            if live_context.epoch != self._generation:
                raise Rejection(ErrorCode.STALE_GENERATION, "恢复上下文 epoch 不正确")
            if self._pre_revoke_context is not None:
                previous = self._pre_revoke_context
                for field_name in (
                    "robot",
                    "boot",
                    "scene_id",
                    "controller",
                    "task_phase",
                    "scene_hash",
                ):
                    if getattr(live_context, field_name) != getattr(previous, field_name):
                        raise Rejection(
                            ErrorCode.CONTEXT_CHANGED,
                            f"恢复时 {field_name} 变化",
                        )
                if live_context.queue_rev < previous.queue_rev:
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "恢复时 queue_rev 回退")
            if self._events.has_gap:
                raise Rejection(ErrorCode.EVIDENCE_GAP, "事件链已有 LOG_GAP")
            self._state = ExecutorState.IDLE
            self._context = None
            self._last_snapshot_capture = snapshot.capture_mono_ns
            self._last_obs_id = snapshot.obs_id
            self._events.append(
                "OUTCOME", outcome="recovered", generation=self._generation, obs_id=snapshot.obs_id
            )
