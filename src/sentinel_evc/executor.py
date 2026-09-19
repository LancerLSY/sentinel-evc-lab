"""最终提交门禁、逐步发送和撤销屏障；唯一调用 controller.submit 的模块。"""

from __future__ import annotations

import math
import threading
from time import monotonic_ns as system_monotonic_ns

from .authority import Authority, MAX_OBS_AGE_NS, START_TUBE
from .contracts import ErrorCode, Lease, LeaseContext, Plan, Rejection, Snapshot
from .events import EventLog


class ExecutorState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FAULT = "FAULT"


class Executor:
    def __init__(self, authority: Authority, controller, events: EventLog,
                 *, monotonic_ns=system_monotonic_ns):
        self._authority = authority
        self._controller = controller
        self._events = events
        self._monotonic_ns = monotonic_ns
        self._lock = threading.RLock()
        # 只保护取消状态；控制器调用和 ACK 等待不占用任何锁。
        self._cancel_lock = threading.Lock()
        self._cancel_calls = 0
        self._state = ExecutorState.IDLE
        self._generation = 0
        self._plan = None
        self._lease = None
        self._pending = []
        self._next_dispatch_ns = 0
        self._accepted_start = 0
        self._observed_start = 0
        self._cancel_reported = False
        self._revoked_mono = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    def _verify_lease(self, lease, plan, context, now_ns):
        if not self._authority.verify_mac(lease) or lease.final_hash != plan.hash:
            raise Rejection(ErrorCode.AUTH_FAILED, "许可未认证或未绑定最终动作")
        if context != lease.context or context.epoch != self._generation:
            raise Rejection(ErrorCode.CONTEXT_CHANGED, "执行上下文变化")
        if now_ns >= lease.deadline_mono:
            raise Rejection(ErrorCode.LEASE_EXPIRED, "许可已过期")

    def _verify_snapshot(self, snapshot, now_ns):
        if not 0 <= now_ns - snapshot.observed_mono <= MAX_OBS_AGE_NS:
            raise Rejection(ErrorCode.STATE_STALE, "观测年龄超出允许范围")

    def commit(self, lease: Lease, plan: Plan, snapshot: Snapshot,
               live_context: LeaseContext, now_ns: int) -> None:
        """所有复核与消费共用临界区；失败不会烧掉尚未消费的许可。"""
        with self._lock:
            if self._state == ExecutorState.FAULT:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "故障状态下不接纳新许可")
            self._verify_lease(lease, plan, live_context, now_ns)
            if snapshot.context != live_context:
                raise Rejection(ErrorCode.CONTEXT_CHANGED, "最新观测上下文不匹配")
            if lease.deadline_mono - now_ns < lease.prefix_len * plan.dt * 1e9:
                raise Rejection(ErrorCode.LEASE_EXPIRED, "剩余期限不足以执行前缀")
            self._verify_snapshot(snapshot, now_ns)
            if math.dist(snapshot.position, plan.points[0]) > START_TUBE:
                raise Rejection(ErrorCode.TRACKING_TUBE, "提交时起点已出管")
            if self._authority.is_consumed(lease.lease_id):
                raise Rejection(ErrorCode.LEASE_REPLAY, lease.lease_id)
            if self._pending or self._controller.free_slots() < self._controller.capacity:
                raise Rejection(ErrorCode.QUEUE_FULL, "已有前缀尚未完成，不能覆盖")
            self._authority.consume(lease)
            self._plan = plan
            self._lease = lease
            self._pending = list(range(lease.prefix_len))
            self._next_dispatch_ns = now_ns
            self._accepted_start = len(self._controller.accepted)
            self._observed_start = len(self._controller.observed)
            self._state = ExecutorState.RUNNING
            self._event("COMMIT", accepted=True, reason_code=None)

    def _event(self, kind, **payload):
        lease = self._lease
        self._events.append(kind, plan_hash=lease.final_hash if lease else None,
                            lease_id=lease.lease_id if lease else None,
                            cert_id=lease.cert_id if lease else None, **payload)

    def _advance(self):
        accepted = len(self._controller.accepted)
        observed = len(self._controller.observed)
        self._controller.advance()
        for index in range(accepted, len(self._controller.accepted)):
            self._event("CONTROLLER_ACK", step_index=index - self._accepted_start,
                        accepted=True)
        for index in range(observed, len(self._controller.observed)):
            self._event("OBSERVED", step_index=index - self._observed_start,
                        position=self._controller.observed[index]["action"], observed=True)
        if (self._state == ExecutorState.RUNNING and not self._pending
                and self._controller.free_slots() == self._controller.capacity):
            self._state = ExecutorState.IDLE

    def tick(self, now_ns: int, live_context: LeaseContext) -> bool:
        """每步核对当前上下文与期限；错过时间槽不在同一时刻补发多步。"""
        with self._lock:
            if self._state != ExecutorState.RUNNING or not self._pending:
                self._advance()
                return False
            try:
                self._verify_lease(self._lease, self._plan, live_context, now_ns)
            except Rejection:
                self._pending.clear()
                self._state = ExecutorState.IDLE
                raise
            if now_ns < self._next_dispatch_ns or self._controller.free_slots() <= 0:
                self._advance()
                return False
            step = self._pending[0]
            submitted = self._controller.submit(self._plan.points[step + 1], self._generation)
            self._event("DISPATCH", step_index=step, submitted=submitted)
            if submitted:
                self._pending.pop(0)
                self._next_dispatch_ns = now_ns + self._plan.dt * 1e9
            self._advance()
            return submitted

    def revoke(self, reason: str) -> None:
        with self._cancel_lock:
            self._cancel_calls += 1
            old_epoch = self._generation
            with self._lock:
                self._generation += 1
                self._pending.clear()
                self._state = ExecutorState.FAULT
            self._revoked_mono = self._monotonic_ns()
            self._event("REVOKE", reason=reason, old_epoch=old_epoch,
                        new_epoch=self._generation)
            self._cancel_reported = False
        self._controller.cancel()
        with self._cancel_lock:
            self._cancel_calls -= 1

    def poll_cancel(self):
        with self._cancel_lock:
            if self._cancel_calls:
                return None
            acked = self._controller.cancel_acked
            if acked is not None and not self._cancel_reported:
                self._event("CANCEL_ACK", confirmed=bool(acked))
                self._cancel_reported = True
            return acked

    def recover(self, snapshot: Snapshot, human_approved: bool) -> None:
        """取消确认、新代次有效观测与人工批准三者缺一不可。"""
        with self._cancel_lock, self._lock:
            if self._state != ExecutorState.FAULT:
                return
            if (self._cancel_calls or self._controller.cancel_acked is not True
                    or human_approved is not True):
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "取消未确认或缺少人工批准")
            self._verify_snapshot(snapshot, self._monotonic_ns())
            if snapshot.observed_mono < self._revoked_mono:
                raise Rejection(ErrorCode.STATE_STALE, "恢复需要撤销后的新观测")
            if snapshot.epoch != self._generation:
                raise Rejection(ErrorCode.CONTEXT_CHANGED, "恢复观测不是当前代次")
            if self._lease is not None:
                previous = self._lease.context
                if snapshot.robot != previous.robot or snapshot.boot != previous.boot:
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "恢复观测不是本机本次启动")
            self._state = ExecutorState.IDLE
