"""Pipelined single-writer executor: chained one-time leases without stop-and-go.

The base ``Executor`` accepts a new lease only after the previous prefix has
been dispatched *and* the controller has drained, so every lease boundary
costs idle control cycles (40 steps take 50 ticks at K=4 and 80 ticks at K=1
with the reference controller).  Fine-grained authorization (small K, i.e.
more frequent re-checks of context, certificate and prediction) therefore
costs throughput.

``PipelinedExecutor`` lets the next lease be committed while the current one is
still executing, provided it *chains* exactly onto what is already committed:

* its plan starts at the last committed knot (bit-identical continuity);
* its context carries ``queue_rev`` = previous lease + 1 and
  ``committed_prefix_hash`` = digest of the committed virtual plan;
* every usual commit-time check still holds (MAC, single use, generation,
  deadline, prediction, fresh controller feedback inside the tracking tube).

Per-step dispatch re-checks the head entry's own lease (generation, deadline,
remaining-step budget, prediction), the environment-owned context fields, the
live queue state against the most recently committed lease, fresh feedback and
the tracking tube against the committed virtual plan.  Revocation clears every
queued entry of every chained lease.

Memory and per-commit cost stay bounded on long runs: observed virtual knots
are trimmed in batches, finished leases are forgotten, and the committed
prefix digest is a rolling hash extended in O(K) per lease.

Context semantics in pipelined mode: at commit, every field of the caller's
live context must equal the lease's context (as in the base executor), and a
chained lease must additionally carry queue_rev = previous + 1 and the digest
of the committed virtual plan.  At dispatch, robot, boot, scene_id,
controller, task_phase, scene_hash and epoch are compared with the head
entry's lease, and queue_rev / committed_prefix_hash with the most recently
committed lease (the head entry's own values have already been superseded by
the chain).
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import deque
from typing import Optional

from .contracts import Context, ErrorCode, Lease, Plan, Rejection, Snapshot
from .executor import START_TUBE, Executor, ExecutorState, _OPEN_PHASES

ENV_FIELDS = ("robot", "boot", "scene_id", "controller", "task_phase", "scene_hash")
_PREFIX_TAG = b"sentinel-committed-prefix-v2\0"
_TRIM_EVERY = 64   # drop already-observed virtual knots in batches


def _fold(state: bytes, knot) -> bytes:
    return hashlib.sha256(state + struct.pack("<3d", *knot)).digest()


def _initial_state() -> bytes:
    return hashlib.sha256(_PREFIX_TAG).digest()


def prefix_digest(knots) -> str:
    """Rolling digest of the committed virtual plan (IEEE-754 bytes of every knot).

    Extending the plan by K knots costs O(K), so long continuous runs do not
    re-hash the whole history at every chained commit.
    """
    state = _initial_state()
    for k in knots:
        state = _fold(state, k)
    return "sha256:" + state.hex()


class PipelinedExecutor(Executor):
    def __init__(self, authority, controller, events, max_ahead: int = 2):
        super().__init__(authority, controller, events)
        if isinstance(max_ahead, bool) or not isinstance(max_ahead, int) or max_ahead < 1:
            raise ValueError("max_ahead must be a positive integer")
        self._max_ahead = max_ahead
        self._queue: deque = deque()        # entries: [lease, plan, step]
        self._virtual: list = []             # committed virtual plan knots not yet trimmed
        self._trimmed = 0                    # knots dropped from the front of _virtual
        self._prefix_state: Optional[bytes] = None
        self._chain_rev: Optional[int] = None
        self._observed_base = 0
        self._lease_dt: dict = {}           # only leases that still own queued steps

    # ------------------------------------------------------------ helpers
    def _leases_ahead(self) -> int:
        return len({e[0].lease_id for e in self._queue})

    def _check_env(self, live: Context, permitted: Context) -> None:
        if live.epoch != permitted.epoch or live.epoch != self._generation:
            raise Rejection(ErrorCode.STALE_GENERATION, "执行代次变化")
        for name in ENV_FIELDS:
            if getattr(live, name) != getattr(permitted, name):
                raise Rejection(ErrorCode.CONTEXT_CHANGED, f"{name} 变化")

    def _observed_index(self) -> int:
        return max(0, self._controller.cursors()["observed"] - self._observed_base)

    def _expected_knot(self):
        local = self._observed_index() - self._trimmed
        return self._virtual[max(0, min(local, len(self._virtual) - 1))]

    def _committed_prefix(self) -> str:
        return "sha256:" + self._prefix_state.hex()

    def _extend_virtual(self, knots) -> None:
        for k in knots:
            self._virtual.append(k)
            self._prefix_state = _fold(self._prefix_state, k)

    def _trim_virtual(self) -> None:
        # keep the knot the robot is expected at and everything after it
        drop = min(self._observed_index() - self._trimmed, len(self._virtual) - 1)
        if drop >= _TRIM_EVERY:
            del self._virtual[:drop]
            self._trimmed += drop

    # ------------------------------------------------------------ commit
    def commit(self, lease: Lease, plan: Plan, snapshot: Snapshot, live_context: Context, now_ns: int) -> None:
        with self._lock:
            if self._state == ExecutorState.FAULT:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "故障状态下不接纳新许可")
            chaining = self._state == ExecutorState.RUNNING
            if chaining and self._leases_ahead() >= self._max_ahead:
                raise Rejection(ErrorCode.CONTROLLER_FULL, "已链入的许可数达到上限")
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
            self._check_env(live_context, lease.context)
            # at commit the caller's live queue state must equal what the lease was issued for
            if (live_context.queue_rev != lease.context.queue_rev
                    or live_context.committed_prefix_hash != lease.context.committed_prefix_hash):
                raise Rejection(ErrorCode.CONTEXT_CHANGED, "queue_rev / committed_prefix_hash 与许可不一致")
            self._check_snapshot_age(snapshot, now_ns)
            if not self._matches_controller_feedback(snapshot, now_ns):
                raise Rejection(ErrorCode.STATE_STALE, "快照不是控制器当前实际反馈")
            if chaining:
                if struct.pack("<3d", *plan.knots[0]) != struct.pack("<3d", *self._virtual[-1]):
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "链入计划没有从已提交的末端节点开始")
                if lease.context.queue_rev != self._chain_rev + 1:
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "链入许可的 queue_rev 不连续")
                if lease.context.committed_prefix_hash != self._committed_prefix():
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "链入许可未绑定已提交前缀")
                if math.dist(snapshot.position, self._expected_knot()) > START_TUBE:
                    raise Rejection(ErrorCode.TRACKING_TUBE, "实际反馈偏离已提交虚拟计划")
            else:
                if math.dist(snapshot.position, plan.knots[0]) > START_TUBE:
                    raise Rejection(ErrorCode.TRACKING_TUBE, "提交时起点已出管")
            self._authority.consume(lease)
            if not chaining:
                self._virtual, self._trimmed = [], 0
                self._prefix_state = _initial_state()
                self._extend_virtual(plan.knots[:1])
                self._observed_base = self._controller.cursors()["observed"]
                self._next_dispatch_ns = now_ns
            for step in range(lease.prefix_len):
                self._queue.append([lease, plan, step])
            self._extend_virtual(plan.knots[1:lease.prefix_len + 1])
            self._chain_rev = lease.context.queue_rev
            self._lease_dt[lease.lease_id] = int(plan.dt * 1e9)
            self._lease, self._plan, self._context = lease, plan, live_context
            self._last_context = live_context
            if not chaining:
                self._last_snapshot_capture = snapshot.capture_mono_ns
                self._last_obs_id = snapshot.obs_id
            self._state = ExecutorState.RUNNING
            self._pending = [0]  # base-class bookkeeping: non-empty while entries are queued
            self._events.append("COMMIT", plan_hash=plan.hash, plan_exact=plan.exact_hash,
                                permit_payload=lease.payload(), lease_id=lease.lease_id, cert_id=lease.cert_id,
                                prediction_hash=lease.prediction_hash, prefix_len=lease.prefix_len,
                                obs_id=snapshot.obs_id, generation=self._generation, chained=chaining,
                                queue_rev=lease.context.queue_rev)
            if self._events.has_gap:
                self._clear_chain()
                self._pending = []
                self._state = ExecutorState.FAULT
                self._generation = self._authority.advance_generation()
                self._controller.cancel()
                raise Rejection(ErrorCode.EVIDENCE_GAP, "COMMIT 触发 LOG_GAP")

    # ------------------------------------------------------------ dispatch
    def tick(self, now_ns: int, snapshot: Optional[Snapshot] = None, live_context: Optional[Context] = None) -> bool:
        with self._lock:
            if self._state != ExecutorState.RUNNING:
                self._controller.tick()
                return False
            if not self._queue:
                self._controller.tick()
                if self._controller.is_drained:
                    self._state = ExecutorState.IDLE
                    self._pending = []
                    self._lease = self._plan = self._context = None
                    self._clear_chain()
                return False
            if self._controller.free_slots() <= 0 or now_ns < self._next_dispatch_ns:
                self._controller.tick()
                return False
            try:
                if snapshot is None or live_context is None:
                    raise Rejection(ErrorCode.STATE_STALE, "新派发缺少 Snapshot 或 Context")
                lease, plan, step = self._queue[0]
                self._authority.validate_runtime(lease, plan.hash, now_ns)
                self._check_env(live_context, lease.context)
                # nobody else may have changed the queue: the live queue state must be the one
                # of the most recently committed (chained) lease
                latest = self._lease.context
                if (live_context.queue_rev != latest.queue_rev
                        or live_context.committed_prefix_hash != latest.committed_prefix_hash):
                    raise Rejection(ErrorCode.CONTEXT_CHANGED, "队列状态与最近提交的许可不一致")
                self._check_snapshot_age(snapshot, now_ns)
                if snapshot.capture_mono_ns <= self._last_snapshot_capture or snapshot.obs_id == self._last_obs_id:
                    raise Rejection(ErrorCode.STATE_STALE, "观测未更新")
                if not self._matches_controller_feedback(snapshot, now_ns):
                    raise Rejection(ErrorCode.STATE_STALE, "快照不是控制器当前实际反馈")
                if math.dist(snapshot.position, self._expected_knot()) > START_TUBE:
                    raise Rejection(ErrorCode.TRACKING_TUBE, "实际反馈偏离已提交虚拟计划")
                remaining = sum(1 for e in self._queue if e[0].lease_id == lease.lease_id)
                if now_ns + remaining * self._lease_dt[lease.lease_id] > lease.deadline_mono_ns:
                    raise Rejection(ErrorCode.LEASE_EXPIRED, "剩余已批准步骤无法在许可期限内完成")
                grip = dict(plan.gripper_events).get(step)
                if grip == "open" and (snapshot.supported is not True or live_context.task_phase not in _OPEN_PHASES):
                    raise Rejection(ErrorCode.TRACKING_TUBE, "open 要求 supported=True 且处于允许释放阶段")
                if not self._events.can_append_without_gap():
                    self._events.note_gap("DISPATCH event would overflow bounded buffer")
                    raise Rejection(ErrorCode.EVIDENCE_GAP, "派发事件无法完整记录")
                ok = self._authority.admit_submission(
                    lease, plan.hash, now_ns,
                    lambda: self._controller.submit(plan.knots[step + 1], self._generation, grip),
                )
                if ok:
                    self._queue.popleft()
                    self._last_snapshot_capture = snapshot.capture_mono_ns
                    self._last_obs_id = snapshot.obs_id
                    self._next_dispatch_ns = now_ns + self._lease_dt[lease.lease_id]
                    if not self._queue or self._queue[0][0].lease_id != lease.lease_id:
                        del self._lease_dt[lease.lease_id]       # the lease owns no more queued steps
                    self._trim_virtual()
                    action_bytes = struct.pack("<3d", *plan.knots[step + 1])
                    self._events.append("DISPATCH", plan_hash=plan.hash, plan_exact=plan.exact_hash,
                                        action_float64_le=action_bytes.hex(),
                                        action_bytes_hash="sha256:" + hashlib.sha256(action_bytes).hexdigest(),
                                        lease_id=lease.lease_id, step=step,
                                        gripper_event=grip, obs_id=snapshot.obs_id, generation=self._generation)
                    if self._events.has_gap:
                        raise Rejection(ErrorCode.EVIDENCE_GAP, "DISPATCH 触发 LOG_GAP")
                self._controller.tick()
                return ok
            except Rejection as exc:
                self._enter_runtime_fault(exc)
                raise

    # ------------------------------------------------------------ revoke
    def _clear_chain(self) -> None:
        self._queue.clear()
        self._virtual, self._trimmed = [], 0
        self._prefix_state = None
        self._chain_rev = None
        self._lease_dt.clear()

    def _enter_runtime_fault(self, rejection: Rejection) -> None:
        self._clear_chain()
        super()._enter_runtime_fault(rejection)

    def revoke(self, reason: str, now_ns: Optional[int] = None) -> None:
        """Same linearization point as the base executor; also drops every chained lease."""
        with self._lock:
            self._generation = self._authority.advance_generation()
            self._revoked_after_capture = self._last_snapshot_capture
            if now_ns is not None:
                self._revoked_at_ns = now_ns
            self._pre_revoke_context = self._context or self._last_context
            dropped = len(self._queue)
            self._clear_chain()
            self._pending = []
            self._state = ExecutorState.FAULT
            lease_id = self._lease.lease_id if self._lease else None
            self._lease = None
            self._plan = None
            self._events.append("REVOKE", reason=reason, lease_id=lease_id, new_generation=self._generation,
                                dropped_queued_steps=dropped)
            if self._events.supports("CANCEL_REQUEST"):
                self._events.append("CANCEL_REQUEST", reason=reason, generation=self._generation)
        self._controller.cancel()
