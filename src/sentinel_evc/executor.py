"""执行器：唯一的命令写入者。

五项发布不变量，每条对应 tests/test_executor.py 里的一个测试：
  1. 未授权通道不能写驱动（只有 Executor 调 controller.submit）
  2. 同一许可不能重复消费
  3. 已提交的不可撤销前缀不能被改写
  4. 撤销后不新增旧代次本地提交
  5. 取消未确认不得恢复旧计划
"""

from __future__ import annotations

import threading
from typing import Optional

from .authority import Authority
from .contracts import ErrorCode, Lease, LeaseContext, Plan, Rejection, Snapshot
from .events import EventLog


class ExecutorState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FAULT = "FAULT"


class Executor:
    """本地唯一命令写入者。"""

    def __init__(self, authority: Authority, controller, events: EventLog):
        self._authority = authority
        self._controller = controller
        self._events = events

        self._lock = threading.RLock()
        self._state = ExecutorState.IDLE
        self._generation = 0

        self._plan: Optional[Plan] = None
        self._lease: Optional[Lease] = None
        self._pending: list = []  # 本地待发前缀
        self._dispatched = 0

    # ------------------------------------------------------------ 属性

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    # ------------------------------------------------------------ Commit

    def commit(
        self,
        lease: Lease,
        plan: Plan,
        snapshot: Snapshot,
        live_context: LeaseContext,
        now_ns: int,
    ) -> None:
        """提交阶段：复核最新状态，消费一次性许可，才开放待发前缀。

        准备阶段的依据和提交时的新观测都要记录 —— 两者可以不同，
        但新观测必须仍在允许管内。
        """
        with self._lock:
            if self._state == ExecutorState.FAULT:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "故障状态下不接纳新许可")

            if not self._authority.verify_mac(lease):
                raise Rejection(ErrorCode.AUTH_FAILED, "许可签名不匹配")

            if lease.final_hash != plan.hash:
                raise Rejection(ErrorCode.AUTH_FAILED, "许可未绑定这个最终动作")

            if now_ns > lease.deadline_mono:
                raise Rejection(ErrorCode.LEASE_EXPIRED, "许可已过期")

            if live_context != lease.context:
                raise Rejection(ErrorCode.CONTEXT_CHANGED, "执行上下文变化")

            # 新快照可以替换旧快照，但必须仍在允许管内。
            # 注意：obs_id 相同但状态已出管，一样要拒绝。
            import math

            from .authority import START_TUBE

            if math.dist(snapshot.position, plan.points[0]) > START_TUBE:
                raise Rejection(ErrorCode.TRACKING_TUBE, "提交时起点已出管")

            # 一次性消费。放在所有检查之后 —— 检查失败不应烧掉许可。
            self._authority.consume(lease)

            self._plan = plan
            self._lease = lease
            self._pending = list(range(lease.prefix_len))
            self._dispatched = 0
            self._state = ExecutorState.RUNNING

            self._events.append(
                "COMMIT",
                plan_hash=plan.hash,
                lease_id=lease.lease_id,
                cert_id=lease.cert_id,
                accepted=True,
                reason_code=None,
            )

    # ------------------------------------------------------------ Dispatch

    def tick(self, now_ns: int) -> bool:
        """推进一步。返回是否实际发出了命令。

        每一步都要重新检查许可状态、代次、期限 —— 不是 commit 时查一次就完。
        错过时间槽之后不能瞬间补发多条旧动作。
        """
        with self._lock:
            if self._state != ExecutorState.RUNNING or not self._pending:
                self._controller.tick()
                return False

            lease = self._lease
            assert lease is not None and self._plan is not None

            if now_ns > lease.deadline_mono:
                self._pending.clear()
                self._state = ExecutorState.IDLE
                self._controller.tick()
                return False

            if self._controller.free_slots() <= 0:
                # 正常背压，不是错误
                self._controller.tick()
                return False

            step = self._pending.pop(0)
            action = self._plan.points[step + 1]
            ok = self._controller.submit(action, self._generation)
            if ok:
                self._dispatched += 1
                self._events.append(
                    "DISPATCH",
                    plan_hash=self._plan.hash,
                    lease_id=lease.lease_id,
                    cert_id=lease.cert_id,
                    step_index=step,
                    submitted=True,
                )
            else:
                self._pending.insert(0, step)

            self._controller.tick()
            return ok

    # ------------------------------------------------------------ Revoke

    def revoke(self, reason: str) -> None:
        """撤销屏障。

        锁内只做三件事然后立刻放开 —— 网络等待绝不能占着撤销锁。
        cancel 发在锁外。
        """
        with self._lock:
            old_epoch = self._generation
            self._generation += 1  # 旧代次立刻失效
            self._pending.clear()  # 清本地待发
            self._state = ExecutorState.FAULT
            lease_id = self._lease.lease_id if self._lease else None
            self._lease = None
            self._events.append(
                "REVOKE",
                lease_id=lease_id,
                reason=reason,
                old_epoch=old_epoch,
                new_epoch=self._generation,
            )

        self._controller.cancel()  # 锁外发，等 ACK 不占锁

    def poll_cancel(self) -> Optional[bool]:
        """查询取消是否已确认。"""
        acked = self._controller.cancel_acked
        if acked is not None:
            self._events.append("CANCEL_ACK", confirmed=bool(acked))
        return acked

    def try_recover(self, operator_approved: bool) -> None:
        """恢复必须同时满足三件事：取消确认、新快照、人工批准。少一样都不恢复。"""
        with self._lock:
            if self._state != ExecutorState.FAULT:
                return
            if self._controller.cancel_acked is not True:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "取消未确认，不恢复")
            if not operator_approved:
                raise Rejection(ErrorCode.CANCEL_UNCONFIRMED, "缺少人工批准")
            self._state = ExecutorState.IDLE
