"""模拟控制器。

这是一个**数值模拟**，不是任何真实控制器的模型。它存在的唯一目的是
让三个游标（submitted / accepted / observed）分离，从而能测试这句话：

    撤销之后不再新增旧代次提交，但撤销之前已经提交的命令仍可能被执行。

清空 Python 队列不等于电机停止。真机演示必须按具体控制器实测，
不能把清队列画成瞬间制动。
"""

from __future__ import annotations

from collections import deque
import math
from typing import Optional


class SimController:
    """容量有限的命令缓冲 + 延迟确认。

    capacity        控制器内部能缓存几步。默认 2，小于典型 4 步前缀 ——
                    这正是为了暴露「前缀要分批提交」这件事。
    ack_delay_ticks 提交后几个 tick 才确认
    exec_delay_ticks 确认后几个 tick 才被观测到实际执行
    """

    def __init__(
        self,
        capacity: int = 2,
        ack_delay_ticks: int = 1,
        exec_delay_ticks: int = 1,
        drop_cancel_ack: bool = False,
        initial_position=(0.0, 0.0, 0.0),
        initial_gripper: str = "closed",
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity 必须是正整数")
        if isinstance(ack_delay_ticks, bool) or not isinstance(ack_delay_ticks, int) or ack_delay_ticks < 1:
            raise ValueError("ack_delay_ticks 必须是正整数")
        if isinstance(exec_delay_ticks, bool) or not isinstance(exec_delay_ticks, int) or exec_delay_ticks < 1:
            raise ValueError("exec_delay_ticks 必须是正整数")
        self.capacity = capacity
        self.ack_delay_ticks = ack_delay_ticks
        self.exec_delay_ticks = exec_delay_ticks
        self.drop_cancel_ack = drop_cancel_ack

        self._inflight = deque()  # [{action, gen, age, state}]
        self.submitted = []  # 已发出
        self.accepted = []  # 控制器已确认
        self.observed = []  # 已实际执行（模拟）
        self.gripper_submitted = []
        self.gripper_accepted = []
        self.gripper_observed = []

        self._position = self._validated_position(initial_position)
        if initial_gripper not in ("open", "closed"):
            raise ValueError("initial_gripper 必须是 open/closed")
        self.gripper_state = initial_gripper

        self._cancel_pending = False
        self._cancel_age = 0
        self.cancel_acked: Optional[bool] = None

    # ------------------------------------------------------------ 提交

    def free_slots(self) -> int:
        return self.capacity - len(self._inflight)

    @staticmethod
    def _validated_position(position) -> tuple:
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            raise ValueError("position 必须是三维向量")
        result = []
        for value in position:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("position 必须是有限三维向量")
            result.append(float(value))
        return tuple(result)

    def set_initial_position(self, position) -> None:
        if self._inflight or self.submitted or self.observed:
            raise RuntimeError("控制器已有命令，不能重设初始位置")
        self._position = self._validated_position(position)

    def submit(self, action, generation: int, gripper_event=None) -> bool:
        """提交一步。队列满时返回 False —— 这不是错误，是正常背压。"""
        if len(self._inflight) >= self.capacity:
            return False
        action = self._validated_position(action)
        if gripper_event is not None and gripper_event not in ("open", "close"):
            raise ValueError("gripper_event 必须是 open/close/None")
        item = {
            "action": action,
            "gen": generation,
            "gripper_event": gripper_event,
            "age": 0,
            "state": "submitted",
        }
        self._inflight.append(item)
        self.submitted.append({"action": action, "gen": generation})
        if gripper_event is not None:
            self.gripper_submitted.append({"event": gripper_event, "gen": generation})
        return True

    # ------------------------------------------------------------ 取消

    def cancel(self) -> None:
        """请求取消。ACK 是异步的，而且可能丢失。"""
        self._cancel_pending = True
        self._cancel_age = 0
        self.cancel_acked = None

    # ------------------------------------------------------------ 推进

    def tick(self) -> None:
        """推进一个控制周期。"""
        # 取消确认：丢弃尚未被确认的命令，但**已经确认的会继续执行完**
        if self._cancel_pending:
            self._cancel_age += 1
            if self._cancel_age >= self.ack_delay_ticks:
                if self.drop_cancel_ack:
                    # 故障注入：ACK 永远不来，状态停在未确认
                    pass
                else:
                    self._inflight = deque(
                        it for it in self._inflight if it["state"] != "submitted"
                    )
                    self._cancel_pending = False
                    self.cancel_acked = True

        done = []
        for item in self._inflight:
            item["age"] += 1
            if item["state"] == "submitted" and item["age"] >= self.ack_delay_ticks:
                item["state"] = "accepted"
                item["age"] = 0
                self.accepted.append({"action": item["action"], "gen": item["gen"]})
                if item["gripper_event"] is not None:
                    self.gripper_accepted.append(
                        {"event": item["gripper_event"], "gen": item["gen"]}
                    )
            elif item["state"] == "accepted" and item["age"] >= self.exec_delay_ticks:
                item["state"] = "observed"
                self.observed.append({"action": item["action"], "gen": item["gen"]})
                self._position = item["action"]
                if item["gripper_event"] is not None:
                    self.gripper_state = "open" if item["gripper_event"] == "open" else "closed"
                    self.gripper_observed.append(
                        {"event": item["gripper_event"], "gen": item["gen"]}
                    )
                done.append(item)

        for item in done:
            self._inflight.remove(item)

    # ------------------------------------------------------------ 查询

    def cursors(self) -> dict:
        return {
            "submitted": len(self.submitted),
            "accepted": len(self.accepted),
            "observed": len(self.observed),
        }

    @property
    def is_drained(self) -> bool:
        return not self._inflight

    def read_feedback(self, now_ns: int) -> dict:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns 必须是非负整数")
        return {
            "position": self._position,
            "capture_mono_ns": now_ns,
            "cursors": self.cursors(),
            "gripper": self.gripper_state,
        }

    def observed_generations(self) -> set:
        return {it["gen"] for it in self.observed}

    def submitted_generations(self) -> set:
        return {it["gen"] for it in self.submitted}
