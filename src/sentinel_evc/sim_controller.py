"""模拟控制器。

这是一个**数值模拟**，不是任何真实控制器的模型。它存在的唯一目的是
让三个游标（submitted / accepted / observed）分离，从而能测试这句话：

    撤销之后不再新增旧代次提交，但撤销之前已经提交的命令仍可能被执行。

清空 Python 队列不等于电机停止。真机演示必须按具体控制器实测，
不能把清队列画成瞬间制动。
"""

from __future__ import annotations

from collections import deque
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
        drop_ack: bool = False,
    ):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity 必须为正整数")
        if any(type(delay) is not int or delay < 0
               for delay in (ack_delay_ticks, exec_delay_ticks)):
            raise ValueError("控制器延迟必须为非负整数周期")
        self.capacity = capacity
        self.ack_delay_ticks = ack_delay_ticks
        self.exec_delay_ticks = exec_delay_ticks
        self.drop_cancel_ack = drop_cancel_ack
        self.drop_ack = drop_ack

        self._inflight = deque()  # [{action, gen, age, state}]
        self.submitted = []  # 已发出
        self.accepted = []  # 控制器已确认
        self.observed = []  # 已实际执行（模拟）

        self._cancel_pending = False
        self._cancel_age = 0
        self.cancel_acked: Optional[bool] = None

    # ------------------------------------------------------------ 提交

    def free_slots(self) -> int:
        return self.capacity - len(self._inflight)

    def submit(self, action, generation: int) -> bool:
        """提交一步。队列满时返回 False —— 这不是错误，是正常背压。"""
        if len(self._inflight) >= self.capacity:
            return False
        item = {"action": action, "gen": generation, "age": 0, "state": "submitted"}
        self._inflight.append(item)
        self.submitted.append({"action": action, "gen": generation})
        return True

    # ------------------------------------------------------------ 取消

    def cancel(self) -> None:
        """请求取消。ACK 是异步的，而且可能丢失。"""
        self._cancel_pending = True
        self._cancel_age = 0
        self.cancel_acked = None

    # ------------------------------------------------------------ 推进

    def advance(self) -> None:
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
                # ACK 丢失只影响主机可见确认，不把控制器已接受的动作当作未执行。
                if not self.drop_ack:
                    self.accepted.append({"action": item["action"], "gen": item["gen"]})
            elif item["state"] == "accepted" and item["age"] >= self.exec_delay_ticks:
                item["state"] = "observed"
                self.observed.append({"action": item["action"], "gen": item["gen"]})
                done.append(item)

        for item in done:
            self._inflight.remove(item)

    def tick(self) -> None:
        """保留旧执行器入口；确定性推进由 advance 实现。"""
        self.advance()

    # ------------------------------------------------------------ 查询

    def cursors(self) -> dict:
        return {
            "submitted": len(self.submitted),
            "accepted": len(self.accepted),
            "observed": len(self.observed),
        }

    def observed_generations(self) -> set:
        return {it["gen"] for it in self.observed}

    def submitted_generations(self) -> set:
        return {it["gen"] for it in self.submitted}
