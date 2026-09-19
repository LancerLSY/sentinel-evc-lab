"""事件记录。

13 种事件类型首版就全部定义，不要后加 —— 事件 schema 是 A 和 B 的
解耦点，冻结之后改动要两人同意。

原始动作、批准动作、发送动作和反馈不能共用一列。没有反馈时输出 unknown，
不得把「发送成功」写成「实际完成」。
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional

from .contracts import canonical_json, sha256_hex

EVENT_TYPES = (
    "PROPOSAL",
    "TRANSFORM",
    "CERTIFICATE",
    "PREPARE",
    "COMMIT",
    "DISPATCH",
    "CONTROLLER_ACK",
    "OBSERVED",
    "REVOKE",
    "CANCEL_ACK",
    "BACKUP",
    "OUTCOME",
    "LOG_GAP",
)

ZERO_HASH = "sha256:" + "0" * 64


class EventLog:
    """有界事件队列。

    队列有上限，满了就显式丢弃并记一条 LOG_GAP —— 不无限堆内存，
    但缺口必须可见。需要完整证据的运行模式下，出现缺口应停止批准新任务。
    """

    def __init__(self, run_id: str, maxlen: int = 100_000):
        self.run_id = run_id
        self._events: deque = deque()
        self._maxlen = maxlen
        self._seq = 0
        self._prev_hash = ZERO_HASH
        self._gap_count = 0

    def append(self, etype: str, **payload) -> dict:
        if etype not in EVENT_TYPES:
            raise ValueError(f"未定义的事件类型: {etype}")

        if len(self._events) >= self._maxlen:
            # 丢最旧的，并把缺口记下来
            self._events.popleft()
            self._gap_count += 1

        ev = {
            "seq": self._seq,
            "run_id": self.run_id,
            "type": etype,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        self._seq += 1
        self._prev_hash = sha256_hex(ev)
        self._events.append(ev)
        return ev

    def note_gap(self, detail: str) -> None:
        self.append("LOG_GAP", detail=detail, dropped=self._gap_count)

    @property
    def tip_hash(self) -> str:
        return self._prev_hash

    @property
    def count(self) -> int:
        return self._seq

    def events(self) -> list:
        return list(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(ev, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for ev in self._events
        ) + "\n"

    def by_type(self, etype: str) -> list:
        return [e for e in self._events if e["type"] == etype]
