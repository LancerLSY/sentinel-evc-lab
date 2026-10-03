"""事件记录。

13 种事件类型首版就全部定义，不要后加 —— 事件 schema 是 A 和 B 的
解耦点，冻结之后改动要两人同意。

原始动作、批准动作、发送动作和反馈不能共用一列。没有反馈时输出 unknown，
不得把「发送成功」写成「实际完成」。
"""

from __future__ import annotations

import json
import threading
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

PRODUCT_EVENT_TYPES = EVENT_TYPES + (
    "PREDICTION",
    "LEASE",
    "CANCEL_REQUEST",
    "CANCEL_ACCEPTED",
)

ZERO_HASH = "sha256:" + "0" * 64


class EventLog:
    """有界事件队列。

    队列有上限，满了就显式丢弃并记一条 LOG_GAP —— 不无限堆内存，
    但缺口必须可见。需要完整证据的运行模式下，出现缺口应停止批准新任务。
    """

    def __init__(
        self, run_id: str, maxlen: int = 100_000, schema_version: str = "v0.1"
    ):
        if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
            raise ValueError("run_id 必须是有界非空字符串")
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < 2:
            raise ValueError("maxlen 至少为 2")
        if schema_version not in ("v0.1", "product-v1"):
            raise ValueError("未知事件 schema_version")
        self.run_id = run_id
        self.schema_version = schema_version
        self._events: deque = deque()
        self._maxlen = maxlen
        self._seq = 0
        self._prev_hash = ZERO_HASH
        self._gap_count = 0
        self._has_gap = False
        self._lock = threading.RLock()
        self._reservations = {}
        self._reserved = 0
        self._pinned = {}

    @property
    def allowed_types(self) -> tuple:
        return EVENT_TYPES if self.schema_version == "v0.1" else PRODUCT_EVENT_TYPES

    def supports(self, etype: str) -> bool:
        return etype in self.allowed_types

    def can_append_without_gap(self, count: int = 1) -> bool:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count 必须是非负整数")
        with self._lock:
            return not self._has_gap and len(self._events) + self._reserved + count <= self._maxlen

    def reserve(self, count: int):
        """Reserve evidence slots, retaining two unreserved emergency slots."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        with self._lock:
            if self._has_gap or len(self._events) + self._reserved + count + 2 > self._maxlen:
                return None
            token = object()
            self._reservations[token] = count
            self._reserved += count
            self._pinned[token] = set()
            return token

    def append_reserved(self, token, etype: str, **payload) -> dict:
        if etype not in self.allowed_types:
            raise ValueError(f"未定义的事件类型: {etype}")
        payload = json.loads(canonical_json(payload).decode("utf-8"))
        with self._lock:
            count = self._reservations.get(token, 0)
            if count < 1:
                raise ValueError("evidence reservation is absent or exhausted")
            self._reservations[token] = count - 1
            self._reserved -= 1
            event = self._append_one(etype, payload)
            self._pinned[token].add(event["seq"])
            return event

    def release(self, token) -> None:
        with self._lock:
            self._reserved -= self._reservations.pop(token, 0)
            self._pinned.pop(token, None)

    def entry_guard(self):
        """Serialize the writer-entry decision with creation of an evidence gap."""
        return self._lock

    def _drop_unreserved(self) -> None:
        pinned = set().union(*self._pinned.values()) if self._pinned else set()
        for event in self._events:
            if event["seq"] not in pinned:
                self._events.remove(event)
                return
        raise RuntimeError("no unreserved evidence slot available")

    def _append_one(self, etype: str, payload: dict) -> dict:
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

    def append(self, etype: str, **payload) -> dict:
        if etype not in self.allowed_types:
            raise ValueError(f"未定义的事件类型: {etype}")
        # Canonical round-trip both rejects NaN/unsupported objects and takes
        # an immutable JSON-compatible copy of mapping proxies/tuples.
        payload = json.loads(canonical_json(payload).decode("utf-8"))
        with self._lock:
            limit = self._maxlen - self._reserved
            if etype == "LOG_GAP":
                while len(self._events) >= limit:
                    self._drop_unreserved()
                    self._gap_count += 1
                self._has_gap = True
                return self._append_one(etype, payload)

            if len(self._events) >= limit:
                dropped_now = 0
                # Reserve one slot for the visible gap marker and one for the
                # event that detected it. A fresh marker stays visible even
                # after repeated overflow.
                while len(self._events) > limit - 2:
                    self._drop_unreserved()
                    dropped_now += 1
                self._gap_count += dropped_now
                self._has_gap = True
                self._append_one(
                    "LOG_GAP",
                    {
                        "detail": "bounded event buffer overflow",
                        "dropped": dropped_now,
                        "dropped_total": self._gap_count,
                    },
                )
            return self._append_one(etype, payload)

    def note_gap(self, detail: str) -> None:
        self.append("LOG_GAP", detail=detail, dropped=self._gap_count)

    @property
    def has_gap(self) -> bool:
        return self._has_gap

    @property
    def tip_hash(self) -> str:
        return self._prev_hash

    @property
    def count(self) -> int:
        return self._seq

    def events(self) -> list:
        with self._lock:
            return list(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(ev, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for ev in self._events
        ) + "\n"

    def by_type(self, etype: str) -> list:
        return [e for e in self._events if e["type"] == etype]
