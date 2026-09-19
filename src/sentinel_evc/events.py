"""有界事件队列与规范 JSONL；排队失败保留序号并显式报告缺口。"""

from __future__ import annotations

import math
import re
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from threading import Lock
from time import monotonic_ns

from .contracts import canonical_json, sha256_hex


PAYLOAD_FIELDS = {
    "PROPOSAL": {"role"},
    "TRANSFORM": {"method", "parent_plan_hashes"},
    "CERTIFICATE": {
        "verdict", "path", "margins", "first_violation_segment",
        "full_checks_used", "inherit_depth", "parent_cert_id",
    },
    "PREPARE": {"prefix_len", "deadline_mono"},
    "COMMIT": {"accepted", "reason_code"},
    "DISPATCH": {"step_index", "submitted"},
    "CONTROLLER_ACK": {"step_index", "accepted"},
    "OBSERVED": {"step_index", "position", "observed"},
    "REVOKE": {"reason", "old_epoch", "new_epoch"},
    "CANCEL_ACK": {"confirmed"},
    "BACKUP": {"reason", "backup_plan_hash"},
    "OUTCOME": {"status", "submitted", "accepted", "observed"},
    "LOG_GAP": {"dropped_count", "first_dropped_seq", "last_dropped_seq"},
}
EVENT_TYPES = tuple(PAYLOAD_FIELDS)
ZERO_HASH = "sha256:" + "0" * 64
PAYLOAD_ENUMS = {
    "role": {"parent", "child"},
    "method": {"mix", "near"},
    "verdict": {"FULL", "INHERITED", "REJECTED"},
    "path": {"full", "delta"},
    "status": {"completed", "rejected", "fault"},
    "reason_code": {
        None, "STATE_STALE", "CONTEXT_CHANGED", "LEASE_REPLAY", "TRACKING_TUBE",
        "CANCEL_UNCONFIRMED", "LEASE_EXPIRED", "AUTH_FAILED", "QUEUE_FULL",
    },
}


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _number(value) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validate_payload(etype: str, payload: dict) -> None:
    if etype not in PAYLOAD_FIELDS:
        raise ValueError(f"未定义的事件类型: {etype}")
    if payload.keys() != PAYLOAD_FIELDS[etype]:
        raise ValueError(f"{etype} payload 字段不符合合同")
    for field, value in payload.items():
        if field in PAYLOAD_ENUMS:
            valid = value in tuple(PAYLOAD_ENUMS[field])
        elif (etype, field) in {
            ("COMMIT", "accepted"),
            ("DISPATCH", "submitted"),
            ("CONTROLLER_ACK", "accepted"),
            ("OBSERVED", "observed"),
            ("CANCEL_ACK", "confirmed"),
        }:
            valid = type(value) is bool
        elif field in {"margins", "position"}:
            valid = isinstance(value, (list, tuple)) and all(_number(v) for v in value)
            if field == "position":
                valid = valid and len(value) == 3
        elif field == "parent_plan_hashes":
            valid = isinstance(value, (list, tuple)) and all(_digest(v) for v in value)
        elif field == "backup_plan_hash":
            valid = _digest(value)
        elif field == "parent_cert_id":
            valid = value is None or (isinstance(value, str) and bool(value))
        elif field == "reason":
            valid = isinstance(value, str)
        elif field == "deadline_mono":
            valid = _number(value)
        elif field == "first_violation_segment" and value is None:
            valid = True
        else:
            valid = type(value) is int and value >= (1 if field in {"prefix_len", "dropped_count"} else 0)
        if not valid:
            raise ValueError(f"{etype}.{field} 不符合合同")
    canonical_json(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EventLog:
    """只持有尚未消费的事件；count/tip_hash 覆盖整个流，包括已 drain 的事件。"""

    def __init__(self, run_id: str, maxlen: int = 1024, *, spool=None,
                 monotonic_ns=monotonic_ns, utc_now=_utc_now):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id 必须为非空字符串")
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("事件队列容量必须为正整数")
        self.run_id = run_id
        # 序号、链摘要、队列和丢弃范围属于同一次记录事务。
        self._lock = Lock()
        self._events: deque = deque()
        self._maxlen = maxlen
        self._spool = spool
        self._spooled_count = 0
        self._type_counts = {event_type: 0 for event_type in EVENT_TYPES}
        self._monotonic_ns = monotonic_ns
        self._utc_now = utc_now
        self._seq = 0
        self._count = 0
        self._prev_hash = ZERO_HASH
        self._gap_count = 0
        self._first_dropped_seq = None
        self._last_dropped_seq = None

    def append(self, etype: str, *, plan_hash=None, lease_id=None,
               cert_id=None, **payload) -> dict | None:
        _validate_payload(etype, payload)
        if plan_hash is not None and not _digest(plan_hash):
            raise ValueError("plan_hash 不符合摘要格式")
        for value in (lease_id, cert_id):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("事件 ID 必须为非空字符串或 null")
        with self._lock:
            if self._spool is not None and len(self._events) >= self._maxlen:
                self._flush_to_spool()
            if self._gap_count and len(self._events) < self._maxlen:
                self._record_gap()
            if len(self._events) >= self._maxlen:
                # 丢当前尝试，保留已排队事件；缺失序号由后续 LOG_GAP 解释。
                if not self._gap_count:
                    self._first_dropped_seq = self._seq
                self._last_dropped_seq = self._seq
                self._gap_count += 1
                self._seq += 1
                return None
            return self._record(etype, payload, plan_hash, lease_id, cert_id)

    def _record(self, etype, payload, plan_hash=None, lease_id=None, cert_id=None):
        event = {
            "seq": self._seq,
            "ts_mono_ns": self._monotonic_ns(),
            "ts_utc": self._utc_now(),
            "run_id": self.run_id,
            "type": etype,
            "plan_hash": plan_hash,
            "lease_id": lease_id,
            "cert_id": cert_id,
            "payload": deepcopy(payload),
            "prev_hash": self._prev_hash,
        }
        self._prev_hash = sha256_hex(event)
        self._seq += 1
        self._count += 1
        self._type_counts[etype] += 1
        self._events.append(event)
        return deepcopy(event)

    def _flush_to_spool(self) -> None:
        for event in self._events:
            self._spool.write(canonical_json(event) + b"\n")
        self._spool.flush()
        self._spooled_count += len(self._events)
        self._events.clear()

    def _record_gap(self) -> dict:
        event = self._record("LOG_GAP", {
            "dropped_count": self._gap_count,
            "first_dropped_seq": self._first_dropped_seq,
            "last_dropped_seq": self._last_dropped_seq,
        })
        self._gap_count = 0
        self._first_dropped_seq = self._last_dropped_seq = None
        return event

    @property
    def tip_hash(self) -> str:
        with self._lock:
            return self._prev_hash

    @property
    def count(self) -> int:
        """累计成功记录数；不包含丢弃尝试，包含 LOG_GAP。"""
        with self._lock:
            return self._count

    def events(self) -> list[dict]:
        with self._lock:
            return deepcopy(list(self._events))

    def type_count(self, event_type: str) -> int:
        with self._lock:
            return self._type_counts.get(event_type, 0)

    def drain(self) -> list[dict]:
        """将当前缓冲事件交给消费者并释放容量；流的序号和链不重置。"""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            if self._gap_count:
                events.append(self._record_gap())
                self._events.clear()
            return events

    def to_jsonl(self) -> bytes:
        with self._lock:
            if self._gap_count:
                raise RuntimeError("存在尚未记录的 LOG_GAP；先 drain 消费完整批次")
            output = BytesIO()
            self._write_jsonl(output)
            return output.getvalue()

    def write_jsonl(self, path) -> int:
        """把已消费和仍在队列中的完整事件流写入目标文件。"""
        with self._lock:
            if self._gap_count:
                raise RuntimeError("存在尚未记录的 LOG_GAP；先 drain 消费完整批次")
            with open(path, "wb") as output:
                self._write_jsonl(output)
            return self._spooled_count + len(self._events)

    def _write_jsonl(self, output) -> None:
        if self._spool is not None:
            position = self._spool.tell()
            self._spool.seek(0)
            while chunk := self._spool.read(65536):
                output.write(chunk)
            self._spool.seek(position)
        for event in self._events:
            output.write(canonical_json(event) + b"\n")

    def by_type(self, etype: str) -> list[dict]:
        with self._lock:
            return deepcopy([event for event in self._events if event["type"] == etype])
