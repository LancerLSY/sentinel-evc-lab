"""事件格式、规范字节和有限容量的真实回归。"""

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from time import monotonic_ns

import pytest

from sentinel_evc.contracts import canonical_json
from sentinel_evc.events import EVENT_TYPES, EventLog, ZERO_HASH


PAYLOADS = {
    "PROPOSAL": {"role": "parent"},
    "TRANSFORM": {"method": "mix", "parent_plan_hashes": [ZERO_HASH]},
    "CERTIFICATE": {
        "verdict": "FULL", "path": "full", "margins": [0.1],
        "first_violation_segment": None, "full_checks_used": 1,
        "inherit_depth": 0, "parent_cert_id": None,
    },
    "PREPARE": {"prefix_len": 4, "deadline_mono": 0.5},
    "COMMIT": {"accepted": True, "reason_code": None},
    "DISPATCH": {"step_index": 0, "submitted": True},
    "CONTROLLER_ACK": {"step_index": 0, "accepted": True},
    "OBSERVED": {"step_index": 0, "position": [0.1, 0.0, 0.3], "observed": True},
    "REVOKE": {"reason": "测试", "old_epoch": 0, "new_epoch": 1},
    "CANCEL_ACK": {"confirmed": True},
    "BACKUP": {"reason": "已选备用", "backup_plan_hash": ZERO_HASH},
    "OUTCOME": {"status": "completed", "submitted": 1, "accepted": 1, "observed": 1},
    "LOG_GAP": {"dropped_count": 1, "first_dropped_seq": 4, "last_dropped_seq": 4},
}
COMMON_FIELDS = {
    "seq", "ts_mono_ns", "ts_utc", "run_id", "type", "plan_hash",
    "lease_id", "cert_id", "payload", "prev_hash",
}


def fixed_log(**kwargs):
    return EventLog("测试", monotonic_ns=lambda: 123,
                    utc_now=lambda: "2026-09-19T00:00:00Z", **kwargs)


@pytest.mark.parametrize("etype", PAYLOADS)
def test_all_thirteen_minimal_payloads(etype):
    log = fixed_log()
    event = log.append(etype, **PAYLOADS[etype])
    assert event.keys() == COMMON_FIELDS
    assert event["payload"] == PAYLOADS[etype]
    assert event["seq"] == 0
    assert event["ts_mono_ns"] == 123
    assert event["ts_utc"] == "2026-09-19T00:00:00Z"
    assert event["run_id"] == "测试"
    assert event["prev_hash"] == ZERO_HASH
    assert event["plan_hash"] is event["lease_id"] is event["cert_id"] is None
    assert set(EVENT_TYPES) == set(PAYLOADS)
    assert len(EVENT_TYPES) == 13


@pytest.mark.parametrize("etype", PAYLOADS)
def test_extra_and_missing_payload_fields_rejected_without_consuming_seq(etype):
    log = fixed_log()
    payload = deepcopy(PAYLOADS[etype])
    with pytest.raises(ValueError):
        log.append(etype, **payload, extra=True)
    payload.pop(next(iter(payload)))
    with pytest.raises(ValueError):
        log.append(etype, **payload)
    assert log.append("PROPOSAL", role="child")["seq"] == 0


@pytest.mark.parametrize("etype,field,value", [
    ("PROPOSAL", "role", "unknown"),
    ("TRANSFORM", "method", "perturb"),
    ("TRANSFORM", "parent_plan_hashes", ["bad"]),
    ("CERTIFICATE", "verdict", "BASELINE"),
    ("CERTIFICATE", "path", "unknown"),
    ("CERTIFICATE", "margins", [float("nan")]),
    ("CERTIFICATE", "inherit_depth", None),
    ("CERTIFICATE", "first_violation_segment", -1),
    ("PREPARE", "prefix_len", 0),
    ("PREPARE", "deadline_mono", float("inf")),
    ("COMMIT", "accepted", 1),
    ("COMMIT", "reason_code", "NEW_ERROR"),
    ("DISPATCH", "submitted", 1),
    ("CONTROLLER_ACK", "accepted", 1),
    ("OBSERVED", "position", [0, 0]),
    ("OBSERVED", "observed", 1),
    ("CANCEL_ACK", "confirmed", 1),
    ("OUTCOME", "status", "recovered"),
    ("LOG_GAP", "dropped_count", 0),
])
def test_payload_values_match_contract(etype, field, value):
    payload = dict(PAYLOADS[etype], **{field: value})
    with pytest.raises(ValueError):
        fixed_log().append(etype, **payload)


def test_chain_and_jsonl_use_same_utf8_bytes_without_lf_in_hash():
    log = fixed_log()
    first = log.append("PREPARE", plan_hash=ZERO_HASH, lease_id="许可",
                       cert_id="证书", **PAYLOADS["PREPARE"])
    second = log.append("PROPOSAL", role="child")
    expected_hash = "sha256:" + hashlib.sha256(canonical_json(first)).hexdigest()
    assert second["prev_hash"] == expected_hash
    assert log.tip_hash == "sha256:" + hashlib.sha256(canonical_json(second)).hexdigest()
    data = log.to_jsonl()
    assert data == canonical_json(first) + b"\n" + canonical_json(second) + b"\n"
    assert "许可".encode("utf-8") in data
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert [json.loads(line) for line in data.splitlines()] == log.events()


def test_overflow_preserves_queued_events_and_reports_dropped_sequence_range():
    log = fixed_log(maxlen=2)
    first = log.append("PROPOSAL", role="parent")
    second = log.append("PROPOSAL", role="child")
    assert log.append("PROPOSAL", role="child") is None
    assert log.append("PROPOSAL", role="child") is None
    assert log.count == 2
    assert log.events() == [first, second]
    old_tip = log.tip_hash
    assert log.drain() == [first, second]
    assert log.events() == []
    assert log.count == 2 and log.tip_hash == old_tip
    normal = log.append("PROPOSAL", role="child")
    gap, recorded = log.events()
    assert gap["seq"] == 4 and normal["seq"] == 5
    assert gap["type"] == "LOG_GAP"
    assert gap["payload"] == {
        "dropped_count": 2, "first_dropped_seq": 2, "last_dropped_seq": 3,
    }
    assert gap["prev_hash"] == old_tip
    assert recorded == normal
    assert normal["prev_hash"] == "sha256:" + hashlib.sha256(canonical_json(gap)).hexdigest()
    assert log.count == 4


def test_capacity_one_prioritizes_gap_and_tracks_new_drop():
    log = fixed_log(maxlen=1)
    log.append("PROPOSAL", role="parent")
    assert log.append("PROPOSAL", role="child") is None
    log.drain()
    assert log.append("PROPOSAL", role="child") is None
    first_gap = log.drain()[0]
    assert first_gap["seq"] == 2
    assert first_gap["payload"]["first_dropped_seq"] == 1
    assert log.append("PROPOSAL", role="child") is None
    next_gap = log.drain()[0]
    assert next_gap["seq"] == 4
    assert next_gap["payload"] == {
        "dropped_count": 1, "first_dropped_seq": 3, "last_dropped_seq": 3,
    }


def test_default_capacity_is_1024():
    log = fixed_log()
    for _ in range(1024):
        assert log.append("PROPOSAL", role="parent") is not None
    assert log.append("PROPOSAL", role="child") is None
    assert log.count == len(log.events()) == 1024


def test_empty_buffer_and_drain_do_not_reset_chain():
    log = fixed_log()
    assert log.to_jsonl() == b""
    assert log.drain() == []
    assert log.count == 0 and log.tip_hash == ZERO_HASH
    log.append("PROPOSAL", role="parent")
    tip = log.tip_hash
    log.drain()
    event = log.append("PROPOSAL", role="child")
    assert event["seq"] == 1 and event["prev_hash"] == tip
    assert log.count == 2


def test_caller_cannot_mutate_queued_event_or_hash_through_alias():
    log = fixed_log()
    position = [0.1, 0.0, 0.3]
    event = log.append("OBSERVED", step_index=0, position=position, observed=True)
    expected = log.to_jsonl(), log.tip_hash
    position[0] = 999.0
    event["payload"]["position"][0] = 999.0
    log.events()[0]["payload"]["position"][0] = 999.0
    log.by_type("OBSERVED")[0]["payload"]["position"][0] = 999.0
    assert (log.to_jsonl(), log.tip_hash) == expected


def test_real_clock_defaults():
    before = monotonic_ns()
    event = EventLog("clock").append("PROPOSAL", role="parent")
    assert before <= event["ts_mono_ns"] <= monotonic_ns()
    assert datetime.fromisoformat(event["ts_utc"]).utcoffset().total_seconds() == 0


@pytest.mark.parametrize("kwargs", [{"run_id": ""}, {"run_id": None},
                                    {"run_id": "x", "maxlen": 0},
                                    {"run_id": "x", "maxlen": True}])
def test_invalid_queue_parameters(kwargs):
    with pytest.raises(ValueError):
        EventLog(**kwargs)


def test_unknown_event_and_invalid_ids_leave_log_empty():
    log = fixed_log()
    with pytest.raises(ValueError):
        log.append("UNKNOWN")
    for ids in ({"plan_hash": "bad"}, {"cert_id": ""}, {"lease_id": 1}):
        with pytest.raises(ValueError):
            log.append("PROPOSAL", role="parent", **ids)
    assert log.count == 0 and log.tip_hash == ZERO_HASH
