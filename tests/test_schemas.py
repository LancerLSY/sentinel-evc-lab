"""冻结 schema 与手写格式 fixture 对拍；不把 fixture 当作运行成绩。"""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EVENT_TYPES = {
    "PROPOSAL", "TRANSFORM", "CERTIFICATE", "PREPARE", "COMMIT", "DISPATCH",
    "CONTROLLER_ACK", "OBSERVED", "REVOKE", "CANCEL_ACK", "BACKUP", "OUTCOME",
    "LOG_GAP",
}


def schema(name):
    return json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text("utf-8"))


def assert_schema(value, rule):
    """只解释这三份固定 schema 使用的关键字，不能作为通用校验器。"""
    if "oneOf" in rule:
        matches = 0
        for alternative in rule["oneOf"]:
            try:
                assert_schema(value, alternative)
            except AssertionError:
                continue
            matches += 1
        assert matches == 1
    if "const" in rule:
        assert value == rule["const"]
    if "enum" in rule:
        assert value in rule["enum"]
    types = rule.get("type", [])
    if isinstance(types, str):
        types = [types]
    actual = {dict: "object", list: "array", str: "string", int: "integer",
              float: "number", bool: "boolean", type(None): "null"}[type(value)]
    if types:
        assert actual in types or (actual == "integer" and "number" in types)
    if isinstance(value, dict):
        properties = rule.get("properties", {})
        assert set(rule.get("required", [])) <= value.keys()
        if rule.get("additionalProperties") is False:
            assert value.keys() <= properties.keys()
        for key in value.keys() & properties.keys():
            assert_schema(value[key], properties[key])
    elif isinstance(value, list):
        assert len(value) >= rule.get("minItems", 0)
        assert len(value) <= rule.get("maxItems", math.inf)
        for item in value:
            assert_schema(item, rule.get("items", {}))
    elif isinstance(value, str):
        assert len(value) >= rule.get("minLength", 0)
        if "pattern" in rule:
            assert re.fullmatch(rule["pattern"], value)
    elif type(value) in (int, float):
        assert math.isfinite(value)
        assert value >= rule.get("minimum", -math.inf)
        assert value > rule.get("exclusiveMinimum", -math.inf)


def load_events():
    return [json.loads(line) for line in (FIXTURES / "sample_events.jsonl").read_bytes().splitlines()]


def test_all_schema_objects_are_closed():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) <= node["properties"].keys()
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    for name in ("event", "scenario", "verdict"):
        document = schema(name)
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        check(document)


def test_event_fixture_covers_all_types_and_validates():
    events = load_events()
    assert len(events) == 20
    assert {event["type"] for event in events} == EVENT_TYPES
    for event in events:
        assert_schema(event, schema("event"))


def test_event_fixture_matches_production_canonical_bytes():
    from sentinel_evc.contracts import canonical_json

    for line in (FIXTURES / "sample_events.jsonl").read_bytes().splitlines():
        assert line == canonical_json(json.loads(line))


def test_event_fixture_chain_and_gap():
    raw = (FIXTURES / "sample_events.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    previous_hash = "sha256:" + "0" * 64
    previous_seq = -1
    for line in raw.splitlines():
        event = json.loads(line)
        assert event["prev_hash"] == previous_hash
        if event["type"] == "LOG_GAP":
            gap = event["payload"]
            assert gap["first_dropped_seq"] == previous_seq + 1
            assert gap["last_dropped_seq"] == event["seq"] - 1
            assert gap["dropped_count"] == event["seq"] - previous_seq - 1
        else:
            assert event["seq"] == previous_seq + 1
        previous_hash = "sha256:" + sha256(line).hexdigest()
        previous_seq = event["seq"]


def test_event_rejects_missing_fields_extra_fields_and_wrong_payload():
    rule = schema("event")
    for event in load_events():
        for field in event:
            bad = deepcopy(event)
            del bad[field]
            with pytest.raises(AssertionError):
                assert_schema(bad, rule)
        for location in ("root", "payload"):
            bad = deepcopy(event)
            target = bad if location == "root" else bad["payload"]
            target["unexpected"] = True
            with pytest.raises(AssertionError):
                assert_schema(bad, rule)
        for field in event["payload"]:
            bad = deepcopy(event)
            del bad["payload"][field]
            with pytest.raises(AssertionError):
                assert_schema(bad, rule)
            bad = deepcopy(event)
            bad["payload"][field] = {}
            with pytest.raises(AssertionError):
                assert_schema(bad, rule)
        bad = deepcopy(event)
        bad["type"] = "UNKNOWN"
        with pytest.raises(AssertionError):
            assert_schema(bad, rule)


def test_event_rejects_wrong_scalar_and_enum_values():
    event = load_events()[0]
    for key, value in (("seq", True), ("ts_mono_ns", -1),
                       ("ts_utc", "2026-09-19T00:00:00+00:00"),
                       ("prev_hash", "sha256:" + "A" * 64),
                       ("plan_hash", "sha256:bad")):
        bad = deepcopy(event)
        bad[key] = value
        with pytest.raises(AssertionError):
            assert_schema(bad, schema("event"))
    bad = deepcopy(event)
    bad["payload"]["role"] = "backup"
    with pytest.raises(AssertionError):
        assert_schema(bad, schema("event"))


def test_scenario_fixture_and_nested_extra_fields():
    scenario = json.loads((FIXTURES / "scenario.json").read_text("utf-8"))
    rule = schema("scenario")
    assert_schema(scenario, rule)
    without_seed = deepcopy(scenario)
    del without_seed["seed"]
    assert_schema(without_seed, rule)
    for location in ("root", "workspace", "obstacle"):
        bad = deepcopy(scenario)
        target = {"root": bad, "workspace": bad["workspace"],
                  "obstacle": bad["obstacles"][0]}[location]
        target["unexpected"] = True
        with pytest.raises(AssertionError):
            assert_schema(bad, rule)


def test_verdict_fixture_is_formal_path_only():
    verdict = json.loads((FIXTURES / "verdict.json").read_text("utf-8"))
    rule = schema("verdict")
    assert_schema(verdict, rule)
    assert set(rule["required"]) == rule["properties"].keys()
    for key, value in (("path", "parent_only"), ("verdict", "baseline_allowed"),
                       ("baseline_allowed", True), ("cert_id", "extra-cert"),
                       ("reason_code", "UNKNOWN")):
        bad = deepcopy(verdict)
        bad[key] = value
        with pytest.raises(AssertionError):
            assert_schema(bad, rule)
    for key in verdict:
        bad = deepcopy(verdict)
        del bad[key]
        with pytest.raises(AssertionError):
            assert_schema(bad, rule)
