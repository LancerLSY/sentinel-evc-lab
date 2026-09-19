"""通过独立 CLI 进程核对故障实验与落盘证据，不复用执行判定逻辑。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FAULT_CODES = {
    "late": "TRACKING_TUBE",
    "expired": "LEASE_EXPIRED",
    "replay": "LEASE_REPLAY",
    "scene-change": "CONTEXT_CHANGED",
    "queue-rev-change": "CONTEXT_CHANGED",
    "revoke-race": None,
    "cancel-unconfirmed": "CANCEL_UNCONFIRMED",
}


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "sentinel_evc", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )


@pytest.fixture(scope="module")
def fault_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fault-cli")
    runs = {}
    for fault in (*FAULT_CODES, "all"):
        out = directory / fault
        result = _cli("fault", "--fault", fault, "--seed", 1234, "--out", out)
        runs[fault] = (out, result)
    return runs


def _artifacts(fault_runs, fault):
    out, result = fault_runs[fault]
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in
              (out / "bundle" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return summary, events


@pytest.mark.parametrize("fault", [*FAULT_CODES, "all"])
def test_successfully_blocked_fault_experiment_exits_zero(fault_runs, fault):
    _, result = fault_runs[fault]
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fault", FAULT_CODES)
def test_single_fault_summary_contains_only_requested_case(fault_runs, fault):
    summary, _ = _artifacts(fault_runs, fault)
    assert [item["fault"] for item in summary["faults"]["details"]] == [fault]


@pytest.mark.parametrize("fault,code", FAULT_CODES.items())
def test_single_fault_is_blocked_for_expected_reason(fault_runs, fault, code):
    summary, _ = _artifacts(fault_runs, fault)
    detail = summary["faults"]["details"][0]
    assert (detail["blocked"], detail["code"]) == (True, code)


@pytest.mark.parametrize("fault", ["late", "expired", "replay", "scene-change",
                                  "queue-rev-change"])
def test_commit_rejection_reason_is_present_in_evidence(fault_runs, fault):
    _, events = _artifacts(fault_runs, fault)
    rejected = [event["payload"]["reason_code"] for event in events
                if event["type"] == "COMMIT" and not event["payload"]["accepted"]]
    assert rejected == [FAULT_CODES[fault]]


@pytest.mark.parametrize("fault", FAULT_CODES)
def test_single_fault_cursors_match_successful_recorded_events(fault_runs, fault):
    summary, events = _artifacts(fault_runs, fault)
    counts = {
        field: sum(event["type"] == kind and event["payload"][field] is True
                   for event in events)
        for field, kind in (("submitted", "DISPATCH"),
                            ("accepted", "CONTROLLER_ACK"), ("observed", "OBSERVED"))
    }
    assert summary["faults"]["details"][0]["cursors"] == counts


@pytest.mark.parametrize("fault", FAULT_CODES)
def test_seven_faults_have_no_submissions_from_revoked_lease(fault_runs, fault):
    summary, events = _artifacts(fault_runs, fault)
    revoked = set()
    after_revoke = []
    for event in events:
        if event["type"] == "REVOKE":
            revoked.add(event["lease_id"])
        elif (event["type"] == "DISPATCH" and event["payload"]["submitted"]
              and event["lease_id"] in revoked):
            after_revoke.append(event)
    assert len(after_revoke) == summary["faults"]["details"][0][
        "stale_submissions_after_revoke"] == 0


def test_all_faults_summary_reports_seven_real_blocked_cases(fault_runs):
    summary, _ = _artifacts(fault_runs, "all")
    faults = summary["faults"]
    assert {item["fault"]: (item["blocked"], item["code"])
            for item in faults["details"]} == {
        fault: (True, code) for fault, code in FAULT_CODES.items()
    }
    assert faults["faults_run"] == faults["blocked"] == 7


def test_all_faults_cursor_totals_match_shared_event_stream(fault_runs):
    summary, events = _artifacts(fault_runs, "all")
    for field, kind in (("submitted", "DISPATCH"),
                        ("accepted", "CONTROLLER_ACK"), ("observed", "OBSERVED")):
        assert sum(item["cursors"][field] for item in summary["faults"]["details"]) == sum(
            event["type"] == kind and event["payload"][field] is True for event in events
        )


def test_revoke_race_preserves_observation_of_previously_accepted_step(fault_runs):
    _, events = _artifacts(fault_runs, "revoke-race")
    revoke = next(event for event in events if event["type"] == "REVOKE")
    accepted = {(event["lease_id"], event["payload"]["step_index"])
                for event in events if event["type"] == "CONTROLLER_ACK"
                and event["payload"]["accepted"] and event["seq"] < revoke["seq"]}
    later = {(event["lease_id"], event["payload"]["step_index"])
             for event in events if event["type"] == "OBSERVED"
             and event["payload"]["observed"] and event["seq"] > revoke["seq"]}
    assert later and later <= accepted


def test_cancel_without_confirmation_does_not_resume_execution(fault_runs):
    _, events = _artifacts(fault_runs, "cancel-unconfirmed")
    revoke = next(event for event in events if event["type"] == "REVOKE")
    later = [event for event in events if event["seq"] > revoke["seq"]]
    assert not [event for event in later if
                (event["type"] == "CANCEL_ACK" and event["payload"]["confirmed"])
                or (event["type"] == "COMMIT" and event["payload"]["accepted"])
                or (event["type"] == "DISPATCH" and event["payload"]["submitted"])]


@pytest.mark.parametrize("fault", [*FAULT_CODES, "all"])
def test_fault_bundle_verifies_with_output_directory_run_id(fault_runs, fault):
    out, _ = fault_runs[fault]
    summary, events = _artifacts(fault_runs, fault)
    assert summary["run_id"] == out.name
    assert {event["run_id"] for event in events} == {out.name}
    result = _cli("verify", "--bundle", out / "bundle", "--public-key",
                  out / "anchors" / "demo.public", "--run-id", out.name)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("populated", [False, True])
def test_existing_output_directory_is_rejected_without_changing_contents(tmp_path, populated):
    out = tmp_path / "existing"
    out.mkdir()
    if populated:
        (out / "keep.txt").write_bytes(b"existing user artifact\r\n")
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*")
              if path.is_file()}
    result = _cli("fault", "--fault", "all", "--seed", 1234, "--out", out)
    assert result.returncode == 1, result.stdout + result.stderr
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*")
            if path.is_file()} == before


def test_unknown_fault_is_parameter_error_and_does_not_create_output(tmp_path):
    out = tmp_path / "invalid"
    result = _cli("fault", "--fault", "unknown", "--seed", 1234, "--out", out)
    assert result.returncode == 2
    assert not out.exists()
