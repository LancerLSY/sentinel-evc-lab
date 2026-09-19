"""七类故障使用真实执行器和控制器生成同流事件。"""

import pytest

from sentinel_evc.events import EventLog
from sentinel_evc.pipeline import FAULTS, _run_one_fault, act_two_faults


@pytest.mark.parametrize("fault,code", [
    ("late", "TRACKING_TUBE"),
    ("expired", "LEASE_EXPIRED"),
    ("replay", "LEASE_REPLAY"),
    ("scene-change", "CONTEXT_CHANGED"),
    ("queue-rev-change", "CONTEXT_CHANGED"),
    ("revoke-race", None),
    ("cancel-unconfirmed", "CANCEL_UNCONFIRMED"),
])
def test_fault_is_blocked_with_expected_reason(fault, code):
    log = EventLog("fault-test")
    result = _run_one_fault(fault, log)
    assert result["fault"] == fault
    assert result["blocked"]
    assert result["code"] == code
    assert result["stale_submissions_after_revoke"] == 0
    assert len(log.by_type("DISPATCH")) == result["cursors"]["submitted"]
    assert len(log.by_type("CONTROLLER_ACK")) == result["cursors"]["accepted"]
    assert len(log.by_type("OBSERVED")) == result["cursors"]["observed"]


def test_seven_faults_share_stream_without_duplicate_observations():
    log = EventLog("all-faults")
    result = act_two_faults(log)
    assert result["faults_run"] == result["blocked"] == 7
    assert result["stale_gen_submissions_after_revoke"] == 0
    assert [item["fault"] for item in result["details"]] == FAULTS
    observed = log.by_type("OBSERVED")
    assert len(observed) == len({(item["lease_id"], item["payload"]["step_index"])
                                 for item in observed})


def test_accepted_step_can_be_observed_after_revoke():
    log = EventLog("revoke-order")
    result = _run_one_fault("revoke-race", log)
    revoke = log.by_type("REVOKE")[0]
    accepted = {item["payload"]["step_index"] for item in log.by_type("CONTROLLER_ACK")
                if item["seq"] < revoke["seq"]}
    late_observed = [item for item in log.by_type("OBSERVED") if item["seq"] > revoke["seq"]]
    assert late_observed
    assert all(item["payload"]["step_index"] in accepted for item in late_observed)
    assert all(item["seq"] < revoke["seq"] for item in log.by_type("DISPATCH"))
    assert result["cursors"]["submitted"] == result["submitted_before_revoke"]
