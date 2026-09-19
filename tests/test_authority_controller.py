"""许可信任边界与确定性控制器回归。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import hmac
import threading

import pytest

from sentinel_evc.authority import Authority
from sentinel_evc.contracts import ErrorCode, LeaseContext, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore, establish_root
from sentinel_evc.events import ZERO_HASH
from sentinel_evc.scenarios import make_parent_pair, make_scene
from sentinel_evc.sim_controller import SimController


def rig():
    scene = make_scene(0)
    plan, _ = make_parent_pair(0)
    cert = establish_root(plan, scene).certificate
    store = CertificateStore()
    store.register(cert)
    context = LeaseContext("robot", 0, 0, scene.scene_id, 0, ZERO_HASH)
    snapshot = Snapshot("obs", **context.summary(), position=plan.points[0],
                        observed_mono=1_000_000_000)
    return Authority(store, key=b"test-only-hmac-key"), plan, cert, context, snapshot


def prepare(r, **kwargs):
    auth, plan, cert, context, snapshot = r
    return auth.prepare(plan, cert, context, snapshot, 1_000_000_000, **kwargs)


@pytest.mark.parametrize("age", [100_000_001, -1, float("nan")])
def test_prepare_rejects_invalid_observation_age(age):
    auth, plan, cert, context, snapshot = rig()
    with pytest.raises(Rejection) as exc:
        auth.prepare(plan, cert, context,
                     replace(snapshot, observed_mono=1_000_000_000 - age),
                     1_000_000_000)
    assert exc.value.code == ErrorCode.STATE_STALE


def test_prepare_accepts_age_and_tube_boundaries():
    auth, plan, cert, context, snapshot = rig()
    # y 起点为 0，避免为边界另加浮点容差。
    snapshot = replace(snapshot, observed_mono=900_000_000,
                       position=(plan.points[0][0], 0.005, plan.points[0][2]))
    lease = auth.prepare(plan, cert, context, snapshot, 1_000_000_000)
    assert lease.prefix_len == 4
    assert lease.deadline_mono == 1_500_000_000


def test_prepare_rejects_outside_five_mm_tube():
    auth, plan, cert, context, snapshot = rig()
    snapshot = replace(snapshot, position=(plan.points[0][0], 0.005001, plan.points[0][2]))
    with pytest.raises(Rejection) as exc:
        auth.prepare(plan, cert, context, snapshot, 1_000_000_000)
    assert exc.value.code == ErrorCode.TRACKING_TUBE


@pytest.mark.parametrize("field,value", [
    ("robot", "other"), ("boot", 1), ("epoch", 1), ("scene_id", "other"),
    ("queue_rev", 1), ("committed_prefix_hash", "sha256:" + "1" * 64),
])
def test_prepare_requires_snapshot_context_match(field, value):
    auth, plan, cert, context, snapshot = rig()
    with pytest.raises(Rejection) as exc:
        auth.prepare(plan, cert, context, replace(snapshot, **{field: value}), 1_000_000_000)
    assert exc.value.code == ErrorCode.CONTEXT_CHANGED


def test_prepare_uses_local_certificate_scene_not_caller_claim():
    auth, plan, cert, context, snapshot = rig()
    context = replace(context, scene_id="changed")
    snapshot = replace(snapshot, scene_id="changed")
    forged = replace(cert, scene_id="changed")
    with pytest.raises(Rejection) as exc:
        auth.prepare(plan, forged, context, snapshot, 1_000_000_000)
    assert exc.value.code == ErrorCode.CONTEXT_CHANGED


@pytest.mark.parametrize("prefix", [0, -1, 17, True, 1.5])
def test_prepare_rejects_invalid_prefix(prefix):
    with pytest.raises(Rejection) as exc:
        prepare(rig(), prefix_len=prefix)
    assert exc.value.code == ErrorCode.AUTH_FAILED


@pytest.mark.parametrize("ttl", [0, -1, 199_999_999, float("nan"), float("inf")])
def test_prepare_rejects_invalid_ttl(ttl):
    with pytest.raises(Rejection) as exc:
        prepare(rig(), ttl_ns=ttl)
    assert exc.value.code == ErrorCode.AUTH_FAILED


def test_prepare_accepts_exact_prefix_budget_without_unapproved_upper_limits():
    assert prepare(rig(), ttl_ns=200_000_000).prefix_len == 4
    assert prepare(rig(), prefix_len=16, ttl_ns=11_000_000_000).prefix_len == 16


def test_hmac_is_sha256_over_complete_lease_without_hmac():
    lease = prepare(rig())
    expected = hmac.new(b"test-only-hmac-key", lease.signing_bytes(), hashlib.sha256).hexdigest()
    assert lease.hmac == expected
    assert len(lease.hmac) == 64


@pytest.mark.parametrize("field,value", [
    ("lease_id", "forged"), ("final_hash", "sha256:" + "1" * 64), ("cert_id", "forged"),
    ("prefix_len", 3), ("deadline_mono", 1_600_000_000), ("hmac", "0" * 64),
    ("hmac", "伪造"), ("deadline_mono", float("nan")),
])
def test_failed_authentication_does_not_consume_valid_lease(field, value):
    r = rig()
    lease = prepare(r)
    with pytest.raises(Rejection) as exc:
        r[0].consume(replace(lease, **{field: value}))
    assert exc.value.code == ErrorCode.AUTH_FAILED
    assert not r[0].is_consumed(lease.lease_id)
    r[0].consume(lease)
    assert r[0].is_consumed(lease.lease_id)


def test_context_tampering_does_not_consume_valid_lease():
    r = rig()
    lease = prepare(r)
    with pytest.raises(Rejection) as exc:
        r[0].consume(replace(lease, context=replace(lease.context, epoch=1)))
    assert exc.value.code == ErrorCode.AUTH_FAILED
    assert not r[0].is_consumed(lease.lease_id)


def test_concurrent_consumption_has_one_winner():
    r = rig()
    lease = prepare(r)
    barrier = threading.Barrier(8)

    def consume_once():
        barrier.wait()
        try:
            r[0].consume(lease)
            return "consumed"
        except Rejection as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: consume_once(), range(8)))
    assert results.count("consumed") == 1
    assert results.count(ErrorCode.LEASE_REPLAY) == 7


def test_controller_advance_separates_ack_observation_and_capacity():
    controller = SimController(ack_delay_ticks=2, exec_delay_ticks=2)
    assert controller.submit((1, 0, 0), 0)
    assert controller.submit((2, 0, 0), 0)
    assert not controller.submit((3, 0, 0), 0)
    controller.advance()
    assert controller.cursors() == {"submitted": 2, "accepted": 0, "observed": 0}
    controller.advance()
    assert controller.cursors() == {"submitted": 2, "accepted": 2, "observed": 0}
    controller.advance()
    controller.advance()
    assert controller.cursors() == {"submitted": 2, "accepted": 2, "observed": 2}
    assert controller.free_slots() == 2


def test_controller_dropped_ack_still_allows_physical_observation():
    controller = SimController(drop_ack=True)
    controller.submit((1, 0, 0), 0)
    controller.advance()
    controller.advance()
    assert controller.cursors() == {"submitted": 1, "accepted": 0, "observed": 1}


def test_cancel_preserves_already_accepted_step_and_discards_pending():
    controller = SimController(exec_delay_ticks=3)
    controller.submit((1, 0, 0), 0)
    controller.advance()
    controller.submit((2, 0, 0), 0)
    controller.cancel()
    for _ in range(3):
        controller.advance()
    assert controller.cancel_acked is True
    assert controller.observed == [{"action": (1, 0, 0), "gen": 0}]


def test_cancel_ack_loss_does_not_claim_confirmation():
    controller = SimController(drop_cancel_ack=True)
    controller.cancel()
    for _ in range(5):
        controller.advance()
    assert controller.cancel_acked is None


@pytest.mark.parametrize("kwargs", [
    {"capacity": 0}, {"capacity": True}, {"ack_delay_ticks": -1},
    {"ack_delay_ticks": 1.5}, {"exec_delay_ticks": -1},
])
def test_controller_rejects_invalid_simulation_parameters(kwargs):
    with pytest.raises(ValueError):
        SimController(**kwargs)
