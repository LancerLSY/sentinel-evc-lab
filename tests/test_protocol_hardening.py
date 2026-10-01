"""Regression tests for the product-v1 protocol boundary."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from sentinel_evc.authority import Authority, MAX_OBS_AGE_NS
from sentinel_evc.contracts import (
    Context,
    ErrorCode,
    Plan,
    Rejection,
    Snapshot,
    TransformRecord,
)
from sentinel_evc.delta_cert import CertificateStore, establish_root, validate_or_inherit
from sentinel_evc.events import EventLog
from sentinel_evc.executor import Executor, ExecutorState
from sentinel_evc.scenarios import make_parent_pair, make_scene, perturb
from sentinel_evc.sim_controller import SimController


NOW = 1_000_000_000


def _rig(*, maxlen=100_000, gripper_events=()):
    scene = make_scene(0)
    original, _ = make_parent_pair(0)
    plan = Plan("test", original.knots, original.dt, gripper_events=gripper_events)
    verdict = establish_root(plan, scene)
    store = CertificateStore()
    store.register(verdict.certificate)
    events = EventLog("hardening", maxlen=maxlen, schema_version="product-v1")
    authority = Authority(store, events=events)
    controller = SimController(initial_position=plan.knots[0])
    executor = Executor(authority, controller, events)
    context = Context(scene_id=scene.scene_id)
    snapshot = Snapshot("obs-0", plan.knots[0], NOW)
    return scene, plan, verdict.certificate, authority, controller, executor, events, context, snapshot


def _fresh(controller, seq, now, *, supported=None):
    feedback = controller.read_feedback(now)
    return Snapshot(
        f"obs-{seq}", feedback["position"], now, supported=supported
    )


def test_plan_rejects_non_finite_wrong_dimension_and_dt():
    factories = [
        lambda: Plan("nan", ((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)), 0.05),
        lambda: Plan("inf", ((0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)), 0.05),
        lambda: Plan("dim", ((0.0, 0.0), (1.0, 0.0)), 0.05),
        lambda: Plan("dt", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0),
    ]
    for factory in factories:
        with pytest.raises((TypeError, ValueError)):
            factory()


def test_hashed_dtos_are_deeply_immutable():
    parent, child = make_parent_pair(0)
    record = TransformRecord("tf", "perturb", parent.hash, child.hash, {"x": [1, 2]})
    with pytest.raises(TypeError):
        record.parameters["x"] = 3
    with pytest.raises((TypeError, AttributeError, FrozenInstanceError)):
        record.parameters["x"].append(4)


def test_wrong_parent_plan_and_transform_hashes_force_full_check():
    scene = make_scene(0)
    safe_parent, other = make_parent_pair(0)
    root = establish_root(safe_parent, scene)
    child, good = perturb(safe_parent, seed=2)

    wrong_parent = validate_or_inherit(child, scene, other, root.certificate, good)
    assert wrong_parent.full_checks_used == 1
    assert wrong_parent.reason_code == "parent_plan_hash_mismatch"

    bad_transform = TransformRecord(
        "bad", "perturb", other.hash, safe_parent.hash, {"magnitude": 0.001}
    )
    wrong_record = validate_or_inherit(
        child, scene, safe_parent, root.certificate, bad_transform
    )
    assert wrong_record.full_checks_used == 1
    assert wrong_record.reason_code == "transform_parent_hash_mismatch"


def test_commit_compares_every_context_field():
    cases = [
        ("robot", "other", ErrorCode.CONTEXT_CHANGED),
        ("task_phase", "place", ErrorCode.CONTEXT_CHANGED),
        ("committed_prefix_hash", "sha256:" + "1" * 64, ErrorCode.CONTEXT_CHANGED),
    ]
    for field, value, code in cases:
        _, plan, cert, authority, _, executor, _, context, snapshot = _rig()
        lease = authority.prepare(plan, cert, context, snapshot, NOW)
        live = Context(**{**context.__dict__, field: value})
        with pytest.raises(Rejection) as exc:
            executor.commit(lease, plan, snapshot, live, NOW)
        assert exc.value.code == code


def test_commit_rechecks_observation_age_and_rejects_running_overwrite():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig()
    lease = authority.prepare(plan, cert, context, snapshot, NOW)
    stale_now = NOW + MAX_OBS_AGE_NS + 1
    with pytest.raises(Rejection) as exc:
        executor.commit(lease, plan, snapshot, context, stale_now)
    assert exc.value.code == ErrorCode.STATE_STALE

    lease = authority.prepare(plan, cert, context, snapshot, NOW)
    executor.commit(lease, plan, snapshot, context, NOW)
    other_lease = authority.prepare(plan, cert, context, snapshot, NOW)
    with pytest.raises(Rejection) as exc:
        executor.commit(other_lease, plan, snapshot, context, NOW)
    assert exc.value.code == ErrorCode.CONTROLLER_FULL
    assert controller.cursors()["submitted"] == 0


def test_tick_requires_fresh_observation_and_context():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig()
    lease = authority.prepare(plan, cert, context, snapshot, NOW)
    executor.commit(lease, plan, snapshot, context, NOW)
    with pytest.raises(Rejection) as exc:
        executor.tick(NOW + 50_000_000)
    assert exc.value.code == ErrorCode.STATE_STALE
    assert executor.state == ExecutorState.FAULT
    assert controller.cursors()["submitted"] == 0


def test_tick_rejects_reused_observation_and_changed_context():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig()
    lease = authority.prepare(plan, cert, context, snapshot, NOW)
    executor.commit(lease, plan, snapshot, context, NOW)
    fresh = _fresh(controller, 1, NOW + 50_000_000)
    assert executor.tick(NOW + 50_000_000, fresh, context)
    changed = Context(**{**context.__dict__, "queue_rev": 1})
    newer = _fresh(controller, 2, NOW + 100_000_000)
    with pytest.raises(Rejection) as exc:
        executor.tick(NOW + 100_000_000, newer, changed)
    assert exc.value.code == ErrorCode.CONTEXT_CHANGED
    assert executor.state == ExecutorState.FAULT


def test_revoke_invalidates_unused_lease_and_recovery_requires_drain_and_fresh_snapshot():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig()
    unused = authority.prepare(plan, cert, context, snapshot, NOW)
    active = authority.prepare(plan, cert, context, snapshot, NOW)
    executor.commit(active, plan, snapshot, context, NOW)
    executor.revoke("test")
    with pytest.raises(Rejection) as exc:
        authority.consume(unused)
    assert exc.value.code == ErrorCode.STALE_GENERATION

    controller.tick()
    current = context.bumped_epoch()
    with pytest.raises(Rejection) as exc:
        executor.try_recover(True, snapshot=snapshot, live_context=current, now_ns=NOW)
    assert exc.value.code == ErrorCode.STATE_STALE

    fresh = _fresh(controller, 1, NOW + 1)
    executor.try_recover(True, snapshot=fresh, live_context=current, now_ns=NOW + 1)
    assert executor.state == ExecutorState.IDLE


def test_recovery_rejects_confirmed_cancel_until_accepted_prefix_drains():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig()
    controller.exec_delay_ticks = 5
    lease = authority.prepare(plan, cert, context, snapshot, NOW)
    executor.commit(lease, plan, snapshot, context, NOW)
    first = _fresh(controller, 1, NOW + 50_000_000)
    executor.tick(NOW + 50_000_000, first, context)
    executor.revoke("drain-test")
    controller.tick()
    assert controller.cancel_acked is True
    assert not controller.is_drained
    current = context.bumped_epoch()
    fresh = _fresh(controller, 2, NOW + 100_000_000)
    with pytest.raises(Rejection) as exc:
        executor.try_recover(True, fresh, current, NOW + 100_000_000)
    assert exc.value.code == ErrorCode.CANCEL_UNCONFIRMED


def test_open_requires_support_and_is_observed_as_gripper_action():
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig(
        gripper_events=((0, "open"),)
    )
    place_context = Context(**{**context.__dict__, "task_phase": "place"})
    lease = authority.prepare(plan, cert, place_context, snapshot, NOW, prefix_len=1)
    executor.commit(lease, plan, snapshot, place_context, NOW)
    unsupported = _fresh(controller, 1, NOW + 50_000_000, supported=False)
    with pytest.raises(Rejection) as exc:
        executor.tick(NOW + 50_000_000, unsupported, place_context)
    assert exc.value.code == ErrorCode.TRACKING_TUBE
    assert not controller.gripper_submitted

    # New rig because a runtime gate failure intentionally faults the first executor.
    _, plan, cert, authority, controller, executor, _, context, snapshot = _rig(
        gripper_events=((0, "open"),)
    )
    place_context = Context(**{**context.__dict__, "task_phase": "place"})
    lease = authority.prepare(plan, cert, place_context, snapshot, NOW, prefix_len=1)
    executor.commit(lease, plan, snapshot, place_context, NOW)
    supported = _fresh(controller, 1, NOW + 50_000_000, supported=True)
    assert executor.tick(NOW + 50_000_000, supported, place_context)
    controller.tick()
    assert controller.gripper_observed[-1]["event"] == "open"
    assert controller.gripper_state == "open"


def test_prediction_registry_binding_and_expiry():
    _, plan, cert, authority, _, _, _, context, snapshot = _rig()
    prediction = SimpleNamespace(
        plan_hash=plan.hash,
        hash="sha256:" + "a" * 64,
        deadline_mono_ns=NOW + 1_000_000_000,
        allowed=True,
    )
    authority.register_prediction(prediction)
    lease = authority.prepare(
        plan, cert, context, snapshot, NOW, prediction=prediction, require_prediction=True
    )
    assert lease.prediction_hash == prediction.hash

    expired = SimpleNamespace(
        plan_hash=plan.hash,
        hash="sha256:" + "b" * 64,
        deadline_mono_ns=NOW - 1,
        allowed=True,
    )
    authority.register_prediction(expired)
    with pytest.raises(Rejection) as exc:
        authority.prepare(
            plan, cert, context, snapshot, NOW, prediction=expired, require_prediction=True
        )
    assert exc.value.code == ErrorCode.MODEL_UNKNOWN


def test_log_overflow_emits_sticky_gap_and_blocks_motion():
    _, plan, cert, authority, _, _, events, context, snapshot = _rig(maxlen=3)
    for i in range(5):
        events.append("PROPOSAL", i=i)
    assert events.has_gap
    assert events.by_type("LOG_GAP")
    with pytest.raises(Rejection) as exc:
        authority.prepare(plan, cert, context, snapshot, NOW)
    assert exc.value.code == ErrorCode.EVIDENCE_GAP


def test_event_vocabularies_are_versioned():
    legacy = EventLog("legacy")
    with pytest.raises(ValueError):
        legacy.append("PREDICTION")
    product = EventLog("product", schema_version="product-v1")
    product.append("PREDICTION", prediction_hash="sha256:" + "0" * 64)
