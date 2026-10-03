"""v2 protocol hardening: byte-exact binding, scene binding, completion budget, revocation time,
idle-revoke context, atomic consumption, and the first unit tests of the native gateway."""

import math
import threading

import pytest

from sentinel_evc.authority import Authority
from sentinel_evc.contracts import Context, ErrorCode, Plan, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore, establish_root
from sentinel_evc.events import EventLog
from sentinel_evc.executor import Executor, ExecutorState
from sentinel_evc.scenarios import make_parent_pair, make_scene
from sentinel_evc.sim_controller import SimController

T0 = 1_000_000_000
DT = 50_000_000


def _rig(scene_hash=False, ttl=500_000_000):
    scene = make_scene(0)
    plan, _ = make_parent_pair(0)
    store = CertificateStore()
    cert = establish_root(plan, scene).certificate
    store.register(cert)
    log = EventLog("hardening")
    auth = Authority(store, events=log)
    ctrl = SimController(initial_position=plan.knots[0])
    ex = Executor(auth, ctrl, log)
    ctx = Context(scene_id=scene.scene_id, scene_hash=scene.hash if scene_hash else None)
    snap = Snapshot("obs-0", plan.knots[0], T0)
    return dict(scene=scene, plan=plan, cert=cert, auth=auth, ctrl=ctrl, ex=ex, ctx=ctx, snap=snap, log=log, ttl=ttl)


def _feedback(r, t, tag):
    f = r["ctrl"].read_feedback(t)
    return Snapshot(f"obs-{tag}", f["position"], t)


def test_one_ulp_change_after_prepare_is_rejected_at_commit():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"], T0)
    p = r["plan"]
    q = Plan(p.plan_id, (p.knots[0],) + tuple((math.nextafter(x, 1.0), y, z) for x, y, z in p.knots[1:]), p.dt)
    assert q.hash == p.hash and q.exact_hash != p.exact_hash
    with pytest.raises(Rejection) as e:
        r["ex"].commit(lease, q, r["snap"], r["ctx"], T0)
    assert e.value.code == ErrorCode.CERTIFICATE_MISS
    assert r["ctrl"].cursors()["submitted"] == 0


def test_certificate_from_another_scene_is_not_authorised():
    r = _rig(scene_hash=True)
    other = make_scene(7)
    ctx = Context(scene_id=r["scene"].scene_id, scene_hash=other.hash)   # same id, different content
    with pytest.raises(Rejection) as e:
        r["auth"].prepare(r["plan"], r["cert"], ctx, r["snap"], T0)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED


def test_remaining_steps_must_finish_before_deadline():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"], T0, prefix_len=4, ttl_ns=4 * DT)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], T0)
    t = T0 + 10_000_000            # 10 ms late: 4 remaining steps can no longer finish by T0 + 200 ms
    with pytest.raises(Rejection) as e:
        r["ex"].tick(t, _feedback(r, t, "late"), r["ctx"])
    assert e.value.code == ErrorCode.LEASE_EXPIRED
    assert r["ctrl"].cursors()["submitted"] == 0


class _StalledFeedback(SimController):
    """Controller whose feedback stream stops updating after ``stall()`` (e.g. a stuck driver thread)."""

    stalled_at = None

    def stall(self, at_ns):
        self.stalled_at = at_ns

    def read_feedback(self, now_ns):
        f = super().read_feedback(now_ns)
        if self.stalled_at is not None:
            f = dict(f, capture_mono_ns=self.stalled_at)
        return f


def test_recovery_feedback_must_postdate_the_revocation_instant():
    r = _rig()
    ctrl = _StalledFeedback(initial_position=r["plan"].knots[0])
    ex = Executor(r["auth"], ctrl, r["log"])
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"], T0, prefix_len=1)
    ex.commit(lease, r["plan"], r["snap"], r["ctx"], T0)
    t = T0
    for i in range(6):                       # finish the one-step prefix; the arm is at rest
        t += DT
        f = ctrl.read_feedback(t)
        ex.tick(t, Snapshot(f"tick-{i}", f["position"], t), r["ctx"])
    assert ex.state == ExecutorState.IDLE
    ctrl.stall(t + 1)                        # last real sample is taken before the revocation
    stale = Snapshot("captured-before-revoke", ctrl.read_feedback(t + 1)["position"], t + 1)
    ex.revoke("operator", now_ns=t + 5)
    for _ in range(4):
        ctrl.tick()
    ex.poll_cancel()
    ctx = Context(scene_id=r["scene"].scene_id, epoch=ex.generation)
    # the stalled sample is younger than MAX_OBS_AGE, newer than the last dispatch and is what the
    # controller reports, so only the revocation-instant rule can reject it
    with pytest.raises(Rejection) as e:
        ex.try_recover(True, stale, ctx, t + 20)
    assert e.value.code == ErrorCode.STATE_STALE
    assert "撤销后" in e.value.detail


def test_idle_revoke_still_checks_stable_context_fields():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"], T0, prefix_len=1)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], T0)
    t = T0
    for i in range(6):
        t += DT
        r["ex"].tick(t, _feedback(r, t, f"idle-{i}"), r["ctx"])
    assert r["ex"].state == ExecutorState.IDLE
    r["ex"].revoke("idle-stop", now_ns=t)
    for _ in range(3):
        r["ctrl"].tick()
    r["ex"].poll_cancel()
    t += DT
    moved = Context(scene_id="another-scene", epoch=r["ex"].generation)
    with pytest.raises(Rejection) as e:
        r["ex"].try_recover(True, _feedback(r, t, "rec"), moved, t)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED


def test_consumption_is_atomic_across_threads():
    import time
    r = _rig()
    original = r["auth"].validate_available

    def slow_validate(lease):
        original(lease)
        time.sleep(0.01)                      # widen the check-to-record window
    r["auth"].validate_available = slow_validate
    for _ in range(5):
        lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"], T0)
        wins = []
        barrier = threading.Barrier(8)

        def go():
            barrier.wait()
            try:
                r["auth"].consume(lease)
                wins.append(1)
            except Rejection:
                pass
        ts = [threading.Thread(target=go) for _ in range(8)]
        [x.start() for x in ts]
        [x.join() for x in ts]
        assert len(wins) == 1


# --------------------------------------------------------------- native gateway

def _native():
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed")
    from sentinel_evc.native_gateway import (NativeActionProfile, NativeContext, NativeGateway,
                                             NativeGatewayDenied, NativeSnapshot)
    prof = NativeActionProfile("test-7d", tuple(f"a{i}" for i in range(7)), (-1e30,) * 7, (1e30,) * 7, "test")
    log = EventLog("native", schema_version="product-v1", maxlen=64)
    gw = NativeGateway(prof, log, lease_ttl_ns=250_000_000, max_feedback_age_ns=500_000_000)
    snap = NativeSnapshot("ep:reset", "sha256:" + "1" * 64, 0, T0)
    ctx = NativeContext("r", "b", "ep", "s", "c", "run", 0)
    gw.observe_feedback(snap, ctx)
    ctx2 = NativeContext("r", "b", "ep", "s", "c", "run", 0, chunk_id=0, action_index_in_chunk=0,
                         policy_input_hash="sha256:" + "2" * 64, policy_input_step=0)
    gw.refresh_context(ctx2, snapshot_hash=snap.digest)
    action = np.arange(7, dtype=np.float32).reshape(1, 7) / 10
    calls = []

    def writer(a, entered):
        entered()
        calls.append(a.copy())
        return "stepped"
    return dict(np=np, gw=gw, log=log, snap=snap, ctx=ctx2, action=action, writer=writer, calls=calls,
                Denied=NativeGatewayDenied)


def test_native_happy_path_then_replay_is_denied_without_writer_call():
    n = _native()
    permit = n["gw"].authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 1)
    assert n["gw"].submit(permit, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 2) == "stepped"
    with pytest.raises(n["Denied"]) as e:
        n["gw"].submit(permit, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 3)
    assert e.value.code == "LEASE_REPLAY" and len(n["calls"]) == 1


def test_native_one_ulp_action_change_is_denied():
    n = _native()
    permit = n["gw"].authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 1)
    changed = n["action"].copy()
    changed[0, 3] = n["np"].nextafter(changed[0, 3], n["np"].float32(1))
    with pytest.raises(n["Denied"]) as e:
        n["gw"].submit(permit, changed, snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 2)
    assert e.value.code == "ACTION_REPLACED" and not n["calls"]


def test_native_revoked_forged_and_reused_feedback_are_denied():
    n = _native()
    gw = n["gw"]
    p1 = gw.authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 1)
    gw.revoke("test", now_ns=T0 + 2)
    with pytest.raises(n["Denied"]) as e:
        gw.submit(p1, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 3)
    assert e.value.code == "GENERATION_REVOKED"
    p2 = gw.authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 4)
    from dataclasses import replace
    forged = replace(p2, authenticator="hmac-sha256:" + "0" * 64)
    with pytest.raises(n["Denied"]) as e:
        gw.submit(forged, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 5)
    assert e.value.code == "INVALID_PERMIT"
    p3 = gw.authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 6)
    gw.submit(p2, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 7)
    with pytest.raises(n["Denied"]) as e:
        gw.submit(p3, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 8)
    assert e.value.code == "FEEDBACK_REUSED" and len(n["calls"]) == 1


def test_native_stale_feedback_and_evidence_gap_block_the_writer():
    n = _native()
    gw = n["gw"]
    p = gw.authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 1)
    with pytest.raises(n["Denied"]) as e:
        gw.submit(p, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 240_000_000 + 600_000_000)
    assert e.value.code in ("LEASE_EXPIRED", "STALE_FEEDBACK")
    p = gw.authorize(n["action"], snapshot=n["snap"], context=n["ctx"], now_ns=T0 + 2)
    n["log"].note_gap("simulated storage failure")
    with pytest.raises(n["Denied"]) as e:
        gw.submit(p, n["action"], snapshot=n["snap"], context=n["ctx"], writer=n["writer"], now_ns=T0 + 3)
    assert e.value.code == "EVIDENCE_GAP" and not n["calls"]


def test_certificate_copy_with_rewritten_scene_stamp_is_refused():
    """The scene binding must read the registered certificate, not the caller's copy."""
    import dataclasses
    from sentinel_evc.contracts import Scene, Sphere
    from sentinel_evc.geometry import full_check
    r = _rig()
    a, p = r["scene"], r["plan"]
    b = Scene(a.scene_id, a.obstacles + (Sphere(p.knots[2], 0.03),), a.ws_lo, a.ws_hi, a.tool_radius, a.tracking_reserve)
    assert full_check(p, a)[0] and not full_check(p, b)[0]
    ctx_b = Context(scene_id=b.scene_id, scene_hash=b.hash)
    with pytest.raises(Rejection) as e:
        r["auth"].prepare(p, r["cert"], ctx_b, r["snap"], T0)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED
    forged = dataclasses.replace(r["cert"], scene_hash=b.hash)
    with pytest.raises(Rejection) as e:
        r["auth"].prepare(p, forged, ctx_b, r["snap"], T0)
    assert e.value.code == ErrorCode.CERTIFICATE_MISS
    assert r["ctrl"].submitted == []
