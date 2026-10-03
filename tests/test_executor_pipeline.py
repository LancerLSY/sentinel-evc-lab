"""PipelinedExecutor: chained leases keep every EVC invariant while removing stop-and-go."""

import random

import pytest

from sentinel_evc.authority import Authority
from sentinel_evc.contracts import Context, ErrorCode, Plan, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore
from sentinel_evc.delta_cert_v2 import full_v2, slice_certificate_v2
from sentinel_evc.events import EventLog
from sentinel_evc.executor import Executor, ExecutorState
from sentinel_evc.executor_pipeline import PipelinedExecutor, prefix_digest
from sentinel_evc.contracts import Certificate
from sentinel_evc.scenarios import make_scene
from sentinel_evc.sim_controller import SimController

DT = 50_000_000
T0 = 1_000_000_000


def _plan(h=40):
    import math
    return Plan("P", tuple((0.1 + 0.6 * i / h, 0.15 * math.sin(math.pi * i / h), 0.3) for i in range(h + 1)), 0.05)


class Rig:
    """Drives an executor through a long plan with leases of K steps (optionally chained)."""

    def __init__(self, pipelined, K, max_ahead=2, cap=2):
        self.scene = make_scene(0)
        self.plan = _plan()
        self.store = CertificateStore()
        self.auth = Authority(self.store, events=None)
        self.log = EventLog("pipe", maxlen=1_000_000)
        self.ctrl = SimController(capacity=cap, initial_position=self.plan.knots[0])
        self.ex = (PipelinedExecutor(self.auth, self.ctrl, self.log, max_ahead=max_ahead) if pipelined
                   else Executor(self.auth, self.ctrl, self.log))
        self.pipelined, self.K = pipelined, K
        self.root = full_v2(self.plan, self.scene).certificate
        self.t, self.seq, self.offset, self.rev = T0, 0, 0, 0
        self.virtual = [self.plan.knots[0]]
        self.lease_plans = {}

    def snap(self):
        self.seq += 1
        f = self.ctrl.read_feedback(self.t)
        return Snapshot(f"o{self.seq}", f["position"], self.t)

    def env(self):
        return Context(scene_id=self.scene.scene_id, scene_hash=self.scene.hash, epoch=self.ex.generation)

    def try_commit(self):
        if self.offset >= self.plan.horizon:
            return False
        if not self.pipelined and self.ex.state != ExecutorState.IDLE:
            return False
        sub = Plan(f"S{self.offset}", self.plan.knots[self.offset:], 0.05)
        c2 = slice_certificate_v2(self.root, self.plan, sub, self.offset)
        cert = Certificate(c2.cert_id, sub.hash, self.scene.hash, sub.dt, sub.horizon, c2.margins, "INHERITED", depth=1, plan_exact=sub.exact_hash)
        self.store.register(cert)
        k = min(self.K, sub.horizon)
        ctx = Context(scene_id=self.scene.scene_id, scene_hash=self.scene.hash, epoch=self.ex.generation, queue_rev=self.rev,
                      committed_prefix_hash=prefix_digest(self.virtual) if self.ex.state == ExecutorState.RUNNING else Context().committed_prefix_hash)
        s = self.snap()
        lease = self.auth.prepare(sub, cert, ctx, s, self.t, prefix_len=k, ttl_ns=2_000_000_000,
                                  chained=self.pipelined and self.ex.state == ExecutorState.RUNNING)
        try:
            self.ex.commit(lease, sub, s, ctx, self.t)
        except Rejection as e:
            if e.code == ErrorCode.CONTROLLER_FULL:
                return False
            raise
        self.lease_plans[lease.lease_id] = (sub, k)
        if self.ex.state == ExecutorState.RUNNING:
            self.virtual = (self.virtual if self.virtual else [sub.knots[0]]) + list(sub.knots[1:k + 1])
        self.offset += k
        self.rev += 1
        return True

    def step(self):
        self.try_commit()
        self.t += DT
        # live context = the queue state of the most recently committed lease (unchanged queue)
        ctx = self.ex._lease.context if self.ex._lease else self.env()
        self.ex.tick(self.t, self.snap(), ctx)


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_pipelined_executes_same_actions_without_stop_and_go(K):
    ticks = {}
    for pipelined in (False, True):
        r = Rig(pipelined, K)
        n = 0
        while len(r.ctrl.observed) < 40 and n < 400:
            r.step(); n += 1
        assert [o["action"] for o in r.ctrl.observed] == [tuple(k) for k in r.plan.knots[1:41]]
        ticks[pipelined] = n
    assert ticks[True] <= 43            # ~ one dispatch per tick + controller latency
    assert ticks[True] < ticks[False]


def test_revoke_drops_every_chained_lease_and_blocks_until_recovery():
    rng = random.Random(3)
    for trial in range(40):
        r = Rig(True, rng.choice([1, 2, 4]), max_ahead=rng.choice([1, 2, 3]))
        for _ in range(rng.randint(3, 25)):
            r.step()
        before = len(r.ctrl.submitted)
        r.ex.revoke("test", now_ns=r.t)
        assert r.ex.state == ExecutorState.FAULT
        for _ in range(10):
            r.t += DT
            r.ex.tick(r.t, r.snap(), r.env())
            with pytest.raises(Rejection):
                r.try_commit()
        assert len(r.ctrl.submitted) == before          # nothing new after the linearization point
        # every submitted action was dispatched under a consumed lease, at most K steps per lease
        disp = [e["payload"] for e in r.log.events() if e["type"] == "DISPATCH"]
        assert len(disp) == before
        per_lease = {}
        for d in disp:
            per_lease[d["lease_id"]] = per_lease.get(d["lease_id"], 0) + 1
        for lid, n in per_lease.items():
            assert lid in r.lease_plans and n <= r.lease_plans[lid][1]
            assert r.auth.is_consumed(lid)


def test_chained_lease_must_start_at_committed_end_and_bind_prefix():
    r = Rig(True, 4)
    r.step()  # first lease committed and running
    assert r.ex.state == ExecutorState.RUNNING
    # a lease whose plan does not start at the committed end knot
    bad = Plan("B", r.plan.knots[2:], 0.05)
    c2 = slice_certificate_v2(r.root, r.plan, bad, 2)
    cert = Certificate(c2.cert_id, bad.hash, r.scene.hash, bad.dt, bad.horizon, c2.margins, "INHERITED", depth=1, plan_exact=bad.exact_hash)
    r.store.register(cert)
    ctx = Context(scene_id=r.scene.scene_id, epoch=r.ex.generation, queue_rev=r.rev, committed_prefix_hash=prefix_digest(r.virtual))
    s = r.snap()
    lease = r.auth.prepare(bad, cert, ctx, s, r.t, prefix_len=2, ttl_ns=2_000_000_000, chained=True)
    with pytest.raises(Rejection) as e:
        r.ex.commit(lease, bad, s, ctx, r.t)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED
    # correct start but wrong committed-prefix digest
    good = Plan("G", r.plan.knots[4:], 0.05)
    c2 = slice_certificate_v2(r.root, r.plan, good, 4)
    cert = Certificate(c2.cert_id, good.hash, r.scene.hash, good.dt, good.horizon, c2.margins, "INHERITED", depth=1, plan_exact=good.exact_hash)
    r.store.register(cert)
    ctx = Context(scene_id=r.scene.scene_id, epoch=r.ex.generation, queue_rev=r.rev)
    s = r.snap()
    lease = r.auth.prepare(good, cert, ctx, s, r.t, prefix_len=2, ttl_ns=2_000_000_000, chained=True)
    with pytest.raises(Rejection) as e:
        r.ex.commit(lease, good, s, ctx, r.t)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED


def test_commit_rejects_live_queue_state_that_differs_from_the_lease():
    r = Rig(True, 4)
    sub = Plan("Q", r.plan.knots, 0.05)
    c2 = slice_certificate_v2(r.root, r.plan, sub, 0)
    cert = Certificate(c2.cert_id, sub.hash, r.scene.hash, sub.dt, sub.horizon, c2.margins, "INHERITED", depth=1, plan_exact=sub.exact_hash)
    r.store.register(cert)
    ctx = Context(scene_id=r.scene.scene_id, epoch=r.ex.generation)
    s = r.snap()
    lease = r.auth.prepare(sub, cert, ctx, s, r.t, prefix_len=4, ttl_ns=2_000_000_000)
    with pytest.raises(Rejection) as e:
        r.ex.commit(lease, sub, s, Context(scene_id=r.scene.scene_id, epoch=r.ex.generation, queue_rev=7), r.t)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED
    assert r.ctrl.submitted == []


def test_dispatch_rejects_a_queue_changed_by_someone_else():
    r = Rig(True, 4)
    r.step()
    assert r.ex.state == ExecutorState.RUNNING and len(r.ctrl.submitted) == 1
    latest = r.ex._lease.context
    foreign = Context(scene_id=latest.scene_id, epoch=latest.epoch, queue_rev=7,
                      committed_prefix_hash=prefix_digest([(9.0, 9.0, 9.0)]))
    r.t += DT
    with pytest.raises(Rejection) as e:
        r.ex.tick(r.t, r.snap(), foreign)
    assert e.value.code == ErrorCode.CONTEXT_CHANGED
    assert len(r.ctrl.submitted) == 1


def test_long_continuous_run_keeps_pipeline_state_bounded():
    import math as _m
    h = 600
    # back-and-forth sweep through the free region of scene 0
    f = lambda i: (0.1 + 0.6 * (1 - abs(1 - (i % 400) / 200)), 0.15 * abs(_m.sin(_m.pi * i / 200)), 0.3)
    plan = Plan("L", tuple(f(i) for i in range(h + 1)), 0.05)
    r = Rig(True, 1)
    r.plan = plan
    r.root = full_v2(plan, r.scene).certificate
    r.virtual = [plan.knots[0]]
    peak_virtual = peak_dt = 0
    n = 0
    digest_checked = False
    while len(r.ctrl.observed) < h and n < 3 * h:
        r.step(); n += 1
        peak_virtual = max(peak_virtual, len(r.ex._virtual))
        peak_dt = max(peak_dt, len(r.ex._lease_dt))
        if n == h // 2:
            assert r.ex.state == ExecutorState.RUNNING and r.ex._trimmed > 0
            # rolling digest over the trimmed state == digest of the whole committed history
            assert r.ex._committed_prefix() == prefix_digest(r.virtual)
            digest_checked = True
    assert digest_checked
    assert [o["action"] for o in r.ctrl.observed] == [tuple(k) for k in plan.knots[1:h + 1]]
    assert n <= h + 3
    assert peak_virtual <= 64 + 8 and peak_dt <= 3
