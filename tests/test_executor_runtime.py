from dataclasses import replace
from threading import Barrier, Event, Thread, current_thread
from time import monotonic_ns

import pytest

from sentinel_evc.authority import Authority, MAX_OBS_AGE_NS
from sentinel_evc.contracts import ErrorCode, LeaseContext, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore, establish_root
from sentinel_evc.events import EventLog, ZERO_HASH
from sentinel_evc.executor import Executor, ExecutorState
from sentinel_evc.scenarios import make_parent_pair, make_scene
from sentinel_evc.sim_controller import SimController


def rig(controller=None, event_clock=monotonic_ns):
    scene = make_scene(0)
    plan, _ = make_parent_pair(0)
    cert = establish_root(plan, scene).certificate
    store = CertificateStore()
    store.register(cert)
    auth = Authority(store)
    context = LeaseContext('robot', 1, 0, scene.scene_id, 0, ZERO_HASH)
    now = [1_000_000_000]
    snap = Snapshot('obs', **context.summary(), position=plan.points[0], observed_mono=now[0])
    log = EventLog('executor', monotonic_ns=event_clock)
    ctrl = controller or SimController()
    ex = Executor(auth, ctrl, log, monotonic_ns=lambda: now[0])
    lease = auth.prepare(plan, cert, context, snap, now[0])
    return ex, auth, ctrl, log, plan, lease, context, snap, now


@pytest.mark.parametrize('field,value', [
    ('robot', 'other'), ('boot', 2), ('epoch', 1), ('scene_id', 'other'),
    ('queue_rev', 1), ('committed_prefix_hash', 'sha256:' + '1' * 64),
])
def test_commit_rechecks_snapshot_context_without_consuming(field, value):
    ex, auth, _, _, plan, lease, ctx, snap, now = rig()
    with pytest.raises(Rejection) as error:
        ex.commit(lease, plan, replace(snap, **{field: value}), ctx, now[0])
    assert error.value.code == ErrorCode.CONTEXT_CHANGED
    assert not auth.is_consumed(lease.lease_id)


@pytest.mark.parametrize('age', [-1, MAX_OBS_AGE_NS + 1])
def test_commit_rechecks_snapshot_age_without_consuming(age):
    ex, auth, _, _, plan, lease, ctx, snap, now = rig()
    with pytest.raises(Rejection) as error:
        ex.commit(lease, plan, replace(snap, observed_mono=now[0] - age), ctx, now[0])
    assert error.value.code == ErrorCode.STATE_STALE
    assert not auth.is_consumed(lease.lease_id)


@pytest.mark.parametrize('short_by', [0, 1])
def test_commit_remaining_budget_boundary(short_by):
    ex, auth, _, _, plan, lease, ctx, snap, now = rig()
    now[0] = lease.deadline_mono - int(lease.prefix_len * plan.dt * 1e9) + short_by
    snap = replace(snap, observed_mono=now[0])
    if short_by:
        with pytest.raises(Rejection) as error:
            ex.commit(lease, plan, snap, ctx, now[0])
        assert error.value.code == ErrorCode.LEASE_EXPIRED
        assert not auth.is_consumed(lease.lease_id)
    else:
        ex.commit(lease, plan, snap, ctx, now[0])
        assert auth.is_consumed(lease.lease_id)


def test_busy_commit_preserves_pending_prefix_and_does_not_consume():
    ex, auth, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    cert = auth._store.get(lease.cert_id)
    other = auth.prepare(plan, cert, ctx, snap, now[0])
    with pytest.raises(Rejection) as error:
        ex.commit(other, plan, snap, ctx, now[0])
    assert error.value.code == ErrorCode.QUEUE_FULL
    assert not auth.is_consumed(other.lease_id)
    assert ex.tick(now[0], ctx)
    assert ctrl.submitted[0]['action'] == plan.points[1]


def test_each_step_rechecks_context_and_stops_old_prefix():
    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    assert ex.tick(now[0], ctx)
    now[0] += 50_000_000
    with pytest.raises(Rejection) as error:
        ex.tick(now[0], replace(ctx, queue_rev=1))
    assert error.value.code == ErrorCode.CONTEXT_CHANGED
    assert len(ctrl.submitted) == 1


def test_step_clock_cannot_burst_missed_slots():
    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    assert ex.tick(now[0], ctx)
    now[0] += 150_000_000
    assert ex.tick(now[0], ctx)
    assert not ex.tick(now[0], ctx)
    assert len(ctrl.submitted) == 2


def test_ack_and_observation_events_preserve_three_cursors_after_revoke():
    ex, _, ctrl, log, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    ex.tick(now[0], ctx)
    assert ctrl.cursors() == {'submitted': 1, 'accepted': 1, 'observed': 0}
    ex.revoke('test')
    ex.tick(now[0] + 50_000_000, ctx)
    assert ctrl.cursors() == {'submitted': 1, 'accepted': 1, 'observed': 1}
    assert len(log.by_type('CONTROLLER_ACK')) == 1
    observed = log.by_type('OBSERVED')
    assert len(observed) == 1
    assert observed[0]['lease_id'] == lease.lease_id
    assert observed[0]['payload']['step_index'] == 0


@pytest.mark.parametrize('failure', ['no_ack', 'no_human', 'old_epoch', 'old_snapshot', 'future_snapshot'])
def test_recover_requires_cancel_fresh_snapshot_and_human(failure):
    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    ex.revoke('test')
    if failure != 'no_ack':
        ctrl.advance()
    now[0] += 50_000_000
    fresh = replace(snap, obs_id='recovery', epoch=1, observed_mono=now[0])
    if failure == 'old_epoch':
        fresh = replace(fresh, epoch=0)
    if failure == 'old_snapshot':
        fresh = replace(fresh, observed_mono=now[0] - MAX_OBS_AGE_NS - 1)
    if failure == 'future_snapshot':
        fresh = replace(fresh, observed_mono=now[0] + 1)
    with pytest.raises(Rejection):
        ex.recover(fresh, human_approved=failure != 'no_human')
    assert ex.state == ExecutorState.FAULT


def test_recover_accepts_fresh_post_cancel_snapshot():
    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    ex.revoke('test')
    ctrl.advance()
    now[0] += 50_000_000
    ex.recover(replace(snap, obs_id='new', epoch=1, observed_mono=now[0]), True)
    assert ex.state == ExecutorState.IDLE


def test_revoke_holds_no_lock_and_cannot_recover_while_cancel_waits():
    entered, release = Barrier(2), Barrier(2)

    class WaitingCancel(SimController):
        def cancel(self):
            entered.wait(timeout=5)
            release.wait(timeout=5)
            super().cancel()

    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig(WaitingCancel())
    ex.commit(lease, plan, snap, ctx, now[0])
    ctrl.cancel_acked = True
    thread = Thread(target=ex.revoke, args=('race',))
    thread.start()
    entered.wait(timeout=5)
    try:
        assert ex._lock.acquire(blocking=False)
        ex._lock.release()
        assert ex._cancel_lock.acquire(blocking=False)
        ex._cancel_lock.release()
        assert ex.poll_cancel() is None
        with pytest.raises(Rejection) as error:
            ex.recover(replace(snap, epoch=1), True)
        assert error.value.code == ErrorCode.CANCEL_UNCONFIRMED
        assert not ex.tick(now[0], ctx)
        assert ctrl.submitted == []
    finally:
        release.wait(timeout=5)
        thread.join(timeout=5)
    assert not thread.is_alive()

@pytest.mark.parametrize('tamper', ['hmac', 'final_hash'])
def test_commit_authentication_failure_does_not_consume(tamper):
    ex, auth, _, _, plan, lease, ctx, snap, now = rig()
    if tamper == 'hmac':
        lease = replace(lease, hmac='0' * 64)
    else:
        plan = replace(plan, task_phase='other')
    with pytest.raises(Rejection) as error:
        ex.commit(lease, plan, snap, ctx, now[0])
    assert error.value.code == ErrorCode.AUTH_FAILED
    assert not auth.is_consumed(lease.lease_id)


def test_failed_dispatch_cannot_replay_consumed_lease():
    class RejectingController(SimController):
        def submit(self, action, generation):
            return False

    ex, auth, _, log, plan, lease, ctx, snap, now = rig(RejectingController())
    ex.commit(lease, plan, snap, ctx, now[0])
    assert not ex.tick(now[0], ctx)
    assert auth.is_consumed(lease.lease_id)
    with pytest.raises(Rejection) as error:
        ex.commit(lease, plan, snap, ctx, now[0])
    assert error.value.code == ErrorCode.LEASE_REPLAY
    assert log.by_type('DISPATCH')[0]['payload']['submitted'] is False


def test_submit_and_revoke_share_linearization_boundary():
    entered, release = Barrier(2), Barrier(2)
    errors = []

    class WaitingSubmit(SimController):
        def submit(self, action, generation):
            entered.wait(timeout=5)
            release.wait(timeout=5)
            return super().submit(action, generation)

    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig(WaitingSubmit())
    ex.commit(lease, plan, snap, ctx, now[0])

    def dispatch():
        try:
            ex.tick(now[0], ctx)
        except Exception as error:
            errors.append(error)

    sender = Thread(target=dispatch)
    sender.start()
    entered.wait(timeout=5)
    revoker = Thread(target=ex.revoke, args=('race',))
    revoker.start()
    release.wait(timeout=5)
    sender.join(timeout=5)
    revoker.join(timeout=5)
    assert not sender.is_alive() and not revoker.is_alive()
    assert errors == []
    assert ex.generation == 1
    assert len(ctrl.submitted) == 1
    assert not ex.tick(now[0] + 50_000_000, ctx)
    assert len(ctrl.submitted) == 1
    assert len(ctrl.observed) == 1


def test_executor_is_only_production_submit_caller():
    import ast
    from pathlib import Path

    callers = []
    for path in (Path(__file__).parents[1] / 'src' / 'sentinel_evc').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'submit'):
                callers.append(path.name)
    assert callers == ['executor.py']


def test_recovery_does_not_reuse_a_recent_pre_revoke_observation():
    ex, _, ctrl, _, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    now[0] += 1
    ex.revoke('test')
    ctrl.advance()
    with pytest.raises(Rejection) as error:
        ex.recover(replace(snap, epoch=1), True)
    assert error.value.code == ErrorCode.STATE_STALE


def test_repeated_cancel_poll_records_confirmation_once():
    ex, _, ctrl, log, plan, lease, ctx, snap, now = rig()
    ex.commit(lease, plan, snap, ctx, now[0])
    ex.revoke('test')
    ctrl.advance()
    assert ex.poll_cancel() is True
    assert ex.poll_cancel() is True
    assert len(log.by_type('CANCEL_ACK')) == 1


def test_concurrent_revoke_and_observed_keep_unique_sequence_and_chain():
    from sentinel_evc.contracts import sha256_hex

    revoke_entered, release_revoke, observation_finished = Event(), Event(), Event()
    errors = []

    def event_clock():
        if current_thread().name == 'revoke-writer':
            revoke_entered.set()
            assert release_revoke.wait(timeout=5)
        return 1

    ex, _, ctrl, log, plan, lease, ctx, snap, now = rig(event_clock=event_clock)
    ex.commit(lease, plan, snap, ctx, now[0])
    ex.tick(now[0], ctx)

    def revoke():
        try:
            ex.revoke('concurrent')
        except Exception as error:
            errors.append(error)

    def observe():
        try:
            ex.tick(now[0] + 50_000_000, ctx)
        except Exception as error:
            errors.append(error)
        finally:
            observation_finished.set()

    revoker = Thread(target=revoke, name='revoke-writer')
    observer = Thread(target=observe)
    revoker.start()
    assert revoke_entered.wait(timeout=5)
    observer.start()
    try:
        # 无锁版本在此完成 OBSERVED，从而与暂停的 REVOKE 使用同一 seq。
        # 有锁版本等待 REVOKE 完成；定时等待只释放测试屏障，不推进模拟时钟。
        observation_finished.wait(timeout=0.2)
    finally:
        release_revoke.set()
        revoker.join(timeout=5)
        observer.join(timeout=5)
    assert not revoker.is_alive() and not observer.is_alive()
    assert errors == []
    assert ctrl.cursors() == {'submitted': 1, 'accepted': 1, 'observed': 1}
    records = log.events()
    assert {'REVOKE', 'OBSERVED'} <= {event['type'] for event in records}
    assert [event['seq'] for event in records] == list(range(len(records)))
    previous = ZERO_HASH
    for event in records:
        assert event['prev_hash'] == previous
        previous = sha256_hex(event)
    assert log.tip_hash == previous
