"""Controller feedback must be the actual fresh sample bound to a Snapshot."""

import pytest

from sentinel_evc.authority import Authority
from sentinel_evc.contracts import Context, ErrorCode, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore, establish_root
from sentinel_evc.events import EventLog
from sentinel_evc.executor import Executor
from sentinel_evc.scenarios import make_parent_pair, make_scene
from sentinel_evc.sim_controller import SimController


NOW = 1_000_000_000


class FeedbackController(SimController):
    def __init__(self, initial_position, mutate):
        super().__init__(initial_position=initial_position)
        self._mutate = mutate

    def read_feedback(self, now_ns):
        feedback = super().read_feedback(now_ns)
        return self._mutate(feedback)


def _rig(mutate):
    scene = make_scene(0)
    plan, _ = make_parent_pair(0)
    verdict = establish_root(plan, scene)
    store = CertificateStore()
    store.register(verdict.certificate)
    events = EventLog("feedback-binding")
    authority = Authority(store, events=events)
    controller = FeedbackController(plan.knots[0], mutate)
    executor = Executor(authority, controller, events)
    context = Context(scene_id=scene.scene_id)
    snapshot = Snapshot("fresh-looking", plan.knots[0], NOW)
    lease = authority.prepare(plan, verdict.certificate, context, snapshot, NOW)
    return executor, lease, plan, snapshot, context


def test_commit_rejects_replayed_controller_feedback_with_fresh_caller_snapshot():
    def stale_timestamp(feedback):
        feedback["capture_mono_ns"] = NOW - 1
        return feedback

    executor, lease, plan, snapshot, context = _rig(stale_timestamp)
    with pytest.raises(Rejection) as exc:
        executor.commit(lease, plan, snapshot, context, NOW)
    assert exc.value.code == ErrorCode.STATE_STALE


@pytest.mark.parametrize(
    "mutate",
    [
        lambda feedback: {key: value for key, value in feedback.items() if key != "capture_mono_ns"},
        lambda feedback: {**feedback, "capture_mono_ns": True},
        lambda feedback: {**feedback, "capture_mono_ns": -1},
        lambda feedback: {**feedback, "capture_mono_ns": 1 << 63},
        lambda feedback: {**feedback, "position": (0.0, float("nan"), 0.0)},
        lambda feedback: {**feedback, "position": (0.0, 0.0)},
        lambda feedback: {**feedback, "valid": False},
    ],
)
def test_commit_rejects_malformed_or_invalid_controller_feedback(mutate):
    executor, lease, plan, snapshot, context = _rig(mutate)
    with pytest.raises(Rejection) as exc:
        executor.commit(lease, plan, snapshot, context, NOW)
    assert exc.value.code == ErrorCode.STATE_STALE
