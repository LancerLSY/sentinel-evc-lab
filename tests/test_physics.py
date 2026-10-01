"""Contract tests for the optional, genuine MuJoCo physics fixture.

The module itself remains importable without MuJoCo.  Tests that instantiate
the engine are explicitly skipped when the optional dependency is unavailable,
including under the repository's minimal ``run_tests.py`` shim.
"""

import importlib.util
import math
import time

import pytest

from sentinel_evc.contracts import DEFAULT_DESCRIPTOR
from sentinel_evc.physics import (
    PHYSICS_DESCRIPTOR,
    PHYSICS_SCOPE,
    MujocoController,
    PhysicsConfig,
    physics_plan,
    physics_scene,
    verify_physics_plan,
)
from sentinel_evc.physics_experiment import _new_root, run_branch


def _has_mujoco():
    return importlib.util.find_spec("mujoco") is not None


def _require_mujoco():
    if not _has_mujoco():
        pytest.skip("MuJoCo optional dependency is not installed")


def _settled(config=None):
    _require_mujoco()
    return _new_root(config or PhysicsConfig())


def test_physics_config_is_strict_about_types_finiteness_and_supported_steps():
    assert PhysicsConfig().timestep == 0.002
    for supported in (0.000125, 0.00025, 0.0005, 0.001, 0.002):
        assert PhysicsConfig(timestep=supported).timestep == supported
    for bad in (True, False, float("nan"), float("inf"), 0.0020000000000001, 0.003):
        with pytest.raises(ValueError):
            PhysicsConfig(timestep=bad)
    for bad in (True, float("nan"), float("inf"), 0.0, 1.01):
        with pytest.raises(ValueError):
            PhysicsConfig(friction=bad)
    for bad in (True, float("nan"), float("inf"), 0.0, 0.21):
        with pytest.raises(ValueError):
            PhysicsConfig(payload_mass=bad)
    for bad in (True, -1, 1_000_000, 1.5):
        with pytest.raises(ValueError):
            PhysicsConfig(seed=bad)


def test_physics_plan_and_scene_expose_the_explicit_geometry_only_profile():
    plan = physics_plan(0.6)
    scene = physics_scene()
    assert plan.descriptor == PHYSICS_DESCRIPTOR
    assert plan.descriptor != DEFAULT_DESCRIPTOR
    assert plan.descriptor.gripper == "absent"
    assert plan.gripper_events == ()
    assert plan.dt == 0.05
    assert plan.horizon == 40
    assert len(plan.knots) == 41
    assert scene.scene_id == "mujoco-tray-scene-v1"
    assert scene.obstacles == ()
    assert scene.tool_radius == 0.17

    controller, _ = _settled()
    start = controller.read_feedback(controller.now_ns)["position"]
    bound = physics_plan(0.6, start)
    certificate = verify_physics_plan(bound, controller, "physics-profile")
    assert certificate.method == "FULL"
    assert certificate.proof_scope == PHYSICS_SCOPE
    assert certificate.plan_hash == bound.hash
    assert certificate.scene_hash == controller.scene_digest()


def test_feedback_reports_actual_qpos_and_interval_completion_not_targets():
    _require_mujoco()
    controller = MujocoController(PhysicsConfig())
    initial = controller.read_feedback(controller.now_ns)
    target = (0.20, 0.05, 0.53)
    assert controller.submit(target, generation=0)

    controller.tick()
    first = controller.read_feedback(controller.now_ns)
    actual_from_engine = tuple(float(x) for x in controller.data.xpos[controller.model.body("tray").id])
    assert first["position"] == actual_from_engine
    assert first["position"] != target
    assert first["tracking_error"] > 0
    assert first["cursors"] == {"submitted": 1, "accepted": 1, "observed": 0}
    assert controller.now_ns - initial["capture_mono_ns"] == 2_000_000

    ticks_per_interval = round(0.05 / controller.poll_dt)
    for _ in range(ticks_per_interval - 2):
        controller.tick()
    assert controller.cursors()["observed"] == 0
    controller.tick()
    final = controller.read_feedback(controller.now_ns)
    assert final["cursors"]["observed"] == 1
    assert controller.observed[0]["action"] == target
    assert final["position"] == tuple(float(x) for x in controller.data.xpos[controller.model.body("tray").id])
    assert all(math.isfinite(value) for value in final["qpos"] + final["qvel"])


def test_feedback_timestamp_must_be_the_current_simulator_clock():
    _require_mujoco()
    controller = MujocoController(PhysicsConfig())
    now = controller.now_ns
    assert controller.read_feedback(now)["capture_mono_ns"] == now
    for wrong in (now - 1, now + 1, True):
        with pytest.raises(ValueError, match="local clock"):
            controller.read_feedback(wrong)


def test_public_tick_keeps_two_millisecond_clock_cadence_at_all_refinements():
    _require_mujoco()
    for timestep in (0.002, 0.001, 0.0005, 0.00025, 0.000125):
        controller = MujocoController(PhysicsConfig(timestep=timestep))
        before_ns = controller.now_ns
        before_physics = controller.data.time
        controller.tick()
        assert controller.now_ns - before_ns == 2_000_000
        assert math.isclose(controller.data.time - before_physics, 0.002, abs_tol=1e-12)


def test_complete_integration_root_restore_replays_identically():
    controller, root = _settled()
    target = (0.16, 0.04, 0.50)
    trajectories = []
    for _ in range(2):
        replay = MujocoController(PhysicsConfig())
        replay.restore_root(root)
        assert replay.submit(target, generation=0)
        rows = []
        for _ in range(round(0.10 / replay.poll_dt)):
            replay.tick()
            feedback = replay.read_feedback(replay.now_ns)
            rows.append((tuple(feedback["qpos"]), tuple(feedback["qvel"]), feedback["contact_count"]))
        trajectories.append(rows)
    assert trajectories[0] == trajectories[1]

    altered = dict(root)
    altered["scene_digest"] = "sha256:" + "0" * 64
    fresh = MujocoController(PhysicsConfig())
    with pytest.raises(ValueError, match="different physics model"):
        fresh.restore_root(altered)


def test_scene_digest_covers_joint_axes_options_and_actuator_dynamics():
    _require_mujoco()

    joint_controller = MujocoController(PhysicsConfig())
    original = joint_controller.scene_digest()
    joint_id = joint_controller.model.joint("slide_x").id
    joint_controller.model.jnt_axis[joint_id, 0] += 0.01
    assert joint_controller.scene_digest() != original

    option_controller = MujocoController(PhysicsConfig())
    original = option_controller.scene_digest()
    option_controller.model.opt.disableflags ^= int(
        option_controller.engine.mjtDisableBit.mjDSBL_GRAVITY
    )
    assert option_controller.scene_digest() != original

    actuator_controller = MujocoController(PhysicsConfig())
    original = actuator_controller.scene_digest()
    actuator_id = actuator_controller.model.actuator("x_servo").id
    actuator_controller.model.actuator_dynprm[actuator_id, 0] += 0.125
    assert actuator_controller.scene_digest() != original


def test_live_geometry_and_friction_mutation_refuse_before_any_submit():
    controller, root = _settled()
    start = controller.read_feedback(controller.now_ns)["position"]
    plan = physics_plan(0.6, start)
    for mutation in ("geometry", "friction"):
        result, mutated, _ = run_branch(
            PhysicsConfig(), root, plan, "mutation-" + mutation, mutate=mutation
        )
        assert result["protocol_error"] == "CONTEXT_CHANGED"
        assert result["cursors"]["submitted"] == 0
        assert mutated.submitted == []


def test_revoke_stops_new_generation_zero_submits_but_physics_tail_continues():
    controller, root = _settled()
    plan = physics_plan(1.6, controller.read_feedback(controller.now_ns)["position"])
    result, _, _ = run_branch(PhysicsConfig(), root, plan, "revoke-tail", revoke_at=3)
    revoke = result["revoke"]
    assert revoke["new_old_generation_submissions"] == 0
    assert revoke["accepted_tail_observed"] >= 1
    assert revoke["physics_elapsed_after_request_s"] >= 0.5
    assert revoke["cancel_confirmed"] is True
    assert revoke["drained"] is True


def test_quick_branch_pair_preserves_negative_and_step_convergence_signals():
    """Keep a bounded real-engine check below the file's ten-second budget."""
    _require_mujoco()
    started = time.monotonic()

    base, base_root = _new_root(PhysicsConfig(timestep=0.002))
    start = base.read_feedback(base.now_ns)["position"]
    fast, _, _ = run_branch(PhysicsConfig(timestep=0.002), base_root, physics_plan(0.6, start), "negative-fast")
    slow, _, _ = run_branch(PhysicsConfig(timestep=0.002), base_root, physics_plan(1.6, start), "negative-slow")
    assert fast["outcome"] == "slip"
    assert slow["outcome"] == "stable"

    # This remains a deliberately weak pilot at 1ms/0.5ms: it checks the
    # classification and peak-offset signal only.  The experiment acceptance
    # gate uses the refined 0.25ms/0.125ms pair and additionally checks final
    # pose, velocity, orientation, tracking and contact outcomes.
    scaled = []
    for step in (0.001, 0.0005):
        controller, root = _new_root(PhysicsConfig(timestep=step))
        plan = physics_plan(1.6, controller.read_feedback(controller.now_ns)["position"])
        result, _, _ = run_branch(PhysicsConfig(timestep=step), root, plan, f"scale-{step}")
        scaled.append(result)
    assert scaled[0]["outcome"] == scaled[1]["outcome"]
    assert abs(scaled[0]["maximum_payload_offset_m"] - scaled[1]["maximum_payload_offset_m"]) <= 0.001
    assert time.monotonic() - started < 10.0
