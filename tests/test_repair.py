from sentinel_evc.contracts import Plan, Scene, Sphere
from sentinel_evc.experiments import experiment_registry
from sentinel_evc.geometry import full_check
from sentinel_evc.repair import repair_plan


def _scene():
    return Scene(
        "repair",
        (Sphere((0.5, 0.0, 0.0), 0.12),),
        (-1.0, -1.0, -1.0),
        (2.0, 2.0, 2.0),
        tool_radius=0.0,
        tracking_reserve=0.0,
    )


def test_repair_is_off_by_default_and_never_changes_endpoints():
    plan = Plan("p", ((0.0, 0.3, 0.0), (0.5, 0.1, 0.0), (1.0, 0.3, 0.0)), 0.05)
    disabled = repair_plan(plan, _scene())
    assert disabled.status == "disabled"
    assert disabled.plan is None

    repaired = repair_plan(plan, _scene(), enabled=True, max_iterations=30, step_size=0.03)
    assert repaired.status == "repaired"
    assert repaired.full_check_passed
    assert repaired.plan.knots[0] == plan.knots[0]
    assert repaired.plan.knots[-1] == plan.knots[-1]
    assert full_check(repaired.plan, _scene())[0]


def test_undefined_gradient_fails_closed():
    plan = Plan("center", ((0.0, 0.3, 0.0), (0.5, 0.0, 0.0), (1.0, 0.3, 0.0)), 0.05)
    result = repair_plan(plan, _scene(), enabled=True)
    assert result.status == "infeasible"
    assert result.reason == "UNDEFINED_GRADIENT"
    assert result.plan is None


def test_external_experiments_are_pending_without_fake_metrics():
    records = tuple(record for record in experiment_registry() if record.status == "pending")
    assert {record.experiment_id for record in records} == {
        "real-vla-shadow",
        "gru-residual-world",
        "physical-robot-controller",
        "end-to-end-cost-benchmark",
    }
    assert all(record.status == "pending" for record in records)
    assert all(record.prerequisites and record.command and record.acceptance for record in records)
    assert all(record.metrics == () for record in records)
