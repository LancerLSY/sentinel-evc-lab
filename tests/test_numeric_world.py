import math

import pytest

from sentinel_evc.numeric_world import (
    HiddenPlantParameters,
    candidate_actions,
    generate_candidates,
    make_numeric_case,
    plant_step,
    rollout_candidate,
)


def test_numeric_case_uses_only_observed_actions_and_four_fixed_candidates():
    case = make_numeric_case(7)
    candidates = generate_candidates()

    assert len(case.history.samples) == 8
    assert all(sample.action == case.actual_actions[i]
               for i, sample in enumerate(case.history.samples))
    assert not hasattr(case.history, "hidden_params")
    assert [plan.plan_id for plan in candidates] == [
        "duration-0.6s", "duration-0.9s", "duration-1.2s", "duration-1.6s"
    ]
    assert all(plan.horizon == 40 and plan.dt == 0.05 for plan in candidates)
    assert all(plan.knots[0] == (0.0, 0.0, 0.0) for plan in candidates)
    assert all(plan.knots[-1] == (0.35, 0.0, 0.0) for plan in candidates)


def test_plant_and_rollout_are_deterministic_and_finite():
    params = HiddenPlantParameters(k=12.0, d=1.5, beta=60.0)
    assert plant_step(0.1, -0.2, 0.3, params) == plant_step(0.1, -0.2, 0.3, params)
    case = make_numeric_case(3)
    plan = generate_candidates()[0]
    outcome = rollout_candidate(case, plan)
    assert len(outcome.r) == plan.horizon
    assert all(math.isfinite(value) for value in outcome.r + outcome.v)

    with pytest.raises(ValueError):
        plant_step(float("nan"), 0.0, 0.0, params)


def test_candidate_action_order_changes_the_rollout():
    case = make_numeric_case(11)
    plan = generate_candidates()[1]
    actions = candidate_actions(plan)
    normal = rollout_candidate(case, plan)
    shuffled = rollout_candidate(case, plan, actions=tuple(reversed(actions)))
    assert normal.r != shuffled.r
