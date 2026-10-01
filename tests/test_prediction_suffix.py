from dataclasses import replace

import pytest

from sentinel_evc.contracts import Plan
from sentinel_evc.numeric_world import generate_candidates, make_numeric_case
from sentinel_evc.prediction import (
    PredictionBindingError,
    build_default_numeric_artifacts,
    evaluate_candidates,
    predict_plan,
    slice_prediction,
)


def _artifacts():
    return build_default_numeric_artifacts(mode="physical", seed=9)


def test_action_changes_change_prediction_and_all_four_results_are_reported():
    case = make_numeric_case(77)
    plans = generate_candidates()
    model, calibration = _artifacts()
    evaluation = evaluate_candidates(
        case.history, plans, model, calibration, now_ns=10, risk_limit=0.12
    )
    assert len(evaluation.candidates) == 4
    assert evaluation.selected is not None
    assert evaluation.selected.allowed
    assert all(len(row.prediction.centers) == 40 for row in evaluation.candidates)
    assert all(row.summary()["plan"]["dt"] == 0.05 for row in evaluation.candidates)

    plan = plans[0]
    shuffled = Plan(
        "shuffled",
        (plan.knots[0],) + tuple(reversed(plan.knots[1:-1])) + (plan.knots[-1],),
        plan.dt,
    )
    normal = predict_plan(case.history, plan, model, calibration, 10, risk_limit=1.0)
    changed = predict_plan(case.history, shuffled, model, calibration, 10, risk_limit=1.0)
    assert normal.centers != changed.centers


def test_out_of_profile_displacement_is_retained_but_fail_closed_unknown():
    case = make_numeric_case(78)
    model, calibration = _artifacts()
    changed_profile = generate_candidates(displacement=0.20)
    evaluation = evaluate_candidates(
        case.history, changed_profile, model, calibration, now_ns=10, risk_limit=1.0
    )
    assert len(evaluation.candidates) == 4
    assert evaluation.selected is None
    assert all(not row.allowed for row in evaluation.candidates)
    assert all(row.reason == "MODEL_UNKNOWN" for row in evaluation.candidates)
    assert all(row.prediction.centers for row in evaluation.candidates)
    assert all(row.prediction.lower == () for row in evaluation.candidates)

    translated = generate_candidates(start=(0.2, -0.1, 0.3))
    translated_evaluation = evaluate_candidates(
        case.history, translated, model, calibration, now_ns=10, risk_limit=1.0
    )
    assert translated_evaluation.allowed_count == 4


def test_exact_suffix_reuse_preserves_root_binding_and_rejects_changed_actions():
    case = make_numeric_case(88)
    original = generate_candidates()[3]
    model, calibration = _artifacts()
    prediction = predict_plan(case.history, original, model, calibration, 100)
    offset = 5
    suffix = Plan("suffix", original.knots[offset:], original.dt)
    sliced = slice_prediction(prediction, original, suffix, offset, 101)
    assert sliced.root_prediction_hash == prediction.hash
    assert sliced.suffix_offset == offset
    assert sliced.centers == prediction.centers[offset:]

    changed_knots = list(suffix.knots)
    changed_knots[1] = (changed_knots[1][0] + 1e-9, *changed_knots[1][1:])
    changed = Plan("changed", tuple(changed_knots), suffix.dt)
    with pytest.raises(PredictionBindingError, match="suffix action mismatch"):
        slice_prediction(prediction, original, changed, offset, 101)


def test_suffix_rejects_expiry_and_model_calibration_rule_mismatch():
    case = make_numeric_case(99)
    original = generate_candidates()[2]
    model, calibration = _artifacts()
    prediction = predict_plan(case.history, original, model, calibration, 1_000, ttl_ns=10)
    suffix = Plan("suffix", original.knots[1:], original.dt)
    with pytest.raises(PredictionBindingError, match="expired"):
        slice_prediction(prediction, original, suffix, 1, 1_011)
    with pytest.raises(PredictionBindingError, match="model mismatch"):
        slice_prediction(prediction, original, suffix, 1, 1_001, model_hash="wrong")

    mismatched_calibration = replace(calibration, model_hash="wrong")
    with pytest.raises(PredictionBindingError, match="model/calibration mismatch"):
        predict_plan(case.history, original, model, mismatched_calibration, 1_000)

    with pytest.raises(ValueError, match="deeply immutable"):
        replace(prediction, plan_knots=[list(knot) for knot in prediction.plan_knots])


def test_earliest_allowed_independent_of_candidate_order():
    from sentinel_evc.numeric_world import make_numeric_case, generate_candidates
    from sentinel_evc.prediction import build_default_numeric_artifacts, evaluate_candidates
    model,calibration=build_default_numeric_artifacts(mode='physical')
    plans=generate_candidates()
    result=evaluate_candidates(make_numeric_case(7).history,tuple(reversed(plans)),model,calibration,0,risk_limit=.5)
    assert result.selected.plan == plans[0]
    assert len(result.candidates)==4

def test_unknown_and_denied_are_distinct():
    from sentinel_evc.numeric_world import make_numeric_case, generate_candidates
    from sentinel_evc.prediction import build_default_numeric_artifacts, evaluate_candidates
    model,calibration=build_default_numeric_artifacts(mode='physical')
    history=make_numeric_case(7).history
    unknown=evaluate_candidates(history,generate_candidates(.2),model,calibration,0)
    assert all(row.status=='unknown' for row in unknown.candidates)
    denied=evaluate_candidates(history,generate_candidates(),model,calibration,0,risk_limit=.001)
    assert all(row.status=='denied' for row in denied.candidates)
