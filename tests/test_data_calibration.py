from dataclasses import replace

import pytest

from sentinel_evc.calibration import fit_root_max_calibration
from sentinel_evc.data import NumericDataset, make_numeric_dataset
from sentinel_evc.prediction import build_default_numeric_artifacts, train_residual_model


def test_root_siblings_cannot_cross_splits():
    dataset = make_numeric_dataset(2, 1, 2, 1, seed=100)
    train_root = dataset.split("train")[0]
    leaked = replace(train_root, split="cal")
    with pytest.raises(ValueError, match="both train and cal"):
        NumericDataset(dataset.roots + (leaked,))


def test_residual_model_is_deterministic_and_has_no_hidden_features():
    dataset = make_numeric_dataset(4, 0, 0, 0, seed=4)
    first = train_residual_model(dataset.split("train"))
    second = train_residual_model(dataset.split("train"))
    assert first.hash == second.hash
    assert set(first.feature_names) == {"r", "v", "actual_action", "r_cubed", "bias"}
    assert all(name not in first.feature_names for name in ("k", "d", "beta", "hidden_params"))
    assert first.summary()["model_hash"] == first.hash


def test_finite_sample_rank_returns_unknown_when_rank_exceeds_root_count():
    train = make_numeric_dataset(4, 0, 0, 0, seed=20)
    model = train_residual_model(train.split("train"))
    small_cal = make_numeric_dataset(0, 0, 8, 0, seed=200)
    calibration = fit_root_max_calibration(small_cal.split("cal"), model, alpha=0.1)
    assert calibration.rank == 9
    assert calibration.q is None
    assert not calibration.finite


def test_direct_calibration_rejects_duplicate_and_train_overlap_roots():
    dataset = make_numeric_dataset(4, 0, 2, 0, seed=300)
    model = train_residual_model(dataset.split("train"))
    cal_root = dataset.split("cal")[0]
    with pytest.raises(ValueError, match="duplicate root"):
        fit_root_max_calibration((cal_root, cal_root), model)

    overlapping = replace(dataset.split("train")[0], split="cal")
    with pytest.raises(ValueError, match="overlap model training roots"):
        fit_root_max_calibration((overlapping,), model)


def test_default_helpers_produce_finite_independent_calibration():
    for mode in ("physical", "residual"):
        model, calibration = build_default_numeric_artifacts(mode=mode, seed=2)
        assert calibration.finite
        assert calibration.root_count == 19
        assert calibration.model_hash == model.hash
        assert calibration.summary()["calibration_hash"] == calibration.hash

        with pytest.raises(ValueError, match="finite"):
            replace(calibration, q=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            replace(model, candidate_sigmas=(0.1, 0.1, 0.1, float("nan")))
