"""Standard-library numeric prediction, binding, and candidate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .contracts import Plan, sha256_hex
from .numeric_world import NumericHistory, candidate_actions


DEFAULT_CANDIDATE_RULE = "fixed-four-duration-0.35m-h40-dt0.05-v1"
DEFAULT_CANDIDATE_RULE_HASH = sha256_hex(DEFAULT_CANDIDATE_RULE)


class PredictionBindingError(ValueError):
    pass


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...]:
    """Small partial-pivot Gaussian elimination used by the ridge baseline."""

    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][i] - factor * augmented[column][i]
                    for i in range(n + 1)
                ]
    answer = tuple(augmented[row][-1] for row in range(n))
    if not _finite(answer):
        raise ValueError("non-finite ridge solution")
    return answer


@dataclass(frozen=True)
class ResidualModel:
    family: str
    version: str
    mode: str
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_mean: float
    target_scale: float
    coefficients: tuple[float, ...]
    candidate_sigmas: tuple[float, float, float, float]
    train_root_ids: tuple[str, ...]
    train_manifest_hash: str
    nominal_parameters: tuple[float, float, float] = (20.0, 1.75, 80.0)

    def __post_init__(self) -> None:
        tuple_fields = (
            self.feature_names,
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            self.candidate_sigmas,
            self.train_root_ids,
            self.nominal_parameters,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise ValueError("model collections must be immutable tuples")
        numeric = (
            self.feature_mean
            + self.feature_scale
            + (self.target_mean, self.target_scale)
            + self.coefficients
            + self.candidate_sigmas
            + self.nominal_parameters
        )
        if not _finite(numeric):
            raise ValueError("model values must be finite")
        if any(value <= 0 for value in self.feature_scale + self.candidate_sigmas):
            raise ValueError("model scales must be positive")
        if self.target_scale <= 0 or len(self.candidate_sigmas) != 4:
            raise ValueError("invalid target/candidate scales")
        if self.mode not in ("physical", "residual"):
            raise ValueError("unknown model mode")

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary(include_hash=False))

    @property
    def model_id(self) -> str:
        return self.hash

    @property
    def sigma(self) -> float:
        return max(self.candidate_sigmas)

    def summary(self, include_hash: bool = True) -> dict:
        result = {
            "family": self.family,
            "version": self.version,
            "mode": self.mode,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "coefficients": list(self.coefficients),
            "candidate_sigmas": list(self.candidate_sigmas),
            "train_root_ids": list(self.train_root_ids),
            "train_manifest_hash": self.train_manifest_hash,
            "nominal_parameters": list(self.nominal_parameters),
            "training_scope": "train roots only; normalization fitted on the same train roots",
            "limitations": [
                "low-dimensional numeric residual baseline",
                "not the v4 GRU ResidualWorld",
                "not a robot or functional-safety model",
            ],
        }
        if include_hash:
            result["model_hash"] = self.hash
        return result


def _features(r: float, v: float, action: float) -> tuple[float, ...]:
    values = (r, v, action, r**3, 1.0)
    if not _finite(values):
        raise ValueError("non-finite model input")
    return values


def _mean_scale(columns: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means, scales = [], []
    for column in columns:
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        means.append(mean)
        scales.append(max(math.sqrt(variance), 1e-12))
    return tuple(means), tuple(scales)


def train_residual_model(examples, ridge: float = 1e-6) -> ResidualModel:
    """Fit a deterministic one-step ridge residual using train roots only."""

    roots = tuple(examples)
    if not roots or any(root.split != "train" for root in roots):
        raise ValueError("model fitting accepts train roots only")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    rows, targets = [], []
    for root in roots:
        for plan, outcome in zip(root.candidates, root.outcomes):
            actions = candidate_actions(plan)
            r = root.history.samples[-1].r
            v = root.history.samples[-1].v
            for action, next_r, next_v in zip(actions, outcome.r, outcome.v):
                rows.append(_features(r, v, action))
                nominal_next_v = v + plan.dt * (
                    -action - 20.0 * r - 1.75 * v - 80.0 * r**3
                )
                targets.append(next_v - nominal_next_v)
                r, v = next_r, next_v
    if not rows or not _finite(targets):
        raise ValueError("no finite training rows")
    columns = tuple(tuple(row[index] for row in rows) for index in range(len(rows[0])))
    feature_mean, feature_scale = _mean_scale(columns)
    target_mean = sum(targets) / len(targets)
    target_variance = sum((target - target_mean) ** 2 for target in targets) / len(targets)
    target_scale = max(math.sqrt(target_variance), 1e-12)
    normalized_rows = [
        tuple((value - feature_mean[i]) / feature_scale[i] for i, value in enumerate(row))
        for row in rows
    ]
    normalized_targets = [(target - target_mean) / target_scale for target in targets]
    width = len(rows[0])
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, target in zip(normalized_rows, normalized_targets):
        for i in range(width):
            rhs[i] += row[i] * target
            for j in range(width):
                gram[i][j] += row[i] * row[j]
    for index in range(width):
        gram[index][index] += ridge
    coefficients = _solve(gram, rhs)
    residuals = []
    for row, target in zip(normalized_rows, targets):
        predicted = target_mean + target_scale * sum(
            coefficient * value for coefficient, value in zip(coefficients, row)
        )
        residuals.append(target - predicted)
    sigma = max(math.sqrt(sum(value * value for value in residuals) / len(residuals)), 0.002)
    root_ids = tuple(sorted(root.root_id for root in roots))
    manifest_hash = sha256_hex(
        [{"root_id": root.root_id, "hash": root.hash} for root in sorted(roots, key=lambda item: item.root_id)]
    )
    provisional = ResidualModel(
        family="stdlib-ridge-residual",
        version="v1",
        mode="residual",
        feature_names=("r", "v", "actual_action", "r_cubed", "bias"),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        coefficients=coefficients,
        candidate_sigmas=(sigma, sigma, sigma, sigma),
        train_root_ids=root_ids,
        train_manifest_hash=manifest_hash,
    )
    candidate_errors = [[] for _ in range(4)]
    for root in roots:
        for index, (plan, outcome) in enumerate(zip(root.candidates, root.outcomes)):
            centers, _ = predict_trajectory(root.history, plan, provisional)
            candidate_errors[index].extend(
                truth - center for truth, center in zip(outcome.r, centers)
            )
    candidate_sigmas = tuple(
        # Candidate-specific train scales prevent a high-error fast candidate from
        # mechanically inflating the slower candidates' envelopes.  Calibration still
        # supplies the independent finite-sample multiplier.
        max(max(abs(error) for error in errors), 0.002)
        for errors in candidate_errors
    )
    return ResidualModel(
        **{**provisional.__dict__, "candidate_sigmas": candidate_sigmas}
    )


def make_physical_model() -> ResidualModel:
    """Return the fixed same-information nominal physical baseline."""

    return ResidualModel(
        family="numeric-physical-baseline",
        version="v1",
        mode="physical",
        feature_names=("r", "v", "actual_action"),
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        target_mean=0.0,
        target_scale=1.0,
        coefficients=(),
        candidate_sigmas=(0.04, 0.025, 0.015, 0.008),
        train_root_ids=(),
        train_manifest_hash=sha256_hex("fixed-nominal-physical-v1"),
    )


def _residual(model: ResidualModel, r: float, v: float, action: float, dt: float) -> float:
    k, d, beta = model.nominal_parameters
    physical = dt * (-k * r - d * v - beta * r**3)
    if model.mode == "physical":
        return physical
    raw = _features(r, v, action)
    normalized = tuple(
        (value - model.feature_mean[index]) / model.feature_scale[index]
        for index, value in enumerate(raw)
    )
    learned = model.target_mean + model.target_scale * sum(
        coefficient * value for coefficient, value in zip(model.coefficients, normalized)
    )
    return physical + learned


def predict_trajectory(
    history: NumericHistory, plan: Plan, model: ResidualModel
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Predict future load displacement from observable history and final actions."""

    r = history.samples[-1].r
    v = history.samples[-1].v
    centers, scales = [], []
    try:
        candidate_index = ("duration-0.6s", "duration-0.9s", "duration-1.2s", "duration-1.6s").index(plan.plan_id)
    except ValueError:
        candidate_index = 0
    for action in candidate_actions(plan):
        v = v - plan.dt * action + _residual(model, r, v, action, plan.dt)
        r = r + plan.dt * v
        if not _finite((r, v)):
            raise ValueError("non-finite model rollout")
        centers.append(r)
        scales.append(model.candidate_sigmas[candidate_index])
    return tuple(centers), tuple(scales)


@dataclass(frozen=True)
class Prediction:
    prediction_id: str
    plan_hash: str
    deadline_mono_ns: int
    allowed: bool
    history_hash: str
    model_hash: str
    calibration_hash: str
    candidate_rule_hash: str
    centers: tuple[float, ...]
    lower_envelope: tuple[float, ...]
    upper_envelope: tuple[float, ...]
    plan_knots: tuple
    dt: float
    descriptor_hash: str
    gripper_events: tuple
    risk_score: Optional[float]
    reason_code: Optional[str] = None
    root_prediction_hash: Optional[str] = None
    suffix_offset: int = 0

    def __post_init__(self) -> None:
        if any(not isinstance(value, tuple) for value in (
            self.centers,
            self.lower_envelope,
            self.upper_envelope,
        )):
            raise ValueError("prediction arrays must be immutable tuples")
        if not isinstance(self.plan_knots, tuple) or any(
            not isinstance(knot, tuple) for knot in self.plan_knots
        ):
            raise ValueError("prediction plan binding must be deeply immutable")
        if not isinstance(self.gripper_events, tuple):
            raise ValueError("prediction events must be immutable")
        numeric = self.centers + self.lower_envelope + self.upper_envelope + (self.dt,)
        numeric += tuple(value for knot in self.plan_knots for value in knot)
        if not _finite(numeric):
            raise ValueError("prediction values must be finite")
        if self.risk_score is not None and not math.isfinite(self.risk_score):
            raise ValueError("prediction risk score must be finite or None")
        if self.dt <= 0 or self.deadline_mono_ns < 0 or self.suffix_offset < 0:
            raise ValueError("invalid prediction timing")
        if self.lower_envelope and (
            len(self.lower_envelope) != len(self.centers)
            or len(self.upper_envelope) != len(self.centers)
        ):
            raise ValueError("prediction envelope length mismatch")

    @property
    def lower(self) -> tuple[float, ...]:
        return self.lower_envelope

    @property
    def upper(self) -> tuple[float, ...]:
        return self.upper_envelope

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary(include_hash=False))

    def summary(self, include_hash: bool = True) -> dict:
        result = {
            "prediction_id": self.prediction_id,
            "plan_hash": self.plan_hash,
            "deadline_mono_ns": self.deadline_mono_ns,
            "allowed": self.allowed,
            "history_hash": self.history_hash,
            "model_hash": self.model_hash,
            "calibration_hash": self.calibration_hash,
            "candidate_rule_hash": self.candidate_rule_hash,
            "centers": list(self.centers),
            "lower_envelope": list(self.lower_envelope),
            "upper_envelope": list(self.upper_envelope),
            "risk_score": self.risk_score,
            "reason_code": self.reason_code,
            "root_prediction_hash": self.root_prediction_hash,
            "suffix_offset": self.suffix_offset,
        }
        if include_hash:
            result["prediction_hash"] = self.hash
        return result


def predict_plan(
    history: NumericHistory,
    plan: Plan,
    model: ResidualModel,
    calibration,
    now_ns: int,
    ttl_ns: int = 60_000_000_000,
    risk_limit: float = 0.12,
    candidate_rule_hash: str = DEFAULT_CANDIDATE_RULE_HASH,
) -> Prediction:
    if ttl_ns <= 0 or not math.isfinite(risk_limit) or risk_limit < 0:
        raise ValueError("invalid prediction validity or risk limit")
    if calibration.model_hash != model.hash:
        raise PredictionBindingError("model/calibration mismatch")
    if calibration.candidate_rule_hash != candidate_rule_hash:
        raise PredictionBindingError("candidate-rule/calibration mismatch")
    centers, scales = predict_trajectory(history, plan, model)
    profile_ok = _matches_frozen_candidate_profile(plan)
    if not profile_ok:
        lower = upper = ()
        risk_score = None
        allowed = False
        reason = "MODEL_UNKNOWN"
    elif calibration.q is None or not math.isfinite(calibration.q):
        lower = upper = ()
        risk_score = None
        allowed = False
        reason = "CALIBRATION_UNKNOWN"
    else:
        lower = tuple(center - calibration.q * scale for center, scale in zip(centers, scales))
        upper = tuple(center + calibration.q * scale for center, scale in zip(centers, scales))
        risk_score = max(max(abs(lo), abs(hi)) for lo, hi in zip(lower, upper))
        allowed = risk_score <= risk_limit
        reason = None if allowed else "RISK_LIMIT"
    descriptor_hash = sha256_hex(plan.descriptor.summary())
    prediction_id = sha256_hex(
        {
            "plan_hash": plan.hash,
            "history_hash": history.hash,
            "model_hash": model.hash,
            "calibration_hash": calibration.hash,
            "candidate_rule_hash": candidate_rule_hash,
            "deadline": now_ns + ttl_ns,
        }
    )
    return Prediction(
        prediction_id=prediction_id,
        plan_hash=plan.hash,
        deadline_mono_ns=now_ns + ttl_ns,
        allowed=allowed,
        history_hash=history.hash,
        model_hash=model.hash,
        calibration_hash=calibration.hash,
        candidate_rule_hash=candidate_rule_hash,
        centers=centers,
        lower_envelope=lower,
        upper_envelope=upper,
        plan_knots=plan.knots,
        dt=plan.dt,
        descriptor_hash=descriptor_hash,
        gripper_events=plan.gripper_events,
        risk_score=risk_score,
        reason_code=reason,
    )


def _matches_frozen_candidate_profile(plan: Plan) -> bool:
    """The bundled calibration covers one translated 0.35 m action family only."""

    from .numeric_world import generate_candidates

    if plan.horizon != 40 or plan.dt != 0.05 or plan.gripper_events:
        return False
    reference_by_id = {
        candidate.plan_id: candidate
        for candidate in generate_candidates(displacement=0.35, start=plan.knots[0])
    }
    reference = reference_by_id.get(plan.plan_id)
    if reference is None:
        return False
    return (
        plan.knots == reference.knots
        and plan.descriptor == reference.descriptor
        and candidate_actions(plan) == candidate_actions(reference)
    )


def slice_prediction(
    prediction: Prediction,
    original_plan: Plan,
    suffix_plan: Plan,
    offset: int,
    now_ns: int,
    history_hash: Optional[str] = None,
    model_hash: Optional[str] = None,
    calibration_hash: Optional[str] = None,
    candidate_rule_hash: Optional[str] = None,
) -> Prediction:
    """Reuse only an exact remaining suffix with unchanged evidence bindings."""

    if now_ns > prediction.deadline_mono_ns:
        raise PredictionBindingError("prediction expired")
    if prediction.plan_hash != original_plan.hash or prediction.plan_knots != original_plan.knots:
        raise PredictionBindingError("original plan mismatch")
    if offset < 0 or offset >= len(original_plan.knots):
        raise PredictionBindingError("invalid suffix offset")
    expected_knots = original_plan.knots[offset:]
    expected_events = tuple(
        (event[0] - offset, *event[1:])
        for event in original_plan.gripper_events
        if event[0] >= offset
    )
    if suffix_plan.knots != expected_knots:
        raise PredictionBindingError("suffix action mismatch")
    if suffix_plan.dt != original_plan.dt or suffix_plan.descriptor != original_plan.descriptor:
        raise PredictionBindingError("suffix timing/descriptor mismatch")
    if suffix_plan.gripper_events != expected_events:
        raise PredictionBindingError("suffix event mismatch")
    bindings = (
        (history_hash, prediction.history_hash, "history"),
        (model_hash, prediction.model_hash, "model"),
        (calibration_hash, prediction.calibration_hash, "calibration"),
        (candidate_rule_hash, prediction.candidate_rule_hash, "candidate rule"),
    )
    for supplied, expected, label in bindings:
        if supplied is not None and supplied != expected:
            raise PredictionBindingError(f"{label} mismatch")
    interval_offset = min(offset, len(prediction.centers))
    return Prediction(
        prediction_id=sha256_hex({"root": prediction.hash, "offset": offset, "plan": suffix_plan.hash}),
        plan_hash=suffix_plan.hash,
        deadline_mono_ns=prediction.deadline_mono_ns,
        allowed=prediction.allowed,
        history_hash=prediction.history_hash,
        model_hash=prediction.model_hash,
        calibration_hash=prediction.calibration_hash,
        candidate_rule_hash=prediction.candidate_rule_hash,
        centers=prediction.centers[interval_offset:],
        lower_envelope=prediction.lower_envelope[interval_offset:],
        upper_envelope=prediction.upper_envelope[interval_offset:],
        plan_knots=suffix_plan.knots,
        dt=suffix_plan.dt,
        descriptor_hash=prediction.descriptor_hash,
        gripper_events=suffix_plan.gripper_events,
        risk_score=prediction.risk_score,
        reason_code=prediction.reason_code,
        root_prediction_hash=prediction.root_prediction_hash or prediction.hash,
        suffix_offset=prediction.suffix_offset + offset,
    )


@dataclass(frozen=True)
class CandidateResult:
    plan: Plan
    physical_allowed: bool
    prediction: Prediction
    allowed: bool
    reason_code: Optional[str]
    geometry_margin: Optional[float] = None

    def __post_init__(self) -> None:
        if self.geometry_margin is not None and not math.isfinite(self.geometry_margin):
            raise ValueError("geometry margin must be finite or None")

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    @property
    def plan_hash(self) -> str:
        return self.plan.hash

    @property
    def status(self) -> str:
        return "allowed" if self.allowed else ("unknown" if self.reason_code in {"MODEL_UNKNOWN", "CALIBRATION_UNKNOWN"} else "denied")

    @property
    def reason(self) -> Optional[str]:
        return self.reason_code

    @property
    def predicted_peak(self) -> float:
        return max((abs(value) for value in self.prediction.centers), default=0.0)

    @property
    def upper_peak(self) -> Optional[float]:
        if not self.prediction.upper:
            return None
        return max(
            max(abs(lo), abs(hi))
            for lo, hi in zip(self.prediction.lower, self.prediction.upper)
        )

    def summary(self) -> dict:
        return {
            "id": self.plan.plan_id,
            "duration": next(index * self.plan.dt for index,knot in enumerate(self.plan.knots) if knot == self.plan.knots[-1]),
            "plan_hash": self.plan.hash,
            "physical_allowed": self.physical_allowed,
            "prediction": self.prediction.summary(),
            "allowed": self.allowed,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "geometry_margin": self.geometry_margin,
            "predicted_peak": self.predicted_peak,
            "upper_peak": self.upper_peak,
            "plan": {
                "knots": [list(knot) for knot in self.plan.knots],
                "dt": self.plan.dt,
            },
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    selected: Optional[CandidateResult]
    results: tuple[CandidateResult, ...]
    candidate_rule_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple):
            raise ValueError("candidate results must be an immutable tuple")
        if self.selected is not None and self.selected not in self.results:
            raise ValueError("selected candidate must belong to results")

    @property
    def candidates(self) -> tuple[CandidateResult, ...]:
        return self.results

    @property
    def selected_plan_hash(self) -> Optional[str]:
        return self.selected.plan.hash if self.selected else None

    @property
    def allowed_count(self) -> int:
        return sum(result.allowed for result in self.results)

    def summary(self) -> dict:
        return {
            "selected_plan_hash": self.selected_plan_hash,
            "allowed_count": self.allowed_count,
            "candidate_rule_hash": self.candidate_rule_hash,
            "results": [result.summary() for result in self.results],
        }


def evaluate_candidates(
    history: NumericHistory,
    candidates: Sequence[Plan],
    model: ResidualModel,
    calibration,
    now_ns: int,
    risk_limit: float = 0.12,
    physical_check: Optional[Callable[[Plan], bool]] = None,
    candidate_rule_hash: str = DEFAULT_CANDIDATE_RULE_HASH,
    ttl_ns: int = 60_000_000_000,
) -> CandidateEvaluation:
    """Evaluate all final candidates without any controller/driver access."""

    if len(candidates) != 4:
        raise ValueError("numeric product requires all four final candidates")
    results = []
    for plan in candidates:
        physical_result = True if physical_check is None else physical_check(plan)
        if isinstance(physical_result, tuple):
            physical_allowed = bool(physical_result[0])
            geometry_margin = float(physical_result[1]) if physical_result[1] is not None else None
        else:
            physical_allowed = bool(physical_result)
            geometry_margin = None
        prediction = predict_plan(
            history,
            plan,
            model,
            calibration,
            now_ns,
            ttl_ns=ttl_ns,
            risk_limit=risk_limit,
            candidate_rule_hash=candidate_rule_hash,
        )
        allowed = physical_allowed and prediction.allowed
        reason = None if allowed else (
            prediction.reason_code if physical_allowed else "PHYSICAL_REJECTED"
        )
        results.append(CandidateResult(plan, physical_allowed, prediction, allowed, reason, geometry_margin))
    allowed_results = [result for result in results if result.allowed]
    selected = min(allowed_results, key=lambda result: (result.summary()["duration"], result.plan.hash)) if allowed_results else None
    return CandidateEvaluation(selected, tuple(results), candidate_rule_hash)


def build_default_numeric_artifacts(
    mode: str = "residual", seed: int = 0
):
    """Build a modest deterministic model/calibration pair for CLI/product use."""

    from .calibration import fit_root_max_calibration
    from .data import make_numeric_dataset

    if mode not in ("physical", "residual"):
        raise ValueError("mode must be 'physical' or 'residual'")
    dataset = make_numeric_dataset(
        train_roots=32 if mode == "residual" else 0,
        dev_roots=8,
        cal_roots=19,
        test_roots=0,
        # Public product scenarios accept seeds through 1,000,000.  Keep the bundled
        # helper's fitting roots in a separate namespace so a displayed product root
        # is never silently one of its train/calibration roots.
        seed=10_000_000 + seed,
    )
    model = (
        train_residual_model(dataset.split("train"))
        if mode == "residual"
        else make_physical_model()
    )
    calibration = fit_root_max_calibration(dataset.split("cal"), model)
    return model, calibration
