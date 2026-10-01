"""Deterministic low-dimensional load plant for the numeric workbench.

The history surface contains observations and actions that were actually applied.
Hidden plant parameters remain on the evaluator-only ``NumericCase`` object.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional

from .contracts import Plan, sha256_hex


CANDIDATE_DURATIONS = (0.6, 0.9, 1.2, 1.6)


def _finite(*values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("numeric plant inputs must be finite")


@dataclass(frozen=True)
class HiddenPlantParameters:
    """Simulator-only parameters; never part of ``NumericHistory`` or model features."""

    k: float
    d: float
    beta: float

    def __post_init__(self) -> None:
        _finite(self.k, self.d, self.beta)
        if self.k <= 0 or self.d <= 0 or self.beta <= 0:
            raise ValueError("plant parameters must be positive")


@dataclass(frozen=True)
class NumericSample:
    r: float
    v: float
    a: float

    def __post_init__(self) -> None:
        _finite(self.r, self.v, self.a)

    @property
    def action(self) -> float:
        return self.a


@dataclass(frozen=True)
class NumericHistory:
    root_id: str
    dt: float
    samples: tuple[NumericSample, ...]

    def __post_init__(self) -> None:
        _finite(self.dt)
        if not isinstance(self.samples, tuple) or any(
            not isinstance(sample, NumericSample) for sample in self.samples
        ):
            raise ValueError("history samples must be an immutable NumericSample tuple")
        if self.dt <= 0 or not self.samples:
            raise ValueError("history requires positive dt and at least one sample")

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary())

    def summary(self) -> dict:
        return {
            "root_id": self.root_id,
            "dt": self.dt,
            "samples": [
                {"r": sample.r, "v": sample.v, "a": sample.a}
                for sample in self.samples
            ],
        }


@dataclass(frozen=True)
class NumericCase:
    root_id: str
    history: NumericHistory
    actual_actions: tuple[float, ...]
    evaluator_params: HiddenPlantParameters
    evaluator_r: float
    evaluator_v: float

    def __post_init__(self) -> None:
        if not isinstance(self.actual_actions, tuple):
            raise ValueError("actual actions must be immutable")
        _finite(*self.actual_actions, self.evaluator_r, self.evaluator_v)
        if len(self.actual_actions) != len(self.history.samples):
            raise ValueError("actual action/history length mismatch")

    @property
    def params(self) -> HiddenPlantParameters:
        return self.evaluator_params

    @property
    def current_r(self) -> float:
        return self.evaluator_r

    @property
    def current_v(self) -> float:
        return self.evaluator_v


@dataclass(frozen=True)
class NumericOutcome:
    root_id: str
    plan_hash: str
    actions: tuple[float, ...]
    r: tuple[float, ...]
    v: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, tuple) for value in (self.actions, self.r, self.v)):
            raise ValueError("outcome arrays must be immutable tuples")
        _finite(*self.actions, *self.r, *self.v)
        if not (len(self.actions) == len(self.r) == len(self.v)):
            raise ValueError("outcome arrays must have equal length")


def plant_step(
    r: float,
    v: float,
    action: float,
    params: HiddenPlantParameters,
    dt: float = 0.05,
) -> tuple[float, float]:
    """Advance ``v_next=v+dt*(-a-k*r-d*v-beta*r^3)`` one interval."""

    _finite(r, v, action, dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    v_next = v + dt * (
        -action - params.k * r - params.d * v - params.beta * r**3
    )
    r_next = r + dt * v_next
    _finite(r_next, v_next)
    return r_next, v_next


def make_numeric_case(seed: int, history_len: int = 8, dt: float = 0.05) -> NumericCase:
    """Create one deterministic root with observable noisy history and hidden truth."""

    if history_len != 8:
        raise ValueError("the product profile fixes history_len=8")
    _finite(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = random.Random(seed)
    params = HiddenPlantParameters(
        k=rng.uniform(5.0, 35.0),
        d=rng.uniform(0.5, 3.0),
        beta=rng.uniform(40.0, 120.0),
    )
    r = rng.uniform(-0.025, 0.025)
    v = rng.uniform(-0.04, 0.04)
    samples = []
    actions = []
    for index in range(history_len):
        action = 0.18 * math.sin((seed + 1) * 0.17 + index * 0.61)
        r, v = plant_step(r, v, action, params, dt)
        observed_r = r + rng.uniform(-0.0015, 0.0015)
        observed_v = v + rng.uniform(-0.003, 0.003)
        samples.append(NumericSample(observed_r, observed_v, action))
        actions.append(action)
    root_id = f"numeric-root-{seed:08d}"
    return NumericCase(
        root_id=root_id,
        history=NumericHistory(root_id, dt, tuple(samples)),
        actual_actions=tuple(actions),
        evaluator_params=params,
        evaluator_r=r,
        evaluator_v=v,
    )


def generate_candidates(
    displacement: float = 0.35,
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dt: float = 0.05,
    horizon: int = 40,
) -> tuple[Plan, ...]:
    """Generate the four fixed-duration final 3-D Plans from one root."""

    if len(start) != 3:
        raise ValueError("start must be a 3-D position")
    _finite(displacement, dt, *start)
    if dt <= 0 or horizon != 40:
        raise ValueError("numeric profile requires dt>0 and horizon=40")
    plans = []
    for duration in CANDIDATE_DURATIONS:
        move_intervals = int(round(duration / dt))
        knots = []
        for index in range(horizon + 1):
            u = min(index / move_intervals, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            knots.append(
                (
                    start[0] + displacement * smooth,
                    start[1],
                    start[2],
                )
            )
        plans.append(Plan(f"duration-{duration:.1f}s", tuple(knots), dt))
    return tuple(plans)


def candidate_actions(plan: Plan) -> tuple[float, ...]:
    """Map absolute position knots to scalar acceleration with initial velocity zero.

    The plant follows ``v_next = v + dt*(-a - ...)`` from the design note, so ``a``
    is acceleration rather than the first-difference velocity.  Holding after transfer
    consequently includes the deceleration needed to bring the commanded velocity to zero.
    """

    _finite(plan.dt)
    if plan.dt <= 0:
        raise ValueError("plan dt must be positive")
    actions = []
    previous_velocity = 0.0
    for index in range(plan.horizon):
        velocity = (
            float(plan.knots[index + 1][0]) - float(plan.knots[index][0])
        ) / plan.dt
        actions.append((velocity - previous_velocity) / plan.dt)
        previous_velocity = velocity
    actions = tuple(actions)
    _finite(*actions)
    return actions


def rollout_candidate(
    case: NumericCase,
    plan: Plan,
    actions: Optional[Iterable[float]] = None,
) -> NumericOutcome:
    """Evaluator-only rollout using hidden parameters and an explicit action sequence."""

    applied = tuple(candidate_actions(plan) if actions is None else actions)
    if len(applied) != plan.horizon:
        raise ValueError("action sequence must match plan horizon")
    r, v = case.evaluator_r, case.evaluator_v
    rs, vs = [], []
    for action in applied:
        r, v = plant_step(r, v, float(action), case.evaluator_params, plan.dt)
        rs.append(r)
        vs.append(v)
    return NumericOutcome(case.root_id, plan.hash, applied, tuple(rs), tuple(vs))
