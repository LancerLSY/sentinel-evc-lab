"""Optional bounded numeric path repair with mandatory final full recheck.

This is a small deterministic position adjustment, not SQP and not a general safety
filter.  It is disabled by default and never runs in the normal candidate path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .contracts import Plan, Scene
from .geometry import full_check


@dataclass(frozen=True)
class RepairResult:
    status: str
    plan: Optional[Plan]
    reason: Optional[str]
    iterations: int
    full_check_passed: bool

    def summary(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "iterations": self.iterations,
            "full_check_passed": self.full_check_passed,
            "plan_hash": self.plan.hash if self.plan else None,
            "method": "bounded-position-adjustment-v1-not-sqp",
        }


def repair_plan(
    plan: Plan,
    scene: Scene,
    enabled: bool = False,
    max_iterations: int = 20,
    step_size: float = 0.01,
    timeout_s: float = 0.05,
) -> RepairResult:
    if not enabled:
        return RepairResult("disabled", None, "OFF_BY_DEFAULT", 0, False)
    if max_iterations < 1 or step_size <= 0 or timeout_s <= 0:
        raise ValueError("repair bounds must be positive")
    if not all(
        math.isfinite(float(value))
        for knot in plan.knots
        for value in knot
    ):
        return RepairResult("infeasible", None, "NON_FINITE", 0, False)
    ok, _, _ = full_check(plan, scene)
    if ok:
        return RepairResult("unchanged", plan, None, 0, True)

    started = time.monotonic()
    original_first, original_last = plan.knots[0], plan.knots[-1]
    knots = [list(knot) for knot in plan.knots]
    for iteration in range(1, max_iterations + 1):
        if time.monotonic() - started > timeout_s:
            return RepairResult("timeout", None, "TIMEOUT", iteration - 1, False)
        moved = False
        for index in range(1, len(knots) - 1):
            point = knots[index]
            for obstacle in scene.obstacles:
                delta = [point[axis] - obstacle.center[axis] for axis in range(3)]
                distance = math.sqrt(sum(value * value for value in delta))
                required = obstacle.radius + scene.tool_radius + scene.tracking_reserve + step_size
                if distance < required:
                    if distance <= 1e-15:
                        return RepairResult(
                            "infeasible", None, "UNDEFINED_GRADIENT", iteration - 1, False
                        )
                    amount = min(required - distance, step_size)
                    for axis in range(3):
                        point[axis] += amount * delta[axis] / distance
                    moved = True
            for axis in range(3):
                point[axis] = min(max(point[axis], scene.ws_lo[axis]), scene.ws_hi[axis])
        if tuple(knots[0]) != original_first or tuple(knots[-1]) != original_last:
            return RepairResult("infeasible", None, "FIXED_ENDPOINT_CHANGED", iteration, False)
        candidate = Plan(
            plan.plan_id + "-repair",
            tuple(tuple(knot) for knot in knots),
            plan.dt,
            plan.gripper_events,
            plan.descriptor,
        )
        ok, _, _ = full_check(candidate, scene)
        if ok:
            return RepairResult("repaired", candidate, None, iteration, True)
        if not moved:
            return RepairResult("infeasible", None, "NO_LOCAL_DIRECTION", iteration, False)
    return RepairResult("infeasible", None, "ITERATION_LIMIT", max_iterations, False)
