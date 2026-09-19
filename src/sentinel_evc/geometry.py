"""几何完整检查。

约束族：静态球障碍 + 轴对齐盒工作空间 + 沿分段线性中心轨迹平移的球形工具。
两类约束在适用范围内的 Lipschitz 常数均为 L=1（见 delta_cert.py 的说明）。

本模块提供两个互相独立的检查实现：
  full_check          —— 主实现，解析求线段最近点
  full_check_sampled  —— 对照实现，密集采样逐点求距

G1 门禁要求两者在全部构造案例上零分歧。同一个思路写两遍不叫独立对照，
所以这两个函数刻意用了完全不同的方法。
"""

from __future__ import annotations

import math
from typing import Optional

from .contracts import Plan, Scene


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def segment_point_distance(p0, p1, c) -> float:
    """线段 p0->p1 上到点 c 的最小欧氏距离。

    这里必须求整段的最近点。只判断两个端点是最典型的漏检：
    端点都在障碍外、线段中部穿过球心的情况会被放行。
    tests/test_geometry_delta.py::test_midpoint_penetration 锁死这个行为。
    """
    d = _sub(p1, p0)
    dd = _dot(d, d)
    if dd == 0.0:
        return _norm(_sub(p0, c))
    t = _dot(_sub(c, p0), d) / dd
    t = max(0.0, min(1.0, t))  # 最近点参数夹到 [0, 1]
    closest = (p0[0] + t * d[0], p0[1] + t * d[1], p0[2] + t * d[2])
    return _norm(_sub(closest, c))


def _segment_box_margin(p0, p1, scene: Scene) -> float:
    """线段到盒边界的最小余量。

    轴向约束沿线段是线性函数，极值必在端点取到，所以取两端点的最坏值即可。
    这不是偷懒 —— 是线性函数的性质。球障碍就不能这么做。
    """
    slack = scene.tool_radius + scene.tracking_reserve
    worst = float("inf")
    for p in (p0, p1):
        for axis in range(3):
            worst = min(worst, p[axis] - scene.ws_lo[axis] - slack)
            worst = min(worst, scene.ws_hi[axis] - p[axis] - slack)
    return worst


def full_check(plan: Plan, scene: Scene):
    """完整几何检查（主实现）。

    返回 (ok, margins, first_violation_segment)：
      margins[k]              第 k 段所有约束中的最小余量，单位米
      first_violation_segment 第一个余量不大于零的段下标，全部通过时为 None
    """
    margins = []
    first_violation: Optional[int] = None

    for k in range(plan.horizon):
        p0, p1 = plan.points[k], plan.points[k + 1]
        seg_margin = _segment_box_margin(p0, p1, scene)

        for obs in scene.obstacles:
            dist = segment_point_distance(p0, p1, obs.center)
            m = dist - obs.radius - scene.tool_radius - scene.tracking_reserve
            seg_margin = min(seg_margin, m)

        margins.append(seg_margin)
        if seg_margin <= 0.0 and first_violation is None:
            first_violation = k

    return (first_violation is None), tuple(margins), first_violation


def full_check_sampled(plan: Plan, scene: Scene, samples_per_segment: int = 200):
    """完整几何检查（独立对照实现）。

    用密集采样代替解析最近点。采样必然比解析实现保守性略差（可能高估余量），
    所以比较时要允许一个与采样密度相称的容差。
    """
    margins = []
    first_violation: Optional[int] = None
    slack = scene.tool_radius + scene.tracking_reserve

    for k in range(plan.horizon):
        p0, p1 = plan.points[k], plan.points[k + 1]
        seg_margin = float("inf")

        for i in range(samples_per_segment + 1):
            t = i / samples_per_segment
            p = (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            )
            for axis in range(3):
                seg_margin = min(seg_margin, p[axis] - scene.ws_lo[axis] - slack)
                seg_margin = min(seg_margin, scene.ws_hi[axis] - p[axis] - slack)
            for obs in scene.obstacles:
                dist = _norm(_sub(p, obs.center))
                seg_margin = min(seg_margin, dist - obs.radius - slack)

        margins.append(seg_margin)
        if seg_margin <= 0.0 and first_violation is None:
            first_violation = k

    return (first_violation is None), tuple(margins), first_violation


def cross_validate(plan: Plan, scene: Scene, tol: float = 1e-4) -> bool:
    """G1 门禁用：两个独立实现的判定是否一致。

    只比较 ok/violation 的布尔结论，余量允许 tol 量级的差异 ——
    采样实现天然会略微高估余量。
    """
    ok_a, margins_a, _ = full_check(plan, scene)
    ok_b, margins_b, _ = full_check_sampled(plan, scene)
    if ok_a != ok_b:
        return False
    for ma, mb in zip(margins_a, margins_b):
        if abs(ma - mb) > tol:
            return False
    return True
