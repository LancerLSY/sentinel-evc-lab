"""场景与轨迹构造。

构造的轨迹**不是** VLA 的输出，也不冒充 VLA 的错误。它们是为了让
「变换让旧结论失效」这件事可复现地发生而设计的数值样例。

真实 VLA 版本要用模型实际输出和官方聚合路径，保持相同初始状态、种子
和预处理 —— 不用脚本制造的动作冒充模型错误。
"""

from __future__ import annotations

import math
import random

from .contracts import Plan, Scene, Sphere, TransformRecord

HORIZON = 16
DT = 0.05

START = (0.10, 0.0, 0.30)
GOAL = (0.70, 0.0, 0.30)


def make_scene(seed: int = 0) -> Scene:
    """球障碍正好挡在起点到终点的直线上。"""
    rng = random.Random(seed)
    jitter = rng.uniform(-0.01, 0.01)
    return Scene(
        scene_id=f"scene-{seed:04d}",
        obstacles=(Sphere(center=(0.40, jitter, 0.30), radius=0.05),),
        ws_lo=(0.0, -0.40, 0.0),
        ws_hi=(0.80, 0.40, 0.60),
        tool_radius=0.02,
        tracking_reserve=0.005,
    )


def _arc(amplitude: float, plan_id: str, noise: float = 0.0, seed: int = 0) -> Plan:
    """从 START 到 GOAL 的分段线性路径，y 方向按正弦鼓起 amplitude。"""
    rng = random.Random(seed)
    knots = []
    for i in range(HORIZON + 1):
        t = i / HORIZON
        x = START[0] + t * (GOAL[0] - START[0])
        y = amplitude * math.sin(math.pi * t)
        z = START[2]
        if noise and 0 < i < HORIZON:
            x += rng.uniform(-noise, noise)
            y += rng.uniform(-noise, noise)
            z += rng.uniform(-noise, noise)
        knots.append((x, y, z))
    return Plan(plan_id=plan_id, knots=tuple(knots), dt=DT)


def make_parent_pair(seed: int = 0):
    """两条从障碍两侧绕过的父轨迹。各自单独验证都能通过。"""
    p1 = _arc(+0.15, f"P1-{seed:06d}")
    p2 = _arc(-0.15, f"P2-{seed:06d}")
    return p1, p2


def mix(p1: Plan, p2: Plan, weight: float = 0.5, plan_id: str = "MIX"):
    """加权混合两条轨迹 —— 这是真实部署链里很常见的一种动作聚合。

    两条都验证通过，混合结果却穿过障碍。这就是整个项目要说的那件事。
    """
    knots = tuple(
        tuple(weight * a[j] + (1 - weight) * b[j] for j in range(3))
        for a, b in zip(p1.knots, p2.knots)
    )
    child = Plan(plan_id=plan_id, knots=knots, dt=p1.dt)
    record = TransformRecord(
        transform_id=f"tf-{plan_id}",
        kind="mix",  # 不在 INHERITABLE 里 —— 混合必须当作新的最终候选
        parent_hash=p1.hash,
        child_hash=child.hash,
        parameters={"weight": weight},
    )
    return child, record


def perturb(p1: Plan, magnitude: float = 0.008, seed: int = 0, plan_id: str = "NEAR"):
    """同侧小扰动 —— 正常的路径微调，应该能被继承。

    这条是必要的负对照：没有它，别人会认为这套机制只是「一变就拒」。
    """
    rng = random.Random(seed)
    knots = [p1.knots[0]]
    for k in p1.knots[1:-1]:
        knots.append(tuple(k[j] + rng.uniform(-magnitude, magnitude) for j in range(3)))
    knots.append(p1.knots[-1])
    child = Plan(plan_id=plan_id, knots=tuple(knots), dt=p1.dt)
    record = TransformRecord(
        transform_id=f"tf-{plan_id}",
        kind="perturb",  # 已登记、可计算上界 —— 允许尝试继承
        parent_hash=p1.hash,
        child_hash=child.hash,
        parameters={"magnitude": magnitude},
    )
    return child, record
