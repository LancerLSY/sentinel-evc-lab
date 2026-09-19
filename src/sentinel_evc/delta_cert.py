"""Δ-Cert 增量重验。

数学基础
--------
设父轨迹对约束 i 在第 k 段已验证剩余下界 rho(i,k)，子轨迹与父轨迹在该段
的状态差异不超过 e(k)，约束函数在相应区域满足 Lipschitz 常数 L(i)，则：

    rho_child(i,k) >= rho_parent(i,k) - L(i) * e(k)

相同时间网格下，第 k 段任意插值点的偏差不超过两端点偏差范数的最大值：

    e(k) = max( ||p'_k - p_k||, ||p'_{k+1} - p_{k+1}|| )

本项目的两类约束 —— 点到固定球的欧氏距离、轴向盒边界 —— 都满足 L=1，
所以每段只需保存一个最小余量，转移检查从 O(H*M) 降到 O(H)。

这是对已有连续性与安全证书思想的应用，不是新的数学定理。项目的增量在于
把它和变换登记、依赖失效、许可生命周期组合起来，并实测成本收益。

两条容易写错的规则
------------------
1. 余量从父的**剩余量**扣减，不是每次回到最初值。
   30mm 先耗 8mm 剩 22，再耗 5mm 剩 17 —— 不是 30-5=25。
2. 继承失败的结论是「无法证明，去做完整检查」，不是「一定会碰撞」。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

from .contracts import MAX_INHERIT_DEPTH, Certificate, Plan, Scene, TransformRecord
from .geometry import full_check

# 本项目约束族的 Lipschitz 常数。整臂推广时这个值不再是 1，
# 拿不到有效界时必须退回 FULL_ONLY，不能沿用。
LIPSCHITZ = 1.0

_cert_counter = itertools.count(1)


def _next_cert_id() -> str:
    return f"cert-{next(_cert_counter):06d}"


@dataclass(frozen=True)
class Verdict:
    """一次判定的完整结果，对应 schemas/verdict.schema.json。"""

    plan_hash: str
    verdict: str  # FULL | INHERITED | REJECTED
    path: str  # full | delta
    certificate: Optional[Certificate] = None
    margins: tuple = ()
    first_violation_segment: Optional[int] = None
    full_checks_used: int = 0
    inherit_depth: int = 0
    parent_cert_id: Optional[str] = None
    reason_code: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "margins", tuple(self.margins))

    def summary(self) -> dict:
        return {
            "plan_hash": self.plan_hash,
            "verdict": self.verdict,
            "path": self.path,
            "parent_cert_id": self.parent_cert_id,
            "margins": list(self.margins),
            "first_violation_segment": self.first_violation_segment,
            "full_checks_used": self.full_checks_used,
            "inherit_depth": self.inherit_depth,
            "reason_code": self.reason_code,
        }


def deviation_bounds(parent: Plan, child: Plan) -> tuple:
    """逐段偏差界 e(k)，要求同时间网格、同节点数。"""
    if parent.horizon != child.horizon or parent.dt != child.dt:
        raise ValueError("时间网格不同，不能按段比较偏差")
    e = []
    for k in range(parent.horizon):
        d0 = math.dist(parent.points[k], child.points[k])
        d1 = math.dist(parent.points[k + 1], child.points[k + 1])
        e.append(max(d0, d1))
    return tuple(e)


def dependency_ok(
    parent_cert: Certificate,
    child: Plan,
    scene: Scene,
    transform: TransformRecord,
) -> Optional[str]:
    """检查继承的前置依赖。返回失效原因，全部满足时返回 None。

    几何余量够用不代表可以继承 —— 时间网格、事件、场景、变换类型
    任何一项变化都要先挡住。
    """
    if not transform.inheritable:
        return "transform_not_registered"
    if parent_cert.scene_id != scene.scene_id:
        return "scene_changed"
    if parent_cert.plan.dt != child.dt:
        return "dt_changed"
    if parent_cert.plan.horizon != child.horizon:
        return "horizon_changed"
    if parent_cert.plan.gripper_events != child.gripper_events:
        return "gripper_events_changed"
    if parent_cert.plan.controller_profile != child.controller_profile:
        return "controller_profile_changed"
    if parent_cert.plan.task_phase != child.task_phase:
        return "task_phase_changed"
    if parent_cert.inherit_depth >= MAX_INHERIT_DEPTH:
        return "max_depth_exceeded"
    return None


def validate_or_inherit(
    child: Plan,
    scene: Scene,
    parent_plan: Optional[Plan] = None,
    parent_cert: Optional[Certificate] = None,
    transform: Optional[TransformRecord] = None,
) -> Verdict:
    """路径 c：先试继承，界不够则完整检查。

    这是产品主路径。路径 a（只验父轨迹）和路径 b（总是完整检查）
    在 pipeline.py 里作为对照实现。
    """
    # --- 情形一：无父证书，只能完整检查
    if parent_cert is None or parent_plan is None or transform is None:
        return _full_path(child, scene)

    # 余量只能绑定证书内的父计划，不能配上另一个父计划低估偏差。
    if (parent_plan != parent_cert.plan
            or transform.parent_hash != parent_cert.plan_hash
            or transform.child_hash != child.hash):
        return _full_path(child, scene, parent_cert_id=parent_cert.cert_id)

    # identity 也走同一条路径，保证继承深度和依赖检查不被绕过。
    reason = dependency_ok(parent_cert, child, scene, transform)
    if reason is not None:
        return _full_path(child, scene, parent_cert_id=parent_cert.cert_id)

    e = deviation_bounds(parent_plan, child)
    # 关键：从父证书的剩余余量扣，不是从最初值扣
    child_margins = tuple(
        parent_cert.margins[k] - LIPSCHITZ * e[k] for k in range(child.horizon)
    )

    if min(child_margins) > 0.0:
        cert = Certificate(
            cert_id=_next_cert_id(),
            plan=child,
            scene_id=scene.scene_id,
            margins=child_margins,
            inherit_depth=parent_cert.inherit_depth + 1,
            parent_cert_id=parent_cert.cert_id,
        )
        return Verdict(
            plan_hash=child.hash,
            verdict="INHERITED",
            path="delta",
            certificate=cert,
            margins=child_margins,
            full_checks_used=0,
            inherit_depth=cert.inherit_depth,
            parent_cert_id=parent_cert.cert_id,
        )

    # 界不够 —— 这只说明「无法证明」，还要真的做完整检查才能下结论
    return _full_path(child, scene, parent_cert_id=parent_cert.cert_id)


def _full_path(
    child: Plan, scene: Scene, parent_cert_id: Optional[str] = None
) -> Verdict:
    ok, margins, first_violation = full_check(child, scene)
    if not ok:
        return Verdict(
            plan_hash=child.hash,
            verdict="REJECTED",
            path="full",
            margins=margins,
            first_violation_segment=first_violation,
            full_checks_used=1,
            parent_cert_id=parent_cert_id,
        )
    cert = Certificate(
        cert_id=_next_cert_id(),
        plan=child,
        scene_id=scene.scene_id,
        margins=margins,
        inherit_depth=0,
        parent_cert_id=None,
    )
    return Verdict(
        plan_hash=child.hash,
        verdict="FULL",
        path="full",
        certificate=cert,
        margins=margins,
        full_checks_used=1,
        parent_cert_id=parent_cert_id,
    )


def establish_root(plan: Plan, scene: Scene) -> Verdict:
    """建立父（根）证书：一次完整几何检查。"""
    return _full_path(plan, scene)


class CertificateStore:
    """本地可信证书登记表。

    只接纳本地验证器产生的记录。上游发来的 safe=true、physical_ok=true
    或自选的证书 ID 一律不构成证据。
    """

    def __init__(self):
        self._certs = {}

    def register(self, cert: Certificate) -> None:
        self._certs[cert.cert_id] = cert

    def get(self, cert_id: str) -> Optional[Certificate]:
        return self._certs.get(cert_id)

    def covers(self, cert_id: str, plan_hash: str) -> bool:
        """该证书是否确实覆盖这个最终动作。"""
        cert = self._certs.get(cert_id)
        return cert is not None and cert.plan_hash == plan_hash

    def __len__(self) -> int:
        return len(self._certs)
