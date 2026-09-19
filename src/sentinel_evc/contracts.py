"""不可变契约对象与规范化哈希。

本模块是整个系统的信任根基：证书、许可、证据都引用这里算出的哈希。
一旦某个对象被哈希过，它就不能再被修改 —— 所有类都是 frozen dataclass。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------- 规范化序列化

FLOAT_FMT = "%.12g"


def _canon(obj: Any) -> Any:
    """把对象递归转成可确定性序列化的形式。

    浮点数统一格式化到 12 位有效数字，避免不同平台 repr 差异导致哈希不一致。
    """
    if isinstance(obj, float):
        return float(FLOAT_FMT % obj)
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    return obj


def canonical_json(obj: Any) -> bytes:
    """确定性 JSON 序列化。

    注意：这是本项目自己的规范化实现，**不是** RFC 8785 的完整实现。
    不要在任何对外材料里声称实现了 RFC 8785。
    """
    return json.dumps(
        _canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


# ---------------------------------------------------------------- 动作描述器


@dataclass(frozen=True)
class ActionDescriptor:
    """动作语义的显式声明。

    7 维数组不自动等于 7 关节速度 —— 单位、坐标系、模式必须写下来。
    数值参考域里固定为世界系 3D 绝对位置。
    """

    descriptor_id: str = "numeric-world-xyz-v1"
    mode: str = "absolute_position"
    units: str = "m"
    frame: str = "world"
    rotation: Optional[str] = None
    gripper: str = "discrete_event"

    def summary(self) -> dict:
        return {
            "descriptor_id": self.descriptor_id,
            "mode": self.mode,
            "units": self.units,
            "frame": self.frame,
            "rotation": self.rotation,
            "gripper": self.gripper,
        }


DEFAULT_DESCRIPTOR = ActionDescriptor()


# ---------------------------------------------------------------- 场景与计划


@dataclass(frozen=True)
class Sphere:
    center: tuple
    radius: float

    def summary(self) -> dict:
        return {"type": "sphere", "center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class Scene:
    """静态场景：球障碍 + 盒工作空间 + 工具半径 + 跟踪预留。"""

    scene_id: str
    obstacles: tuple  # tuple[Sphere, ...]
    ws_lo: tuple
    ws_hi: tuple
    tool_radius: float = 0.02
    tracking_reserve: float = 0.005

    def summary(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "obstacles": [o.summary() for o in self.obstacles],
            "workspace": {"lo": list(self.ws_lo), "hi": list(self.ws_hi)},
            "tool_radius": self.tool_radius,
            "tracking_reserve": self.tracking_reserve,
        }

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary())


@dataclass(frozen=True)
class Plan:
    """一个最终候选动作：H+1 个三维位置节点 + 均匀 dt + 夹爪事件。

    knots 是 tuple of tuple，不可变。改一个点就是一个新 Plan、新哈希。
    """

    plan_id: str
    knots: tuple
    dt: float
    gripper_events: tuple = ()
    descriptor: ActionDescriptor = field(default=DEFAULT_DESCRIPTOR)

    def __post_init__(self):
        if len(self.knots) < 2:
            raise ValueError("Plan 至少需要 2 个节点")

    @property
    def horizon(self) -> int:
        """段数 H（节点数为 H+1）。"""
        return len(self.knots) - 1

    def summary(self) -> dict:
        return {
            "knots": [list(k) for k in self.knots],
            "dt": self.dt,
            "gripper_events": [list(e) for e in self.gripper_events],
            "descriptor": self.descriptor.summary(),
        }

    @property
    def hash(self) -> str:
        """最终动作摘要。证书和许可都绑定这个值。"""
        return sha256_hex(self.summary())

    def prefix(self, n: int) -> tuple:
        """前 n 段对应的节点。"""
        return self.knots[: n + 1]


# ---------------------------------------------------------------- 变换记录


@dataclass(frozen=True)
class TransformRecord:
    """一次动作变换的登记。

    变换登记器不是让上游声明「这个变换安全」就信任 —— 每种允许继承的
    变换类型必须有代码、测试和适用条件。未知类型强制完整重验。
    """

    transform_id: str
    kind: str  # "mix" | "perturb" | "identity" | "unknown"
    parent_hash: str
    child_hash: str
    parameters: dict = field(default_factory=dict)
    version: str = "v1"

    # 只有列在这里的变换类型才允许尝试 Δ-Cert 继承
    INHERITABLE = frozenset({"identity", "perturb"})

    @property
    def inheritable(self) -> bool:
        return self.kind in self.INHERITABLE

    def summary(self) -> dict:
        return {
            "transform_id": self.transform_id,
            "kind": self.kind,
            "parent_hash": self.parent_hash,
            "child_hash": self.child_hash,
            "parameters": self.parameters,
            "version": self.version,
        }


# ---------------------------------------------------------------- 上下文与快照


@dataclass(frozen=True)
class Context:
    """执行上下文版本。任何一项变化都可能使已有证书或许可失效。"""

    robot: str = "numeric-robot-0"
    boot: int = 1
    epoch: int = 0
    scene_id: str = "scene-000"
    controller: str = "sim-controller-v1"
    task_phase: str = "transfer"
    queue_rev: int = 0
    committed_prefix_hash: str = "sha256:" + "0" * 64

    def summary(self) -> dict:
        return {
            "robot": self.robot,
            "boot": self.boot,
            "epoch": self.epoch,
            "scene_id": self.scene_id,
            "controller": self.controller,
            "task_phase": self.task_phase,
            "queue_rev": self.queue_rev,
            "committed_prefix_hash": self.committed_prefix_hash,
        }

    def bumped_epoch(self) -> "Context":
        return Context(
            robot=self.robot,
            boot=self.boot,
            epoch=self.epoch + 1,
            scene_id=self.scene_id,
            controller=self.controller,
            task_phase=self.task_phase,
            queue_rev=self.queue_rev,
            committed_prefix_hash=self.committed_prefix_hash,
        )


@dataclass(frozen=True)
class Snapshot:
    """状态快照。不可变，有唯一 obs_id。"""

    obs_id: str
    position: tuple
    capture_mono_ns: int
    valid: bool = True

    def summary(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "position": list(self.position),
            "capture_mono_ns": self.capture_mono_ns,
            "valid": self.valid,
        }


# ---------------------------------------------------------------- 证书与许可


@dataclass(frozen=True)
class Certificate:
    """对特定 Plan、场景、profile 的约束证明记录。

    margins[k] = 第 k 段的**剩余**最小余量（米）。继承时从剩余量扣减，
    不是每次回到最初值 —— 这是最容易写错、后果最严重的地方。
    """

    cert_id: str
    plan_hash: str
    scene_hash: str
    dt: float
    horizon: int
    margins: tuple
    method: str  # "FULL" | "INHERITED"
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    depth: int = 0
    proof_scope: str = "numeric-sphere-box-L1-v1"

    def summary(self) -> dict:
        return {
            "cert_id": self.cert_id,
            "plan_hash": self.plan_hash,
            "scene_hash": self.scene_hash,
            "dt": self.dt,
            "horizon": self.horizon,
            "margins": list(self.margins),
            "method": self.method,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "depth": self.depth,
            "proof_scope": self.proof_scope,
        }


@dataclass(frozen=True)
class Lease:
    """本地执行许可：有限、一次性、有到期时间、可撤销。

    有证书不代表现在可以发送 —— Lease 才是执行器唯一认的凭据。
    """

    lease_id: str
    plan_hash: str
    context: Context
    cert_id: str
    prefix_len: int
    deadline_mono_ns: int
    mac: str = ""

    def payload(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "plan_hash": self.plan_hash,
            "context": self.context.summary(),
            "cert_id": self.cert_id,
            "prefix_len": self.prefix_len,
            "deadline_mono_ns": self.deadline_mono_ns,
        }


# ---------------------------------------------------------------- 错误码


class ErrorCode:
    """结构化拒绝原因。

    这些不能统一返回一个无说明的 None —— 不同原因对应不同的恢复前提。
    """

    INPUT_SCHEMA = "INPUT_SCHEMA"
    STATE_STALE = "STATE_STALE"
    CONTEXT_CHANGED = "CONTEXT_CHANGED"
    CERTIFICATE_MISS = "CERTIFICATE_MISS"
    GEOMETRY_VIOLATION = "GEOMETRY_VIOLATION"
    MODEL_UNKNOWN = "MODEL_UNKNOWN"
    LEASE_REPLAY = "LEASE_REPLAY"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    LEASE_UNKNOWN = "LEASE_UNKNOWN"
    TRACKING_TUBE = "TRACKING_TUBE"
    CANCEL_UNCONFIRMED = "CANCEL_UNCONFIRMED"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    STALE_GENERATION = "STALE_GENERATION"
    CONTROLLER_FULL = "CONTROLLER_FULL"


class Rejection(Exception):
    """带错误码的拒绝。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
