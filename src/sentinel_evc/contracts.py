"""不可变契约对象与规范化哈希。

本模块是整个系统的信任根基：证书、许可、证据都引用这里算出的哈希。
一旦某个对象被哈希过，它就不能再被修改 —— 所有类都是 frozen dataclass。
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
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
    if isinstance(obj, Mapping):
        return {str(k): _canon(v) for k, v in obj.items()}
    return obj


def canonical_json(obj: Any) -> bytes:
    """确定性 JSON 序列化。

    注意：这是本项目自己的规范化实现，**不是** RFC 8785 的完整实现。
    不要在任何对外材料里声称实现了 RFC 8785。
    """
    return json.dumps(
        _canon(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


# ---------------------------------------------------------------- 动作描述器


def _nonempty(value: Any, name: str, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ValueError(f"{name} 必须是 1..{max_len} 字符字符串")
    return value


def _digest(value: Any, name: str) -> str:
    value = _nonempty(value, name, 71)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{name} 必须是 sha256 摘要")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 sha256 摘要") from exc
    return value


def _finite_real(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} 必须是{'正' if positive else ''}有限数值")
    return result


def _vec3(value: Any, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} 必须是 3 维向量")
    return tuple(_finite_real(v, f"{name}[{i}]") for i, v in enumerate(value))


def _freeze(value: Any) -> Any:
    """Recursively copy mutable inputs before they enter a hashed DTO."""
    if isinstance(value, Mapping):
        if len(value) > 256 or any(not isinstance(k, str) or not k or len(k) > 128 for k in value):
            raise ValueError("映射键必须是最多 256 个有界非空字符串")
        frozen = {k: _freeze(v) for k, v in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        if len(value) > 4096:
            raise ValueError("序列超过不可变 DTO 上限")
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        if len(value) > 4096:
            raise ValueError("集合超过不可变 DTO 上限")
        return frozenset(_freeze(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("不可变 DTO 不接受 NaN/Inf")
    if isinstance(value, str) and len(value) > 4096:
        raise ValueError("字符串超过不可变 DTO 上限")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError("不可变 DTO 含不支持的值类型")
    return value


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

    def __post_init__(self):
        _nonempty(self.descriptor_id, "descriptor_id")
        _nonempty(self.mode, "mode")
        _nonempty(self.units, "units")
        _nonempty(self.frame, "frame")
        if self.rotation is not None:
            _nonempty(self.rotation, "rotation")
        _nonempty(self.gripper, "gripper")

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

    def __post_init__(self):
        object.__setattr__(self, "center", _vec3(self.center, "sphere.center"))
        object.__setattr__(self, "radius", _finite_real(self.radius, "sphere.radius", positive=True))

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

    def __post_init__(self):
        _nonempty(self.scene_id, "scene_id")
        obstacles = tuple(self.obstacles)
        if len(obstacles) > 1024 or any(not isinstance(o, Sphere) for o in obstacles):
            raise ValueError("obstacles 必须是最多 1024 个 Sphere")
        lo, hi = _vec3(self.ws_lo, "ws_lo"), _vec3(self.ws_hi, "ws_hi")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("工作空间下界必须严格小于上界")
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "ws_lo", lo)
        object.__setattr__(self, "ws_hi", hi)
        object.__setattr__(self, "tool_radius", _finite_real(self.tool_radius, "tool_radius"))
        object.__setattr__(self, "tracking_reserve", _finite_real(self.tracking_reserve, "tracking_reserve"))
        if self.tool_radius < 0.0 or self.tracking_reserve < 0.0:
            raise ValueError("工具半径与跟踪预留不能为负")

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
        return self.exact_hash

    @property
    def exact_hash(self) -> str:
        """Execution identity includes every float64 bit of the static scene.

        A new domain tag separates these identities from legacy rounded hashes.
        The public summary stays readable and existing signed archives stay valid.
        """
        cached = self.__dict__.get("_hash_cache")
        if cached is None:
            def exact(value):
                if isinstance(value, float):
                    return {"float64_le": struct.pack("<d", value).hex()}
                if isinstance(value, dict):
                    return {key: exact(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [exact(item) for item in value]
                return value
            cached = sha256_hex({"schema": "sentinel-scene-exact-v1", "scene": exact(self.summary())})
            object.__setattr__(self, "_hash_cache", cached)
        return cached

    @property
    def legacy_hash(self) -> str:
        """Rounded identity for reading historical records; never use for admission."""
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
        _nonempty(self.plan_id, "plan_id")
        knots = tuple(_vec3(k, f"knots[{i}]") for i, k in enumerate(self.knots))
        object.__setattr__(self, "knots", knots)
        if len(self.knots) < 2:
            raise ValueError("Plan 至少需要 2 个节点")
        if len(self.knots) > 100_001:
            raise ValueError("Plan 节点数量超过上限")
        object.__setattr__(self, "dt", _finite_real(self.dt, "dt", positive=True))
        if not isinstance(self.descriptor, ActionDescriptor):
            raise TypeError("descriptor 必须是 ActionDescriptor")
        events = []
        seen_steps = set()
        for i, event in enumerate(self.gripper_events):
            if not isinstance(event, (tuple, list)) or len(event) != 2:
                raise ValueError(f"gripper_events[{i}] 必须是 (step, event)")
            step, kind = event
            if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(knots) - 1:
                raise ValueError(f"gripper_events[{i}] step 越界")
            if kind not in ("open", "close"):
                raise ValueError(f"gripper_events[{i}] 类型无效")
            if step in seen_steps:
                raise ValueError("同一步只能有一个夹爪事件")
            seen_steps.add(step)
            events.append((step, kind))
        object.__setattr__(self, "gripper_events", tuple(events))

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
        """最终动作摘要（规范化，12 位有效数字）。证书和事件引用这个值。

        Plan 构造后深度不可变，所以摘要只算一次并缓存（v2 改进：避免继承路径反复重算）。
        """
        cached = self.__dict__.get("_hash_cache")
        if cached is None:
            cached = sha256_hex(self.summary())
            object.__setattr__(self, "_hash_cache", cached)
        return cached

    @property
    def exact_hash(self) -> str:
        """逐位精确摘要：节点与 dt 按 IEEE-754 float64 小端字节打包。

        规范化摘要在 12 位有效数字之后会碰撞；许可绑定与 R0 同一性复用用这个值，
        保证「被写入的就是被验证的」在字节层面成立。
        """
        cached = self.__dict__.get("_exact_hash_cache")
        if cached is None:
            h = hashlib.sha256(b"sentinel-plan-exact-v1\0")
            h.update(struct.pack("<Id", len(self.knots), self.dt))
            for knot in self.knots:
                h.update(struct.pack("<3d", *knot))
            h.update(canonical_json({"gripper_events": [list(e) for e in self.gripper_events],
                                     "descriptor": self.descriptor.summary()}))
            cached = "sha256:" + h.hexdigest()
            object.__setattr__(self, "_exact_hash_cache", cached)
        return cached

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

    def __post_init__(self):
        _nonempty(self.transform_id, "transform_id")
        _nonempty(self.kind, "kind")
        _digest(self.parent_hash, "parent_hash")
        _digest(self.child_hash, "child_hash")
        _nonempty(self.version, "version")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters 必须是映射")
        object.__setattr__(self, "parameters", _freeze(self.parameters))

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
    # v2：可选的场景内容摘要。设置后，证书的 scene_hash 必须与之相等才能签发 / 提交许可。
    scene_hash: Optional[str] = None

    def __post_init__(self):
        for field_name in ("robot", "scene_id", "controller", "task_phase"):
            _nonempty(getattr(self, field_name), field_name)
        _digest(self.committed_prefix_hash, "committed_prefix_hash")
        if self.scene_hash is not None:
            _digest(self.scene_hash, "scene_hash")
        for field_name in ("boot", "epoch", "queue_rev"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")

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
            # 只在设置时出现，保持旧上下文的摘要与 MAC 不变
            **({"scene_hash": self.scene_hash} if self.scene_hash is not None else {}),
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
            scene_hash=self.scene_hash,
        )


@dataclass(frozen=True)
class Snapshot:
    """状态快照。不可变，有唯一 obs_id。"""

    obs_id: str
    position: tuple
    capture_mono_ns: int
    valid: bool = True
    supported: Optional[bool] = None

    def __post_init__(self):
        _nonempty(self.obs_id, "obs_id")
        object.__setattr__(self, "position", _vec3(self.position, "snapshot.position"))
        if isinstance(self.capture_mono_ns, bool) or not isinstance(self.capture_mono_ns, int) or self.capture_mono_ns < 0:
            raise ValueError("capture_mono_ns 必须是非负整数")
        if type(self.valid) is not bool:
            raise TypeError("valid 必须是 bool")
        if self.supported is not None and type(self.supported) is not bool:
            raise TypeError("supported 必须是 bool 或 None")

    def summary(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "position": list(self.position),
            "capture_mono_ns": self.capture_mono_ns,
            "valid": self.valid,
            "supported": self.supported,
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
    # v2: IEEE-754 byte digest of the plan the margins were computed on (Plan.exact_hash).
    # When set, Authority.prepare only issues a lease for a plan with exactly these bytes.
    plan_exact: Optional[str] = None

    def __post_init__(self):
        for field_name in ("cert_id", "method", "proof_scope"):
            _nonempty(getattr(self, field_name), field_name)
        _digest(self.plan_hash, "plan_hash")
        _digest(self.scene_hash, "scene_hash")
        if self.plan_exact is not None:
            _digest(self.plan_exact, "plan_exact")
        object.__setattr__(self, "dt", _finite_real(self.dt, "certificate.dt", positive=True))
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("certificate.horizon 必须为正整数")
        margins = tuple(_finite_real(v, "certificate.margin") for v in self.margins)
        if len(margins) != self.horizon:
            raise ValueError("certificate.margins 长度必须等于 horizon")
        object.__setattr__(self, "margins", margins)
        if self.method not in ("FULL", "INHERITED"):
            raise ValueError("certificate.method 无效")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("certificate.depth 必须是非负整数")

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
            **({"plan_exact": self.plan_exact} if self.plan_exact is not None else {}),
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
    prediction_hash: Optional[str] = None
    authority_generation: int = 0
    mac: str = ""
    plan_exact: Optional[str] = None

    def __post_init__(self):
        for field_name in ("lease_id", "cert_id"):
            _nonempty(getattr(self, field_name), field_name)
        _digest(self.plan_hash, "plan_hash")
        if not isinstance(self.context, Context):
            raise TypeError("context 必须是 Context")
        if isinstance(self.prefix_len, bool) or not isinstance(self.prefix_len, int) or self.prefix_len < 1:
            raise ValueError("prefix_len 必须是正整数")
        for field_name in ("deadline_mono_ns", "authority_generation"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")
        if self.prediction_hash is not None:
            _digest(self.prediction_hash, "prediction_hash")
        if self.plan_exact is not None:
            _digest(self.plan_exact, "plan_exact")

    def payload(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "plan_hash": self.plan_hash,
            "context": self.context.summary(),
            "cert_id": self.cert_id,
            "prefix_len": self.prefix_len,
            "deadline_mono_ns": self.deadline_mono_ns,
            "prediction_hash": self.prediction_hash,
            "authority_generation": self.authority_generation,
            **({"plan_exact": self.plan_exact} if self.plan_exact is not None else {}),
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
