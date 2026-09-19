"""不可变领域合同与本 Demo 的规范字节；不是 RFC 8785。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

MAX_INHERIT_DEPTH = 4


def _encode(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("JSON 浮点必须有限")
        return format(0.0 if obj == 0.0 else obj, ".16e")
    if isinstance(obj, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in obj):
            raise ValueError("JSON 字符串不能包含孤立 surrogate")
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(value) for value in obj) + "]"
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj):
            raise TypeError("JSON 对象键必须是字符串")
        return "{" + ",".join(_encode(key) + ":" + _encode(obj[key]) for key in sorted(obj)) + "}"
    raise TypeError(f"不支持的 JSON 类型: {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    return _encode(obj).encode("utf-8")


def strict_json_loads(source: str | bytes) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 键: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"非法 JSON 常量: {value}")

    if isinstance(source, bytes):
        source = source.decode("utf-8")
    value = json.loads(source, object_pairs_hook=unique_object, parse_constant=reject_constant)
    # 同时拦截指数溢出和解码后的孤立 surrogate。
    canonical_json(value)
    return value


def sha256_hex(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def _point(values) -> tuple[float, float, float]:
    point = tuple(float(value) for value in values)
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError("坐标必须是三维有限数")
    return point


def _nonempty(*values):
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("ID 必须是非空字符串")


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", _point(self.center))

    def summary(self) -> dict:
        return {"type": "sphere", "center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class Scene:
    scene_id: str
    obstacles: tuple[Sphere, ...]
    ws_lo: tuple[float, float, float]
    ws_hi: tuple[float, float, float]
    tool_radius: float = 0.02
    tracking_reserve: float = 0.005

    def __post_init__(self):
        _nonempty(self.scene_id)
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "ws_lo", _point(self.ws_lo))
        object.__setattr__(self, "ws_hi", _point(self.ws_hi))

    def summary(self) -> dict:
        return {"scene_id": self.scene_id, "obstacles": [o.summary() for o in self.obstacles],
                "workspace": {"lo": list(self.ws_lo), "hi": list(self.ws_hi)},
                "tool_radius": self.tool_radius, "tracking_reserve": self.tracking_reserve}

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary())


@dataclass(frozen=True)
class Plan:
    points: tuple[tuple[float, float, float], ...]
    dt: float
    gripper_events: tuple[str, ...]
    controller_profile: str
    task_phase: str

    def __post_init__(self):
        object.__setattr__(self, "points", tuple(_point(point) for point in self.points))
        object.__setattr__(self, "gripper_events", tuple(self.gripper_events))
        object.__setattr__(self, "dt", float(self.dt))
        if len(self.points) < 2:
            raise ValueError("Plan 至少需要 2 个节点")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt 必须是正有限数")
        if any(not isinstance(event, str) for event in self.gripper_events):
            raise ValueError("gripper_events 必须包含字符串")
        if not isinstance(self.controller_profile, str) or not isinstance(self.task_phase, str):
            raise ValueError("controller_profile 和 task_phase 必须是字符串")

    @property
    def horizon(self) -> int:
        return len(self.points) - 1

    def summary(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary())

    def prefix(self, n: int) -> tuple:
        return self.points[:n + 1]


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class TransformRecord:
    transform_id: str
    kind: str
    parent_hash: str
    child_hash: str
    parameters: tuple = field(default_factory=tuple)
    version: str = "v1"

    INHERITABLE = frozenset({"identity", "perturb"})

    def __post_init__(self):
        _nonempty(self.transform_id)
        object.__setattr__(self, "parameters", _freeze(self.parameters))

    @property
    def inheritable(self) -> bool:
        return self.kind in self.INHERITABLE

    def summary(self) -> dict:
        return {"transform_id": self.transform_id, "kind": self.kind,
                "parent_hash": self.parent_hash, "child_hash": self.child_hash,
                "parameters": dict(self.parameters), "version": self.version}


@dataclass(frozen=True)
class LeaseContext:
    robot: str
    boot: int
    epoch: int
    scene_id: str
    queue_rev: int
    committed_prefix_hash: str

    def __post_init__(self):
        _nonempty(self.robot, self.scene_id)

    def summary(self) -> dict:
        return asdict(self)

    def bumped_epoch(self) -> LeaseContext:
        return replace(self, epoch=self.epoch + 1)


@dataclass(frozen=True)
class Snapshot:
    obs_id: str
    robot: str
    boot: int
    epoch: int
    scene_id: str
    queue_rev: int
    committed_prefix_hash: str
    position: tuple[float, float, float]
    observed_mono: float

    def __post_init__(self):
        _nonempty(self.obs_id, self.robot, self.scene_id)
        object.__setattr__(self, "position", _point(self.position))

    @property
    def context(self) -> LeaseContext:
        return LeaseContext(self.robot, self.boot, self.epoch, self.scene_id,
                            self.queue_rev, self.committed_prefix_hash)

    def summary(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Certificate:
    cert_id: str
    plan: Plan
    scene_id: str
    margins: tuple[float, ...]
    inherit_depth: int
    parent_cert_id: str | None

    def __post_init__(self):
        _nonempty(self.cert_id, self.scene_id)
        object.__setattr__(self, "margins", tuple(float(value) for value in self.margins))
        if len(self.margins) != self.plan.horizon:
            raise ValueError("余量数量必须与 Plan 段数一致")
        if type(self.inherit_depth) is not int or not 0 <= self.inherit_depth <= MAX_INHERIT_DEPTH:
            raise ValueError("继承深度必须在 0 到 4 之间")
        if (self.inherit_depth == 0) != (self.parent_cert_id is None):
            raise ValueError("完整检查无父证书；继承证书必须有父证书")
        if self.parent_cert_id is not None:
            _nonempty(self.parent_cert_id)

    @property
    def plan_hash(self) -> str:
        return self.plan.hash

    def summary(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Lease:
    lease_id: str
    final_hash: str
    context: LeaseContext
    cert_id: str
    prefix_len: int
    deadline_mono: float
    hmac: str = ""

    def __post_init__(self):
        _nonempty(self.lease_id, self.cert_id)

    def summary(self) -> dict:
        return asdict(self)

    def payload(self) -> dict:
        result = self.summary()
        del result["hmac"]
        return result

    def signing_bytes(self) -> bytes:
        return canonical_json(self.payload())


class ErrorCode:
    STATE_STALE = "STATE_STALE"
    CONTEXT_CHANGED = "CONTEXT_CHANGED"
    LEASE_REPLAY = "LEASE_REPLAY"
    TRACKING_TUBE = "TRACKING_TUBE"
    CANCEL_UNCONFIRMED = "CANCEL_UNCONFIRMED"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    AUTH_FAILED = "AUTH_FAILED"
    QUEUE_FULL = "QUEUE_FULL"


class Rejection(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
