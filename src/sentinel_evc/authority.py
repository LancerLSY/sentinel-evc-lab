"""执行许可签发方。

信任边界：许可由本地 Authority 签发，上游模型进程不持有签发密钥。
这里用 HMAC 防止消息误用和引用伪造 —— 它**不能**抵御已经控制同一
Python 进程内存的攻击者。生产环境需要进程、权限和密钥隔离。

Authority 的 HMAC 密钥与 Evidence 的 Ed25519 密钥用途不同，
必须分开，不共享。
"""

from __future__ import annotations

import hmac
import itertools
import os
from hashlib import sha256
from typing import Optional

from .contracts import (
    Certificate,
    Context,
    ErrorCode,
    Lease,
    Rejection,
    Snapshot,
    canonical_json,
)
from .delta_cert import CertificateStore

_lease_counter = itertools.count(1)

# 资源保护上限，不代表真实工位允许这么久的盲执行
MAX_TTL_NS = 10_000_000_000  # 10 s
MAX_PREFIX_LEN = 8

# 起点允许偏差管（米）
START_TUBE = 0.005

# 观测最大年龄
MAX_OBS_AGE_NS = 200_000_000  # 200 ms


class Authority:
    def __init__(
        self,
        store: CertificateStore,
        key: Optional[bytes] = None,
        events=None,
        max_predictions: int = 1024,
        max_leases: int = 100_000,
    ):
        if not isinstance(store, CertificateStore):
            raise TypeError("store 必须是 CertificateStore")
        if not isinstance(max_predictions, int) or isinstance(max_predictions, bool) or max_predictions < 1:
            raise ValueError("max_predictions 必须是正整数")
        if not isinstance(max_leases, int) or isinstance(max_leases, bool) or max_leases < 1:
            raise ValueError("max_leases 必须是正整数")
        self._store = store
        self._key = key or os.urandom(32)
        self._issued: dict = {}
        self._consumed: set = set()
        self._events = events
        self._generation = 0
        self._predictions: dict = {}
        self._max_predictions = max_predictions
        self._max_leases = max_leases

    @property
    def generation(self) -> int:
        return self._generation

    def _check_evidence(self) -> None:
        if self._events is not None and self._events.has_gap:
            raise Rejection(ErrorCode.EVIDENCE_GAP, "事件链已有 LOG_GAP")

    def register_prediction(self, prediction) -> None:
        """Register a locally verified prediction without importing its module."""
        try:
            prediction_hash = prediction.hash
            plan_hash = prediction.plan_hash
            deadline = prediction.deadline_mono_ns
            allowed = prediction.allowed
        except AttributeError as exc:
            raise TypeError("prediction 缺少核心绑定字段") from exc
        if not isinstance(prediction_hash, str) or not prediction_hash:
            raise ValueError("prediction.hash 无效")
        if not isinstance(plan_hash, str) or not plan_hash:
            raise ValueError("prediction.plan_hash 无效")
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 0:
            raise ValueError("prediction.deadline_mono_ns 无效")
        if type(allowed) is not bool:
            raise TypeError("prediction.allowed 必须是 bool")
        existing = self._predictions.get(prediction_hash)
        if existing is not None and existing is not prediction and (
            existing.plan_hash != plan_hash
            or existing.deadline_mono_ns != deadline
            or existing.allowed != allowed
        ):
            raise ValueError("prediction hash collision")
        if existing is None and len(self._predictions) >= self._max_predictions:
            raise ValueError("prediction registry capacity exceeded")
        self._predictions[prediction_hash] = prediction

    def advance_generation(self) -> int:
        """Invalidate every lease issued under the previous generation."""
        self._generation += 1
        return self._generation

    def validate_runtime(self, lease: Lease, plan_hash: str, now_ns: int) -> None:
        """Recheck revocation, evidence and learned-evidence validity at dispatch."""
        self._check_evidence()
        if lease.authority_generation != self._generation:
            raise Rejection(ErrorCode.STALE_GENERATION, "许可代次已失效")
        if lease.plan_hash != plan_hash:
            raise Rejection(ErrorCode.LEASE_UNKNOWN, "许可与当前计划不匹配")
        if now_ns > lease.deadline_mono_ns:
            raise Rejection(ErrorCode.LEASE_EXPIRED, "许可已过期")
        if lease.prediction_hash is not None:
            prediction = self._predictions.get(lease.prediction_hash)
            if prediction is None or prediction.plan_hash != plan_hash:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 登记或动作绑定失效")
            if prediction.allowed is not True or now_ns > prediction.deadline_mono_ns:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 已拒绝或过期")

    def validate_available(self, lease: Lease) -> None:
        """Check identity/replay/revocation without consuming a permit."""
        self._check_evidence()
        if lease.lease_id not in self._issued:
            raise Rejection(ErrorCode.LEASE_UNKNOWN, lease.lease_id)
        if not self.verify_mac(lease):
            raise Rejection(ErrorCode.LEASE_UNKNOWN, "签名不匹配，疑似伪造或篡改")
        if lease.lease_id in self._consumed:
            raise Rejection(ErrorCode.LEASE_REPLAY, lease.lease_id)
        if lease.authority_generation != self._generation:
            raise Rejection(ErrorCode.STALE_GENERATION, "许可代次已失效")

    # ------------------------------------------------------------ 签名

    def _mac(self, lease: Lease) -> str:
        return hmac.new(self._key, canonical_json(lease.payload()), sha256).hexdigest()

    def verify_mac(self, lease: Lease) -> bool:
        return hmac.compare_digest(self._mac(lease), lease.mac)

    # ------------------------------------------------------------ Prepare

    def prepare(
        self,
        plan,
        cert: Certificate,
        context: Context,
        snapshot: Snapshot,
        now_ns: int,
        prefix_len: int = 4,
        ttl_ns: int = 500_000_000,
        prediction=None,
        require_prediction: bool = False,
    ) -> Lease:
        """准备阶段：核对内容与范围，签发许可。**不向驱动发任何命令。**"""
        self._check_evidence()
        if prefix_len < 1 or prefix_len > MAX_PREFIX_LEN:
            raise Rejection(ErrorCode.INPUT_SCHEMA, f"prefix_len={prefix_len}")
        if prefix_len > plan.horizon:
            raise Rejection(ErrorCode.INPUT_SCHEMA, "前缀长于计划本身")
        if isinstance(ttl_ns, bool) or not isinstance(ttl_ns, int) or ttl_ns < 1 or ttl_ns > MAX_TTL_NS:
            raise Rejection(ErrorCode.INPUT_SCHEMA, "ttl 超过资源保护上限")
        if context.epoch != self._generation:
            raise Rejection(ErrorCode.STALE_GENERATION, "上下文 epoch 与签发代次不一致")
        if len(self._issued) >= self._max_leases:
            raise Rejection(ErrorCode.INPUT_SCHEMA, "lease registry capacity exceeded")
        if (
            self._events is not None
            and self._events.supports("LEASE")
            and not self._events.can_append_without_gap()
        ):
            self._events.note_gap("LEASE event would overflow bounded buffer")
            raise Rejection(ErrorCode.EVIDENCE_GAP, "签发事件无法完整记录")

        # 证书必须来自本地登记表，且确实覆盖这个最终动作
        if not self._store.covers(cert.cert_id, plan.hash):
            raise Rejection(ErrorCode.CERTIFICATE_MISS, "证书未覆盖该最终动作")

        if not snapshot.valid:
            raise Rejection(ErrorCode.STATE_STALE, "快照无效")
        age = now_ns - snapshot.capture_mono_ns
        if age < 0 or age > MAX_OBS_AGE_NS:
            raise Rejection(ErrorCode.STATE_STALE, f"观测年龄 {age}ns")

        # 起点必须落在允许管内
        import math

        if math.dist(snapshot.position, plan.knots[0]) > START_TUBE:
            raise Rejection(ErrorCode.TRACKING_TUBE, "起点偏离允许管")

        # 期限要覆盖整个前缀的预计执行时间
        prefix_ns = int(prefix_len * plan.dt * 1e9)
        if ttl_ns < prefix_ns:
            raise Rejection(ErrorCode.INPUT_SCHEMA, "期限不足以跑完前缀")

        prediction_hash = None
        if prediction is None and require_prediction:
            raise Rejection(ErrorCode.MODEL_UNKNOWN, "缺少必需 prediction")
        if prediction is not None:
            prediction_hash = getattr(prediction, "hash", None)
            registered = self._predictions.get(prediction_hash)
            if registered is None or registered is not prediction:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 未在本地登记")
            if prediction.plan_hash != plan.hash:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 未绑定最终动作")
            if prediction.allowed is not True:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 不允许执行")
            if prediction.deadline_mono_ns < now_ns + prefix_ns:
                raise Rejection(ErrorCode.MODEL_UNKNOWN, "prediction 已过期或不能覆盖前缀")

        lease = Lease(
            lease_id=f"lease-{next(_lease_counter):06d}",
            plan_hash=plan.hash,
            context=context,
            cert_id=cert.cert_id,
            prefix_len=prefix_len,
            deadline_mono_ns=now_ns + ttl_ns,
            prediction_hash=prediction_hash,
            authority_generation=self._generation,
        )
        lease = Lease(**{**lease.__dict__, "mac": self._mac(lease)})
        self._issued[lease.lease_id] = lease
        if self._events is not None and self._events.supports("LEASE"):
            self._events.append(
                "LEASE",
                lease_id=lease.lease_id,
                plan_hash=lease.plan_hash,
                cert_id=lease.cert_id,
                prediction_hash=lease.prediction_hash,
                prefix_len=lease.prefix_len,
                generation=lease.authority_generation,
            )
            self._check_evidence()
        return lease

    # ------------------------------------------------------------ 消费

    def consume(self, lease: Lease) -> None:
        """一次性消费。同一许可第二次使用必须失败。"""
        self.validate_available(lease)
        self._consumed.add(lease.lease_id)

    def is_consumed(self, lease_id: str) -> bool:
        return lease_id in self._consumed
