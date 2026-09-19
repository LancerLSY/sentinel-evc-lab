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
import math
import os
import threading
from dataclasses import replace
from hashlib import sha256
from typing import Optional

from .contracts import (
    Certificate,
    ErrorCode,
    Lease,
    LeaseContext,
    Rejection,
    Snapshot,
)
from .delta_cert import CertificateStore

_lease_counter = itertools.count(1)

# 起点允许偏差管（米）
START_TUBE = 0.005

# 观测最大年龄
MAX_OBS_AGE_NS = 100_000_000  # 100 ms


class Authority:
    def __init__(self, store: CertificateStore, key: Optional[bytes] = None):
        self._store = store
        self._key = key or os.urandom(32)
        self._issued: dict = {}
        self._consumed: set = set()
        self._consume_lock = threading.Lock()

    # ------------------------------------------------------------ 签名

    def _mac(self, lease: Lease) -> str:
        return hmac.new(self._key, lease.signing_bytes(), sha256).hexdigest()

    def verify_mac(self, lease: Lease) -> bool:
        try:
            return hmac.compare_digest(self._mac(lease), lease.hmac)
        except (TypeError, ValueError):
            # 非规范消息无法形成合法签名，同样属于认证失败。
            return False

    # ------------------------------------------------------------ Prepare

    def prepare(
        self,
        plan,
        cert: Certificate,
        context: LeaseContext,
        snapshot: Snapshot,
        now_ns: int,
        prefix_len: int = 4,
        ttl_ns: int = 500_000_000,
    ) -> Lease:
        """准备阶段：核对内容与范围，签发许可。**不向驱动发任何命令。**"""
        if type(prefix_len) is not int or not 1 <= prefix_len <= plan.horizon:
            raise Rejection(ErrorCode.AUTH_FAILED, f"prefix_len={prefix_len}")
        if not math.isfinite(ttl_ns) or ttl_ns < prefix_len * plan.dt * 1e9:
            raise Rejection(ErrorCode.AUTH_FAILED, "期限不足以跑完前缀")

        # 证书必须来自本地登记表，且确实覆盖这个最终动作
        local_cert = self._store.get(cert.cert_id)
        if local_cert is None or local_cert.plan_hash != plan.hash:
            raise Rejection(ErrorCode.AUTH_FAILED, "证书未覆盖该最终动作")
        if local_cert.scene_id != context.scene_id or snapshot.context != context:
            raise Rejection(ErrorCode.CONTEXT_CHANGED, "证书或观测的执行上下文不匹配")

        age = now_ns - snapshot.observed_mono
        if not 0 <= age <= MAX_OBS_AGE_NS:
            raise Rejection(ErrorCode.STATE_STALE, f"观测年龄 {age}ns")

        # 起点必须落在允许管内
        if math.dist(snapshot.position, plan.points[0]) > START_TUBE:
            raise Rejection(ErrorCode.TRACKING_TUBE, "起点偏离允许管")

        lease = Lease(
            lease_id=f"lease-{next(_lease_counter):06d}",
            final_hash=plan.hash,
            context=context,
            cert_id=cert.cert_id,
            prefix_len=prefix_len,
            deadline_mono=now_ns + ttl_ns,
        )
        lease = replace(lease, hmac=self._mac(lease))
        self._issued[lease.lease_id] = lease
        return lease

    # ------------------------------------------------------------ 消费

    def consume(self, lease: Lease) -> None:
        """由 Commit 在状态复核完成后调用；共享 Authority 时仍只能消费一次。"""
        with self._consume_lock:
            if lease.lease_id not in self._issued:
                raise Rejection(ErrorCode.AUTH_FAILED, lease.lease_id)
            if not self.verify_mac(lease):
                raise Rejection(ErrorCode.AUTH_FAILED, "签名不匹配，疑似伪造或篡改")
            if lease.lease_id in self._consumed:
                raise Rejection(ErrorCode.LEASE_REPLAY, lease.lease_id)
            self._consumed.add(lease.lease_id)

    def is_consumed(self, lease_id: str) -> bool:
        return lease_id in self._consumed
