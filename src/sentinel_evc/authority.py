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

# 资源保护上限，不代表真实工位允许这么久的盲执行
MAX_TTL_NS = 10_000_000_000  # 10 s
MAX_PREFIX_LEN = 8

# 起点允许偏差管（米）
START_TUBE = 0.01

# 观测最大年龄
MAX_OBS_AGE_NS = 200_000_000  # 200 ms


class Authority:
    def __init__(self, store: CertificateStore, key: Optional[bytes] = None):
        self._store = store
        self._key = key or os.urandom(32)
        self._issued: dict = {}
        self._consumed: set = set()

    # ------------------------------------------------------------ 签名

    def _mac(self, lease: Lease) -> str:
        return hmac.new(self._key, lease.signing_bytes(), sha256).hexdigest()

    def verify_mac(self, lease: Lease) -> bool:
        return hmac.compare_digest(self._mac(lease), lease.hmac)

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
        if prefix_len < 1 or prefix_len > MAX_PREFIX_LEN:
            raise Rejection(ErrorCode.AUTH_FAILED, f"prefix_len={prefix_len}")
        if prefix_len > plan.horizon:
            raise Rejection(ErrorCode.AUTH_FAILED, "前缀长于计划本身")
        if ttl_ns > MAX_TTL_NS:
            raise Rejection(ErrorCode.AUTH_FAILED, "ttl 超过资源保护上限")

        # 证书必须来自本地登记表，且确实覆盖这个最终动作
        if not self._store.covers(cert.cert_id, plan.hash):
            raise Rejection(ErrorCode.AUTH_FAILED, "证书未覆盖该最终动作")

        age = now_ns - snapshot.observed_mono
        if age > MAX_OBS_AGE_NS:
            raise Rejection(ErrorCode.STATE_STALE, f"观测年龄 {age}ns")

        # 起点必须落在允许管内
        import math

        if math.dist(snapshot.position, plan.points[0]) > START_TUBE:
            raise Rejection(ErrorCode.TRACKING_TUBE, "起点偏离允许管")

        # 期限要覆盖整个前缀的预计执行时间
        prefix_ns = int(prefix_len * plan.dt * 1e9)
        if ttl_ns < prefix_ns:
            raise Rejection(ErrorCode.AUTH_FAILED, "期限不足以跑完前缀")

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
        """一次性消费。同一许可第二次使用必须失败。"""
        if lease.lease_id not in self._issued:
            raise Rejection(ErrorCode.AUTH_FAILED, lease.lease_id)
        if not self.verify_mac(lease):
            raise Rejection(ErrorCode.AUTH_FAILED, "签名不匹配，疑似伪造或篡改")
        if lease.lease_id in self._consumed:
            raise Rejection(ErrorCode.LEASE_REPLAY, lease.lease_id)
        self._consumed.add(lease.lease_id)

    def is_consumed(self, lease_id: str) -> bool:
        return lease_id in self._consumed
