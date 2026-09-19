"""证据包：哈希链 + 清单 + Ed25519 签名。

签名只能证明**相对于指定信任源的记录完整性**。它不证明传感器诚实、
不证明动作物理上发生过、不判断事故责任。演示公钥不是客户 PKI。

verify_bundle 刻意写成不依赖本模块任何其它函数的独立实现：
它只读文件、自己算哈希。校验器复用写入方的代码就证明不了什么。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

if TYPE_CHECKING:
    from .events import EventLog


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_bundle(log: EventLog, out_dir: str) -> dict:
    """把一次运行的事件导出成可独立校验的包。"""
    from .contracts import canonical_json

    out = Path(out_dir)
    bundle = out / "bundle"
    anchors = out / "anchors"
    bundle.mkdir(parents=True, exist_ok=True)
    anchors.mkdir(parents=True, exist_ok=True)

    events_path = bundle / "events.jsonl"
    events_path.write_bytes(log.to_jsonl())

    manifest = {
        "run_id": log.run_id,
        "event_count": log.count,
        "tip_hash": log.tip_hash,
        "files": {"events.jsonl": _sha256_file(events_path)},
        "signer": "sentinel-evc-lab demo key (NOT a customer PKI)",
    }
    manifest_bytes = canonical_json(manifest)
    (bundle / "manifest.json").write_bytes(manifest_bytes)

    # Evidence 用 Ed25519，与 Authority 的 HMAC 密钥完全分开
    private_key = Ed25519PrivateKey.generate()
    signature = private_key.sign(manifest_bytes)
    (bundle / "manifest.sig").write_bytes(signature)

    pub_path = anchors / "demo.public"
    pub_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )

    return {
        "bundle_dir": str(bundle),
        "public_key": str(pub_path),
        "event_count": log.count,
        "tip_hash": log.tip_hash,
    }


# ---------------------------------------------------------------- 独立校验器


LAYERS = ("files_present", "signature", "run_id", "file_digest",
          "event_count", "hash_chain", "tip_hash")


def verify_layers(bundle_dir: str, public_key_path: str, run_id: str) -> dict:
    """逐层校验，**不短路**。

    每一层独立判定并各自给出原因。不短路是刻意的：不同的篡改方式会留下
    不同的失败组合（改字节 → 摘要+链断；删尾 → 摘要+条数不符），
    只报第一个失败层会把这两种情况混成一样的输出。
    """
    bundle = Path(bundle_dir)
    events_path = bundle / "events.jsonl"
    manifest_path = bundle / "manifest.json"
    sig_path = bundle / "manifest.sig"
    r = {k: None for k in LAYERS}

    missing = [p.name for p in (events_path, manifest_path, sig_path)
               if not p.exists()]
    if missing:
        r["files_present"] = f"MISSING_FILE: {', '.join(missing)}"
        return r
    r["files_present"] = True

    # --- 签名
    manifest_bytes = manifest_path.read_bytes()
    try:
        pub = Ed25519PublicKey.from_public_bytes(Path(public_key_path).read_bytes())
        pub.verify(sig_path.read_bytes(), manifest_bytes)
        r["signature"] = True
    except InvalidSignature:
        r["signature"] = "SIGNATURE_MISMATCH: 签名校验失败（公钥不对或清单被改）"
    except Exception as exc:
        r["signature"] = f"BAD_PUBLIC_KEY: {exc}"

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        r["run_id"] = f"MANIFEST_UNREADABLE: {exc}"
        return r

    # --- run_id
    r["run_id"] = True if manifest.get("run_id") == run_id else (
        f"RUN_ID_MISMATCH: 清单里是 {manifest.get('run_id')}，要求的是 {run_id}")

    # --- 文件摘要
    declared = manifest.get("files", {}).get("events.jsonl")
    r["file_digest"] = True if declared == _sha256_file(events_path) else (
        "FILE_DIGEST_MISMATCH: events.jsonl 内容与清单摘要不符")

    # --- 条数
    lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln]
    declared_n = manifest.get("event_count")
    r["event_count"] = True if len(lines) == declared_n else (
        f"EVENT_COUNT_MISMATCH: 文件里 {len(lines)} 条，清单声明 {declared_n} 条")

    # --- 哈希链逐条重算
    prev = "sha256:" + "0" * 64
    chain_ok = True
    for i, line in enumerate(lines):
        try:
            ev = json.loads(line)
        except Exception:
            r["hash_chain"] = f"EVENT_UNREADABLE: 第 {i} 条无法解析"
            chain_ok = False
            break
        if ev.get("prev_hash") != prev:
            r["hash_chain"] = f"CHAIN_BROKEN: 哈希链在第 {i} 条断裂"
            chain_ok = False
            break
        try:
            canonical = _canonical_json(ev)
        except (TypeError, ValueError) as exc:
            r["hash_chain"] = f"EVENT_UNREADABLE: 第 {i} 条不符合规范字节规则: {exc}"
            chain_ok = False
            break
        if canonical != line.encode("utf-8"):
            r["hash_chain"] = f"EVENT_NONCANONICAL: 第 {i} 条不是规范 JSON"
            chain_ok = False
            break
        prev = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if chain_ok:
        r["hash_chain"] = True

    r["tip_hash"] = True if prev == manifest.get("tip_hash") else (
        "TIP_HASH_MISMATCH: 末尾摘要与清单不符")
    return r


def verify_bundle(bundle_dir: str, public_key_path: str, run_id: str) -> Tuple[bool, str]:
    """独立校验。只接受三个参数，只读文件，自己重算一切。

    返回 (是否通过, 说明)。通过要求每一层都过。
    """
    r = verify_layers(bundle_dir, public_key_path, run_id)
    failed = [v for v in r.values() if v is not None and v is not True]
    if not failed and all(v is True for v in r.values()):
        n = len(Path(bundle_dir, "events.jsonl")
                .read_text(encoding="utf-8").splitlines())
        return True, f"OK: {n} 条事件，{len(LAYERS)} 层校验全部通过"
    return False, " | ".join(failed)


def failed_layers(bundle_dir: str, public_key_path: str, run_id: str) -> tuple:
    """返回失败的层名，用于区分不同的篡改方式。"""
    r = verify_layers(bundle_dir, public_key_path, run_id)
    return tuple(k for k in LAYERS if r[k] is not None and r[k] is not True)


def _encode(obj) -> str:
    """独立实现规范 JSON，verify 不复用生产侧协议代码。"""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("浮点必须有限")
        return format(0.0 if obj == 0.0 else obj, ".16e")
    if isinstance(obj, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in obj):
            raise ValueError("字符串包含孤立 surrogate")
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return "[" + ",".join(_encode(value) for value in obj) + "]"
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj):
            raise TypeError("对象键必须是字符串")
        return "{" + ",".join(
            _encode(key) + ":" + _encode(obj[key]) for key in sorted(obj)
        ) + "}"
    raise TypeError(f"不支持的 JSON 类型: {type(obj).__name__}")


def _canonical_json(obj) -> bytes:
    return _encode(obj).encode("utf-8")
