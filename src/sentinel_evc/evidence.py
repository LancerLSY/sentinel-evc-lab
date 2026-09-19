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
from typing import Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_bundle(log, out_dir: str) -> dict:
    """把一次运行的事件导出成可独立校验的包。"""
    from .contracts import canonical_json

    out = Path(out_dir)
    bundle = out / "bundle"
    anchors = out / "anchors"
    bundle.mkdir(parents=True)
    anchors.mkdir(parents=True)

    events_path = bundle / "events.jsonl"
    written = log.write_jsonl(events_path)
    if written == 0 or written != log.count:
        raise ValueError("证据包必须包含非空完整事件流，不能只打包 drain 后的残余队列")

    manifest = {
        "run_id": log.run_id,
        "event_count": log.count,
        "tip_hash": log.tip_hash,
        "files": {"events.jsonl": _sha256_file(events_path)},
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


LAYERS = ("files_present", "run_id", "hash_chain", "event_count",
          "tip_hash", "file_digest", "signature")


def _strict_json(raw: bytes):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("重复 JSON 键")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"非有限 JSON 数值: {value}")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                       parse_constant=reject_constant)
    # 1e999 以及嵌套孤立 surrogate 也必须拒绝。
    _canonical_json(value)
    return value


def verify_layers(bundle_dir: str, public_key_path: str, run_id: str) -> dict:
    """按冻结顺序验证，并保留已有七层返回结构与各层失败原因。"""
    bundle = Path(bundle_dir)
    result = {key: None for key in LAYERS}
    names = ("events.jsonl", "manifest.json", "manifest.sig")
    try:
        contents = {name: (bundle / name).read_bytes() for name in names}
    except OSError as exc:
        result["files_present"] = f"MISSING_FILE: {exc}"
        return result
    result["files_present"] = True
    manifest_bytes = contents["manifest.json"]
    try:
        manifest = _strict_json(manifest_bytes)
        if not isinstance(manifest, dict) or set(manifest) != {
            "run_id", "event_count", "tip_hash", "files"
        }:
            raise ValueError("manifest 字段不符合合同")
    except (ValueError, TypeError, UnicodeError) as exc:
        result["run_id"] = f"MANIFEST_UNREADABLE: {exc}"
        return result

    result["run_id"] = True if manifest["run_id"] == run_id else "RUN_ID_MISMATCH: 清单 run_id 不符"
    raw = contents["events.jsonl"]
    # 只按原始 LF 分帧；禁止 universal-newline 修复 CRLF 或忽略空行。
    lines = raw.split(b"\n")[:-1] if raw.endswith(b"\n") else raw.split(b"\n")
    framing_ok = bool(raw) and raw.endswith(b"\n") and b"\r" not in raw and all(lines)
    events = []
    parse_error = None
    for index, line in enumerate(lines):
        try:
            event = _strict_json(line)
            if not isinstance(event, dict):
                raise ValueError("事件必须是对象")
            events.append(event)
            if event.get("run_id") != run_id:
                result["run_id"] = f"RUN_ID_MISMATCH: 第 {index} 条事件 run_id 不符"
        except (ValueError, TypeError, UnicodeError) as exc:
            parse_error = f"EVENT_UNREADABLE: 第 {index} 条: {exc}"
            break

    previous = "sha256:" + "0" * 64
    previous_seq = -1
    if not framing_ok:
        result["hash_chain"] = "EVENT_FRAMING: 必须是非空、以 LF 结尾的规范 JSONL"
    elif parse_error:
        result["hash_chain"] = parse_error
    elif _canonical_json(manifest) != manifest_bytes:
        result["hash_chain"] = "MANIFEST_NONCANONICAL: 清单不是规范 JSON"
    else:
        result["hash_chain"] = True
        for index, (line, event) in enumerate(zip(lines, events)):
            if event.get("prev_hash") != previous:
                result["hash_chain"] = f"CHAIN_BROKEN: 哈希链在第 {index} 条断裂"
                break
            if _canonical_json(event) != line:
                result["hash_chain"] = f"EVENT_NONCANONICAL: 第 {index} 条不是规范 JSON"
                break
            seq = event.get("seq")
            if type(seq) is not int or seq < 0:
                result["hash_chain"] = f"EVENT_SEQUENCE: 第 {index} 条序号无效"
                break
            if event.get("type") == "LOG_GAP":
                gap = event.get("payload")
                valid_seq = (
                    isinstance(gap, dict)
                    and all(type(gap.get(key)) is int for key in (
                        "first_dropped_seq", "last_dropped_seq", "dropped_count"
                    ))
                    and gap["first_dropped_seq"] == previous_seq + 1
                    and gap["last_dropped_seq"] == seq - 1
                    and gap["dropped_count"] == seq - previous_seq - 1
                    and gap["dropped_count"] > 0
                )
            else:
                valid_seq = seq == previous_seq + 1
            if not valid_seq:
                result["hash_chain"] = f"EVENT_SEQUENCE: 第 {index} 条缺口与 LOG_GAP 不符"
                break
            previous = "sha256:" + hashlib.sha256(line).hexdigest()
            previous_seq = seq

    count = manifest["event_count"]
    result["event_count"] = True if type(count) is int and count > 0 and count == len(lines) else "EVENT_COUNT_MISMATCH: 事件数与非空清单声明不符"
    result["tip_hash"] = True if previous == manifest["tip_hash"] else "TIP_HASH_MISMATCH: 末尾摘要与清单不符"
    files = manifest["files"]
    actual_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    result["file_digest"] = True if files == {"events.jsonl": actual_digest} else "FILE_DIGEST_MISMATCH: events.jsonl 原始字节摘要或文件集合不符"

    try:
        pub = Ed25519PublicKey.from_public_bytes(Path(public_key_path).read_bytes())
        pub.verify(contents["manifest.sig"], manifest_bytes)
        result["signature"] = True
    except InvalidSignature:
        result["signature"] = "SIGNATURE_MISMATCH: 签名校验失败"
    except (OSError, ValueError) as exc:
        result["signature"] = f"BAD_PUBLIC_KEY: {exc}"
    return result


def verify_bundle(bundle_dir: str, public_key_path: str, run_id: str) -> Tuple[bool, str]:
    """独立校验。只接受三个参数，只读文件，自己重算一切。

    返回 (是否通过, 说明)。通过要求每一层都过。
    """
    r = verify_layers(bundle_dir, public_key_path, run_id)
    failed = [v for v in r.values() if v is not None and v is not True]
    if not failed and all(v is True for v in r.values()):
        n = Path(bundle_dir, "events.jsonl").read_bytes().count(b"\n")
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
