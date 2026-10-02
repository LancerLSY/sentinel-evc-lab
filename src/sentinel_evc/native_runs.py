"""Verified native VLA recordings for the local workbench.

Importing a recording never executes a policy or grants device motion authority.
The selected public key is an integrity anchor, not proof of physical execution.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from .evidence import verify_bundle
from .scenario import strict_json

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
MEMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
MAX_ARCHIVE = 256 * 1024 * 1024
MAX_EXPANDED = 768 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MAX_MEMBERS = 160
UPLOAD_LIMIT = 64 * 1024 * 1024
SCHEMA = "sentinel-native-vla-run-v1"


def _numbers(value, width=None, maximum=4096):
    if not isinstance(value, list) or len(value) > maximum or (width is not None and len(value) != width):
        raise ValueError("回放数值数组的维度超出支持范围。")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not -1e300 <= item <= 1e300 or not math.isfinite(item) for item in value):
        raise ValueError("回放数组必须包含有限数值。")


def _validate_model(model):
    if not isinstance(model, dict):
        raise ValueError("任务模型必须是 JSON 对象。")
    if model.get("schema") == "sentinel-native-mujoco-model-index-v1":
        rows = model.get("models")
        if not isinstance(rows, list) or len(rows) > 64:
            raise ValueError("任务模型索引超出范围。")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("asset"), str) or not MEMBER.fullmatch(row["asset"]):
                raise ValueError("任务模型索引文件名不合法。")
        return
    bodies, meshes = model.get("bodies"), model.get("meshes")
    if not isinstance(bodies, list) or not 0 < len(bodies) <= 512:
        raise ValueError("任务模型 body 数量超出范围。")
    if not isinstance(meshes, list) or not 0 < len(meshes) <= 512:
        raise ValueError("任务模型 mesh 数量超出范围。")
    body_indices = [body.get("bodyIndex") if isinstance(body, dict) else None for body in bodies]
    if any(type(index) is not int for index in body_indices) or body_indices != list(range(len(bodies))):
        raise ValueError("任务模型必须使用连续且唯一的 bodyIndex。")
    vertices, indices = 0, 0
    for mesh in meshes:
        if not isinstance(mesh, dict) or type(mesh.get("bodyIndex")) is not int or mesh["bodyIndex"] not in range(len(bodies)):
            raise ValueError("网格引用了不支持的 bodyIndex。")
        points, normals, faces = mesh.get("positions"), mesh.get("normals"), mesh.get("indices")
        _numbers(points, maximum=3_000_000)
        _numbers(normals, width=len(points), maximum=3_000_000)
        if not points or len(points) % 3 or not isinstance(faces, list) or not faces or len(faces) % 3:
            raise ValueError("任务模型三角网格维度不合法。")
        vertices += len(points) // 3
        indices += len(faces)
        if vertices > 1_000_000 or indices > 4_000_000:
            raise ValueError("任务模型超过一百万顶点或四百万索引。")
        if any(type(index) is not int or not 0 <= index < len(points) // 3 for index in faces):
            raise ValueError("任务模型包含越界索引。")
        _numbers(mesh.get("color"), width=3)


def _validate_replay(replay, result):
    if not isinstance(replay, dict) or replay.get("schema") != "sentinel-native-vla-replay-v1" or replay.get("run_id") != result["run_id"]:
        raise ValueError("回放版本或运行编号不匹配。")
    episodes = replay.get("episodes")
    if not isinstance(episodes, list) or len(episodes) > 500:
        raise ValueError("回放 episode 数量超出范围。")
    result_ids = [row.get("episode_id") if isinstance(row, dict) else None for row in result["episodes"]]
    replay_ids = [row.get("episode_id") if isinstance(row, dict) else None for row in episodes]
    if any(not isinstance(key, str) or not 0 < len(key) <= 256 for key in replay_ids) or replay_ids != result_ids or len(set(replay_ids)) != len(replay_ids):
        raise ValueError("回放与结果的 episode 身份不一致。")
    total = 0
    for episode in episodes:
        frames = episode.get("frames")
        if not isinstance(frames, list) or len(frames) > 5000:
            raise ValueError("每个 episode 最多支持 5000 个回放帧。")
        total += len(frames)
        if total > 100_000:
            raise ValueError("回放总帧数超出十万帧。")
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("回放帧必须是对象。")
            action = frame.get("action")
            if action is not None:
                _numbers(action, width=7)
            for key, width in (("bodyWorldPosition", 3), ("bodyWorldRotation", 9)):
                rows = frame.get(key, [])
                if not isinstance(rows, list) or len(rows) > 512:
                    raise ValueError("回放 body 数量超出范围。")
                for row in rows:
                    _numbers(row, width=width)
            for key in ("qpos", "qvel"):
                if frame.get(key) is not None:
                    _numbers(frame[key], maximum=2048)


def _run_id(value: str) -> str:
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise ValueError("运行编号仅支持 1–96 个字母、数字、下划线或连字符。")
    return value


class NativeRunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("记录目录不能是符号链接。")
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, run_id: str) -> Path:
        path = self.root / _run_id(run_id)
        if path.is_symlink() or not path.is_dir():
            raise KeyError("native run not found")
        return path

    @staticmethod
    def _verify(directory: Path, run_id: str) -> dict:
        bundle, public_key = directory / "bundle", directory / "public.key"
        if bundle.is_symlink() or public_key.is_symlink():
            raise ValueError("记录核验失败：符号链接不受支持。")
        ok, message = verify_bundle(str(bundle), str(public_key), run_id)
        if not ok:
            raise ValueError("记录完整性核验失败：" + message)
        manifest = strict_json((bundle / "manifest.json").read_bytes())
        if not {"result.json", "replay.json"}.issubset(manifest["files"]):
            raise ValueError("原生运行必须签名覆盖 result.json 和 replay.json。")
        return manifest

    @staticmethod
    def _result(directory: Path, run_id: str) -> dict:
        source = directory / "bundle" / "result.json"
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("原生运行结果超出 8 MiB。")
        value = strict_json(source.read_bytes())
        if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("run_id") != run_id:
            raise ValueError("原生运行结果版本或运行编号不匹配。")
        if not isinstance(value.get("episodes"), list) or len(value["episodes"]) > 500:
            raise ValueError("原生运行 episode 列表不合法。")
        if not isinstance(value.get("metrics", {}), dict):
            raise ValueError("原生运行指标不合法。")
        if not isinstance(value.get("fault_attempts", []), list) or len(value.get("fault_attempts", [])) > 8000:
            raise ValueError("故障尝试数量超出范围。")
        # A verified demo key asserts file integrity only. The result's explicit
        # physical/model scope remains visible and is never promoted here.
        return {**value, "id": run_id, "episode_count": len(value["episodes"]),
                "verification": {"ok": True, "scope": "signed-record-integrity",
                                 "trust": "user-selected-public-key", "hardware_execution_proven": False}}

    def get(self, run_id: str) -> dict:
        directory = self._directory(run_id)
        self._verify(directory, run_id)
        return self._result(directory, run_id)

    def list(self) -> list[dict]:
        rows = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.name, reverse=True):
            if not RUN_ID.fullmatch(path.name) or path.is_symlink() or not path.is_dir():
                continue
            try:
                run = self.get(path.name)
                rows.append({key: run[key] for key in (
                    "id", "name", "mode", "status", "profile", "metrics", "episode_count", "verification"
                ) if key in run})
            except (OSError, ValueError, KeyError, RuntimeError):
                rows.append({"id": path.name, "name": path.name, "status": "failed",
                             "verification": {"ok": False}, "error": "记录完整性核验失败"})
        return rows

    def asset(self, run_id: str, name: str) -> bytes:
        if not isinstance(name, str) or not MEMBER.fullmatch(name) or name in {"manifest.json", "manifest.sig"}:
            raise KeyError("unsupported native asset")
        directory = self._directory(run_id)
        manifest = self._verify(directory, run_id)
        if name not in manifest["files"]:
            raise KeyError("asset not signed")
        path = directory / "bundle" / name
        if path.stat().st_size > MAX_FILE:
            raise ValueError("记录资产超出大小限制。")
        data = path.read_bytes()
        if name.startswith("viewer-model") and name.endswith(".json"):
            _validate_model(strict_json(data))
        return data

    def replay(self, run_id: str) -> dict:
        value = strict_json(self.asset(run_id, "replay.json"))
        _validate_replay(value, self.get(run_id))
        return value

    def import_archive(self, archive: bytes, run_id: str, public_key: bytes) -> dict:
        run_id = _run_id(run_id)
        if not isinstance(archive, bytes) or not 0 < len(archive) <= MAX_ARCHIVE:
            raise ValueError("归档为空或超出 256 MiB。")
        if not isinstance(public_key, bytes) or len(public_key) != 32:
            raise ValueError("需要所选 Ed25519 原始公钥（32 字节）。")
        target = self.root / run_id
        if target.exists() or target.is_symlink():
            raise RuntimeError("同编号记录已经存在；不会覆盖已有证据。")
        staging = Path(tempfile.mkdtemp(prefix=".native-import-", dir=self.root))
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as source:
                entries = source.infolist()
                if not 0 < len(entries) <= MAX_MEMBERS:
                    raise ValueError("归档成员数量超出限制。")
                total, names = 0, set()
                for entry in entries:
                    name = entry.filename
                    if entry.is_dir() and name in {"bundle/", "anchors/"}:
                        continue
                    if name in names:
                        raise ValueError("归档包含重复成员。")
                    names.add(name)
                    kind = stat.S_IFMT(entry.external_attr >> 16)
                    parts = name.split("/")
                    safe = (len(parts) == 2 and parts[0] == "bundle" and MEMBER.fullmatch(parts[1]))
                    safe = bool(safe or name == "anchors/demo.public")
                    if not safe or kind not in {0, stat.S_IFREG} or entry.flag_bits & 1:
                        raise ValueError("仅接受 bundle/ 平面文件和 anchors/demo.public，无链接或加密成员。")
                    total += entry.file_size
                    if entry.file_size > MAX_FILE or total > MAX_EXPANDED:
                        raise ValueError("展开后的证据大小超出限制。")
                    destination = staging.joinpath(*parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(entry) as input_file, destination.open("xb") as output_file:
                        copied = 0
                        while chunk := input_file.read(65536):
                            copied += len(chunk)
                            if copied > entry.file_size or copied > MAX_FILE:
                                raise ValueError("归档成员大小不匹配。")
                            output_file.write(chunk)
            anchor = staging / "anchors" / "demo.public"
            if anchor.is_file() and anchor.read_bytes() != public_key:
                raise ValueError("归档公钥与独立选择的公钥不匹配。")
            (staging / "public.key").write_bytes(public_key)
            manifest = self._verify(staging, run_id)
            actual = {path.name for path in (staging / "bundle").iterdir()}
            if actual != set(manifest["files"]) | {"manifest.json", "manifest.sig"}:
                raise ValueError("归档包含未签名资产。")
            result = self._result(staging, run_id)
            _validate_replay(strict_json((staging / "bundle" / "replay.json").read_bytes()), result)
            for name in manifest["files"]:
                if name.startswith("viewer-model") and name.endswith(".json"):
                    _validate_model(strict_json((staging / "bundle" / name).read_bytes()))
            # No replace: concurrent imports may not overwrite a retained run.
            os.rename(staging, target)
            return result
        except zipfile.BadZipFile as exc:
            raise ValueError("无法读取 ZIP 证据包。") from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def import_request(self, body: dict) -> dict:
        if not isinstance(body, dict) or set(body) != {"archive_base64", "run_id", "public_key_base64"}:
            raise ValueError("需要 archive_base64、run_id 和独立选择的 public_key_base64。")
        if not isinstance(body["archive_base64"], str) or len(body["archive_base64"]) > ((UPLOAD_LIMIT + 2) // 3) * 4:
            raise ValueError("网页导入限 64 MiB ZIP；更大证据请使用 native-import 命令。")
        if not isinstance(body["public_key_base64"], str) or len(body["public_key_base64"]) > 48:
            raise ValueError("公钥格式不合法。")
        try:
            archive = base64.b64decode(body["archive_base64"], validate=True)
            public = base64.b64decode(body["public_key_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("证据或公钥不是合法 Base64。") from exc
        if len(archive) > UPLOAD_LIMIT:
            raise ValueError("网页导入限 64 MiB ZIP。")
        return self.import_archive(archive, body["run_id"], public)

    def export(self, run_id: str) -> Path:
        directory = self._directory(run_id)
        manifest = self._verify(directory, run_id)
        descriptor, filename = tempfile.mkstemp(prefix=".native-export-", suffix=".zip", dir=self.root)
        os.close(descriptor)
        output = Path(filename)
        try:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(set(manifest["files"]) | {"manifest.json", "manifest.sig"}):
                    archive.write(directory / "bundle" / name, "bundle/" + name)
                    if output.stat().st_size > MAX_ARCHIVE:
                        raise ValueError("导出归档超出 256 MiB。")
                archive.write(directory / "public.key", "anchors/demo.public")
            if output.stat().st_size > MAX_ARCHIVE:
                raise ValueError("导出归档超出 256 MiB。")
            return output
        except BaseException:
            output.unlink(missing_ok=True)
            raise
