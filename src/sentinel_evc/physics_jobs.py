"""Persisted, isolated physics jobs shared by the App and CLI.

A completed job may contain failed scientific gates.  Re-reading a terminal job
verifies its signed experiment and bindings; no displayed result is invented.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
import zipfile

from .contracts import canonical_json
from .physics import PhysicsConfig
from .runstore import _WorkspaceLock, utc_now
from .scenario import InputError, strict_json
from .ssh_experiment import _expected_source_manifest_sha256, _verify_result_tree

JOB_ID = re.compile(r"^physics-job-[0-9a-f]{32}$")
ACTIVE = {"queued", "running"}


def physics_engine_status():
    """Return an honest optional-engine capability without making it mandatory."""
    try:
        if importlib.util.find_spec("mujoco") is None:
            return {"available": False, "version": None,
                    "reason": "MuJoCo is not installed; install sentinel-evc-lab[physics]."}
        module = importlib.import_module("mujoco")
        return {"available": True, "version": str(getattr(module, "__version__", "unknown")), "reason": None}
    except Exception as exc:
        return {"available": False, "version": None,
                "reason": "MuJoCo import failed: " + type(exc).__name__}


def _atomic(path, value):
    if path.is_symlink():
        raise ValueError("symlink metadata refused")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class PhysicsJobs:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._ownership = _WorkspaceLock(self.root / "jobs.lock")
        self._lock = threading.RLock()
        self._workers = {}
        self._processes = {}
        self._closed = False
        self._engine = physics_engine_status()
        for path in self.root.glob("physics-job-*/job.json"):
            if not JOB_ID.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                record = strict_json(path.read_bytes())
                if record["status"] in ACTIVE:
                    record.update(status="interrupted", error="进程重启，物理试验已中断；请创建新作业。")
                    self._save(record)
            except (ValueError, KeyError, OSError):
                continue

    def engine_status(self):
        return dict(self._engine)

    def directory(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise KeyError("physics job not found")
        directory = self.root / job_id
        if directory.is_symlink() or directory.resolve().parent != self.root:
            raise KeyError("physics job not found")
        return directory

    def _save(self, record):
        record["updated_at"] = utc_now()
        _atomic(self.directory(record["id"]) / "job.json", record)

    def _raw(self, job_id):
        path = self.directory(job_id) / "job.json"
        if path.is_symlink():
            raise KeyError("physics job not found")
        try:
            record = strict_json(path.read_bytes())
            if record["id"] != job_id:
                raise ValueError("job identity mismatch")
            return record
        except (OSError, ValueError, KeyError) as exc:
            raise KeyError("physics job not found") from exc

    def start(self, value):
        if not isinstance(value, dict) or set(value) - {"seed", "friction", "render"}:
            raise InputError("仅支持 seed、friction、render 物理参数。")
        seed, friction, render = value.get("seed", 7), value.get("friction", .35), value.get("render", False)
        if type(seed) is not int:
            raise InputError("seed 必须为整数。")
        if isinstance(friction, bool) or not isinstance(friction, (int, float)) or not math.isfinite(friction):
            raise InputError("friction 必须为有限数值。")
        if type(render) is not bool:
            raise InputError("render 必须为布尔值。")
        try:
            PhysicsConfig(seed=seed, friction=float(friction))
        except (TypeError, ValueError) as exc:
            raise InputError(str(exc)) from exc
        if not self._engine["available"]:
            raise InputError(self._engine["reason"])
        with self._lock:
            if self._closed:
                raise RuntimeError("物理作业服务已关闭。")
            if any(thread.is_alive() for thread in self._workers.values()):
                raise RuntimeError("已有三维物理作业运行中，请等待完成。")
            job_id = "physics-job-" + uuid.uuid4().hex
            directory = self.directory(job_id)
            source_root = Path(__file__).resolve().parents[2]
            expected_source = _expected_source_manifest_sha256(source_root)
            directory.mkdir()
            record = {"id": job_id, "name": "MuJoCo 三维托盘实验", "status": "queued",
                      "seed": seed, "friction": float(friction), "render": render,
                      "created_at": utc_now(), "error": None, "summary": None,
                      "source_manifest_sha256": expected_source, "scientific_gates_pass": None,
                      "scope": "mujoco-tray-geometry-v1"}
            self._save(record)
            thread = threading.Thread(target=self._execute, args=(job_id,), daemon=True)
            self._workers[job_id] = thread
            thread.start()
            return dict(record)

    def _execute(self, job_id):
        directory = self.directory(job_id)
        process = None
        try:
            with self._lock:
                record = self._raw(job_id)
                if self._closed:
                    raise RuntimeError("服务正在关闭。")
                record["status"] = "running"
                self._save(record)
            command = [sys.executable, "-m", "sentinel_evc", "physics", "--out", str(directory / "experiment"),
                       "--seed", str(record["seed"]), "--friction", str(record["friction"])]
            if record["render"]:
                command.append("--render")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            with (directory / "worker.log").open("xb") as log:
                with self._lock:
                    if self._closed:
                        raise RuntimeError("服务正在关闭。")
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    self._processes[job_id] = process
                return_code = process.wait(timeout=600)
            if return_code not in (0, 3):
                raise RuntimeError("物理进程未完整完成（退出码 " + str(return_code) + "）。")
            verified = _verify_result_tree(directory / "experiment")
            completion = strict_json((directory / "experiment/COMPLETE.json").read_bytes())
            if completion["source_manifest_sha256"] != record["source_manifest_sha256"]:
                raise RuntimeError("试验期间源码版本发生变化，未发布验收结论。")
            config = verified["job_config"]
            if any(config.get(key) != record[key] for key in ("seed", "friction", "render")):
                raise RuntimeError("签名配置与作业请求不一致。")
            with self._lock:
                record.update(status="completed", error=None, scientific_gates_pass=verified["summary"]["infrastructure_gates_pass"])
                self._save(record)
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            with self._lock:
                record = self._raw(job_id)
                record.update(status="failed", error=str(exc)[:1000], scientific_gates_pass=None)
                self._save(record)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def get(self, job_id):
        with self._lock:
            record = self._raw(job_id)
            if record["status"] == "completed":
                try:
                    verified = _verify_result_tree(self.directory(job_id) / "experiment")
                    complete = strict_json((self.directory(job_id) / "experiment/COMPLETE.json").read_bytes())
                    if complete["source_manifest_sha256"] != record["source_manifest_sha256"]:
                        raise ValueError("source mismatch")
                    if any(verified["job_config"].get(key) != record[key] for key in ("seed", "friction", "render")):
                        raise ValueError("job configuration mismatch")
                    record["summary"] = verified["summary"]
                    record["scientific_gates_pass"] = record["summary"]["infrastructure_gates_pass"]
                    record["verification"] = {"ok": True, "trial_count": verified["trial_count"],
                                              "trust": "self-contained demo integrity keys"}
                except (ValueError, OSError, KeyError, RuntimeError) as exc:
                    record.update(status="failed", summary=None, scientific_gates_pass=None,
                                  error="EVIDENCE_INVALID: " + str(exc)[:300], verification={"ok": False})
                    self._save(record)
            return record

    def list(self):
        with self._lock:
            rows = []
            for path in self.root.glob("physics-job-*/job.json"):
                try:
                    row = self._raw(path.parent.name)
                    rows.append({key: row.get(key) for key in ("id", "name", "status", "seed", "friction", "created_at", "error", "scientific_gates_pass")})
                except KeyError:
                    continue
            return sorted(rows, key=lambda row: row["created_at"], reverse=True)[:200]

    def trace(self, job_id):
        record = self.get(job_id)
        if not record.get("verification", {}).get("ok"):
            raise ValueError("完整签名试验完成后才可回放。")
        trial_id = record["summary"]["integrated"]["run_id"]
        if trial_id != "physics-" + str(record["seed"]) + "-integrated":
            raise ValueError("signed trace identity mismatch")
        path = self.directory(job_id) / "experiment" / trial_id / "bundle/trace.json"
        rows = strict_json(path.read_bytes())
        stride = max(1, (len(rows) + 1499) // 1500)
        sampled = rows[::stride]
        if rows and sampled[-1] != rows[-1]:
            sampled.append(rows[-1])
        return {"trace": sampled, "run_id": trial_id, "source_sample_count": len(rows),
                "scope": "recorded MuJoCo feedback; geometry-only selected branch", "sample_stride": stride}

    def export(self, job_id):
        record = self.get(job_id)
        if not record.get("verification", {}).get("ok"):
            raise ValueError("证据校验通过后才能导出。")
        directory = self.directory(job_id)
        source = directory / "experiment"
        target = directory / "evidence.zip"
        temp = directory / ("evidence-" + uuid.uuid4().hex + ".tmp")
        try:
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                roots = [source / "experiment-index"]
                anchors = strict_json((source / "experiment-index/bundle/trial_anchors.json").read_bytes())
                roots.extend(source / run_id for run_id in anchors)
                paths = [source / "COMPLETE.json", source / "summary.json"]
                for root in roots:
                    manifest = strict_json((root / "bundle/manifest.json").read_bytes())
                    paths.extend([root / "bundle/manifest.json", root / "bundle/manifest.sig", root / "anchors/demo.public"])
                    paths.extend(root / "bundle" / name for name in manifest["files"])
                for path in paths:
                    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
                        raise ValueError("unsafe evidence export")
                    archive.write(path, path.relative_to(source))
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def close(self):
        with self._lock:
            self._closed = True
            for process in self._processes.values():
                if process.poll() is None:
                    process.terminate()
            workers = list(self._workers.values())
        for worker in workers:
            worker.join(timeout=2)
        with self._lock:
            for process in self._processes.values():
                if process.poll() is None:
                    process.kill()
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=2)
        self._ownership.close()
