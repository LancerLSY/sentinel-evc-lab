"""Operator-configured unified VLA jobs. HTTP cannot select programs or paths.

The ML/physics runtime stays in its own Python environment. A stopped simulator
process is not a confirmed physical-device stop. Live files are provisional;
terminal records and raw artifacts are signed only after the process exits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

from .contracts import canonical_json
from .events import EventLog
from .evidence import build_bundle, verify_bundle
from .physics_jobs import _atomic
from .runstore import _WorkspaceLock, utc_now
from .scenario import InputError, strict_json

PROFILE = "smolvla-libero-panda"
JOB_ID = re.compile(r"loop-[0-9a-f]{32}\Z")
ACTIVE = {"queued", "preparing", "running", "stopping"}
TERMINAL = {"completed", "stopped", "rejected", "failed", "interrupted"}
MAX_ASSETS = 256 * 1024 * 1024
WORKSPACE_BUDGET = 8 * 1024 * 1024 * 1024


def _read(path, limit=4 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("missing, oversized or unsafe local artifact")
    return strict_json(path.read_bytes())


def _source_digest(source):
    digest = hashlib.sha256()
    paths = sorted((source / "src" / "sentinel_evc").rglob("*.py"))
    paths += sorted((source / "experiments" / "vla").glob("*.py"))
    paths += sorted((source / "experiments" / "vla").rglob("*.json"))
    for path in paths:
        if path.is_symlink():
            raise ValueError("symlink runtime source refused")
        digest.update(path.relative_to(source).as_posix().encode() + b"\0" + path.read_bytes())
    return "sha256:" + digest.hexdigest()


class LoopJobs:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._ownership = _WorkspaceLock(self.root / "jobs.lock")
        self._lock = threading.RLock()
        self._workers = {}
        self._processes = {}
        self._closed = False
        self._runtime_cache = None
        for path in self.root.glob("loop-*/job.json"):
            try:
                record = self._raw(path.parent.name)
                if record["status"] in ACTIVE:
                    # A surviving runner sees this before its next writer entry.
                    (path.parent / "stop.request").touch(exist_ok=True)
                    record.update(status="interrupted", stage="interrupted",
                                  error="服务已重启；本次运行中断，请创建新运行。")
                    self._save(record)
            except (OSError, ValueError, KeyError):
                continue

    def directory(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise KeyError("loop job not found")
        path = self.root / job_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise KeyError("loop job not found")
        return path

    def _save(self, record):
        record["updated_at"] = utc_now()
        _atomic(self.directory(record["id"]) / "job.json", record)

    def _raw(self, job_id):
        value = _read(self.directory(job_id) / "job.json")
        if (not isinstance(value, dict) or value.get("id") != job_id
                or value.get("status") not in ACTIVE | TERMINAL):
            raise KeyError("invalid loop record")
        return value

    def configure(self, python, source, config):
        source = Path(source).expanduser().resolve()
        python = Path(python).expanduser().absolute()
        config = Path(config).expanduser().resolve()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("需要已有 VLA 环境中可执行的 Python 绝对路径。")
        if not (source / "experiments/vla/product_session.py").is_file():
            raise ValueError("源码目录缺少 experiments/vla/product_session.py。")
        value = _read(config)
        if not isinstance(value, dict):
            raise ValueError("闭环配置必须是 JSON 对象。")
        with self._lock:
            if any(worker.is_alive() for worker in self._workers.values()):
                raise RuntimeError("运行期间不能更换闭环环境。")
            # Local operator configuration, never returned through HTTP.
            _atomic(self.root / "runtime.json", {"python": str(python), "source": str(source),
                    "config": value, "source_digest": _source_digest(source)})
            self._runtime_cache = None
        return self.runtime(refresh=True)

    def _runtime(self):
        value = _read(self.root / "runtime.json")
        if set(value) != {"python", "source", "config", "source_digest"}:
            raise ValueError("invalid operator runtime")
        if _source_digest(Path(value["source"])) != value["source_digest"]:
            raise ValueError("源码已更改，请重新运行 loop-configure 绑定当前版本。")
        return value

    def runtime(self, refresh=False):
        with self._lock:
            if not refresh and self._runtime_cache and time.monotonic() - self._runtime_cache[0] < 15:
                return self._runtime_cache[1]
        status = {"configured": False, "ready": False, "profile": PROFILE, "checks": []}
        if not (self.root / "runtime.json").exists():
            status["checks"] = [{"name": "operator_configuration", "ok": False,
                                  "detail": "先通过 CLI 绑定已有 VLA Python、源码和任务配置。"}]
            return status
        try:
            runtime = self._runtime()
            status["configured"] = True
            config_path = self.root / "check-config.json"
            _atomic(config_path, runtime["config"])
            command = [runtime["python"], str(Path(runtime["source"]) / "experiments/vla/product_session.py"),
                       "--config", str(config_path), "--check"]
            process = subprocess.run(command, capture_output=True, timeout=45, check=False, cwd=runtime["source"])
            report = strict_json(process.stdout)
            checks = report.get("checks")
            if not isinstance(checks, list):
                raise ValueError("invalid runtime readiness report")
            status["checks"] = checks
            status["ready"] = process.returncode == 0 and report.get("ready") is True
        except (OSError, ValueError, subprocess.SubprocessError):
            status["checks"] = [{"name": "runtime_check", "ok": False,
                                  "detail": "运行环境检查失败；检查本地 CLI 配置、源码版本和依赖。"}]
        with self._lock:
            self._runtime_cache = (time.monotonic(), status)
        return status

    def start(self, body):
        if body != {}:
            raise InputError("闭环启动使用已绑定配置，不接收程序、路径或任意驱动参数。")
        with self._lock:
            if self._closed or any(thread.is_alive() for thread in self._workers.values()):
                raise RuntimeError("闭环服务已关闭或已有运行，请等待当前运行结束。")
            try:
                ownership = _WorkspaceLock(self.root / "execution.lock")
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("旧闭环进程尚未退出，请等待运行锁释放。") from exc
            ownership.close()
            used = sum(path.stat().st_size for path in self.root.rglob("*")
                       if not path.is_symlink() and path.is_file())
            if used + 3 * MAX_ASSETS > WORKSPACE_BUDGET or shutil.disk_usage(self.root).free < 3 * MAX_ASSETS:
                raise RuntimeError("闭环工作区预算不足；先核验并备份已有证据，再移出旧作业目录。")
            if not self.runtime(refresh=True)["ready"]:
                raise InputError("闭环环境尚未就绪，请先完成页面列出的环境检查。")
            runtime = self._runtime()
            job_id = "loop-" + uuid.uuid4().hex
            directory = self.directory(job_id)
            directory.mkdir()
            _atomic(directory / "config.json", runtime["config"])
            record = {"id": job_id, "name": "SmolVLA · Panda 闭环", "profile": PROFILE,
                      "status": "queued", "stage": "queued", "created_at": utc_now(),
                      "error": None, "summary": None, "verification": None,
                      "source_digest": runtime["source_digest"],
                      "scope": "实际模型推理与 MuJoCo 仿真；仿真几何检查，不构成实机停止证明。"}
            self._save(record)
            worker = threading.Thread(target=self._execute, args=(job_id, runtime), daemon=False)
            self._workers[job_id] = worker
            worker.start()
            return record

    def _execute(self, job_id, runtime):
        directory = self.directory(job_id)
        output = directory / "output"
        record = self._raw(job_id)
        try:
            with self._lock:
                record = self._raw(job_id)
                if (directory / "stop.request").exists():
                    record.update(status="stopped", stage="stopped")
                    self._finalize(record, output)
                    return
                record.update(status="preparing", stage="environment")
                self._save(record)
                command = [runtime["python"], str(Path(runtime["source"]) / "experiments/vla/product_session.py"),
                           "--config", str(directory / "config.json"), "--output-dir", str(output),
                           "--stop-file", str(directory / "stop.request")]
                with (directory / "runner.log").open("wb") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               cwd=runtime["source"], start_new_session=(os.name != "nt"))
                self._processes[job_id] = process
            returncode = process.wait()
            record = self._raw(job_id)
            manifest = _read(output / "manifest.json", 32 * 1024 * 1024) if (output / "manifest.json").exists() else {}
            status = manifest.get("status")
            if returncode == 0 and status == "stopped":
                record.update(status="stopped", stage="stopped")
            elif returncode == 0 and status == "complete":
                record.update(status="completed", stage="completed")
            elif returncode == 0 and status == "rejected":
                record.update(status="rejected", stage="rejected")
            else:
                record.update(status="failed", stage="failed", error="闭环运行未完成；原始诊断保存在本地 runner.log。")
            if _source_digest(Path(runtime["source"])) != runtime["source_digest"]:
                record.update(status="failed", stage="source_changed", error="运行期间源码发生变化，本次结果不能用于版本验收。")
            rows = manifest.get("results", [])
            public_rows = [{key: value for key, value in row.items() if key in {
                "task_id", "state_index", "seed", "episode_id", "success", "crashed", "steps",
                "denied_candidates", "unwanted_collisions", "timing", "error_type", "stop_requested",
                "elapsed_seconds", "full_incremental_disagreements"}} for row in rows]
            record["summary"] = {"episodes": len(rows), "successes": sum(row.get("success") is True for row in rows),
                                 "steps": sum(int(row.get("steps", 0)) for row in rows),
                                 "denied_candidates": sum(int(row.get("denied_candidates", 0)) for row in rows),
                                 "unwanted_collisions": sum(int(row.get("unwanted_collisions", 0)) for row in rows),
                                 "task_success": all(row.get("success") is True for row in rows) if rows else None,
                                 "results": public_rows, "runner_status": status}
            self._finalize(record, output)
        except Exception:
            record.update(status="failed", stage="failed", verification=None,
                          error="运行或证据封装失败；未生成通过核验的完成记录。")
            self._save(record)
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _finalize(self, record, output):
        artifacts = {}
        inventory = []
        total = 0
        if output.exists():
            for path in sorted(output.rglob("*")):
                if path.is_symlink():
                    raise ValueError("symlink runner output refused")
                if not path.is_file():
                    continue
                total += path.stat().st_size
                if total > MAX_ASSETS or len(inventory) >= 1024:
                    raise ValueError("runner artifacts exceed export budget")
                relative = path.relative_to(output).as_posix()
                name = "asset-" + hashlib.sha256(relative.encode()).hexdigest()[:24] + path.suffix
                artifacts[name] = path.read_bytes()
                inventory.append({"path": relative, "asset": name})
        record["updated_at"] = utc_now()
        record["verification"] = None
        artifacts["job.json"] = canonical_json(record)
        artifacts["config.json"] = (self.directory(record["id"]) / "config.json").read_bytes()
        artifacts["inventory.json"] = canonical_json(inventory)
        log = EventLog(record["id"])
        log.append("OUTCOME", status=record["status"], source_digest=record["source_digest"],
                   note="Post-exit artifact receipt; gateway events are separately included raw artifacts.")
        evidence = build_bundle(log, str(self.directory(record["id"])), artifacts)
        ok, message = verify_bundle(evidence["bundle_dir"], evidence["public_key"], record["id"])
        if not ok:
            raise RuntimeError("evidence finalization failed")
        record["verification"] = {"ok": ok, "message": message, "trust": "local demo key"}
        self._save(record)

    def get(self, job_id):
        record = self._raw(job_id)
        if record["status"] in ACTIVE:
            try:
                progress = _read(self.directory(job_id) / "output/product-progress.json")
                record["stage"] = progress.get("stage", record["stage"])
                if record["status"] != "stopping" and progress.get("status") in {"preparing", "running"}:
                    record["status"] = progress["status"]
            except (OSError, ValueError):
                pass
        elif (self.directory(job_id) / "bundle").exists():
            ok, message = verify_bundle(str(self.directory(job_id) / "bundle"),
                                        str(self.directory(job_id) / "anchors/demo.public"), job_id)
            if ok:
                signed = _read(self.directory(job_id) / "bundle/job.json")
                if {k: v for k, v in record.items() if k not in {"verification", "updated_at"}} != {
                        k: v for k, v in signed.items() if k not in {"verification", "updated_at"}}:
                    ok, message = False, "SIGNED_RECORD_MISMATCH"
            record["verification"] = {"ok": ok, "message": message, "trust": "local demo key"}
            if not ok:
                record.update(status="failed", error="证据完整性检查失败，不能导出。", summary=None)
        else:
            record["verification"] = {"ok": False, "message": "UNFINALIZED_RECORD", "trust": "local demo key"}
            if record["status"] == "completed":
                record.update(status="failed", error="缺少已封装证据，不能确认运行完成。", summary=None)
        return record

    def list(self):
        records = []
        for path in self.root.glob("loop-*/job.json"):
            try:
                records.append(self.get(path.parent.name))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(records, key=lambda value: value["created_at"], reverse=True)

    def live(self, job_id):
        record = self.get(job_id)
        path = self.directory(job_id) / "output/product-live.json"
        try:
            live = None
            # Final views must use the signed snapshot, not mutable runner files.
            if record["status"] in TERMINAL:
                if (record.get("verification") or {}).get("ok"):
                    inventory = _read(self.directory(job_id) / "bundle/inventory.json")
                    item = next((row for row in inventory if row["path"] == "product-live.json"), None)
                    if item:
                        live = _read(self.directory(job_id) / "bundle" / item["asset"], 32 * 1024 * 1024)
            else:
                live = _read(path, 32 * 1024 * 1024)
        except (OSError, ValueError):
            live = None
        return {"job": record, "live": live}

    def model(self, job_id):
        record = self.get(job_id)
        if record["status"] in TERMINAL:
            if not (record.get("verification") or {}).get("ok"):
                raise KeyError("no verified scene model")
            inventory = _read(self.directory(job_id) / "bundle/inventory.json")
            item = next((row for row in inventory if row["path"] == "viewer-model.json"), None)
            if not item:
                raise KeyError("no scene model")
            return _read(self.directory(job_id) / "bundle" / item["asset"], 32 * 1024 * 1024)
        return _read(self.directory(job_id) / "output/viewer-model.json", 32 * 1024 * 1024)

    def stop(self, job_id):
        with self._lock:
            record = self._raw(job_id)
            if record["status"] in ACTIVE:
                (self.directory(job_id) / "stop.request").touch(exist_ok=True)
                record.update(status="stopping", stage="stop_requested")
                self._save(record)
        return self.get(job_id)

    def export(self, job_id):
        record = self.get(job_id)
        if not (record.get("verification") or {}).get("ok") or record["status"] in ACTIVE:
            raise RuntimeError("只可导出进程退出后、通过独立核验的证据包。")
        directory = self.directory(job_id)
        target = directory / "evidence.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((directory / "bundle").iterdir()):
                archive.write(path, "bundle/" + path.name)
            archive.write(directory / "anchors/demo.public", "anchors/demo.public")
        return target

    def close(self):
        with self._lock:
            self._closed = True
            workers = list(self._workers.items())
            for job_id, worker in workers:
                if worker.is_alive():
                    self.stop(job_id)
        for job_id, worker in workers:
            worker.join(timeout=10)
            process = self._processes.get(job_id)
            if worker.is_alive() and process is not None:
                # Simulator-only process cleanup. This is not device cancellation.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                worker.join(timeout=5)
            if worker.is_alive():
                raise RuntimeError("闭环工作进程尚未退出，保留工作区所有权。")
        if self._ownership is not None:
            self._ownership.close()
            self._ownership = None
