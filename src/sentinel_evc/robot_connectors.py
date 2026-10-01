"""Persistent robot connection profiles and bounded read-only diagnostics."""
from __future__ import annotations

import json
import ipaddress
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid
from datetime import datetime, timezone

from .contracts import canonical_json


ROBOT_ID = re.compile(r"^robot-[0-9a-f]{32}$")
_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
DRIVERS = frozenset({"mock", "ur_dashboard_readonly"})
MAX_PROFILES = 100
MAX_LINE = 4096
_ROBOT_MODES = frozenset({
    "NO_CONTROLLER", "DISCONNECTED", "CONFIRM_SAFETY", "BOOTING", "POWER_OFF",
    "POWER_ON", "IDLE", "BACKDRIVE", "RUNNING", "UPDATING_FIRMWARE",
})
_SAFETY_STATUSES = frozenset({
    "NORMAL", "REDUCED", "PROTECTIVE_STOP", "RECOVERY", "SAFEGUARD_STOP",
    "SYSTEM_EMERGENCY_STOP", "ROBOT_EMERGENCY_STOP", "VIOLATION", "FAULT",
    "AUTOMATIC_MODE_SAFEGUARD_STOP", "SYSTEM_THREE_POSITION_ENABLING_STOP",
})
_READ_ONLY_COMMANDS = ("get robot model", "robotmode", "safetystatus")


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("symlink storage refused")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _LineChannel:
    def __init__(self, connection):
        self.connection = connection
        self.buffer = bytearray()
        self.commands_sent = []

    def read(self) -> str:
        while b"\n" not in self.buffer:
            if len(self.buffer) > MAX_LINE:
                raise ValueError("dashboard response exceeds line limit")
            chunk = self.connection.recv(min(512, MAX_LINE + 1 - len(self.buffer)))
            if not chunk:
                raise ValueError("dashboard connection closed before a complete response")
            self.buffer.extend(chunk)
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        if len(line) > MAX_LINE or b"\x00" in line:
            raise ValueError("invalid dashboard response")
        text = line.rstrip(b"\r").decode("utf-8")
        if not text or any(ord(character) < 32 for character in text):
            raise ValueError("invalid dashboard response")
        return text

    def request(self, command: str) -> str:
        if command not in _READ_ONLY_COMMANDS:
            raise ValueError("dashboard command is not in the read-only allowlist")
        self.connection.sendall((command + "\n").encode("ascii"))
        self.commands_sent.append(command)
        return self.read()


def _valid_host(host) -> bool:
    if not isinstance(host, str) or not 1 <= len(host) <= 253 or host.endswith("."):
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        labels = host.split(".")
        return all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels)


def _parse_ur_responses(greeting: str, responses: dict[str, str]) -> dict:
    if "dashboard server" not in greeting.lower() or len(greeting) > 256:
        raise ValueError("unexpected Universal Robots dashboard greeting")
    model = responses["get robot model"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._/-]{0,127}", model):
        raise ValueError("invalid robot model response")
    robot_match = re.fullmatch(r"Robotmode: ([A-Z_]+)", responses["robotmode"], re.IGNORECASE)
    safety_match = re.fullmatch(r"Safetystatus: ([A-Z_]+)", responses["safetystatus"], re.IGNORECASE)
    robot_mode = robot_match.group(1).upper() if robot_match else None
    safety_status = safety_match.group(1).upper() if safety_match else None
    if robot_mode not in _ROBOT_MODES or safety_status not in _SAFETY_STATUSES:
        raise ValueError("dashboard returned an unsupported robot or safety status")
    return {"model": model, "robot_mode": robot_mode, "safety_status": safety_status}


class RobotRegistry:
    """Stores non-secret profiles and performs explicitly read-only diagnostics."""

    def __init__(self, root):
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise ValueError("symlink robot root refused")
        self.root = requested.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _directory(self, robot_id: str) -> Path:
        if not isinstance(robot_id, str) or not ROBOT_ID.fullmatch(robot_id):
            raise KeyError("robot profile not found")
        path = self.root / robot_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise KeyError("robot profile not found")
        return path

    def create(self, name, driver, host=None, port=None) -> dict:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("name must be 1..128 printable characters")
        if not isinstance(driver, str) or driver not in DRIVERS:
            raise ValueError("driver must be mock or ur_dashboard_readonly")
        if driver == "mock":
            if host is not None or port is not None:
                raise ValueError("mock driver does not accept a network endpoint")
            endpoint = None
        else:
            if not _valid_host(host):
                raise ValueError("host must be a bounded hostname or IP literal")
            if port is None:
                port = 29999
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError("port must be an integer from 1 to 65535")
            endpoint = {"host": host, "port": port}
        with self._lock:
            if len(self.list()) >= MAX_PROFILES:
                raise ValueError("robot profile limit reached")
            robot_id = "robot-" + uuid.uuid4().hex
            record = {
                "id": robot_id,
                "name": name,
                "driver": driver,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "mode": "demo" if driver == "mock" else "read_only_diagnostic",
                "hardware_motion": False,
                "credentials_stored": False,
                "capabilities": {
                    "diagnose": True,
                    "write": False,
                    "feedback": False,
                    "queue": False,
                    "cancel": False,
                    "executor_ready": False,
                },
                "last_diagnostic": None,
            }
            directory = self._directory(robot_id)
            directory.mkdir()
            try:
                _atomic_write(directory / "robot.json", canonical_json(record))
            except Exception:
                directory.rmdir()
                raise
            return dict(record)

    def get(self, robot_id) -> dict:
        with self._lock:
            try:
                record = json.loads((self._directory(robot_id) / "robot.json").read_text("utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise KeyError("robot profile not found") from exc
            driver = record.get("driver")
            if record.get("id") != robot_id or not isinstance(driver, str) or driver not in DRIVERS:
                raise KeyError("robot profile not found")
            return record

    def list(self) -> list[dict]:
        with self._lock:
            records = []
            for path in self.root.iterdir():
                if path.is_dir() and not path.is_symlink() and ROBOT_ID.fullmatch(path.name):
                    try:
                        records.append(self.get(path.name))
                    except KeyError:
                        continue
            return sorted(records, key=lambda row: row["id"], reverse=True)[:MAX_PROFILES]

    def diagnose(self, robot_id, *, timeout=1.5) -> dict:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be finite")
        timeout = float(timeout)
        if not math.isfinite(timeout) or not 0.1 <= timeout <= 5.0:
            raise ValueError("timeout must be within 0.1..5.0 seconds")
        with self._lock:
            record = self.get(robot_id)
        started = time.monotonic()
        if record["driver"] == "mock":
            result = {
                "robot_id": robot_id,
                "ok": True,
                "status": "demo_ready",
                "driver": "mock",
                "hardware_connected": False,
                "readings": {"model": "MOCK", "robot_mode": "MOCK_IDLE", "safety_status": "MOCK_NORMAL"},
                "scope": "in-process protocol demonstration; no hardware connection",
                "commands_sent": [],
            }
        else:
            endpoint = record["endpoint"]
            commands_sent = []
            channel = None
            try:
                with socket.create_connection((endpoint["host"], endpoint["port"]), timeout=timeout) as connection:
                    connection.settimeout(timeout)
                    channel = _LineChannel(connection)
                    greeting = channel.read()
                    responses = {}
                    for command in _READ_ONLY_COMMANDS:
                        responses[command] = channel.request(command)
                        commands_sent = list(channel.commands_sent)
                readings = _parse_ur_responses(greeting, responses)
                result = {
                    "robot_id": robot_id,
                    "ok": True,
                    "status": "read_only_connected",
                    "driver": "ur_dashboard_readonly",
                    "hardware_connected": True,
                    "readings": readings,
                    "scope": "Universal Robots Dashboard read-only diagnostic",
                    "commands_sent": commands_sent,
                }
            except (OSError, TimeoutError, UnicodeError, ValueError) as exc:
                if channel is not None:
                    commands_sent = list(channel.commands_sent)
                result = {
                    "robot_id": robot_id,
                    "ok": False,
                    "status": "diagnostic_failed",
                    "driver": "ur_dashboard_readonly",
                    "hardware_connected": False,
                    "error": {"code": "READ_ONLY_DIAGNOSTIC_FAILED", "message": str(exc)[:256]},
                    "scope": "No motion or state-changing command was sent.",
                    "commands_sent": commands_sent,
                }
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        with self._lock:
            latest = self.get(robot_id)
            latest["last_diagnostic"] = result
            _atomic_write(self._directory(robot_id) / "robot.json", canonical_json(latest))
        return result
