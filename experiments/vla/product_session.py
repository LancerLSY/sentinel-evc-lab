#!/usr/bin/env python3
"""Launch one managed SmolVLA -> Panda geometry -> EVC -> MuJoCo session.

The heavy VLA and simulator dependencies remain optional.  ``--check`` only
validates the selected local profile and reports missing runtime pieces as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "config_unified_libero_A6_20261004.json"
DEFAULT_PROTOCOL_SHA256 = "sha256:53d09ea749705e0297b9f512eff6e5776da405e6e5cc94365e7648d6e35a938f"
PROFILE = "smolvla-libero-panda"
REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "mujoco": "mujoco",
    "lerobot": "lerobot",
    "libero": "hf-libero",
    "robosuite": "robosuite",
    "safetensors": "safetensors",
}
PINNED_PACKAGES = {"mujoco": "3.3.7", "lerobot": "0.6.1", "hf-libero": "0.1.4", "robosuite": "1.4.0"}
EXPECTED_CHECKPOINT_FILES = {
    "config.json": "sha256:5c9f3ba9f5f37ea7024c9501b0b20b1941f232989c2167853cb46d9071a70dd7",
    "model.safetensors": "sha256:9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8",
    "policy_postprocessor.json": "sha256:3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "sha256:b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
    "policy_preprocessor.json": "sha256:122ec5106602b1bf129f49690d05ab2f49748a0ac6119de55ea0677d4e90d248",
    "policy_preprocessor_step_5_normalizer_processor.safetensors": "sha256:b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9",
}
EXPECTED_BACKBONE_FILES = {
    "config.json": "sha256:ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17",
    "preprocessor_config.json": "sha256:149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374",
    "processor_config.json": "sha256:f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c",
    "special_tokens_map.json": "sha256:2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60",
    "tokenizer.json": "sha256:5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    "tokenizer_config.json": "sha256:dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_product_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("product session config must be one JSON object")
    allowed = {
        "profile", "protocol_config", "checkpoint", "backbone",
        "task_id", "initial_state_index", "seed", "max_steps", "device", "mujoco_gl",
        "torch_threads", "lease_ttl_ms", "max_feedback_age_ms", "aggregation",
    }
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(f"unknown product session config fields: {unknown}")
    if source.get("profile", PROFILE) != PROFILE:
        raise ValueError(f"profile must be {PROFILE!r}")
    required = {"checkpoint", "backbone", "task_id", "initial_state_index", "seed", "max_steps"}
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"missing product session config fields: {missing}")
    protocol = Path(source.get("protocol_config", DEFAULT_PROTOCOL)).expanduser()
    if not protocol.is_absolute():
        protocol = REPO_ROOT / protocol
    protocol = protocol.resolve()
    if not protocol.is_file():
        raise FileNotFoundError(f"protocol_config does not exist: {protocol}")
    if _sha256(protocol) != DEFAULT_PROTOCOL_SHA256:
        raise ValueError("protocol_config must match the frozen A6 product basis")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    if tuple(frozen.get("branches", ("parent_only", "full_final", "delta_evc"))) != (
        "parent_only", "full_final", "delta_evc"
    ):
        raise ValueError("protocol_config is not the frozen three-branch unified protocol")
    def bounded_int(name: str, low: int, high: int) -> int:
        value = source[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        return value

    task_id = bounded_int("task_id", 0, 9)
    supported_tasks = {int(item) for item in frozen["formal"]["task_ids"]}
    if task_id not in supported_tasks:
        raise ValueError(f"task_id must be one of the A6-supported tasks {sorted(supported_tasks)}")
    state_index = bounded_int("initial_state_index", 0, 49)
    max_steps = bounded_int("max_steps", 1, 280)
    if not 0 <= task_id <= 9:
        raise ValueError("task_id must be in [0, 9] for libero_spatial")
    if not 0 <= state_index <= 49:
        raise ValueError("initial_state_index must be in [0, 49]")
    if not 1 <= max_steps <= 280:
        raise ValueError("max_steps must be in [1, 280]")
    seed = bounded_int("seed", 0, 2**32 - 1)
    checkpoint = Path(source["checkpoint"]).expanduser()
    backbone = Path(source["backbone"]).expanduser()
    if not checkpoint.is_absolute() or not backbone.is_absolute():
        raise ValueError("checkpoint and backbone must be absolute local directories")
    checkpoint, backbone = checkpoint.resolve(), backbone.resolve()
    torch_threads = source.get("torch_threads", frozen.get("torch_threads", 4))
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or not 1 <= torch_threads <= 64:
        raise ValueError("torch_threads must be an integer in [1, 64]")
    def positive_finite(name: str, default: float) -> float:
        value = source.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        value = float(value)
        if not 1.0 <= value <= 60_000.0:
            raise ValueError(f"{name} must be in [1, 60000] milliseconds")
        return value

    lease_ttl_ms = positive_finite("lease_ttl_ms", float(frozen.get("lease_ttl_ms", 1000.0)))
    max_feedback_age_ms = positive_finite("max_feedback_age_ms", float(frozen.get("max_feedback_age_ms", 2000.0)))
    if str(source.get("device", frozen.get("device", "cuda"))) != "cuda":
        raise ValueError("this product profile requires device='cuda'")
    aggregation = str(source.get("aggregation", "native"))
    if aggregation not in {"native", "fixed_overlap"}:
        raise ValueError("aggregation must be 'native' or 'fixed_overlap'")
    merged = dict(frozen)
    merged.update({
        "checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "device": str(source.get("device", frozen.get("device", "cuda"))),
        "mujoco_gl": str(source.get("mujoco_gl", frozen.get("mujoco_gl", "egl"))),
        "torch_threads": torch_threads,
        "lease_ttl_ms": lease_ttl_ms,
        "max_feedback_age_ms": max_feedback_age_ms,
        "preflight": {"task_id": task_id, "initial_state_index": state_index, "seed": seed, "max_episode_steps": 1},
        "formal": {"task_ids": [task_id], "initial_state_indices": [state_index], "seed_base": seed},
        "deadline_timestamp": None,
        "_formal_step_limit": max_steps,
        "_product_profile": PROFILE,
        "_product_aggregation": aggregation,
    })
    identity = {
        "profile": PROFILE,
        "product_config": {**source, "protocol_config": str(protocol)},
        "product_config_path": str(path),
        "product_config_sha256": _sha256(path),
        "protocol_config": str(protocol),
        "protocol_config_sha256": _sha256(protocol),
    }
    return merged, identity


def model_identity_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for label, root, expected in (
        ("checkpoint", Path(config["checkpoint"]), EXPECTED_CHECKPOINT_FILES),
        ("backbone", Path(config["backbone"]), EXPECTED_BACKBONE_FILES),
    ):
        missing: list[str] = []
        mismatched: list[str] = []
        for name, digest in expected.items():
            path = root / name
            if not path.is_file():
                missing.append(name)
            elif _sha256(path) != digest:
                mismatched.append(name)
        ok = not missing and not mismatched
        detail = "frozen file identity matched" if ok else f"missing={missing}; mismatched={mismatched}"
        checks.append({"name": f"identity:{label}", "ok": ok, "detail": detail})
    return checks


def check(path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        config, identity = load_product_config(path)
        checks.append({"name": "config", "ok": True, "detail": "bounded product session config"})
    except Exception as error:
        return {
            "ready": False, "profile": PROFILE,
            "checks": [{"name": "config", "ok": False, "detail": f"{type(error).__name__}: invalid or unavailable local configuration"}],
        }
    for key in ("checkpoint", "backbone"):
        target = Path(config[key])
        exists = target.is_dir()
        checks.append({"name": key, "ok": exists, "detail": "directory exists" if exists else "directory missing"})
    checks.extend(model_identity_checks(config))
    protocol = Path(identity["protocol_config"])
    protocol_ok = protocol.is_file()
    checks.append({"name": "protocol_config", "ok": protocol_ok, "detail": "frozen protocol found" if protocol_ok else "frozen protocol missing"})
    linux_ok = sys.platform.startswith("linux")
    checks.append({"name": "platform:linux", "ok": linux_ok, "detail": sys.platform})
    for module, package in REQUIRED_PACKAGES.items():
        available = importlib.util.find_spec(module) is not None
        detail = "missing"
        if available:
            try:
                detail = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                detail = "importable (distribution version unavailable)"
        checks.append({"name": f"dependency:{module}", "ok": available, "detail": detail})
    for package, expected in PINNED_PACKAGES.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        checks.append({
            "name": f"version:{package}", "ok": installed == expected,
            "detail": f"{installed}; required {expected}",
        })
    device = str(config.get("device", "cuda"))
    if device.startswith("cuda") and importlib.util.find_spec("torch") is not None:
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
            checks.append({"name": "cuda", "ok": cuda_ok, "detail": torch.cuda.get_device_name(0) if cuda_ok else "unavailable"})
        except Exception as error:
            checks.append({"name": "cuda", "ok": False, "detail": f"{type(error).__name__}: {error}"})
    elif device.startswith("cuda"):
        checks.append({"name": "cuda", "ok": False, "detail": "torch unavailable"})
    return {"ready": all(item["ok"] for item in checks), "checks": checks, "profile": PROFILE}


class ProductSessionIO:
    """Atomically publish the latest planned and actually executed simulator state."""

    def __init__(self, output_dir: Path, stop_file: Path, config_identity: dict[str, Any]):
        self.output_dir = output_dir
        self.stop_file = stop_file
        self.config_identity = config_identity
        self.parent_pid = os.getppid()
        self.live: dict[str, Any] = {
            "schema": "sentinel-product-live-v1", "status": "preparing", "stage": "config",
            "profile": PROFILE, "branch": "delta_evc", "candidate": None, "actual": None,
            "aggregation": config_identity["product_config"].get("aggregation", "native"),
            "decision": None, "cursors": {"submitted": 0, "accepted": 0, "observed": 0},
            "frames": [],
            "task_success": False, "updated_at": _utc_now(),
        }
        self.exporter: Any | None = None
        self.progress("preparing", "config")
        self._write_live()

    def _write_live(self) -> None:
        self.live["updated_at"] = _utc_now()
        _atomic_json(self.output_dir / "product-live.json", self.live)

    def progress(self, status: str, stage: str, **details: Any) -> None:
        payload = {
            "schema": "sentinel-product-progress-v1", "status": status, "stage": stage,
            "profile": PROFILE, "updated_at": _utc_now(), **details,
        }
        _atomic_json(self.output_dir / "product-progress.json", payload)
        self.live.update({"status": status, "stage": stage})
        self._write_live()

    def stop_requested(self, stage: str) -> bool:
        config_path = Path(self.config_identity["product_config_path"])
        protocol_path = Path(self.config_identity["protocol_config"])
        if _sha256(config_path) != self.config_identity["product_config_sha256"]:
            raise RuntimeError("product session config changed during execution")
        if _sha256(protocol_path) != self.config_identity["protocol_config_sha256"]:
            raise RuntimeError("frozen protocol config changed during execution")
        if os.getppid() != self.parent_pid:
            self.progress("stopping", stage, stop_requested=True, reason="runner_parent_exited")
            return True
        if not self.stop_file.exists():
            return False
        self.progress("stopping", stage, stop_requested=True)
        return True

    def bind_scene(self, environment: Any, *, task_id: int, task_name: str, state_index: int, seed: int) -> None:
        from export_native_scene import NativeSceneExporter

        self.exporter = NativeSceneExporter.from_environment(
            environment, task_id, task_suite="libero_spatial", task_name=task_name
        )
        scene = self.exporter.export_scene()
        _atomic_json(self.output_dir / "viewer-model.json", scene)
        self.live.update({
            "task_id": task_id, "task_name": task_name, "initial_state_index": state_index,
            "seed": seed, "model_id": self.exporter.model_id,
            "actual": self.exporter.capture_frame(0, step=0),
        })
        self.live["frames"] = [self.live["actual"]]
        self.progress("running", "episode_ready")

    def publish_candidate(self, *, episode_id: str, cycle: int, step: int, action: Any, qpos: Any, decision: dict[str, Any], reason: str | None) -> None:
        self.progress("running", "candidate", cycle=cycle, step=step)
        self.live.update({
            "episode_id": episode_id, "cycle": cycle, "step": step,
            "candidate": {"action": action, "predicted_qpos": qpos},
            "decision": {"allowed": bool(decision.get("allowed")), "status": decision.get("status"), "reason": reason},
        })
        self._write_live()

    def publish_actual(self, *, step: int, cycle: int, action: Any, reward: float, decision: dict[str, Any], permit_id: str, success: bool) -> None:
        if self.exporter is None:
            raise RuntimeError("live scene was not bound before actual execution")
        observed = step + 1
        frame = self.exporter.capture_frame(observed, step=observed)
        frame["action"] = action
        if len(self.live["frames"]) >= 281:
            raise RuntimeError("product observed-frame budget exceeded")
        self.live["frames"].append(frame)
        frame["reward"] = reward
        frame["permitId"] = permit_id
        self.live.update({
            "status": "running", "stage": "executing", "cycle": cycle, "step": observed,
            "actual": frame, "decision": decision,
            "cursors": {"submitted": observed, "accepted": observed, "observed": observed},
            "task_success": bool(success),
        })
        self._write_live()

    def finalize(self, *, status: str, result: dict[str, Any]) -> None:
        self.progress(status, "finalized", steps=int(result.get("steps", 0)), task_success=bool(result.get("success")))
        self.live.update({"status": status, "stage": "finalized", "task_success": bool(result.get("success"))})
        self._write_live()
        artifacts = {}
        for name in ("product-progress.json", "product-live.json", "viewer-model.json", "manifest.json"):
            path = self.output_dir / name
            if path.is_file():
                artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        _atomic_json(self.output_dir / "product-manifest.json", {
            "schema": "sentinel-product-session-manifest-v1", "status": status,
            "profile": PROFILE, "branch": "delta_evc",
            "claim_boundary": "MuJoCo/LIBERO product session; no physical-stop or functional-safety claim",
            "config_identity": self.config_identity, "result": result, "artifacts": artifacts,
            "finished_at": _utc_now(),
        })


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if args.check:
        result = check(config_path)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ready"] else 2
    if args.output_dir is None or args.stop_file is None:
        raise ValueError("--output-dir and --stop-file are required unless --check is used")
    readiness = check(config_path)
    if not readiness["ready"]:
        print(json.dumps(readiness, sort_keys=True))
        return 2
    output_dir = args.output_dir.expanduser().resolve()
    stop_file = args.stop_file.expanduser().resolve()
    arguments = [
        str(HERE / "run_unified_libero.py"), "--config", str(config_path),
        "--product-session", "--output-dir", str(output_dir),
        "--stop-file", str(stop_file),
    ]
    # Keep policy, simulator and writer in this one owned process. A wrapper
    # with an independent grandchild cannot establish writer exit on shutdown.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sentinel_evc.runstore import _WorkspaceLock
    try:
        ownership = _WorkspaceLock(stop_file.parent.parent / "execution.lock")
    except (OSError, RuntimeError, ValueError):
        print(json.dumps({"status": "failed", "reason": "execution_workspace_busy"}))
        return 2
    original_argv = sys.argv
    try:
        readiness = check(config_path)
        if not readiness["ready"]:
            print(json.dumps(readiness, sort_keys=True))
            return 2
        import run_unified_libero
        sys.argv = arguments
        return run_unified_libero.main()
    finally:
        sys.argv = original_argv
        ownership.close()


if __name__ == "__main__":
    raise SystemExit(main())
