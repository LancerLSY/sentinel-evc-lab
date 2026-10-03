"""Recorded-action qualification for a VLA deployment adapter.

The experiment replays signed camera commitments, state values, chunk cursors
and final float32 actions through a deterministic adapter.  It does not load a
model, recover camera pixels, run physics, or command a robot.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import statistics
import struct
import subprocess
import tempfile
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from sentinel_evc.contracts import canonical_json
from sentinel_evc.evidence import verify_bundle
from sentinel_evc.launch_gate import (
    QualifiedNativeWriter,
    build_qualification_capsule,
    freeze_probe_pack,
    qualify,
    reproduce_qualification_capsule,
)
from sentinel_evc.native_gateway import NativeGatewayDenied
from sentinel_evc.scenario import strict_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "sentinel-launch-qualification-experiment-v1"
SOURCE_SCHEMA = "sentinel-launch-recorded-inputs-v1"
STATE_NAMES = [
    "eef_x", "eef_y", "eef_z",
    "eef_axis_angle_x", "eef_axis_angle_y", "eef_axis_angle_z",
    "gripper_left", "gripper_right",
]
STATE_UNITS = ["m", "m", "m", "rad", "rad", "rad", "m", "m"]
CASES = (
    "unchanged",
    "identity_only",
    "mismatched_input",
    "camera_swap",
    "camera_route_renamed",
    "state_index_swap",
    "state_units_scaled",
    "action_order_swap",
    "sign_inversion",
    "gripper_inversion",
    "chunk_off_by_one",
    "postprocess_scale",
    "action_byte_change",
)
SEMANTIC_FAULTS = CASES[3:]
QUALIFICATION_SAMPLES = 200
MAX_SOURCE_FILES = 160
MAX_SOURCE_FILE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 768 * 1024 * 1024
SAFE_SOURCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    """Strict JSON using Python's shortest-roundtrip finite-float encoding."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _save(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value) + b"\n")


def _empty_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)


def _tensor_identity(dtype: str, shape: list[int], raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json(shape))
    digest.update(b"\0")
    digest.update(raw)
    return "sha256:" + digest.hexdigest()


def _float32_bytes(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _selected_episodes(result: dict[str, Any]) -> dict[str, int]:
    selected: dict[str, int] = {}
    for task in range(10):
        rows = [row for row in result.get("episodes", [])
                if row.get("task_id") == task and row.get("initial_state_index") == 46]
        if len(rows) != 1 or not isinstance(rows[0].get("episode_id"), str):
            raise RuntimeError(f"expected one signed task {task}/state 46 episode")
        selected[rows[0]["episode_id"]] = task
    return selected


def _read_bounded_regular(path: Path, maximum: int, *, exact: int | None = None) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"source is not a regular file: {path.name}")
        if metadata.st_size > maximum or (exact is not None and metadata.st_size != exact):
            raise RuntimeError(f"source size is outside its bound: {path.name}")
        chunks: list[bytes] = []
        length = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - length))
            if not block:
                break
            chunks.append(block)
            length += len(block)
            if length > maximum:
                raise RuntimeError(f"source grew beyond its bound: {path.name}")
        if exact is not None and length != exact:
            raise RuntimeError(f"source length changed while reading: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _snapshot_source(bundle: Path, public_key: Path) -> dict[str, Any]:
    """Snapshot and authenticate every source byte consumed by this experiment."""
    public_key_bytes = _read_bounded_regular(public_key, 32, exact=32)
    manifest_bytes = _read_bounded_regular(bundle / "manifest.json", 1024 * 1024)
    signature_bytes = _read_bounded_regular(bundle / "manifest.sig", 64, exact=64)
    manifest = strict_json(manifest_bytes)
    Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature_bytes, manifest_bytes)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("run_id"), str):
        raise RuntimeError("source manifest is not a signed run manifest")
    declared = manifest.get("files")
    if (not isinstance(declared, dict) or not 1 <= len(declared) <= MAX_SOURCE_FILES
            or any(not isinstance(name, str) or not SAFE_SOURCE_NAME.fullmatch(name)
                   for name in declared)):
        raise RuntimeError("signed source file set is outside the bounded snapshot schema")
    assets: dict[str, bytes] = {}
    total = 0
    for name, digest in declared.items():
        raw = _read_bounded_regular(bundle / name, MAX_SOURCE_FILE_BYTES)
        total += len(raw)
        if total > MAX_SOURCE_TOTAL_BYTES:
            raise RuntimeError("signed source snapshot exceeds 768 MiB")
        if _sha(raw) != digest:
            raise RuntimeError(f"snapshotted {name} does not match the signed manifest")
        assets[name] = raw
    required = {"result.json", "native_trace.jsonl", "replay.json", "events.jsonl"}
    if not required.issubset(assets):
        raise RuntimeError("signed source snapshot lacks launch-probe inputs or events")
    with tempfile.TemporaryDirectory(prefix="sentinel-launch-source-") as temporary:
        root = Path(temporary)
        snapshot_bundle = root / "bundle"
        snapshot_anchor = root / "source.public"
        snapshot_bundle.mkdir()
        snapshot_anchor.write_bytes(public_key_bytes)
        for name, raw in assets.items():
            (snapshot_bundle / name).write_bytes(raw)
        (snapshot_bundle / "manifest.json").write_bytes(manifest_bytes)
        (snapshot_bundle / "manifest.sig").write_bytes(signature_bytes)
        ok, reason = verify_bundle(str(snapshot_bundle), str(snapshot_anchor), manifest["run_id"])
        if not ok:
            raise RuntimeError("source snapshot verification failed: " + reason)
    return {
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "signature_bytes": signature_bytes,
        "public_key_bytes": public_key_bytes,
        "assets": {name: assets[name] for name in
                   ("result.json", "native_trace.jsonl", "replay.json")},
        "bundle_bytes": total + len(manifest_bytes) + len(signature_bytes),
        "verification": reason,
    }


def _read_recorded_inputs(result_raw: bytes, trace_raw: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = strict_json(result_raw)
    episodes = _selected_episodes(result)
    events: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    with io.BytesIO(trace_raw) as stream:
        for line in stream:
            value = strict_json(line)
            episode_id = value.get("episode_id")
            step = value.get("step")
            event = value.get("event")
            if episode_id in episodes and step in (0, 1) and event in {
                "policy_input", "selected_normalized_action", "official_postprocessed_action"
            }:
                slot = events.setdefault((episode_id, step), {})
                if event in slot:
                    raise RuntimeError(f"duplicate {event} in {episode_id} step {step}")
                slot[event] = value

    records: list[dict[str, Any]] = []
    selected_identities: dict[str, str] = {}
    for episode_id, task in sorted(episodes.items(), key=lambda item: item[1]):
        for step in (0, 1):
            slot = events.get((episode_id, step), {})
            if set(slot) != {"policy_input", "selected_normalized_action", "official_postprocessed_action"}:
                raise RuntimeError(f"incomplete signed trace for {episode_id} step {step}")
            policy, selected, final = (slot[name] for name in (
                "policy_input", "selected_normalized_action", "official_postprocessed_action"))
            observation = policy["observation"]
            state = observation["observation.state"]
            state_values = state.get("values")
            if (state.get("dtype") != "float32" or state.get("shape") != [1, 8]
                    or not isinstance(state_values, list) or len(state_values) != 1
                    or len(state_values[0]) != 8):
                raise RuntimeError("recorded state is outside the frozen eight-component adapter")
            cameras = {
                key: observation[key]["sha256"]
                for key in ("observation.images.image", "observation.images.image2")
            }
            camera_rows = [observation[key]
                           for key in ("observation.images.image", "observation.images.image2")]
            if any(row.get("dtype") != "float32" or row.get("shape") != [1, 3, 360, 360]
                   for row in camera_rows):
                raise RuntimeError("recorded camera commitments are outside the frozen same-shape routes")
            normalized_values = selected.get("values")
            if not isinstance(normalized_values, list) or len(normalized_values) != 1:
                raise RuntimeError("selected normalized action is missing")
            normalized_raw = _float32_bytes(normalized_values[0])
            if _tensor_identity("float32", [1, 7], normalized_raw) != selected["identity"]["sha256"]:
                raise RuntimeError("selected normalized action identity does not reproduce")
            final_values = final.get("values")
            if not isinstance(final_values, list) or len(final_values) != 7:
                raise RuntimeError("official postprocessed action is missing")
            action_raw = _float32_bytes(final_values)
            if _tensor_identity("float32", [1, 7], action_raw) != final["identity"]["sha256"]:
                raise RuntimeError("official postprocessed action identity does not reproduce")
            if selected.get("chunk_id") != final.get("chunk_id") or selected.get("action_index_in_chunk") != final.get("action_index_in_chunk"):
                raise RuntimeError("selected and postprocessed cursor differ")
            probe_id = f"task{task:02d}-state46-step{step:02d}"
            selected_identities[probe_id] = selected["identity"]["sha256"]
            records.append({
                "id": probe_id,
                "episode_id": episode_id,
                "task_id": task,
                "step": step,
                "input_hash": policy["observation_hash"],
                "cameras": cameras,
                "state": {"names": list(STATE_NAMES), "units": list(STATE_UNITS),
                          "values": [float(value) for value in state_values[0]]},
                "cursor": {"chunk_id": final["chunk_id"],
                           "action_index": final["action_index_in_chunk"]},
                "action": {"dtype": "float32", "shape": [1, 7], "byte_order": "little",
                           "bytes_hex": action_raw.hex()},
            })
    if len(records) != 20:
        raise RuntimeError(f"expected 20 recorded probes, found {len(records)}")
    return records, selected_identities


class RecordedActionAdapter:
    """Practical replay adapter over signed commitments and recorded actions.

    A production runner can pass the same fields after real preprocessing and
    postprocessing.  Here they come from a verified trace, so the experiment
    measures adapter mapping compatibility without pretending to rerun vision.
    """

    def __init__(self, mode: str):
        if mode not in CASES:
            raise ValueError("unsupported adapter mode")
        self.mode = mode

    def observe(self, source: dict[str, Any]) -> dict[str, Any]:
        cameras = dict(source["cameras"])
        names = list(source["state"]["names"])
        units = list(source["state"]["units"])
        values = list(source["state"]["values"])
        cursor = dict(source["cursor"])
        action = dict(source["action"])
        raw = bytearray.fromhex(action["bytes_hex"])
        floats = list(struct.unpack("<7f", raw))
        input_hash = source["input_hash"]

        if self.mode == "mismatched_input":
            input_hash = _sha((input_hash + ":different-capture").encode())
        elif self.mode == "camera_swap":
            first, second = sorted(cameras)
            cameras[first], cameras[second] = cameras[second], cameras[first]
        elif self.mode == "camera_route_renamed":
            value = cameras.pop("observation.images.image")
            cameras["observation.images.front"] = value
        elif self.mode == "state_index_swap":
            for sequence in (names, units, values):
                sequence[0], sequence[1] = sequence[1], sequence[0]
        elif self.mode == "state_units_scaled":
            units[0] = "mm"
            values[0] *= 1000.0
        elif self.mode == "action_order_swap":
            floats[0], floats[1] = floats[1], floats[0]
            raw = bytearray(_float32_bytes(floats))
        elif self.mode == "sign_inversion":
            floats[0] = -floats[0]
            raw = bytearray(_float32_bytes(floats))
        elif self.mode == "gripper_inversion":
            floats[6] = -floats[6]
            raw = bytearray(_float32_bytes(floats))
        elif self.mode == "chunk_off_by_one":
            cursor["action_index"] += 1
        elif self.mode == "postprocess_scale":
            floats[:6] = [value * 0.5 for value in floats[:6]]
            raw = bytearray(_float32_bytes(floats))
        elif self.mode == "action_byte_change":
            first_bits = struct.unpack("<I", raw[:4])[0]
            raw[:4] = struct.pack("<I", first_bits ^ 1)

        return {
            "id": source["id"],
            "input_hash": input_hash,
            "consumed": {"cameras": cameras,
                         "state": {"names": names, "units": units, "values": values}},
            "cursor": cursor,
            "action": {**action, "bytes_hex": bytes(raw).hex()},
        }


def _adapter_digest(mode: str) -> str:
    identity = "identity_only_refactor" if mode == "identity_only" else mode
    return _sha(_json_bytes({
        "implementation": "RecordedActionAdapter-v1",
        "source_sha256": _sha(Path(__file__).read_bytes()),
        "identity": identity,
    }))


def _make_pack(records: list[dict[str, Any]], mode: str, provenance: dict[str, Any]) -> bytes:
    adapter = RecordedActionAdapter(mode)
    probes = [adapter.observe(record) for record in records]
    pack = {
        "schema": "sentinel-vla-probe-pack-v1",
        "suite_id": provenance["suite_id"],
        "adapter_digest": _adapter_digest(mode),
        "provenance": {**provenance, "adapter_mode": mode,
                       "capture": "signed recorded-action adapter replay"},
        "probes": probes,
    }
    return freeze_probe_pack(pack)


def prepare(args: argparse.Namespace) -> None:
    _empty_directory(args.output)
    snapshot = _snapshot_source(args.bundle, args.public_key)
    manifest = snapshot["manifest"]
    records, selected_identities = _read_recorded_inputs(
        snapshot["assets"]["result.json"], snapshot["assets"]["native_trace.jsonl"])
    suite_material = [{key: value for key, value in record.items() if key != "episode_id"}
                      for record in records]
    suite_id = _sha(_json_bytes(suite_material))
    source = {
        "schema": SOURCE_SCHEMA,
        "suite_id": suite_id,
        "source": {
            "run_id": manifest["run_id"],
            "manifest_sha256": _sha(snapshot["manifest_bytes"]),
            "signature_sha256": _sha(snapshot["signature_bytes"]),
            "public_key_sha256": _sha(snapshot["public_key_bytes"]),
            "native_trace_sha256": _sha(snapshot["assets"]["native_trace.jsonl"]),
            "verification": snapshot["verification"],
        },
        "selected_normalized_action_identities": selected_identities,
        "records": records,
    }
    protocol = {
        "schema": SCHEMA,
        "suite_id": suite_id,
        "selection": "tasks 0..9, state 46, policy steps 0 and 1",
        "probe_count": len(records),
        "cases": list(CASES),
        "semantic_faults": list(SEMANTIC_FAULTS),
        "qualification_samples_per_case": QUALIFICATION_SAMPLES,
        "scope": "recorded adapter mapping; no new inference, pixels, physics or robot motion",
        "source_files": {
            "experiments/product/launch_qualification.py": _sha(Path(__file__).read_bytes()),
            "src/sentinel_evc/launch_gate.py": _sha((ROOT / "src/sentinel_evc/launch_gate.py").read_bytes()),
            "src/sentinel_evc/launch_diagnosis.py": _sha((ROOT / "src/sentinel_evc/launch_diagnosis.py").read_bytes()),
        },
    }
    _save(args.output / "source.json", source)
    _save(args.output / "protocol.json", protocol)
    print(json.dumps({"prepared": True, "probe_count": len(records),
                      "suite_id": suite_id, "output": str(args.output)}, indent=2))


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def _measure(reference: bytes, candidate: bytes) -> dict[str, int]:
    samples: list[int] = []
    for _ in range(QUALIFICATION_SAMPLES):
        start = time.perf_counter_ns()
        qualify(reference, candidate)
        samples.append(time.perf_counter_ns() - start)
    return {
        "samples": len(samples),
        "mean_ns": round(statistics.fmean(samples)),
        "p50_ns": _percentile(samples, 0.50),
        "p95_ns": _percentile(samples, 0.95),
        "max_ns": max(samples),
    }


def _writer_probe(reference: bytes, candidate: bytes, current_digest: str) -> dict[str, Any]:
    downstream: list[bytes] = []
    entered: list[bool] = []
    try:
        writer = QualifiedNativeWriter(reference, candidate, lambda: current_digest,
                                       lambda request: downstream.append(request) or "software-receipt")
        receipt = writer(b"recorded-request", lambda: entered.append(True))
        decision, reason = "DISPATCHED", None
    except NativeGatewayDenied as exc:
        receipt, decision, reason = None, "DENIED", exc.code
    return {"decision": decision, "reason": reason, "entry_acknowledgements": len(entered),
            "downstream_calls": len(downstream), "receipt": receipt}


def _digest_drift_probe(reference: bytes, candidate: bytes, expected: str) -> dict[str, Any]:
    calls = 0
    downstream: list[bytes] = []
    entered: list[bool] = []

    def current() -> str:
        nonlocal calls
        calls += 1
        return expected if calls == 1 else _sha(b"adapter changed after qualification")

    writer = QualifiedNativeWriter(reference, candidate, current,
                                   lambda request: downstream.append(request) or "unexpected")
    try:
        writer(b"recorded-request", lambda: entered.append(True))
        decision, reason = "DISPATCHED", None
    except NativeGatewayDenied as exc:
        decision, reason = "DENIED", exc.code
    return {"decision": decision, "reason": reason, "digest_reads": calls,
            "entry_acknowledgements": len(entered), "downstream_calls": len(downstream)}


def run(args: argparse.Namespace) -> None:
    _empty_directory(args.output)
    source = strict_json((args.data / "source.json").read_bytes())
    protocol = strict_json((args.data / "protocol.json").read_bytes())
    if source.get("schema") != SOURCE_SCHEMA or protocol.get("schema") != SCHEMA:
        raise RuntimeError("prepared launch qualification data has an unsupported schema")
    current_sources = {
        "experiments/product/launch_qualification.py": _sha(Path(__file__).read_bytes()),
        "src/sentinel_evc/launch_gate.py": _sha((ROOT / "src/sentinel_evc/launch_gate.py").read_bytes()),
        "src/sentinel_evc/launch_diagnosis.py": _sha((ROOT / "src/sentinel_evc/launch_diagnosis.py").read_bytes()),
    }
    if current_sources != protocol.get("source_files"):
        raise RuntimeError("experiment or launch-gate source changed after prepare")
    snapshot = _snapshot_source(args.bundle, args.public_key)
    current_source = {
        "run_id": snapshot["manifest"]["run_id"],
        "manifest_sha256": _sha(snapshot["manifest_bytes"]),
        "signature_sha256": _sha(snapshot["signature_bytes"]),
        "public_key_sha256": _sha(snapshot["public_key_bytes"]),
        "native_trace_sha256": _sha(snapshot["assets"]["native_trace.jsonl"]),
        "verification": snapshot["verification"],
    }
    if current_source != source.get("source"):
        raise RuntimeError("signed source snapshot differs from the prepared source")
    records, selected_identities = _read_recorded_inputs(
        snapshot["assets"]["result.json"], snapshot["assets"]["native_trace.jsonl"])
    suite_material = [{key: value for key, value in record.items() if key != "episode_id"}
                      for record in records]
    rebuilt_source = {
        "schema": SOURCE_SCHEMA,
        "suite_id": _sha(_json_bytes(suite_material)),
        "source": current_source,
        "selected_normalized_action_identities": selected_identities,
        "records": records,
    }
    if _json_bytes(rebuilt_source) != _json_bytes(source):
        raise RuntimeError("prepared probe inputs differ from the fresh signed source snapshot")
    provenance = {
        "suite_id": source["suite_id"],
        "source_run_id": source["source"]["run_id"],
        "source_manifest_sha256": source["source"]["manifest_sha256"],
        "source_native_trace_sha256": source["source"]["native_trace_sha256"],
        "probe_window": "tasks 0..9, state 46, steps 0 and 1",
        "camera_material": "signed tensor identity commitments; raw pixels unavailable",
    }
    packs_dir = args.output / "packs"
    capsules_dir = args.output / "capsules"
    packs_dir.mkdir()
    capsules_dir.mkdir()
    reference = _make_pack(records, "unchanged", provenance)
    (packs_dir / "reference.json").write_bytes(reference)
    reference_value = json.loads(reference)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        candidate = _make_pack(records, case, provenance)
        (packs_dir / f"{case}.json").write_bytes(candidate)
        candidate_value = json.loads(candidate)
        report = qualify(reference, candidate)
        first_reference = reference_value["probes"][0]
        first_candidate = candidate_value["probes"][0]
        if case in SEMANTIC_FAULTS and _json_bytes(first_reference) == _json_bytes(first_candidate):
            raise RuntimeError(f"fault {case} did not affect the first probe")
        expected = ("PASS" if case in {"unchanged", "identity_only"}
                    else "REVIEW" if case == "mismatched_input" else "BLOCK")
        if report["verdict"] != expected:
            raise RuntimeError(f"{case}: expected {expected}, got {report['verdict']}")
        writer = _writer_probe(reference, candidate, candidate_value["adapter_digest"])
        if report["verdict"] == "BLOCK" and writer["downstream_calls"] != 0:
            raise RuntimeError(f"blocked {case} reached the software writer")
        capsule_root = capsules_dir / case
        capsule_report, receipt = build_qualification_capsule(
            reference, candidate, capsule_root, f"launch-{case.replace('_', '-')}")
        reproduction = reproduce_qualification_capsule(
            receipt["bundle_dir"], receipt["public_key"], f"launch-{case.replace('_', '-')}")
        if capsule_report != report or reproduction["reproduction"] != "PASS":
            raise RuntimeError(f"{case}: signed capsule did not reproduce")
        rows.append({
            "case": case,
            "expected_verdict": expected,
            "report": report,
            "qualification_time": _measure(reference, candidate),
            "writer": writer,
            "capsule": {
                "bytes": _directory_bytes(capsule_root),
                "bundle_dir": os.path.relpath(receipt["bundle_dir"], args.output),
                "public_key": os.path.relpath(receipt["public_key"], args.output),
                "reproduction": reproduction,
            },
        })

    identity_pack = (packs_dir / "identity_only.json").read_bytes()
    identity_value = json.loads(identity_pack)
    drift = _digest_drift_probe(reference, identity_pack, identity_value["adapter_digest"])
    if drift["decision"] != "DENIED" or drift["reason"] != "ADAPTER_DIGEST_CHANGED" or drift["downstream_calls"]:
        raise RuntimeError("live adapter digest drift was not denied before downstream writing")

    selected = ["native_trace.jsonl", "result.json", "replay.json"]
    source_sizes = {
        "selected_signed_payload_bytes": sum(len(snapshot["assets"][name]) for name in selected),
        "selected_files": {name: len(snapshot["assets"][name]) for name in selected},
        "whole_source_bundle_bytes": snapshot["bundle_bytes"],
    }
    result = {
        "schema": SCHEMA,
        "source_commit": _git_commit(),
        "source_files": current_sources,
        "source": source["source"],
        "scope": {
            "measured": "deterministic recorded-action adapter mapping and software-writer launch gate",
            "not_measured": ["new model inference", "raw camera pixels", "task success",
                             "collision or dynamics", "physical robot motion or stop"],
        },
        "probe_count": len(records),
        "retained_windows": 10,
        "actions_per_window": 2,
        "cases": rows,
        "acceptance": {
            "unchanged_pass": rows[0]["report"]["verdict"] == "PASS",
            "identity_only_pass": rows[1]["report"]["verdict"] == "PASS",
            "mismatched_input_review": rows[2]["report"]["verdict"] == "REVIEW",
            "semantic_faults_blocked": sum(row["report"]["verdict"] == "BLOCK" for row in rows[3:]),
            "semantic_fault_count": len(SEMANTIC_FAULTS),
            "blocked_downstream_calls": sum(row["writer"]["downstream_calls"] for row in rows
                                             if row["report"]["verdict"] == "BLOCK"),
            "live_digest_drift": drift,
        },
        "sizes": {
            "reference_pack_bytes": len(reference),
            "all_candidate_pack_bytes": sum((packs_dir / f"{case}.json").stat().st_size for case in CASES),
            "all_capsules_bytes": _directory_bytes(capsules_dir),
            "source": source_sizes,
        },
        "measurement_host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "qualification_clock": "time.perf_counter_ns",
        },
        "comparison_scope": {
            "neighbor": "RLSOK binds release/configuration and canonical commands",
            "this_measurement": "automatic finite probe comparison plus a downstream writer deny on mismatch",
            "claim": "workflow and retained-case behavior only; not an overall competitor ranking",
        },
    }
    if result["acceptance"]["semantic_faults_blocked"] != len(SEMANTIC_FAULTS):
        raise RuntimeError("not every semantic adapter fault was blocked")
    _save(args.output / "result.json", result)
    shutil.copy2(args.data / "protocol.json", args.output / "protocol.json")
    print(json.dumps({"completed": True, "probe_count": len(records),
                      "semantic_faults_blocked": len(SEMANTIC_FAULTS),
                      "blocked_downstream_calls": 0,
                      "result": str(args.output / 'result.json')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument("--public-key", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--bundle", type=Path, required=True)
    run_parser.add_argument("--public-key", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
