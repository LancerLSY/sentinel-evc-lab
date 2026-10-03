#!/usr/bin/env python3
"""Bounded randomized drill for the public NativeGateway request boundary.

This is an experiment runner, not a test suite.  It freezes its protocol before
executing any case and writes every case as JSONL so the aggregate can be
recomputed without trusting the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sentinel_evc.events import EventLog
from sentinel_evc.native_gateway import (
    NativeActionProfile,
    NativeContext,
    NativeGateway,
    NativeGatewayDenied,
    NativeSnapshot,
)


MASTER_SEED = 0x5E71_1E1C
SAMPLES_PER_CLASS = 200
BASE_TIME_NS = 1_000_000_000
LEASE_TTL_NS = 250_000_000
MAX_FEEDBACK_AGE_NS = 2_000_000_000

CASES = (
    ("action_ulp_replaced", "ACTION_REPLACED"),
    ("permit_replay", "LEASE_REPLAY"),
    ("permit_expired", "LEASE_EXPIRED"),
    ("feedback_replaced", "FEEDBACK_CHANGED"),
    ("context_replaced", "CONTEXT_CHANGED"),
    ("generation_revoked", "GENERATION_REVOKED"),
    ("evidence_gap", "EVIDENCE_GAP"),
    ("legal_control", "ALLOWED"),
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def action_digest(action: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(action.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(canonical_bytes(list(action.shape)))
    h.update(b"\0")
    h.update(action.tobytes(order="C"))
    return "sha256:" + h.hexdigest()


def case_seed(case_index: int, sample_index: int) -> int:
    raw = f"{MASTER_SEED}:{case_index}:{sample_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def profile() -> NativeActionProfile:
    return NativeActionProfile(
        "native-fault-drill-7d-v1",
        tuple(f"component_{index}" for index in range(7)),
        (-1.0,) * 7,
        (1.0,) * 7,
        "opaque normalized 7-D request used only for gateway contract drilling",
    )


def make_rig(seed: int, *, maxlen: int = 64) -> dict[str, Any]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    log = EventLog(
        f"native-fault-{seed:016x}", schema_version="product-v1", maxlen=maxlen
    )
    secret = hashlib.sha256(f"secret:{seed}".encode("ascii")).digest()
    gateway = NativeGateway(
        profile(),
        log,
        lease_ttl_ns=LEASE_TTL_NS,
        max_feedback_age_ns=MAX_FEEDBACK_AGE_NS,
        secret=secret,
    )
    snapshot = NativeSnapshot(
        f"feedback-{seed:016x}-0",
        digest_text(f"observation:{seed}:0"),
        0,
        BASE_TIME_NS,
    )
    base_context = NativeContext(
        f"robot-{rng.randrange(8)}",
        f"boot-{rng.randrange(4)}",
        f"episode-{seed:016x}",
        f"scene-{rng.randrange(16)}",
        f"controller-{rng.randrange(4)}",
        rng.choice(("approach", "transfer", "place")),
        0,
        dependencies_hash=digest_text(f"dependencies:{seed}"),
    )
    gateway.observe_feedback(snapshot, base_context)
    context = NativeContext(
        base_context.robot_id,
        base_context.boot_id,
        base_context.episode_id,
        base_context.scene_id,
        base_context.controller_id,
        base_context.task_phase,
        0,
        dependencies_hash=base_context.dependencies_hash,
        chunk_id=rng.randrange(1_000_000),
        action_index_in_chunk=rng.randrange(1_000),
        policy_input_hash=digest_text(f"policy:{seed}:0"),
        policy_input_step=0,
    )
    gateway.refresh_context(context, snapshot_hash=snapshot.digest)
    action = np_rng.uniform(-0.9, 0.9, size=(1, 7)).astype(np.float32)
    return {
        "rng": rng,
        "gateway": gateway,
        "log": log,
        "snapshot": snapshot,
        "context": context,
        "action": action,
    }


class SpyWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = 0

    def __call__(self, action: np.ndarray, entered) -> str:
        self.calls += 1
        entered()
        self.entered += 1
        return "stepped"


def run_case(case_name: str, expected: str, seed: int, sample_index: int) -> dict[str, Any]:
    rig = make_rig(seed)
    gateway: NativeGateway = rig["gateway"]
    log: EventLog = rig["log"]
    snapshot: NativeSnapshot = rig["snapshot"]
    context: NativeContext = rig["context"]
    action: np.ndarray = rig["action"]
    writer = SpyWriter()
    permit = gateway.authorize(
        action, snapshot=snapshot, context=context, now_ns=BASE_TIME_NS + 1
    )
    submitted_action = action
    submitted_snapshot = snapshot
    submitted_context = context
    submitted_now = BASE_TIME_NS + 2
    mutation: dict[str, Any] | None = None

    if case_name == "action_ulp_replaced":
        submitted_action = action.copy()
        component = rig["rng"].randrange(7)
        original_value = float(submitted_action[0, component])
        direction = np.float32(1.0 if submitted_action[0, component] < 0.99 else -1.0)
        submitted_action[0, component] = np.nextafter(
            submitted_action[0, component], direction, dtype=np.float32
        )
        mutation = {
            "component": component,
            "original_float32_le": struct.pack("<f", original_value).hex(),
            "submitted_float32_le": struct.pack(
                "<f", float(submitted_action[0, component])
            ).hex(),
            "operation": "numpy.nextafter float32 (one representable step)",
        }
    elif case_name == "permit_replay":
        gateway.submit(
            permit,
            action,
            snapshot=snapshot,
            context=context,
            writer=writer,
            now_ns=BASE_TIME_NS + 2,
        )
        submitted_now = BASE_TIME_NS + 3
    elif case_name == "permit_expired":
        submitted_now = permit.deadline_mono_ns + 1
    elif case_name == "feedback_replaced":
        fresh_snapshot = NativeSnapshot(
            f"feedback-{seed:016x}-1",
            digest_text(f"observation:{seed}:1"),
            1,
            BASE_TIME_NS + 3,
        )
        fresh_context = NativeContext(
            context.robot_id,
            context.boot_id,
            context.episode_id,
            context.scene_id,
            context.controller_id,
            context.task_phase,
            1,
            dependencies_hash=context.dependencies_hash,
            chunk_id=context.chunk_id,
            action_index_in_chunk=context.action_index_in_chunk,
            policy_input_hash=digest_text(f"policy:{seed}:1"),
            policy_input_step=1,
        )
        gateway.observe_feedback(fresh_snapshot, fresh_context)
        submitted_now = BASE_TIME_NS + 4
    elif case_name == "context_replaced":
        fresh_context = NativeContext(
            context.robot_id,
            context.boot_id,
            context.episode_id,
            context.scene_id,
            context.controller_id,
            context.task_phase,
            context.queue_rev,
            dependencies_hash=context.dependencies_hash,
            chunk_id=context.chunk_id,
            action_index_in_chunk=context.action_index_in_chunk + 1,
            policy_input_hash=digest_text(f"policy:{seed}:replacement"),
            policy_input_step=context.policy_input_step,
        )
        gateway.refresh_context(fresh_context, snapshot_hash=snapshot.digest)
    elif case_name == "generation_revoked":
        gateway.revoke("fault-drill", now_ns=BASE_TIME_NS + 2)
        submitted_now = BASE_TIME_NS + 3
    elif case_name == "evidence_gap":
        log.note_gap("fault-drill injected evidence gap")

    calls_before = writer.calls
    entered_before = writer.entered
    try:
        gateway.submit(
            permit,
            submitted_action,
            snapshot=submitted_snapshot,
            context=submitted_context,
            writer=writer,
            now_ns=submitted_now,
        )
        actual = "ALLOWED"
    except NativeGatewayDenied as exc:
        actual = exc.code
    except BaseException as exc:  # keep unexpected evidence visible in the raw artifact
        actual = f"UNEXPECTED:{type(exc).__name__}"

    target_calls = writer.calls - calls_before
    target_entries = writer.entered - entered_before
    expected_calls = 1 if expected == "ALLOWED" else 0
    valid_entry = (
        actual == expected
        and target_calls == expected_calls
        and target_entries == expected_calls
    )
    return {
        "case": case_name,
        "sample_index": sample_index,
        "seed": seed,
        "expected_code": expected,
        "actual_code": actual,
        "writer_calls_before_target": calls_before,
        "writer_calls_on_target": target_calls,
        "writer_entries_on_target": target_entries,
        "writer_calls_total": writer.calls,
        "valid_entry": valid_entry,
        "action_hash": action_digest(action),
        "submitted_action_hash": action_digest(submitted_action),
        "mutation": mutation,
        "context_hash": context.digest,
        "log_has_gap": log.has_gap,
        "retained_event_types": [event["type"] for event in log.events()],
    }


def concurrency_probe(seed: int) -> dict[str, Any]:
    rig = make_rig(seed)
    gateway: NativeGateway = rig["gateway"]
    log: EventLog = rig["log"]
    snapshot: NativeSnapshot = rig["snapshot"]
    context: NativeContext = rig["context"]
    action: np.ndarray = rig["action"]
    permit = gateway.authorize(
        action, snapshot=snapshot, context=context, now_ns=BASE_TIME_NS + 1
    )
    entered_event = threading.Event()
    release_event = threading.Event()
    result: list[str] = []
    calls = 0

    def writer(_action, entered):
        nonlocal calls
        calls += 1
        entered()
        entered_event.set()
        release_event.wait(2.0)
        return "stepped"

    def submitter() -> None:
        try:
            gateway.submit(
                permit,
                action,
                snapshot=snapshot,
                context=context,
                writer=writer,
                now_ns=BASE_TIME_NS + 2,
            )
            result.append("ALLOWED")
        except BaseException as exc:
            result.append(f"UNEXPECTED:{type(exc).__name__}")

    thread = threading.Thread(target=submitter, name="native-fault-drill-submit")
    thread.start()
    entered_observed = entered_event.wait(2.0)
    if entered_observed:
        gateway.revoke("concurrent-revoke", now_ns=BASE_TIME_NS + 3)
    release_event.set()
    thread.join(2.0)
    event_types = [event["type"] for event in log.events()]
    valid = (
        entered_observed
        and not thread.is_alive()
        and result == ["ALLOWED"]
        and calls == 1
        and "REVOKE" in event_types
        and "CONTROLLER_ACK" in event_types
    )
    return {
        "name": "concurrent_revoke_after_entry",
        "seed": seed,
        "result": result,
        "writer_calls": calls,
        "entered_observed": entered_observed,
        "thread_completed": not thread.is_alive(),
        "generation_after_revoke": gateway.generation,
        "retained_event_types": event_types,
        "valid": valid,
    }


def baseexception_probe(seed: int) -> dict[str, Any]:
    rig = make_rig(seed)
    gateway: NativeGateway = rig["gateway"]
    log: EventLog = rig["log"]
    snapshot: NativeSnapshot = rig["snapshot"]
    context: NativeContext = rig["context"]
    action: np.ndarray = rig["action"]
    permit = gateway.authorize(
        action, snapshot=snapshot, context=context, now_ns=BASE_TIME_NS + 1
    )
    calls = 0

    def writer(_action, entered):
        nonlocal calls
        calls += 1
        entered()
        raise KeyboardInterrupt("fault-drill")

    actual = None
    try:
        gateway.submit(
            permit,
            action,
            snapshot=snapshot,
            context=context,
            writer=writer,
            now_ns=BASE_TIME_NS + 2,
        )
        actual = "RETURNED"
    except KeyboardInterrupt:
        actual = "KeyboardInterrupt"
    acks = log.by_type("CONTROLLER_ACK")
    valid = (
        actual == "KeyboardInterrupt"
        and calls == 1
        and len(acks) == 1
        and acks[0]["payload"].get("accepted") is False
        and acks[0]["payload"].get("status") == "writer_raised"
        and acks[0]["payload"].get("error_type") == "KeyboardInterrupt"
    )
    return {
        "name": "baseexception_after_entry",
        "seed": seed,
        "actual": actual,
        "writer_calls": calls,
        "ack_payloads": [
            {
                "accepted": event["payload"].get("accepted"),
                "status": event["payload"].get("status"),
                "error_type": event["payload"].get("error_type"),
            }
            for event in acks
        ],
        "retained_event_types": [event["type"] for event in log.events()],
        "valid": valid,
    }


def source_manifest(repo: Path, script: Path) -> dict[str, Any]:
    files = (
        script,
        repo / "src/sentinel_evc/native_gateway.py",
        repo / "src/sentinel_evc/events.py",
        repo / "src/sentinel_evc/contracts.py",
    )
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    return {
        "git_head": git_head,
        "files": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in files
        },
    }


def default_output_dir(script: Path) -> Path:
    task_root = script.resolve().parents[3]
    return task_root / "results/native-fault-drill"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    repo = script.parents[2]
    output_dir = (args.output_dir or default_output_dir(script)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_id": "sentinel-native-fault-drill-v1",
        "frozen_before_execution": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "case_count": len(CASES),
        "total_randomized_cases": len(CASES) * SAMPLES_PER_CLASS,
        "cases": [
            {"name": name, "expected_code": expected} for name, expected in CASES
        ],
        "entry_criterion": {
            "denied_case": "actual_code equals expected_code and target writer calls/entries are both zero",
            "legal_case": "actual_code is ALLOWED and target writer calls/entries are both one",
            "aggregate": "zero invalid entries across all frozen cases",
        },
        "independent_probes": [
            "concurrent revoke after writer entry",
            "KeyboardInterrupt after writer entry",
        ],
        "claim_boundary": [
            "deterministic bounded protocol, not a statistical robustness rate",
            "no comparison with another product",
            "NativeGateway schema/bounds/evidence contract only; no dynamics or functional-safety claim",
        ],
        "source": source_manifest(repo, script),
    }
    protocol["protocol_sha256"] = "sha256:" + hashlib.sha256(
        canonical_bytes(protocol)
    ).hexdigest()
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    raw_path = output_dir / "raw.jsonl"
    results: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for case_index, (name, expected) in enumerate(CASES):
            for sample_index in range(SAMPLES_PER_CLASS):
                seed = case_seed(case_index, sample_index)
                record = run_case(name, expected, seed, sample_index)
                results.append(record)
                raw.write(json.dumps(record, sort_keys=True) + "\n")

    independent = [
        concurrency_probe(case_seed(len(CASES), 0)),
        baseexception_probe(case_seed(len(CASES), 1)),
    ]
    by_case: dict[str, Any] = {}
    for name, expected in CASES:
        selected = [record for record in results if record["case"] == name]
        by_case[name] = {
            "expected_code": expected,
            "cases": len(selected),
            "actual_codes": dict(sorted(Counter(r["actual_code"] for r in selected).items())),
            "invalid_entries": sum(not r["valid_entry"] for r in selected),
            "target_writer_calls": sum(r["writer_calls_on_target"] for r in selected),
        }
    invalid = [record for record in results if not record["valid_entry"]]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "source": protocol["source"],
        "total_cases": len(results),
        "invalid_entries": len(invalid),
        "all_frozen_cases_valid": not invalid,
        "by_case": by_case,
        "independent_probes": independent,
        "all_independent_probes_valid": all(item["valid"] for item in independent),
        "raw_jsonl_sha256": sha256_file(raw_path),
        "claim_boundary": protocol["claim_boundary"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_lines = []
    for name in ("protocol.json", "raw.jsonl", "summary.json"):
        path = output_dir / name
        manifest_lines.append(f"{sha256_file(path)[7:]}  {name}")
    (output_dir / "manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not invalid and summary["all_independent_probes_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
