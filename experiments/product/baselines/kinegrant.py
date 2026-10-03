#!/usr/bin/env python3
"""Run frozen action fixtures through installed KineGrant reference APIs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from kinegrant.adapters._context import trusted_context
from kinegrant.adapters.ros2 import ros_action_request
from kinegrant.capability import CapabilityIssuer
from kinegrant.canonical import canonical_json
from kinegrant.crypto import Ed25519KeyPair
from kinegrant.gate import ActionGate, SQLiteReplayStore
from kinegrant.gatekeeper import Gatekeeper
from kinegrant.models import ActionRequest, PolicyRule, parse_time
from kinegrant.policy import PolicyEngine
from kinegrant.receipt import ReceiptLog, verify_receipt_chain
from kinegrant.revocation import RevocationList
from kinegrant.sequence import ActionJournal, SequencePolicy

UPSTREAM_COMMIT = "3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f"
UPSTREAM_VERSION = "2.65.5"
PROTOCOL_SHA256 = "4b2d01d6f9b648fc8b778dd9c002b3c3a569ad1556b5e870a88167cb721bcc77"
INPUTS_SHA256 = "02d9decd745f215174bee8b6827f8050a1f91979081efed00bb84e7431f90a76"
PAYLOADS_SHA256 = "34fe19434a6e3d0de3eeef1059518f8e54507741324ff8c688428f351cef9acd"
EVIDENCE_HASH = f"sha256:{INPUTS_SHA256}"


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def mutate_hex(value: str) -> str:
    return ("00" if value[:2] != "00" else "ff") + value[2:]


def native_identity(dtype: str, shape: list[int], raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(
        dtype.encode("utf-8") + b"\0" + canonical_json(shape) + b"\0" + raw
    ).hexdigest()


def validate_row(row: dict[str, Any]) -> tuple[str, str]:
    if row["dtype"] != "float32" or row["shape"] != [1, 7]:
        raise ValueError(f"unexpected frozen action profile in {row['case_id']}")
    action_bytes = bytes.fromhex(row["action_hex"])
    if len(action_bytes) != 28:
        raise ValueError(f"unexpected float32 payload length in {row['case_id']}")
    raw_hash = "sha256:" + hashlib.sha256(action_bytes).hexdigest()
    identity = native_identity(row["dtype"], row["shape"], action_bytes)
    if identity != row["source_action_bytes_hash"]:
        raise ValueError(f"native identity mismatch in {row['case_id']}")
    return raw_hash, identity


def live_values(row: dict[str, Any], scenario: str) -> dict[str, Any]:
    feedback = row["source_feedback_hash"]
    context_digest = sha256_text(f"frozen-context:{row['task_id']}:{row['source_step']}")
    generation = f"{row['source_episode_id']}:generation:0"
    if scenario == "feedback_changed":
        feedback = sha256_text(f"receiver-feedback-changed:{row['case_id']}")
    if scenario == "context_changed":
        context_digest = sha256_text(f"receiver-context-changed:{row['case_id']}")
    if scenario == "revoked_generation":
        generation = f"{row['source_episode_id']}:generation:1"
    return {
        "receiver_feedback_hash": feedback,
        "receiver_context_digest": context_digest,
        "receiver_step": row["source_step"],
        "receiver_generation": generation,
    }


def request_for(row: dict[str, Any], scenario: str, submitted_hex: str | None = None) -> ActionRequest:
    action_hex = submitted_hex or row["submitted_action_hex"]
    caller_context = {
        "canonical_action": {
            "encoding": "hex",
            "bytes_hex": action_hex,
            "dtype": row["dtype"],
            "shape": row["shape"],
            "source_action_bytes_hash": row["source_action_bytes_hash"],
            "native_request_identity": native_identity(row["dtype"], row["shape"], bytes.fromhex(action_hex)),
        },
        "source_episode_id": row["source_episode_id"],
        "source_manifest_sha256": row["source_manifest_sha256"],
    }
    receiver_context = trusted_context(live_values(row, scenario), caller_context)
    return ros_action_request(
        node_identity="urn:kinegrant:agent:sentinel-vla",
        action_name="kg.action.move",
        physical_target=f"urn:kinegrant:robot:libero-panda:task-{row['task_id']}",
        purpose="frozen-vla-execution-comparison",
        request_id=f"urn:kinegrant:request:freeze-v1:task-{row['task_id']}:step-{row['source_step']}",
        namespace="/sentinel_vla",
        context=receiver_context,
    )


def setup(row: dict[str, Any], scratch: Path) -> tuple[ActionRequest, dict[str, Any], Gatekeeper, ReceiptLog, RevocationList]:
    authorized = request_for(row, "valid", row["action_hex"])
    authority = Ed25519KeyPair.generate()
    policy = PolicyRule(
        policy_id=f"urn:kinegrant:policy:freeze-v1:{row['case_id']}",
        issuer=authority.kid,
        target=authorized.target,
        effect="allow",
        actions=(authorized.action,),
        subjects=(authorized.agent,),
        purposes=(authorized.purpose,),
        constraints={"required_context": authorized.context, "min_approval_tier": 1},
        obligations=("emitActionReceipt",),
    )
    engine = PolicyEngine([policy], trusted_policy_issuers={authority.kid}, require_known_actions=True)
    decision = engine.evaluate(authorized)
    if not decision.allowed:
        raise RuntimeError(f"authorized request unexpectedly denied: {decision.reason}")
    capability = CapabilityIssuer(authority).issue_scoped(
        authorized, decision, ttl_seconds=1, approval_tier=decision.required_approval_tier, wire_version="1.0"
    )
    replay_path = scratch / f"{row['case_id']}.sqlite"
    replay_path.unlink(missing_ok=True)
    revocations = RevocationList()
    gate = ActionGate(trusted_issuers={authority.kid}, replay_store=SQLiteReplayStore(replay_path), revocation_list=revocations)
    receipts = ReceiptLog(Ed25519KeyPair.generate())
    keeper = Gatekeeper(gate=gate, sequence=SequencePolicy(()), journal=ActionJournal(), receipt_log=receipts, revocation_list=revocations)
    return authorized, capability, keeper, receipts, revocations


def config() -> dict[str, Any]:
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_version": UPSTREAM_VERSION,
        "action_request": "official ros_action_request + trusted_context",
        "policy": "default-deny, trusted allow, full required_context equality",
        "capability": "CapabilityIssuer.issue_scoped(wire_version=1.0, ttl_seconds=1)",
        "gate": "Gatekeeper + ActionGate + SQLiteReplayStore + RevocationList",
        "receipt": "ReceiptLog v1.0 obligation result, evidence_hash=inputs.json SHA-256",
        "adapter_protocol_sha256": PROTOCOL_SHA256,
    }


def receipt_ok(receipts: ReceiptLog) -> bool:
    return verify_receipt_chain(receipts.entries, trusted_executors={receipts.executor_key.kid})


def save_receipts(receipts: ReceiptLog, prefix: str, receipt_dir: Path) -> None:
    for index, envelope in enumerate(receipts.entries):
        (receipt_dir / f"{prefix}-{index}.json").write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(row: dict[str, Any], outcome: Any, counters: dict[str, Any], before_callback: int, before_dispatch: int, precondition_callback: int, precondition_dispatch: int, receipts: ReceiptLog, observed_hex: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "scenario": row["scenario"],
        "decision": "allow" if outcome.allowed else "deny",
        "reason": outcome.reason or outcome.stage,
        "stage": outcome.stage,
        "precondition_writer_calls": precondition_dispatch,
        "precondition_callback_invocations": precondition_callback,
        "callback_invocations": counters["callback_invocations"] - before_callback,
        "downstream_dispatch": counters["downstream_dispatch"] - before_dispatch,
        "writer_calls": counters["downstream_dispatch"] - before_dispatch,
        "receipt_count": len(receipts.entries),
        "receipt_chain_valid": receipt_ok(receipts),
        "writer_observed_hex": observed_hex,
        "upstream_version": UPSTREAM_VERSION,
        "configuration": config(),
        **extra,
    }


def run_case(row: dict[str, Any], scratch: Path, receipt_dir: Path) -> dict[str, Any]:
    recomputed_raw_sha, recomputed_native_identity = validate_row(row)
    authorized, capability, keeper, receipts, revocations = setup(row, scratch)
    scenario = row["scenario"]
    actual = authorized
    writer = {"callback_invocations": 0, "downstream_dispatch": 0, "observed_hex": None}

    def actuator(_verified: Any) -> None:
        writer["callback_invocations"] += 1
        writer["downstream_dispatch"] += 1
        writer["observed_hex"] = actual.context["canonical_action"]["bytes_hex"]

    precondition_callback, precondition_dispatch, now = 0, 0, None
    if scenario == "action_replacement":
        actual = request_for(row, scenario, row["submitted_action_hex"])
    elif scenario in {"feedback_changed", "context_changed", "revoked_generation"}:
        actual = request_for(row, scenario, row["submitted_action_hex"])
    elif scenario == "lease_replay":
        first = keeper.execute(capability, actual, actuator, evidence_hash=EVIDENCE_HASH, obligation_results=[{"obligation": "emitActionReceipt", "status": "satisfied"}])
        if not first.allowed:
            raise RuntimeError(f"replay precondition failed: {first.reason}")
        precondition_callback = writer["callback_invocations"]
        precondition_dispatch = writer["downstream_dispatch"]
    elif scenario == "expired_permit":
        now = parse_time(capability["payload"]["expires_at"])
    if scenario == "revoked_generation":
        revocations.revoke(capability["payload"]["capability_id"], reason="frozen generation revoked")
    before_callback = writer["callback_invocations"]
    before_dispatch = writer["downstream_dispatch"]
    outcome = keeper.execute(capability, actual, actuator, now=now, evidence_hash=EVIDENCE_HASH, obligation_results=[{"obligation": "emitActionReceipt", "status": "satisfied"}])
    save_receipts(receipts, row["case_id"], receipt_dir)
    return record(row, outcome, writer, before_callback, before_dispatch, precondition_callback, precondition_dispatch, receipts, writer["observed_hex"], authorized_request_digest=authorized.digest, actual_request_digest=actual.digest, fixture_declared_source_action_hash=row["source_action_bytes_hash"], recomputed_raw_sha256=recomputed_raw_sha, recomputed_native_identity=recomputed_native_identity, source_identity_matches=recomputed_native_identity == row["source_action_bytes_hash"])


def run_entry_race(row: dict[str, Any], kind: str, scratch: Path, receipt_dir: Path) -> dict[str, Any]:
    recomputed_raw_sha, recomputed_native_identity = validate_row(row)
    authorized, capability, keeper, receipts, _revocations = setup(row, scratch)
    writer = {"callback_invocations": 0, "downstream_dispatch": 0, "buffer_hex": row["action_hex"], "feedback": authorized.context["receiver_feedback_hash"], "context": authorized.context["receiver_context_digest"]}

    def actuator(_verified: Any) -> None:
        writer["callback_invocations"] += 1
        if kind == "caller_array_changed":
            writer["buffer_hex"] = mutate_hex(writer["buffer_hex"])
        elif kind == "entry_feedback_changed":
            writer["feedback"] = sha256_text(f"entry-feedback:{row['case_id']}")
        elif kind == "entry_context_changed":
            writer["context"] = sha256_text(f"entry-context:{row['case_id']}")
        elif kind == "entry_deadline_passed":
            time.sleep(1.05)
        writer["downstream_dispatch"] += 1

    outcome = keeper.execute(capability, authorized, actuator, evidence_hash=EVIDENCE_HASH, obligation_results=[{"obligation": "emitActionReceipt", "status": "satisfied"}])
    save_receipts(receipts, f"entryrace-{kind}-{row['case_id']}", receipt_dir)
    race_row = {**row, "case_id": f"entryrace-{kind}-{row['case_id']}", "scenario": kind}
    return record(race_row, outcome, writer, 0, 0, 0, 0, receipts, writer["buffer_hex"], authorized_request_digest=authorized.digest, actual_request_digest=authorized.digest, fixture_declared_source_action_hash=row["source_action_bytes_hash"], recomputed_raw_sha256=recomputed_raw_sha, recomputed_native_identity=recomputed_native_identity, source_identity_matches=recomputed_native_identity == row["source_action_bytes_hash"], writer_live_feedback_hash=writer["feedback"], writer_live_context_digest=writer["context"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-races", action="store_true")
    parser.add_argument("--entry-races-only", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.payloads.read_text(encoding="utf-8").splitlines() if line]
    scratch, receipt_dir = args.output.parent / "scratch", args.output.parent / "receipts"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if args.entry_races_only and not args.entry_races:
        parser.error("--entry-races-only requires --entry-races")
    records = [] if args.entry_races_only else [run_case(row, scratch, receipt_dir) for row in rows]
    if args.entry_races:
        valid_rows = [row for row in rows if row["scenario"] == "valid"]
        for kind in ("caller_array_changed", "entry_feedback_changed", "entry_context_changed", "entry_deadline_passed"):
            records.extend(run_entry_race(row, kind, scratch, receipt_dir) for row in valid_rows)
    args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    print(json.dumps({"records": len(records), "formal_records": 0 if args.entry_races_only else len(rows), "entry_race_records": len(records) if args.entry_races_only else len(records) - len(rows), "denied": sum(item["decision"] == "deny" for item in records), "downstream_dispatch": sum(item["downstream_dispatch"] for item in records), "receipt_chains_valid": all(item["receipt_chain_valid"] for item in records)}, sort_keys=True))


if __name__ == "__main__":
    main()
