"""Recorded-action experiment at a secure transport / application boundary.

NumPy is an experiment dependency. The product gateway does not import it.
DDS delivery and application admission are separate measured stages. This
experiment neither reruns policy inference nor commands a physical robot.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import secrets
import struct
import sys
import threading
import time

import numpy as np

from sentinel_evc.contracts import canonical_json
from sentinel_evc.evidence import build_bundle, verify_bundle
from sentinel_evc.events import EventLog
from sentinel_evc.native_gateway import (
    NativeActionProfile, NativeContext, NativeGateway, NativeGatewayDenied,
    NativeSnapshot, _native_request_identity,
)

FAULTS = ("valid", "action_replacement", "lease_replay", "expired_permit",
          "feedback_changed", "context_changed", "revoked_generation")
SYSTEMS = ("secure_dds", "dds_schema", "dds_hmac_nonce_ttl", "sentinel")
RACES = ("caller_array_changed", "entry_feedback_changed",
         "entry_context_changed", "entry_deadline_passed")
ORIGIN = 1_000_000_000
TTL = 250_000_000
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("src/sentinel_evc/native_gateway.py", "src/sentinel_evc/events.py",
                "src/sentinel_evc/contracts.py", "src/sentinel_evc/evidence.py",
                "experiments/product/execution_boundary.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def action_from(value) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(value["action_hex"]), dtype="<f4").copy().reshape(value["shape"])


def profile_from(value) -> NativeActionProfile:
    names = ("schema_id", "component_names", "lower", "upper", "semantics",
             "request_shape", "request_dtypes", "unsupported_checks")
    return NativeActionProfile(**{key: value[key] for key in names})


def fixture(case, profile, gateway_class=NativeGateway):
    log = EventLog("boundary-" + case["case_id"], schema_version="product-v1")
    gate = gateway_class(profile, log, lease_ttl_ns=TTL,
                         max_feedback_age_ns=1_500_000_000)
    snapshot = NativeSnapshot(case["source_episode_id"] + ":reset",
                              case["source_feedback_hash"], 0, ORIGIN)
    context = NativeContext("recorded-panda", "local-probe", case["source_episode_id"],
                            "libero-task-" + str(case["task_id"]), "instrumented-writer",
                            "recorded-action-probe", 0,
                            dependencies_hash="sha256:" + case["source_manifest_sha256"],
                            chunk_id=0, action_index_in_chunk=0,
                            policy_input_hash=case["source_feedback_hash"], policy_input_step=0)
    gate.observe_feedback(snapshot, context)
    return gate, log, snapshot, context


def prepare(args) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    ok, detail = verify_bundle(str(bundle), str(args.public_key), manifest["run_id"])
    if not ok:
        raise RuntimeError("source bundle verification failed: " + detail)
    replay = json.loads((bundle / "replay.json").read_text())
    result = json.loads((bundle / "result.json").read_text())
    inputs, messages = [], []
    for task in range(10):
        episode = next(e for e in replay["episodes"]
                       if e["task_id"] == task and e["initial_state_index"] == 46)
        reset, frame = episode["frames"][0:2]
        if reset["step"] != 0 or frame["step"] != 1:
            raise RuntimeError("source steps differ from the declared input grid")
        raw = struct.pack("<7f", *frame["action"])
        array = np.frombuffer(raw, dtype="<f4").reshape(1, 7)
        if _native_request_identity(array)[3] != frame["action_bytes_hash"]:
            raise RuntimeError("source float32 action reconstruction changed its identity")
        item = {"task_id": task, "source_episode_id": episode["episode_id"],
                "initial_state_index": 46, "source_step": 1,
                "source_feedback_hash": reset["observation_hash"],
                "source_manifest_sha256": sha((bundle / "manifest.json").read_bytes()),
                "shape": [1, 7], "dtype": "float32", "action_hex": raw.hex(),
                "source_action_bytes_hash": frame["action_bytes_hash"]}
        inputs.append(item)
        for fault in FAULTS:
            message = {**item, "case_id": f"task{task:02d}-{fault}", "scenario": fault}
            if fault == "action_replacement":
                changed = array.copy()
                changed.flat[0] += np.float32(0.125)
                message["submitted_action_hex"] = changed.tobytes(order="C").hex()
            else:
                message["submitted_action_hex"] = raw.hex()
            messages.append(message)
    protocol = {
        "schema": "sentinel-execution-boundary-protocol-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha((bundle / "manifest.json").read_bytes()),
        "source_public_key_sha256": sha(args.public_key.read_bytes()),
        "source_signature_sha256": sha((bundle / "manifest.sig").read_bytes()),
        "input_selection": "tasks 0..9, state 46, first postprocessed action (step 1)",
        "source_validation": {"verified": ok, "detail": detail},
        "source_files": {name: sha((ROOT / name).read_bytes()) for name in SOURCE_FILES},
        "profile": result["profile"], "systems": list(SYSTEMS),
        "scenarios": list(FAULTS), "cases_per_system": 70,
        "permit_lifetime_ns": TTL, "fixture_clock_origin_ns": ORIGIN,
        "entry_races": list(RACES), "entry_race_delay_seconds": 0.30,
        "race_implementations": ["9161cfd62625e5658b04d477d743ae6351c408b5", "current-source-digests"],
        "dds_upstream": {"version": "11.0.1", "commit": "e54e991f75a3e67f8e628da3171122e36ea5b872"},
        "stage_contract": "actual secure DDS delivery first; application gates then probe those exact received payloads with deterministic local state/clock injections",
        "writer_contract": "instrumented software writer records input digest and returns a receipt; no simulator or robot motion",
        "controls": ["DDS legal peer/topic", "DDS denied topic/identity", "schema wrong shape", "HMAC modified authenticator", "valid action for each gate"],
        "comparison_scope": "DDS is transport security. Schema/HMAC gates are experiment reference implementations, not installed competing robot products.",
    }
    save(out / "protocol.json", protocol)
    save(out / "inputs.json", inputs)
    (out / "payloads.ndjson").write_text("".join(json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" for m in messages))
    save(out / "freeze.json", {name: sha((out / name).read_bytes())
                               for name in ("protocol.json", "inputs.json", "payloads.ndjson")})
    print(json.dumps({"prepared_cases": len(messages), "protocol_sha256": sha((out / "protocol.json").read_bytes())}))


def gate_probe(system, case, profile):
    source_action = action_from(case)
    submitted = np.frombuffer(bytes.fromhex(case["submitted_action_hex"]), dtype="<f4").copy().reshape(1, 7)
    gate, log, snapshot, context = fixture(case, profile)
    permit = gate.authorize(source_action, snapshot=snapshot, context=context, now_ns=ORIGIN + 1_000_000)
    writers, feedback, callbacks, acknowledgements = [], [], [], []

    def writer(action, entered=lambda: None):
        callbacks.append(True)
        entered()
        acknowledgements.append(True)
        digest = _native_request_identity(action)[3]
        writers.append(digest)
        receipt = {"writer_input_hash": digest, "sequence": len(writers)}
        feedback.append(receipt)
        return receipt

    nonce = case["case_id"]
    secret = secrets.token_bytes(32)
    signed = {"action_hex": case["action_hex"], "nonce": nonce,
              "deadline_ns": ORIGIN + 1_000_000 + TTL,
              "snapshot_hash": snapshot.digest, "context_hash": context.digest,
              "generation": gate.generation}
    mac = hmac.new(secret, canonical_json(signed), hashlib.sha256).digest()
    consumed = set()

    def admit(action, now):
        if system == "secure_dds":
            return writer(action)
        profile.validate(_native_request_identity(action)[0])
        if action.shape != profile.request_shape or str(action.dtype) not in profile.request_dtypes:
            raise NativeGatewayDenied("UNSUPPORTED_ACTION_SCHEMA", "reference schema check")
        if system == "dds_schema":
            return writer(action)
        if system == "dds_hmac_nonce_ttl":
            if not hmac.compare_digest(mac, hmac.new(secret, canonical_json(signed), hashlib.sha256).digest()):
                raise NativeGatewayDenied("INVALID_MAC", "reference message authenticator")
            if action.tobytes(order="C").hex() != signed["action_hex"]:
                raise NativeGatewayDenied("ACTION_REPLACED", "reference exact-action binding")
            if now > signed["deadline_ns"]:
                raise NativeGatewayDenied("LEASE_EXPIRED", "reference deadline")
            if signed["nonce"] in consumed:
                raise NativeGatewayDenied("LEASE_REPLAY", "reference nonce")
            consumed.add(signed["nonce"])
            return writer(action)
        return gate.submit(permit, action, snapshot=snapshot, context=context,
                           writer=writer, now_ns=now)

    scenario = case["scenario"]
    now = ORIGIN + 2_000_000
    if scenario == "wrong_shape_control":
        submitted = submitted.reshape(7)
    elif scenario == "modified_authenticator_control":
        mac = bytes(32)
        permit = replace(permit, authenticator="hmac-sha256:" + "0" * 64)
    elif scenario == "lease_replay":
        admit(source_action, now)
    elif scenario == "expired_permit":
        now = ORIGIN + 1_000_000 + TTL + 1
    elif scenario == "feedback_changed":
        gate.observe_feedback(replace(snapshot, step=1, capture_mono_ns=ORIGIN + 1_500_000,
                                      feedback_id=snapshot.feedback_id + ":new"),
                              replace(context, queue_rev=1))
    elif scenario == "context_changed":
        gate.refresh_context(replace(context, chunk_id=1), snapshot_hash=snapshot.digest)
    elif scenario == "revoked_generation":
        gate.revoke("frozen receiver-side revocation", now_ns=ORIGIN + 1_500_000)
    precondition_calls = len(writers)
    precondition_callbacks = len(callbacks)
    precondition_acknowledgements = len(acknowledgements)
    start = time.perf_counter_ns()
    try:
        admit(submitted, now)
        decision, reason = "ACCEPTED", None
    except NativeGatewayDenied as error:
        decision, reason = "REJECTED", error.code
    elapsed = time.perf_counter_ns() - start
    result = {"system": system, "case_id": case["case_id"], "scenario": scenario,
              "task_id": case["task_id"], "transport_received": True,
              "decision": decision, "reason": reason,
              "precondition_writer_calls": precondition_calls,
              "callback_invocations": len(callbacks) - precondition_callbacks,
              "entry_acknowledgements": len(acknowledgements) - precondition_acknowledgements,
              "downstream_dispatches": len(writers) - precondition_calls,
              "writer_calls": len(writers) - precondition_calls,
              "writer_input_hashes": writers[precondition_calls:],
              "feedback_return_count": len(feedback) - precondition_calls,
              "gate_wall_ns": elapsed,
              "authorized_action_hash": _native_request_identity(source_action)[3],
              "submitted_action_hash": _native_request_identity(submitted)[3]}
    return result, log.to_jsonl() if system == "sentinel" else ""


def races(inputs, profile, old_source):
    spec = importlib.util.spec_from_file_location("sentinel_evc._boundary_original", old_source)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    rows, traces = [], []
    for item in inputs:
        for scenario in RACES:
            for label, module in (("original", old), ("hardened", sys.modules[NativeGateway.__module__])):
                case = {**item, "case_id": f"task{item['task_id']:02d}-{scenario}-{label}"}
                action = action_from(case)
                profile_local = module.NativeActionProfile(**{key: profile.summary()[key] for key in
                    ("schema_id", "component_names", "lower", "upper", "semantics", "request_shape", "request_dtypes", "unsupported_checks")})
                log = EventLog("race-" + case["case_id"], schema_version="product-v1")
                gate = module.NativeGateway(profile_local, log, lease_ttl_ns=TTL,
                                             max_feedback_age_ns=1_500_000_000)
                snapshot = module.NativeSnapshot("source-reset", item["source_feedback_hash"], 0, ORIGIN)
                context = module.NativeContext("panda", "local-probe", item["source_episode_id"], "scene",
                    "instrumented-writer", "probe", 0, chunk_id=0, action_index_in_chunk=0,
                    policy_input_hash=item["source_feedback_hash"], policy_input_step=0)
                gate.observe_feedback(snapshot, context)
                permit = gate.authorize(action, snapshot=snapshot, context=context, now_ns=ORIGIN + 1_000_000)
                waiting, proceed = threading.Event(), threading.Event()
                calls, errors, callbacks, acknowledgements = [], [], [], []

                def writer(request, entered):
                    callbacks.append(True)
                    waiting.set()
                    if not proceed.wait(3):
                        raise RuntimeError("race barrier timed out")
                    entered()
                    acknowledgements.append(True)
                    digest = _native_request_identity(request)[3]
                    calls.append(digest)
                    return {"writer_input_hash": digest}

                def submit():
                    try:
                        gate.submit(permit, action, snapshot=snapshot, context=context, writer=writer,
                                    now_ns=ORIGIN + 2_000_000)
                    except Exception as error:
                        errors.append({"type": type(error).__name__, "reason": getattr(error, "code", str(error))})

                thread = threading.Thread(target=submit)
                thread.start()
                if not waiting.wait(3):
                    proceed.set()
                    thread.join(3)
                    raise RuntimeError("writer did not reach the admission/entry barrier")
                if scenario == "caller_array_changed":
                    action.flat[0] += np.float32(0.125)
                elif scenario == "entry_feedback_changed":
                    gate.observe_feedback(replace(snapshot, step=1, capture_mono_ns=ORIGIN + 1_500_000,
                                                  feedback_id="new-feedback"), replace(context, queue_rev=1))
                elif scenario == "entry_context_changed":
                    gate.refresh_context(replace(context, chunk_id=1), snapshot_hash=snapshot.digest)
                else:
                    time.sleep(0.30)
                proceed.set()
                thread.join(3)
                if thread.is_alive():
                    raise RuntimeError("gateway did not leave the entry barrier")
                unauthorized = sum(digest != item["source_action_bytes_hash"] for digest in calls) if scenario == "caller_array_changed" else len(calls)
                rows.append({"implementation": label, "task_id": item["task_id"], "scenario": scenario,
                             "callback_invocations": len(callbacks),
                             "entry_acknowledgements": len(acknowledgements),
                             "downstream_dispatches": len(calls),
                             "writer_calls": len(calls), "unauthorized_writer_calls": unauthorized,
                             "authorized_action_hash": item["source_action_bytes_hash"],
                             "writer_input_hashes": calls, "errors": errors})
                traces.append(log.to_jsonl())
    return rows, "".join(traces)


def run(args) -> None:
    data = args.data
    freeze = json.loads((data / "freeze.json").read_text())
    for name, digest in freeze.items():
        if sha((data / name).read_bytes()) != digest:
            raise RuntimeError("frozen input changed: " + name)
    protocol = json.loads((data / "protocol.json").read_text())
    for name, digest in protocol["source_files"].items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise RuntimeError("frozen experiment source changed: " + name)
    messages = [json.loads(line) for line in (data / "payloads.ndjson").read_text().splitlines()]
    received = [json.loads(line) for line in args.received.read_text().splitlines()]
    expected = {m["case_id"]: m for m in messages}
    if len(received) != len(messages) or {m["case_id"] for m in received} != set(expected):
        raise RuntimeError("DDS did not deliver the complete unique frozen grid")
    for message in received:
        if message != expected[message["case_id"]]:
            raise RuntimeError("DDS payload differs from its frozen input")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    profile = profile_from(protocol["profile"])
    rows, traces, controls = [], [], []
    for system in SYSTEMS:
        for case in received:
            row, trace = gate_probe(system, case, profile)
            rows.append(row)
            traces.append(trace)
        if system != "secure_dds":
            for item in json.loads((data / "inputs.json").read_text()):
                for scenario in ("wrong_shape_control", "modified_authenticator_control"):
                    case = {**item, "case_id": f"task{item['task_id']:02d}-{scenario}",
                            "submitted_action_hex": item["action_hex"], "scenario": scenario}
                    row, trace = gate_probe(system, case, profile)
                    row["transport_received"] = False
                    row["derived_from_received_source_action"] = True
                    controls.append(row)
                    traces.append(trace)
    race_rows, race_trace = races(json.loads((data / "inputs.json").read_text()), profile, args.old_source)
    summary = {system: {scenario: {
        "cases": len([r for r in rows if r["system"] == system and r["scenario"] == scenario]),
        "rejected": sum(r["decision"] == "REJECTED" for r in rows if r["system"] == system and r["scenario"] == scenario),
        "writer_calls": sum(r["writer_calls"] for r in rows if r["system"] == system and r["scenario"] == scenario),
    } for scenario in FAULTS} for system in SYSTEMS}
    result = {"schema": "sentinel-execution-boundary-result-v1",
              "protocol_sha256": sha((data / "protocol.json").read_bytes()),
              "dds_received_file_sha256": sha(args.received.read_bytes()),
              "old_gateway_source_sha256": sha(args.old_source.read_bytes()),
              "source_files": protocol["source_files"], "numpy_version": np.__version__,
              "python_version": sys.version.split()[0], "summary": summary,
              "application_rows": rows, "application_controls": controls, "entry_races": race_rows,
              "scope": protocol["stage_contract"], "writer_contract": protocol["writer_contract"]}
    result["counter_contract"] = {
        "callback_invocations": "trusted writer wrapper invocation, before its preparation barrier",
        "entry_acknowledgements": "successful entered() return in the instrumented wrapper",
        "downstream_dispatches": "recorded software dispatch after successful entry acknowledgement",
        "writer_calls": "legacy alias of downstream_dispatches, not wrapper invocations",
        "unauthorized_writer_calls": "downstream dispatch with substituted bytes or after the injected stale-entry condition",
    }
    save(output / "result.json", result)
    (output / "gateway-traces.jsonl").write_text("".join(traces))
    (output / "entry-race-traces.jsonl").write_text(race_trace)
    audit = EventLog("execution-boundary-20261003", schema_version="product-v1")
    audit.append("OUTCOME", result_sha256=sha((output / "result.json").read_bytes()), protocol_sha256=result["protocol_sha256"])
    build_bundle(audit, str(output / "evidence"), artifacts={
        "result.json": (output / "result.json").read_bytes(),
        "protocol.json": (data / "protocol.json").read_bytes(),
        "inputs.json": (data / "inputs.json").read_bytes(),
        "received.ndjson": args.received.read_bytes(),
        "gateway-traces.jsonl": (output / "gateway-traces.jsonl").read_bytes(),
        "entry-race-traces.jsonl": (output / "entry-race-traces.jsonl").read_bytes(),
    })
    print(json.dumps({"dds_received": len(received), "application_rows": len(rows),
                      "entry_race_rows": len(race_rows), "summary": summary}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument("--public-key", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--received", type=Path, required=True)
    run_parser.add_argument("--old-source", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
