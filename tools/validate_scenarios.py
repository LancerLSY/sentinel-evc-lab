#!/usr/bin/env python3
"""Run reproducible reliability scenarios through the shipped production paths.

The fast numeric runs use the product's logical clock.  Their durations are not
robot real-time measurements.  The harness intentionally records expected and
actual outcomes so fail-closed negative cases remain successful validations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinel_evc.evidence import verify_bundle  # noqa: E402
from sentinel_evc.geometry import full_check  # noqa: E402
from sentinel_evc.numeric_world import candidate_actions, generate_candidates, make_numeric_case  # noqa: E402
from sentinel_evc.prediction import (  # noqa: E402
    build_default_numeric_artifacts,
    evaluate_candidates,
)
from sentinel_evc.product_pipeline import ProductManager  # noqa: E402
from sentinel_evc.runstore import RunStore  # noqa: E402
from sentinel_evc.scenario import Scenario  # noqa: E402
from sentinel_evc.ssh_experiment import _public_source_manifest_sha256  # noqa: E402


SCHEMA = "sentinel-additional-scenarios-v1"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _source_digest() -> str:
    return "sha256:" + _public_source_manifest_sha256(ROOT)


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("output directory must be empty")
    else:
        path.mkdir(parents=True)


def _run_product(output: Path, label: str, scenario: Scenario) -> tuple[dict, Path]:
    artifact = output / "artifacts" / label
    store = RunStore(artifact)
    manager = ProductManager(store, realtime=False)
    try:
        created = manager.start(scenario)
        session = manager._sessions[created["id"]]
        session.thread.join(30)
        if session.thread.is_alive():
            raise TimeoutError(label + " did not finish")
        record = manager.read(created["id"])
        run_dir = store.directory(created["id"])
        okay, detail = verify_bundle(
            str(run_dir / "bundle"), str(run_dir / "anchors" / "demo.public"), created["id"]
        )
        if not okay:
            raise ValueError(label + " evidence failed independent verification: " + detail)
        archive = store.export(created["id"])
        evidence = {
            "manifest": _digest(run_dir / "bundle" / "manifest.json"),
            "events": _digest(run_dir / "bundle" / "events.jsonl"),
            "export": _digest(archive),
            "verified": True,
        }
        compact = {
            "status": record["status"],
            "error_code": (record.get("error") or {}).get("code"),
            "selected": record.get("selected"),
            "candidate_statuses": [row["status"] for row in record["result"]["candidates"]],
            "candidate_reasons": [row["reason_code"] for row in record["result"]["candidates"]],
            "cursors": record["result"]["cursors"],
            "observed_steps": len(record["result"]["steps"]),
        }
        return {"actual": compact, "evidence": evidence}, run_dir
    finally:
        manager.close()
        store.close()


def _case(case_id: str, scope: str, expected: dict, actual: dict, evidence=None) -> dict:
    passed = all(actual.get(key) == value for key, value in expected.items())
    return {
        "case_id": case_id,
        "scope": scope,
        "expected": expected,
        "actual": actual,
        "evidence": evidence or {},
        "passed": passed,
    }


def _product_cases(output: Path, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        result, _ = _run_product(output, f"physical-seed-{seed}", Scenario(seed=seed))
        rows.append(_case(
            f"physical-covered-seed-{seed}",
            "fixed 0.35 m numeric candidate family; logical clock, not robot real time",
            {"status": "completed", "error_code": None, "observed_steps": 40},
            result["actual"], result["evidence"],
        ))

    result, _ = _run_product(output, "risk-rejection", Scenario(seed=7, risk_limit=.001))
    rows.append(_case(
        "strict-risk-rejection",
        "physical baseline with a consequence limit below every calibrated envelope",
        {"status": "rejected", "error_code": "NO_ALLOWED_CANDIDATE", "candidate_reasons": ["RISK_LIMIT"] * 4},
        result["actual"], result["evidence"],
    ))

    result, _ = _run_product(output, "residual-conservative", Scenario(seed=7, prediction_mode="residual"))
    rows.append(_case(
        "residual-conservative-rejection",
        "bundled stdlib residual baseline under the default 0.12 m consequence limit",
        {"status": "rejected", "error_code": "NO_ALLOWED_CANDIDATE", "candidate_reasons": ["RISK_LIMIT"] * 4},
        result["actual"], result["evidence"],
    ))

    blocked_scene = {
        "obstacles": [{"center": [.175, 0, 0], "radius": .03}],
        "workspace": {"lo": [-.5, -.5, -.5], "hi": [1, .5, .5]},
        "tool_radius": .01,
        "tracking_reserve": .005,
    }
    result, _ = _run_product(output, "obstacle-blocked", Scenario.parse({"seed": 7, "scene": blocked_scene}))
    rows.append(_case(
        "obstacle-blocks-all-actions",
        "static spherical obstacle intersects every fixed candidate path",
        {"status": "rejected", "error_code": "NO_ALLOWED_CANDIDATE", "candidate_reasons": ["PHYSICAL_REJECTED"] * 4},
        result["actual"], result["evidence"],
    ))

    result, _ = _run_product(output, "changed-displacement", Scenario(seed=7, displacement=.2, risk_limit=.5))
    rows.append(_case(
        "changed-displacement-is-unknown",
        "0.20 m actions are outside the frozen calibrated 0.35 m candidate family",
        {"status": "rejected", "error_code": "NO_ALLOWED_CANDIDATE", "candidate_reasons": ["MODEL_UNKNOWN"] * 4},
        result["actual"], result["evidence"],
    ))
    return rows


def _translation_case() -> dict:
    case = make_numeric_case(7)
    model, calibration = build_default_numeric_artifacts(mode="physical")
    base = generate_candidates()
    translated = generate_candidates(start=(.1, -.1, .1))
    scene = Scenario.parse({"scene": {
        "obstacles": [], "workspace": {"lo": [-.5, -.5, -.5], "hi": [1, .5, .5]},
        "tool_radius": .01, "tracking_reserve": .005,
    }}).scene
    normal = evaluate_candidates(case.history, base, model, calibration, 1, physical_check=lambda plan: full_check(plan, scene)[0])
    moved = evaluate_candidates(case.history, translated, model, calibration, 1, physical_check=lambda plan: full_check(plan, scene)[0])
    action_deltas = [
        max(abs(x - y) for x, y in zip(candidate_actions(a.plan), candidate_actions(b.plan)))
        for a, b in zip(normal.candidates, moved.candidates)
    ]
    prediction_deltas = [
        max(abs(x - y) for x, y in zip(a.prediction.centers, b.prediction.centers))
        for a, b in zip(normal.candidates, moved.candidates)
    ]
    actual = {
        "base_allowed": normal.allowed_count,
        "translated_allowed": moved.allowed_count,
        "equivalent_actions": [delta <= 1e-12 for delta in action_deltas],
        "equivalent_predictions": [delta <= 1e-12 for delta in prediction_deltas],
        "same_decisions": [a.status == b.status for a, b in zip(normal.candidates, moved.candidates)],
        "translated_geometry_allowed": [row.physical_allowed for row in moved.candidates],
    }
    return _case(
        "translated-scene-action-equivalence",
        "the calibrated action family is translation invariant when geometry remains supported",
        {"base_allowed": 1, "translated_allowed": 1, "equivalent_actions": [True] * 4, "equivalent_predictions": [True] * 4, "same_decisions": [True] * 4, "translated_geometry_allowed": [True] * 4},
        actual,
        {"base_evaluation": _json_digest(normal.summary()), "translated_evaluation": _json_digest(moved.summary())},
    )


def _persistence_case(output: Path) -> dict:
    artifact = output / "artifacts" / "persistence-recovery"
    _, run_dir = _run_product(output, "persistence-recovery", Scenario(seed=11))
    run_id = run_dir.name
    signed_result = run_dir / "bundle" / "result.json"
    before = _digest(signed_result)

    restarted_store = RunStore(artifact)
    try:
        restarted = restarted_store.read(run_id)
        restarted_export = restarted_store.export(run_id)
        restarted_exported = restarted_export.is_file()
    finally:
        restarted_store.close()

    tampered = json.loads(signed_result.read_text("utf-8"))
    tampered["tampered"] = True
    signed_result.write_text(json.dumps(tampered, sort_keys=True), "utf-8")
    after = _digest(signed_result)

    store = RunStore(artifact)
    manager = ProductManager(store, realtime=False)
    export_refused = False
    try:
        invalid = store.read(run_id)
        try:
            store.export(run_id)
        except ValueError:
            export_refused = True
        created = manager.start(Scenario(seed=23))
        session = manager._sessions[created["id"]]
        session.thread.join(30)
        if session.thread.is_alive():
            raise TimeoutError("recovery run did not finish")
        recovered = manager.read(created["id"])
        recovery_dir = store.directory(created["id"])
        verified, _ = verify_bundle(str(recovery_dir / "bundle"), str(recovery_dir / "anchors" / "demo.public"), created["id"])
        recovery_export = store.export(created["id"])
        actual = {
            "restart_read_before_tamper": restarted["status"],
            "restart_export_before_tamper": restarted_exported,
            "tampered_status": invalid["status"],
            "tampered_error_code": (invalid.get("error") or {}).get("code"),
            "tampered_export_refused": export_refused,
            "independent_next_run_status": recovered["status"],
            "independent_next_run_verified": verified,
        }
        evidence = {
            "signed_result_before_tamper": before,
            "signed_result_after_tamper": after,
            "next_manifest": _digest(recovery_dir / "bundle" / "manifest.json"),
            "next_export": _digest(recovery_export),
        }
    finally:
        manager.close()
        store.close()
    return _case(
        "restart-tamper-refusal-and-recovery",
        "persisted signed run is rechecked after restart; tampering fails closed without poisoning a new run",
        {"restart_read_before_tamper": "completed", "restart_export_before_tamper": True, "tampered_status": "failed", "tampered_error_code": "EVIDENCE_INVALID", "tampered_export_refused": True, "independent_next_run_status": "completed", "independent_next_run_verified": True},
        actual, evidence,
    )


def run_validation(output: Path, seeds: tuple[int, ...]) -> dict:
    _prepare_output(output)
    rows = _product_cases(output, seeds)
    rows.append(_translation_case())
    rows.append(_persistence_case(output))
    raw = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
    (output / "cases.jsonl").write_bytes(raw)
    metadata = {
        "schema": SCHEMA,
        "source_digest": _source_digest(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "clock_scope": "numeric scenario matrix uses deterministic logical execution time; no robot real-time claim",
        "seeds": list(seeds),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), "utf-8")
    failures = [row["case_id"] for row in rows if not row["passed"]]
    summary = {
        "schema": SCHEMA,
        "status": "completed" if not failures else "failed",
        "case_count": len(rows),
        "passed": len(rows) - len(failures),
        "failed": failures,
        "cases_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "metadata_sha256": _digest(output / "metadata.json"),
        "claims": {"real_robot": False, "vla": False, "robot_realtime": False, "mujoco": False},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="0,7,23")
    args = parser.parse_args(argv)
    try:
        seeds = tuple(int(value) for value in args.seeds.split(","))
    except ValueError as exc:
        parser.error("--seeds must be comma-separated integers")
    if not seeds or any(seed < 0 or seed > 1_000_000 for seed in seeds):
        parser.error("--seeds must contain supported product seeds")
    summary = run_validation(args.out, seeds)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
