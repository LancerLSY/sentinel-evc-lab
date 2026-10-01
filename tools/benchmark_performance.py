"""Reproducible wall-clock benchmarks for the shipped Sentinel EVC paths.

The measurements are deliberately end to end:

* numeric runs use the realtime ProductManager through signed finalization and
  an independent bundle verification;
* geometry compares final-full-check and delta/fallback paths, with parent
  certificate creation included in both sides of every paired sample;
* physics runs the complete 25-trial MuJoCo experiment through signed index and
  trial verification.

This is a descriptive benchmark, not a hard-realtime or functional-safety
claim.  Run it on an otherwise idle host and retain the generated artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_evc.delta_cert import establish_root, validate_or_inherit  # noqa: E402
from sentinel_evc.evidence import build_bundle, verify_bundle  # noqa: E402
from sentinel_evc.events import EventLog  # noqa: E402
from sentinel_evc.product_pipeline import ProductManager  # noqa: E402
from sentinel_evc.runstore import RunStore  # noqa: E402
from sentinel_evc.scenario import Scenario  # noqa: E402
from sentinel_evc.scenarios import make_parent_pair, make_scene, mix, perturb  # noqa: E402


SCHEMA = "sentinel-performance-benchmark-v1"


def nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile without interpolation."""
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]


def duration_summary(values: list[float]) -> dict:
    if not values:
        return {"sample_count": 0, "status": "no_samples"}
    result = {
        "sample_count": len(values),
        "min_seconds": min(values),
        "p50_seconds": nearest_rank(values, 50),
        "p95_seconds": nearest_rank(values, 95),
        "p99_seconds": nearest_rank(values, 99),
        "max_seconds": max(values),
        "mean_seconds": sum(values) / len(values),
        "quantile_method": "nearest-rank",
    }
    if len(values) < 100:
        result["sample_size_note"] = (
            "With fewer than 100 samples, nearest-rank p99 is descriptive "
            "and may equal the observed maximum."
        )
    return result


def ratio_summary(values: list[float]) -> dict:
    if not values:
        return {"sample_count": 0, "status": "no_samples"}
    result = {
        "sample_count": len(values),
        "min_ratio": min(values),
        "p50_ratio": nearest_rank(values, 50),
        "p95_ratio": nearest_rank(values, 95),
        "p99_ratio": nearest_rank(values, 99),
        "max_ratio": max(values),
        "mean_ratio": sum(values) / len(values),
        "quantile_method": "nearest-rank",
    }
    if len(values) < 100:
        result["sample_size_note"] = (
            "With fewer than 100 samples, nearest-rank p99 is descriptive "
            "and may equal the observed maximum."
        )
    return result


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _source_metadata() -> dict:
    paths = sorted((ROOT / "src" / "sentinel_evc").rglob("*.py"))
    paths.extend([ROOT / "tools" / "benchmark_performance.py", ROOT / "pyproject.toml"])
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
        if path.is_file()
    ]
    status = _git_value("status", "--porcelain")
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_dirty": bool(status),
        "measured_source_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "measured_file_count": len(rows),
    }


def _environment_metadata() -> dict:
    packages = {}
    for name in ("cryptography", "mujoco"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware = {"processor": platform.processor() or None, "memory_bytes": None}
    if platform.system() == "Darwin":
        hardware["processor"] = _system_value("sysctl", "-n", "machdep.cpu.brand_string") or hardware["processor"]
        memory = _system_value("sysctl", "-n", "hw.memsize")
        hardware["memory_bytes"] = int(memory) if memory and memory.isdigit() else None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "hardware": hardware,
        "packages": packages,
        "clock": "time.perf_counter_ns monotonic wall clock",
    }


def _system_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _artifact_ref(path: Path, output: Path) -> str:
    return path.relative_to(output).as_posix()


def run_numeric_iteration(
    output: Path, mode: str, phase: str, index: int, *, realtime: bool = True
) -> dict:
    artifact = output / "artifacts" / "numeric" / mode / f"{phase}-{index:03d}"
    started = time.perf_counter_ns()
    store = manager = None
    record = None
    try:
        store = RunStore(artifact)
        manager = ProductManager(store, realtime=realtime)
        created = manager.start(
            Scenario(name=f"benchmark-{mode}", seed=7, prediction_mode=mode)
        )
        session = manager._sessions[created["id"]]
        session.thread.join(timeout=120 if realtime else 30)
        if session.thread.is_alive():
            raise TimeoutError("numeric benchmark did not finish")
        verification_started = time.perf_counter_ns()
        record = manager.read(created["id"])
        directory = store.directory(created["id"])
        verified, detail = verify_bundle(
            str(directory / "bundle"),
            str(directory / "anchors" / "demo.public"),
            created["id"],
        )
        if not verified:
            raise ValueError("numeric evidence verification failed: " + detail)
        verification_seconds = (time.perf_counter_ns() - verification_started) / 1_000_000_000
        result = record.get("result") or {}
        if (
            record.get("status") != "completed"
            or result.get("cursors") != {"submitted": 40, "accepted": 40, "observed": 40}
            or len(result.get("steps", ())) != 40
        ):
            raise ValueError("numeric run did not complete 40 observed steps")
        export_started = time.perf_counter_ns()
        archive = store.export(created["id"])
        export_seconds = (time.perf_counter_ns() - export_started) / 1_000_000_000
        if not archive.is_file():
            raise ValueError("numeric evidence export failed")
    finally:
        if manager is not None:
            manager.close()
        if store is not None:
            store.close()
    duration = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "benchmark": "numeric_realtime_signed_run",
        "mode": mode,
        "phase": phase,
        "iteration": index,
        "duration_seconds": duration,
        "status": record["status"],
        "observed_steps": record["result"]["cursors"]["observed"],
        "evidence_verified": True,
        "verification_seconds": verification_seconds,
        "export_seconds": export_seconds,
        "artifact": _artifact_ref(artifact, output),
    }


def _geometry_workload(
    cases: int, strategy: str, artifact: Path, run_id: str
) -> tuple[float, dict, dict]:
    counts = {
        "cases": cases,
        "root_full_checks": 0,
        "final_full_checks": 0,
        "fallback_full_checks": 0,
        "inherited": 0,
        "unsafe_rejected": 0,
        "safe_passed": 0,
    }
    stages = {
        "parent_certificate_seconds": 0.0,
        "final_validation_seconds": 0.0,
        "fallback_full_seconds": 0.0,
        "successful_inheritance_seconds": 0.0,
        "evidence_finalization_seconds": 0.0,
    }
    log = EventLog(run_id)
    started = time.perf_counter_ns()
    for case in range(cases):
        scene = make_scene(case % 50)
        parent, sibling = make_parent_pair(case)
        stage_started = time.perf_counter_ns()
        root = establish_root(parent, scene)
        stages["parent_certificate_seconds"] += (time.perf_counter_ns() - stage_started) / 1_000_000_000
        if root.verdict != "FULL" or root.certificate is None:
            raise ValueError("parent certificate creation failed")
        counts["root_full_checks"] += root.full_checks_used
        log.append(
            "CERTIFICATE",
            case=case,
            role="parent",
            strategy=strategy,
            **root.certificate.summary(),
        )
        if case % 2 == 0:
            child, transform = mix(parent, sibling, plan_id=f"MIX-{case:06d}")
            expected_safe = False
        else:
            child, transform = perturb(
                parent, seed=case, plan_id=f"NEAR-{case:06d}"
            )
            expected_safe = True
        log.append("TRANSFORM", case=case, strategy=strategy, **transform.summary())

        if strategy == "full":
            stage_started = time.perf_counter_ns()
            verdict = validate_or_inherit(child, scene)
            stages["final_validation_seconds"] += (time.perf_counter_ns() - stage_started) / 1_000_000_000
            counts["final_full_checks"] += verdict.full_checks_used
        elif strategy == "delta":
            stage_started = time.perf_counter_ns()
            verdict = validate_or_inherit(
                child, scene, parent, root.certificate, transform
            )
            stage_seconds = (time.perf_counter_ns() - stage_started) / 1_000_000_000
            stages["final_validation_seconds"] += stage_seconds
            if verdict.full_checks_used:
                stages["fallback_full_seconds"] += stage_seconds
            else:
                stages["successful_inheritance_seconds"] += stage_seconds
            counts["fallback_full_checks"] += verdict.full_checks_used
            counts["inherited"] += int(verdict.verdict == "INHERITED")
        else:
            raise ValueError("unknown geometry strategy")

        accepted = verdict.verdict in {"FULL", "INHERITED"}
        if accepted != expected_safe:
            raise ValueError("geometry benchmark correctness invariant failed")
        counts["safe_passed"] += int(expected_safe and accepted)
        counts["unsafe_rejected"] += int(not expected_safe and not accepted)
    evidence_started = time.perf_counter_ns()
    log.append("OUTCOME", strategy=strategy, counts=counts)
    encoded = json.dumps(
        {"strategy": strategy, "counts": counts},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    bundle = build_bundle(log, str(artifact), {"result.json": encoded})
    verified, detail = verify_bundle(bundle["bundle_dir"], bundle["public_key"], run_id)
    if not verified:
        raise ValueError("geometry evidence verification failed: " + detail)
    stages["evidence_finalization_seconds"] = (
        time.perf_counter_ns() - evidence_started
    ) / 1_000_000_000
    duration = (time.perf_counter_ns() - started) / 1_000_000_000
    return duration, counts, stages


def run_geometry_pair(output: Path, cases: int, phase: str, index: int) -> list[dict]:
    order = ("full", "delta") if index % 2 == 0 else ("delta", "full")
    rows = []
    for position, strategy in enumerate(order):
        artifact = output / "artifacts" / "geometry" / f"{phase}-{index:03d}" / strategy
        run_id = f"perf-geometry-{strategy}-{phase}-{index:03d}"
        duration, counts, stages = _geometry_workload(
            cases, strategy, artifact, run_id
        )
        rows.append(
            {
                "benchmark": "geometry_parent_and_final_validation",
                "strategy": strategy,
                "phase": phase,
                "iteration": index,
                "pair_order": position,
                "duration_seconds": duration,
                "status": "completed",
                "counts": counts,
                "stage_seconds": stages,
                "evidence_verified": True,
                "artifact": _artifact_ref(artifact, output),
            }
        )
    return rows


def run_physics_iteration(output: Path, phase: str, index: int, seed: int, friction: float) -> dict:
    from sentinel_evc.physics_experiment import run_physics_experiment
    from sentinel_evc.ssh_experiment import _verify_result_tree

    artifact = output / "artifacts" / "physics" / f"{phase}-{index:03d}"
    started = time.perf_counter_ns()
    summary = run_physics_experiment(artifact, seed=seed, friction=friction)
    verified = _verify_result_tree(artifact)
    duration = (time.perf_counter_ns() - started) / 1_000_000_000
    if verified.get("trial_count") != 25:
        raise ValueError("physics benchmark did not verify all 25 trials")
    trial_rows = list(summary["results"]) + [summary["integrated"]] + list(summary["faults"].values())
    return {
        "benchmark": "mujoco_25_trial_signed_experiment",
        "phase": phase,
        "iteration": index,
        "duration_seconds": duration,
        "status": "completed",
        "trial_count": verified["trial_count"],
        "signed_evidence_verified": True,
        "experiment_stage_seconds": summary["wall_time_s"],
        "simulated_trial_time_seconds": sum(row["physics_time_s"] for row in trial_rows),
        "infrastructure_gates_pass": bool(summary["infrastructure_gates_pass"]),
        "acceptance": summary["acceptance"],
        "scientific_observations": {
            "integrated_outcome": summary["integrated"]["outcome"],
            "failed_convergence_durations_seconds": [
                row["duration"] for row in summary["convergence"] if not row["passed"]
            ],
        },
        "artifact": _artifact_ref(artifact, output),
    }


def _write_json(path: Path, value) -> None:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _write_samples(path: Path, samples: list[dict]) -> None:
    raw = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in samples
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def _summary_for_samples(samples: list[dict]) -> dict:
    measured = [row for row in samples if row.get("phase") == "measured"]
    output = {}
    for mode in sorted({row["mode"] for row in measured if row["benchmark"].startswith("numeric")}):
        values = [row["duration_seconds"] for row in measured if row.get("mode") == mode]
        output[f"numeric_{mode}"] = {
            "scope": "realtime 40-observation ProductManager run, signed finalization, independent verification, and evidence export",
            "durations": duration_summary(values),
            "verification_stage": duration_summary(
                [row["verification_seconds"] for row in measured if row.get("mode") == mode]
            ),
            "export_stage": duration_summary(
                [row["export_seconds"] for row in measured if row.get("mode") == mode]
            ),
        }
    geometry = [row for row in measured if row["benchmark"].startswith("geometry")]
    if geometry:
        by_strategy = {
            strategy: [row for row in geometry if row["strategy"] == strategy]
            for strategy in ("full", "delta")
        }
        ratios = []
        differences = []
        for iteration in sorted({row["iteration"] for row in geometry}):
            pair = {row["strategy"]: row for row in geometry if row["iteration"] == iteration}
            ratios.append(pair["delta"]["duration_seconds"] / pair["full"]["duration_seconds"])
            differences.append(pair["delta"]["duration_seconds"] - pair["full"]["duration_seconds"])
        output["geometry"] = {
            "scope": "paired construction, parent certificate, transform, final full or delta-with-fallback validation, signed finalization, and independent verification",
            "full": duration_summary([row["duration_seconds"] for row in by_strategy["full"]]),
            "delta": duration_summary([row["duration_seconds"] for row in by_strategy["delta"]]),
            "paired_delta_over_full_ratio": ratio_summary(ratios),
            "paired_delta_minus_full_seconds": duration_summary(differences),
            "operation_counts_per_sample": {
                strategy: by_strategy[strategy][0]["counts"] for strategy in by_strategy
            },
            "stage_totals_per_sample": {
                strategy: {
                    stage: duration_summary([row["stage_seconds"][stage] for row in rows])
                    for stage in rows[0]["stage_seconds"]
                }
                for strategy, rows in by_strategy.items()
            },
            "claim_boundary": "Call-count reduction and wall-clock ratio are reported separately; neither is whole-system speedup.",
        }
    physics = [row for row in measured if row["benchmark"].startswith("mujoco")]
    if physics:
        output["physics"] = {
            "scope": "complete local 25-trial MuJoCo fixture, signed finalization, and independent verification",
            "durations": duration_summary([row["duration_seconds"] for row in physics]),
            "experiment_stage": duration_summary([row["experiment_stage_seconds"] for row in physics]),
            "simulated_trial_time": duration_summary([row["simulated_trial_time_seconds"] for row in physics]),
            "infrastructure_gates_pass_each_run": [row["infrastructure_gates_pass"] for row in physics],
            "acceptance_each_run": [row["acceptance"] for row in physics],
            "scientific_observations_each_run": [row["scientific_observations"] for row in physics],
            "claim_boundary": "Local numerical fixture only; no VLA checkpoint, robot arm, hard-realtime, or functional-safety claim.",
        }
    return output


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new or empty output directory")
    parser.add_argument("--numeric-modes", default="physical", help="comma-separated physical,residual")
    parser.add_argument("--numeric-repeats", type=_positive, default=10)
    parser.add_argument("--numeric-warmup", type=_nonnegative, default=1)
    parser.add_argument("--geometry-cases", type=_positive, default=1000)
    parser.add_argument("--geometry-repeats", type=_positive, default=20)
    parser.add_argument("--geometry-warmup", type=_nonnegative, default=2)
    parser.add_argument("--physics-repeats", type=_positive, default=3)
    parser.add_argument("--physics-warmup", type=_nonnegative, default=0)
    parser.add_argument("--physics-seed", type=int, default=7)
    parser.add_argument("--physics-friction", type=float, default=0.35)
    parser.add_argument("--skip-numeric", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-physics", action="store_true")
    args = parser.parse_args(argv)
    modes = tuple(part.strip() for part in args.numeric_modes.split(",") if part.strip())
    if not modes or any(mode not in {"physical", "residual"} for mode in modes):
        parser.error("--numeric-modes must contain physical and/or residual")
    if not math.isfinite(args.physics_friction) or not 0 < args.physics_friction <= 1:
        parser.error("--physics-friction must be finite and in (0, 1]")
    if args.skip_numeric and args.skip_geometry and args.skip_physics:
        parser.error("at least one benchmark must be enabled")
    args.numeric_modes = modes
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output = Path(args.out).expanduser()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("benchmark output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    failures = []
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": _source_metadata(),
        "environment": _environment_metadata(),
        "configuration": {
            "numeric_modes": list(args.numeric_modes),
            "numeric_repeats": args.numeric_repeats,
            "numeric_warmup": args.numeric_warmup,
            "geometry_cases": args.geometry_cases,
            "geometry_repeats": args.geometry_repeats,
            "geometry_warmup": args.geometry_warmup,
            "physics_repeats": args.physics_repeats,
            "physics_warmup": args.physics_warmup,
            "physics_seed": args.physics_seed,
            "physics_friction": args.physics_friction,
            "enabled": {
                "numeric": not args.skip_numeric,
                "geometry": not args.skip_geometry,
                "physics": not args.skip_physics,
            },
        },
        "methodology": {
            "execution": "sequential in one process; run on an otherwise idle host",
            "warmups": "recorded in raw samples and excluded from result quantiles",
            "geometry_order": "paired full/delta order alternates by iteration",
            "timing": "totals are measured directly; stage measurements are diagnostics and percentile totals are never formed by summing stage percentiles",
            "artifacts": "all successful and failed iteration directories are retained under artifacts/",
            "claim_boundary": "descriptive local wall-clock results; no hard-realtime, robot, VLA, or functional-safety claim",
        },
        "status": "running",
        "failures": failures,
        "results": {},
        "raw_samples": "samples.jsonl",
    }

    def persist() -> None:
        report["results"] = _summary_for_samples(samples)
        _write_samples(output / "samples.jsonl", samples)
        report["raw_samples_sha256"] = hashlib.sha256(
            (output / "samples.jsonl").read_bytes()
        ).hexdigest()
        _write_json(output / "benchmark.json", report)

    def capture(label: str, callback) -> None:
        try:
            rows = callback()
            samples.extend(rows if isinstance(rows, list) else [rows])
        except Exception as exc:  # preserve prior artifacts and publish a bounded failure
            failures.append({"benchmark": label, "exception_type": type(exc).__name__})
            report["status"] = "failed"
            persist()
            raise
        persist()

    if not args.skip_geometry:
        for phase, count in (("warmup", args.geometry_warmup), ("measured", args.geometry_repeats)):
            for index in range(count):
                print(f"geometry {phase} {index + 1}/{count}", flush=True)
                capture("geometry", lambda p=phase, i=index: run_geometry_pair(output, args.geometry_cases, p, i))

    if not args.skip_numeric:
        for mode in args.numeric_modes:
            for phase, count in (("warmup", args.numeric_warmup), ("measured", args.numeric_repeats)):
                for index in range(count):
                    print(f"numeric/{mode} {phase} {index + 1}/{count}", flush=True)
                    capture("numeric_" + mode, lambda m=mode, p=phase, i=index: run_numeric_iteration(output, m, p, i))

    if not args.skip_physics:
        for phase, count in (("warmup", args.physics_warmup), ("measured", args.physics_repeats)):
            for index in range(count):
                print(f"physics {phase} {index + 1}/{count}", flush=True)
                capture("physics", lambda p=phase, i=index: run_physics_iteration(output, p, i, args.physics_seed, args.physics_friction))

    report["status"] = "completed"
    persist()
    print("benchmark completed; see benchmark.json and samples.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
