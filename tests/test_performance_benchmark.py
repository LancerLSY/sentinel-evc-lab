import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_performance", ROOT / "tools" / "benchmark_performance.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_nearest_rank_and_small_sample_qualification():
    values = [0.4, 0.1, 0.3, 0.2]
    assert benchmark.nearest_rank(values, 50) == 0.2
    assert benchmark.nearest_rank(values, 95) == 0.4
    assert benchmark.nearest_rank(values, 99) == 0.4
    summary = benchmark.duration_summary(values)
    assert summary["quantile_method"] == "nearest-rank"
    assert "descriptive" in summary["sample_size_note"]
    with pytest.raises(ValueError):
        benchmark.nearest_rank([], 50)


def test_geometry_pair_includes_parent_cost_failed_fallbacks_and_evidence(tmp_path):
    rows = benchmark.run_geometry_pair(
        tmp_path, cases=10, phase="measured", index=0
    )
    by_strategy = {row["strategy"]: row for row in rows}
    assert set(by_strategy) == {"full", "delta"}
    assert by_strategy["full"]["counts"] == {
        "cases": 10,
        "root_full_checks": 10,
        "final_full_checks": 10,
        "fallback_full_checks": 0,
        "inherited": 0,
        "unsafe_rejected": 5,
        "safe_passed": 5,
    }
    assert by_strategy["delta"]["counts"] == {
        "cases": 10,
        "root_full_checks": 10,
        "final_full_checks": 0,
        "fallback_full_checks": 5,
        "inherited": 5,
        "unsafe_rejected": 5,
        "safe_passed": 5,
    }
    assert all(row["duration_seconds"] > 0 for row in rows)
    assert all(row["evidence_verified"] is True for row in rows)
    assert all((tmp_path / row["artifact"] / "bundle" / "manifest.json").is_file() for row in rows)


def test_fast_numeric_iteration_completes_and_verifies_signed_evidence(tmp_path):
    row = benchmark.run_numeric_iteration(
        tmp_path, "physical", "measured", 0, realtime=False
    )
    assert row["status"] == "completed"
    assert row["observed_steps"] == 40
    assert row["evidence_verified"] is True
    assert not Path(row["artifact"]).is_absolute()
    assert (tmp_path / row["artifact"] / "workspace.lock").is_file()
    assert len(list((tmp_path / row["artifact"]).glob("run-*/evidence.zip"))) == 1


def test_cli_writes_public_summary_and_refuses_nonempty_output(tmp_path):
    output = tmp_path / "benchmark"
    assert benchmark.main(
        [
            "--out",
            str(output),
            "--skip-numeric",
            "--skip-physics",
            "--geometry-cases",
            "10",
            "--geometry-warmup",
            "0",
            "--geometry-repeats",
            "2",
        ]
    ) == 0
    report = json.loads((output / "benchmark.json").read_text("utf-8"))
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text("utf-8").splitlines()]
    assert report["status"] == "completed"
    assert report["schema"] == benchmark.SCHEMA
    assert report["results"]["geometry"]["full"]["sample_count"] == 2
    assert len(samples) == 4
    assert report["raw_samples_sha256"] == benchmark.hashlib.sha256(
        (output / "samples.jsonl").read_bytes()
    ).hexdigest()
    assert str(tmp_path) not in (output / "benchmark.json").read_text("utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        benchmark.main(
            ["--out", str(output), "--skip-numeric", "--skip-physics"]
        )
