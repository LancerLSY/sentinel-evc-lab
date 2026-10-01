import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_scenarios", ROOT / "tools" / "validate_scenarios.py"
)
scenarios = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scenarios)


def test_additional_scenario_matrix_uses_real_product_paths_and_passes(tmp_path):
    output = tmp_path / "scenario-validation"
    assert scenarios.main(["--out", str(output), "--seeds", "0,7"]) == 0
    summary = json.loads((output / "summary.json").read_text("utf-8"))
    metadata = json.loads((output / "metadata.json").read_text("utf-8"))
    rows = [json.loads(line) for line in (output / "cases.jsonl").read_text("utf-8").splitlines()]

    assert summary["status"] == "completed"
    assert summary["passed"] == summary["case_count"] == len(rows)
    assert summary["claims"] == {
        "mujoco": False,
        "real_robot": False,
        "robot_realtime": False,
        "vla": False,
    }
    assert metadata["seeds"] == [0, 7]
    assert metadata["source_digest"].startswith("sha256:")
    by_id = {row["case_id"]: row for row in rows}
    assert by_id["strict-risk-rejection"]["actual"]["candidate_reasons"] == ["RISK_LIMIT"] * 4
    assert by_id["changed-displacement-is-unknown"]["actual"]["candidate_reasons"] == ["MODEL_UNKNOWN"] * 4
    assert by_id["obstacle-blocks-all-actions"]["actual"]["candidate_reasons"] == ["PHYSICAL_REJECTED"] * 4
    assert by_id["restart-tamper-refusal-and-recovery"]["actual"]["tampered_export_refused"] is True
    assert all(row["evidence"] for row in rows)
    public_text = (output / "summary.json").read_text("utf-8") + (output / "metadata.json").read_text("utf-8") + (output / "cases.jsonl").read_text("utf-8")
    assert str(tmp_path) not in public_text


def test_scenario_cli_refuses_nonempty_output(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("keep", "utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        scenarios.main(["--out", str(output)])
