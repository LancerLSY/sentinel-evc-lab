"""从命令行生成真实产物，验证证据与离线安装边界。"""

from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(*args, cwd=ROOT, env=None):
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "sentinel_evc", *map(str, args)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=60,
    )


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli-evidence") / "seeded-demo"
    result = cli("demo", "--cases", 8, "--seed", 1234, "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    return out


def verify(out, *, bundle=None, public=None, run_id=None):
    return cli(
        "verify", "--bundle", bundle or out / "bundle",
        "--public-key", public or out / "anchors" / "demo.public",
        "--run-id", run_id or out.name,
    )


@pytest.mark.parametrize("occupied", [False, True], ids=["empty", "contains-file"])
def test_demo_rejects_existing_output_directory_without_writing(tmp_path, occupied):
    out = tmp_path / "existing"
    out.mkdir()
    if occupied:
        (out / "keep.txt").write_bytes(b"existing data\x00\xff")
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    result = cli("demo", "--cases", 8, "--seed", 1234, "--out", out)
    assert result.returncode in {1, 2}
    assert "unrecognized arguments" not in result.stderr
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_demo_writes_only_the_approved_artifacts(demo_output):
    actual = {path.relative_to(demo_output).as_posix()
              for path in demo_output.rglob("*") if path.is_file()}
    assert actual == {
        "scenario.json", "summary.json", "report.html", "anchors/demo.public",
        "bundle/events.jsonl", "bundle/manifest.json", "bundle/manifest.sig",
    }


def test_demo_bundle_contains_only_the_three_signed_evidence_files(demo_output):
    assert {path.name for path in (demo_output / "bundle").iterdir()} == {
        "events.jsonl", "manifest.json", "manifest.sig",
    }


def test_demo_uses_raw_ed25519_signature_and_external_public_key(demo_output):
    assert len((demo_output / "bundle" / "manifest.sig").read_bytes()) == 64
    assert len((demo_output / "anchors" / "demo.public").read_bytes()) == 32


def test_demo_uses_directory_name_for_every_run_identity(demo_output):
    summary = json.loads((demo_output / "summary.json").read_bytes())
    manifest = json.loads((demo_output / "bundle" / "manifest.json").read_bytes())
    events = [json.loads(line) for line in
              (demo_output / "bundle" / "events.jsonl").read_bytes().splitlines()]
    assert {summary["run_id"], manifest["run_id"],
            *(event["run_id"] for event in events)} == {demo_output.name}
    assert summary["bundle"]["bundle_dir"] == "bundle"
    assert summary["bundle"]["public_key"] == "anchors/demo.public"


def test_demo_manifest_describes_the_complete_raw_event_file(demo_output):
    raw = (demo_output / "bundle" / "events.jsonl").read_bytes()
    manifest = json.loads((demo_output / "bundle" / "manifest.json").read_bytes())
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert manifest["event_count"] == raw.count(b"\n")
    assert manifest["files"] == {"events.jsonl": "sha256:" + sha256(raw).hexdigest()}
    assert manifest["tip_hash"] == "sha256:" + sha256(raw.split(b"\n")[-2]).hexdigest()


def test_demo_evidence_includes_geometry_execution_and_outcome_in_one_stream(demo_output):
    events = [json.loads(line) for line in
              (demo_output / "bundle" / "events.jsonl").read_bytes().splitlines()]
    types = {event["type"] for event in events}
    assert {"PROPOSAL", "TRANSFORM", "CERTIFICATE", "PREPARE", "COMMIT",
            "DISPATCH", "CONTROLLER_ACK", "OBSERVED", "REVOKE", "CANCEL_ACK",
            "OUTCOME"} <= types
    assert "LOG_GAP" not in types
    assert [event["seq"] for event in events] == list(range(len(events)))


def test_verify_accepts_the_original_demo_bundle(demo_output):
    result = verify(demo_output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


@pytest.mark.parametrize("tamper,first_failure", [
    ("middle-event", "CHAIN_BROKEN"),
    ("last-three-lines", "EVENT_COUNT_MISMATCH"),
    ("public-key", "SIGNATURE_MISMATCH"),
    ("run-id", "RUN_ID_MISMATCH"),
])
def test_verify_reports_distinct_first_failure_for_each_tamper(
    demo_output, tmp_path, tamper, first_failure,
):
    bundle = tmp_path / "bundle"
    shutil.copytree(demo_output / "bundle", bundle)
    public = demo_output / "anchors" / "demo.public"
    run_id = demo_output.name
    path = bundle / "events.jsonl"
    if tamper == "middle-event":
        lines = path.read_bytes().splitlines(keepends=True)
        candidates = [index for index, line in enumerate(lines[:-1])
                      if b'"method":"mix"' in line]
        index = candidates[len(candidates) // 2]
        lines[index] = lines[index].replace(b'"method":"mix"', b'"method":"near"')
        path.write_bytes(b"".join(lines))
    elif tamper == "last-three-lines":
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-3]))
    elif tamper == "public-key":
        public = tmp_path / "wrong.public"
        public.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes_raw())
    else:
        run_id = "different-run"
    result = verify(demo_output, bundle=bundle, public=public, run_id=run_id)
    assert result.returncode == 1, result.stdout + result.stderr
    assert first_failure in result.stdout.split(" | ", 1)[0]


@pytest.mark.parametrize("args", [(), ("verify",), ("verify", "--bundle", "missing")])
def test_missing_required_cli_arguments_exit_with_two(args):
    result = cli(*args)
    assert result.returncode == 2


def test_report_displays_the_same_path_counts_as_summary(demo_output):
    report = (demo_output / "report.html").read_text(encoding="utf-8")
    geo = json.loads((demo_output / "summary.json").read_bytes())["geometry"]
    rows = [[html.unescape(re.sub("<[^>]+>", "", cell)).strip()
             for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            for row in re.findall(r"<tr>(.*?)</tr>", report, re.S)]
    for path in "abc":
        row = next(row for row in rows if row and row[0].startswith(path + " · "))
        assert row[1] == str(geo[f"path_{path}_wrong_release"])
        if path in "bc":
            assert row[2] == str(geo[f"path_{path}_full_checks"])


def test_report_displays_the_actual_manifest_and_verifier_result(demo_output):
    report = (demo_output / "report.html").read_text(encoding="utf-8")
    manifest = json.loads((demo_output / "bundle" / "manifest.json").read_bytes())
    result = verify(demo_output)
    message = result.stdout.strip().split(" — ", 1)[1]
    assert html.escape(message) in report
    assert manifest["tip_hash"] in report
    assert f'事件条数 {manifest["event_count"]}' in report


def test_package_installs_offline_and_runs_outside_the_checkout(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", project)
    shutil.copytree(ROOT / "src" / "sentinel_evc", project / "src" / "sentinel_evc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    installed = tmp_path / "installed"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--no-build-isolation", "--no-compile", "--target", str(installed), str(project)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # 复用预装运行依赖，但包自身必须从非 editable 的新安装目录导入。
    env = {**os.environ, "PYTHONPATH": str(installed), "PYTHONUTF8": "1"}
    imported = subprocess.run(
        [sys.executable, "-c", "import sentinel_evc; print(sentinel_evc.__file__)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert imported.returncode == 0, imported.stderr
    assert Path(imported.stdout.strip()).resolve().is_relative_to(installed.resolve())
    result = cli("--help", cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(command in result.stdout for command in ("demo", "geometry", "fault", "verify"))
