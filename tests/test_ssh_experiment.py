"""Offline contract tests for strict SSH physics transport and receipts."""

import ast
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import sys

import pytest

from sentinel_evc.contracts import canonical_json
from sentinel_evc.evidence import build_bundle
from sentinel_evc.events import EventLog
from sentinel_evc import ssh_experiment as remote


def _fake_result(directory: Path, *, seed=7, friction=0.35, render=False):
    directory.mkdir()
    anchors = {}
    run_ids = sorted(remote._expected_trial_ids(seed))
    for index, run_id in enumerate(run_ids):
        log = EventLog(run_id, schema_version="product-v1")
        log.append("OUTCOME", outcome="stable", trial=index)
        info = build_bundle(log, str(directory / run_id), artifacts={
            "result.json": canonical_json({"outcome": "stable", "trial": index}),
            "root.json": canonical_json({"root": index}),
            "plan.json": canonical_json({"plan": index}),
            "physics.xml": b"<mujoco/>",
            "trace.json": canonical_json([]),
            "config.json": canonical_json({"timestep": 0.002}),
        })
        trial = directory / run_id
        anchors[run_id] = {
            "run_id": run_id,
            "tip_hash": info["tip_hash"],
            "verification": "test fixture",
            "public_key_sha256": hashlib.sha256(
                (trial / "anchors" / "demo.public").read_bytes()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (trial / "bundle" / "manifest.json").read_bytes()
            ).hexdigest(),
        }

    experiment_id = f"physics-experiment-{seed}"
    summary = {
        "schema": "mujoco-experiment-v1",
        "scope": "test fixture",
        "acceptance": {"static_stability": True},
        "infrastructure_gates_pass": True,
    }
    summary_bytes = canonical_json(summary)
    source_manifest = canonical_json({"src/sentinel_evc/physics.py": "0" * 64})
    signed_completion = {
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "experiment_id": experiment_id,
        "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
    }
    index_log = EventLog(experiment_id, schema_version="product-v1")
    index_log.append("OUTCOME", acceptance=summary["acceptance"], infrastructure_gates_pass=True)
    index = build_bundle(index_log, str(directory / "experiment-index"), artifacts={
        "summary.json": summary_bytes,
        "completion.json": canonical_json(signed_completion),
        "source_manifest.json": source_manifest,
        "environment.json": canonical_json({"python": "test", "mujoco": "test"}),
        "trial_anchors.json": canonical_json(anchors),
        "job_config.json": canonical_json({
            "seed": seed,
            "friction": friction,
            "render": render,
            "command": "python -m sentinel_evc physics --out <empty-job-dir>",
            "profile": "mujoco-test",
        }),
    })
    index_dir = directory / "experiment-index"
    complete = {
        **signed_completion,
        "index_tip": index["tip_hash"],
        "public_key_sha256": hashlib.sha256(
            (index_dir / "anchors" / "demo.public").read_bytes()
        ).hexdigest(),
    }
    (directory / "summary.json").write_bytes(summary_bytes)
    (directory / "COMPLETE.json").write_bytes(canonical_json(complete))
    return complete


def _archive_result(directory: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.add(path, arcname="result/" + path.relative_to(directory).as_posix())
    return stream.getvalue()


def test_remote_input_rejects_shell_targets_paths_and_nonfinite_values(tmp_path):
    bad_calls = [
        {"host": "user@example", "out": tmp_path / "a"},
        {"host": "--proxy", "out": tmp_path / "b"},
        {"host": "lab", "out": tmp_path / "c", "python": "/usr/bin/python3"},
        {"host": "lab", "out": tmp_path / "d", "friction": float("nan")},
        {"host": "lab", "out": tmp_path / "e", "render": 1},
    ]
    for kwargs in bad_calls:
        with pytest.raises(ValueError):
            remote.run_remote_physics(**kwargs)


def test_public_source_archive_has_a_fixed_nonsecret_surface():
    root = Path(remote.__file__).resolve().parents[2]
    data, digest = remote._source_archive(root)
    assert hashlib.sha256(data).hexdigest() == digest
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "pyproject.toml" in names
    assert "README.md" in names
    assert "LICENSE" in names
    assert "src/sentinel_evc/ssh_experiment.py" in names
    assert "src/sentinel_evc/native/SentinelApp.swift" in names
    assert "tests/test_ssh_experiment.py" in names
    assert "tools/benchmark_performance.py" in names
    assert "tools/validate_scenarios.py" in names
    assert not any(".git" in Path(name).parts for name in names)
    assert not any(name.endswith((".token", ".pem", ".key")) for name in names)
    assert "AGENTS.local.md" not in names


@pytest.mark.parametrize('link_target', ['README.md', 'missing-file'])
def test_public_source_archive_refuses_symlinked_benchmark(tmp_path, link_target):
    (tmp_path / "src/sentinel_evc").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "src/sentinel_evc/module.py").write_text("", encoding="utf-8")
    (tmp_path / "tests/test_module.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "tools/benchmark_performance.py").symlink_to(tmp_path / link_target)

    with pytest.raises(remote.RemoteExperimentError, match="unsafe"):
        remote._source_files(tmp_path)


@pytest.mark.parametrize("location", ["src", "src/sentinel_evc/nested.py"])
def test_public_source_archive_refuses_root_or_nested_symlink(tmp_path, location):
    (tmp_path / "src/sentinel_evc").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/sentinel_evc/module.py").write_text("", encoding="utf-8")
    (tmp_path / "tests/test_module.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    path = tmp_path / location
    if path.is_dir():
        shutil.rmtree(path)
        path.symlink_to(tmp_path / "tests", target_is_directory=True)
    else:
        path.symlink_to(tmp_path / "README.md")
    with pytest.raises(remote.RemoteExperimentError, match="unsafe"):
        remote._source_files(tmp_path)


def test_public_source_archive_refuses_symlinked_checkout_root(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(remote.RemoteExperimentError, match="root is unsafe"):
        remote._source_files(alias)


def test_public_manifest_digest_includes_scenario_harness(tmp_path):
    root = Path(remote.__file__).resolve().parents[2]
    before = remote._public_source_manifest_sha256(root)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for source in remote._source_files(root):
        target = checkout / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    harness = checkout / "tools/validate_scenarios.py"
    harness.write_bytes(harness.read_bytes() + b"\n# digest-change\n")
    after = remote._public_source_manifest_sha256(checkout)
    assert after != before


def test_extracted_public_source_archive_runs_full_validation(tmp_path):
    if importlib.machinery.PathFinder.find_spec('pytest') is None:
        pytest.skip('public archive full validation requires optional pytest')
    root = Path(remote.__file__).resolve().parents[2]
    package, _ = remote._source_archive(root)
    checkout = tmp_path / 'extracted-public-source'
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(package), mode='r:gz') as archive:
        for member in archive.getmembers():
            target = checkout / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            assert member.isfile()
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = archive.extractfile(member)
            assert incoming is not None
            target.write_bytes(incoming.read())
    environment = os.environ.copy()
    current_src = (root / 'src').resolve()
    dependencies = [
        str(Path(item).resolve()) for item in sys.path if item and Path(item).resolve() != current_src
    ]
    environment['PYTHONPATH'] = os.pathsep.join([str(checkout / 'src'), *dependencies])
    result = subprocess.run(
        # Exclude only this rehearsal to avoid recursively unpacking and
        # launching another complete suite from inside itself.
        [sys.executable, '-m', 'pytest', '-q', '-k',
         'not extracted_public_source_archive_runs_full_validation'],
        cwd=checkout, env=environment,
        text=True, capture_output=True, timeout=180,
    )
    (tmp_path / 'archive-validation.log').write_text(result.stdout + result.stderr, 'utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 deselected' in result.stdout, result.stdout


def test_remote_bootstrap_runs_package_verification_with_the_dependency_venv():
    tree = ast.parse(remote._REMOTE_BOOTSTRAP)
    bootstrap_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(module and module.startswith("sentinel_evc") for module in bootstrap_imports)
    assert "from sentinel_evc.ssh_experiment import _verify_result_tree" in remote._REMOTE_BOOTSTRAP
    assert "call('verification.log',[str(py),'-c',verify_program,str(output)])" in remote._REMOTE_BOOTSTRAP
    install = remote._REMOTE_BOOTSTRAP.index("if call('install.log'")
    source_path = remote._REMOTE_BOOTSTRAP.index("env['PYTHONPATH']=str(source/'src')")
    tests = remote._REMOTE_BOOTSTRAP.index("if call('tests.log'")
    assert install < source_path < tests


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_result_archive_refuses_traversal_and_links(tmp_path, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        if kind == "traversal":
            data = b"bad"
            info = tarfile.TarInfo("result/../../escape")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        else:
            info = tarfile.TarInfo("result/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
    with pytest.raises(remote.RemoteExperimentError, match="unsafe"):
        remote._safe_extract_result(stream.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_run_uses_verified_complete_archive_and_writes_receipt_last(tmp_path, monkeypatch):
    source = tmp_path / "remote-result"
    complete = _fake_result(source)
    archive = _archive_result(source)
    package = b"bounded-public-source"
    package_digest = hashlib.sha256(package).hexdigest()
    job_id_holder = {}

    monkeypatch.setattr(remote, "_host_pin", lambda host: {
        "host_alias": host,
        "resolved_host": "physics.internal",
        "port": 22,
        "known_host_fingerprints": ["SHA256:test-host-key"],
    })
    monkeypatch.setattr(remote, "_source_archive", lambda root: (package, package_digest))
    monkeypatch.setattr(
        remote,
        "_expected_source_manifest_sha256",
        lambda root: complete["source_manifest_sha256"],
    )

    def finish(host, command, sent, timeout):
        assert sent == package
        assert "StrictHostKeyChecking" not in command  # options belong to ssh argv, never remote shell
        job_id = next(part for part in command.split() if part.startswith("job-"))
        job_id_holder["id"] = job_id
        return {
            "schema": "sentinel-ssh-complete-v1",
            "job_id": job_id,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_size": len(archive),
            "package_sha256": package_digest,
            "experiment_id": complete["experiment_id"],
            "index_tip": complete["index_tip"],
            "index_public_key_sha256": complete["public_key_sha256"],
            "source_manifest_sha256": complete["source_manifest_sha256"],
            "experiment_exit_code": 0,
        }

    monkeypatch.setattr(remote, "_run_remote_job", finish)
    monkeypatch.setattr(remote, "_fetch_remote_archive", lambda host, python, job_id, timeout: archive)
    output = tmp_path / "downloaded"
    result = remote.run_remote_physics("physics-lab", output)
    assert result["ok"] is True
    assert result["trial_count"] == 25
    assert result["known_host_fingerprints"] == ["SHA256:test-host-key"]
    assert "at collection time" in result["trust"]
    assert "not a transferable attestation" in result["trust"]
    assert (output / "REMOTE_RECEIPT.json").is_file()
    assert remote.verify_physics_experiment_result(output)["ok"] is True

    trial = output / sorted(remote._expected_trial_ids(7))[0] / "bundle" / "result.json"
    trial.write_bytes(trial.read_bytes() + b" ")
    with pytest.raises(remote.RemoteExperimentError):
        remote.verify_physics_experiment_result(output)


def test_verifier_rejects_an_omitted_signed_trial_even_when_remaining_bundles_verify(tmp_path):
    output = tmp_path / "incomplete"
    _fake_result(output)
    # Removing a trial invalidates the exact experiment shape before any claim can pass.
    shutil_target = output / sorted(remote._expected_trial_ids(7))[0]
    shutil.rmtree(shutil_target)
    with pytest.raises(remote.RemoteExperimentError):
        remote._verify_result_tree(output)


def test_verifier_rejects_top_level_summary_that_differs_from_signed_summary(tmp_path):
    output = tmp_path / "altered-summary"
    _fake_result(output)
    (output / "summary.json").write_bytes(canonical_json({
        "schema": "mujoco-experiment-v1",
        "acceptance": {"static_stability": False},
        "infrastructure_gates_pass": True,
    }))

    with pytest.raises(remote.RemoteExperimentError, match="top-level summary differs"):
        remote._verify_result_tree(output)


def test_partial_or_unbound_job_never_creates_a_receipt(tmp_path, monkeypatch):
    package = b"source"
    digest = hashlib.sha256(package).hexdigest()
    monkeypatch.setattr(remote, "_host_pin", lambda host: {
        "host_alias": host, "resolved_host": host, "port": 22,
        "known_host_fingerprints": ["SHA256:test"],
    })
    monkeypatch.setattr(remote, "_source_archive", lambda root: (package, digest))
    monkeypatch.setattr(remote, "_expected_source_manifest_sha256", lambda root: "0" * 64)
    monkeypatch.setattr(remote, "_run_remote_job", lambda *args: {
        "schema": "sentinel-ssh-complete-v1",
        "job_id": "job-wrong",
        "package_sha256": digest,
    })
    output = tmp_path / "partial"
    with pytest.raises(remote.RemoteExperimentError, match="not bound"):
        remote.run_remote_physics("physics-lab", output)
    assert not (output / "REMOTE_RECEIPT.json").exists()


def test_invalid_receipt_binding_is_rejected_before_result_is_published(tmp_path, monkeypatch):
    source = tmp_path / "remote-result"
    complete = _fake_result(source)
    archive = _archive_result(source)
    package = b"bounded-public-source"
    package_digest = hashlib.sha256(package).hexdigest()

    monkeypatch.setattr(remote, "_host_pin", lambda host: {
        "host_alias": host,
        "resolved_host": "physics.internal",
        "port": 22,
        "known_host_fingerprints": ["SHA256:test-host-key"],
    })
    monkeypatch.setattr(remote, "_source_archive", lambda root: (package, package_digest))
    monkeypatch.setattr(
        remote,
        "_expected_source_manifest_sha256",
        lambda root: complete["source_manifest_sha256"],
    )

    def finish(host, command, sent, timeout):
        job_id = next(part for part in command.split() if part.startswith("job-"))
        return {
            "schema": "sentinel-ssh-complete-v1",
            "job_id": job_id,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_size": len(archive),
            "package_sha256": package_digest,
            "experiment_id": complete["experiment_id"],
            "index_tip": "sha256:" + "f" * 64,
            "index_public_key_sha256": complete["public_key_sha256"],
            "source_manifest_sha256": complete["source_manifest_sha256"],
            "experiment_exit_code": 0,
        }

    monkeypatch.setattr(remote, "_run_remote_job", finish)
    monkeypatch.setattr(remote, "_fetch_remote_archive", lambda host, python, job_id, timeout: archive)
    output = tmp_path / "unpublished"

    with pytest.raises(remote.RemoteExperimentError, match="does not bind"):
        remote.run_remote_physics("physics-lab", output)

    assert not output.exists()


def test_ssh_job_invocation_enforces_noninteractive_strict_options(monkeypatch):
    seen = {}

    class Result:
        returncode = 0
        stderr = b""
        stdout = b'{"schema":"sentinel-ssh-complete-v1"}\n'

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(remote, "_run", fake_run)
    remote._run_remote_job("physics-lab", "python3 -c fixed", b"package", 60)
    args = seen["args"]
    assert args[0] == "ssh"
    for option in (
        "StrictHostKeyChecking=yes",
        "BatchMode=yes",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "NumberOfPasswordPrompts=0",
    ):
        assert option in args
    assert seen["kwargs"]["input_bytes"] == b"package"
