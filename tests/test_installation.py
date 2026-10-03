from __future__ import annotations

import json
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from sentinel_evc import installation


def _source(root: Path) -> Path:
    source = root / "checkout"
    (source / "src/sentinel_evc").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "tools").mkdir()
    (source / "src/sentinel_evc/__init__.py").write_text("", encoding="utf-8")
    (source / "src/sentinel_evc/native").mkdir()
    (source / "src/sentinel_evc/native/SentinelApp.swift").write_text(
        "@main struct App { static func main() {} }\n", encoding="utf-8"
    )
    (source / "tests/test_sample.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (source / "tools/benchmark_performance.py").write_text(
        "print('benchmark')\n", encoding="utf-8"
    )
    (source / "tools/validate_scenarios.py").write_text(
        "print('scenarios')\n", encoding="utf-8"
    )
    (source / "src/sentinel_evc/__pycache__").mkdir()
    (source / "src/sentinel_evc/__pycache__/bad.pyc").write_bytes(b"private cache")
    for name, content in {
        "pyproject.toml": "[project]\nname='sentinel-evc-lab'\nversion='0'\n",
        "README.md": "# Sentinel\n",
        "LICENSE": "MIT\n",
    }.items():
        (source / name).write_text(content, encoding="utf-8")
    return source


class FakeRunner:
    def __init__(self, fail_on: int | None = None):
        self.commands: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, command, check, **kwargs):
        self.commands.append(command)
        assert "PYTHONPATH" not in kwargs.get("env", {})
        if self.fail_on == len(self.commands):
            raise subprocess.CalledProcessError(1, command)
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / ("Scripts/python.exe" if installation.os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


def test_install_system_publishes_complete_core_install(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "installed"
    runner = FakeRunner()

    result = installation.install_system(target, "core", source, build_app=False,
                                         python_executable="python-test", runner=runner)

    assert result.target == target
    assert result.profile == "core"
    assert result.command.is_file()
    assert result.command.stat().st_mode & stat.S_IXUSR
    assert "source/src" in result.command.read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in result.command.read_text(encoding="utf-8")
    assert (target / "source/tests/test_sample.py").is_file()
    assert (target / "source/tools/benchmark_performance.py").is_file()
    assert (target / "source/tools/validate_scenarios.py").is_file()
    assert (target / "source/src/sentinel_evc/native/SentinelApp.swift").is_file()
    assert not (target / "source/src/sentinel_evc/__pycache__").exists()
    manifest = json.loads((target / "installation.json").read_text())
    assert manifest["desktop_app"] is False
    assert manifest["product_version"] == "0.3.0.dev0"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["source_frozen_at"].endswith("+00:00")
    assert runner.commands[1][-1].endswith("/build-source")
    assert not (target / "build-source").exists()


@pytest.mark.parametrize('link_target', ['README.md', 'missing-file'])
def test_public_source_refuses_symlinked_benchmark(tmp_path, link_target):
    source = _source(tmp_path)
    benchmark = source / "tools/benchmark_performance.py"
    benchmark.unlink()
    benchmark.symlink_to(source / link_target)

    with pytest.raises(ValueError, match="regular file"):
        installation._copy_public_source(source, tmp_path / "public")


@pytest.mark.parametrize("location", ["src", "src/sentinel_evc/nested.py"])
def test_public_source_refuses_root_or_nested_symlink_before_copy(tmp_path, location):
    source = _source(tmp_path)
    path = source / location
    if path.is_dir():
        shutil.rmtree(path)
        path.symlink_to(source / "tests", target_is_directory=True)
    else:
        path.symlink_to(source / "README.md")
    destination = tmp_path / "public"
    with pytest.raises(ValueError, match="symlink"):
        installation._copy_public_source(source, destination)
    assert not destination.exists()


def test_public_source_refuses_symlinked_checkout_root_before_copy(tmp_path):
    source = _source(tmp_path)
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(source, target_is_directory=True)
    destination = tmp_path / "public"
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        installation._copy_public_source(alias, destination)
    assert not destination.exists()


def test_source_digest_ignores_runtime_bytecode_cache(tmp_path):
    source = _source(tmp_path)
    before = installation._source_digest(source)
    cache = source / "src/sentinel_evc/__pycache__"
    (cache / "new.cpython-312.pyc").write_bytes(b"runtime only")
    assert installation._source_digest(source) == before


def test_install_system_all_profile_requests_physics_extra(tmp_path):
    runner = FakeRunner()
    installation.install_system(tmp_path / "installed", "all", _source(tmp_path),
                                build_app=False, runner=runner)
    assert runner.commands[1][-1].endswith("/build-source[physics]")


def test_install_refuses_existing_target_before_running_commands(tmp_path):
    target = tmp_path / "installed"
    target.mkdir()
    runner = FakeRunner()
    with pytest.raises(FileExistsError):
        installation.install_system(target, source=_source(tmp_path),
                                    build_app=False, runner=runner)
    assert runner.commands == []


@pytest.mark.parametrize("tree", ("src", "tests"))
def test_install_refuses_target_inside_copied_source_before_commands(tmp_path, tree):
    source = _source(tmp_path)
    target = source / tree / "nested-install"
    runner = FakeRunner()

    with pytest.raises(ValueError, match="installation target cannot be inside"):
        installation.install_system(target, source=source, build_app=False, runner=runner)

    assert runner.commands == []
    assert not target.exists()
    assert not list((source / tree).glob(".nested-install-*"))


def test_install_refuses_symlinked_target_resolving_inside_source(tmp_path):
    source = _source(tmp_path)
    alias = tmp_path / "source-alias"
    alias.symlink_to(source / "src", target_is_directory=True)
    runner = FakeRunner()

    with pytest.raises(ValueError, match="installation target cannot be inside"):
        installation.install_system(alias / "nested-install", source=source,
                                    build_app=False, runner=runner)

    assert runner.commands == []
    assert not (source / "src/nested-install").exists()


@pytest.mark.parametrize("failure", (BrokenPipeError, KeyboardInterrupt, RuntimeError))
def test_published_install_survives_progress_observer_failure(tmp_path, failure):
    source = _source(tmp_path)
    target = tmp_path / "installed"
    runner = FakeRunner()
    events = []

    def observer(step, event):
        events.append((step, event))
        if (step, event) == ("publish", "done"):
            raise failure("observer failed after publish")

    result = installation.install_system(
        target, "core", source, build_app=False,
        python_executable="python-test", runner=runner, on_progress=observer,
    )

    assert result.target == target
    assert (target / "installation.json").is_file()
    assert (target / "bin").is_dir()
    assert events[-1] == ("publish", "done")
    assert not list(tmp_path.glob(".installed-*"))


def test_partial_failure_never_publishes_target(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "installed"
    runner = FakeRunner(fail_on=2)
    with pytest.raises(subprocess.CalledProcessError):
        installation.install_system(target, source=source, build_app=False, runner=runner)
    assert not target.exists()
    assert not list(tmp_path.glob(".installed-*"))


def test_build_macos_app_records_final_runtime_and_cleans_failure(tmp_path, monkeypatch):
    source = tmp_path / "SentinelApp.swift"
    source.write_text("@main struct App { static func main() {} }", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(installation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installation.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(installation.shutil, "which", lambda name: "/usr/bin/xcrun")

    build_commands = []

    def compile_ok(command, check, **kwargs):
        build_commands.append(command)
        if "-o" in command:
            executable = Path(command[command.index("-o") + 1])
            executable.write_text("binary", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    app = installation.build_macos_app(
        tmp_path / "Built.app", python, tmp_path / "data", source_file=source,
        source_dir=tmp_path / "final/source", configured_python=tmp_path / "final/python",
        runner=compile_ok,
    )
    config = json.loads((app / "Contents/Resources/app-config.json").read_text())
    assert config["python"] == str((tmp_path / "final/python").resolve())
    assert config["source_dir"] == str((tmp_path / "final/source").resolve())
    assert (app / "Contents/Info.plist").is_file()
    assert any("arm64-apple-macosx12.0" in command for command in build_commands)
    assert any("--sign" in command for command in build_commands)


def test_build_macos_app_preserves_venv_python_symlink(tmp_path, monkeypatch):
    source = tmp_path / "SentinelApp.swift"
    source.write_text("@main struct App { static func main() {} }", encoding="utf-8")
    base = tmp_path / "base-python"
    base.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    monkeypatch.setattr(installation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installation.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(installation.shutil, "which", lambda name: "/usr/bin/xcrun")

    def compile_ok(command, check, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_text("binary", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    app = installation.build_macos_app(tmp_path / "App.app", venv_python, tmp_path / "data",
                                       source_file=source, runner=compile_ok)
    config = json.loads((app / "Contents/Resources/app-config.json").read_text())
    assert config["python"] == str(venv_python.absolute())
    assert config["python"] != str(base.resolve())


def test_build_macos_app_requires_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(installation.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="macOS"):
        installation.build_macos_app(tmp_path / "App.app", "/usr/bin/python3", tmp_path)


def test_native_app_waits_for_cleanup_then_reaps_backend():
    source = (Path(installation.__file__).parent / "native/SentinelApp.swift").read_text()
    assert "addingTimeInterval(15.0)" in source
    assert "kill(task.processIdentifier, SIGKILL)" in source
    assert "task.waitUntilExit()" in source


def test_native_backend_discovery_requires_nonce_and_exact_marker():
    source=(Path(installation.__file__).parent/'native/SentinelApp.swift').read_text()
    assert 'line.hasPrefix("SENTINEL_READY ")' in source
    assert 'ready.nonce == launchNonce' in source
    assert 'environment["SENTINEL_LAUNCH_NONCE"] = launchNonce' in source
    assert 'guard backendPort == nil' in source
    assert 'url.user == nil, url.password == nil' in source
    assert 'line.range(of: "http://127.0.0.1:")' not in source


def test_native_app_routes_only_backend_downloads_to_save_panel():
    source = (Path(installation.__file__).parent / "native/SentinelApp.swift").read_text()
    assert "action.shouldPerformDownload" in source
    assert "response.canShowMIMEType ? .allow : .download" in source
    assert source.count("url.host == \"127.0.0.1\"") >= 3
    assert "guard backendResponse || isBackendBlob(url) else" in source
    assert "decisionHandler(.cancel)" in source
    assert "download.delegate = self" in source
    assert "NSSavePanel()" in source


def test_native_app_has_quit_and_first_responder_edit_menu():
    source = (Path(installation.__file__).parent / "native/SentinelApp.swift").read_text()
    assert "configureMainMenu()" in source
    assert "#selector(NSApplication.terminate(_:))" in source
    assert 'keyEquivalent: "q"' in source
    assert "#selector(NSText.copy(_:))" in source
    assert "#selector(NSText.paste(_:))" in source
    assert "#selector(NSText.selectAll(_:))" in source
    assert "NSApp.mainMenu = mainMenu" in source
