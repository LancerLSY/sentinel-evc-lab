import io
from pathlib import Path
import subprocess
import sys

from sentinel_evc import cli, install_wizard as wizard
from test_installation import FakeRunner, _source


def _terminal(answers):
    answers = iter(answers)
    output = io.StringIO()
    return wizard.Terminal(output, lambda prompt: next(answers)), output


def _fake_commands(monkeypatch, tmp_path, fail_on=None):
    runner = FakeRunner(fail_on)
    monkeypatch.setattr(wizard.sys, "platform", "linux")
    monkeypatch.setattr(wizard.subprocess, "run", runner)
    original = wizard.tempfile.mkstemp
    monkeypatch.setattr(wizard.tempfile, "mkstemp", lambda **kwargs: original(dir=tmp_path, **kwargs))
    return runner


def test_wizard_installs_selected_profile_with_actual_stage_events(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path)
    terminal, output = _terminal(["2", str(tmp_path / "installed"), "y", "3"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 0
    assert not runner.commands[1][-1].endswith("[physics]")
    assert (tmp_path / "installed/bin/sentinel-evc").exists()
    text = output.getvalue()
    for name in ("独立 Python", "公开源码", "所选依赖", "CLI 启动器", "安装记录"):
        assert name in text
    assert "[5/5]" in text and "安装完成" in text
    assert "\033" not in text
    assert (tmp_path / "installed/installation.json").is_file()


def test_wizard_reprompts_invalid_choice_and_existing_directory(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("existing user data")
    terminal, output = _terminal(["9", "1", str(target), str(tmp_path / "new"), "y", "3"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 0
    assert runner.commands[1][-1].endswith("[physics]")
    assert marker.read_text() == "existing user data"
    assert "目录已存在" in output.getvalue() and "请输入 1–2" in output.getvalue()


def test_wizard_cancel_creates_no_target_or_commands(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path)
    terminal, output = _terminal(["", str(tmp_path / "cancelled"), "n"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 0
    assert not runner.commands and not (tmp_path / "cancelled").exists()
    assert "已取消" in output.getvalue()


def test_wizard_quit_and_input_eof_are_clean(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path)
    terminal, _ = _terminal(["q"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 130
    def eof(prompt):
        raise EOFError
    assert wizard.run_wizard(source=tmp_path / "checkout", terminal=wizard.Terminal(io.StringIO(), eof)) == 130
    assert not runner.commands


def test_wizard_failed_dependency_retains_log_and_removes_partial_install(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path, fail_on=2)
    target = tmp_path / "failed"
    terminal, output = _terminal(["1", str(target), "y"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 2
    assert len(runner.commands) == 2
    assert not target.exists() and not list(tmp_path.glob(".failed-*"))
    assert "阶段    安装所选依赖" in output.getvalue()
    assert list(tmp_path.glob("sentinel-install-*.log"))


def test_wizard_ssh_next_step_prints_command_without_connecting(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path)
    terminal, output = _terminal(["2", str(tmp_path / "ssh"), "y", "2"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 0
    assert len(runner.commands) == 2
    assert "YOUR_SSH_HOST" in output.getvalue() and "remote-physics" in output.getvalue()


def test_wizard_native_app_option_and_real_stage_count(tmp_path, monkeypatch):
    _fake_commands(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(wizard.shutil, "which", lambda _: "/usr/bin/tool")
    from sentinel_evc import installation
    def build(out, *args, **kwargs):
        out.mkdir()
        return out
    monkeypatch.setattr(installation.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installation, "build_macos_app", build)
    terminal, output = _terminal(["2", str(tmp_path / "native"), "y", "y", "3"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 0
    assert (tmp_path / "native/Sentinel EVC.app").is_dir()
    assert "[6/6]" in output.getvalue() and "构建 macOS App" in output.getvalue()


def test_wizard_launch_failure_preserves_completed_install(tmp_path, monkeypatch):
    runner = _fake_commands(monkeypatch, tmp_path, fail_on=3)
    terminal, output = _terminal(["2", str(tmp_path / "installed"), "y", "1"])
    assert wizard.run_wizard(source=_source(tmp_path), terminal=terminal) == 2
    assert len(runner.commands) == 3
    assert (tmp_path / "installed/installation.json").is_file()
    assert "启动失败" in output.getvalue() and "安装结果仍保留" in output.getvalue()


def test_interactive_cli_does_not_wait_for_input_in_automation(monkeypatch):
    class NonTTY:
        def isatty(self):
            return False
    monkeypatch.setattr(cli.sys, "stdin", NonTTY())
    assert cli.main(["install"]) == 2
    assert cli.main(["install", "--interactive", "--target", "unused"]) == 2


def test_noninteractive_cli_keeps_target_based_installation(tmp_path, monkeypatch):
    from sentinel_evc import installation
    calls = []
    def install(target, **kwargs):
        calls.append((target, kwargs))
        return installation.InstallationResult(Path(target), tmp_path / "python", tmp_path / "command", None, "core")
    monkeypatch.setattr(installation, "install_system", install)
    assert cli.main(["install", "--target", str(tmp_path / "target"), "--profile", "core", "--no-app"]) == 0
    assert len(calls) == 1 and calls[0][1]["profile"] == "core"
