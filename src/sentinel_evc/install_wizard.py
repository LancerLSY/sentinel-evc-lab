"""Dependency-free terminal installation wizard; uses the shared installer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .installation import install_system


STEPS = {
    "environment": "创建独立 Python 环境",
    "source": "保存公开源码快照",
    "dependencies": "安装所选依赖",
    "launchers": "创建 CLI 启动器",
    "app": "构建 macOS App",
    "publish": "保存安装记录并发布目录",
}


class Terminal:
    def __init__(self, stream=None, input_fn=None):
        self.stream = stream if stream is not None else sys.stdout
        self.input_fn = input_fn if input_fn is not None else input
        self.color = (self.stream.isatty() and "NO_COLOR" not in os.environ
                      and (os.name != "nt" or "WT_SESSION" in os.environ))
        self.step = None
        self.index = 0
        self.total = 5
        self._stop = threading.Event()
        self._spinner = None
        self._started = 0.0

    def write(self, message="", *, accent=False):
        value = f"\033[36m{message}\033[0m" if accent and self.color else message
        print(value, file=self.stream, flush=True)

    def heading(self, title):
        self.write()
        self.write("─" * 60, accent=True)
        self.write("  " + title, accent=True)
        self.write("─" * 60, accent=True)

    def ask(self, prompt, default=""):
        suffix = f" [{default}]" if default else ""
        value = self.input_fn(f"  {prompt}{suffix} › ").strip()
        if value.lower() == "q":
            raise KeyboardInterrupt
        return value or default

    def choice(self, prompt, options, default=1):
        self.write()
        for i, label in enumerate(options, 1):
            self.write(f"  {i}  {label}")
        while True:
            value = self.ask(prompt, str(default))
            if value.isdigit() and 1 <= int(value) <= len(options):
                return int(value)
            self.write(f"  请输入 1–{len(options)}，或 q 退出。")

    def yes(self, prompt, default=True):
        while True:
            value = self.ask(prompt + "（y/n）", "y" if default else "n").lower()
            if value in {"y", "yes", "是"}:
                return True
            if value in {"n", "no", "否"}:
                return False
            self.write("  请输入 y 或 n，或 q 退出。")

    def progress(self, step, event):
        if event == "start":
            self.stop_spinner()
            self.step = step
            self.index += 1
            self._started = time.monotonic()
            self._stop.clear()
            self.write(f"  [{self.index}/{self.total}] {STEPS[step]} …")
            if self.color:
                self._spinner = threading.Thread(target=self._animate, daemon=True)
                self._spinner.start()
        elif event == "done":
            self.stop_spinner()
            elapsed = time.monotonic() - self._started
            self.write(f"        ✓ {STEPS[step]} · {elapsed:.1f}s", accent=True)

    def _animate(self):
        frames = ("◐", "◓", "◑", "◒")
        i = 0
        while not self._stop.wait(.2):
            elapsed = time.monotonic() - self._started
            self.stream.write(f"\r\033[2K        {frames[i % 4]} 正在执行 · {elapsed:.1f}s")
            self.stream.flush()
            i += 1

    def stop_spinner(self):
        self._stop.set()
        if self._spinner is not None:
            self._spinner.join()
            self.stream.write("\r\033[2K")
            self.stream.flush()
            self._spinner = None


def _command(command):
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run_wizard(*, source=None, target=None, profile="all", python_executable=None,
               no_app=False, terminal=None):
    terminal = terminal if terminal is not None else Terminal()
    python = python_executable or sys.executable
    source = Path(source or Path(__file__).resolve().parents[2]).expanduser().resolve()
    log_path = None
    installed = False
    try:
        terminal.heading("SENTINEL EVC  /  安装向导")
        terminal.write("  回车使用默认值 · 输入 q 或 Ctrl+C 退出")
        if sys.version_info < (3, 10):
            raise ValueError("需要 Python 3.10 或更新版本。")
        if not (source / "pyproject.toml").is_file():
            raise ValueError("请从完整源码检出运行安装向导（缺少 pyproject.toml）。")
        terminal.write(f"  Python  {sys.version.split()[0]}")
        terminal.write(f"  平台    {sys.platform}")
        selected = terminal.choice("安装类型", (
            "完整版  · CLI、工作台、MuJoCo 三维实验",
            "轻量版  · CLI、工作台、模型预览、只读诊断",
        ), default=1 if profile == "all" else 2)
        profile = "all" if selected == 1 else "core"
        default_target = str(Path(target or Path.cwd() / "sentinel-evc-install").expanduser().absolute())
        while True:
            destination = Path(terminal.ask("安装目录", default_target)).expanduser().resolve()
            if destination.exists():
                terminal.write("  此目录已存在，请选择新目录；现有安装和实验不会被覆盖。")
                continue
            if destination.parent.exists() and not destination.parent.is_dir():
                terminal.write("  父路径不是目录，请重新选择。")
                continue
            break
        build_app = False
        if sys.platform == "darwin" and not no_app:
            tools_ready = bool(shutil.which("xcrun") and shutil.which("codesign"))
            build_app = terminal.yes("同时创建 macOS App", default=tools_ready)
            if build_app and not tools_ready:
                terminal.write("  未找到 Apple Command Line Tools；可先安装 CLI，之后再构建 App。")
                build_app = False
        elif sys.platform != "darwin":
            terminal.write("  本平台安装 CLI；可用浏览器打开工作台。")
        terminal.heading("安装配置")
        terminal.write(f"  类型    {'完整版' if profile == 'all' else '轻量版'}")
        terminal.write(f"  目录    {destination}")
        terminal.write(f"  App     {'创建原生 macOS App' if build_app else '使用 CLI / 浏览器入口'}")
        terminal.write("  首次安装需要下载依赖。每个阶段完成后会显示实际耗时。")
        if not terminal.yes("开始安装"):
            terminal.write("  已取消，未创建安装目录。")
            return 0
        fd, raw_path = tempfile.mkstemp(prefix="sentinel-install-", suffix=".log")
        os.close(fd)
        log_path = Path(raw_path)
        terminal.total = 6 if build_app else 5
        terminal.heading("正在安装")
        terminal.write(f"  详细日志  {log_path}")
        with log_path.open("a", encoding="utf-8") as log:
            def runner(command, **kwargs):
                log.write("\n$ " + _command(command) + "\n")
                log.flush()
                return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
            result = install_system(destination, profile=profile, source=source,
                                    python_executable=python, build_app=build_app,
                                    runner=runner, on_progress=terminal.progress)
        installed = True
        terminal.heading("安装完成")
        terminal.write(f"  CLI     {result.command}")
        if result.app:
            terminal.write(f"  App     {result.app}")
        terminal.write(f"  数据    {result.target / 'data'}")
        terminal.write(f"  日志    {log_path}")
        _finish(result, terminal)
        return 0
    except (KeyboardInterrupt, EOFError):
        terminal.stop_spinner()
        terminal.write("\n  已退出向导。" if installed else "\n  安装已取消；未发布的临时安装会清理。")
        if log_path:
            terminal.write(f"  日志保留在 {log_path}")
        return 0 if installed else 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        terminal.stop_spinner()
        terminal.heading("启动失败" if installed else "安装失败")
        if terminal.step and not installed:
            terminal.write(f"  阶段    {STEPS[terminal.step]}")
        terminal.write(f"  原因    {exc}")
        if log_path:
            terminal.write(f"  详情    {log_path}")
        terminal.write("  安装结果仍保留，可稍后手动启动。" if installed else "  未发布的临时安装会清理；已有目录不会被覆盖。")
        return 2
    finally:
        terminal.stop_spinner()


def _finish(result, terminal):
    options = ["打开原生 App" if result.app else "打开浏览器工作台",
               "查看 SSH 实验接入命令", "完成并退出"]
    choice = terminal.choice("下一步", options, default=3)
    if choice == 1:
        if result.app:
            subprocess.run(["open", "-n", str(result.app)], check=True)
        else:
            subprocess.run([str(result.command), "app", "--browser", "--data-dir",
                            str(result.target / "data")], check=True)
    elif choice == 2:
        terminal.write("  使用已有 OpenSSH 主机别名；需要可用的密钥认证和已确认的主机指纹。")
        terminal.write("  将 YOUR_SSH_HOST 替换为自己的主机别名：")
        terminal.write("  " + _command([str(result.command), "remote-physics", "--host",
                                       "YOUR_SSH_HOST", "--out", str(result.target / "data/remote-physics")]))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Sentinel EVC 交互式安装向导")
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--profile", choices=("all", "core"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-app", action="store_true")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        print("安装向导需要交互终端。无人值守安装请使用 tools/install.sh TARGET 或 install --target TARGET。", file=sys.stderr)
        return 2
    return run_wizard(source=args.source, target=args.target, profile=args.profile,
                      python_executable=args.python, no_app=args.no_app)


if __name__ == "__main__":
    raise SystemExit(main())
