"""Reproducible local installation and macOS application packaging.

The installer deliberately creates a managed virtual environment instead of
claiming to ship a standalone Python runtime.  It installs from a source
checkout so the exact public source used by SSH jobs remains inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Sequence


RunCommand = Callable[..., subprocess.CompletedProcess]
_PROFILES = {"core": "", "all": "physics"}


@dataclass(frozen=True)
class InstallationResult:
    target: Path
    python: Path
    command: Path
    app: Path | None
    profile: str


def _run(command: Sequence[str], *, runner: RunCommand = subprocess.run, **kwargs) -> None:
    runner(list(command), check=True, **kwargs)


def _clean_install_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        environment.pop(name, None)
    return environment


def _compile_swift(source: Path, executable: Path, cache: Path,
                   *, runner: RunCommand) -> None:
    architecture = platform.machine()
    if architecture not in {"arm64", "x86_64"}:
        raise RuntimeError(f"unsupported macOS architecture: {architecture}")
    target = f"{architecture}-apple-macosx12.0"
    base = ["swiftc", "-target", target, "-parse-as-library",
            "-module-cache-path", str(cache)]
    suffix = [str(source), "-o", str(executable),
              "-framework", "AppKit", "-framework", "WebKit"]
    xcrun = shutil.which("xcrun")
    if not xcrun:
        raise RuntimeError("xcrun is required; install Apple Command Line Tools")
    try:
        _run([xcrun, *base, *suffix], runner=runner)
        return
    except subprocess.CalledProcessError as first_error:
        # A partially updated Command Line Tools install can leave the default
        # SDK newer than its Swift standard library.  Try other installed SDKs
        # without modifying the developer directory or global toolchain.
        sdk_root = Path("/Library/Developer/CommandLineTools/SDKs")
        candidates = sorted(
            (path for path in sdk_root.glob("MacOSX*.sdk") if not path.is_symlink()),
            reverse=True,
        )
        for sdk in candidates:
            try:
                _run([xcrun, "swiftc", "-sdk", str(sdk),
                      "-target", target, "-parse-as-library",
                      "-module-cache-path", str(cache),
                      *suffix], runner=runner)
                return
            except subprocess.CalledProcessError:
                continue
        raise first_error


def _require_new_target(target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"installation target already exists: {target}")
    if not target.parent.exists():
        target.parent.mkdir(parents=True)
    if not target.parent.is_dir():
        raise NotADirectoryError(str(target.parent))


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _write_posix_wrapper(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "install_root=$(CDPATH= cd -- \"$(dirname -- \"$0\")/..\" && pwd)\n"
        "PYTHONPATH=\"$install_root/source/src${PYTHONPATH:+:$PYTHONPATH}\"\n"
        "PYTHONDONTWRITEBYTECODE=1\n"
        "export PYTHONPATH PYTHONDONTWRITEBYTECODE\n"
        "exec \"$install_root/.venv/bin/python\" -m sentinel_evc \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_windows_wrapper(path: Path) -> None:
    path.write_text(
        "@echo off\r\n"
        "set \"INSTALL_ROOT=%~dp0..\"\r\n"
        "set \"PYTHONPATH=%INSTALL_ROOT%\\source\\src;%PYTHONPATH%\"\r\n"
        "set \"PYTHONDONTWRITEBYTECODE=1\"\r\n"
        "\"%INSTALL_ROOT%\\.venv\\Scripts\\python.exe\" -m sentinel_evc %*\r\n",
        encoding="utf-8",
    )


def _package_requirement(source: str | os.PathLike[str] | None, extra: str) -> str:
    raw = "." if source is None else os.fspath(source)
    candidate = Path(raw).expanduser()
    if candidate.exists():
        raw = str(candidate.resolve())
    return raw + (f"[{extra}]" if extra else "")


def build_macos_app(
    out: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    *,
    source_file: str | os.PathLike[str] | None = None,
    source_dir: str | os.PathLike[str] | None = None,
    configured_python: str | os.PathLike[str] | None = None,
    runner: RunCommand = subprocess.run,
) -> Path:
    """Compile the AppKit/WebKit shell and return the ``.app`` directory.

    This requires macOS developer command-line tools and an interpreter where
    ``sentinel_evc`` is installed.  No Python runtime is copied into the app.
    """
    if platform.system() != "Darwin":
        raise RuntimeError("native app builds require macOS")
    app = Path(out).expanduser().resolve()
    if app.suffix != ".app":
        app = app / "Sentinel EVC.app"
    if app.exists():
        raise FileExistsError(f"app target already exists: {app}")
    # Keep the venv entry path.  Resolving its symlink would silently launch
    # the base interpreter without the environment's installed packages.
    python = Path(python_executable).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    source = (
        Path(source_file).expanduser().resolve()
        if source_file is not None
        else Path(__file__).resolve().parent / "native/SentinelApp.swift"
    )
    if not source.is_file():
        raise FileNotFoundError(
            "native Swift source is unavailable; pass source_file from a source checkout"
        )

    contents = app / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    executable = macos / "SentinelEVC"
    try:
        cache = contents / ".module-cache"
        cache.mkdir()
        _compile_swift(source, executable, cache, runner=runner)
        shutil.rmtree(cache)
        (contents / "Info.plist").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>SentinelEVC</string>
<key>CFBundleIdentifier</key><string>lab.sentinel.evc</string>
<key>CFBundleName</key><string>Sentinel EVC</string>
<key>CFBundleDisplayName</key><string>Sentinel EVC</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.3.0</string>
<key>CFBundleVersion</key><string>0.3.0</string>
<key>LSMinimumSystemVersion</key><string>12.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>\n""",
            encoding="utf-8",
        )
        config = {
            "schema_version": "native-app-v1",
            "python": str(Path(configured_python).expanduser().absolute()) if configured_python else str(python),
            "data_dir": str(Path(data_dir).expanduser().resolve()),
            "source_dir": str(Path(source_dir).resolve()) if source_dir else "",
            "bind": "127.0.0.1",
        }
        (resources / "app-config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        codesign = shutil.which("codesign")
        if not codesign:
            raise RuntimeError("codesign is required to finalize the macOS app bundle")
        _run([codesign, "--force", "--deep", "--sign", "-", str(app)], runner=runner)
    except BaseException:
        shutil.rmtree(app, ignore_errors=True)
        raise
    return app


def _resolve_source_checkout(source: str | os.PathLike[str] | None) -> Path:
    if source is None:
        packaged_checkout = Path(__file__).resolve().parents[2]
        origin = packaged_checkout if (packaged_checkout / "pyproject.toml").is_file() else Path.cwd()
    else:
        origin = Path(source)
    origin = origin.expanduser()
    if origin.is_symlink():
        raise ValueError("installation source root cannot be a symlink")
    origin = origin.resolve()
    if not origin.is_dir() or not (origin / "pyproject.toml").is_file():
        raise ValueError("installation source must be a source checkout with pyproject.toml")
    return origin


def _reject_recursive_target(destination: Path, source: Path) -> None:
    for copied_tree in (source / "src", source / "tests"):
        copied_tree = copied_tree.resolve()
        if destination == copied_tree or copied_tree in destination.parents:
            raise ValueError(
                f"installation target cannot be inside source/{copied_tree.name}"
            )


def _copy_public_source(source: str | os.PathLike[str] | None, destination: Path) -> Path:
    origin = _resolve_source_checkout(source)
    required = ("src", "tests", "pyproject.toml", "README.md", "LICENSE")
    for name in required:
        item = origin / name
        if item.is_symlink():
            raise ValueError(f"required public source asset is a symlink: {name}")
        if not item.exists():
            raise FileNotFoundError(f"required public source asset is missing: {name}")
        if name in {"src", "tests"}:
            if not item.is_dir():
                raise ValueError(f"required public source directory is unsafe: {name}")
            for nested in item.rglob("*"):
                if nested.is_symlink() or not (nested.is_dir() or nested.is_file()):
                    raise ValueError(f"public source tree contains a symlink or special file: {name}")
        elif not item.is_file():
            raise ValueError(f"required public source file is unsafe: {name}")
    tools = tuple(origin / "tools" / name for name in (
        "benchmark_performance.py", "validate_scenarios.py",
    ))
    for tool in tools:
        if tool.exists() or tool.is_symlink() or tool.parent.is_symlink():
            if tool.is_symlink() or tool.parent.is_symlink() or not tool.is_file():
                raise ValueError(f"public tool must be a regular file: {tool.name}")

    destination.mkdir()
    for name in required:
        item = origin / name
        target = destination / name
        if item.is_dir():
            shutil.copytree(
                item, target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
            )
        else:
            shutil.copy2(item, target)
    # Older source checkouts predate these public validators, so absence remains
    # compatible.  When present, copy only the audited tools rather than the
    # whole tools directory, which may contain local operator material.
    for tool in tools:
        if not tool.is_file():
            continue
        target = destination / "tools" / tool.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(tool, target)
    # Ship only the curated native VLA entrypoints. Heavy inference packages
    # stay in a separately selected environment; core installation stays small.
    native_files = (
        "experiments/vla/run_sentinel_libero.py",
        "experiments/vla/libero_native_profile.py",
        "experiments/vla/profiles/libero_native_qualified.json",
        "experiments/vla/export_native_scene.py",
        "experiments/vla/summarize_native_runs.py",
        "experiments/vla/run_libero_closedloop.py",
        "experiments/vla/libero_protocol.json",
        "experiments/vla/libero_portable_assets.json",
        "docs/native_vla_gateway.md",
    )
    for relative in native_files:
        item = origin / relative
        if not item.exists() and not item.is_symlink():
            continue
        if item.is_symlink() or any(parent.is_symlink() for parent in item.parents if parent != origin and origin in parent.parents) or not item.is_file():
            raise ValueError("native VLA source must be a regular public file")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    return destination


def _source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in source.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    ):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def install_system(
    target: str | os.PathLike[str],
    profile: str = "all",
    source: str | os.PathLike[str] | None = None,
    *,
    python_executable: str | os.PathLike[str] = sys.executable,
    build_app: bool = True,
    native_source: str | os.PathLike[str] | None = None,
    runner: RunCommand = subprocess.run,
    on_progress: Callable[[str, str], None] | None = None,
) -> InstallationResult:
    """Install into a new directory, publishing it only after every step passes."""
    if profile not in _PROFILES:
        raise ValueError("profile must be 'core' or 'all'")
    destination = Path(target).expanduser().resolve()
    source_checkout = _resolve_source_checkout(source)
    _reject_recursive_target(destination, source_checkout)
    _require_new_target(destination)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    def progress(step: str, event: str) -> None:
        if on_progress is not None:
            on_progress(step, event)
    try:
        progress("environment", "start")
        venv = stage / ".venv"
        clean_environment = _clean_install_environment()
        _run([os.fspath(python_executable), "-m", "venv", str(venv)],
             runner=runner, env=clean_environment)
        progress("environment", "done")
        progress("source", "start")
        python = _venv_python(venv)
        public_source = _copy_public_source(source_checkout, stage / "source")
        source_sha256 = _source_digest(public_source)
        build_source = stage / "build-source"
        shutil.copytree(public_source, build_source)
        progress("source", "done")
        progress("dependencies", "start")
        requirement = _package_requirement(build_source, _PROFILES[profile])
        _run([str(python), "-m", "pip", "install", requirement],
             runner=runner, env=clean_environment)
        shutil.rmtree(build_source)
        progress("dependencies", "done")

        progress("launchers", "start")
        commands = stage / "bin"
        commands.mkdir()
        command = commands / ("sentinel-evc.cmd" if os.name == "nt" else "sentinel-evc")
        final_python_expected = destination / python.relative_to(stage)
        final_source_expected = destination / public_source.relative_to(stage)
        if os.name == "nt":
            _write_windows_wrapper(command)
        else:
            _write_posix_wrapper(command)
        progress("launchers", "done")
        app = None
        if build_app:
            progress("app", "start")
            if platform.system() != "Darwin":
                raise RuntimeError("the desktop App is currently available on macOS only")
            app = build_macos_app(
                stage / "Sentinel EVC.app", python, destination / "data",
                source_file=native_source, source_dir=final_source_expected,
                configured_python=final_python_expected, runner=runner,
            )
            progress("app", "done")
        progress("publish", "start")
        (stage / "data").mkdir(exist_ok=True)
        (stage / "installation.json").write_text(
            json.dumps({
                "schema_version": "sentinel-install-v1",
                "profile": profile,
                "product_version": "0.3.0.dev0",
                "python_prerequisite": ">=3.10",
                "managed_venv": ".venv",
                "desktop_app": app is not None,
                "source_sha256": source_sha256,
                "source_frozen_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.rename(destination)
        final_python = destination / python.relative_to(stage)
        final_command = destination / command.relative_to(stage)
        final_app = destination / app.relative_to(stage) if app is not None else None
        result = InstallationResult(destination, final_python, final_command, final_app, profile)
        try:
            progress("publish", "done")
        except BaseException:
            # Publication already completed atomically.  A broken terminal or
            # optional observer cannot turn a successful install into a false
            # failure (including during Ctrl+C delivery at this exact point).
            pass
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
