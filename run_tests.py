#!/usr/bin/env python3
"""备用测试运行器：在没有 pytest 的环境里跑测试。

正式开发请用 `python -m pytest -q`。这个脚本只实现 pytest 的一个极小子集
（`raises`、`skip`、`parametrize` 、`monkeypatch.setattr` 和 `tmp_path`），存在的意义是让「我这台机器装不上依赖」不成为
跑不了测试的借口。CI 里用真的 pytest。
"""

from __future__ import annotations

import re
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path


class _Raises:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__}，但没有抛出")
        if not issubclass(et, self.exc_type):
            return False
        if self.match is not None and not re.search(self.match,str(ev)):
            raise AssertionError("异常信息与预期模式不符")
        self.value = ev
        return True


class _Skipped(Exception):
    pass


def _skip(reason):
    raise _Skipped(reason)


class _MonkeyPatch:
    def __init__(self):
        self._changes = []

    def setattr(self, target, name, value):
        previous = getattr(target, name)
        self._changes.append((target, name, previous))
        setattr(target, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for target, name, previous in reversed(self._changes):
            setattr(target, name, previous)


def _parametrize(names, values):
    names = tuple(part.strip() for part in names.split(","))
    def decorate(fn):
        cases = []
        for value in values:
            row = (value,) if len(names) == 1 else tuple(value)
            if len(row) != len(names):
                raise ValueError("parameter count mismatch")
            cases.append(dict(zip(names, row)))
        previous = getattr(fn, "_parameter_cases", [{}])
        fn._parameter_cases = [{**first, **second} for first in previous for second in cases]
        return fn
    return decorate


def _install_pytest_shim() -> None:
    if "pytest" in sys.modules:
        return
    shim = types.ModuleType("pytest")
    shim.raises = _Raises
    shim.skip = _skip
    shim.mark = types.SimpleNamespace(parametrize=_parametrize)
    sys.modules["pytest"] = shim


def main() -> int:
    _install_pytest_shim()
    root = Path(__file__).parent
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))

    import importlib

    modules = sorted(p.stem for p in (root / "tests").glob("test_*.py"))
    passed = failed = skipped = 0
    failures = []

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        for name, fn in sorted(vars(mod).items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            for case_index, case in enumerate(getattr(fn, "_parameter_cases", [{}])):
                with tempfile.TemporaryDirectory() as tmp, _MonkeyPatch() as patch:
                    kwargs = dict(case)
                    if "tmp_path" in inspect.signature(fn).parameters:
                        kwargs["tmp_path"] = Path(tmp)
                    if "monkeypatch" in inspect.signature(fn).parameters:
                        kwargs["monkeypatch"] = patch
                    try:
                        fn(**kwargs)
                        passed += 1
                        print(".", end="", flush=True)
                    except _Skipped:
                        skipped += 1
                        print("s", end="", flush=True)
                    except Exception:
                        failed += 1
                        print("F", end="", flush=True)
                        failures.append((f"{mod_name}::{name}[{case_index}]", traceback.format_exc()))

    print()
    for label, tb in failures:
        print(f"\n{'=' * 60}\nFAILED {label}\n{'-' * 60}\n{tb}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
