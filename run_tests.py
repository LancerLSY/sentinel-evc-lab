#!/usr/bin/env python3
"""备用测试运行器：在没有 pytest 的环境里跑测试。

正式开发请用 `python -m pytest -q`。这个脚本只实现 pytest 的一个极小子集
（`raises` 和 `tmp_path`），存在的意义是让「我这台机器装不上依赖」不成为
跑不了测试的借口。CI 里用真的 pytest。
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__}，但没有抛出")
        if not issubclass(et, self.exc_type):
            return False
        self.value = ev
        return True


def _install_pytest_shim() -> None:
    if "pytest" in sys.modules:
        return
    shim = types.ModuleType("pytest")
    shim.raises = _Raises
    sys.modules["pytest"] = shim


def main() -> int:
    _install_pytest_shim()
    root = Path(__file__).parent
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))

    import importlib

    modules = sorted(p.stem for p in (root / "tests").glob("test_*.py"))
    passed = failed = 0
    failures = []

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        for name, fn in sorted(vars(mod).items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = tempfile.mkdtemp()
                kwargs["tmp_path"] = Path(tmp)
            try:
                fn(**kwargs)
                passed += 1
                print(".", end="", flush=True)
            except Exception:
                failed += 1
                print("F", end="", flush=True)
                failures.append((f"{mod_name}::{name}", traceback.format_exc()))

    print()
    for label, tb in failures:
        print(f"\n{'=' * 60}\nFAILED {label}\n{'-' * 60}\n{tb}")

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
