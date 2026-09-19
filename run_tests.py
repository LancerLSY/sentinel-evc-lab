#!/usr/bin/env python3
"""用仓库批准的 pytest 依赖运行完整测试。"""

from __future__ import annotations

import sys

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", *sys.argv[1:]]))
