"""仓库骨架与公开声明的最小回归约束。"""

import importlib
import importlib.util
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_package_import_resolves_to_src_layout():
    package = importlib.import_module("sentinel_evc")
    assert Path(package.__file__).resolve().parent == ROOT / "src" / "sentinel_evc"


def test_package_has_a_python_module_entrypoint():
    assert importlib.util.find_spec("sentinel_evc.__main__") is not None


def test_dependency_groups_contain_only_approved_packages():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    groups = {
        "runtime": (project["dependencies"], {"numpy", "cryptography"}),
        "test": (project["optional-dependencies"]["test"], {"pytest"}),
    }
    for group, (requirements, expected) in groups.items():
        names = {
            re.match(r"[A-Za-z0-9_.-]+", item).group().lower()
            for item in requirements
        }
        assert names == expected, group


def test_readmes_mark_out_of_scope_capabilities_as_unimplemented():
    readmes = {
        "README.md": (
            ("Real VLA integration", "Not started"),
            ("WorldGuard", "public interface undefined"),
            ("Physical robot", "Not started"),
        ),
        "README.zh-CN.md": (
            ("真实 VLA", "未开始"),
            ("WorldGuard", "公开接口未定义"),
            ("真机", "未开始"),
        ),
    }
    for name, expected in readmes.items():
        rows = [
            line for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
            if line.startswith("|")
        ]
        for capability, status in expected:
            matching = [line for line in rows if capability in line]
            assert matching, f"{name} 能力状态表缺少 {capability}"
            assert status in matching[0]


def test_readmes_match_the_active_mit_license():
    assert (ROOT / "LICENSE").is_file()
    assert not (ROOT / "LICENSE.proposed").exists()
    for name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / name).read_text(encoding="utf-8")
        assert "MIT" in readme
        assert "[LICENSE](LICENSE)" in readme


def test_gitignore_covers_secrets_runs_and_local_handoff():
    rules = {
        line.strip().removeprefix("/")
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for pattern in ("*.pem", "*.key", "runs/", "SESSION_HANDOFF.md"):
        assert pattern in rules
