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


def test_readme_marks_out_of_scope_capabilities_as_unimplemented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = [line.split("|") for line in readme.splitlines() if line.startswith("|")]
    for capability in ("真实 VLA", "WorldGuard", "真机"):
        statuses = [cells[2] for cells in rows if capability in cells[1]]
        assert statuses, f"能力状态表缺少 {capability}"
        for status in statuses:
            assert re.search(r"未开始|无实现|未实现|不在本轮范围", status)
            assert not re.search(r"已实现|已接入|已完成", status)


def test_readme_keeps_the_proposed_license_inactive():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paragraphs = [part for part in readme.split("\n\n") if "LICENSE.proposed" in part]
    assert (ROOT / "LICENSE.proposed").is_file()
    assert not (ROOT / "LICENSE").exists()
    assert any(re.search(r"尚未生效|未生效|尚未正式授权开源", part) for part in paragraphs)


def test_gitignore_covers_secrets_runs_and_local_handoff():
    rules = {
        line.strip().removeprefix("/")
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for pattern in ("*.pem", "*.key", "runs/", "SESSION_HANDOFF.md"):
        assert pattern in rules
