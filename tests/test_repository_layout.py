from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_directories_exist():
    for name in (
        "src/sentinel_evc", "schemas", "docs", "worldguard", "tests", ".github/workflows"
    ):
        assert (ROOT / name).is_dir(), name


def test_package_entry_and_repository_files_exist():
    for name in (
        "src/sentinel_evc/__init__.py",
        "src/sentinel_evc/__main__.py",
        "pyproject.toml",
        "README.md",
        ".github/workflows/ci.yml",
        "worldguard/README_WHY_EMPTY.md",
    ):
        assert (ROOT / name).is_file(), name


def test_worldguard_contains_only_its_scope_explanation():
    files = {
        path.relative_to(ROOT / "worldguard").as_posix()
        for path in (ROOT / "worldguard").rglob("*") if path.is_file()
    }
    assert files == {"README_WHY_EMPTY.md"}
