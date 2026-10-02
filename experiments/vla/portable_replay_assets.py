#!/usr/bin/env python3
"""Strict content binding for reconstructed LIBERO replay assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "sentinel-libero-portable-assets-v1"
REPO_ID = "lerobot/libero-assets"
REPO_TYPE = "dataset"
REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
HEX = frozenset("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and set(value) <= HEX


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    _require(
        isinstance(value, str)
        and value == path.as_posix()
        and not path.is_absolute()
        and bool(path.parts)
        and all(part not in ("", ".", "..") for part in path.parts),
        f"invalid portable asset path: {value!r}",
    )
    return path


def load_portable_assets_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"portable asset manifest is missing or a symlink: {path}")
    receipt_sha = _sha256(path)
    _require(receipt_sha == expected_sha256, "portable asset manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == SCHEMA, "portable asset manifest schema mismatch")
    _require(manifest.get("repo_id") == REPO_ID, "portable asset repository mismatch")
    _require(manifest.get("repo_type") == REPO_TYPE, "portable asset repository type mismatch")
    _require(manifest.get("revision") == REVISION, "portable asset revision mismatch")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "portable asset manifest has no files")
    for relative, entry in files.items():
        _relative_path(relative)
        _require(isinstance(entry, dict), f"invalid portable asset entry: {relative}")
        _require(type(entry.get("bytes")) is int and entry["bytes"] >= 0, f"invalid byte count: {relative}")
        _require(_is_digest(entry.get("sha256"), 64), f"invalid SHA-256: {relative}")
        official = entry.get("official")
        _require(isinstance(official, dict), f"official digest missing: {relative}")
        algorithm = official.get("algorithm")
        digest = official.get("digest")
        _require(algorithm in ("sha256", "git-blob-sha1"), f"unsupported official digest: {relative}")
        _require(_is_digest(digest, 64 if algorithm == "sha256" else 40), f"invalid official digest: {relative}")
    return manifest


def verify_portable_assets(root: Path, manifest_path: Path, expected_sha256: str) -> dict[str, Any]:
    manifest = load_portable_assets_manifest(manifest_path, expected_sha256)
    _require(root.is_dir() and not root.is_symlink(), f"LIBERO asset root is missing or a symlink: {root}")
    declared = set(manifest["files"])
    actual: set[str] = set()
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = [name for name in names if name != ".cache"]
        for name in names:
            _require(not (directory_path / name).is_symlink(), f"asset directory symlink is forbidden: {directory_path / name}")
        for name in filenames:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if relative.endswith(".incomplete"):
                continue
            _require(not path.is_symlink(), f"asset file symlink is forbidden: {relative}")
            _require(path.is_file(), f"non-regular asset path: {relative}")
            actual.add(relative)
    _require(actual == declared, f"portable asset coverage mismatch: undeclared={sorted(actual - declared)}, missing={sorted(declared - actual)}")

    records: list[str] = []
    total_bytes = 0
    for relative in sorted(declared):
        pure = _relative_path(relative)
        path = root.joinpath(*pure.parts)
        cursor = root
        for part in pure.parts:
            cursor = cursor / part
            _require(not cursor.is_symlink(), f"asset path symlink is forbidden: {relative}")
        entry = manifest["files"][relative]
        size = path.stat().st_size
        _require(size == entry["bytes"], f"asset byte count mismatch: {relative}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(b"blob " + str(size).encode("ascii") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
        computed = sha256.hexdigest()
        _require(computed == entry["sha256"], f"asset SHA-256 mismatch: {relative}")
        official = entry["official"]
        observed_official = computed if official["algorithm"] == "sha256" else git_blob.hexdigest()
        _require(observed_official == official["digest"], f"official asset digest mismatch: {relative}")
        total_bytes += size
        records.append(f"{relative}\0{computed}\0{size}\n")
    canonical = hashlib.sha256("".join(records).encode("utf-8")).hexdigest()
    return {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": expected_sha256, "bytes": manifest_path.stat().st_size},
        "source": {"repo_id": REPO_ID, "repo_type": REPO_TYPE, "revision": REVISION},
        "content": {
            "scope": "canonical_reconstructed_content",
            "file_count": len(declared),
            "total_bytes": total_bytes,
            "tree_sha256": canonical,
            "coverage": "all regular non-cache non-incomplete files under configured LIBERO assets root",
            "excluded": [".cache/**", "*.incomplete"],
        },
        "root": str(root.resolve()),
        "historical_whole_tree_match_claimed": False,
    }
