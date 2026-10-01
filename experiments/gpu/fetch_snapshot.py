#!/usr/bin/env python3
"""Fetch a fixed public Hub snapshot against metadata obtained from its owner.

The metadata JSON must come from the official Hugging Face API with blobs=true.
Mirrors supply bytes only; LFS SHA256 or Git blob IDs establish their integrity.
No authentication or upload is performed.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path, algorithm, prefix=b""):
    value = hashlib.new(algorithm)
    value.update(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def fetch_file(item, *, root, repo, revision, dataset, endpoint):
    relative = item["rfilename"]
    path = root / relative
    if path.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError("invalid snapshot path")
    path.parent.mkdir(parents=True, exist_ok=True)

    def verified(candidate):
        if not candidate.is_file() or candidate.stat().st_size != item["size"]:
            return False
        lfs = item.get("lfs")
        if lfs:
            return digest(candidate, "sha256") == lfs["sha256"]
        prefix = f"blob {item['size']}\0".encode()
        return digest(candidate, "sha1", prefix) == item["blobId"]

    if not verified(path):
        partial = path.with_name(path.name + ".partial")
        prefix = "datasets/" if dataset else ""
        url = f"{endpoint.rstrip('/')}/{prefix}{repo}/resolve/{revision}/{relative}"
        subprocess.run(["curl", "-fL", "--retry", "2", "--connect-timeout", "20",
                        "--max-time", "3600", "-o", str(partial), url], check=True)
        if not verified(partial):
            raise ValueError(f"official digest mismatch: {relative}")
        partial.replace(path)
    result = {"path": relative, "size": path.stat().st_size,
              "sha256": digest(path, "sha256"), "official_blob_id": item.get("blobId")}
    print(json.dumps({"downloaded_and_verified": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-type", choices=("dataset", "model"), required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    revision, repo = metadata["sha"], metadata["id"]
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.repo_type == "dataset"
    files = [item for item in metadata["siblings"]
             if not item["rfilename"].endswith((".gif", ".ipynb"))
             and not item["rfilename"].startswith("onnx/")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: fetch_file(item, root=args.out, repo=repo,
                            revision=revision, dataset=dataset, endpoint=args.endpoint), files))
    (args.out / "download_manifest.json").write_text(json.dumps({
        "repo_id": repo, "revision": revision, "official_metadata_sha256": digest(args.metadata, "sha256"),
        "transport_endpoint": args.endpoint, "files": results, "completed_unix": time.time(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
