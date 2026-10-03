#!/usr/bin/env python3
"""Frozen N=4 extension of the counterbalanced arm protocol."""

import hashlib
import json
import sys
from pathlib import Path

import run_counterbalanced_arm as base


HERE = Path(__file__).resolve()
RUNNER = Path(base.__file__).resolve()
base.NS = (4,)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extension_hashes():
    return {"n4_wrapper.py": digest(HERE), "run_counterbalanced_arm.py": digest(RUNNER)}


def argument(name):
    i = sys.argv.index(name)
    return Path(sys.argv[i + 1])


def main():
    command = sys.argv[1]
    if command == "prepare":
        trace_dir = argument("--trace-dir")
        pre = {"status": "N4_EXTENSION_DECLARED_BEFORE_TRACE_OR_TIMING",
               "source_sha256": extension_hashes()}
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "n4-extension.pre-timing.json").write_text(json.dumps(pre, indent=2))
        base.main()
        frozen = trace_dir / "protocol.frozen.json"
        protocol = json.load(open(frozen))
        protocol["extension"] = pre
        frozen.write_text(json.dumps(protocol, indent=2))
        requested = argument("--protocol")
        if requested.resolve() != frozen.resolve():
            requested.write_text(json.dumps(protocol, indent=2))
    elif command == "run":
        protocol = json.load(open(argument("--trace-dir") / "protocol.frozen.json"))
        if protocol.get("extension", {}).get("source_sha256") != extension_hashes():
            raise RuntimeError("N=4 wrapper or underlying frozen runner changed")
        base.main()
    else:
        base.main()


if __name__ == "__main__":
    main()
