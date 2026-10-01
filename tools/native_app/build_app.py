#!/usr/bin/env python3
"""Build the native shell from a source checkout."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sentinel_evc.installation import build_macos_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", default="runs/workbench")
    args = parser.parse_args()
    app = build_macos_app(args.out, args.python, args.data_dir, source_dir=ROOT)
    print(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
