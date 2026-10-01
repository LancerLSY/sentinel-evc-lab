"""Import one local model into a Sentinel asset store and print its preview metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_evc.assets import AssetStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--format", choices=("obj", "stl", "mjcf", "urdf"), required=True)
    parser.add_argument("--store", type=Path, default=Path("runs/assets"))
    arguments = parser.parse_args()
    store = AssetStore(arguments.store)
    asset = store.import_asset(arguments.path.name, arguments.format, content=arguments.path.read_bytes())
    print(json.dumps({"asset": asset, "geometry": store.geometry(asset["id"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
