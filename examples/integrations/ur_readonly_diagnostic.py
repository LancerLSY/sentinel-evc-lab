"""Create a local UR Dashboard profile and run the three read-only diagnostic queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_evc.robot_connectors import RobotRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="Robot hostname or IP; no credentials are accepted")
    parser.add_argument("--port", type=int, default=29999)
    parser.add_argument("--store", type=Path, default=Path("runs/robots"))
    arguments = parser.parse_args()
    registry = RobotRegistry(arguments.store)
    profile = registry.create("UR read-only diagnostic", "ur_dashboard_readonly", arguments.host, arguments.port)
    print(json.dumps(registry.diagnose(profile["id"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
