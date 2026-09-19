"""Sentinel-EVC v0.1 数值 Demo 的四个命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from .contracts import canonical_json
from .events import EventLog
from .evidence import verify_bundle
from .pipeline import FAULTS, act_one_geometry, act_three_evidence, act_two_faults
from .report import write_report
from .scenarios import DT, HORIZON, make_scene


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def _new_output(path: str) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在: {output}")
    output.mkdir(parents=True)
    return output


def _scenario(seed: int) -> dict:
    value = make_scene(seed).summary()
    value.update(dt=DT, horizon=HORIZON, seed=seed)
    return value


def _write_json(path: Path, value) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _events(bundle_dir: str) -> list[dict]:
    raw = Path(bundle_dir, "events.jsonl").read_bytes()
    return [json.loads(line) for line in raw.splitlines()]


def _bundle_summary(bundle: dict) -> dict:
    return {
        "bundle_dir": "bundle",
        "public_key": "anchors/demo.public",
        "event_count": bundle["event_count"],
        "tip_hash": bundle["tip_hash"],
    }


def _geometry_ok(stats: dict) -> bool:
    return (
        stats["path_b_wrong_release"] == 0
        and stats["path_c_wrong_release"] == 0
        and stats["path_c_false_reject"] == 0
        and stats["safe_children_passed"] == stats["safe_children_total"]
        and stats["cross_check_disagreements"] == 0
    )


def _faults_ok(stats: dict) -> bool:
    return (
        stats["faults_run"] == stats["blocked"]
        and stats["stale_gen_submissions_after_revoke"] == 0
    )


def cmd_demo(args) -> int:
    try:
        output = _new_output(args.out)
    except (FileExistsError, OSError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1

    run_id = output.name
    started = perf_counter()
    with tempfile.TemporaryFile() as spool:
        log = EventLog(run_id, spool=spool)
        geometry = act_one_geometry(args.cases, log, seed=args.seed)
        faults = act_two_faults(log, seed=args.seed)
        bundle = act_three_evidence(log, str(output))

    ok, message = verify_bundle(bundle["bundle_dir"], bundle["public_key"], run_id)
    _write_json(output / "scenario.json", _scenario(args.seed))
    geometry["end_to_end_wall_seconds"] = round(perf_counter() - started, 6)
    _write_json(output / "summary.json", {
        "run_id": run_id,
        "geometry": {key: value for key, value in geometry.items() if key != "samples"},
        "faults": faults,
        "bundle": _bundle_summary(bundle),
    })
    write_report(
        str(output / "report.html"), run_id=run_id, geo=geometry, faults=faults,
        bundle=bundle, verify_msg=message, events=_events(bundle["bundle_dir"]),
    )
    success = ok and _geometry_ok(geometry) and _faults_ok(faults)
    print(("PASS" if success else "FAIL") + f" — {message}")
    return 0 if success else 1


def cmd_geometry(args) -> int:
    try:
        output = _new_output(args.out)
    except (FileExistsError, OSError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1

    started = perf_counter()
    with tempfile.TemporaryFile() as spool:
        stats = act_one_geometry(
            args.cases, EventLog(output.name, spool=spool), seed=args.seed,
        )
    stats["end_to_end_wall_seconds"] = round(perf_counter() - started, 6)
    _write_json(output / "scenario.json", _scenario(args.seed))
    _write_json(output / "summary.json", {"run_id": output.name, "geometry": stats})
    success = _geometry_ok(stats)
    print(("PASS" if success else "FAIL") +
          f" — {stats['cases']} cases, {stats['cross_check_disagreements']} disagreements")
    return 0 if success else 1


def cmd_fault(args) -> int:
    try:
        output = _new_output(args.out)
    except (FileExistsError, OSError) as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryFile() as spool:
        log = EventLog(output.name, spool=spool)
        faults = act_two_faults(log, fault=args.fault, seed=args.seed)
        bundle = act_three_evidence(log, str(output))
    ok, message = verify_bundle(bundle["bundle_dir"], bundle["public_key"], output.name)
    _write_json(output / "summary.json", {
        "run_id": output.name, "faults": faults, "bundle": _bundle_summary(bundle),
    })
    success = ok and _faults_ok(faults)
    print(("PASS" if success else "FAIL") + f" — {message}")
    return 0 if success else 1


def cmd_verify(args) -> int:
    ok, message = verify_bundle(args.bundle, args.public_key, args.run_id)
    print(("PASS — " if ok else "FAIL — ") + message)
    return 0 if ok else 1


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="sentinel_evc")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="运行完整三幕并生成证据包")
    demo.add_argument("--cases", type=_positive_int, default=1000)
    demo.add_argument("--seed", type=int, default=1234)
    demo.add_argument("--out", default="runs/demo01")
    demo.set_defaults(func=cmd_demo)

    geometry = commands.add_parser("geometry", help="运行固定几何案例")
    geometry.add_argument("--cases", type=_positive_int, default=1000)
    geometry.add_argument("--seed", type=int, default=1234)
    geometry.add_argument("--out", default="runs/geometry")
    geometry.set_defaults(func=cmd_geometry)

    fault = commands.add_parser("fault", help="运行执行故障实验")
    fault.add_argument("--fault", choices=(*FAULTS, "all"), required=True)
    fault.add_argument("--seed", type=int, default=1234)
    fault.add_argument("--out", default="runs/fault")
    fault.set_defaults(func=cmd_fault)

    verify = commands.add_parser("verify", help="独立校验证据包")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--public-key", required=True)
    verify.add_argument("--run-id", required=True)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
