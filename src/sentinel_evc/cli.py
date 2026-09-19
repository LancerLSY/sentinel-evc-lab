"""命令行入口。

    python -m sentinel_evc demo   --cases 1000 --out runs/demo01
    python -m sentinel_evc verify --bundle runs/demo01/bundle \\
                                  --public-key runs/demo01/anchors/demo.public \\
                                  --run-id demo01
    python -m sentinel_evc tamper --out runs/demo01   # 四种篡改各跑一次
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .events import EventLog
from .evidence import failed_layers, verify_bundle
from .pipeline import act_one_geometry, act_three_evidence, act_two_faults
from .report import write_report


def _fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()


def cmd_demo(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"错误：输出目录 {out} 非空。新实验请用空目录，不要覆盖已有结果。",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or out.name
    log = EventLog(run_id=run_id)

    print(f"\n运行 {run_id} · {args.cases} 个几何案例\n")

    print("第一幕 · 变换让旧结论失效")
    geo = act_one_geometry(args.cases, log)

    w = (26, 12, 12)
    print("  " + _fmt_row(("判定路径", "错误放行", "全检次数"), w))
    print("  " + "-" * 50)
    print("  " + _fmt_row(
        ("a 只验父轨迹", geo["path_a_wrong_release"], 0), w))
    print("  " + _fmt_row(
        ("b 最终全检（基线）", geo["path_b_wrong_release"],
         geo["path_b_full_checks"]), w))
    print("  " + _fmt_row(
        ("c Δ-Cert + 必要全检", geo["path_c_wrong_release"],
         geo["path_c_full_checks"]), w))
    print(f"\n  实际违规的子轨迹        {geo['path_a_total_violating']}")
    print(f"  安全子轨迹通过          {geo['safe_children_passed']} / "
          f"{geo['safe_children_total']}（误拒 {geo['path_c_false_reject']}）")
    print(f"  直接继承                {geo['path_c_inherited']}")
    print(f"  完整检查调用减少        {geo.get('full_check_reduction_pct','—')}%"
          "  ← 是验证阶段调用次数，不是整机提速")
    print(f"  独立实现交叉验证分歧    {geo['cross_check_disagreements']}")

    print("\n第二幕 · 撤销与故障注入")
    faults = act_two_faults(log)
    for d in faults["details"]:
        mark = "✓" if d["blocked"] else "✗"
        print(f"  {mark} {d['fault']:<22} {str(d['code'] or '—'):<22}"
              f" 游标 {d['cursors'].get('submitted',0)}/"
              f"{d['cursors'].get('accepted',0)}/{d['cursors'].get('observed',0)}")
    print(f"\n  撤销后旧代次新增提交    "
          f"{faults['stale_gen_submissions_after_revoke']}  ← 这个数必须是 0")

    print("\n第三幕 · 证据导出与独立校验")
    bundle = act_three_evidence(log, str(out))
    ok, msg = verify_bundle(bundle["bundle_dir"], bundle["public_key"], run_id)
    print(f"  事件条数                {bundle['event_count']}")
    print(f"  独立校验                {'PASS' if ok else 'FAIL'} — {msg}")

    (out / "summary.json").write_text(
        json.dumps({"run_id": run_id, "geometry": {k: v for k, v in geo.items()
                                                   if k != "samples"},
                    "faults": faults, "bundle": bundle},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    report_path = write_report(str(out / "report.html"), run_id=run_id, geo=geo,
                               faults=faults, bundle=bundle, verify_msg=msg)
    print(f"\n  报告                    {report_path}")
    print(f"  证据包                  {bundle['bundle_dir']}")
    print(f"  公钥                    {bundle['public_key']}\n")
    return 0 if ok else 1


def cmd_verify(args) -> int:
    ok, msg = verify_bundle(args.bundle, args.public_key, args.run_id)
    print(("PASS — " if ok else "FAIL — ") + msg)
    return 0 if ok else 1


def cmd_tamper(args) -> int:
    """四种篡改，各自应给出不同的失败原因。"""
    src = Path(args.out)
    run_id = args.run_id or src.name
    bundle = src / "bundle"
    pub = src / "anchors" / "demo.public"

    if not bundle.exists():
        print(f"错误：{bundle} 不存在，请先运行 demo。", file=sys.stderr)
        return 2

    ok, msg = verify_bundle(str(bundle), str(pub), run_id)
    print(f"原包                  {'PASS' if ok else 'FAIL'} — {msg}\n")

    tmp = src / "_tamper"
    cases = []

    # 1. 改一个字节
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(bundle, tmp)
    p = tmp / "events.jsonl"
    data = bytearray(p.read_bytes())
    mid = len(data) // 2
    for i, b in enumerate(data):
        if chr(b).isdigit() and i > mid:
            data[i] = ord("9") if chr(b) != "9" else ord("8")
            break
    p.write_bytes(bytes(data))
    cases.append(("改事件里 1 个字节", failed_layers(str(tmp), str(pub), run_id),
                  verify_bundle(str(tmp), str(pub), run_id)))

    # 2. 删尾
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(bundle, tmp)
    p = tmp / "events.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    cases.append(("删掉最后 3 行", failed_layers(str(tmp), str(pub), run_id),
                  verify_bundle(str(tmp), str(pub), run_id)))

    # 3. 换公钥
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    wrong_pub = src / "_wrong.public"
    wrong_pub.write_bytes(
        Ed25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw))
    cases.append(("换一把公钥", failed_layers(str(bundle), str(wrong_pub), run_id),
                  verify_bundle(str(bundle), str(wrong_pub), run_id)))

    # 4. 换 run_id
    cases.append(("改 run_id", failed_layers(str(bundle), str(pub), "wrong-run"),
                  verify_bundle(str(bundle), str(pub), "wrong-run")))

    for name, layers, (o, m) in cases:
        print(f"{name:<22}{'PASS ← 不该通过！' if o else 'FAIL'}")
        print(f"{'':<22}失败层 {', '.join(layers)}")
        print(f"{'':<22}{m}\n")

    shutil.rmtree(tmp, ignore_errors=True)
    wrong_pub.unlink(missing_ok=True)

    all_failed = all(not o for _, _, (o, _) in cases)
    profiles = {layers for _, layers, _ in cases}
    print(f"四种篡改全部失败：{'是' if all_failed else '否'}"
          f"　不同失败特征数：{len(profiles)} / 4")
    return 0 if all_failed and len(profiles) == 4 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sentinel_evc", description="Sentinel EVC Lab · 数值参考实现")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="跑完整三幕")
    d.add_argument("--cases", type=int, default=1000)
    d.add_argument("--out", default="runs/demo01")
    d.add_argument("--run-id", default=None)
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("verify", help="独立校验一个证据包")
    v.add_argument("--bundle", required=True)
    v.add_argument("--public-key", required=True)
    v.add_argument("--run-id", required=True)
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("tamper", help="四种篡改测试")
    t.add_argument("--out", required=True)
    t.add_argument("--run-id", default=None)
    t.set_defaults(func=cmd_tamper)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
