"""核对 `RESULTS.md` 里写的数字与随包 `sample_run/` 的实际输出。

用法：
    python tools/check_results.py [--sample-run sample_run] [--results RESULTS.md]

它在查三件事：

1. `sample_run/summary.json` 里的值 = 本文件登记的基线（与 AGENTS.md 第 2 节一致）；
2. `RESULTS.md` 的表格里，每一行都真的写了那个值；
3. 两边的 `run_id` 对得上。

为什么要这个脚本：`RESULTS.md` 是手写的，`sample_run/` 是机器生成的。手写的那份
一旦和随包输出脱节，整页「结果与复现」就变成了一段说法而不是证据。CI 里会跑它。

注意它**不**核对「测试通过 N 项」——那个数由 pytest 自己产生，
CI 里另有一个作业跑测试；这个脚本只管数字与运行产物。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# (RESULTS.md 里的指标名, summary.json 里的取值函数, 期望渲染出来的样子)
ROWS = [
    ("构造案例总数", lambda s: s["geometry"]["cases"], "1000"),
    ("实际违规的子轨迹", lambda s: s["geometry"]["path_a_total_violating"], "500"),
    ("路径 a 错误放行", lambda s: s["geometry"]["path_a_wrong_release"], "500"),
    ("路径 b 错误放行", lambda s: s["geometry"]["path_b_wrong_release"], "0"),
    ("路径 c 错误放行", lambda s: s["geometry"]["path_c_wrong_release"], "0"),
    ("路径 b 完整检查调用", lambda s: s["geometry"]["path_b_full_checks"], "1000"),
    ("路径 c 完整检查调用", lambda s: s["geometry"]["path_c_full_checks"], "500"),
    ("直接继承次数", lambda s: s["geometry"]["path_c_inherited"], "500"),
    ("安全子轨迹通过", lambda s: s["geometry"]["safe_children_passed"], "500 / 500"),
    ("误拒", lambda s: s["geometry"]["path_c_false_reject"], "0"),
    ("独立实现交叉验证分歧", lambda s: s["geometry"]["cross_check_disagreements"], "0"),
    ("父证书建立的完整检查", lambda s: s["geometry"]["root_full_checks"], "1000"),
    ("注入故障类型数", lambda s: s["faults"]["faults_run"], "7"),
    ("阻断的故障数", lambda s: s["faults"]["blocked"], "7"),
    ("撤销后旧代次新增提交", lambda s: s["faults"]["stale_gen_submissions_after_revoke"], "0"),
    ("事件条数", lambda s: s["bundle"]["event_count"], "2031"),
]


def _row_value(md: str, label: str):
    """从 markdown 表格里取出 `| label | value | ... |` 的 value 单元格。"""
    # 标签可能被 ** 包起来，值也可能被 ** 包起来，都容忍
    pat = re.compile(
        r"^\|\s*\**\s*" + re.escape(label) + r"\s*\**\s*\|([^|]*)\|", re.M)
    m = pat.search(md)
    return m.group(1).strip() if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="核对 RESULTS.md 与 sample_run/")
    ap.add_argument("--sample-run", default="sample_run")
    ap.add_argument("--results", default="RESULTS.md")
    args = ap.parse_args(argv)

    summary_path = Path(args.sample_run) / "summary.json"
    results_path = Path(args.results)
    problems = []

    if not summary_path.exists():
        print(f"找不到 {summary_path}", file=sys.stderr)
        return 2
    if not results_path.exists():
        print(f"找不到 {results_path}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    md = results_path.read_text(encoding="utf-8")

    print(f"run_id      {summary.get('run_id')}")
    print(f"对照文件    {summary_path}  ↔  {results_path}")
    print(f"{'指标':<24}{'summary.json':>14}{'RESULTS.md':>14}  期望")
    print("-" * 68)

    for label, getter, expected in ROWS:
        try:
            actual = getter(summary)
        except (KeyError, TypeError) as exc:
            problems.append(f"{label}: summary.json 里取不到值（{exc}）")
            continue
        rendered = f"{actual} / {actual}" if expected == "500 / 500" else str(actual)
        cell = _row_value(md, label)
        mark = "OK"
        if rendered != expected:
            problems.append(
                f"{label}: 随包输出是 {rendered}，登记的基线是 {expected}")
            mark = "MISMATCH"
        elif cell is None:
            problems.append(f"{label}: RESULTS.md 里找不到这一行")
            mark = "MISSING"
        elif expected not in cell:
            problems.append(
                f"{label}: RESULTS.md 写的是「{cell}」，随包输出是「{expected}」")
            mark = "STALE"
        print(f"{label:<24}{rendered:>14}{str(cell):>14}  {expected}   {mark}")

    # run_id 一致性：manifest 与 summary 必须是同一次运行
    run_id = summary.get("run_id")
    if run_id and run_id not in md:
        problems.append(f"RESULTS.md 里没有出现 run_id「{run_id}」")

    # 行尾守卫：证据包被 checkout 转换过，独立校验就一定失败。
    # Linux 上的 CI 不会转换，所以必须在这里显式挡住（见仓库根的 .gitattributes）。
    events_path = Path(args.sample_run) / "bundle" / "events.jsonl"
    if events_path.exists():
        raw = events_path.read_bytes()
        cr = raw.count(b"\r")
        print(f"{'events.jsonl 行尾':<24}{'CR 字节 ' + str(cr):>28}")
        if cr:
            problems.append(
                f"events.jsonl 里有 {cr} 个 CR 字节：工作区被做过 LF→CRLF 转换，"
                "随包证据包已不可独立校验（检查 .gitattributes / core.autocrlf）")

    print("-" * 68)
    if problems:
        print(f"不一致 {len(problems)} 处：", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        print("RESULTS.md 与 sample_run/ 已经脱节：改数字还是改文档，都得说清原因。",
              file=sys.stderr)
        return 1

    print(f"全部一致：{len(ROWS)} 行，run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
