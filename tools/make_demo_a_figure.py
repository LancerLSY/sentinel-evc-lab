"""生成「第一幕 · 变换让旧结论失效」的对照图（Demo A）。

用法：
    python tools/make_demo_a_figure.py --out docs/demo_a.svg --cases 1000

三条硬约束（来自《开源 Demo 实施方案》第十一节 README 骨架）：

1. **配图必须来自真实运行。** 本脚本自己跑一遍第一幕流水线，
   图上每个数字都是这次运行算出来的，脚本会先跟基线断言，不一致就报错退出。
2. **纯 Python 字符串模板 + 内联 SVG。** 没有依赖、没有 CDN、没有构建步骤，
   生成的文件双击就能看，断网也能看。
3. **不说过头话。** 这是构造的数值几何场景，不是真机、不是 VLA、不是安全证明。

图里画的是一个具体案例（案例 #0），不是三次运行拼起来的：
同一条父轨迹 P1、P2 各自完整检查都通过，异侧混合之后却从障碍里穿过去。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从仓库根直接 `python tools/make_demo_a_figure.py` 运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_evc.contracts import Plan  # noqa: E402
from sentinel_evc.events import EventLog  # noqa: E402
from sentinel_evc.geometry import full_check  # noqa: E402
from sentinel_evc.pipeline import act_one_geometry  # noqa: E402
from sentinel_evc.scenarios import make_parent_pair, make_scene  # noqa: E402

# 与 AGENTS.md 第 2 节 / RESULTS.md 第一幕数字一致的基线。
# 这里断言不是为了「对齐数字」，而是为了不让图比仓库的状态先行一步。
BASELINE = {
    "cases": 1000,
    "path_a_wrong_release": 500,
    "path_a_total_violating": 500,
    "path_b_wrong_release": 0,
    "path_b_full_checks": 1000,
    "path_c_wrong_release": 0,
    "path_c_full_checks": 500,
    "path_c_inherited": 500,
    "path_c_false_reject": 0,
    "safe_children_passed": 500,
    "safe_children_total": 500,
    "cross_check_disagreements": 0,
}

W, H = 1040, 560
PX, PY, SZ = 64.0, 74.0, 380.0  # 轨迹绘图区：左上角 + 边长（正方形，按米等比）
PANEL_X = 500.0  # 右侧文字/表格起点

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',"
        "'Microsoft YaHei',sans-serif")

C_ACC = "#3a5f8a"   # 父轨迹 P1
C_OK = "#2d7a4f"    # 父轨迹 P2
C_BAD = "#b3452f"   # 混合后的最终动作
C_MUT = "#6b6b66"
C_LINE = "#d8d7d2"
C_FG = "#1a1a18"


def mm(x: float) -> str:
    """米 → 带符号的毫米字符串。"""
    return f"{x * 1000:+.1f}"


def run_act_one(cases: int) -> dict:
    """真跑一遍第一幕。返回流水线统计（含前 8 个案例的轨迹样本）。"""
    log = EventLog(run_id="demo-a-figure")
    geo = act_one_geometry(cases, log)
    bad = {
        k: (geo.get(k), v)
        for k, v in BASELINE.items()
        if geo.get(k) != v
    }
    if bad:
        print("第一幕结果与基线不一致，拒绝出图：", file=sys.stderr)
        for k, (got, want) in bad.items():
            print(f"  {k}: 本次 {got} · 基线 {want}", file=sys.stderr)
        raise SystemExit(2)
    return geo


def _plot_mapping(scene):
    """米 → 像素。视野取整个工作空间盒（x∈[0,0.8]，y∈[-0.4,0.4]），等比缩放。"""
    x0, y0 = scene.ws_lo[0], scene.ws_lo[1]
    x1, y1 = scene.ws_hi[0], scene.ws_hi[1]
    s = min(SZ / (x1 - x0), SZ / (y1 - y0))

    def tx(p):
        return PX + (p[0] - x0) * s, PY + SZ - (p[1] - y0) * s

    return tx, s


def _polyline(pts, tx, **attrs) -> str:
    d = " ".join(
        ("M" if i == 0 else "L") + f"{tx(p)[0]:.1f},{tx(p)[1]:.1f}"
        for i, p in enumerate(pts)
    )
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" fill="none" {a}/>'


def _segments(plan):
    return [(plan.knots[k], plan.knots[k + 1]) for k in range(plan.horizon)]


def _closest_point_on_segment(p0, p1, c):
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    dd = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    if dd == 0.0:
        return p0
    t = ((c[0] - p0[0]) * d[0] + (c[1] - p0[1]) * d[1]
         + (c[2] - p0[2]) * d[2]) / dd
    t = max(0.0, min(1.0, t))
    return tuple(p0[i] + t * d[i] for i in range(3))


def build_svg(geo: dict, cases: int) -> str:
    scene = make_scene(0)
    p1, p2 = make_parent_pair(0)
    sample = geo["samples"][0]          # 案例 #0 —— 异侧混合，实际违规
    child = sample["child_knots"]
    child = [tuple(p) for p in child]
    p1_pts = [tuple(p) for p in sample["parent_knots"]]

    obs = scene.obstacles[0]
    clearance = obs.radius + scene.tool_radius + scene.tracking_reserve
    ok_p1, m_p1, _ = full_check(p1, scene)
    ok_p2, m_p2, _ = full_check(p2, scene)
    child_plan = Plan(plan_id=f"MIX-{sample['case']:06d}", knots=tuple(child),
                      dt=p1.dt)
    ok_c, m_c, first_bad = full_check(child_plan, scene)
    child_segments = _segments(child_plan)

    tx, s = _plot_mapping(scene)
    cx, cy = tx(obs.center)
    r_obs = obs.radius * s
    r_clr = clearance * s

    # 障碍轮廓 + 禁区（球半径 + 工具半径 + 跟踪预留）
    parts = [
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_obs:.1f}" '
        f'fill="{C_BAD}" opacity=".16"/>',
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_obs:.1f}" fill="none" '
        f'stroke="{C_BAD}" stroke-width="1.5" opacity=".65"/>',
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_clr:.1f}" fill="none" '
        f'stroke="{C_BAD}" stroke-width="1" stroke-dasharray="3 3" opacity=".5"/>',
    ]

    # 工作空间盒
    x0p, y0p = tx((scene.ws_lo[0], scene.ws_hi[1], 0))
    parts.append(
        f'<rect x="{x0p:.1f}" y="{y0p:.1f}" width="{SZ:.1f}" height="{SZ:.1f}" '
        f'fill="none" stroke="{C_LINE}" stroke-width="1"/>')

    # 两条父轨迹
    parts.append(_polyline(p1_pts, tx, stroke=C_ACC, stroke_width="2.2",
                           opacity=".85"))
    parts.append(_polyline(p2.knots, tx, stroke=C_OK, stroke_width="2.2",
                           opacity=".85"))

    # 混合后的最终动作：违规段加粗
    parts.append(_polyline(child, tx, stroke=C_BAD, stroke_width="2.6",
                           opacity=".55"))
    worst_k, worst_m = min(enumerate(m_c), key=lambda kv: kv[1])
    for k, m in enumerate(m_c):
        if m < 0.0:
            parts.append(_polyline(child_segments[k], tx, stroke=C_BAD,
                                   stroke_width="5", stroke_linecap="round"))
    p0, p1s = child_segments[worst_k]
    wp = _closest_point_on_segment(p0, p1s, obs.center)
    wx, wy = tx(wp)

    # 最深点标记
    parts.append(
        f'<circle cx="{wx:.1f}" cy="{wy:.1f}" r="4.5" fill="#fff" '
        f'stroke="{C_BAD}" stroke-width="2"/>')
    parts.append(
        f'<line x1="{wx:.1f}" y1="{wy:.1f}" x2="{wx + 66:.1f}" '
        f'y2="{wy - 52:.1f}" stroke="{C_BAD}" stroke-width="1" opacity=".7"/>')
    parts.append(
        f'<text x="{wx + 70:.1f}" y="{wy - 56:.1f}" font-size="12" '
        f'fill="{C_BAD}" font-weight="600">第 {first_bad} 段起穿障</text>')
    parts.append(
        f'<text x="{wx + 70:.1f}" y="{wy - 41:.1f}" font-size="12" '
        f'fill="{C_BAD}">最小余量 {mm(worst_m)} mm</text>')

    # 起点 / 终点
    for pt, label, anchor in ((p1_pts[0], "起点", "start"),
                              (p1_pts[-1], "终点", "end")):
        px_, py_ = tx(pt)
        parts.append(f'<circle cx="{px_:.1f}" cy="{py_:.1f}" r="3.5" '
                     f'fill="{C_FG}"/>')
        parts.append(f'<text x="{px_ + (8 if anchor == "start" else -8):.1f}" '
                     f'y="{py_ - 9:.1f}" font-size="11.5" fill="{C_MUT}" '
                     f'text-anchor="{anchor}">{label}</text>')

    # 坐标刻度
    for xv in (0.0, 0.2, 0.4, 0.6, 0.8):
        px_, py_ = tx((xv, scene.ws_lo[1], 0))
        parts.append(f'<text x="{px_:.1f}" y="{py_ + 16:.1f}" font-size="10.5" '
                     f'fill="{C_MUT}" text-anchor="middle">{xv:g}</text>')
    for yv in (-0.4, -0.2, 0.0, 0.2, 0.4):
        px_, py_ = tx((scene.ws_lo[0], yv, 0))
        parts.append(f'<text x="{px_ - 7:.1f}" y="{py_ + 3.5:.1f}" font-size="10.5" '
                     f'fill="{C_MUT}" text-anchor="end">{yv:g}</text>')
    parts.append(f'<text x="{PX + SZ / 2:.1f}" y="{PY + SZ + 34:.1f}" font-size="11" '
                 f'fill="{C_MUT}" text-anchor="middle">x（米）</text>')
    parts.append(f'<text x="{PX - 40:.1f}" y="{PY + SZ / 2:.1f}" font-size="11" '
                 f'fill="{C_MUT}" text-anchor="middle" '
                 f'transform="rotate(-90 {PX - 40:.1f} {PY + SZ / 2:.1f})">y（米）</text>')

    # 图例
    ly = PY + SZ + 40
    for dx, color, text, dash in (
        (0, C_ACC, "父轨迹 P1（完整检查通过）", ""),
        (250, C_OK, "父轨迹 P2（完整检查通过）", ""),
        (500, C_BAD, "0.5·P1 + 0.5·P2（最终动作，穿障）", ""),
    ):
        parts.append(f'<line x1="{PX + dx:.1f}" y1="{ly:.1f}" x2="{PX + dx + 26:.1f}" '
                     f'y2="{ly:.1f}" stroke="{color}" stroke-width="3" {dash}/>')
        parts.append(f'<text x="{PX + dx + 33:.1f}" y="{ly + 4:.1f}" font-size="11.5" '
                     f'fill="{C_MUT}">{text}</text>')
    parts.append(
        f'<text x="{PX:.1f}" y="{ly + 22:.1f}" font-size="11.5" fill="{C_MUT}">'
        f'虚线圈 = 禁区半径 {clearance * 1000:.0f} mm'
        f'（球障碍 {obs.radius * 1000:.0f} + 工具半径 {scene.tool_radius * 1000:.0f}'
        f' + 跟踪预留 {scene.tracking_reserve * 1000:.0f}）</text>')
    parts.append(
        f'<text x="{PX:.1f}" y="{ly + 44:.1f}" font-size="11.5" fill="{C_MUT}">'
        f'顶视 xy 投影，z 恒为 {p1.knots[0][2]:.2f} m；'
        f'轨迹是构造的数值样例，不是 VLA 输出、不是真机记录</text>')

    # ---------------------------------------------------------------- 右侧面板
    g = geo
    panel = []
    panel.append(
        f'<text x="{PANEL_X}" y="46" font-size="17" font-weight="700" '
        f'fill="{C_FG}">第一幕 · 变换让旧结论失效</text>')
    panel.append(
        f'<text x="{PANEL_X}" y="68" font-size="12" fill="{C_MUT}">'
        f'同一场景、同一父轨迹、同一批子轨迹，三条判定路径各跑一遍</text>')

    col_path, col_release, col_checks = PANEL_X, 812.0, 1008.0
    rows = [
        ("a 只验父轨迹就放行", g["path_a_wrong_release"], "500", 0, C_BAD),
        ("b 最终全检（基线）", g["path_b_wrong_release"], "500",
         g["path_b_full_checks"], C_OK),
        ("c Δ-Cert + 必要全检", g["path_c_wrong_release"], "500",
         g["path_c_full_checks"], C_OK),
    ]
    y = 112.0
    panel.append(
        f'<text x="{col_path}" y="{y}" font-size="11.5" fill="{C_MUT}" '
        f'font-weight="600">判定路径</text>')
    panel.append(
        f'<text x="{col_release}" y="{y}" font-size="11.5" fill="{C_MUT}" '
        f'font-weight="600" text-anchor="end">错误放行的违规轨迹</text>')
    panel.append(
        f'<text x="{col_checks}" y="{y}" font-size="11.5" fill="{C_MUT}" '
        f'font-weight="600" text-anchor="end">完整检查调用</text>')
    panel.append(f'<line x1="{col_path}" y1="{y + 8:.1f}" x2="{col_checks}" '
                 f'y2="{y + 8:.1f}" stroke="{C_LINE}" stroke-width="1"/>')
    y += 30
    for name, wrong, total, checks, color in rows:
        panel.append(f'<text x="{col_path}" y="{y}" font-size="13" fill="{C_FG}">'
                     f'{name}</text>')
        panel.append(
            f'<text x="{col_release}" y="{y}" font-size="13.5" font-weight="700" '
            f'fill="{color}" text-anchor="end">{wrong} / {total}</text>')
        panel.append(
            f'<text x="{col_checks}" y="{y}" font-size="13.5" fill="{C_FG}" '
            f'text-anchor="end">{checks}</text>')
        y += 17
        panel.append(f'<line x1="{col_path}" y1="{y - 6:.1f}" x2="{col_checks}" '
                     f'y2="{y - 6:.1f}" stroke="{C_LINE}" stroke-width=".6" '
                     f'opacity=".7"/>')
        y += 15

    y += 14
    for text, color in (
        (f"路径 a 那一列就是本项目要解决的问题："
         f"{g['path_a_wrong_release']} 条真正违规的轨迹被放行。", C_BAD),
        (f"安全子轨迹通过 {g['safe_children_passed']} / "
         f"{g['safe_children_total']} 条，误拒 {g['path_c_false_reject']} 条"
         f"（负对照：这套机制不是「一变就拒」）。", C_MUT),
        (f"直接继承 {g['path_c_inherited']} 次；独立实现交叉验证分歧 "
         f"{g['cross_check_disagreements']} 处。", C_MUT),
        (f"完整检查调用减少 {g.get('full_check_reduction_pct', 0):.1f}% —— "
         f"指验证阶段的调用次数，不是整机提速。", C_MUT),
    ):
        panel.append(f'<text x="{col_path}" y="{y}" font-size="12" fill="{color}">'
                     f'{text}</text>')
        y += 21

    # 案例说明框
    y += 16
    box_h = 152.0
    box_top = y - 20.0
    panel.append(
        f'<rect x="{PANEL_X - 14:.1f}" y="{box_top:.1f}" '
        f'width="{col_checks - PANEL_X + 14:.1f}" height="{box_h:.1f}" rx="4" '
        f'fill="#f7f6f3" stroke="{C_LINE}" stroke-width="1"/>')
    panel.append(
        f'<text x="{PANEL_X}" y="{y}" font-size="12.5" font-weight="700" '
        f'fill="{C_FG}">图中这一组（案例 #0）为什么成立</text>')
    y += 24
    for text in (
        f"P1 最小余量 {mm(min(m_p1))} mm，P2 {mm(min(m_p2))} mm"
        f" —— 两条父轨迹各自完整检查都通过。",
        f"按 0.5 / 0.5 混合后：最小余量 {mm(min(m_c))} mm，"
        f"第 {first_bad} 段起穿过禁区。",
        f"父轨迹的结论对混合结果不成立 —— 这就是本项目要解决的问题。",
        f"Δ-Cert 界不够（父余量扣不掉逐段偏差），回退完整检查 → "
        f"判定 {sample['verdict']}。",
        f"「界不够」是「无法证明」，不是「一定会碰撞」—— 所以必须真的再查一遍。",
    ):
        panel.append(f'<text x="{PANEL_X}" y="{y}" font-size="12" fill="{C_FG}">'
                     f'{text}</text>')
        y += 23

    y = box_top + box_h + 24.0
    panel.append(
        f'<text x="{col_path}" y="{y}" font-size="11.5" fill="{C_MUT}">'
        f'复现：python -m sentinel_evc demo --cases {cases} --out runs/demo_a</text>')
    panel.append(
        f'<text x="{col_path}" y="{y + 18:.1f}" font-size="11.5" fill="{C_MUT}">'
        f'本图由同一次运行的输出生成；脚本先跟基线断言，不一致就拒绝出图。</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="第一幕：两条各自验证通过的父轨迹，混合之后穿过障碍">
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  <g font-family="{FONT}">
    {chr(10).join("    " + p for p in parts)}
    {chr(10).join("    " + p for p in panel)}
  </g>
</svg>
"""
    return svg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成第一幕（Demo A）对照图")
    ap.add_argument("--out", default="docs/demo_a.svg")
    ap.add_argument("--cases", type=int, default=1000)
    args = ap.parse_args(argv)

    geo = run_act_one(args.cases)
    svg = build_svg(geo, args.cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 显式写 LF：这个文件要进版本库，行尾必须与平台无关，
    # 否则在 Windows 上生成一次就产生一次无意义的 diff。
    out.write_text(svg, encoding="utf-8", newline="\n")

    print(f"第一幕 {geo['cases']} 个案例：路径 a 错误放行 "
          f"{geo['path_a_wrong_release']} / {geo['path_a_total_violating']}，"
          f"路径 b 全检 {geo['path_b_full_checks']} 次，"
          f"路径 c 全检 {geo['path_c_full_checks']} 次，"
          f"误拒 {geo['path_c_false_reject']}，"
          f"交叉验证分歧 {geo['cross_check_disagreements']}")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
