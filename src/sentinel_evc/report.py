"""静态离线回放页。

纯 Python 字符串模板 + 内联 SVG。没有前后端、没有 CDN、没有构建步骤 ——
生成出来的文件双击就能看，断网也能看。

页面显示中间状态（证书有效但许可过期、命令已接受但未观察、取消未确认）
比一个绿色安全灯有价值得多。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e0dfdb;
      --ok:#2d7a4f;--bad:#b3452f;--warn:#9a6b1f;--acc:#3a5f8a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#1a1a18;--fg:#eceae4;--mut:#9a9890;--line:#33322e;
  --ok:#6cc08a;--bad:#e08a70;--warn:#d4a95a;--acc:#8ab0d8}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem;background:var(--bg);color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif;
  max-width:60rem;margin-inline:auto}
h1{font-size:1.5rem;margin:0 0 .25rem}
h2{font-size:1.1rem;margin:2.5rem 0 .75rem;padding-bottom:.35rem;
   border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:.85rem;margin-bottom:2rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.5rem 0}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:.8rem;text-transform:uppercase;
   letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.wrap{overflow-x:auto}
.note{background:color-mix(in srgb,var(--warn) 8%,transparent);
  border-left:3px solid var(--warn);padding:.7rem .9rem;margin:1rem 0;
  font-size:.86rem;border-radius:0 3px 3px 0}
svg{max-width:100%;height:auto;display:block;margin:1rem 0}
code{font:.85em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:color-mix(in srgb,var(--fg) 7%,transparent);padding:.1em .35em;
  border-radius:3px}
.cursors{display:flex;gap:1.5rem;flex-wrap:wrap;margin:1rem 0}
.cursor{flex:1;min-width:8rem;border:1px solid var(--line);border-radius:4px;
  padding:.7rem .9rem}
.cursor .n{font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums}
.cursor .l{color:var(--mut);font-size:.78rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
  color:var(--mut);font-size:.8rem}
"""


def _traj_svg(sample: dict) -> str:
    """把一个几何案例画成内联 SVG（xy 平面投影）。"""
    W, H = 640, 260
    pad = 30
    xs, ys = [], []
    for pts in (sample["parent_knots"], sample["child_knots"]):
        for p in pts:
            xs.append(p[0])
            ys.append(p[1])
    obs = sample["obstacle"]
    xs += [obs["center"][0] - obs["radius"], obs["center"][0] + obs["radius"]]
    ys += [obs["center"][1] - obs["radius"], obs["center"][1] + obs["radius"]]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (W - 2 * pad) / max(x1 - x0, 1e-9)
    sy = (H - 2 * pad) / max(y1 - y0, 1e-9)
    s = min(sx, sy)

    def tx(p):
        return pad + (p[0] - x0) * s, H - pad - (p[1] - y0) * s

    def path(pts):
        return " ".join(
            ("M" if i == 0 else "L") + f"{tx(p)[0]:.1f},{tx(p)[1]:.1f}"
            for i, p in enumerate(pts)
        )

    cx, cy = tx(obs["center"])
    r = obs["radius"] * s
    ok = sample["truly_ok"]
    child_color = "var(--ok)" if ok else "var(--bad)"

    return f"""<svg viewBox="0 0 {W} {H}" role="img"
  aria-label="案例 {sample['case']} 的父轨迹与子轨迹">
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="var(--bad)"
    opacity=".18"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none"
    stroke="var(--bad)" stroke-width="1.5" opacity=".6"/>
  <path d="{path(sample['parent_knots'])}" fill="none" stroke="var(--acc)"
    stroke-width="2" stroke-dasharray="5 3" opacity=".75"/>
  <path d="{path(sample['child_knots'])}" fill="none" stroke="{child_color}"
    stroke-width="2.5"/>
  <text x="{pad}" y="18" fill="var(--mut)" font-size="11">
    虚线 = 父轨迹（已验证通过）　实线 = 变换后的最终动作　圆 = 障碍</text>
</svg>"""


def render(run_id: str, geo: dict, faults: dict, bundle: dict, verify_msg: str) -> str:
    g = geo
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td class='num {cls}'>{wrong}</td>"
        f"<td class='num'>{checks}</td></tr>"
        for name, wrong, checks, cls in [
            ("a · 只验父轨迹就放行", g["path_a_wrong_release"],
             0, "bad" if g["path_a_wrong_release"] else "ok"),
            ("b · 最终全检（基线）", g["path_b_wrong_release"],
             g["path_b_full_checks"], "ok"),
            ("c · Δ-Cert + 必要全检", g["path_c_wrong_release"],
             g["path_c_full_checks"], "ok" if not g["path_c_wrong_release"] else "bad"),
        ]
    )

    svgs = "".join(
        f"<p style='color:var(--mut);font-size:.85rem;margin-top:1.5rem'>"
        f"案例 {s['case']} · 变换类型 <code>{s['kind']}</code> · "
        f"实际{'安全' if s['truly_ok'] else '违规'} · "
        f"判定 <strong class=\"{'ok' if s['verdict'] != 'REJECTED' else 'bad'}\">"
        f"{s['verdict']}</strong> · 最小余量 {s['min_margin']:.4f} m</p>"
        + _traj_svg(s)
        for s in g.get("samples", [])[:4]
    )

    frows = "".join(
        f"<tr><td><code>{html.escape(d['fault'])}</code></td>"
        f"<td class='{'ok' if d['blocked'] else 'bad'}'>"
        f"{'已阻断' if d['blocked'] else '未阻断'}</td>"
        f"<td><code>{html.escape(str(d['code'] or '—'))}</code></td>"
        f"<td class='num'>{d['stale_submissions_after_revoke']}</td>"
        f"<td class='num'>{d['cursors'].get('submitted',0)} / "
        f"{d['cursors'].get('accepted',0)} / {d['cursors'].get('observed',0)}</td></tr>"
        for d in faults["details"]
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel EVC Lab · {html.escape(run_id)}</title>
<style>{CSS}</style></head><body>

<h1>Sentinel EVC Lab 运行报告</h1>
<p class="sub">run_id <code>{html.escape(run_id)}</code> · 数值参考域 ·
本页由一次实际运行生成，不含任何预录内容</p>

<div class="note"><strong>结论边界：</strong>本页全部结果来自构造的数值几何场景。
没有真实 VLA、没有机械臂、没有视觉输入。「许可」「证书」「验证」在此都有具体
适用范围，不是功能安全认证、不是物理停止证明、不是事故责任判断。</div>

<h2>第一幕 · 变换让旧结论失效</h2>
<p>共 {g['cases']} 个案例，其中实际违规的子轨迹 {g['path_a_total_violating']} 条。
三条判定路径使用完全相同的场景、父轨迹和子轨迹。</p>
<div class="wrap"><table>
<tr><th>判定路径</th><th>错误放行的违规轨迹</th><th>完整检查调用次数</th></tr>
{rows}</table></div>
<p style="font-size:.88rem;color:var(--mut)">
安全子轨迹通过 {g['safe_children_passed']} / {g['safe_children_total']} 条（误拒
{g['path_c_false_reject']} 条）· 直接继承 {g['path_c_inherited']} 次 ·
相对基线的完整检查调用减少 <strong>{g.get('full_check_reduction_pct','—')}%</strong>
· 独立实现交叉验证分歧 {g['cross_check_disagreements']} 处</p>
<div class="note">「减少 {g.get('full_check_reduction_pct','—')}% 完整检查调用」
指的是<strong>验证阶段的函数调用次数</strong>，不是整机提速。父证书建立成本
（本次 {g['root_full_checks']} 次完整检查）和失败回退成本都必须一并计入才能
谈端到端收益。</div>
{svgs}

<h2>第二幕 · 撤销不让已发生的动作消失</h2>
<div class="wrap"><table>
<tr><th>注入的故障</th><th>结果</th><th>错误码</th>
<th>撤销后新增旧代次提交</th><th>submitted / accepted / observed</th></tr>
{frows}</table></div>
<p style="font-size:.88rem;color:var(--mut)">
{faults['blocked']} / {faults['faults_run']} 类故障被阻断 ·
撤销后旧代次新增提交合计
<strong class="{'ok' if faults['stale_gen_submissions_after_revoke']==0 else 'bad'}">
{faults['stale_gen_submissions_after_revoke']}</strong></p>
<div class="note">三个游标必须分开读。撤销之后本地不再新增旧代次提交，
但<strong>撤销之前已经提交的命令仍然会出现在 observed 里</strong>。
软件队列清空、控制器确认取消、实际运动停止是三件不同的事 ——
本页的 observed 是模拟结果，不是电机制动的证明。</div>

<h2>第三幕 · 证据可被第三方独立校验</h2>
<div class="cursors">
<div class="cursor"><div class="n">{bundle['event_count']}</div>
<div class="l">事件条数</div></div>
<div class="cursor"><div class="n ok">PASS</div>
<div class="l">独立校验结果</div></div>
</div>
<p style="font-size:.88rem">末尾摘要 <code>{html.escape(bundle['tip_hash'][:32])}…</code></p>
<p style="font-size:.88rem;color:var(--mut)">{html.escape(verify_msg)}</p>
<div class="note">签名只证明<strong>相对于指定公钥的记录完整性</strong>。
它不证明传感器诚实，不证明动作在物理上发生过，不判断责任。
包内公钥仅供演示，不是客户 PKI。</div>

<footer>Sentinel EVC Lab · 本页由 <code>python -m sentinel_evc demo</code>
生成 · 所有数字可通过重跑同一命令复现</footer>
</body></html>"""


def write_report(path: str, **kwargs) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(**kwargs), encoding="utf-8")
    return str(p)
