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
  aria-label="案例 {html.escape(str(sample['case']))} 的父轨迹与子轨迹">
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="var(--bad)"
    opacity=".18"/>
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none"
    stroke="var(--bad)" stroke-width="1.5" opacity=".6"/>
  <path d="{path(sample['parent_knots'])}" fill="none" stroke="var(--acc)"
    stroke-width="2" stroke-dasharray="5 3" opacity=".75"/>
  <path d="{path(sample['child_knots'])}" fill="none" stroke="{child_color}"
    stroke-width="2.5"/>
  <text x="{pad}" y="18" fill="var(--mut)" font-size="11">
    虚线 = 父轨迹　实线 = 变换后的最终动作　圆 = 障碍</text>
</svg>"""


def _text(value) -> str:
    return html.escape("未记录" if value is None else str(value))


def _cursors(values: dict) -> str:
    return " / ".join(_text(values.get(key)) for key in ("submitted", "accepted", "observed"))


def render(run_id: str, geo: dict, faults: dict, bundle: dict, verify_msg: str,
           *, events=()) -> str:
    rows = []
    for path, name in (("a", "只验父轨迹就放行"), ("b", "最终全检（基线）"),
                       ("c", "Δ-Cert + 必要全检")):
        wrong = geo.get(f"path_{path}_wrong_release")
        cls = "warn" if wrong is None else ("bad" if wrong else "ok")
        rows.append(f"<tr><td>{path} · {name}</td><td class='num {cls}'>{_text(wrong)}</td>"
                    f"<td class='num'>{_text(geo.get(f'path_{path}_full_checks'))}</td></tr>")

    metrics = (
        ("cases", "构造案例数"), ("path_a_total_violating", "实际违规子轨迹数"),
        ("safe_children_passed", "安全子轨迹通过数"), ("safe_children_total", "安全子轨迹总数"),
        ("path_c_false_reject", "路径 c 误拒数"), ("path_c_inherited", "直接继承次数"),
        ("cross_check_disagreements", "独立对照分歧数"),
        ("root_full_checks", "父证书建立完整检查次数"),
        ("path_c_full_checks", "继承失败回退完整检查次数"),
        ("end_to_end_wall_seconds", "端到端墙钟时间（秒）"),
    )
    stats = "".join(f"<tr><td>{label}</td><td class='num'>{_text(geo.get(key))}</td></tr>"
                    for key, label in metrics)
    svgs = "".join(
        f"<p>案例 {_text(s['case'])} · 变换类型 <code>{_text(s['kind'])}</code> · "
        f"实际{'合规' if s['truly_ok'] else '违规'} · 判定 {_text(s['verdict'])} · "
        f"最小余量 {s['min_margin']:.4f} m</p>" + _traj_svg(s)
        for s in geo.get("samples", [])[:4]
    )
    frows = "".join(
        f"<tr><td><code>{_text(d['fault'])}</code></td>"
        f"<td class='{'ok' if d.get('blocked') else 'bad'}'>"
        f"{'未记录' if d.get('blocked') is None else ('已阻断' if d['blocked'] else '未阻断')}</td>"
        f"<td><code>{_text(d.get('code'))}</code></td>"
        f"<td class='num'>{_text(d.get('stale_submissions_after_revoke'))}</td>"
        f"<td class='num'>{_cursors(d.get('cursors', {}))}</td></tr>"
        for d in faults.get("details", [])
    )
    erows = "".join(
        f"<tr><td>{_text(event['seq'])}</td><td>{_text(event['ts_mono_ns'])}</td>"
        f"<td>{_text(event['type'])}</td>"
        f"<td><code>{_text(json.dumps(event['payload'], ensure_ascii=False))}</code></td>"
        f"<td>{_cursors(event['payload']) if event['type'] == 'OUTCOME' else '—'}</td></tr>"
        for event in events
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel EVC Lab · {_text(run_id)}</title>
<style>{CSS}</style></head><body>
<h1>Sentinel EVC Lab 运行报告</h1>
<p class="sub">run_id <code>{_text(run_id)}</code> · 数值参考域 · 按输入记录展示</p>
<div class="note"><strong>结论边界：</strong>构造数值场景不代表真实 VLA、机械臂或视觉输入。
「许可」「证书」「验证」不是功能安全认证、不是物理停止证明、不是事故责任判断。
零次观测到失效也不等于任意场景零事故。dt、前缀长度和 tracking reserve 仅适用于本 Demo。</div>
<h2>第一幕 · 变换让旧结论失效</h2>
<p>下列统计仅适用于输入所述构造案例及约束族；未提供的数据标记为“未记录”。</p>
<div class="wrap"><table><tr><th>判定路径</th><th>错误放行的违规轨迹</th>
<th>完整检查调用次数</th></tr>{''.join(rows)}</table>
<table><tr><th>统计项</th><th>记录值</th></tr>{stats}</table></div>
<div class="note">完整检查调用次数不是整机提速。评估收益还需父证书建立成本、
失败回退成本和端到端墙钟时间；本页不从调用次数推断性能收益。
继承失败回退成本以上方完整检查次数记录；墙钟时间未提供时显示“未记录”。</div>
{svgs}
<h2>第二幕 · 撤销与三个游标</h2>
<div class="wrap"><table><tr><th>注入的故障</th><th>结果</th><th>错误码</th>
<th>撤销后新增旧代次提交</th><th>submitted / accepted / observed</th></tr>{frows}</table></div>
<p>记录的阻断数 {_text(faults.get('blocked'))} / {_text(faults.get('faults_run'))}；
撤销后旧代次新增提交合计 {_text(faults.get('stale_gen_submissions_after_revoke'))}。</p>
<div class="note">三个游标分别表示提交、控制器接受和观测；不能相互代替。
撤销之前已接受的命令仍可能随后 observed。软件队列清空、控制器确认取消、
实际运动停止是三件不同的事；模拟 observed 不是电机制动证明。</div>
<h2>第三幕 · 独立证据校验</h2>
<p>事件条数 {_text(bundle.get('event_count'))}</p>
<p>末尾摘要 <code>{_text(bundle.get('tip_hash'))}</code></p>
<p>校验器返回：{_text(verify_msg)}</p>
<div class="note">签名只证明相对于指定公钥的记录完整性，不证明传感器诚实、
不证明动作在物理上发生过，也不判断责任。bundle 外公钥为演示用，非客户 PKI。</div>
<h2>事件时间线</h2>
<div class="wrap"><table><tr><th>seq</th><th>单调时间 ns</th><th>类型</th>
<th>原始 payload</th><th>submitted / accepted / observed</th></tr>{erows}</table></div>
<footer>Sentinel EVC Lab · 离线报告只呈现输入记录，不重新计算几何或协议判定。</footer>
</body></html>"""


def write_report(path: str, **kwargs) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(**kwargs), encoding="utf-8")
    return str(p)
