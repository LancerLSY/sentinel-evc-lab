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


def _use_utf8_output() -> None:
    """把标准输出/错误固定成 UTF-8。

    Windows 上输出被管道或重定向接走时，Python 会用本地编码（简体中文机器上是 GBK），
    而第二幕要打印 ✓ / ✗ —— GBK 里没有这两个字符，`print` 会直接抛
    UnicodeEncodeError，把 demo 打断在第二幕，退出码 1。
    CI 跑在 UTF-8 环境，永远抓不到这一类问题（同类的坑见仓库根 .gitattributes 的注释）。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


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
    ok, msg = verify_bundle(args.bundle, args.public_key, args.run_id,args.expected_tip)
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


def cmd_serve(args):
    from .server import serve
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be 0–65535")
    serve(args.data_dir,args.port)
    return 0


def cmd_native_import(args) -> int:
    from .native_runs import MAX_ARCHIVE, NativeRunStore
    from .runstore import RunStore
    archive, public_key = Path(args.archive), Path(args.public_key)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError('需要不超过 256 MiB 的本地 ZIP 证据包。')
    if public_key.is_symlink() or not public_key.is_file() or public_key.stat().st_size != 32:
        raise ValueError('需要独立选择的 32 字节 Ed25519 原始公钥。')
    # Follow the same workspace ownership contract as the numeric service.
    workspace = RunStore(args.data_dir)
    try:
        store = NativeRunStore(workspace.root / 'native-runs')
        record = store.import_archive(archive.read_bytes(), args.run_id, public_key.read_bytes())
    finally:
        workspace.close()
    print(json.dumps({'id':record['id'],'verification':record['verification'],
                      'episode_count':record['episode_count'], 'open':'sentinel-evc serve --data-dir '+str(args.data_dir)},ensure_ascii=False,indent=2))
    return 0


def cmd_native_run(args) -> int:
    import subprocess
    source = Path(args.source).resolve() if args.source else Path(__file__).resolve().parents[2]
    runner = source / 'experiments' / 'vla' / 'run_sentinel_libero.py'
    if not runner.is_file():
        raise ValueError('此安装没有 VLA runner；请用 --source 指向仓库检出目录。')
    configuration = Path(args.config).resolve()
    if not configuration.is_file():
        raise ValueError('需要本地 VLA 配置 JSON。')
    return subprocess.run([args.python, str(runner), '--config', str(configuration),
                           '--output-dir', str(Path(args.out).resolve())], check=False).returncode


def cmd_run(args):
    from .scenario import Scenario, strict_json
    from .runstore import RunStore
    from .product_pipeline import ProductManager
    scenario=Scenario.parse(strict_json(Path(args.scenario).read_bytes())) if args.scenario else Scenario.parse({"name":"CLI 数值运行","seed":args.seed,"prediction_mode":args.mode,"risk_limit":args.risk_limit})
    store=RunStore(args.out)
    manager=ProductManager(store,realtime=not args.fast)
    try:
        record=manager.start(scenario)
        session=manager._sessions[record['id']]
        session.thread.join()
        result=manager.read(record['id'])
        print(json.dumps({"id":result['id'],"status":result['status'],"selected":result['selected'],"cursors":result['result']['cursors'],"error":result['error'],"verification":result.get('verification')},ensure_ascii=False,indent=2))
        return 0 if result['status']=="completed" else 3
    finally:
        manager.close();store.close()


def cmd_baseline(args):
    from .baseline import run_baseline
    print(json.dumps(run_baseline(args.out,args.seed,args.mode,args.risk_limit),ensure_ascii=False,indent=2))
    return 0


def cmd_experiments(args):
    from .experiments import experiment_registry
    records=[e.summary() for e in experiment_registry()]
    if args.experiment_id:
        records=[e for e in records if e['id']==args.experiment_id]
        if not records:
            print("unknown experiment",file=sys.stderr);return 2
        print(json.dumps({"status":"PREREQUISITES_REQUIRED" if records[0]["status"] == "pending" else records[0]["status"],"experiment":records[0]},ensure_ascii=False,indent=2))
        return 3
    print(json.dumps({"experiments":records},ensure_ascii=False,indent=2))
    return 0


def cmd_physics(args):
    from .physics_experiment import run_physics_experiment
    result = run_physics_experiment(args.out, seed=args.seed, friction=args.friction, render=args.render)
    print(json.dumps({"scope": result["scope"], "acceptance": result["acceptance"],
                      "integrated_outcome": result["integrated"]["outcome"],
                      "out": args.out}, ensure_ascii=False, indent=2))
    return 0 if result["infrastructure_gates_pass"] else 3


def cmd_remote_physics(args):
    from .ssh_experiment import run_remote_physics
    result = run_remote_physics(args.host, args.out, seed=args.seed,
                                friction=args.friction, render=args.render,
                                python=args.python, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["infrastructure_gates_pass"] else 3


def cmd_verify_physics(args):
    from .ssh_experiment import verify_physics_experiment_result
    result = verify_physics_experiment_result(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_install(args):
    from dataclasses import asdict
    from .installation import install_system
    source = args.source or str(Path(__file__).resolve().parents[2])
    if args.interactive or args.target is None:
        from .install_wizard import run_wizard
        if not sys.stdin.isatty():
            raise ValueError("交互安装需要终端；无人值守安装请指定 --target。")
        return run_wizard(source=source, target=args.target, profile=args.profile,
                          python_executable=args.python, no_app=args.no_app)
    result = install_system(args.target, profile=args.profile, source=source,
                            python_executable=args.python, build_app=(not args.no_app and sys.platform == "darwin"))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_build_app(args):
    from .installation import build_macos_app
    root = Path(__file__).resolve().parents[2]
    path = build_macos_app(args.out, args.python, args.data_dir, source_dir=root if (root / "pyproject.toml").is_file() else None)
    print(path)
    return 0


def cmd_app(args):
    from .server import serve
    if args.browser or sys.platform != "darwin":
        serve(args.data_dir, args.port, open_browser=True)
        return 0
    import subprocess
    from .installation import build_macos_app
    import hashlib
    data_dir = Path(args.data_dir).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    source_dir = root if (root / "pyproject.toml").is_file() else None
    config = {"schema_version": "native-app-v1",
              "python": str(Path(sys.executable).expanduser().absolute()),
              "data_dir": str(data_dir),
              "source_dir": str(source_dir) if source_dir else "",
              "bind": "127.0.0.1"}
    native = Path(__file__).resolve().parent / "native/SentinelApp.swift"
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + native.read_bytes()).hexdigest()[:20]
    app = data_dir.parent / ".sentinel-apps" / ("Sentinel EVC-" + identity + ".app")
    if not app.exists():
        build_macos_app(app, sys.executable, data_dir, source_dir=source_dir)
    saved = json.loads((app / "Contents/Resources/app-config.json").read_text(encoding="utf-8"))
    if saved != config:
        raise ValueError("桌面App配置与当前工作区不一致，请使用新的构建目录。")
    subprocess.run(["open", "-n", str(app)], check=True)
    print("已打开 Sentinel EVC 桌面App。")
    return 0


def cmd_models(args):
    from .assets import AssetStore
    store = AssetStore(Path(args.data_dir)/"models")
    if args.model_action == "import":
        path = Path(args.file)
        value = store.import_asset(args.name or path.stem, args.format or path.suffix.lstrip("."), content=path.read_bytes())
    elif args.model_action == "list":
        value = {"assets": store.list()}
    elif args.model_action == "check":
        value = store.check(args.id)
    else:
        value = store.geometry(args.id)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def cmd_robot(args):
    from .robot_connectors import RobotRegistry
    registry = RobotRegistry(Path(args.data_dir)/"robots")
    if args.robot_action == "add":
        value = registry.create(args.name, args.driver, args.host, args.port)
    elif args.robot_action == "list":
        value = {"profiles": registry.list()}
    else:
        value = registry.diagnose(args.id)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    _use_utf8_output()
    ap = argparse.ArgumentParser(
        prog="sentinel_evc", description="Sentinel EVC · CLI、桌面App与机器人实验入口")
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
    v.add_argument("--expected-tip",default=None,help="独立保留的末尾锚点")
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser("tamper", help="四种篡改测试")
    t.add_argument("--out", required=True)
    t.add_argument("--run-id", default=None)
    t.set_defaults(func=cmd_tamper)

    ui=sub.add_parser("serve",help="本地单用户数值操作台")
    ui.add_argument("--port",type=int,default=8765)
    ui.add_argument("--data-dir",default="runs/workbench")
    ui.set_defaults(func=cmd_serve)

    native_run=sub.add_parser('native-run',help='使用已有 VLA 环境运行原生策略和执行授权门禁')
    native_run.add_argument('--config',required=True)
    native_run.add_argument('--out',required=True)
    native_run.add_argument('--python',default=sys.executable,help='已有 LeRobot/LIBERO 环境的 Python')
    native_run.add_argument('--source',default=None,help='包含 experiments/vla 的仓库检出目录')
    native_run.set_defaults(func=cmd_native_run)

    native_import=sub.add_parser('native-import',help='核验并导入原生 VLA ZIP 记录到 App/工作台')
    native_import.add_argument('--archive',required=True)
    native_import.add_argument('--public-key',required=True)
    native_import.add_argument('--run-id',required=True)
    native_import.add_argument('--data-dir',default='runs/workbench')
    native_import.set_defaults(func=cmd_native_import)

    run=sub.add_parser("run",help="执行四候选数值闭环并保存签名证据")
    run.add_argument("--out",default="runs/workbench")
    run.add_argument("--scenario",default=None)
    run.add_argument("--seed",type=int,default=7)
    run.add_argument("--mode",choices=("physical","residual"),default="physical")
    run.add_argument("--risk-limit",type=float,default=.12)
    run.add_argument("--fast",action="store_true",help="仅用于数值验证的逻辑时钟模式")
    run.set_defaults(func=cmd_run)

    baseline=sub.add_parser("train-baseline",help="小规模根分组训练、校准和留出评价")
    baseline.add_argument("--out",required=True)
    baseline.add_argument("--seed",type=int,default=10_000_000)
    baseline.add_argument("--mode",choices=("physical","residual"),default="residual")
    baseline.add_argument("--risk-limit",type=float,default=.12)
    baseline.set_defaults(func=cmd_baseline)

    experiments=sub.add_parser("experiments",help="列出尚需模型、设备或数据的实验")
    experiments.add_argument("--experiment-id",default=None)
    experiments.set_defaults(func=cmd_experiments)

    physics = sub.add_parser("physics", help="真实 MuJoCo 三维托盘接触实验（可选依赖）")
    physics.add_argument("--out", required=True)
    physics.add_argument("--seed", type=int, default=7)
    physics.add_argument("--friction", type=float, default=.35)
    physics.add_argument("--render", action="store_true")
    physics.set_defaults(func=cmd_physics)

    remote = sub.add_parser("remote-physics", help="通过严格 OpenSSH 执行隔离的三维物理作业")
    remote.add_argument("--host", required=True, help="现有 OpenSSH 主机别名")
    remote.add_argument("--out", required=True)
    remote.add_argument("--python", default="python3", help="远端 Python 3.10+ 可执行文件名")
    remote.add_argument("--timeout", type=int, default=7200)
    remote.add_argument("--seed", type=int, default=7)
    remote.add_argument("--friction", type=float, default=.35)
    remote.add_argument("--render", action="store_true", help="远端 EGL 渲染实际轨迹")
    remote.set_defaults(func=cmd_remote_physics)

    check_physics = sub.add_parser("verify-physics", help="校验 SSH 回传收据、签名索引与全部物理试验")
    check_physics.add_argument("--out", required=True)
    check_physics.set_defaults(func=cmd_verify_physics)

    install = sub.add_parser("install", help="一键安装独立CLI环境与桌面App")
    install.add_argument("--target", default=None)
    install.add_argument("--interactive", action="store_true", help="打开终端安装向导；不传target时自动打开")
    install.add_argument("--profile", choices=("core","all"), default="all")
    install.add_argument("--source", default=None)
    install.add_argument("--python", default=sys.executable)
    install.add_argument("--no-app", action="store_true")
    install.set_defaults(func=cmd_install)

    build_app = sub.add_parser("build-app", help="构建原生macOS桌面入口（使用已有Python环境）")
    build_app.add_argument("--out", required=True)
    build_app.add_argument("--python", default=sys.executable)
    build_app.add_argument("--data-dir", default="runs/app")
    build_app.set_defaults(func=cmd_build_app)

    app = sub.add_parser("app", help="打开桌面App或浏览器操作台")
    app.add_argument("--data-dir", default="runs/app")
    app.add_argument("--browser", action="store_true")
    app.add_argument("--port", type=int, default=0)
    app.set_defaults(func=cmd_app)

    models = sub.add_parser("models", help="导入、预览与检查3D模型")
    models.add_argument("--data-dir", default="runs/app")
    model_actions = models.add_subparsers(dest="model_action", required=True)
    model_import = model_actions.add_parser("import")
    model_import.add_argument("--file", required=True)
    model_import.add_argument("--format", choices=("obj","stl","mjcf","urdf"))
    model_import.add_argument("--name", default=None)
    model_actions.add_parser("list")
    for action in ("check","geometry"):
        parser = model_actions.add_parser(action)
        parser.add_argument("--id", required=True)
    models.set_defaults(func=cmd_models)

    robot = sub.add_parser("robot", help="配置机械臂驱动入口与只读诊断")
    robot.add_argument("--data-dir", default="runs/app")
    robot_actions = robot.add_subparsers(dest="robot_action", required=True)
    robot_add = robot_actions.add_parser("add")
    robot_add.add_argument("--name", required=True)
    robot_add.add_argument("--driver", choices=("mock","ur_dashboard_readonly"), required=True)
    robot_add.add_argument("--host", default=None)
    robot_add.add_argument("--port", type=int, default=None)
    robot_actions.add_parser("list")
    robot_diagnose = robot_actions.add_parser("diagnose")
    robot_diagnose.add_argument("--id", required=True)
    robot.set_defaults(func=cmd_robot)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError,OSError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
