# 结果与复现

本页只记录可定位到实际运行、代码提交和输出文件的结果。当前样例由提交
`d6cf66e992b34b395158f2aabd32b993aef07b6b` 生成，run_id 为 `sample_run`。

## 复现命令

```bash
python -m sentinel_evc demo --cases 1000 --seed 1234 --out runs/sample_run
python -m sentinel_evc verify \
  --bundle runs/sample_run/bundle \
  --public-key runs/sample_run/anchors/demo.public \
  --run-id sample_run
```

输出目录必须尚不存在。仓库中的 `sample_run/` 是上述运行通过后保存的同一份产物，
包含 `scenario.json`、`summary.json`、`report.html`、bundle 三文件和 bundle 外演示公钥。

## G0 · 来源

| 项 | 记录 |
| --- | --- |
| 生成代码 | `d6cf66e992b34b395158f2aabd32b993aef07b6b` |
| 输入 profile | `sample_run/scenario.json` |
| 汇总 | `sample_run/summary.json` |
| 事件清单 | `sample_run/bundle/manifest.json` |
| 平台 | Windows 11 `10.0.26200`，Python 3.13.5 |
| 依赖 | numpy 2.3.3，cryptography 48.0.0；测试 pytest 8.4.2 |
| 核心流水线墙钟时间 | 11.909279 秒，见 `geometry.end_to_end_wall_seconds`；不含最终 summary/report 写入 |
| 完整命令外部墙钟时间 | 12.171510 秒，PowerShell `Measure-Command` 单次本机记录 |

墙钟时间只有本机单次测量，不是性能基准。外部计时未进入签名 bundle；这里保留命令、
代码提交和读数，避免把核心流水线字段误写成整条命令耗时。

## G1 · 几何

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 构造案例 | 1000 | `geometry.cases` |
| 实际违规子轨迹 | 500 | `geometry.path_a_total_violating` |
| 路径 a / b / c 错误放行 | 500 / 0 / 0 | 对应 `path_*_wrong_release` |
| 安全子轨迹通过 | 500 / 500 | `safe_children_passed` / `safe_children_total` |
| 误拒 | 0 | `path_c_false_reject` |
| 独立对照分歧 | 0 | `cross_check_disagreements` |

独立对照对每段取 1001 个采样点，不复用生产侧解析几何公式。固定 1000 案例的
独立门禁命令为：

```bash
python -m pytest -q tests/test_gate_geometry.py
```

## G2 · Δ-Cert 与成本

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 两条父轨迹建立证书 | 2000 次完整检查 | `geometry.root_full_checks` |
| 路径 b 最终全检 | 1000 次 | `geometry.path_b_full_checks` |
| 路径 c 继承失败回退 | 500 次完整检查 | `geometry.path_c_full_checks` |
| 路径 c 直接继承 | 500 次 | `geometry.path_c_inherited` |
| 子轨迹完整检查调用减少 | 50.0% | `geometry.full_check_reduction_pct` |

50.0% 只比较路径 c 与路径 b 的**子轨迹完整检查调用次数**。它不包含 2000 次父证书
建立成本，也不等于整机提速。累计余量 `30−8−5=17mm`、六类依赖变化回退、深度 4
回退和继承失败后安全轨迹仍可由完整检查通过，均由 `tests/test_geometry_delta.py` 锁定。

## G3 · 执行与故障

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 注入故障类型 | 7 | `faults.faults_run` |
| 正确阻断 | 7 | `faults.blocked` |
| 撤销后旧代次新增提交 | 0 | `faults.stale_gen_submissions_after_revoke` |
| 撤销竞态中撤销前已提交、随后 observed | 3 | `faults.details[fault=revoke-race]` |
| cancel 未确认 | 阻断，`CANCEL_UNCONFIRMED` | `faults.details[fault=cancel-unconfirmed]` |

这些游标来自确定性数值控制器。它们不是物理停止或实机制动证明。

## G4 · 证据

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 事件条数 | 7049 | `sample_run/bundle/manifest.json` |
| 末尾摘要 | `sha256:40e6c3fba173750057f82c077c63e4fa9c3e684e61645ae7b238149ff3e23f19` | 同上 |
| `events.jsonl` 摘要 | `sha256:2f8d0b5a9a393bc0d8bec5631ef9fe20618c0a50ce0d5a2af955a522eaac0e14` | 同上 |
| 独立 verify | PASS，7049 条事件、7 层全过 | 上方 verify 命令 |

四类篡改门禁由下列测试实际重建副本并修改，不依赖生产写入方的协议实现：

```bash
python -m pytest -q tests/test_gate_evidence_cli.py \
  -k "verify_reports_distinct_first_failure or verify_accepts"
```

首个失败层分别为：中间事件 `hash_chain`、删尾三行 `event_count`、错误公钥
`signature`、错误 run_id `run_id`。

## 回归与发布边界

```bash
python -m pytest -q
python -m compileall -q src tests run_tests.py
```

本轮全量结果为 **397 passed**。G0–G4 已在本地通过；G5 尚未完成。子 agent、CI、
同一作者换环境或离线安装 smoke 都不能代替真实非项目成员按 README 复现并说明边界，
因此版本保持 candidate。

本仓库没有运行真实 VLA、MuJoCo、ROS 2、物理机器人、视觉模型训练、mTLS、
多用户权限、工业级持久化或跨进程隔离。
