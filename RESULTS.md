# 结果与复现

每个数字后面直接跟产生它的命令和输出文件路径。任何人质疑任何一个数字，
回应是「你自己跑一遍」，而不是一段解释。

本页全部数据来自 `sample_run/`，run_id 为 `sample-001`。

## 复现命令

```bash
python -m sentinel_evc demo --cases 1000 --out runs/repro --run-id repro
```

输出目录必须为空。随包的 `sample_run/` 请勿覆盖。

本页是手写的，`sample_run/` 是机器生成的。两者是否还对得上，由脚本核对，CI 里也会跑：

```bash
python tools/check_results.py
```

## 第一幕 · 几何判定

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 构造案例总数 | 1000 | `sample_run/summary.json` → `geometry.cases` |
| 实际违规的子轨迹 | 500 | `geometry.path_a_total_violating` |
| 路径 a 错误放行 | 500 | `geometry.path_a_wrong_release` |
| 路径 b 错误放行 | 0 | `geometry.path_b_wrong_release` |
| 路径 c 错误放行 | 0 | `geometry.path_c_wrong_release` |
| 路径 b 完整检查调用 | 1000 | `geometry.path_b_full_checks` |
| 路径 c 完整检查调用 | 500 | `geometry.path_c_full_checks` |
| 直接继承次数 | 500 | `geometry.path_c_inherited` |
| 安全子轨迹通过 | 500 / 500 | `geometry.safe_children_passed` |
| 误拒 | 0 | `geometry.path_c_false_reject` |
| 独立实现交叉验证分歧 | 0 | `geometry.cross_check_disagreements` |
| 父证书建立的完整检查 | 1000 | `geometry.root_full_checks` |

**完整检查调用减少 50.0%**，即 500 次
对 1000 次。

图中那一组（案例 #0）的数字，同样来自真实运行：

| 指标 | 值 | 来源 |
| --- | --- | --- |
| P1 最小余量 | +67.7 mm | `docs/demo_a.svg` · `python tools/make_demo_a_figure.py --out docs/demo_a.svg` |
| P2 最小余量 | +75.0 mm | 同上 |
| 0.5·P1 + 0.5·P2 最小余量 | −68.1 mm（第 6 段起穿障） | 同上 |

该脚本自己跑一遍第一幕，先与本节基线断言，不一致就报错退出 —— 配图里没有手填数字。

这个数字的适用范围：
- 它是**验证阶段的函数调用次数**，不是整机提速，不是端到端时延改善。
- 父证书建立成本（1000 次完整检查）不在分子里，谈端到端收益时必须加回去。
- 命中率取决于变换幅度分布。本次构造集是一半可继承、一半不可继承，属于人为设定的比例。
- 本仓库**没有**端到端耗时测量，因此不作任何提速主张。

## 第二幕 · 故障注入

| 指标 | 值 |
| --- | --- |
| 注入故障类型数 | 7 |
| 阻断的故障数 | 7 |
| **撤销后旧代次新增提交** | **0** |

逐类结果见 `sample_run/summary.json` → `faults.details`，或 `report.html`。

撤销/取消 ACK 均为数值模拟，**不是实机制动证明**。

## 第三幕 · 证据

| 指标 | 值 |
| --- | --- |
| 事件条数 | 2031 |
| 末尾摘要 | `sha256:872d9a5779ae246e6ca9aba5a2bf83c06…` |
| 独立校验 | PASS（7 层全过） |

```bash
python -m sentinel_evc verify \
  --bundle sample_run/bundle \
  --public-key sample_run/anchors/demo.public \
  --run-id sample-001

python -m sentinel_evc tamper --out sample_run --run-id sample-001
```

## 自动化测试

```bash
python -m pytest -q      # 或 python run_tests.py
```

本轮：**24 项通过**。

这个数字只属于本仓库。技术文档和历史参考包里的 57 / 50 / 12 / 65 / 51 项来自
不同代码库、不同时间，**任何形式的相加或并列都不成立**。

## 运行环境

| 项 | 值 |
| --- | --- |
| Python | 3.12.3 |
| 平台 | Linux x86_64 |

其它平台未测试。依赖范围不是兼容性证明。

## 本仓库没有做过的事

真实 VLA 推理、MuJoCo、ROS 2、物理机器人、视觉模型训练、mTLS、多用户权限、
工业级持久化、跨进程隔离、真机停止与后备验证。
