# Sentinel EVC Lab

[English](README.md) · **中文**

<p align="center">
  <img src="docs/media/sentinel-hero.svg" alt="Sentinel EVC 最终动作校验工作台" width="100%">
</p>

**[项目主页](https://lansiyao.com/research/sentinel-vla/project/)** · [GitHub](https://github.com/LancerLSY/sentinel-evc-lab) · [Hugging Face](https://huggingface.co/LancerLSY/sentinel-smolvla-so100)

Sentinel EVC 把授权门放在机器人策略动作块的最后提交点：校验**真正要执行**
的最终动作，签发有时限、一次性的执行许可，并留下可由第三方独立校验的证据。

本地执行工作台提供 CLI、macOS 原生 App、浏览器回放、SSH 实验入口、
受限三维模型检查和只读机械臂诊断，可与已有策略环境一起使用。
每个受支持的 profile 分别声明授权与校验契约。

> **v0.3 · 本地开发者预览版 · MIT**
>
> 产品运行时：Python 3.10+ · macOS App：由 Python 环境管理并在本机构建 · 服务仅监听 `127.0.0.1`

[安装](#五分钟安装) · [机制](#执行许可如何产生) ·
[实测结果](#实测结果) · [最新实验包下载](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/native-vla-product-20261003) ·
[研究展示导航](docs/research_showcase.md) · [CI](https://github.com/LancerLSY/sentinel-evc-lab/actions/workflows/ci.yml)

## 执行差异诊断与回归门

**定位第一次变化，直达它的证据。** App 和 CLI 可以比较两份分别选择公钥的原生记录，检查动作、反馈、执行游标及结果，并链接两侧源记录的三维帧。下载的报告可离线重算；`--fail-on` 指定回归判定字段。

完整 40 组记录中，5,420 次动作一致，产品找出第 4 步第二路相机的一次反馈差异。实际安装的 Rerun 0.38.1 CLI 对照定位到差异 Episode chunk；Sentinel 工作流进一步给出首个步骤、VLA 字段与两侧源帧引用。[实测对照、范围与复现](docs/research/2026-10-03/execution-diff/RESULTS.md)。


## 当前可用入口

| 入口 | 已可用 | 边界 |
|---|---|---|
| **CLI + 本地服务** | 安装、运行、校验、导出；数值与固定 MuJoCo profile | 单机、单工作区 |
| **macOS 原生 App** | 共用本地工作台；独立公钥核验导入原生运行、按任务三维回放、导出签名 ZIP | 依赖受管 Python 环境；本机构建，尚无分发签名与公证 |
| **SSH 实验** | 严格 OpenSSH 启动、源码绑定、返回证据校验 | 需要已有主机别名、密钥代理和 `known_hosts` |
| **机械臂入口** | mock 诊断与 Universal Robots 只读 dashboard 探测 | 尚无物理运动写入适配器 |
| **三维模型入口** | OBJ、STL、MJCF、URDF 检查与离线 Canvas 预览 | 预览不会获得执行授权 |
| **原生 VLA 接入** | 官方 SmolVLA/Panda 推理、精确 7-D 请求授权、一次性许可与真实环境反馈 | LIBERO 仿真；尚无 Panda 碰撞、动力学、WorldGuard 或物理停止验证 |
| **GPU / 模型研究** | WorldGuard 模型对照与固定 SO100 数据上的 SmolVLA 微调 | 各实验 profile 分别报告；SO100 overlay 与原生 Panda 检查点不同 |

工作台把“运行是否完成”“证据是否完整”“科学验收是否通过”分开显示。
一次实验可以正确完成，同时保留未通过的科学门。

## 为什么在现有 VLA 系统中加入 Sentinel

**保留已有策略，控制最终请求。** Sentinel 把官方后处理器输出的精确
shape、dtype 和字节绑定到当前反馈及执行上下文。短时许可在环境写入器前
只能消费一次，动作、判定和实际反馈共同形成可携带的签名记录。

| 原生接入实测 | 结果 |
|---|---|
| 直接执行与 Sentinel 主动授权 | **33/40 对 33/40** 次任务成功，七个失败全部保留 |
| 完整配对动作序列 | **40/40 字节一致**，覆盖 **5,420** 个请求 |
| 六类无效授权 | **60/60 阻断**，**0** 次越权写入器调用 |
| 授权加写入前准入耗时 | **均值 0.802 ms · P95 1.040 ms**，不含推理和 `env.step` |
| 实际 App 导入与导出 | 独立公钥核验完整 40 条运行记录，导出 ZIP 再次核验通过 |

固定工程对照在 **RTX 4090 D（24 GB）** 上运行，两组都使用官方
SmolVLA/LIBERO 实际推理。原生 profile 建立进程内请求完整性，物理检查由
其他 profile 分别声明。
[完整结果与每个失败](docs/research/2026-10-03/native-product/RESULTS.md) ·
[冻结方案](docs/research/2026-10-03/native-product/RAW_REQUEST_PROTOCOL.md) ·
[原生证据包与视频](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/native-vla-product-20261003)。

LeRobot 提供策略部署与 RTC，MoveIt Pro 提供运动与物理控制，Foxglove/Rerun
提供可视化工作流。Sentinel 专注最终请求授权与可核验的写入证据，适合与
这些系统配合使用；当前没有已安装竞品的性能排名。
[有来源的职责对照](docs/native_method_position.md)。

![macOS App 查看已签名的正式原生运行](docs/screenshots/native-formal-active-v3-overview.png)

## 五分钟安装

### 交互安装器

```bash
git clone https://github.com/LancerLSY/sentinel-evc-lab.git
cd sentinel-evc-lab
./tools/install.sh
```

选择完整 profile 可安装 MuJoCo，选择 core profile 可得到更小的运行环境。
向导会创建隔离环境、保留公开源码快照、显示真实阶段进度，并可构建 macOS App。
如果 `python3` 版本过低，可执行
`PYTHON=/path/to/python3.12 ./tools/install.sh`。Windows 用户在 PowerShell
中运行 `.\tools\install.ps1`。

### 无人值守安装与启动

```bash
./tools/install.sh "$HOME/Applications/Sentinel-EVC"
open "$HOME/Applications/Sentinel-EVC/Sentinel EVC.app"
```

安装后的 CLI 位于 `Sentinel-EVC/bin/sentinel-evc`。其他桌面平台或不使用 App
时，执行 `sentinel-evc serve`，打开命令输出的回环地址。

### 从源码运行

```bash
python -m pip install -e ".[physics]"
python -m sentinel_evc serve --data-dir runs/workbench --port 8765
```

打开 `http://127.0.0.1:8765`，可新建或导入场景、比较四个最终候选、
运行有记录的仿真、查看事件并导出签名 ZIP 证据。

[App 安装与分发边界](docs/install_app.md) ·
[模型、机械臂与执行器接口](docs/integration_ports.md) ·
[产品验证记录](docs/product_validation.md)

### 使用已有 VLA 环境

模型依赖保留在已有环境中。安装后的 CLI 使用该环境的 Python 启动原生运行：

```bash
sentinel-evc native-run --python /path/to/vla/bin/python \
  --config /path/to/native-active.json --out /path/to/new-run
sentinel-evc native-import --archive /path/to/sentinel-native-vla-bundle.zip \
  --public-key /path/to/independently-retained.public --run-id ACTUAL_RUN_ID \
  --data-dir /path/to/workbench-data
```

在 App 中进入“原生 VLA 运行”，选择任务与保存的时间点。三维场景使用该任务
真实的 MuJoCo 可视网格与 body 姿态，同一帧显示动作请求、授权判定和
submitted / accepted / observed 游标。导入记录不会启动推理或授予机械臂运动权限。

[原生门禁契约与配置](docs/native_vla_gateway.md) ·
[原生请求配对协议](docs/research/2026-10-03/native-product/RAW_REQUEST_PROTOCOL.md) ·
[方法与相关工具](docs/native_method_position.md)

![原生请求授权与实际环境反馈](docs/media/native-request-flow.svg)

## 执行许可如何产生

<p align="center">
  <img src="docs/media/execution-pipeline.svg" alt="上游动作经过变换、最终物理检查和 WorldGuard、一次性许可、仿真或驱动适配器，最终生成签名证据" width="100%">
</p>

1. 上游 VLA、规划器或已记录策略给出动作块。
2. 重定时、修复、坐标转换等步骤生成最终候选。
3. 按所选 profile 校验最终候选：受限数值 profile 执行物理与 WorldGuard 检查；
   原生 Panda profile 检查请求身份、格式与执行上下文。
4. Authority 签发与计划、上下文绑定且有期限的一次性许可。
5. Executor 复核最新状态，并作为唯一组件调用 controller。
6. submitted、accepted、observed 三个游标进入可独立校验的签名日志。

撤销会阻止旧代次继续提交。恢复需要取消确认、排空、最新反馈、当前 epoch
和明确批准。签名只能证明相对于指定公钥的记录完整性，不能证明传感器诚实或
动作确实在物理世界发生。

[机制实现矩阵](docs/implementation_matrix.md) ·
[产品契约](docs/product_contracts.md) ·
[设计追踪](docs/design_alignment.md)

## 冻结实验方案与清晰的三维演示

[**冻结协议与完整结果**](docs/research/2026-10-02/paper-v2/RESULTS.zh-CN.md) ·
[**数据、源码与 1080p 视频**](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/paper-validation-20261002)

| 新实验 | 实测结果 | 主要限制 |
|---|---|---|
| 支持域路由，600 个新 roots | 完成 300、拒绝 300，零次观测到不安全选择 | 固定 4.8 秒完成全部 600；拒绝计为未完成 |
| 物理压力，27 组 / 324 个 roots | 4.8 秒完成 324/324；1.6 秒完成 184/324、掉落 78 次 | 下界策略等于固定 4.8 秒；有限仿真样本，耗时三倍 |
| 未来隐藏摩擦变化，100 对 roots | 相同输入下，1.6 秒动作的结果在 100/100 对中不同 | 需要可信物理边界或补充观测 |
| 公平 UR5e 校验成本，60 个 roots | 全检与增量 60/60 判定一致；平均边际节省 0.996 ms | 增量 P95 更差，危险类型仍有限 |
| 原始官方 SmolVLA/LIBERO 闭环，100 次固定任务 | 成功 58/100；task5 失败 10/10 | 官方 Panda 模型、仅观察记录；与 SO100 overlay 分开 |
| 配对版本闭环，40 次新状态执行 | task5：0/10 → 7/10；对照：10/10 → 10/10 | 完整 MuJoCo 3.8.1 / 3.3.7 处理；基准兼容性，不是物理精度结论 |
| 全任务新初态确认，100 次执行 | **成功93/100**；零崩溃；task4仍为6/10 | 官方 Panda 模型、隔离MuJoCo3.3.7、执行间隔10；与原58/100为不同网格 |

[![完整 UR5e 网格：任务、计划变化、判定与结果](docs/research/2026-10-02/paper-v2/video/sentinel-ur5e-paper-poster.png)](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/paper-validation-20261002/sentinel-ur5e-paper.mp4)

**UR5e，1080p · 19.95 秒。** 红色画面展示被拒绝计划的反事实运动，该计划没有下发；旁边展示放行执行或不下发候选。视频使用官方完整机械臂网格与保存的 MuJoCo 关节状态。三个案例解释早期机制 fixture，不代表新成本实验或真机录像。

[![相同初态：快速动作打滑掉落，慢速动作完成运输](docs/research/2026-10-02/paper-v2/friction-video/friction-failure-vs-fallback-poster.png)](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/paper-validation-20261002/friction-failure-vs-fallback.mp4)

**接触失效，1080p · 10.2 秒。** 同一个新低摩擦样本对比 1.6 秒和 4.8 秒动作：载荷掉落与完成运输。实时位移曲线解释风险阈值；后备动作耗时为三倍。视频重建真实保存的仿真状态，带中英双语说明。

[![官方 SmolVLA 原生 Panda 任务、动作链与结果](docs/research/2026-10-02/paper-v2/libero-video/libero-task00-state00-poster.png)](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/paper-validation-20261002/libero-task00-state00-explained.mp4)

**原始 Panda 配置，4.15 秒。** 预先指定的 task0/state0 在 MuJoCo 3.8.1、每个预测片段执行 50 个动作的配置下，经过 83 个实际动作后成功。360×360 仿真 RGB 放在中英双语 1080p 画布上，按真实 20 Hz 控制时钟播放。这个视频属于原始 58/100 研究，全部 42 次失败仍被保留；93/100 确认实验使用另外的网格与配置。

[保留的失败视频](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/paper-validation-20261002/task5_state0_failure_bilingual_20hz.mp4)精确重放原始 task5/state0 动作：零奖励、任务失败、目标碗体中心最高仅上升 1.674 mm。它是事后诊断，不增加基准样本。

## 实测结果

以下数字均来自冻结流程与保留的原始产物，只描述指定数据和 profile，
不构成通用“机器人安全率”。

### 实测硬件：RTX 4090 D（24 GB）

下表 GPU 数字全部实际测自 **RTX 4090 D（24 GB）**。

| 实验 | 实测结果 | 含义 |
|---|---:|---|
| SmolVLA，5,000 步 | 留出动作标准化 MAE **0.673166 → 0.245984**（下降 **63.46%**） | 固定 SO100 记录与冻结模型/数据版本 |
| SmolVLA 图像到动作推理 | **P50 229.26 ms · P95 236.90 ms** | 4 线程，内存图像输入至反标准化输出 |
| W0/W1/W2 ensembles | 共 **36 个已训练成员** | 数值、MuJoCo 对象、视觉对象、真实数据关节研究 |
| 固定相机视觉 profile | 观测到 **0/500** 不安全选择 | 同一固定仿真测试组；低于名义目标的覆盖率仍保留 |
| 相机变化、沿用原校准 | **47/500** 不安全选择 | 说明相机身份必须进入受支持 profile |
| 匹配新 profile 的重校准对照 | **0/500**，动作全部为 **1.6 s** | 只是在既有压力集上恢复，不是新的独立部署证明 |
| SO100 状态/动作与视觉/状态/动作 | 加图像后标准化误差增加 **10.0%** | 该任务中更简单的状态/动作模型更好 |

最强的物理辨识基线也优于严格学习型数值模型。负结果与正结果一起保留。

### 新分布 W2 审计

一轮独立记录评估在 **600 个新 MuJoCo roots** 上运行既有状态与图像 ensemble：
六类场景，每类 100 roots，每个 root 四个同胞动作。

| 新独立场景 | 状态 gate | 图像 gate | 主要结论 |
|---|---:|---:|---|
| 标称分布 | **0/100** 不安全选择 | **0/100** | 本组没有观测到不安全选择 |
| 相机变化 | **0/100** | **7/100**（root bootstrap 95% 区间 **3%–12%**） | 不依赖相机的状态 gate 是更强对照 |
| 隐藏低摩擦 | **100/100** | **100/100** | 四个候选全部不安全，但两个学习 gate 仍各自选出一个 |

1.35× 位移超出已训练动作 contract。完整状态覆盖率降至状态模型 15%、图像模型
8%，因此产品声明结果为 `MODEL_UNKNOWN`，拒绝全部 100 roots；裸模型选择只作为
退化诊断保留。该场景实验使用 RTX 4090 D，总运行时间 30.53 s；物理由 MuJoCo workers
计算，因此该时间不是模型推理延迟基准。

[场景报告](docs/worldguard_scenario_results.md) ·
[运行 manifest](docs/research/2026-10-02/worldguard-scenarios/manifest.json) ·
[原始 metrics](docs/research/2026-10-02/worldguard-scenarios/metrics.json)

### 低摩擦物理降级方案

低摩擦审计暴露了明确边界：原 2 s 动作族的四个候选全部不安全，校准可以拒绝，
却无法凭空产生可执行动作。因此另建一个 **5 s 物理 profile**，在 100 个新 roots
上比较 1.6、3.2 和 4.8 s 计划。

| 新 profile 里的计划 | 不安全 | 掉落 | 托盘任务完成 | 代价 |
|---|---:|---:|---:|---:|
| 固定 1.6 s | **100/100** | **58/100** | **0/100** | 基准时长 |
| 固定 4.8 s | **0/100** | **0/100** | **100/100** | 动作时长 **3×** |

100 个 roots 中零次观测到不安全，对应 95% Wilson 上界为 **3.70%**，不是零风险
保证。筛选规则使用配置值 `μ_min=0.015`，系统并未测量摩擦系数；它选择 4.8 s。
未被采用的 3.2 s 计划在
100 个 roots 中同样全部安全，但被保守规则拒绝。这是新的、尚未部署的 MuJoCo
profile，不是对原 2 s 神经 WorldGuard 的修复，也不证明真机未知摩擦下可用。

[低摩擦报告](docs/low_friction_fallback.md) ·
[运行 manifest](docs/research/2026-10-02/low-friction-fallback/manifest.json)

[完整 GPU 报告](docs/gpu_training_results.md) ·
[模型卡与加载方式](docs/gpu_model_cards.md) ·
[已评估 SmolVLA overlay 模型卡](docs/huggingface/smolvla_model_card.md) ·
[复现命令](experiments/gpu/README.md) ·
[已校验实验包](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/gpu-experiments-20261002) ·
[Hugging Face 模型](https://huggingface.co/LancerLSY/sentinel-smolvla-so100)

已评估的 SmolVLA overlay、加载器与模型卡已发布至 Hugging Face。21 个模型包文件
均与选中的本地版本一致，详见 [Hub 发布校验记录](docs/huggingface/smolvla_hub_publication.json)。
该 overlay 需要配合固定的原始 SmolVLA base 与冻结骨干加载。
打包 loader 已重建有限值 `[1,50,6]` dev-only 输出，并与固定参考逐位相同
（`maximum_absolute_difference: 0.0`）；详见[加载校验](docs/huggingface/smolvla_loading_verification.json)。

## 三维机械臂研究

<p align="center">
  <img src="docs/media/ur5e-validation.gif" alt="MuJoCo UR5e late_suffix root 10000 的真实轨迹；单独实现的高分辨率复核器回放发生碰撞，最终计划完整校验将其拒绝" width="82%">
</p>

当前 MuJoCo Menagerie UR5e 结果覆盖 **180 个同根比较**：六类场景，每类
30 roots。一个单独实现、不导入 gate runner 的高分辨率复核器使用 0.005 rad
静态采样和 1 ms MuJoCo 步长，标记 139 个不安全 roots、41 个安全 roots。
动画使用 `late_suffix / root 10000` 保存的关节轨迹：只校验父计划会放行，
最终计划完整校验会拒绝，高分辨率回放记录到碰撞。
GIF 是结果条件化展示：它是在复核标签已知后选出的首个不安全 `late_suffix` 案例，
用于解释失效，不用于估计发生率。

| 策略 | 放行 | 误放行 / 139 个不安全 | 误拒 / 41 个安全 | 平均实测校验墙钟 |
|---|---:|---:|---:|---:|
| 只校验父计划 | 166/180 | **125/139** | 0/41 | 101.15 ms |
| 最终计划完整校验 | 41/180 | **0/139** | **0/41** | **48.74 ms** |
| 增量校验 + 强制回退 | 41/180 | **0/139** | **0/41** | **总计 147.65 ms** |
| 保守拒绝所有变换 | 30/180 | 0/139 | **11/41** | 0 ms；没有校验调用 |

增量路径在 106/180 roots 上授权静态前缀复用，在 74/180 roots 上完整回退；
14 个父记录缺失或父计划被拒绝的 roots 全部回退。增量阶段平均 46.50 ms，
父计划校验另需 101.15 ms；端到端计入后，增量路径**慢于 48.74 ms 的完整校验**。
动态校验始终从第零帧完整回放；只复用绑定后的静态前缀记录，绑定失效就完整回退。
这个进程内记录绑定前缀、模型、资产、时间步、初始状态和上下文，但不是完整 v4
外部签名证书。

复用子集平均为 112.18 + 44.55 = 156.73 ms；回退子集平均为
85.34 + 49.30 = 134.64 ms。这是固定顺序、静态检查可提前结束的一次矩阵观测，
不是算法基准。高分辨率复核沿用早期运行的同一组 180 个构造 roots，障碍位置带有
对抗性设计，不是新的自然分布留出集；`0/139` 不是广义可靠性证明。

<p align="center">
  <img src="docs/media/ur5e-comparison.svg" alt="UR5e 同根场景下父计划、完整、增量与保守策略对比" width="92%">
</p>

[静态帧](docs/media/ur5e-validation.png) ·
[七案例 720p 合集](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/simulation-research-20261002/ur5e-validation-montage.mp4) ·
[v3 运行 manifest](docs/research/2026-10-02/ur5e/v3/manifest.json) ·
[v3 逐 root 结果](docs/research/2026-10-02/ur5e/v3/per_root.json) ·
[高分辨率复核](docs/research/2026-10-02/ur5e/v3/review/reviewed_metrics.json) ·
[媒体 manifest](docs/research/2026-10-02/ur5e/media_manifest.json) ·
[仿真研究 Release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/simulation-research-20261002)

合集覆盖七个代表案例：六类场景各一个，另加一个安全窄道案例；案例是在知道复核
结果后选取。287 个保存的关节状态帧按 1× 展示，共 14.35 s；33.75 s 总时长还包含
明确标注的标题与停留画面。GIF 将 29 帧逻辑序列编码为 21 帧、2.9 s。媒体渲染
不会重新运行物理或重新计算 gate。

Release 归档还提供可独立加载的便携 UR5e 模型，共 27 个文件，其中 26 个为已核验的
官方 Menagerie 资产。[原 v2 manifest](docs/research/2026-10-02/ur5e/manifest.json)、
[逐 root 记录](docs/research/2026-10-02/ur5e/per_root.json)和
[复核记录](docs/research/2026-10-02/ur5e/original_review.json)继续公开作为历史资料；
旧版时间不再作为默认结果。

该机械臂研究验证的是独立实验 profile，并不表示 UR5e 模型、学习型 WorldGuard
或物理机械臂驱动已经接入产品核心 Executor。当前产品的机械臂接口仍为只读。

[研究展示与实验边界](docs/research_showcase.md)

## 工作台

<p align="center">
  <img src="docs/screenshots/physics-workbench.jpg" alt="显示三维轨迹、证据状态和科学验收门的 MuJoCo 工作台" width="92%">
</p>

```bash
python -m sentinel_evc run --mode physical --out runs/numeric01
python -m sentinel_evc train-baseline --mode residual --out runs/residual01
python -m sentinel_evc physics --out runs/physics01 --seed 7 --render
python -m sentinel_evc verify-physics --out runs/physics01
```

内置数值校准只覆盖固定 0.35 m 候选族。未支持的动作族保持 `unknown`，
物理或风险违规为 `denied`。固定接触 fixture 保留负结果：只按几何选择的
最快候选可能滑移，且最快分支未通过终态姿态收敛门。

## SSH 复现

```bash
# 需要已有 OpenSSH alias、Python 3.10+、密钥代理与 known_hosts
python -m sentinel_evc remote-physics \
  --host YOUR_ALIAS --out runs/remote01 --seed 7
python -m sentinel_evc verify-physics --out runs/remote01
```

SSH 路径在远端运行同一可信源码，取回证据包，并对照提交源码与传输收据校验。
它不读取密码，并拒绝变化的主机密钥。远端渲染需要可用 EGL 驱动。

[物理与 SSH 设计](docs/physics_ssh_design.md) ·
[性能协议](docs/performance_protocol.md) ·
[实测性能](docs/performance_results.md)

## 复现参考机制

紧凑 CPU Demo 只依赖 core profile，并要求输出到新的空目录。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m sentinel_evc demo --out runs/demo --run-id demo --cases 1000
python -m sentinel_evc verify \
  --bundle runs/demo/bundle \
  --public-key runs/demo/anchors/demo.public \
  --run-id demo
```

打开 `runs/demo/report.html` 查看离线回放。三幕分别证明：校验后的变换会使旧结论失效；撤销不能抹去已接受的动作；
第三方无需信任产生证据的运行时，也能校验签名记录。准确复现命令与历史测量见
[RESULTS.md](RESULTS.md)。

## 当前科学与产品边界

- 数值、固定 MuJoCo、视觉/GPU 与机械臂研究是彼此独立的实验 profile。
- 导入的 OBJ/STL/MJCF/URDF 只会解析和预览，不会获得运动授权。
- ROS 2 运动链、物理机器人执行、mTLS、生产 PKI 与公证分发仍待完成。
- “零次观测到失效”只描述指定样本；覆盖率、误放行、任务成功和物理事故是不同指标。
- 证据签名只建立记录完整性，不建立传感器真实性、物理因果或责任判断。

[实验计划](docs/experiment_plan.md) ·
[可靠性记录](docs/reliability_results.md) ·
[能力矩阵](docs/implementation_matrix.md)

## 开发

```bash
python -m pytest -q
python run_tests.py
```

核心运行依赖是 `cryptography`；`physics` profile 增加 MuJoCo 和 Pillow。
本地 UI 不使用 web 框架、前端构建步骤或 CDN。逐包实测许可见
[依赖与许可证登记](docs/依赖许可.md)。

贡献请附测试、输入输出样例和已知限制。特别欢迎能推翻仓库结论的反例；
负结果会保留。参见[当前任务板](docs/任务板.md)。

## 许可

代码采用 [MIT License](LICENSE)。MIT 不含专利授权条款。核心机制另有专利
申请 **202611458350.1**；软件许可证不构成明示或默示的专利许可。
第三方依赖保留各自许可证。
