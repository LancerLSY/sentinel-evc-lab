# 从 App 或 CLI 运行统一闭环

这一入口把已有的统一实验接进产品生命周期：启动真实 SmolVLA 推理，生成原生动作块（可选固定重叠聚合），检查 Panda 预测运动，取得一次性许可，调用实际 MuJoCo writer，再从环境反馈进入下一轮。App 的「闭环运行」管理这条路径，原来的「AI 执行记录」继续用于导入和回放。

支持组合是 **官方 SmolVLA LIBERO 检查点 + LIBERO Spatial/Panda 的任务 0、4、5 + Linux/MuJoCo 3.3.7**。SO100 微调检查点不能替换这个 Panda 检查点。核心安装不新增 ML 依赖，现有数值和接触实验保持各自的运行环境。

## 配置一次，重复运行

在已有、符合[统一实验依赖](research/2026-10-04/unified/REPRODUCE.md)的 Linux VLA 环境中安装核心包：

```bash
/path/to/vla/bin/python -m pip install -e .
cp experiments/vla/product_session.example.json /path/to/session.json
```

编辑 `session.json` 中 checkpoint、backbone 的绝对路径，以及 task、initial state、seed 和 max_steps。采用 A6 的检查与短前缀策略，`aggregation: "native"` 默认保留模型原生动作；`fixed_overlap` 显式启用固定重叠聚合。每次只运行 `delta_evc`，不会运行跳过最终检查的对照分支。

```bash
sentinel-evc loop-configure \
  --python /path/to/vla/bin/python \
  --source /path/to/sentinel-evc-lab \
  --config /path/to/session.json \
  --data-dir runs/workbench
sentinel-evc loop-status --data-dir runs/workbench
sentinel-evc loop-run --data-dir runs/workbench
```

`loop-configure` 保存本地运行绑定并报告缺失项；环境未就绪时退出码为 3。修改绑定的 Python/JSON 源码后必须重新配置。配置操作与 CLI 运行需要独占该工作区，先关闭使用同一目录的工作台。

启动 App/浏览器操作台时使用同一个 data-dir：

```bash
sentinel-evc serve --data-dir runs/workbench --port 8765
```

打开 `http://127.0.0.1:8765/#loop`。环境检查通过后，点击「启动闭环」。浏览器只能启动已绑定配置，不能提交程序路径、SSH 密码或任意控制命令。每个工作区一次只运行一个统一 VLA 作业。

如果模型在 SSH 服务器上运行，在服务器启动上述服务，在电脑上建立现有可信 SSH 连接的端口转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 YOUR_EXISTING_HOST_ALIAS
```

随后打开同一个本地 URL。使用已登记的主机密钥；服务器服务继续只监听 loopback。

作业原始资产上限为 256 MiB；工作区启动预算为 8 GiB，每次预留原始文件、签名副本及 ZIP 的三份空间。空间不足时拒绝新作业，不自动删除证据。关闭服务后，可把已核验并备份的完整旧作业目录移出 `loop-jobs`，或选择新的 data-dir。

## 操作和结果

页面分别展示候选动作、几何判定、已执行动作、实际姿态、执行游标和任务结果。三维画面来自该任务实际模型与环境反馈，拒绝的候选不会成为已执行运动。没有反馈时显示等待状态。

「停止」或 CLI 的 Ctrl+C 创建停止请求，runner 在下一次新 writer 入口前检查它并撤销许可。已经进入的同步仿真步可以结束，最后反馈会保留。停止后要新建运行，不能复用旧许可恢复。服务退出会请求协作停止，再清理未退出的仿真进程。这些行为只适用于软件仿真；实机取消与制动还需要设备适配和测量。

运行结束后下载 ZIP，保留独立公钥与运行编号，再用现有校验器核验：

```bash
sentinel-evc verify --bundle EXTRACTED/bundle \
  --public-key INDEPENDENTLY_RETAINED_PUBLIC_KEY --run-id loop-ACTUAL_ID
```

证据包保留任务配置、源码摘要、几何/许可/实际写入记录、运行结果和最终反馈。`inventory.json` 把原始相对文件名映射到签名包中的平面文件名；gateway 原始事件作为独立签名资产保留。外层事件记录是进程退出后的交付收据，不冒充逐动作 gateway 事件。默认公钥是本地演示信任源。

运行 `completed` 表示软件流程正常结束。`task_success` 和 success/episodes 表示任务结果；达到步数上限、发生拒绝或任务未完成不会被改成成功。失去服务进程、封装失败或记录被改动会显示中断/失败，并禁止导出未经核验的完成包。

## 验收边界

已有 A6 统一实验结果仍为最终检查与增量路径各 **3/6**；未聚合原生策略的 **6/6** 属于单独的事后对照。新产品入口不改变这些结果，不新增成功率结论。新入口需要在可用 Linux/GPU 环境中完成独立的实际运行验收。[本轮接入验证记录](managed_closed_loop_validation.md)保留已通过的检查、物理实验负结果和待验收项目。

本路径可显式启用已声明的固定重叠聚合。LeRobot 的官方 [RTC](https://huggingface.co/docs/lerobot/main/rtc) 通过异步生成和前缀引导衔接动作块；这次接入没有把固定平均操作称为完整 RTC。其他策略或设备应通过明确的动作语义和 writer 适配接入，不能仅凭文件导入就获得运动权限。官方部署接口见 [Policy Deployment](https://huggingface.co/docs/lerobot/main/inference)。
