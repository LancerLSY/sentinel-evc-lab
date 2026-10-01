# 模型资产与机械臂接入端口

## 模型资产

`AssetStore(root)` 只接受单文件、大小受限的 `OBJ`、`STL`、`MJCF` 和 `URDF`。导入接口为：

```python
store.import_asset(name, format, content=raw_bytes)
store.import_asset(name, format, content_base64=encoded_text)
store.list()
store.get(asset_id)
store.geometry(asset_id)
store.check(asset_id)
```

OBJ 和 STL 的结果是实际解析得到的三角形顶点与索引。MJCF 和 URDF 导入阶段给出基本几何体预览；安装 `sentinel-evc-lab[physics]` 后，`check` 会用 MuJoCo 编译模型，执行五个有界仿真步，并把盒、球、圆柱和胶囊体转换成编译后的世界坐标预览。

XML 导入拒绝 DTD、实体、处理指令、include、plugin、attach、外部文件和 URL。为避免编译前资源膨胀，它也拒绝 replicate、composite、flex、mesh、hfield、skin、texture、显式 `<size>` 内存配置及过大的 solver iteration。当前入口最多接受 2,000 个 XML 元素、1,000 个几何体、128 个关节和 128 个执行器，且基本几何尺寸与位置有有限幅值上限。源文件按 SHA-256 记录，写入采用临时文件加原子替换。

模型检查只说明文件能被当前 MuJoCo 版本编译、在五步推进中保持有限状态且没有 MuJoCo warning。它不生成几何安全证书，不把任意机器人模型加入可信 fixture，也不批准 Executor 派发或训练。

## 只读机械臂诊断

`RobotRegistry(root)` 保存名称、driver、host 和 port，不接受或保存密码、密钥、令牌：

```python
registry.create(name, "mock")
registry.create(name, "ur_dashboard_readonly", host, port=29999)
registry.list()
registry.get(robot_id)
registry.diagnose(robot_id, timeout=1.5)
```

`mock` 是进程内协议演示，结果始终标明没有连接硬件。`ur_dashboard_readonly` 使用 Universal Robots Dashboard Server，连接超时和响应长度均受限，只发送以下三个换行结尾的查询：

- `get robot model`
- `robotmode`
- `safetystatus`

Universal Robots 官方资料说明 Dashboard Server 使用 TCP 29999 且命令以换行结束；官方命令表将上述三项定义为型号、机器人模式和安全状态查询：

- [Dashboard Server e-Series, port 29999](https://www.universal-robots.com/articles/ur/dashboard-server-e-series-port-29999/)
- [Dashboard Examples](https://www.universal-robots.com/manuals/EN/HTML/SW5_24/Content/prod-dashboard/DashboardExamples.htm)

此连接器不发送 `play`、`stop`、`power on`、`brake release`、解锁保护停机或任何运动命令。成功诊断只代表在本次有界 TCP 会话中收到了符合协议的只读响应。

## Executor 控制器端口

真实硬件 motion adapter 需要经过独立设备评审和实机验证。`examples/integrations/controller_adapter_contract.py` 记录当前 Executor 实际调用的端口：

- `free_slots()`：返回控制器当前可接纳的真实队列容量。
- `submit(action, generation, gripper_event)`：唯一写入口；只在设备确认接纳时返回 `True`。
- `cursors()`：分别报告 submitted、accepted、observed。写入成功不能冒充 observed。
- `read_feedback(now_ns)`：返回设备实际位置、同一单调时钟域的采样时间和真实游标。
- `cancel()` 与 `cancel_acked`：取消请求及设备确认是两个状态。
- `is_drained`：只在已接纳命令的设备侧尾部全部结束后为真。
- `tick()`：推进或轮询一次有界 I/O 周期。

连接器还必须支持现有 Executor 的撤销流程：停止旧代次的新提交，等待取消确认，排空已接纳尾部，再使用撤销后的新反馈恢复。当前 `RobotRegistry` 只提供配置与诊断，不会把 profile 自动注册为 motion adapter。
