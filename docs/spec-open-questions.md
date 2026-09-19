# Sentinel-EVC v0.1 规范决策台账

本台账区分 **来源记载** 与 **维护者确认的工程补充**，不把建议、目标或历史成绩写成官方原文或本次实测。原文见 [DOCX 转录](临界哨兵_Sentinel-EVC_开源_Demo_实施方案_转录.md)；以下“§六 6.1”等均指该转录中的原文章节。

Q01–Q11 沿用《Sentinel-EVC_v0.1_逐分支实施规划》§2 编号。补充依据为维护者在本会话明确要求实施的《Sentinel-EVC v0.1 分批并行实施计划》§2–§5。批准规则不等于已经实现、冻结 schema 文件或验收通过；每批实际状态由 README 和本地交接记录给出。

## 一、原文未决

此表列的是原文留下的空白；“已有补充”只表示下面有维护者决定，不能反推原文曾规定这些细节。未被决定覆盖的公共合同不得由实现自行补全。

| 编号 | 原文依据与空白 | 本轮决定覆盖情况 |
| --- | --- | --- |
| Q01 规范化字节 | §五、§六 6.5 只要求排序键、固定浮点格式、UTF-8 无 BOM；未给精度、舍入、负零、非有限数、摘要文本和 HMAC 确切编码。 | 下文 Q01 已有补充。 |
| Q02 领域合同 | §四、§六 6.2–6.3 给出 Plan/Certificate/Lease 的用途和 Lease 字段；Plan、Certificate、Snapshot 的完整字段及深度上限未给全。 | 已确认不可变数据内容和深度。StateStore 的独立公开接口未定义，不另建公开协议。 |
| Q03 schema 与事件 | §七给三种示例、13 个事件名；未给 required/可空/额外字段、完整 payload、BACKUP、路径 a、seq 和溢出行为。 | 已确认共同字段、payload、路径与溢出规则；具体 schema 文件在批次 01 按已确认合同形成，不能现在声称已冻结文件。 |
| Q04 证据格式 | §六 6.5 给链和 manifest 公式，§四给四种篡改；首项、空流、文件覆盖范围、签名/公钥编码、失败返回结构未给全。 | 已确认字节、范围及校验顺序；verify 的结构化失败返回格式未定义，不新增对外结果字段。 |
| Q05 几何边界 | §六 6.1–6.2 给非退化段公式与 L=1；零长度、零余量、容差、盒边界预留未定。 | 下文 Q05 已有补充。 |
| Q06 执行 profile | §四、§六 6.3–6.4 给 dt/prefix/TTL 示例、容量、时钟和恢复前提；观测年龄、起点管、剩余时长边界及调度细节未给全。 | 已确认阈值、剩余预算、确定性控制器、恢复入口；未给逐项 ACK 延迟数值或新的 StateStore 协议，不自行扩张。 |
| Q07 错误与消费 | §四、§六 6.3–6.4 给五个错误码及部分对应关系；过期、认证失败、队列满、全部失败映射和消费时机未定。 | 已确认三个补充码与消费时机；未明确的逐项错误映射在相关批次核对，不以新增错误码解决。 |
| Q08 数据构造与统计 | §四给 P1/P2、C_mix/C_near 和三路径；扰动细节、场景采样范围、选用父证书、汇总完整字段未定。 | 已确认固定场景与轨迹构造；C_mix 选择哪张父证书、summary 的完整公开字段仍未明确，不把汇总目标当新 schema。 |
| Q09 CLI 与产物 | §一、§五给四子命令与部分参数；fault 标识、run_id、目录和退出码未给全。 | 下文 Q09 已有补充。 |
| Q10 独立对照 | §十要求不同方法、1000 案例零分歧；密集采样举例没有指定密度。 | 已确认每段 1001 点；不得把有限采样零分歧推广成任意连续轨迹证明。 |
| Q11 WorldGuard | §四要求占位说明和接口，未提供任何接口名称、签名或输入输出。 | 维护者明确只保留空目录说明；接口仍未定义，不编造，不阻塞数值主链。 |

## 二、已由维护者确认的工程补充

以下均是维护者确认的本 Demo 规则，**不是对 DOCX 原文的逐字转录**。凡原文已经明确且没有补充的机制，直接引用原文章节，避免另写一套公式或业务规范。

### Q01 规范化字节、摘要与 HMAC

- 有限浮点按 binary64 实际值以 `.16e` 输出：17 位有效数字、保留尾零、小写 `e`、指数带符号；最近值、中点取偶，不预先 `round`。
- `-0.0`、`+0.0` 都编码为正零。拒绝 NaN、Infinity、孤立 surrogate、非字符串对象键、重复 JSON 键。
- 对象递归按 Unicode 码点排序；数组保序；整数、浮点、布尔严格区分。无缩进、无额外空白、无末尾换行，UTF-8 无 BOM，不做 Unicode 归一化。
- 摘要文本为 `sha256:` 加 64 位小写十六进制。
- Lease 使用 HMAC-SHA256：输入为移除 `hmac` 字段后的完整 Lease 规范字节；输出为无前缀的 64 位小写十六进制。
- 固定字节/摘要向量、跨新进程稳定性由合同测试验证。独立 verify 自己实现规范化与解析，不复用生产侧协议实现；不声称 RFC 8785（原文 §六 6.5）。

### Q02 领域对象

领域对象全部为 `frozen dataclass`；嵌套容器构造时复制为 tuple，不能通过输入别名修改。

| 对象 | 维护者确认的内容 |
| --- | --- |
| Plan | `points: tuple[tuple[float, float, float], ...]`、`dt: float`、`gripper_events: tuple[str, ...]`、`controller_profile: str`、`task_phase: str`。哈希覆盖全部字段；后三项只作不透明身份和相等性比较。 |
| Certificate | `cert_id`、内嵌不可变 Plan、`scene_id`、每段最小剩余余量、`inherit_depth`、可空 `parent_cert_id`。完整检查为深度 0 且无父证书；继承为父深度加一。 |
| LeaseContext | 严格六键：`robot`、`boot`、`epoch`、`scene_id`、`queue_rev`、`committed_prefix_hash`。 |
| Lease | `lease_id`、`final_hash`、`context`、`cert_id`、`prefix_len`、`deadline_mono`、`hmac`。 |
| Snapshot | `obs_id`、Lease context 的六个值、当前 `position`、`observed_mono`。 |

最大继承深度为 4；父深度小于 4 才能继承，父深度等于 4 回退完整检查。必要不变量限于有限坐标、三维点、至少两节点、正 `dt`、余量数量匹配、ID 非空、深度与父证书关系正确；不引入泛化校验框架。余量累计扣减和六类依赖失效按原文 §六 6.2；继承失败必须转完整检查，不能直接判 REJECTED。

### Q05、Q06、Q07 几何与执行 profile

场景采用原文 §七的数值示例作为固定 Demo 场景：工作空间 `[0,-0.4,0]` 至 `[0.8,0.4,0.6]`；球心 `[0.4,0,0.3]`，半径 `0.05`；工具半径 `0.02`；tracking reserve `0.005`；`dt=0.05`、horizon `16`。

- 零长度段按单点检查；球和收缩后盒边界净余量必须严格大于 0，等于 0 违规。盒边界同时扣工具半径和 tracking reserve，不设隐藏 epsilon。非退化段公式原样采用 §六 6.1。
- 默认 `prefix_len=4`、Lease TTL `500ms`、控制器队列容量 2、观测最大年龄 `100ms`、起点允许管 `5mm`。Commit 时剩余 deadline 必须大于等于 `prefix_len × dt`。
- 错误码保留原文五项 `STATE_STALE`、`CONTEXT_CHANGED`、`LEASE_REPLAY`、`TRACKING_TUBE`、`CANCEL_UNCONFIRMED`；仅补充 `LEASE_EXPIRED`、`AUTH_FAILED`、`QUEUE_FULL`。
- 所有校验通过后才在 Commit 锁内原子消费 Lease；校验失败不消费；消费后的 dispatch 失败不能重放。
- revoke 锁内三操作和锁外 cancel/等待按 §六 6.4。`recover(snapshot, human_approved)` 要求 cancel 已确认、新快照有效、人工批准为真。
- SimController 用确定性 `advance()`，无后台线程；ACK 延迟/丢失可配置。线程与 barrier 用于竞态测试，不靠随机 sleep 冒充撤销验证。

### Q03 事件与 schema

三份 schema 为 event/scenario/verdict，`additionalProperties` 均为 false。事件公共字段全部必填：`seq`、`ts_mono_ns`、`ts_utc`、`run_id`、`type`、`plan_hash`、`lease_id`、`cert_id`、`payload`、`prev_hash`。不适用的 hash/id 使用 null，禁止额外字段。固定 13 个类型来自 §七；以下 payload 为维护者补充。

| 类型 | 严格最小 payload |
| --- | --- |
| PROPOSAL | `role: parent/child` |
| TRANSFORM | `method: mix/near`、`parent_plan_hashes` |
| CERTIFICATE | `verdict`、`path: full/delta`、`margins`、`first_violation_segment`、`full_checks_used`、`inherit_depth`、`parent_cert_id` |
| PREPARE | `prefix_len`、`deadline_mono` |
| COMMIT | `accepted`、`reason_code` |
| DISPATCH | `step_index`、`submitted` |
| CONTROLLER_ACK | `step_index`、`accepted` |
| OBSERVED | `step_index`、`position`、`observed` |
| REVOKE | `reason`、`old_epoch`、`new_epoch` |
| CANCEL_ACK | `confirmed` |
| BACKUP | `reason`、`backup_plan_hash`；只表示选择已验证备用计划，不表示提交，也不是 Δ-Cert 回退 |
| OUTCOME | `status: completed/rejected/fault`、三个游标 `submitted` / `accepted` / `observed` |
| LOG_GAP | `dropped_count`、`first_dropped_seq`、`last_dropped_seq` |

正式 verdict 保留 `FULL / INHERITED / REJECTED`，只用于路径 b/c；路径 a 使用独立 `baseline_allowed` 与实际违规记录。用标准库和 pytest 检查固定合同，不引入 jsonschema。

`seq` 从 0 开始；队列默认容量 1024，测试可以更小。溢出时为被丢事件保留序号并累计范围；有空间后先发一条 LOG_GAP，再继续正常事件。JSONL 每条为规范 JSON 加一个 LF，保留文件末尾 LF；链哈希不含 LF。样例约 20 条，覆盖全部类型、REVOKE 和 LOG_GAP，标明是格式 fixture，不是运行成绩。

### Q04 证据和独立 verify

- 首事件 `prev_hash` 为 `sha256:` 加 64 个零；空事件流拒绝。
- bundle 只含 `events.jsonl`、`manifest.json`、`manifest.sig`。manifest 字段采用原文 §六 6.5；`manifest.files` 只摘要 `events.jsonl` 的原始字节。
- Ed25519 签名为原始 64 字节，公钥为 bundle 外原始 32 字节。签名输入仍为原文规定的规范 manifest 字节。
- verify 顺序：run_id → JSON/链 → event_count/tip → 文件摘要 → 签名；不导入生产侧 contracts/events/执行核，不复用生产侧规范化和解析。
- 原包须通过；改中间事件、删尾三行、换公钥、改 run_id 分别命中链、count/tip、签名、run_id。四种篡改分别用原包副本，不能串改。

### Q08、Q10 轨迹与独立对照

- 固定场景只对轨迹作有种子扰动，不另作全随机场景。
- 17 个节点，x 从 0.1 线性到 0.7，z=0.3；P1/P2 的 y 为 `±a·sin(πt)`，`a` 由种子在 `[0.10,0.14]` 采样。
- 1000 子案例由 500 个 `C_mix=0.5·P1+0.5·P2` 与 500 个 C_near 组成；C_mix 穿球；C_near 对 P1 内部节点 y/z 加 `[-0.003,0.003]` 有种子扰动，端点不变。
- 独立对照每段取 1001 个密集采样点，不复用生产几何公式；同一 1000 案例零分歧是门禁，不能预写通过。
- 继承不足仍回退完整检查；统计同时报告父证书成本、回退成本、完整检查次数和端到端墙钟时间（表述边界见原文 §十）。

### Q09 CLI 与产物

```text
demo --cases 1000 --seed 1234 --out runs/demo01
geometry --cases 1000 --seed 1234 --out runs/geometry
fault --fault {late,expired,replay,scene-change,queue-rev-change,revoke-race,cancel-unconfirmed,all} --seed 1234 --out runs/fault
verify --bundle ... --public-key ... --run-id ...
```

成功退出 0，业务或验证失败退出 1，参数错误退出 2。run_id 取输出目录最后一段；目标目录已存在即失败，不覆盖。demo 输出 `scenario.json`、`summary.json`、`report.html`、`bundle/` 三文件和 `anchors/demo.public`。Authority HMAC 与 Evidence Ed25519 私钥每次运行分别临时生成，只存在内存；仅保存演示公钥。报告只用 Python 字符串模板和内联 SVG，消费真实结果，不重做判定。

### Q11 与发布边界

WorldGuard 只留 `README_WHY_EMPTY.md`，不编造接口。G0–G5 通过条件与失败降级见原文 §十；仅 G0–G4 真实通过后收录一次 `sample_run/` 和 RESULTS。没有真实非项目成员完成 G5 时保持 candidate。子 agent、自测、CI 或同一作者换环境均不能代替 G5。

候选发布材料采用英文 README 与中文 README.zh-CN；能力、数字、限制一致。`LICENSE.proposed` 不自动改名，CI 未实测绿色不挂徽章。首版范围和禁止的安全/性能主张仍按原文 §三、§十、§十一。

## 三、实施安排与来源冲突

本轮执行采用维护者批准的批次 00–05 分支和每批验收后确认节奏，替代旧单人规划的 00–12 排期；本台账保留旧 Q 编号，不另造 Q 系列。子 agent 不操作 Git，由主 agent 对每个独立切片完成测试、diff 审查和中文提交。同步远端只允许快进；合并、推送等待批次确认。

原文 §五及批准计划列运行依赖 `numpy`、`cryptography`，而后续提供的 AGENTS 写“只有 cryptography”；批准计划指定 Python 3.13 CI，AGENTS 记载本机 3.14.6。二者不是同一层面的事实：依赖规则冲突需主 agent 按当前授权核对，版本必须实际检查；不能把 AGENTS 中的 24 测试、2031 事件等历史数字当作本轮基线实测。此差异不改写来源转录，不据此新增业务规则。
