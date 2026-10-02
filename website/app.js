"use strict";

const zh = {
  downloadNativeData:"原生产品证据包",downloadNativeDataDesc:"原始签名运行、公钥、配对结果、失败、审计视频与模型/姿态数据",downloadNativeSource:"产品源码快照",downloadNativeSourceDesc:"CLI、App、网关、运行脚本、网页代码与保存的协议",downloadNativePage:"便携项目页",downloadNativePageDesc:"双语界面、交互任务网格、原回放记录与双视角影片",nativeAssetIndex:"原生产品下载索引与 SHA-256",
  nativeVideoScope:"所选任务的独立视频播放器 · 两个 720×720 展示视角、20 fps。字幕显示保存的动作、授权、整次运行游标与实际结果。回放时逐步核对了每个动作、已签名的 360×360 观测哈希及当前步结果。影片不增加基准样本，失败序列属于事后诊断。视频播放器与三维控制各自独立。",
  nativeEvidenceKicker:"实际写入边界上的测量",
  nativeEvidenceTitle:"保留原请求，拒绝无效权限，留下可核验的证据。",
  nativeEvidenceIntro:"40 对 LIBERO 任务，两组均使用官方 SmolVLA 实际推理，GPU 为 RTX 4090 D（24 GB）。所有计划样本与失败均保留。",
  nativeMetricOutcome:"直接执行与主动授权的任务成功次数一致",
  nativeMetricBytes:"动作字节保持一致，40/40 条完整配对序列相同",
  nativeMetricFaults:"无效授权全部拒绝，越权写入器调用为零",
  nativeMetricCost:"授权加写入前准入的均值；P95 为 1.040 ms",
  nativeEvidenceDetails:"查看六类拒绝记录与所有计划任务格",
  faultCondition:"注入条件",
  faultReason:"保存的拒绝原因",
  faultBlocked:"阻断 / 尝试",
  faultWriter:"越权写入器调用",
  nativeEvidenceScope:"两组保留相同的七个任务失败，固定网格披露四个先前诊断格。60 次故障来自独立的合成软件故障组，过期通过注入越过期限的时间戳实现。网关计时不含推理、env.step 与记录开销。20 Hz 仿真与回放时钟不是墙钟实时保证。这里评价执行接入，不表示策略提升或已安装竞品的性能优势。",
  nativeEvidenceReport:"完整结果、失败与计时边界",
  nativeEvidenceDownload:"签名证据包、独立公钥与视频",
  nativeEvidenceData:"精确对照 JSON",
  faultReplaced:"授权后动作被替换",
  faultContext:"执行上下文变化",
  faultExpired:"许可过期",
  faultReplay:"已消费许可重放",
  faultFeedback:"反馈身份变化",
  faultGeneration:"旧授权代次被撤销",
  replayIntro:"查看最终候选动作，再比较两种执行判定。",
  modelGeometry:"原仿真网格",
  viewOrbit:"三维",
  viewSide:"侧视",
  viewTop:"俯视",
  obstacleLabel:"障碍物 A",obstacleLabelB:"障碍物 B",
  wristLabel:"手腕 Wrist 3",
  previousFrame:"上一帧",
  nextFrame:"下一帧",
  speedLabel:"速度",
  showTrace:"手腕路径",
  downloadGeometry:"模型网格与姿态来源",
  collisionExplanation:"父计划校验后，动作后缀发生了变化。最终运动进入固定障碍物；只检查父计划仍会放行。",
  safeExplanation:"保存的运动从两个固定障碍物之间通过。独立整轨迹复核记为安全，最终计划校验允许执行。",
  rejectedNote:"最终判定：拒绝。播放候选动作可以查看运动原因；这条被拒绝的动作未下发执行。",
  acceptedNote:"最终判定：在这个仿真案例中放行。保存的单个案例不能证明普遍安全。",
  counterfactualMotion:"被拒绝动作 · 反事实回放",
  acceptedMotion:"获准计划 · 仿真回放",
  webglError:"三维渲染暂时无法使用，请查看上方保存的仿真影片。",
  confirmationTitle:"全任务新初态确认：93/100",
  confirmationBody:"提前冻结源码的100次全任务确认成功93次，描述性95% Wilson区间86.25%–96.57%，零次崩溃。七个280步失败全部保留：抽屉任务四次，任务3、5、8各一次。11,592组官方逐步结果与未改动的仿真动作均经独立复核。这是官方原生Panda模型的评价，与SO100 overlay及Sentinel干预收益分开。",
  confirmationScope:"原58/100与新93/100的初态、完整仿真版本和执行间隔不同，两个百分比独立报告，不能作为配对提升。100次固定分母保留每个失败；抽屉任务仍只有6/10成功。",
  figureHint:"点击研究图表可查看原尺寸版本；每张图下方说明数量与证据范围。",
  backendCaption:"先冻结协议再运行的 40 次新状态闭环：任务 5 在 MuJoCo 3.8.1 为 0/10，在 3.3.7 为 7/10；对照两边均为 10/10。主端点配对描述性 95% 区间为 +40–100 个百分点。处理改变完整仿真版本，不能归因为单一重置机制，也不证明物理精度或模型总体成功率。",
  failureCaption:"原始 task5/state0 失败 · 精确重放 280 个动作 · 720p 字幕画布 · 14.05 秒。初始状态、相机和每个动作与原 trace 一致；目标碗体中心最高仅上升 1.674 mm，奖励一直为零。事后诊断，排除在基准样本之外；距离指标使用物体中心和观测 EEF 位置。",
  diagnosticReplaySummary:"查看失败诊断：版本对照",
  backendReplayCaption:"任务 5 · 初始状态索引 21 · seed 43022 · 原正式动作精确回放。MuJoCo 3.8.1 执行 280 步后失败、奖励为 0；MuJoCo 3.3.7 执行 85 步后成功、奖励为 1。两个 backend 各自保留正式记录中的相机哈希。成功侧随后仅为同步影片而停留终态画面 195 帧（9.75 秒）；该停留不增加动作、物理步、奖励或基准样本。",
  horizonCaption:"40 次新配对闭环：任务 5 仅从 0/10 到 1/10，对照从 8/10 到 10/10。配对区间包含零，频繁推理尚未解决严重失败。",resetSummary:"仿真版本是否在第一步动作之前就改变任务？",resetBody:"40 次无策略重置，固定任务文件、资产和配对 seed，对比 MuJoCo 3.8.1 与 3.3.7。任务 5 的碗位置在全部十对中相差 35.327 mm，对照平均相差 0.688 mm。这确认初态对版本敏感，尚不能解释策略失败原因；旧版本用于基准兼容性对照，不说明其物理更准确。",
  statPhysical:"慢速完成 / 分层接触物理样本",statNative:"官方 SmolVLA 成功 / 声明兼容配置的新初态",
  downloadPaperData:"冻结设计的实验资料",downloadPaperDataDesc:"原始样本、动作 trace、配对结果、媒体与来源",downloadPaperSource:"实验源码与协议",downloadPaperSourceDesc:"运行脚本、冻结设计、复现命令与完整报告",paperIndex:"论文实验文件索引与 SHA-256",
  liberoVideoCaption:"官方 SmolVLA · 原生 LIBERO/Panda · 预先指定 task0/state0，83 步成功。360×360 原始 RGB 放在 1080p 字幕画布上，按真实 20 Hz 时钟播放。原始 MuJoCo3.8.1、执行间隔50的研究成功58/100，观察器不改动作；新兼容配置在下方独立报告。",
  liberoTitle:"原始原生配置：58/100",
  liberoBody:"原始 MuJoCo3.8.1、每50步重新观察的配置下，官方 SmolVLA LIBERO 模型在100个固定任务中成功58次，描述性 95% Wilson 区间为 48.21%–67.20%。小盅上的黑碗任务失败 10/10；42 次失败全部达到 280 步上限。17,868 组后处理到 env.step 动作未被改变。这是官方匹配模型与记录路径的评估，和 SO100 overlay、干预收益分开报告。",
  broadBody:"另一份冻结协议覆盖 27 组、324 个新物理样本。固定 4.8 秒完成 324/324，零次观测到风险或掉落；1.6 秒完成 184/324、掉落 78 次。下界规则始终等于固定 4.8 秒，耗时三倍。描述性 95% Wilson 零事件上界：整体 1.17%，单组 n=12 仍为 24.25%，不能据此认证连续参数域。",
  frictionVideoCaption:"相同初态、两种时长 · 1080p · 10.2 秒。1.6 秒动作打滑并掉落，4.8 秒完成运输，耗时为三倍。曲线展示载荷位移与风险阈值，画面来自真实保存的 MuJoCo 接触状态。",
  paperStudy:"先冻结方案，再比较安全与任务效用",
  paperBody:"协议与源码哈希在正式运行前提交。声明摩擦下界时，长时后备完成了 100/100 个低摩擦任务；相机不匹配时路由到不依赖相机的状态模型。质量或位移超出原支持域时，仍拒绝了 300/600 个样本。",
  paperScope:"路由完成 300/600，固定 4.8 秒完成 600/600。拒绝计为未完成。每组 100 个样本中零次观测不安全事件，对应 Wilson 上界 3.70%。可信配置声明不等于测出了摩擦。",
  paperCost:"60 个预设 UR5e 样本中，障碍物在生成计划前独立采样；全检与增量判定一致。边际平均节省 0.996 ms，但增量 P95 更差，不主张尾延迟改善。",
  paperReport:"冻结设计、失败原因、配对结果与区间",
  navProduct:"产品",navNative:"原生回放",navOverview:"概览",navDemo:"演示",navMethod:"机制",navResults:"实验",navReproduce:"复现",
  eyebrow:"本地执行控制 · 可核验回放",headline:"在最终动作上，作出执行决定。",hero:"面向机器人策略请求的本地执行边界：绑定最终精确字节、单次授权、观察实际写入，并保留签名回放。",personalHome:"个人主页",researchOverview:"研究概览",artifacts:"数据与源码",hfModel:"SO100 模型 · Hugging Face",videoLink:"原生回放",forthcoming:"待发布",
  productKicker:"产品 / 从策略到证据",productHeadline:"从最终请求到实际反馈，一条可以检查的事务链。",productLead:"Sentinel EVC 在本地运行，位于策略集成与实际写入器之间。原生 Panda profile 把官方后处理器输出与当前反馈、上下文绑定到有期限且只能使用一次的许可；数值与 MuJoCo profile 分别提供各自有边界的检查。",
  accessCli:"安装并运行本地工作台",accessApp:"导入、检查与回放已核验记录",accessSsh:"源码绑定的远程实验",accessVla:"接入现有 LeRobot / LIBERO Python 环境",
  boundaryRequest:"最终请求",boundaryRequestBody:"后处理后的 shape · dtype · 精确字节",boundaryPermit:"单次许可",boundaryPermitBody:"反馈 · profile · 上下文 · 有效期",boundaryWriter:"实际写入器",boundaryWriterBody:"在环境调用前一刻完成准入",boundaryReplay:"签名回放",boundaryReplayBody:"submitted · accepted · observed · outcome",
  transactionTitle:"控制事务 ≠ 共享日志",transactionBody:"共享日志记录各组件报告的内容；控制事务决定这一个精确请求是否可以进入写入器。Sentinel 同时保留二者，不把一条日志本身当作执行权限。",
  adoptionKicker:"为什么在现有系统中加入 Sentinel？",adoptionTitle:"保留正在使用的 VLA 技术栈，在实际写入器前增加事务边界。",adoptionLead:"Sentinel 不替代策略推理、运动规划或可视化。它让最后一次软件交接可以检查：授权后处理后的精确请求，依据当前执行上下文单次消费，在写入器处确认，并保留可独立核验的证据。",
  adoptionExact:"授权真正将被写入的字节",adoptionExactBody:"绑定 shape、dtype 与后处理器输出的精确字节，并在环境写入器前一刻重新核对同一个请求。",adoptionPermit:"让权限绑定实时上下文",adoptionPermitBody:"短有效期、单次使用，并绑定实际反馈、profile、上下文与 generation，避免许可变成通用权限。",adoptionEvidence:"保留执行状态证据",adoptionEvidenceBody:"分开记录 submitted、accepted、observed 游标；独立核验 Ed25519 签名包；在同一回放中查看保存的动作、判定与任务网格。",adoptionSurface:"保持较小的接入面",adoptionSurfaceBody:"标准库网关与密码学核验器在现有机器学习环境旁运行。接入 Sentinel 本身不要求增加 ROS 技术栈。",
  integrationKicker:"已有系统职责对照",integrationTitle:"周边技术栈负责什么，Sentinel 增加什么。",integrationScope:"这是基于文档的原生基线职责对照，不是已安装竞品的性能基准或排名。",stackOwns:"现有栈负责",sentinelAdds:"Sentinel 增加",lerobotOwns:"策略部署、官方处理器，以及针对 reset、过期 action chunk 和 epoch 的 RTC 处理。",lerobotAdds:"授权最终生成的精确动作字节，并在捕获的写入器处执行准入检查。",moveitOwns:"物理限制、自碰撞/世界碰撞检查、力中止、轨迹拼接与下发。",moveitAdds:"为自身网关路径增加请求完整性证据；原生 Panda profile 不替代这些物理控制。",foxgloveOwns:"三维遥测可视化与记录数据回放工作流。",foxgloveAdds:"在精确请求、授权判定与观察到的写入器状态转移之间建立签名关联。",rerunOwns:"共享记录、日志与可视化。",rerunAdds:"增加单次许可语义，并在回放中对齐保存的任务网格、动作、判定与结果。",
  productInstallLink:"安装 CLI 与 App",gatewayLink:"原生 VLA 网关契约",positionLink:"机制与生态定位",so100Link:"Hugging Face 上已发布的 SO100 模型",productBoundaryScope:"原生 Panda profile 只建立进程内请求完整性，不证明无碰撞、WorldGuard 支持、传感器真实、物理制动或任务完成。物理与数值验证使用独立声明的 profile。",
  demoKicker:"01 / 仿真演示",demoTitle:"计划变了，校验结论也应重新判断。",videoCaption:"UR5e 全网格 MuJoCo 回放 · 1080p · 19.95 秒。三个清晰案例：安全对照、后缀变化、场景变化；被拒绝的运动明确标为反事实且未下发。",mediaSource:"媒体来源与哈希",
  overviewKicker:"02 / 项目概览",overviewTitle:"把检查放在真正提交动作的地方。",abstract1:"机器人策略生成的动作块，在执行前可能经过重定时、修复、坐标转换或拼接。原计划上的判定不会自动覆盖变换后的运动。Sentinel EVC 将校验结论绑定到最终动作和当前执行上下文。",abstract2:"本地研究工作台包含最终计划校验、带完整回退的受限证书复用、控制器单写者与签名证据。独立研究 profile 分别评估 UR5e 运动、WorldGuard 后果预测，以及 SO100 记录数据上的 SmolVLA 动作重建。",statArm:"误放行 / 139 个不安全的 UR5e 构造 roots",statSmol:"标准化动作 MAE 降幅 / 五个留出 episodes",statWG:"新 MuJoCo roots / 六类分布变化场景",
  methodKicker:"03 / 核心机制",methodTitle:"让判定对应真正要执行的动作。",pipe1:"提出动作",pipe1s:"策略动作块",pipe2:"生成变换",pipe2s:"重定时 · 修复 · 拼接",pipe3:"最终复核",pipe3s:"最终计划 + 上下文",pipe4:"执行许可",pipe4s:"单次使用 · 有效期",pipe5:"提交执行",pipe5s:"控制器单写者",pipe6:"验证证据",pipe6s:"签名事件记录",bindingTitle:"动作与上下文绑定。",bindingBody:"计划、模型/profile 身份和当前上下文共同确定判定的适用范围。修改的后缀、变化的场景或过期的观测不能继承无关结论。",fallbackTitle:"复用有明确边界。",fallbackBody:"UR5e 研究只复用精确匹配的静态前缀。动态校验始终从第零帧开始；父记录无效时，执行完整校验。",evidenceTitle:"执行留下可核查的记录。",evidenceBody:"submitted、accepted、observed 三种状态分别记录。撤销阻止旧代次新增提交；事件与哈希支持独立证据验证。",methodScope:"产品核心支持受限数值与固定接触 profile。训练后的神经模型和 UR5e 研究属于独立实验 profile；真机运动写入适配器尚未启用。",designTrace:"查看机制实现与设计对应关系",
  replayKicker:"05 / 旧版 UR5E 记录轨迹三维回放",replayTitle:"查看判定背后的实际运动。",caseCollision:"01 · 后缀变换",caseSafe:"02 · 窄道通过",orbitHelp:"拖动旋转 · 滚轮缩放",resetView:"重置视角",play:"播放",pause:"暂停",replayTime:"回放时间",recordedCase:"保存的案例",parentVerdict:"只校验父计划",finalVerdict:"最终计划校验",recordedOutcome:"整条轨迹复核",armLegend:"UR5e 原始网格模型",obstacleLegend:"固定球形障碍物",traceLegend:"保存的手腕路径",loadingReplay:"正在加载保存的轨迹…",downloadReplay:"下载轨迹与源文件哈希",replayScope:"原始 UR5e 可视网格，每例使用 41 帧保存的关节状态进行前向计算。不重新积分动力学，也不插值姿态。被拒绝动作展示为反事实回放，未下发执行。两个案例在复核后选取；碰撞标签对应整条轨迹，不表示当前帧。蓝色路径是 Wrist 3 的轨迹，不是安装工具的末端路径。UR5e 模型来自 MuJoCo Menagerie，采用 BSD-3-Clause 许可。",lateCase:"后缀变换",safeCase:"窄道案例",allowed:"放行",denied:"拒绝",collision:"碰撞",safe:"安全",replayReady:"41 帧原始姿态 · 米 / 弧度 / 秒",replayError:"轨迹暂时无法加载，可下载原始 JSON 查看。",
  nativeReplayKicker:"产品演示 / 原生 PANDA 记录",nativeReplayTitle:"在同一时间轴查看策略动作、授权判定和观测执行。",nativeReplayIntro:"选择一条保存的 SmolVLA/LIBERO Episode。保存的 Panda 任务网格、策略动作、授权判定和控制器游标在同一条 20 Hz 保存时间轴上推进。",nativeInstructionLabel:"当前任务指令",nativeTerminalLabel:"Episode 结果",nativeTaskLabel:"保存的任务",nativeGeometryLabel:"保存的任务网格 · 为便于观察隐藏封闭房间外壳",nativePoseBadge:"物理姿态记录",nativeNoInterpolation:"不插值",nativeLoading:"正在加载保存的原生回放…",nativeOrbitHelp:"拖动旋转 · 滚轮缩放 · 方向键旋转",nativeTimelineLabel:"保存的指令时间轴",nativeCommandKicker:"保存的指令与授权",nativeActionLabel:"策略动作",nativeActionHelp:"7 个模型输出值：相对平移 Δx/y/z、相对旋转 Δrx/ry/rz 与夹爪指令。它们是模型动作单位，不是实测 SI 速度。",nativeDecisionLabel:"请求完整性授权",nativeReasonLabel:"记录原因",nativeCursorsLabel:"整次运行游标 submitted / accepted / observed",nativeStateLabel:"保存的状态摘要",nativeOutcomeLabel:"当前步结果",nativeRawLabel:"原始保存哈希与结果字段",nativeAttemptsTitle:"保存的故障授权尝试",nativeReplayScope:"查看器只绘制保存的 MuJoCo body 姿态，不运行新动力学，也不插值缺失姿态。permit 记录该次请求的指令完整性与授权，不等同于碰撞校验。被拒绝的故障尝试只展示其真实保存记录，绝不会生成虚构机器人运动。源证据包在发布前已完成核验；当前浏览器页面不执行签名验证。请使用保存的公钥独立核验下载的证据包。",
  resultsKicker:"06 / 实验对比",resultsTitle:"比较决策、误差与实际代价。",armStudy:"UR5e 最终计划校验",armStudyMeta:"180 个构造 roots · 139 个不安全 · 41 个安全",armStudyBody:"六类场景在同一组 roots 上比较四种策略。独立复核器使用更密集的静态采样和 1 ms MuJoCo 回放，不导入 gate runner。",falseAllows:"误放行",falseRejects:"误拒绝",cost:"实测代价",armScope:"106 次静态前缀复用，74 次完整回退。增量路径计入父计划后为 147.65 ms，完整校验为 48.74 ms，不主张提速。这些数据来自构造案例上的固定顺序观测，未建立自然分布留出或连续碰撞证明。",armReport:"方法、分母与代价统计",parentOnly:"只校验父计划",full:"最终计划完整校验",incremental:"增量校验 + 回退",conservative:"拒绝所有变换",faCaption:"误放行 / 139 个不安全 roots · 越低越好",frCaption:"误拒绝 / 41 个安全 roots · 越低越好",costCaption:"平均端到端校验墙钟，包含父计划成本 · ms",
  smolStudy:"SO100 记录动作上的 SmolVLA",smolMeta:"五个留出 episodes · 1,926 个重叠窗口",smolBody:"冻结视觉语言骨干，微调动作专家。固定基础模型与开发集选中的 overlay 使用相同采样噪声，进行一次配对留出评估。",smolScope:"衡量动作重建，不是真机任务成功率。窗口之间重叠，独立单位是 episode；数据集未声明原生物理单位。",smolMetric:"动作 MAE / 训练动作标准差",smolImprovement:"标准化 MAE 降幅",modelCard:"模型卡、检查点与加载方法",base:"固定基础模型",fine:"微调 overlay",
  wgStudy:"分布变化下的 WorldGuard",scenario:"场景",nominal:"标称分布",cameraShift:"相机变化",lowFriction:"低摩擦",massLow:"低载荷质量",massHigh:"高载荷质量",displacementShift:"未支持的位移",wgBody:"600 个新 roots、六类场景，每个 root 有四个同胞计划。状态和视觉 ensemble 沿用原始校准，困难案例保留在对比中。",wgMetric:"不安全选择 / 100 个 roots · 越低越好",wgReport:"完整场景结果与不确定性",fastest:"最快几何候选",fixed:"固定 1.6 s",state:"状态 WorldGuard",visual:"视觉 WorldGuard",wgNominal:"两个学习 gate 在这 100 个标称 roots 中均未观测到不安全选择；零观测不代表普遍零风险。",wgCamera:"视觉 gate 产生 7/100 个不安全选择，XY 联合覆盖率为 59%；相机 profile 变化不会被原始校准自动覆盖。",wgFriction:"四个原始候选全部不安全。两个学习 gate 均选出不安全动作，表明原动作族没有可用的安全替代。",wgMass:"隐藏载荷质量超出训练范围。这组数据中状态与视觉 gate 均为 0/100 个不安全选择；结论仅适用于记录的场景。",wgUnknown:"1.35× 位移超出已训练动作族。产品 contract 以 MODEL_UNKNOWN 拒绝全部 100 个 roots；没有动作被选中，拒绝不计为任务成功。",unsupported:"超出受支持的动作 contract",
  fallbackKicker:"MuJoCo 物理 profile 降级方案",fallbackStudy:"原候选都失败时，评估新的动作族。",fallbackStudyBody:"独立的 5 s MuJoCo profile 使用 100 个新低摩擦 roots。4.8 s 候选为 0/100 个不安全选择、100/100 个托盘终点任务，运动时长是 1.6 s 的三倍。",fallbackScope:"规则假定 μ_min = 0.015，系统未测量摩擦。3.2 s 候选在这组数据中也全部安全，但被保守规则拒绝。零次失败的 Wilson 上界为 3.70%。该 profile 尚未部署，也不是对原神经 gate 的修复。",fallbackReport:"降级规则与保留结果",hardware:"GPU 实测使用 RTX 4090 D（24 GB）。全部图表取自保存的结果文件，证据下载包含源文件身份与哈希。",downloadEvidence:"下载图表数据与来源身份",
  workbenchKicker:"APP / 本地工作台",workbenchTitle:"本地运行，查看实际证据。",productTitle:"CLI、桌面和 SSH 入口",productBody:"使用独立选择的公钥导入签名原生运行；选择任务，检查真实网格、动作与授权，并导出原始证据。App 与 CLI 共用本地引擎。",product1:"OBJ、STL、MJCF、URDF 检查",product2:"严格 SSH 实验与源码绑定结果",product3:"机械臂只读诊断",product4:"独立证据验证",productScope:"本地开发者预览版。实际 App 已导入全部 40 条正式运行，导出 ZIP 再次独立核验通过。App 分发签名与公证仍待完善，真机接口仍为只读。",installGuide:"安装方式与支持的能力",
  reproduceKicker:"08 / 复现",reproduceTitle:"从源码到可验证的运行记录。",quickStart:"本地快速开始",copy:"复制",copied:"已复制",copyFailed:"请选择文本复制",reproduceBody:"使用 Python 3.10+ 从源码运行，或通过交互安装器选择 core/physics profile 并构建 macOS App。CLI 输出本地工作台地址，仿真保存实际结果与证据。",trainingEnv:"GPU 训练使用独立的固定环境。模型 overlay 需要原始冻结的 SmolVLA base/backbone，不是独立策略检查点。",trainingGuide:"训练、校准与评估命令",replaySourceNote:"原仿真目录校验包含下载缓存；诊断回放源码逐文件绑定官方模型内容。",replaySourceLink:"查看回放复现说明",downloadResults:"仿真数据与结果",downloadResultsDesc:"保留 roots、模型、校准、预测和日志",downloadArm:"UR5e 演示包",downloadArmDesc:"便携官方模型、保存的轨迹和视频",downloadModel:"SmolVLA 可训练 overlay",downloadModelDesc:"选中权重、已校验加载器和模型卡",downloadData:"固定 SO100 数据集",downloadDataDesc:"50 个记录 episodes、两个相机与来源身份",licenseNote:"发布索引包含文件大小与 SHA-256，归档包含逐文件身份与许可。Sentinel 代码采用 MIT；MIT 不授予专利权。上游来源和核心机制专利说明见仓库。",artifactIndex:"文件索引与校验和",
  citationKicker:"09 / 引用",citationTitle:"引用研究软件与仿真产物。",citationBody:"当前软件和仿真资料可引用项目仓库。论文引用将在 arXiv 记录公开后补入。",footer:"开放代码 · 保存的实验 · 可复现的证据"
};
const english = new Map();
document.querySelectorAll("[data-i18n]").forEach(el => english.set(el.dataset.i18n, el.textContent));
let locale = "en";
try { if (localStorage.getItem("sentinel-language") === "zh") locale = "zh"; } catch {}
let evidence = null, replay = null, metric = "false_allow", caseIndex = 0, frameIndex = 0, playing = false;
let viewer = null, animation = 0, playbackStart = 0, playbackFrame = 0;
const canvas = document.getElementById("replayCanvas");

const tr = (key, fallback) => locale === "zh" ? (zh[key] || fallback) : (english.get(key) || fallback);
const setText = (id, value) => { document.getElementById(id).textContent = value; };

function languageUpdate() {
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  document.body.lang = locale;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = tr(el.dataset.i18n, el.textContent); });
  const button = document.getElementById("language");
  button.textContent = locale === "zh" ? "English" : "中文";
  button.setAttribute("aria-label", locale === "zh" ? "Switch to English" : "切换到中文");
  document.title = locale === "zh" ? "Sentinel EVC — 最终机器人动作校验" : "Sentinel EVC — Verify the Final Robot Action";
  if (evidence) { renderArm(); renderSmol(); renderWorldguard(); }
  if (replay) { updateReplay(); }
  window.dispatchEvent(new CustomEvent("sentinel-language",{detail:{locale}}));
}

function bars(target, rows, maximum) {
  const container = document.getElementById(target);
  container.replaceChildren();
  rows.forEach(row => {
    const el = document.createElement("div"); el.className = "bar-row " + (row.kind || "");
    const label = document.createElement("span"); label.className = "bar-name"; label.textContent = row.name;
    const track = document.createElement("div"); track.className = "bar-track"; track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div"); fill.className = "bar-fill"; fill.style.width = Math.max(0, Math.min(100, row.value / maximum * 100)) + "%";
    const value = document.createElement("span"); value.className = "bar-value"; value.textContent = row.display;
    track.append(fill); el.append(label, track, value); container.append(el);
  });
}

function renderArm() {
  const names = {parent_only:tr("parentOnly","Parent only"),full:tr("full","Full final-plan check"),incremental:tr("incremental","Incremental + fallback"),conservative:tr("conservative","Reject all transforms")};
  const rows = ["parent_only","full","incremental","conservative"].map(key => {
    const m = evidence.ur5e.methods[key];
    const value = metric === "cost" ? m.mean_ms : metric === "false_allow" ? m.false_allow_count : m.false_reject_count;
    const denom = metric === "false_allow" ? evidence.ur5e.unsafe : evidence.ur5e.safe;
    return {name:names[key],value,display:metric === "cost" ? value.toFixed(2) + " ms" : value + "/" + denom,kind:key === "parent_only" ? "negative" : key === "conservative" ? "muted" : ""};
  });
  setText("armMetricLabel", metric === "cost" ? tr("costCaption","Mean end-to-end gate cost, including parent validation · ms") : metric === "false_allow" ? tr("faCaption","False allows / 139 unsafe roots · lower is better") : tr("frCaption","False rejects / 41 safe roots · lower is better"));
  bars("armChart", rows, metric === "cost" ? 150 : metric === "false_allow" ? evidence.ur5e.unsafe : evidence.ur5e.safe);
}

function renderSmol() {
  const values = evidence.smolvla.normalized_mae;
  bars("smolChart", [{name:tr("base","Pinned base"),value:values.base,display:values.base.toFixed(6),kind:"muted"},{name:tr("fine","Fine-tuned overlay"),value:values.fine,display:values.fine.toFixed(6)}], values.base);
}

function renderWorldguard() {
  const selected = document.getElementById("wgScenario").value;
  const scenario = evidence.worldguard[selected];
  if (selected === "displacement_shift") {
    const notice = document.createElement("p"); notice.className = "hardware-note";
    notice.textContent = "MODEL_UNKNOWN · " + scenario.product_contract.rejected + "/" + scenario.roots;
    document.getElementById("wgChart").replaceChildren(notice);
    setText("wgMetricLabel",tr("unsupported","Outside the supported action contract"));
    setText("wgNote",tr("wgUnknown","The 1.35× displacement is outside the trained action family. The product contract rejects all 100 roots as MODEL_UNKNOWN. No action is selected; rejection is not counted as task success."));
    return;
  }
  const names = {fastest:tr("fastest","Fastest geometry candidate"),fixed:tr("fixed","Fixed 1.6 s"),state:tr("state","State WorldGuard"),visual:tr("visual","Visual WorldGuard")};
  bars("wgChart",Object.keys(names).map(key=>({name:names[key],value:scenario.unsafe[key],display:scenario.unsafe[key]+"/"+scenario.roots,kind:key==="fastest"?"muted":scenario.unsafe[key]>0?"negative":""})),scenario.roots);
  setText("wgMetricLabel",tr("wgMetric","Unsafe selections / 100 roots · lower is better"));
  const notes = {
    nominal:tr("wgNominal","Both learned gates observe 0/100 unsafe selections on these nominal roots. Zero observations do not establish general zero risk."),
    camera_shift:tr("wgCamera","The visual gate allows 7/100 unsafe selections and its joint XY coverage falls to 59%. Camera-profile changes are not automatically covered by the original calibration."),
    low_friction:tr("wgFriction","All four original candidates are unsafe. Both learned gates select an unsafe plan; the original action family offers no safe alternative."),
    mass_low:tr("wgMass","Hidden payload mass lies outside the training range. Both learned gates have 0/100 unsafe selections in this recorded sample; this does not establish arbitrary-physics robustness."),
    mass_high:tr("wgMass","Hidden payload mass lies outside the training range. Both learned gates have 0/100 unsafe selections in this recorded sample; this does not establish arbitrary-physics robustness.")
  };
  setText("wgNote",notes[selected]);
}

function drawReplay() {
  if (!replay || !viewer) return;
  viewer.draw(frameIndex,document.getElementById("showTrace").checked);
  const record=replay.cases[caseIndex],frame=record.frames[frameIndex];
  for (const [id,point] of [["obstacleLabel",record.obstacles[0].centerWorldM],["obstacleLabelB",record.obstacles[1].centerWorldM],["wristLabel",frame.bodyWorldPositionM[6]]]) {
    const el=document.getElementById(id),p=viewer.project(point);
    el.hidden=!p?.visible;
    if(p?.visible){el.style.left=(p.x*100)+"%";el.style.top=(p.y*100)+"%";}
  }
}

function updateReplay() {
  const c=replay.cases[caseIndex],frame=c.frames[frameIndex];
  setText("caseName",caseIndex===0?tr("lateCase","Changed suffix"):tr("safeCase","Narrow passage"));
  setText("caseId",locale==="zh"?"Root "+c.rootId+" · 41 帧原始记录":"Root "+c.rootId+" · 41 saved frames");
  [ ["parentDecision",c.savedOutcome.gateDecisionsAllow.parent_only], ["finalDecision",c.savedOutcome.gateDecisionsAllow.full] ].forEach(([id,allowed])=>{const el=document.getElementById(id);el.textContent=allowed?tr("allowed","Allow"):tr("denied","Deny");el.className=allowed?"verdict-allowed":"verdict-denied";});
  const unsafe=c.savedOutcome.reviewUnsafe;
  setText("caseExplanation",caseIndex===0?tr("collisionExplanation","The suffix changed after the parent plan was checked. The final motion reaches the fixed obstacle; checking only the parent still allows it."):tr("safeExplanation","This saved motion passes between two fixed obstacles. The independent whole-trajectory review records it as safe; the final-plan gate allows it."));
  setText("decisionNote",unsafe?tr("rejectedNote","Final decision: reject. Play the candidate to inspect why; this rejected motion was not dispatched."):tr("acceptedNote","Final decision: allow in this simulation case. This is a recorded example, not a general safety guarantee."));
  const badge=document.getElementById("motionType");badge.textContent=unsafe?tr("counterfactualMotion","Rejected motion · counterfactual"):tr("acceptedMotion","Accepted plan · simulation replay");badge.classList.toggle("is-safe",!unsafe);
  setText("frameCount",String(frameIndex+1).padStart(2,"0")+" / "+c.frames.length);
  document.getElementById("replayTime").setAttribute("aria-valuetext",frame.timeSeconds.toFixed(2)+" s");
  document.getElementById("previousFrame").disabled=frameIndex===0;
  document.getElementById("nextFrame").disabled=frameIndex===c.frames.length-1;
  setText("caseOutcome",unsafe?tr("collision","Collision"):tr("safe","Safe"));
  document.getElementById("caseOutcome").className=unsafe?"verdict-denied":"verdict-allowed";
  setText("replayClock",frame.timeSeconds.toFixed(2)+" / "+c.frames.at(-1).timeSeconds.toFixed(2)+" s");
  document.getElementById("replayTime").value=String(frameIndex);
  setText("playReplay",playing?tr("pause","Pause"):tr("play","Play"));
  setText("replayStatus",tr("replayReady","41 saved frames · meters / radians / seconds"));
  drawReplay();
}

function stopPlayback(){playing=false;cancelAnimationFrame(animation);if(replay)updateReplay();}
function tick(now){if(!playing)return;const count=replay.cases[caseIndex].frames.length;const next=Math.min(count-1,playbackFrame+Math.floor((now-playbackStart)*Number(document.getElementById("replaySpeed").value)/(replay.sampling.frameDtSeconds*1000)));if(next!==frameIndex){frameIndex=next;updateReplay();}if(frameIndex===count-1){stopPlayback();return;}animation=requestAnimationFrame(tick);}
document.getElementById("playReplay").addEventListener("click",()=>{if(!replay)return;if(playing){stopPlayback();return;}if(frameIndex===40)frameIndex=0;playing=true;playbackFrame=frameIndex;playbackStart=performance.now();animation=requestAnimationFrame(tick);updateReplay();});
document.getElementById("replayTime").addEventListener("input",event=>{const next=Number(event.target.value);stopPlayback();frameIndex=next;if(replay)updateReplay();});
document.querySelectorAll("[data-case]").forEach(button=>button.addEventListener("click",()=>{if(!replay)return;stopPlayback();caseIndex=Number(button.dataset.case);frameIndex=0;viewer.setCase(replay.cases[caseIndex]);document.querySelectorAll("[data-case]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));updateReplay();}));
document.querySelectorAll("[data-metric]").forEach(button=>button.addEventListener("click",()=>{metric=button.dataset.metric;document.querySelectorAll("[data-metric]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));if(evidence)renderArm();}));
document.getElementById("wgScenario").addEventListener("change",()=>{if(evidence)renderWorldguard();});
document.getElementById("language").addEventListener("click",()=>{locale=locale==="en"?"zh":"en";try{localStorage.setItem("sentinel-language",locale);}catch{}languageUpdate();});
function clearViewPreset(){document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed","false"));}
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{if(!viewer)return;viewer.preset(button.dataset.view);document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));drawReplay();}));
document.getElementById("resetView").addEventListener("click",()=>{if(!viewer)return;viewer.preset("orbit");document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view==="orbit")));drawReplay();});
document.getElementById("showTrace").addEventListener("change",drawReplay);
document.getElementById("replaySpeed").addEventListener("change",stopPlayback);
for(const [id,step] of [["previousFrame",-1],["nextFrame",1]])document.getElementById(id).addEventListener("click",()=>{if(!replay)return;stopPlayback();frameIndex=Math.max(0,Math.min(replay.cases[caseIndex].frames.length-1,frameIndex+step));updateReplay();});
let drag=null;
canvas.addEventListener("pointerdown",event=>{if(!viewer)return;drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{if(!drag||!viewer)return;clearViewPreset();viewer.autoFit=false;viewer.yaw-=(event.clientX-drag.x)*.008;viewer.pitch=Math.max(.04,Math.min(1.52,viewer.pitch+(event.clientY-drag.y)*.008));drag={x:event.clientX,y:event.clientY};drawReplay();});
["pointerup","pointercancel","lostpointercapture"].forEach(name=>canvas.addEventListener(name,()=>{drag=null;}));
canvas.addEventListener("wheel",event=>{if(!viewer)return;event.preventDefault();viewer.autoFit=false;viewer.distance=Math.max(.8,Math.min(7,viewer.distance+event.deltaY*.002));drawReplay();},{passive:false});
canvas.addEventListener("keydown",event=>{if(!viewer||!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","-","="].includes(event.key))return;event.preventDefault();clearViewPreset();viewer.autoFit=false;if(event.key==="ArrowLeft")viewer.yaw-=.1;if(event.key==="ArrowRight")viewer.yaw+=.1;if(event.key==="ArrowUp")viewer.pitch=Math.min(1.52,viewer.pitch+.08);if(event.key==="ArrowDown")viewer.pitch=Math.max(.04,viewer.pitch-.08);if(event.key==="+"||event.key==="=")viewer.distance=Math.max(.8,viewer.distance-.15);if(event.key==="-")viewer.distance=Math.min(7,viewer.distance+.15);drawReplay();});
canvas.addEventListener("webglcontextlost",event=>{event.preventDefault();stopPlayback();document.getElementById("viewerMessage").hidden=false;setText("viewerMessage",tr("webglError","3D rendering is unavailable. View the recorded simulation film above."));});
canvas.addEventListener("webglcontextrestored",()=>window.location.reload());
new ResizeObserver(()=>drawReplay()).observe(canvas);
document.addEventListener("visibilitychange",()=>{if(document.hidden)stopPlayback();});
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&playing)stopPlayback();}).observe(canvas);
document.querySelectorAll("[data-copy]").forEach(button=>button.addEventListener("click",async()=>{const text=document.getElementById(button.dataset.copy).textContent;try{await navigator.clipboard.writeText(text);button.textContent=tr("copied","Copied");}catch{const range=document.createRange();range.selectNodeContents(document.getElementById(button.dataset.copy));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent=tr("copyFailed","Select text to copy");}setTimeout(()=>{button.textContent=tr("copy","Copy");},1800);}));
languageUpdate();
fetch("assets/evidence.json").then(r=>{if(!r.ok)throw new Error("evidence");return r.json();}).then(e=>{evidence=e;languageUpdate();}).catch(()=>{setText("armMetricLabel",locale==="zh"?"图表数据暂时无法加载，请查看来源报告。":"Chart data could not load. Use the linked source reports.");});
Promise.all([fetch("assets/ur5e-trajectory-replay.json").then(r=>{if(!r.ok)throw new Error("replay");return r.json();}),fetch("assets/ur5e-viewer-model.json").then(r=>{if(!r.ok)throw new Error("geometry");return r.json();})]).then(([r,m])=>{viewer=new window.SentinelReplayViewer(canvas,m);replay=r;viewer.setCase(replay.cases[0]);document.getElementById("viewerMessage").hidden=true;document.getElementById("playReplay").disabled=false;document.getElementById("replayTime").disabled=false;languageUpdate();}).catch(()=>{setText("replayStatus",tr("replayError","Replay could not load. Download the source JSON to inspect the recorded data."));setText("viewerMessage",tr("webglError","3D rendering is unavailable. View the recorded simulation film above."));});
