"use strict";

const zh = {
  navOverview:"概览",navDemo:"演示",navMethod:"机制",navResults:"实验",navReproduce:"复现",
  eyebrow:"机器人策略校验 · 开源研究",headline:"在最终动作上，作出执行决定。",hero:"为经过变换的机器人计划设置执行门，通过一次性许可与可独立校验的证据，连接动作、决策和实际记录。",artifacts:"数据与模型",hfModel:"Hugging Face 模型",videoLink:"视频",forthcoming:"待发布",
  demoKicker:"01 / 仿真演示",demoTitle:"计划变了，校验结论也应重新判断。",videoCaption:"七段保存的 UR5e MuJoCo 案例 · 720p · 33.75 秒。案例在评估后选取，视频包含标题与最终状态停留画面。",mediaSource:"媒体来源与哈希",
  overviewKicker:"02 / 项目概览",overviewTitle:"把检查放在真正提交动作的地方。",abstract1:"机器人策略生成的动作块，在执行前可能经过重定时、修复、坐标转换或拼接。原计划上的判定不会自动覆盖变换后的运动。Sentinel EVC 将校验结论绑定到最终动作和当前执行上下文。",abstract2:"本地研究工作台包含最终计划校验、带完整回退的受限证书复用、控制器单写者与签名证据。独立研究 profile 分别评估 UR5e 运动、WorldGuard 后果预测，以及 SO100 记录数据上的 SmolVLA 动作重建。",statArm:"误放行 / 139 个不安全的 UR5e 构造 roots",statSmol:"标准化动作 MAE 降幅 / 五个留出 episodes",statWG:"新 MuJoCo roots / 六类分布变化场景",
  methodKicker:"03 / 核心机制",methodTitle:"让判定对应真正要执行的动作。",pipe1:"提出动作",pipe1s:"策略动作块",pipe2:"生成变换",pipe2s:"重定时 · 修复 · 拼接",pipe3:"最终复核",pipe3s:"最终计划 + 上下文",pipe4:"执行许可",pipe4s:"单次使用 · 有效期",pipe5:"提交执行",pipe5s:"控制器单写者",pipe6:"验证证据",pipe6s:"签名事件记录",bindingTitle:"动作与上下文绑定。",bindingBody:"计划、模型/profile 身份和当前上下文共同确定判定的适用范围。修改的后缀、变化的场景或过期的观测不能继承无关结论。",fallbackTitle:"复用有明确边界。",fallbackBody:"UR5e 研究只复用精确匹配的静态前缀。动态校验始终从第零帧开始；父记录无效时，执行完整校验。",evidenceTitle:"执行留下可核查的记录。",evidenceBody:"submitted、accepted、observed 三种状态分别记录。撤销阻止旧代次新增提交；事件与哈希支持独立证据验证。",methodScope:"产品核心支持受限数值与固定接触 profile。训练后的神经模型和 UR5e 研究属于独立实验 profile；真机运动写入适配器尚未启用。",designTrace:"查看机制实现与设计对应关系",
  replayKicker:"04 / 记录轨迹的三维回放",replayTitle:"查看判定背后的实际运动。",caseCollision:"后缀变换 · 碰撞",caseSafe:"窄道案例 · 安全",orbitHelp:"拖动旋转 · 滚轮缩放",resetView:"重置视角",play:"播放",pause:"暂停",replayTime:"回放时间",recordedCase:"保存的案例",parentVerdict:"只校验父计划",finalVerdict:"最终计划校验",recordedOutcome:"整条轨迹复核",armLegend:"UR5e 连杆原点",obstacleLegend:"固定球形障碍",traceLegend:"Wrist 3 轨迹",loadingReplay:"正在加载保存的轨迹…",downloadReplay:"下载轨迹与源文件哈希",replayScope:"保存的实际关节位置，经固定 MuJoCo 模型前向计算得到。显示连杆示意结构，不含 CAD 网格，也不重新积分动力学。这两个案例在复核后选取；标签对应整条轨迹，不表示当前帧的碰撞状态。",lateCase:"后缀变换",safeCase:"窄道案例",allowed:"放行",denied:"拒绝",collision:"碰撞",safe:"安全",replayReady:"41 帧原始记录 · 米 / 弧度 / 秒",replayError:"轨迹暂时无法加载，可下载原始 JSON 查看。",
  resultsKicker:"05 / 实验对比",resultsTitle:"比较决策、误差与实际代价。",armStudy:"UR5e 最终计划校验",armStudyMeta:"180 个构造 roots · 139 个不安全 · 41 个安全",armStudyBody:"六类场景在同一组 roots 上比较四种策略。独立复核器使用更密集的静态采样和 1 ms MuJoCo 回放，不导入 gate runner。",falseAllows:"误放行",falseRejects:"误拒绝",cost:"实测代价",armScope:"106 次静态前缀复用，74 次完整回退。增量路径计入父计划后为 147.65 ms，完整校验为 48.74 ms，不主张提速。这些数据来自构造案例上的固定顺序观测，未建立自然分布留出或连续碰撞证明。",armReport:"方法、分母与代价统计",parentOnly:"只校验父计划",full:"最终计划完整校验",incremental:"增量校验 + 回退",conservative:"拒绝所有变换",faCaption:"误放行 / 139 个不安全 roots · 越低越好",frCaption:"误拒绝 / 41 个安全 roots · 越低越好",costCaption:"平均端到端校验墙钟，包含父计划成本 · ms",
  smolStudy:"SO100 记录动作上的 SmolVLA",smolMeta:"五个留出 episodes · 1,926 个重叠窗口",smolBody:"冻结视觉语言骨干，微调动作专家。固定基础模型与开发集选中的 overlay 使用相同采样噪声，进行一次配对留出评估。",smolScope:"衡量动作重建，不是真机任务成功率。窗口之间重叠，独立单位是 episode；数据集未声明原生物理单位。",smolMetric:"动作 MAE / 训练动作标准差",smolImprovement:"标准化 MAE 降幅",modelCard:"模型卡、检查点与加载方法",base:"固定基础模型",fine:"微调 overlay",
  wgStudy:"分布变化下的 WorldGuard",scenario:"场景",nominal:"标称分布",cameraShift:"相机变化",lowFriction:"低摩擦",massLow:"低载荷质量",massHigh:"高载荷质量",displacementShift:"未支持的位移",wgBody:"600 个新 roots、六类场景，每个 root 有四个同胞计划。状态和视觉 ensemble 沿用原始校准，困难案例保留在对比中。",wgMetric:"不安全选择 / 100 个 roots · 越低越好",wgReport:"完整场景结果与不确定性",fastest:"最快几何候选",fixed:"固定 1.6 s",state:"状态 WorldGuard",visual:"视觉 WorldGuard",wgNominal:"两个学习 gate 在这 100 个标称 roots 中均未观测到不安全选择；零观测不代表普遍零风险。",wgCamera:"视觉 gate 产生 7/100 个不安全选择，XY 联合覆盖率为 59%；相机 profile 变化不会被原始校准自动覆盖。",wgFriction:"四个原始候选全部不安全。两个学习 gate 均选出不安全动作，表明原动作族没有可用的安全替代。",wgMass:"隐藏载荷质量超出训练范围。这组数据中状态与视觉 gate 均为 0/100 个不安全选择；结论仅适用于记录的场景。",wgUnknown:"1.35× 位移超出已训练动作族。产品 contract 以 MODEL_UNKNOWN 拒绝全部 100 个 roots；没有动作被选中，拒绝不计为任务成功。",unsupported:"超出受支持的动作 contract",
  fallbackKicker:"MuJoCo 物理 profile 降级方案",fallbackStudy:"原候选都失败时，评估新的动作族。",fallbackStudyBody:"独立的 5 s MuJoCo profile 使用 100 个新低摩擦 roots。4.8 s 候选为 0/100 个不安全选择、100/100 个托盘终点任务，运动时长是 1.6 s 的三倍。",fallbackScope:"规则假定 μ_min = 0.015，系统未测量摩擦。3.2 s 候选在这组数据中也全部安全，但被保守规则拒绝。零次失败的 Wilson 上界为 3.70%。该 profile 尚未部署，也不是对原神经 gate 的修复。",fallbackReport:"降级规则与保留结果",hardware:"GPU 实测使用 RTX 4090 D（24 GB）。RTX 5090 为未实测的目标配置。全部图表取自保存的结果文件，证据下载包含源文件身份与哈希。",downloadEvidence:"下载图表数据与来源身份",
  workbenchKicker:"06 / 研究工作台",workbenchTitle:"本地运行，查看实际证据。",productTitle:"CLI、桌面和 SSH 入口",productBody:"可安装 CLI 与 macOS App 共用本地引擎。运行数值和固定 MuJoCo 接触 profile，回放保存的轨迹，导出签名证据包。",product1:"OBJ、STL、MJCF、URDF 检查",product2:"严格 SSH 实验与源码绑定结果",product3:"机械臂只读诊断",product4:"独立证据验证",productScope:"研究原型。macOS App 在本地构建，未提供分发签名与公证；模型导入不会获得运动许可，真机接口仍为只读。",installGuide:"安装方式与支持的能力",
  reproduceKicker:"07 / 复现",reproduceTitle:"从源码到可验证的运行记录。",quickStart:"本地快速开始",copy:"复制",copied:"已复制",copyFailed:"请选择文本复制",reproduceBody:"使用 Python 3.10+ 从源码运行，或通过交互安装器选择 core/physics profile 并构建 macOS App。CLI 输出本地工作台地址，仿真保存实际结果与证据。",trainingEnv:"GPU 训练使用独立的固定环境。模型 overlay 需要原始冻结的 SmolVLA base/backbone，不是独立策略检查点。",trainingGuide:"训练、校准与评估命令",downloadResults:"仿真数据与结果",downloadResultsDesc:"保留 roots、模型、校准、预测和日志",downloadArm:"UR5e 演示包",downloadArmDesc:"便携官方模型、保存的轨迹和视频",downloadModel:"SmolVLA 可训练 overlay",downloadModelDesc:"选中权重、已校验加载器和模型卡",downloadData:"固定 SO100 数据集",downloadDataDesc:"50 个记录 episodes、两个相机与来源身份",licenseNote:"发布索引包含文件大小与 SHA-256，归档包含逐文件身份与许可。Sentinel 代码采用 MIT；MIT 不授予专利权。上游来源和核心机制专利说明见仓库。",artifactIndex:"文件索引与校验和",
  citationKicker:"08 / 引用",citationTitle:"引用研究软件与仿真产物。",citationBody:"当前软件和仿真资料可引用项目仓库。论文引用将在 arXiv 记录公开后补入。",footer:"开放代码 · 保存的实验 · 可复现的证据"
};
const english = new Map();
document.querySelectorAll("[data-i18n]").forEach(el => english.set(el.dataset.i18n, el.textContent));
let locale = "en";
try { if (localStorage.getItem("sentinel-language") === "zh") locale = "zh"; } catch {}
let evidence = null, replay = null, metric = "false_allow", caseIndex = 0, frameIndex = 0, playing = false;
let yaw = -.7, pitch = .38, distance = 2.35, animation = 0, playbackStart = 0, playbackFrame = 0;
const canvas = document.getElementById("replayCanvas");
const ctx = canvas.getContext("2d");
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

function project(point,w,h) {
  const x=point[0]+.06,y=point[1]-.34,z=point[2]-.34;
  const a=x*Math.cos(yaw)-y*Math.sin(yaw), b=x*Math.sin(yaw)+y*Math.cos(yaw);
  const vertical=z*Math.cos(pitch)-b*Math.sin(pitch), depth=Math.max(.25,distance-b*Math.cos(pitch)-z*Math.sin(pitch));
  const scale=Math.min(w,h)*1.28/depth;
  return {x:w*.5+a*scale,y:h*.51-vertical*scale,depth,scale};
}

function drawReplay() {
  if (!replay || !ctx) return;
  const box=canvas.getBoundingClientRect(),w=box.width,h=box.height,dpr=Math.min(window.devicePixelRatio || 1,2);
  const cw=Math.round(w*dpr),ch=Math.round(h*dpr);
  if (canvas.width!==cw || canvas.height!==ch) {canvas.width=cw;canvas.height=ch;}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  ctx.fillStyle="#f5f7fb";ctx.fillRect(0,0,w,h);
  function line(a,b,color,width=1,dash=[]) { const p=project(a,w,h),q=project(b,w,h);ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.lineWidth=width;ctx.strokeStyle=color;ctx.stroke();ctx.setLineDash([]); }
  for(let i=-6;i<=10;i++){const n=i*.1;line([n,-.6,0],[n,1,0],"#dce3ef");line([-.6,n,0],[1,n,0],"#dce3ef");}
  const c=replay.cases[caseIndex],frame=c.frames[frameIndex];
  for(let i=1;i<c.frames.length;i++)line(c.frames[i-1].bodyWorldPositionM[6],c.frames[i].bodyWorldPositionM[6],"#9baec8",1.5,[3,4]);
  const items=[];
  c.obstacles.forEach(obstacle=>{const p=project(obstacle.centerWorldM,w,h);items.push({depth:p.depth,draw(){const radius=obstacle.radiusM*p.scale;ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fillStyle="rgba(213,94,82,.72)";ctx.fill();ctx.lineWidth=1;ctx.strokeStyle="#b94e45";ctx.stroke();}});});
  for(let i=1;i<frame.bodyWorldPositionM.length;i++){const a=frame.bodyWorldPositionM[i-1],b=frame.bodyWorldPositionM[i];items.push({depth:(project(a,w,h).depth+project(b,w,h).depth)/2,draw(){line(a,b,"#436ab4",8);line(a,b,"#75a0e7",3);}});}
  items.sort((a,b)=>b.depth-a.depth).forEach(item=>item.draw());
  frame.bodyWorldPositionM.forEach((point,i)=>{const p=project(point,w,h);ctx.beginPath();ctx.arc(p.x,p.y,i===0?8:5,0,Math.PI*2);ctx.fillStyle=i===0?"#283b60":"#fff";ctx.fill();ctx.strokeStyle="#365b9e";ctx.lineWidth=2;ctx.stroke();});
  const origin=[-.48,-.26,0],axes=[[[-.30,-.26,0],"#b85a51","X"],[[-.48,-.08,0],"#4c9079","Y"],[[-.48,-.26,.18],"#487dc9","Z"]];
  axes.forEach(([tip,color,label])=>{line(origin,tip,color,2);const p=project(tip,w,h);ctx.font="12px system-ui";ctx.fillStyle=color;ctx.fillText(label,p.x+5,p.y-4);});
  ctx.font="12px system-ui";ctx.fillStyle="#76839a";ctx.fillText("UR5e · "+tr("recordedCase","RECORDED CASE")+" · m",18,25);
}

function updateReplay() {
  const c=replay.cases[caseIndex],frame=c.frames[frameIndex];
  setText("caseName",caseIndex===0?tr("lateCase","Late suffix"):tr("safeCase","Narrow passage"));
  setText("caseId",locale==="zh"?"Root "+c.rootId+" · 41 帧原始记录":"Root "+c.rootId+" · 41 saved frames");
  [ ["parentDecision",c.savedOutcome.gateDecisionsAllow.parent_only], ["finalDecision",c.savedOutcome.gateDecisionsAllow.full] ].forEach(([id,allowed])=>{const el=document.getElementById(id);el.textContent=allowed?tr("allowed","Allow"):tr("denied","Deny");el.className=allowed?"verdict-allowed":"verdict-denied";});
  const unsafe=c.savedOutcome.reviewUnsafe;
  setText("caseOutcome",unsafe?tr("collision","Collision"):tr("safe","Safe"));
  document.getElementById("caseOutcome").className=unsafe?"verdict-denied":"verdict-allowed";
  setText("replayClock",frame.timeSeconds.toFixed(2)+" / "+c.frames.at(-1).timeSeconds.toFixed(2)+" s");
  document.getElementById("replayTime").value=String(frameIndex);
  setText("playReplay",playing?tr("pause","Pause"):tr("play","Play"));
  setText("replayStatus",tr("replayReady","41 saved frames · meters / radians / seconds"));
  drawReplay();
}

function stopPlayback(){playing=false;cancelAnimationFrame(animation);if(replay)updateReplay();}
function tick(now){if(!playing)return;const count=replay.cases[caseIndex].frames.length;frameIndex=Math.min(count-1,playbackFrame+Math.floor((now-playbackStart)/50));updateReplay();if(frameIndex===count-1){stopPlayback();return;}animation=requestAnimationFrame(tick);}
document.getElementById("playReplay").addEventListener("click",()=>{if(!replay)return;if(playing){stopPlayback();return;}if(frameIndex===40)frameIndex=0;playing=true;playbackFrame=frameIndex;playbackStart=performance.now();animation=requestAnimationFrame(tick);updateReplay();});
document.getElementById("replayTime").addEventListener("input",event=>{stopPlayback();frameIndex=Number(event.target.value);if(replay)updateReplay();});
document.querySelectorAll("[data-case]").forEach(button=>button.addEventListener("click",()=>{if(!replay)return;stopPlayback();caseIndex=Number(button.dataset.case);frameIndex=0;document.querySelectorAll("[data-case]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));updateReplay();}));
document.querySelectorAll("[data-metric]").forEach(button=>button.addEventListener("click",()=>{metric=button.dataset.metric;document.querySelectorAll("[data-metric]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));if(evidence)renderArm();}));
document.getElementById("wgScenario").addEventListener("change",()=>{if(evidence)renderWorldguard();});
document.getElementById("language").addEventListener("click",()=>{locale=locale==="en"?"zh":"en";try{localStorage.setItem("sentinel-language",locale);}catch{}languageUpdate();});
document.getElementById("resetView").addEventListener("click",()=>{yaw=-.7;pitch=.38;distance=2.35;drawReplay();});
let drag=null;
canvas.addEventListener("pointerdown",event=>{drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener("pointermove",event=>{if(!drag)return;yaw+=(event.clientX-drag.x)*.008;pitch=Math.max(-.3,Math.min(1.1,pitch+(event.clientY-drag.y)*.008));drag={x:event.clientX,y:event.clientY};drawReplay();});
["pointerup","pointercancel"].forEach(name=>canvas.addEventListener(name,()=>{drag=null;}));
canvas.addEventListener("wheel",event=>{event.preventDefault();distance=Math.max(1.1,Math.min(4.5,distance+event.deltaY*.002));drawReplay();},{passive:false});
canvas.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","-","="].includes(event.key))return;event.preventDefault();if(event.key==="ArrowLeft")yaw-=.1;if(event.key==="ArrowRight")yaw+=.1;if(event.key==="ArrowUp")pitch=Math.min(1.1,pitch+.08);if(event.key==="ArrowDown")pitch=Math.max(-.3,pitch-.08);if(event.key==="+"||event.key==="=")distance=Math.max(1.1,distance-.15);if(event.key==="-")distance=Math.min(4.5,distance+.15);drawReplay();});
new ResizeObserver(()=>drawReplay()).observe(canvas);
document.addEventListener("visibilitychange",()=>{if(document.hidden)stopPlayback();});
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&playing)stopPlayback();}).observe(canvas);
document.querySelectorAll("[data-copy]").forEach(button=>button.addEventListener("click",async()=>{const text=document.getElementById(button.dataset.copy).textContent;try{await navigator.clipboard.writeText(text);button.textContent=tr("copied","Copied");}catch{const range=document.createRange();range.selectNodeContents(document.getElementById(button.dataset.copy));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent=tr("copyFailed","Select text to copy");}setTimeout(()=>{button.textContent=tr("copy","Copy");},1800);}));
languageUpdate();
Promise.all([fetch("assets/evidence.json").then(r=>{if(!r.ok)throw new Error("evidence");return r.json();}),fetch("assets/ur5e-trajectory-replay.json").then(r=>{if(!r.ok)throw new Error("replay");return r.json();})]).then(([e,r])=>{evidence=e;replay=r;document.getElementById("playReplay").disabled=false;document.getElementById("replayTime").disabled=false;languageUpdate();}).catch(()=>{setText("replayStatus",tr("replayError","Replay could not load. Download the source JSON to inspect the recorded data."));setText("armMetricLabel",locale==="zh"?"图表数据暂时无法加载，请查看来源报告。":"Chart data could not load. Use the linked source reports.");});
