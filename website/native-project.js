"use strict";

(() => {
  const MANIFEST = "assets/native-demo-index.json", INDEX_LIMIT = 512 * 1024, REPLAY_LIMIT = 64 * 1024 * 1024, MODEL_LIMIT = 192 * 1024 * 1024;
  const ui = id => document.getElementById(id);
  const canvas = ui("nativeProjectCanvas");
  let locale = document.body.lang === "zh" ? "zh" : "en";
  let manifest = null, demo = null, episode = null, replayData = null, frames = [], viewer = null;
  let frameIndex = 0, playing = false, animation = 0, playbackOrigin = 0, frameOrigin = 0, loadGeneration = 0, artifactStatusKey = "loadingIndex";

  const copy = {
    en: {
      diagnosticLabel:"Post-hoc failure diagnostic",declaredLabel:"Predeclared demo",declaredScope:"Predeclared episode: every displayed pose is retained in the signed source. The companion replay adds no benchmark samples.",diagnosticScope:"Post-hoc failure diagnostic: the source episode remains one of the seven counted failures. All 29 retained poses match; 252 additional poses are derived from the audited action replay. The diagnosis adds no benchmark samples.",
      loadingIndex:"Loading retained artifact index…", missingIndex:"Native replay assets are not included in this build. Use the source or release links below.",
      loadingReplay:"Loading retained model and replay…", ready:"Retained model and replay loaded", failed:"Native replay could not be loaded",
      pass:"PASS", reject:"REJECT", unknown:"UNKNOWN", allowed:"Accepted native action", denied:"Rejected authorization fault", noAttempts:"No fault-authorization attempt records were retained for this episode.", omitted:"additional retained fault attempts not expanded",
      success:"Task success", failure:"Task did not succeed", crashed:"Episode crashed", terminalUnknown:"Terminal result not declared",
      frame:"Frame", step:"step", permit:"permit", noAction:"No action submitted", noReason:"No recorded reason", noOutcome:"No recorded step outcome",
      eef:"EEF body xyz",qpos:"qpos[0:7]",contacts:"contacts",reward:"reward",stepSuccess:"success",terminated:"terminated",truncated:"truncated",rawUnavailable:"No raw hash summary was retained.",
      provenance:"Provenance", readme:"Replay notes", source:"Source", release:"Artifact release", bundle:"Verified source bundle", publicKey:"Public key", play:"Play", pause:"Pause",
      resetLabel:"Reset observed",resetMeaning:"Initial observation recorded; no command was submitted.",baselineLabel:"Baseline direct",baselineMeaning:"The native action went directly to env.step without a Sentinel permit.",shadowLabel:"Shadow passthrough",shadowMeaning:"Sentinel observed the request without intervening; the native action passed through unchanged.",shadowRejectLabel:"Shadow would reject",shadowRejectMeaning:"Shadow evaluation would reject, but shadow mode did not intervene and the native action still passed through unchanged.",activeLabel:"Active authorized",activeMeaning:"Sentinel authorized the integrity-bound exact request and submitted it with a retained permit; this does not claim collision validation.",blockedBeforeStep:"Blocked before env.step",envStepCalls:"env.step calls",expected:"expected",
    },
    zh: {
      diagnosticLabel:"事后失败诊断",declaredLabel:"预先选定的演示",declaredScope:"预先选定的 Episode：每个显示姿态均保存在已签名的源记录中。媒体回放不增加基准样本。",diagnosticScope:"事后失败诊断：原 Episode 仍是已计入分母的七个失败之一。29 个保存姿态全部匹配，另外 252 个姿态来自审计后的动作重放。诊断不增加基准样本。",
      loadingIndex:"正在加载保存的产物索引…", missingIndex:"当前构建未包含原生回放资产，请使用下方源码或发布链接。",
      loadingReplay:"正在加载保存的模型与回放…", ready:"保存的模型与回放已加载", failed:"原生回放无法加载",
      pass:"通过", reject:"拒绝", unknown:"未知", allowed:"已接受的原生动作", denied:"被拒绝的授权故障", noAttempts:"本 Episode 没有保存故障授权尝试记录。", omitted:"条其他故障尝试未展开",
      success:"任务成功", failure:"任务未成功", crashed:"Episode 运行崩溃", terminalUnknown:"记录未声明终态结果",
      frame:"帧", step:"步", permit:"许可", noAction:"没有提交动作", noReason:"记录未提供原因", noOutcome:"没有保存当前步结果",
      eef:"末端 body xyz",qpos:"qpos[0:7]",contacts:"接触数",reward:"奖励",stepSuccess:"成功",terminated:"终止",truncated:"截断",rawUnavailable:"没有保存原始哈希摘要。",
      provenance:"来源与哈希", readme:"回放说明", source:"源码", release:"产物发布", bundle:"已核验源证据包", publicKey:"核验公钥", play:"播放", pause:"暂停",
      resetLabel:"已观测重置",resetMeaning:"已记录初始观测，没有提交控制指令。",baselineLabel:"基线直接执行",baselineMeaning:"原生动作直接进入 env.step，没有使用 Sentinel permit。",shadowLabel:"影子旁路",shadowMeaning:"Sentinel 只观察、不干预，原生动作保持不变并继续执行。",shadowRejectLabel:"影子模式本会拒绝",shadowRejectMeaning:"影子判定本会拒绝，但影子模式不干预，原生动作仍保持不变并继续执行。",activeLabel:"主动授权执行",activeMeaning:"Sentinel 对完整性绑定的精确请求完成授权，并使用保存的 permit 提交；这不声称完成碰撞校验。",blockedBeforeStep:"在 env.step 前阻断",envStepCalls:"env.step 调用次数",expected:"预期",
    },
  };
  const t = key => copy[locale][key] || copy.en[key] || key;
  const decisions = {
    RESET_OBSERVED:{kind:"reset",label:"resetLabel",meaning:"resetMeaning"},
    BASELINE_DIRECT:{kind:"direct",label:"baselineLabel",meaning:"baselineMeaning"},
    SHADOW_PASSTHROUGH:{kind:"shadow",label:"shadowLabel",meaning:"shadowMeaning"},
    SHADOW_WOULD_REJECT_PASSTHROUGH:{kind:"shadow-reject",label:"shadowRejectLabel",meaning:"shadowRejectMeaning"},
    ACTIVE_AUTHORIZED:{kind:"pass",label:"activeLabel",meaning:"activeMeaning"},
  };

  function localAsset(path) {
    return typeof path === "string" && /^assets\/[A-Za-z0-9][A-Za-z0-9_.\/-]*$/.test(path) && !path.includes("..") && !path.includes("//");
  }

  async function readBytes(stream, limit) {
    const reader=stream.getReader(), chunks=[];let size=0;
    while(true){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>limit){await reader.cancel();throw new Error("Artifact exceeds the site playback size limit.");}chunks.push(value);}
    const output=new Uint8Array(size);let offset=0;for(const chunk of chunks){output.set(chunk,offset);offset+=chunk.byteLength;}return output;
  }

  async function digest(bytes) {
    if(!globalThis.crypto?.subtle)return null;
    const value=await globalThis.crypto.subtle.digest("SHA-256",bytes);
    return [...new Uint8Array(value)].map(item=>item.toString(16).padStart(2,"0")).join("");
  }

  async function fetchJSON(path, {limit=REPLAY_LIMIT, sha256=null, decodedSha256=null}={}) {
    if (!localAsset(path)) throw new Error("Artifact path is outside the project asset directory.");
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    const declared=Number(response.headers.get("content-length"));if(Number.isFinite(declared)&&declared>limit)throw new Error("Artifact exceeds the site playback size limit.");
    if(!response.body)throw new Error("Artifact response has no readable body.");
    const raw=await readBytes(response.body,limit), gzipPayload=raw[0]===0x1f&&raw[1]===0x8b;
    const expected=path.endsWith(".gz")&&!gzipPayload?decodedSha256:sha256;
    if(sha256&&!expected)throw new Error("Decoded artifact SHA-256 is missing from the retained index.");
    if(expected){const actual=await digest(raw);if(!actual||actual!==expected)throw new Error("Artifact SHA-256 does not match the retained index.");}
    let bytes=raw;
    if(gzipPayload){
      if(typeof DecompressionStream!=="function")throw new Error("This browser cannot decompress the retained gzip model.");
      bytes=await readBytes(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip")),MODEL_LIMIT);
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  function validateManifest(value) {
    if (!value || value.schema !== "sentinel-native-demo-index-v1" || !Array.isArray(value.demos) || !value.demos.length) throw new Error("Unsupported native demo index.");
    value.demos.forEach((item,index) => {
      if (!(typeof item.taskId === "string" || Number.isInteger(item.taskId)) || !Number.isInteger(item.stateIndex) || !localAsset(item.model) || !localAsset(item.replay)) {
        throw new Error(`Invalid demo entry ${index}.`);
      }
      if (item.video && !localAsset(item.video)) throw new Error(`Invalid video path in demo entry ${index}.`);
      if (item.poster && !localAsset(item.poster)) throw new Error(`Invalid poster path in demo entry ${index}.`);
      for(const key of ["modelSha256","modelDecodedSha256","replaySha256","sourceManifestSha256","sourceResultSha256"]){if(item[key]!==undefined&&!/^[a-f0-9]{64}$/.test(item[key]))throw new Error(`Invalid ${key} in demo entry ${index}.`);}
    });
    return value;
  }

  function instruction(value) {
    if (typeof value === "string") return value;
    if (value && typeof value === "object") return value[locale] || value.en || value.zh || "—";
    return "—";
  }

  function findEpisode(replayValue, selectedDemo) {
    const candidates = Array.isArray(replayValue.episodes) ? replayValue.episodes : replayValue.frames ? [replayValue] : [];
    return candidates.find(item => String(item.task_id ?? item.taskId) === String(selectedDemo.taskId) && Number(item.initial_state_index ?? item.stateIndex) === selectedDemo.stateIndex) || null;
  }

  function setMessage(message, isError=false) {
    ui("nativeProjectMessage").hidden = false;
    ui("nativeProjectMessage").textContent = message;
    ui("nativeProjectMessage").classList.toggle("is-error", isError);
  }

  function setArtifactState(key) { artifactStatusKey=key;ui("nativeArtifactState").textContent=t(key); }

  function setReady(ready) {
    for (const id of ["nativePlay","nativePrevious","nativeNext","nativeTimeline","nativeReset"]) ui(id).disabled = !ready;
    if (ready) ui("nativeProjectMessage").hidden = true;
  }

  function stop() {
    playing = false; cancelAnimationFrame(animation); animation = 0;
    ui("nativePlay").textContent = t("play");
  }

  function terminalText(item) {
    if (item?.crashed) return t("crashed");
    if (item?.success === true) return t("success");
    if (item?.success === false) return t("failure");
    return t("terminalUnknown");
  }

  function decisionKind(value) {
    const exact=decisions[String(value||"").toUpperCase()];if(exact)return exact.kind;
    const normalized = String(value || "").toLowerCase();
    if (["allowed","allow","accepted","pass","permit","permitted"].includes(normalized)) return "pass";
    if (["denied","deny","rejected","reject","blocked","fail"].includes(normalized)) return "reject";
    return "unknown";
  }

  function decisionInfo(value) {
    const code=String(value||"").toUpperCase(), known=decisions[code];
    if(known)return {code,kind:known.kind,label:t(known.label),meaning:t(known.meaning)};
    const kind=decisionKind(value);return {code:String(value||t("unknown")),kind,label:kind==="pass"?t("pass"):kind==="reject"?t("reject"):t("unknown"),meaning:""};
  }

  function permitSummary(frame) {
    const permit=frame.permit ?? frame.authorization?.permit ?? frame.permit_id;
    if(!permit)return "";
    if(typeof permit==="string")return `${t("permit")} ${permit}`;
    if(typeof permit!=="object")return "";
    const id=permit.permit_id ?? permit.id ?? permit.lease_id, details=[];
    if(id)details.push(String(id));if(permit.step!==undefined)details.push(`${t("step")} ${permit.step}`);if(permit.generation!==undefined)details.push(`gen ${permit.generation}`);
    return details.length?`${t("permit")} ${details.join(" · ")}`:t("permit");
  }

  function actionText(action) {
    if (!Array.isArray(action)) return t("noAction");
    if(action.length===7){
      const values=action.map(value=>Number.isFinite(value)?value.toFixed(4):String(value));
      return `Δxyz [${values.slice(0,3).join(", ")}] · Δrot [${values.slice(3,6).join(", ")}] · gripper ${values[6]}`;
    }
    return `[${action.map(value => Number.isFinite(value) ? value.toFixed(4) : String(value)).join(", ")}]`;
  }

  function scalar(value) {
    return Array.isArray(value) ? value[0] : value;
  }

  function number(value) {
    return Number.isFinite(value) ? Number(value).toFixed(4) : "—";
  }

  function stateSummary(frame) {
    const positions=frame.bodyWorldPosition, eefIndex=viewer?.traceBodyIndex;
    const eef=Array.isArray(positions)&&Number.isInteger(eefIndex)?positions[eefIndex]:null;
    const parts=[];
    if(Array.isArray(eef)&&eef.length===3)parts.push(`${t("eef")} [${eef.map(number).join(", ")}]`);
    if(Array.isArray(frame.qpos)&&frame.qpos.length)parts.push(`${t("qpos")} [${frame.qpos.slice(0,7).map(number).join(", ")}]`);
    if(Array.isArray(frame.contacts))parts.push(`${t("contacts")} ${frame.contacts.length}`);
    return parts.join(" · ")||"—";
  }

  function outcomeSummary(outcome) {
    if(!outcome||typeof outcome!=="object")return t("noOutcome");
    const info=outcome.info&&typeof outcome.info==="object"?outcome.info:{}, parts=[];
    const fields=[["reward",scalar(outcome.reward)],["stepSuccess",scalar(info.is_success)],["terminated",scalar(outcome.terminated)],["truncated",scalar(outcome.truncated)]];
    for(const [label,value] of fields)if(value!==undefined&&value!==null)parts.push(`${t(label)} ${typeof value==="number"?number(value):String(value)}`);
    return parts.join(" · ")||t("noOutcome");
  }

  function rawEvidence(frame) {
    const retained={observation_hash:frame.observation_hash,action_bytes_hash:frame.action_bytes_hash,state:frame.state,outcome:frame.outcome};
    for(const key of Object.keys(retained))if(retained[key]===undefined)delete retained[key];
    return Object.keys(retained).length?JSON.stringify(retained,null,2):t("rawUnavailable");
  }

  function attemptsForEpisode(replayValue, selectedEpisode) {
    const retained = selectedEpisode.authorization_attempts || selectedEpisode.attempts || replayValue.authorization_attempts || replayValue.attempts;
    if (Array.isArray(retained)) return retained.filter(item => item.episode_id === undefined || String(item.episode_id) === String(selectedEpisode.episode_id));
    return [];
  }

  function renderAttempts(replayValue) {
    const target = ui("nativeAttemptList"); target.replaceChildren();
    const allAttempts = attemptsForEpisode(replayValue,episode), attempts=allAttempts.slice(0,8);
    target.closest(".native-attempts").hidden = allAttempts.length === 0;
    if (!attempts.length) { const empty=document.createElement("p");empty.className="small";empty.textContent=t("noAttempts");target.append(empty);return; }
    attempts.forEach(attempt => {
      const item=document.createElement("article"), header=document.createElement("div"), title=document.createElement("strong"), meta=document.createElement("span"), reason=document.createElement("p");
      const kind=attempt.blocked===true?"reject":decisionKind(attempt.decision), cursors=attempt.cursors || {};
      item.className=`native-attempt ${kind}`;
      const fault=attempt.fault?` · ${attempt.fault}`:"";
      title.textContent=(kind === "pass" ? t("allowed") : kind === "reject" ? t("denied") : t("unknown"))+fault;
      meta.textContent=`${t("step")} ${attempt.step ?? "—"} · ${cursors.submitted ?? "—"}/${cursors.accepted ?? "—"}/${cursors.observed ?? "—"}`;
      const reasonParts=[attempt.blocked===true?t("blockedBeforeStep"):null,attempt.reason,attempt.expected_reason?`${t("expected")} ${attempt.expected_reason}`:null,attempt.env_step_calls!==undefined?`${t("envStepCalls")} ${attempt.env_step_calls}`:null].filter(Boolean);
      reason.textContent=reasonParts.join(" · ")||t("noReason");
      header.append(title,meta);item.append(header,reason);
      if(Array.isArray(attempt.action)){const action=document.createElement("code");action.textContent=actionText(attempt.action);item.append(action);}
      target.append(item);
    });
    if(allAttempts.length>attempts.length){const more=document.createElement("p");more.className="small";more.textContent=locale==="zh"?`${allAttempts.length-attempts.length} ${t("omitted")}`:`${allAttempts.length-attempts.length} ${t("omitted")}`;target.append(more);}
  }

  function renderLinks(selectedDemo) {
    const target=ui("nativeLinks");target.replaceChildren();
    for(const [key,label] of [["provenance","provenance"],["readme","readme"],["source","source"],["release","release"],["bundle","bundle"],["publicKey","publicKey"]]){
      const href=selectedDemo[key] || manifest?.[key];
      if(typeof href!=="string"||!(localAsset(href)||/^https:\/\//.test(href)))continue;
      const link=document.createElement("a");link.href=href;link.textContent=t(label);link.target="_blank";link.rel="noopener noreferrer";target.append(link);
    }
    for(const [key,label] of [["modelSha256","model"],["replaySha256","replay"],["sourceManifestSha256","source manifest"],["sourceResultSha256","source result"]]){
      const hash=selectedDemo[key];if(!hash)continue;const value=document.createElement("code");value.className="native-hash";value.title=hash;value.textContent=`${label} sha256 ${hash.slice(0,12)}…`;target.append(value);
    }
  }

  function optionText(item) { return `Task ${item.taskId} · state ${item.stateIndex} · ${t(item.purpose === "posthoc_failure_diagnostic" ? "diagnosticLabel" : "declaredLabel")} · ${instruction(item.instruction)}`; }

  function renderDynamicText() {
    if (!demo) return;
    ui("nativeInstruction").textContent=instruction(demo.instruction || episode?.instruction);
    ui("nativeTerminal").textContent=terminalText(episode);
    ui("nativeSelectionScope").textContent=t(demo.purpose === "posthoc_failure_diagnostic" ? "diagnosticScope" : "declaredScope");
    if (episode) renderFrame();
  }

  function renderFrame() {
    if (!frames.length) return;
    const frame=frames[frameIndex], cursors=frame.cursors || {}, info=decisionInfo(frame.decision), kind=info.kind;
    viewer.draw(frameIndex);
    ui("nativeTimeline").value=String(frameIndex);
    ui("nativeFrame").textContent=`${t("frame")} ${frameIndex+1}/${frames.length}`;
    ui("nativeStep").textContent=`${t("step")} ${frame.step ?? frameIndex}`;
    ui("nativeClock").textContent=`${Number.isFinite(frame.timeSeconds)?frame.timeSeconds.toFixed(2):(frameIndex/20).toFixed(2)} s`;
    ui("nativeAction").textContent=actionText(frame.action);
    ui("nativeDecision").textContent=`${info.label} · ${info.code}`;
    ui("nativeDecision").className=`native-decision-${kind}`;
    ui("nativeReason").textContent=[info.meaning,frame.reason,permitSummary(frame)].filter(Boolean).join(" · ")||t("noReason");
    ui("nativeCursors").textContent=`${cursors.submitted ?? "—"} / ${cursors.accepted ?? "—"} / ${cursors.observed ?? "—"}`;
    ui("nativeExecutorState").textContent=stateSummary(frame);
    ui("nativeOutcome").textContent=outcomeSummary(frame.outcome);
    ui("nativeRawEvidence").textContent=rawEvidence(frame);
    ui("nativePrevious").disabled=frameIndex===0;
    ui("nativeNext").disabled=frameIndex===frames.length-1;
  }

  async function selectDemo(indexValue) {
    stop(); setReady(false); viewer?.clear();
    ui("nativeVideoFigure").hidden=true;ui("nativeVideo").pause();
    const generation=++loadGeneration, index=Math.max(0,Math.min(manifest.demos.length-1,Number(indexValue)||0));
    demo=manifest.demos[index];episode=null;replayData=null;frames=[];frameIndex=0;
    ui("nativeTask").value=String(index);ui("nativeInstruction").textContent=instruction(demo.instruction);ui("nativeTerminal").textContent="—";
    setArtifactState("loadingReplay");setMessage(t("loadingReplay"));renderLinks(demo);
    try {
      const [model,replayValue]=await Promise.all([fetchJSON(demo.model,{limit:MODEL_LIMIT,sha256:demo.modelSha256,decodedSha256:demo.modelDecodedSha256}),fetchJSON(demo.replay,{limit:REPLAY_LIMIT,sha256:demo.replaySha256})]);
      if(generation!==loadGeneration)return;
      episode=findEpisode(replayValue,demo);
      if(!episode||!Array.isArray(episode.frames)||!episode.frames.length)throw new Error("The selected task/state has no retained frames.");
      frames=episode.frames;
      replayData=replayValue;
      viewer.setRecord(model,episode);
      ui("nativeTimeline").max=String(frames.length-1);ui("nativeTimeline").value="0";
      const video=ui("nativeVideo");
      if(demo.video){video.src=demo.video;video.poster=demo.poster||"";video.hidden=false;video.load();ui("nativeVideoFigure").hidden=false;}else{video.removeAttribute("src");video.hidden=true;video.load();}
      renderAttempts(replayValue);renderDynamicText();setReady(true);
      setArtifactState("ready");
    } catch(error) {
      if(generation!==loadGeneration)return;
      viewer?.clear();setMessage(`${t("failed")}: ${error.message}`,true);setArtifactState("failed");
    }
  }

  function tick(now) {
    if(!playing)return;
    const speed=Number(ui("nativeSpeed").value), startTime=Number.isFinite(frames[frameOrigin]?.timeSeconds)?frames[frameOrigin].timeSeconds:frameOrigin/20;
    const target=startTime+(now-playbackOrigin)*speed/1000;
    let next=frameOrigin;
    while(next+1<frames.length){const time=Number.isFinite(frames[next+1].timeSeconds)?frames[next+1].timeSeconds:(next+1)/20;if(time>target)break;next+=1;}
    if(next!==frameIndex){frameIndex=next;renderFrame();}
    if(frameIndex>=frames.length-1){stop();return;}
    animation=requestAnimationFrame(tick);
  }

  function playPause() {
    if(!frames.length)return;
    if(playing){stop();return;}
    if(frameIndex===frames.length-1){frameIndex=0;renderFrame();}
    playing=true;frameOrigin=frameIndex;playbackOrigin=performance.now();ui("nativePlay").textContent=t("pause");animation=requestAnimationFrame(tick);
  }

  function step(delta) { if(!frames.length)return;stop();frameIndex=Math.max(0,Math.min(frames.length-1,frameIndex+delta));renderFrame(); }

  function bind() {
    ui("nativeTask").addEventListener("change",event=>selectDemo(event.target.value));
    ui("nativePlay").addEventListener("click",playPause);
    ui("nativePrevious").addEventListener("click",()=>step(-1));ui("nativeNext").addEventListener("click",()=>step(1));
    ui("nativeTimeline").addEventListener("input",event=>{stop();frameIndex=Number(event.target.value);renderFrame();});
    ui("nativeSpeed").addEventListener("change",()=>{if(playing){stop();playPause();}});
    ui("nativeReset").addEventListener("click",()=>viewer?.reset());
    ui("nativeVideo").addEventListener("error",event=>{const video=event.currentTarget;video.hidden=true;video.removeAttribute("src");});
    canvas.addEventListener("keydown",event=>{
      if(!viewer||!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","-","="].includes(event.key))return;
      event.preventDefault();
      if(event.key==="ArrowLeft")viewer.yaw-=.1;if(event.key==="ArrowRight")viewer.yaw+=.1;
      if(event.key==="ArrowUp")viewer.pitch=Math.min(1.45,viewer.pitch+.08);if(event.key==="ArrowDown")viewer.pitch=Math.max(-1.45,viewer.pitch-.08);
      if(event.key==="+"||event.key==="=")viewer.distance=Math.max(viewer.minimumDistance||.12,viewer.distance*.88);if(event.key==="-")viewer.distance=Math.min(viewer.maximumDistance||30,viewer.distance*1.12);
      viewer.draw(frameIndex);
    });
    document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
    new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe(canvas);
    window.addEventListener("sentinel-language",event=>{
      locale=event.detail?.locale==="zh"?"zh":"en";stop();
      setArtifactState(artifactStatusKey);
      if(manifest)ui("nativeTask").querySelectorAll("option").forEach((option,index)=>{const item=manifest.demos[index];option.textContent=optionText(item);});
      renderDynamicText();if(episode&&replayData)renderAttempts(replayData);renderLinks(demo||{});
    });
  }

  async function start() {
    bind();
    try { viewer=new window.SentinelNativeViewer(canvas); viewer.clear(); }
    catch(error){setMessage(`${t("failed")}: ${error.message}`,true);return;}
    try {
      manifest=validateManifest(await fetchJSON(MANIFEST,{limit:INDEX_LIMIT}));
      ui("nativeTask").replaceChildren(...manifest.demos.map((item,index)=>{const option=document.createElement("option");option.value=String(index);option.textContent=optionText(item);return option;}));
      ui("nativeTask").disabled=false;
      await selectDemo(0);
    } catch(error) {
      setMessage(`${t("missingIndex")} (${error.message})`,true);setArtifactState("missingIndex");renderLinks({});
    }
  }

  start();
})();
