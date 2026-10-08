'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {queued:'排队',preparing:'准备',running:'运行中',stopping:'正在撤销',stopped:'已停止',completed:'已完成',rejected:'已拒绝',failed:'失败',allowed:'允许',denied:'拒绝',unknown:'未知',pending:'待实验',blocked:'待接入',implemented:'已实现',completed_numeric:'数值已验证'};
let token = '', templates = [], selected = null, current = null, eventData = [], timer = null, busy = false;
function state(value) { return `<span class="state ${escapeHTML(value)}">${escapeHTML(labels[value] || value)}</span>`; }
function notice(message) { $('notice').textContent=message; $('notice').hidden=!message; }
async function api(path,body) { const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Sentinel-Token':token},body:JSON.stringify(body)}; const response=await fetch(path,options); const value=await response.json(); if(!response.ok)throw Error(value.error?.message || '请求失败'); return value; }
function number(v,d=3) { return Number.isFinite(v)?v.toFixed(d):'—'; }
window.SentinelWorkbench = {api, escapeHTML, notice, number};
function fill(s) { for(const field of ['name','seed','displacement','risk_limit','prediction_mode'])$(field).value=s[field]; }
async function listRuns() { const {runs}=await api('/api/runs'); $('run-list').innerHTML=runs.length?runs.map(run=>`<button class="run-item ${run.id===selected?'selected':''}" data-id="${escapeHTML(run.id)}"><span>${escapeHTML(run.name)}</span>${state(run.status)}<small>${escapeHTML(new Date(run.created_at).toLocaleString())}</small></button>`).join(''):'<p class="empty">还没有运行记录。<br>创建第一条数值场景开始。</p>'; $('run-list').querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>loadRun(b.dataset.id)); }
function plot(history,limit) { if(!history.length)return '<p class="empty">等待实际控制器反馈…</p>'; const w=650,h=190,p=28,max=Math.max(limit||0,...history.map(x=>Math.abs(x.r)),.01)*1.2,total=Math.max(2,history.at(-1).time); const x=t=>p+t/total*(w-p*2),y=r=>h/2-r/max*(h/2-p);const path=history.map((v,i)=>`${i?'L':'M'}${x(v.time).toFixed(2)},${y(v.r).toFixed(2)}`).join(' '); return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="实际观测负载位移随数值时间变化"><line x1="${p}" x2="${w-p}" y1="${h/2}" y2="${h/2}"/><line class="limit" x1="${p}" x2="${w-p}" y1="${y(limit)}" y2="${y(limit)}"/><line class="limit" x1="${p}" x2="${w-p}" y1="${y(-limit)}" y2="${y(-limit)}"/><path d="${path}"/><text x="${p}" y="185">0 s</text><text x="${w-p-20}" y="185">${number(total,1)} s</text><text x="${p}" y="16">r · m</text></svg>`; }
function renderEvents() { const filter=$('event-filter')?.value || ''; const rows=eventData.filter(e=>!filter||e.type===filter); $('event-list').innerHTML=rows.length?rows.map(e=>`<div class="event"><code>#${e.seq} ${escapeHTML(e.type)}</code><span>${escapeHTML(JSON.stringify(e.payload))}</span></div>`).join(''):'<p class="empty">暂无对应事件。</p>'; }
function renderStep() { const steps=current?.result?.steps || []; const index=Number($('step-slider').value); const row=steps[index]; $('step-label').textContent=row?'第 '+row.step+' 步':'等待反馈'; $('step-data').textContent=row?JSON.stringify(row,null,2):'暂无执行记录'; }
function renderRun(run) { const oldStep=$('step-slider')?.value, oldFilter=$('event-filter')?.value || ''; current=run; const r=run.result||{}, rows=r.candidates||[], history=r.observed_history||[], cursors=r.cursors||{}, stopped=run.status==='stopped'; $('run-detail').innerHTML=`<section class="panel"><div class="panel-head"><div><h2 class="detail-title">${escapeHTML(run.name)}</h2><div class="detail-sub">${escapeHTML(run.id)}</div></div>${state(run.status)}</div>${run.error?`<div class="error-box"><strong>${escapeHTML(run.error.code)}</strong><br>${escapeHTML(run.error.message)}</div>`:''}<div class="actions"><button id="stop-run" class="danger" ${!['running','preparing','queued'].includes(run.status)?'disabled':''}>停止运行</button>${stopped?'<label class="approval"><input id="approved" type="checkbox">确认取消已完成，批准恢复</label><button id="resume-run" disabled>恢复执行</button>':''}<button id="export-run" ${!['completed','failed','rejected'].includes(run.status)?'disabled':''}>导出证据包</button><span id="download-slot"></span></div><div class="metrics"><div class="metric"><span>实际观测步骤</span><strong>${cursors.observed||0}<small>/ 40</small></strong></div><div class="metric"><span>批准候选</span><strong>${escapeHTML(r.selected?.replace('duration-','').replace('s','') || '—')}<small>${r.selected?'s':''}</small></strong></div><div class="metric"><span>主机运行时长</span><strong>${number(r.wall_seconds,2)}<small>s</small></strong></div><div class="metric"><span>证据独立核验</span><strong>${run.verification?run.verification.ok?'PASS':'FAIL':'—'}</strong></div></div></section><section class="panel"><div class="panel-head"><h2>四个最终候选</h2><span class="number">同一根状态 · 同一观察窗</span></div><div class="table-wrap"><table><thead><tr><th>转运时长</th><th>判定</th><th>几何余量</th><th>后果上界</th><th>原因 / 最终摘要</th></tr></thead><tbody>${rows.map(c=>`<tr class="${c.id===r.selected?'chosen':''}"><td>${number(c.duration,1)} s${c.id===r.selected?' · 已选':''}</td><td>${state(c.status)}</td><td>${number(c.geometry_margin)} m</td><td>${number(c.upper_peak)} m</td><td>${escapeHTML(c.reason||'通过')}<small title="${escapeHTML(c.plan_hash)}">${escapeHTML(c.plan_hash)}</small></td></tr>`).join('')||'<tr><td colspan="5">正在生成与验证候选…</td></tr>'}</tbody></table></div></section><section class="panel"><div class="panel-head"><h2>已观测负载</h2><span class="number">数值时间 · 实际反馈推进</span></div>${plot(history,run.scenario.risk_limit)}<div class="chart-note"><span>绿色：已观测位移 · 虚线：后果上限</span><span>提交 ${cursors.submitted||0} / 确认 ${cursors.accepted||0} / 观测 ${cursors.observed||0}</span></div></section><div class="split"><section class="panel"><div class="panel-head"><h2>记录回放</h2><span class="number">保存的数据</span></div><div class="timeline"><input id="step-slider" type="range" min="0" max="${Math.max(0,(r.steps||[]).length-1)}" value="${Math.min(Number(oldStep??39),Math.max(0,(r.steps||[]).length-1))}" aria-label="执行步"><span id="step-label"></span></div><pre id="step-data" class="trace"></pre></section><section class="panel"><div class="panel-head"><h2>版本与绑定</h2></div><div class="trace">模型：${escapeHTML(r.model?.id||'等待模型')}<br>校准：${escapeHTML(r.calibration?.id||'等待校准')}<br>预测：${escapeHTML(run.scenario.prediction_mode)}<br>范围：${escapeHTML(run.scope)}<br>演示签名密钥 · 非客户 PKI</div></section></div><section class="panel"><div class="panel-head"><h2>执行事件</h2><select id="event-filter" class="event-filter-control" aria-label="事件过滤"><option value="">全部事件</option>${[...new Set(eventData.map(e=>e.type))].map(t=>`<option value="${escapeHTML(t)}">${escapeHTML(t)}</option>`).join('')}</select></div><div id="event-list" class="events"></div></section>`;
$('stop-run').onclick=()=>action('stop',{}); $('export-run').onclick=async()=>{try{const value=await api(`/api/runs/${selected}/export`,{});$('download-slot').innerHTML=`<a class="export-link" href="${escapeHTML(value.download_url)}" download>下载 ZIP ↓</a>`;notice('证据包已生成。使用 CLI 与指定公钥、运行编号独立核验。');}catch(e){notice(e.message);}}; if(stopped){$('approved').onchange=()=>{$('resume-run').disabled=!$('approved').checked;};$('resume-run').onclick=()=>action('resume',{operator_approved:true});} $('step-slider').oninput=renderStep;$('event-filter').value=oldFilter;$('event-filter').onchange=renderEvents;renderStep();renderEvents(); }
async function action(name,body) { try{await api(`/api/runs/${selected}/${name}`,body);await loadRun(selected);notice(name==='stop'?'已请求撤销，等待控制器取消确认与排空。':'恢复请求已提交，执行器将重新核验观测、代次与预测有效期。');}catch(e){notice(e.message);} }
async function loadRun(id) { if(busy)return;busy=true;try{selected=id;const [detail,events]=await Promise.all([api('/api/runs/'+id),api('/api/runs/'+id+'/events')]);eventData=events.events;renderRun(detail.run);await listRuns();}catch(e){notice(e.message);}finally{busy=false;} }
$('run-form').onsubmit=async event=>{event.preventDefault();$('start').disabled=true;try{const text=$('scenario-json').value.trim();let scenario;if(text){ // Preserve raw JSON duplicate-key checking on the server.
const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json','X-Sentinel-Token':token},body:text});const value=await response.json();if(!response.ok)throw Error(value.error?.message||'创建失败');await loadRun(value.run.id);
}else{scenario={schema_version:'product-v1',name:$('name').value,seed:Number($('seed').value),displacement:Number($('displacement').value),risk_limit:Number($('risk_limit').value),prediction_mode:$('prediction_mode').value};const value=await api('/api/runs',scenario);await loadRun(value.run.id);}notice('运行已创建，正在检查四个最终候选。');}catch(e){notice(e.message);}finally{$('start').disabled=false;}};
$('template').onchange=()=>fill(templates.find(t=>t.id===$('template').value).scenario);$('refresh').onclick=()=>listRuns().catch(e=>notice(e.message));
const pageCopy = {
  loop: ['闭环运行', '启动模型、检查最终动作、执行仿真，并从实际反馈继续。', 'SmolVLA · Panda · MuJoCo'],
  runs: ['实验工作台', '比较候选、检查许可，并追踪实际控制器反馈。', 'L0 · 数值参考'],
  native: ['AI执行', '核验并回放 VLA 的真实保存记录、动作和授权游标。', '签名记录 · 非物理证明'],
  physics: ['三维实验', '检查真实接触动力学、轨迹与科学验收门。', 'MuJoCo · 真实物理'],
  assets: ['模型资产', '导入、预览并检查机器人和场景几何。', 'OBJ · STL · MJCF · URDF'],
  robots: ['机械臂接入', '创建驱动配置并执行不会产生运动的只读诊断。', '只读诊断'],
  experiments: ['实验计划', '查看已经实现、正在验证与等待接入的阶段。', '验证路线图'],
};
let activePage = 'runs';
let selectedAsset = null;
let selectedJob = null;
let physicsFrames = [];
let playTimer = null;
let physicsPollTimer = null;
let physicsPollBusy = false;
let loadedTraceJob = null;
let displayedJobStatus = null;
let physicsEngine = {available: false, version: null, reason: '正在检查 MuJoCo 运行时。'};
let physicsJobsAvailable = false;
let selectedAssetFormat = null;
const physicsViewer = new SentinelViewer($('physics-viewer'), {mode: 'trace'});
const assetViewer = new SentinelViewer($('asset-viewer'), {mode: 'mesh'});

function showPage(name) {
  if (activePage !== name) notice('');
  if (physicsPollTimer) {
    clearInterval(physicsPollTimer);
    physicsPollTimer = null;
  }
  if (name !== 'physics') stopPhysicsPlayback();
  if (name !== 'native') window.SentinelNativeUI?.deactivate();
  if (name !== 'loop') window.SentinelLoopUI?.deactivate();
  activePage = name;
  document.querySelectorAll('.page').forEach(element => {
    element.hidden = element.id !== `${name}-page`;
  });
  document.querySelectorAll('.nav').forEach(element => {
    element.classList.toggle('active', element.dataset.page === name);
  });
  const copy = pageCopy[name];
  $('page-title').textContent = copy[0];
  $('page-description').textContent = copy[1];
  $('scope-badge').textContent = copy[2];
  location.hash = name === 'runs' ? '' : name;
  loadPage(name);
}

function ensurePhysicsPolling() {
  if (!physicsPollTimer && activePage === 'physics') {
    physicsPollTimer = setInterval(refreshSelectedPhysics, 1200);
  }
}

async function refreshSelectedPhysics() {
  if (activePage !== 'physics' || physicsPollBusy) return;
  physicsPollBusy = true;
  try {
    const jobs = await listPhysics();
    const selected = jobs.find(job => job.id === selectedJob);
    if (selected && (['queued', 'running'].includes(selected.status) || selected.status !== displayedJobStatus)) {
      await loadPhysics(selectedJob);
    }
    if (!jobs.some(job => ['queued', 'running'].includes(job.status))) {
      clearInterval(physicsPollTimer);
      physicsPollTimer = null;
    }
  } catch (error) {
    notice(error.message);
  } finally {
    physicsPollBusy = false;
  }
}

async function loadPage(name) {
  try {
    if (name === 'runs') await listRuns();
    if (name === 'native') await window.SentinelNativeUI.load();
    if (name === 'loop') await window.SentinelLoopUI.load();
    if (name === 'physics') {
      const jobs = await listPhysics();
      if (selectedJob) await loadPhysics(selectedJob);
      if (jobs.some(job => ['queued', 'running'].includes(job.status))) ensurePhysicsPolling();
    }
    if (name === 'assets') await listAssets();
    if (name === 'robots') await listRobots();
    if (name === 'experiments') await listExperiments();
  } catch (error) {
    notice(error.message);
  }
}

document.querySelectorAll('.nav').forEach(element => {
  element.onclick = () => showPage(element.dataset.page);
});
$('global-refresh').onclick = () => loadPage(activePage);
document.querySelectorAll('[data-view-action]').forEach(element => {
  element.onclick = () => element.dataset.viewAction === 'reset'
    ? physicsViewer.reset()
    : assetViewer.reset();
});

async function listPhysics() {
  const {jobs} = await api('/api/physics-jobs');
  $('physics-job-list').innerHTML = jobs.length
    ? jobs.map(job => `<button class="job-item ${job.id === selectedJob ? 'selected' : ''}" data-job="${escapeHTML(job.id)}"><span>种子 ${escapeHTML(job.seed)} · μ ${escapeHTML(job.friction)}</span>${state(job.status)}<small>${escapeHTML(new Date(job.created_at).toLocaleString())}</small></button>`).join('')
    : '<p class="empty">还没有三维实验。<br>设置参数并启动第一项 MuJoCo 验证。</p>';
  $('physics-job-list').querySelectorAll('[data-job]').forEach(element => {
    element.onclick = () => loadPhysics(element.dataset.job);
  });
  return jobs;
}

$('new-physics').onclick = async () => {
  const button = $('new-physics');
  if (!physicsJobsAvailable || !physicsEngine.available) {
    notice(physicsEngine.reason || '当前安装未提供 MuJoCo 三维物理运行时。');
    return;
  }
  button.disabled = true;
  try {
    const value = await api('/api/physics-jobs', {
      seed: Number($('physics-seed').value),
      friction: Number($('physics-friction').value),
      render: $('physics-render').checked,
    });
    selectedJob = value.job.id;
    notice('三维实验已进入后台队列。完成前不会展示推测轨迹。');
    await listPhysics();
    await loadPhysics(selectedJob);
    ensurePhysicsPolling();
  } catch (error) {
    notice(error.message);
  } finally {
    button.disabled = false;
  }
};

async function loadPhysics(id) {
  selectedJob = id;
  const {job} = await api(`/api/physics-jobs/${id}`);
  if (selectedJob !== id) return;
  displayedJobStatus = job.status;
  $('physics-view-title').textContent = `实验 ${id.slice(-8)}`;
  if (job.status === 'completed') {
    if (loadedTraceJob !== id) {
      const traceResult = await api(`/api/physics-jobs/${id}/trace`);
      physicsFrames = traceResult.trace || [];
      physicsViewer.setTrace(physicsFrames);
      loadedTraceJob = id;
      $('physics-timeline').max = Math.max(0, physicsFrames.length - 1);
      setPhysicsFrame(physicsFrames.length - 1);
    }
    $('physics-view-empty').hidden = physicsFrames.length > 0;
    $('physics-timeline').disabled = !physicsFrames.length;
    $('physics-play').disabled = !physicsFrames.length;
    $('physics-timeline').max = Math.max(0, physicsFrames.length - 1);
    const summary = job.summary || {};
    const accepted = summary.acceptance === true || job.scientific_gates_pass === true;
    $('physics-result').innerHTML = `<div class="result-grid"><div class="result-card"><span>科学验收</span><strong>${accepted ? '通过' : '未全通过'}</strong></div><div class="result-card"><span>综合结果</span><strong>${escapeHTML(summary.integrated?.outcome || summary.integrated_outcome || '—')}</strong></div><div class="result-card"><span>证据核验</span><strong>${job.verification?.ok ? 'PASS' : '—'}</strong></div><div class="result-card"><span>实验状态</span><strong>已完成</strong></div></div><div class="actions"><button id="export-physics">导出签名证据</button></div>`;
    $('export-physics').onclick = async () => {
      try {
        const result = await api(`/api/physics-jobs/${id}/export`, {});
        const link = document.createElement('a');
        link.href = result.download_url;
        link.download = `${id}-evidence.zip`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        notice('签名证据包已生成，下载已开始。');
      } catch (error) {
        notice(`证据导出失败：${error.message}`);
      }
    };
  } else {
    clearPhysicsTrace();
    $('physics-view-empty').hidden = false;
    $('physics-result').innerHTML = job.error
      ? `<div class="error-box">${escapeHTML(job.error.message || job.error)}</div>`
      : `<div class="result-grid"><div class="result-card"><span>任务状态</span><strong>${escapeHTML(labels[job.status] || job.status)}</strong></div></div>`;
  }
}

function stopPhysicsPlayback() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  $('physics-play').textContent = '▶';
}

function clearPhysicsTrace() {
  stopPhysicsPlayback();
  physicsFrames = [];
  loadedTraceJob = null;
  physicsViewer.setTrace([]);
  $('physics-timeline').value = 0;
  $('physics-timeline').max = 0;
  $('physics-timeline').disabled = true;
  $('physics-play').disabled = true;
  $('physics-time').textContent = '—';
}

function setPhysicsFrame(index) {
  const bounded = Math.max(0, Math.min(physicsFrames.length - 1, Number(index) || 0));
  $('physics-timeline').value = bounded;
  physicsViewer.setFrame(bounded);
  const frame = physicsFrames[bounded];
  $('physics-time').textContent = frame ? `${number(frame.physics_time ?? frame.time, 3)} s` : '—';
}

$('physics-timeline').oninput = event => setPhysicsFrame(event.target.value);
$('physics-play').onclick = () => {
  if (playTimer) {
    stopPhysicsPlayback();
    return;
  }
  $('physics-play').textContent = 'Ⅱ';
  if (Number($('physics-timeline').value) >= physicsFrames.length - 1) setPhysicsFrame(0);
  playTimer = setInterval(() => {
    const next = Number($('physics-timeline').value) + 1;
    if (next >= physicsFrames.length) {
      clearInterval(playTimer);
      playTimer = null;
      $('physics-play').textContent = '▶';
      return;
    }
    setPhysicsFrame(next);
  }, 30);
};

async function listAssets(){const {assets}=await api('/api/assets');$('asset-list').innerHTML=assets.length?assets.map(a=>`<button class="asset-item ${a.id===selectedAsset?'selected':''}" data-asset="${escapeHTML(a.id)}"><span>${escapeHTML(a.name)}</span><span class="state">${escapeHTML(String(a.format).toUpperCase())}</span><small>${escapeHTML((a.source_sha256||'').slice(0,18))}${a.source_sha256?'…':''}</small></button>`).join(''):'<p class="empty">资产库为空。<br>导入一个本地模型开始几何预览。</p>';$('asset-list').querySelectorAll('[data-asset]').forEach(el=>el.onclick=()=>loadAsset(el.dataset.asset));}
$('asset-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;const ext=file.name.split('.').pop().toLowerCase(),format=ext==='xml'?'mjcf':ext;try{const content=await file.arrayBuffer(),bytes=new Uint8Array(content);let binary='';for(let i=0;i<bytes.length;i+=0x8000)binary+=String.fromCharCode(...bytes.subarray(i,i+0x8000));const {asset}=await api('/api/assets',{name:file.name,format,content_base64:btoa(binary)});selectedAsset=asset.id;notice('模型已导入。预览与物理许可保持独立。');await listAssets();await loadAsset(asset.id);}catch(e){notice(e.message);}finally{event.target.value='';}};
async function loadAsset(id){selectedAsset=id;const [{asset},{geometry}]=await Promise.all([api('/api/assets/'+id),api(`/api/assets/${id}/geometry`)]);selectedAssetFormat=asset.format;assetViewer.setGeometry(geometry);$('asset-view-empty').hidden=!!(geometry.vertices?.length||geometry.primitives?.length);$('asset-view-title').textContent=asset.name;$('asset-check').disabled=false;$('asset-meta').innerHTML=`<span>格式 ${escapeHTML(String(asset.format).toUpperCase())}</span><span>顶点 ${geometry.vertex_count??'—'}</span><span>三角面 ${geometry.triangle_count??'—'}</span><span>几何体 ${geometry.primitive_count??'—'}</span><span title="${escapeHTML(asset.source_sha256||'')}">摘要 ${escapeHTML((asset.source_sha256||'—').slice(0,19))}</span>`;$('asset-check-result').innerHTML=asset.last_check?`<div class="check-result"><strong>物理检查：${escapeHTML(asset.last_check.status)}</strong><br>${escapeHTML(asset.last_check.scope||asset.last_check.reason||'')}<br>模型运动许可：未授权</div>`:['mjcf','urdf'].includes(asset.format)&&!physicsEngine.available?`<div class="check-result"><strong>MuJoCo 检查不可用</strong><br>${escapeHTML(physicsEngine.reason||'当前安装未提供物理引擎；仍可进行有界几何预览。')}</div>`:'';await listAssets();}
$('asset-check').onclick=async()=>{if(!selectedAsset)return;$('asset-check').disabled=true;try{const {check}=await api(`/api/assets/${selectedAsset}/check`,{}),message=check.message||check.reason||check.error?.message||check.scope||JSON.stringify(check);$('asset-check-result').innerHTML=`<div class="check-result"><strong>物理检查：${escapeHTML(check.status||'已返回')}</strong><br>${escapeHTML(message)}</div>`;if(check.geometry_updated)await loadAsset(selectedAsset);}catch(e){notice(e.message);}finally{$('asset-check').disabled=false;}};

async function listRobots(){const {profiles}=await api('/api/robots');$('robot-list').innerHTML=profiles.length?profiles.map(p=>`<article class="robot-item"><span>${escapeHTML(p.name||p.id)}</span>${state(p.last_diagnostic?.status||'configured')}<small>${escapeHTML(p.driver)} · ${escapeHTML(p.endpoint?p.endpoint.host+':'+p.endpoint.port:'本地协议演示')} · 运动 ${p.hardware_motion?'已启用':'禁用'}</small><div class="robot-actions"><button data-diagnose="${escapeHTML(p.id)}">运行只读诊断</button></div></article>`).join(''):'<p class="empty">还没有机械臂配置。<br>可先添加 Mock 驱动验证完整接入流程。</p>';$('robot-list').querySelectorAll('[data-diagnose]').forEach(el=>el.onclick=()=>diagnoseRobot(el.dataset.diagnose));}
$('show-robot-form').onclick=()=>$('robot-dialog').showModal();$('close-robot').onclick=()=>$('robot-dialog').close();
$('robot-driver').onchange=()=>{const network=$('robot-driver').value!=='mock';for(const id of ['robot-host','robot-port']){$(id).disabled=!network;$(id).closest('label').classList.toggle('disabled-field',!network);}};$('robot-driver').onchange();
$('robot-form').onsubmit=async event=>{event.preventDefault();try{const driver=$('robot-driver').value,body={name:$('robot-name').value,driver};if(driver!=='mock')Object.assign(body,{host:$('robot-host').value,port:Number($('robot-port').value)});await api('/api/robots',body);$('robot-dialog').close();notice('机械臂连接配置已保存；当前能力仅限诊断。');await listRobots();}catch(e){notice(e.message);}};
async function diagnoseRobot(id){$('robot-diagnostic').className='diagnostic-output';$('robot-diagnostic').textContent='正在执行只读诊断…';try{const {diagnostic}=await api(`/api/robots/${id}/diagnose`,{});$('robot-diagnostic').textContent=JSON.stringify(diagnostic,null,2);}catch(e){$('robot-diagnostic').textContent='诊断失败\n'+e.message;}}

async function listExperiments(){const {experiments}=await api('/api/experiments');$('experiment-list').innerHTML=experiments.map(e=>`<section class="panel experiment-card"><div class="panel-head"><div><span class="kicker">${escapeHTML(e.id||'EXPERIMENT')}</span><h2>${escapeHTML(e.title||e.name||e.id)}</h2></div>${state(e.status)}</div><dl><dt>范围</dt><dd>${escapeHTML(e.scope||e.description||'见验收说明')}</dd><dt>前置条件</dt><dd>${escapeHTML(Array.isArray(e.prerequisites)?e.prerequisites.join('；'):e.prerequisites)}</dd><dt>验收标准</dt><dd>${escapeHTML(Array.isArray(e.acceptance)?e.acceptance.join('；'):e.acceptance)}</dd><dt>输出</dt><dd>${escapeHTML(Array.isArray(e.artifacts)?e.artifacts.join('；'):e.artifacts)}</dd></dl><pre>${escapeHTML(e.command||e.entrypoint||'需完成外部集成后运行')}</pre></section>`).join('');}

function configureCapabilities(session) {
  physicsJobsAvailable = Boolean(session.capabilities?.physics_jobs);
  physicsEngine = session.physics_engine || {
    available: physicsJobsAvailable,
    version: null,
    reason: physicsJobsAvailable ? null : '当前安装未提供 MuJoCo 三维物理运行时。',
  };
  const engineState = document.createElement('div');
  engineState.id = 'physics-engine-state';
  engineState.className = `engine-state ${physicsEngine.available ? 'available' : 'unavailable'}`;
  engineState.textContent = physicsEngine.available
    ? `MuJoCo ${physicsEngine.version || ''} 已就绪 · 本机真实物理计算`
    : `MuJoCo 不可用 · ${physicsEngine.reason || '请安装 physics 组件。'}`;
  $('physics-form').before(engineState);
  $('new-physics').disabled = !physicsJobsAvailable || !physicsEngine.available;
  pageCopy.physics[2] = physicsEngine.available
    ? `MuJoCo ${physicsEngine.version || ''} · 可用`
    : 'MuJoCo · 未安装';

  const fileInput = $('asset-file');
  fileInput.hidden = false;
  fileInput.classList.add('file-input');
  fileInput.setAttribute('aria-label', '选择并导入三维模型文件');
}

(async()=>{try{const session=await api('/api/session');token=session.token;templates=session.templates;configureCapabilities(session);$('template').innerHTML=templates.map(t=>`<option value="${escapeHTML(t.id)}">${escapeHTML(t.name)}</option>`).join('');const route=location.hash.slice(1);showPage(pageCopy[route]?route:'runs');timer=setInterval(()=>{if(selected&&activePage==='runs'&&['queued','preparing','running','stopping'].includes(current?.status))loadRun(selected);},700);}catch(e){notice('本地服务连接失败：'+e.message);$('start').disabled=true;}})();
