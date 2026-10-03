'use strict';

window.SentinelNativeUI = (() => {
  const LIMIT = 64 * 1024 * 1024;
  let selectedId = null;
  let selectedRun = null;
  let replay = null;
  let episode = null;
  let viewer = null;
  let viewerError = null;
  let playTimer = null;
  let initialized = false;

  const element = id => document.getElementById(id);
  const workbench = () => window.SentinelWorkbench;
  const escape = value => workbench().escapeHTML(value);
  const api = (path, body) => workbench().api(path, body);
  const announce = message => workbench().notice(message);
  const decisions = {RESET_OBSERVED:'初始反馈已记录', BASELINE_DIRECT:'基线直接执行', SHADOW_PASSTHROUGH:'影子观察通过', SHADOW_WOULD_REJECT_PASSTHROUGH:'影子判定拒绝，未干预', ACTIVE_AUTHORIZED:'一次授权通过'};
  const reasons = {ACTION_REPLACED:'动作被替换', LEASE_REPLAY:'许可已消费，禁止重放', LEASE_EXPIRED:'许可已过期', FEEDBACK_CHANGED:'反馈身份已变化', OLD_FEEDBACK:'反馈过旧', CONTEXT_CHANGED:'执行上下文已变化', GENERATION_REVOKED:'授权代次已撤销', STALE_FEEDBACK:'反馈超过有效期'};
  const safeAssetName = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(value);

  function init() {
    if (initialized) return;
    initialized = true;
    try {
      viewer = new window.SentinelNativeViewer(element('native-viewer'));
    } catch (error) {
      viewerError = error.message;
    }
    element('native-import-toggle').onclick = () => { element('native-import-panel').hidden = false; element('native-run-id').focus(); };
    element('native-import-close').onclick = () => { element('native-import-panel').hidden = true; };
    element('native-refresh').onclick = () => listRuns(true).catch(showError);
    element('native-reset-view').onclick = () => viewer?.reset();
    element('native-episode').onchange = () => selectEpisode(element('native-episode').value).catch(showError);
    element('native-timeline').oninput = event => setFrame(event.target.value);
    element('native-play').onclick = togglePlayback;
    element('native-archive').onchange = updateFiles;
    element('native-public-key').onchange = updateFiles;
    element('native-import-form').onsubmit = importRun;
  }

  function updateFiles() {
    const archive = element('native-archive').files[0], key = element('native-public-key').files[0];
    element('native-archive-name').textContent = archive ? `${archive.name} · ${formatBytes(archive.size)}` : '请选择不超过 64 MiB 的 ZIP';
    element('native-key-name').textContent = key ? `${key.name} · ${key.size} 字节` : '必须单独选择 32 字节原始公钥';
  }

  function formatBytes(value) {
    if (value < 1024) return `${value} B`;
    return value < 1024 * 1024 ? `${(value/1024).toFixed(1)} KiB` : `${(value/1024/1024).toFixed(1)} MiB`;
  }

  function bytesToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let index=0; index<bytes.length; index+=0x8000) binary += String.fromCharCode(...bytes.subarray(index,index+0x8000));
    return btoa(binary);
  }

  async function importRun(event) {
    event.preventDefault();
    const button = element('native-import-submit');
    const archive = element('native-archive').files[0], key = element('native-public-key').files[0];
    const runId = element('native-run-id').value.trim();
    if (!archive || !key) return showError(new Error('请同时选择 VLA ZIP 记录和独立公钥。'));
    if (archive.size > LIMIT) return showError(new Error('网页导入上限为 64 MiB；请使用下方 native-import CLI 导入较大记录。'));
    if (key.size !== 32) return showError(new Error('Ed25519 原始公钥必须恰好为 32 字节。'));
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(runId)) return showError(new Error('记录编号仅支持字母、数字、下划线或连字符。'));
    button.disabled = true;
    button.querySelector('span').textContent = '正在核验签名…';
    try {
      const [archiveBuffer,keyBuffer] = await Promise.all([archive.arrayBuffer(),key.arrayBuffer()]);
      const {run} = await api('/api/native-runs/import', {
        archive_base64: bytesToBase64(archiveBuffer),
        run_id: runId,
        public_key_base64: bytesToBase64(keyBuffer),
      });
      element('native-import-form').reset(); updateFiles(); element('native-import-panel').hidden = true;
      announce('记录已按独立选择的公钥完成签名核验并导入。');
      await listRuns(false); await loadRun(run.id);
    } catch (error) {
      showError(error, '导入失败');
    } finally {
      button.disabled = false;
      button.querySelector('span').textContent = '核验签名并导入';
    }
  }

  async function listRuns(preserve=true) {
    init();
    const {runs} = await api('/api/native-runs');
    const list = element('native-run-list');
    list.innerHTML = runs.length ? runs.map(run => {
      const verified = run.verification?.ok === true;
      const metrics = run.metrics || {};
      return `<button class="native-run-item ${run.id===selectedId?'selected':''}" data-native-run="${escape(run.id)}">
        <span>${escape(run.name || run.id)}</span><strong class="native-integrity ${verified?'pass':'fail'}">${verified?'SIGNED':'FAILED'}</strong>
        <small>${escape(run.mode || 'VLA')} · ${escape(run.episode_count ?? metrics.episodes ?? 0)} episodes · ${escape(run.status || 'unknown')}</small>
      </button>`;
    }).join('') : '<p class="empty">还没有已核验的 VLA 记录。<br>导入签名 ZIP，或先使用 native-run 生成记录。</p>';
    list.querySelectorAll('[data-native-run]').forEach(button => { button.onclick = () => loadRun(button.dataset.nativeRun).catch(showError); });
    if (preserve && selectedId && !runs.some(run => run.id === selectedId)) clearSelection('所选记录已不存在或完整性核验失败。');
    window.SentinelCompareUI?.updateRuns(runs);
    return runs;
  }

  async function loadRun(id) {
    stopPlayback();
    clearFailure();
    selectedId = id;
    element('native-view-empty').hidden = false;
    element('native-view-empty').querySelector('strong').textContent = '正在核验并载入记录';
    const [{run}, replayValue] = await Promise.all([api(`/api/native-runs/${id}`), api(`/api/native-runs/${id}/replay`)]);
    if (selectedId !== id) return;
    selectedRun = run; replay = replayValue;
    if (!run.verification?.ok) throw new Error('记录签名完整性核验未通过，已拒绝回放。');
    if (!Array.isArray(replay.episodes) || !replay.episodes.length) throw new Error('记录中没有可回放的 Episode。');
    element('native-view-title').textContent = run.name || run.id;
    element('native-view-id').textContent = `${run.id} · ${run.mode || 'VLA'} · ${run.status || 'unknown'}`;
    element('native-download').href = `/api/native-runs/${id}/download`;
    element('native-download').download = `${id}-native-vla.zip`;
    element('native-download').hidden = false;
    element('native-episode').disabled = false;
    element('native-episode').innerHTML = replay.episodes.map((item,index) => `<option value="${index}">${escape(item.episode_id ?? `Episode ${index+1}`)} · task ${escape(item.task_id ?? '—')} · ${item.frames?.length || 0} 帧</option>`).join('');
    renderIntegrity(run);
    await selectEpisode('0');
    await listRuns(true);
  }

  function renderIntegrity(run) {
    const cells = element('native-verdicts').children;
    cells[0].querySelector('strong').textContent = run.verification?.ok ? 'PASS' : 'FAIL';
    cells[0].className = run.verification?.ok ? 'pass' : 'fail';
    cells[0].querySelector('small').textContent = run.verification?.trust === 'user-selected-public-key' ? '独立所选公钥' : '信任锚未知';
    cells[2].querySelector('strong').textContent = '未支持';
    cells[2].querySelector('small').textContent = run.physical_validation?.reason || '不作物理安全证明';
  }

  async function selectEpisode(indexValue) {
    stopPlayback(); clearFailure();
    const index = Math.max(0,Math.min(replay.episodes.length-1,Number(indexValue)||0));
    episode = replay.episodes[index];
    element('native-episode').value = String(index);
    clearReplayView('正在载入当前任务模型');
    if (!Array.isArray(episode.frames) || !episode.frames.length) throw new Error('此 Episode 没有保存回放帧。');
    let modelLoaded = false;
    try {
      if (viewerError) throw new Error(viewerError);
      const model = await loadModel(selectedId, episode, replay.episodes);
      viewer.setRecord(model, episode);
      modelLoaded = true;
    } catch (error) {
      viewer?.clear();
      element('native-view-empty').querySelector('strong').textContent = '当前记录提供动作时间轴';
      element('native-view-empty').querySelector('small').textContent = error.message;
    }
    element('native-view-empty').hidden = modelLoaded;
    element('native-reset-view').disabled = !modelLoaded;
    element('native-timeline').disabled = false;
    element('native-play').disabled = false;
    element('native-timeline').min = 0;
    element('native-timeline').max = episode.frames.length - 1;
    element('native-timeline').value = 0;
    renderEpisodeDetail(index);
    configureVideo(index);
    setFrame(0);
  }

  async function fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) {
      let message = '模型资产读取失败。';
      try { message = (await response.json()).error?.message || message; } catch (_) { /* raw response */ }
      throw new Error(message);
    }
    return response.json();
  }

  async function loadModel(runId, selectedEpisode, allEpisodes) {
    let asset = selectedEpisode.model_asset;
    if (asset !== null && asset !== undefined && !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(asset)) throw new Error('Episode 的模型文件名不合法。');
    if (!asset) {
      const index = await fetchJSON(`/api/native-runs/${runId}/model`);
      if (Array.isArray(index.meshes)) return index;
      const entries = Array.isArray(index.models) ? index.models : [];
      const match = entries.find(item => String(item.task_id) === String(selectedEpisode.task_id));
      asset = match?.asset;
      const multipleTasks = new Set(allEpisodes.map(item => String(item.task_id))).size > 1;
      if (!asset) throw new Error(multipleTasks ? '多任务记录缺少当前 task 的独立模型映射，已停止渲染以防错配。' : '记录缺少当前 Episode 的模型资产。');
    }
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/.test(asset)) throw new Error('任务模型文件名不合法。');
    return fetchJSON(`/api/native-runs/${runId}/files/${asset}`);
  }

  function episodeSummary(index) {
    const metadata = selectedRun.episodes?.find(item => String(item.episode_id) === String(episode.episode_id)) || selectedRun.episodes?.[index] || {};
    return {...metadata,...episode};
  }

  function renderEpisodeDetail(index) {
    const summary = episodeSummary(index);
    const limits = Array.isArray(selectedRun.claim_limits) ? selectedRun.claim_limits.join('；') : selectedRun.claim_limits;
    element('native-detail').innerHTML = `<dl>
      <div><dt>指令</dt><dd>${escape(summary.instruction || '记录未提供')}</dd></div>
      <div><dt>任务</dt><dd>${escape(summary.task_id ?? '—')} · initial state ${escape(summary.initial_state_index ?? '—')}</dd></div>
      <div><dt>种子</dt><dd>${escape(summary.seed ?? '—')}</dd></div>
      <div><dt>结果</dt><dd>${summary.crashed?'运行崩溃':summary.success===true?'任务成功':summary.success===false?'任务未成功':'记录未声明'}</dd></div>
      <div><dt>模型</dt><dd>${escape(summary.model_asset || '由签名模型索引解析')}</dd></div>
      ${(summary.fault_attempts || []).length ? `<div><dt>故障注入记录</dt><dd>${summary.fault_attempts.slice(0,16).map(attempt => `<p>步骤 ${escape(attempt.step)} · ${escape(reasons[attempt.reason] || attempt.reason)} · ${attempt.blocked?'已阻断':'未阻断'} · 环境写入 ${escape(attempt.env_step_calls)} 次</p>`).join('')}</dd></div>` : ''}
      <div><dt>声明限制</dt><dd>${escape(limits || '不证明物理安全或硬件执行')}</dd></div>
    </dl>`;
    const failure = selectedRun.fatal_failure || (summary.crashed ? '此 Episode 记录为崩溃状态。' : '');
    if (failure) showFailure(typeof failure === 'string' ? failure : failure.message || JSON.stringify(failure));
  }

  function permitValue(frame) {
    return frame.permit ?? frame.authorization?.permit ?? frame.permit_id;
  }

  function permitSummary(frame) {
    const permit = permitValue(frame);
    if (!permit) return '';
    if (typeof permit === 'string') return `permit ${permit}`;
    if (typeof permit !== 'object') return '';
    const id = permit.permit_id ?? permit.id ?? permit.lease_id;
    const details = [];
    if (id) details.push(String(id));
    if (permit.step !== undefined) details.push(`step ${permit.step}`);
    if (permit.generation !== undefined) details.push(`gen ${permit.generation}`);
    return details.length ? `permit ${details.join(' · ')}` : '记录含 permit 对象';
  }

  function actionSummary(action) {
    if (!Array.isArray(action)) return '无动作提交';
    const values = action.map(value => Number.isFinite(value) ? value.toFixed(4) : String(value));
    if (values.length === 7) return `Δxyz [${values.slice(0,3).join(', ')}] · Δrot [${values.slice(3,6).join(', ')}] · gripper ${values[6]}`;
    return `[${values.join(', ')}]`;
  }

  function scalar(value) {
    return Array.isArray(value) ? value[0] : value;
  }

  function number(value) {
    return Number.isFinite(value) ? Number(value).toFixed(4) : '—';
  }

  function stateSummary(frame) {
    const positions = frame.bodyWorldPosition;
    const eefIndex = viewer?.traceBodyIndex;
    const eef = Array.isArray(positions) && Number.isInteger(eefIndex) ? positions[eefIndex] : null;
    const parts = [];
    if (Array.isArray(eef) && eef.length === 3) parts.push(`EEF [${eef.map(number).join(', ')}]`);
    if (Array.isArray(frame.qpos) && frame.qpos.length) parts.push(`qpos[0:7] [${frame.qpos.slice(0,7).map(number).join(', ')}]`);
    if (Array.isArray(frame.contacts)) parts.push(`contacts ${frame.contacts.length}`);
    return parts.join(' · ') || '记录未提供可读状态摘要';
  }

  function outcomeSummary(outcome) {
    if (typeof outcome === 'string') return outcome;
    if (!outcome || typeof outcome !== 'object') return '记录未提供 outcome';
    const info = outcome.info && typeof outcome.info === 'object' ? outcome.info : {};
    const fields = [['reward',scalar(outcome.reward)],['success',scalar(info.is_success)],['terminated',scalar(outcome.terminated)],['truncated',scalar(outcome.truncated)]];
    const parts = fields.filter(([,value]) => value !== undefined && value !== null).map(([label,value]) => `${label} ${typeof value === 'number' ? number(value) : String(value)}`);
    return parts.join(' · ') || '记录未提供可读 outcome 摘要';
  }

  function rawEvidence(frame) {
    const retained = {observation_hash:frame.observation_hash,action_bytes_hash:frame.action_bytes_hash,state:frame.state,outcome:frame.outcome};
    for (const key of Object.keys(retained)) if (retained[key] === undefined) delete retained[key];
    return Object.keys(retained).length ? JSON.stringify(retained,null,2) : '此帧未保存原始摘要字段。';
  }

  function configureVideo(index) {
    const block = element('native-video-block'), video = element('native-video'), note = element('native-video-note');
    const asset = episode.video_asset;
    video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true; block.hidden = true;
    if (!asset) return;
    block.hidden = false;
    if (!safeAssetName(asset)) {
      note.textContent = '视频文件名不合法，已拒绝加载。';
      return;
    }
    const videoEpisodes = replay.episodes.filter(item => item.video_asset);
    const videoAssets = new Set(videoEpisodes.map(item => item.video_asset));
    const hashes = selectedRun?.verified_video_sha256;
    const verifiedHashes = hashes && typeof hashes === 'object' && !Array.isArray(hashes) ? hashes : null;
    if (videoEpisodes.length > 1 && (!verifiedHashes || [...videoAssets].some(name => typeof verifiedHashes[name] !== 'string'))) {
      note.textContent = '此包包含多个 Episode 视频，但当前核验 API 未提供签名资产哈希，无法确认各文件是否为独立内容；为避免误导，已禁用包内视频。姿态、动作与授权时间轴仍可检查。';
      return;
    }
    const hash = verifiedHashes?.[asset];
    if (hash) {
      const aliases = videoEpisodes.filter(item => verifiedHashes[item.video_asset] === hash);
      const taskStates = new Set(aliases.map(item => `${item.task_id ?? '—'}\u0000${item.initial_state_index ?? '—'}`));
      if (taskStates.size > 1) {
        note.textContent = '源记录器媒体别名（source recorder media alias）：同一已核验视频内容被绑定到不同 task/state，无法证明视频对应当前 Episode，已禁用所有歧义视频。姿态、动作与授权时间轴仍按各自记录显示。';
        return;
      }
      const firstHashIndex = replay.episodes.findIndex(item => item.video_asset && verifiedHashes[item.video_asset] === hash);
      if (firstHashIndex >= 0 && firstHashIndex < index) {
        const first = replay.episodes[firstHashIndex];
        note.textContent = `此 Episode 与 ${first.episode_id ?? `Episode ${firstHashIndex+1}`} 的视频内容哈希相同，已禁用重复播放。记录器视频独立播放，不与姿态时间轴同步。`;
        return;
      }
    }
    const firstIndex = replay.episodes.findIndex(item => item.video_asset === asset);
    if (firstIndex >= 0 && firstIndex < index) {
      const first = replay.episodes[firstIndex];
      note.textContent = `此 Episode 与 ${first.episode_id ?? `Episode ${firstIndex+1}`} 引用同一保存视频文件，已禁用重复播放。记录器视频独立播放，不与姿态时间轴同步。`;
      return;
    }
    note.textContent = '记录器视频，独立播放，不与姿态时间轴同步。';
    video.src = `/api/native-runs/${selectedId}/files/${asset}`;
    video.hidden = false;
    video.onerror = () => {
      video.hidden = true; video.removeAttribute('src'); video.load();
      note.textContent = '记录器视频无法读取；姿态、动作与授权时间轴仍可独立检查。';
    };
  }

  function setFrame(indexValue) {
    if (!episode) return;
    const index=Math.max(0,Math.min(episode.frames.length-1,Number(indexValue)||0));
    element('native-timeline').value=index;
    viewer?.draw(index);
    const frame=episode.frames[index], decision=String(frame.decision ?? 'unknown');
    const poses = viewer?.poseFrames || [];
    const priorPose = poses.findLast ? poses.findLast(item => item.index <= index) : [...poses].reverse().find(item => item.index <= index);
    const pose = priorPose || poses[0];
    element('native-pose-note').textContent = pose ? (pose.index===index ? `三维姿态：当前 step ${frame.step} 的保存记录` : `动作与反馈：step ${frame.step}；该步姿态未保留，三维显示${priorPose?'此前':'此后'} step ${pose.frame.step} 的保存姿态。`) : '当前任务没有可显示的保存姿态。';

    element('native-time').textContent=`${Number.isFinite(frame.timeSeconds)?frame.timeSeconds.toFixed(3):'—'} s · 帧 ${index+1}/${episode.frames.length}`;
    element('native-action').textContent=actionSummary(frame.action);
    element('native-action-hash').textContent=frame.action_bytes_hash?`bytes ${String(frame.action_bytes_hash).slice(0,18)}…`:'此帧没有动作字节摘要';
    element('native-decision').textContent=decisions[decision] || decision;
    element('native-reason').textContent=frame.reason ? `${reasons[frame.reason] || frame.reason} (${frame.reason})` : (permitSummary(frame) || decision);
    const cursors=frame.cursors || {};
    element('native-cursors').textContent=`${cursors.submitted ?? '—'} / ${cursors.accepted ?? '—'} / ${cursors.observed ?? '—'}`;
    element('native-state').textContent=stateSummary(frame);
    element('native-outcome').textContent=outcomeSummary(frame.outcome);
    element('native-raw-evidence').textContent=rawEvidence(frame);
    const verdict=element('native-verdicts').children[1], pass=isAllowed(frame), reject=isRejected(frame);
    verdict.className=pass?'pass':reject?'fail':'';
    verdict.querySelector('strong').textContent=pass?'PASS':reject?'REJECT':decision;
    verdict.querySelector('small').textContent=pass?(permitValue(frame)?'记录含 permit；仅代表请求完整性授权':'记录判定允许；仅代表请求完整性授权'):frame.reason||'逐帧记录判定';
  }

  function isAllowed(frame) {
    return Boolean(permitValue(frame)) || String(frame.decision).toUpperCase() === 'ACTIVE_AUTHORIZED' || ['allowed','allow','accepted','pass','permit','permitted'].includes(String(frame.decision).toLowerCase());
  }

  function isRejected(frame) {
    return ['denied','deny','rejected','reject','blocked','fail'].includes(String(frame.decision).toLowerCase());
  }

  function togglePlayback() {
    if (playTimer) return stopPlayback();
    if (!episode?.frames.length) return;
    if (Number(element('native-timeline').value) >= episode.frames.length-1) setFrame(0);
    element('native-play').textContent='Ⅱ';
    playTimer=setInterval(()=>{
      const next=Number(element('native-timeline').value)+1;
      if(next>=episode.frames.length)return stopPlayback();
      setFrame(next);
    },50);
  }

  function stopPlayback() {
    if(playTimer)clearInterval(playTimer);
    playTimer=null;
    if(element('native-play'))element('native-play').textContent='▶';
  }

  function clearFailure() {
    element('native-failure').hidden=true;
    element('native-failure').textContent='';
  }

  function showFailure(message) {
    element('native-failure').textContent=message;
    element('native-failure').hidden=false;
  }

  function showError(error, prefix='无法载入记录') {
    const message=`${prefix}：${error?.message || error}`;
    clearReplayView('当前 Episode 无法安全渲染');showFailure(message); announce(message); stopPlayback();
  }

  function clearReplayView(message) {
    viewer?.clear();
    element('native-view-empty').hidden=false;
    element('native-view-empty').querySelector('strong').textContent=message || '尚未载入回放';
    element('native-reset-view').disabled=true;element('native-play').disabled=true;element('native-timeline').disabled=true;
    element('native-timeline').value=0;element('native-timeline').max=0;element('native-time').textContent='—';
    const video=element('native-video');video.pause();video.removeAttribute('src');video.load();video.hidden=true;element('native-video-block').hidden=true;
  }

  function clearSelection(reason) {
    selectedId=null;selectedRun=null;replay=null;episode=null;stopPlayback();
    element('native-view-title').textContent='选择一条执行记录';
    element('native-view-id').textContent='—';
    clearReplayView('尚未载入回放');
    element('native-episode').disabled=true;element('native-play').disabled=true;element('native-timeline').disabled=true;element('native-reset-view').disabled=true;element('native-download').hidden=true;
    if(reason)showFailure(reason);
  }

  async function load() {
    init();
    await listRuns(true);
    if(selectedId && selectedRun)viewer?.draw(Number(element('native-timeline').value)||0);
  }

  function deactivate() { stopPlayback(); }

  async function jumpToEvidence(ref) {
    if (!ref || typeof ref.run_id !== 'string' || !Number.isInteger(ref.episode_index) || !Number.isInteger(ref.frame_index)) throw new Error('分歧引用不合法。');
    await loadRun(ref.run_id);
    if (selectedId !== ref.run_id) return;
    if (ref.episode_index < 0 || ref.episode_index >= replay.episodes.length) throw new Error('分歧 Episode 不存在。');
    await selectEpisode(String(ref.episode_index));
    if (ref.frame_index < 0 || ref.frame_index >= episode.frames.length || episode.frames[ref.frame_index].step !== ref.step) throw new Error('分歧帧引用不匹配。');
    setFrame(ref.frame_index);
    element('native-view-title').scrollIntoView({block:'start'});
  }

  return {load,deactivate,jumpToEvidence};
})();
