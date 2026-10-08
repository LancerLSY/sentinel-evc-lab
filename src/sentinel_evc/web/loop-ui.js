'use strict';

/* Live SmolVLA/LIBERO/Panda control surface. It renders only server-reported state. */
(() => {
  const ACTIVE = new Set(['queued', 'preparing', 'running', 'stopping']);
  const TERMINAL = new Set(['stopped', 'completed', 'rejected', 'failed', 'interrupted']);
  const STATUS = {
    queued: '排队', preparing: '准备运行时', running: '闭环运行中', stopping: '正在停止',
    stopped: '已停止', completed: '已完成', rejected: '已拒绝', failed: '失败', interrupted: '已中断',
  };
  const STAGES = ['runtime', 'policy', 'execution', 'evidence'];
  const STAGE_ALIASES = {
    queued: 0, preparing: 0, runtime: 0, loading: 1, model: 1, policy: 1,
    inference: 2, aggregation: 2, verification: 2, permit: 2, dispatch: 2,
    execution: 2, executing: 2, observing: 2, running: 2, stopping: 3, evidence: 3,
    config: 0, environment: 0, stop_requested: 3, finalized: 3, finalizing: 3,
    completed: 3, rejected: 3, stopped: 3, failed: 3, interrupted: 3, source_changed: 3,
  };

  const element = id => document.getElementById(id);
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
  const finite = value => Number.isFinite(Number(value));
  const first = (...values) => values.find(value => value !== undefined && value !== null);
  const boolText = value => value === true ? '是' : value === false ? '否' : '—';
  const timestamp = value => {
    if (!value) return '—';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  };
  const duration = value => finite(value) ? `${Number(value).toFixed(Number(value) < 1 ? 3 : 2)} s` : '—';
  const milliseconds = value => finite(value) ? `${Number(value).toFixed(Number(value) < 10 ? 2 : 1)} ms` : '—';
  const secondsBetween = (start, finish) => {
    const firstTime = Date.parse(start || ''), lastTime = Date.parse(finish || '');
    return Number.isFinite(firstTime) && Number.isFinite(lastTime) && lastTime >= firstTime ? (lastTime - firstTime) / 1000 : null;
  };
  const compact = value => {
    if (value === undefined || value === null) return '—';
    if (Array.isArray(value)) {
      const values = value.slice(0, 9).map(item => finite(item) ? Number(item).toFixed(4) : String(item));
      return `[${values.join(', ')}${value.length > values.length ? ', …' : ''}]`;
    }
    if (typeof value === 'object') {
      const text = JSON.stringify(value);
      return text.length > 240 ? `${text.slice(0, 237)}…` : text;
    }
    return String(value);
  };
  const errorText = value => {
    if (!value) return '';
    if (typeof value === 'string') return value;
    return [value.code, value.message || value.detail].filter(Boolean).join(' · ') || compact(value);
  };

  let request = null;
  let announce = () => {};
  let initialized = false;
  let selectedId = null;
  let currentJob = null;
  let currentLive = null;
  let runtime = null;
  let jobs = [];
  let viewer = null;
  let pollTimer = null;
  let refreshing = false;
  let viewerKey = null;
  let viewerFrames = [];
  const models = new Map();
  const missingModels = new Set();

  function renderRuntime() {
    const box = element('loop-runtime');
    const start = element('loop-start');
    const checks = Array.isArray(runtime?.checks) ? runtime.checks : [];
    const ready = runtime?.configured === true && runtime?.ready === true;
    const active = jobs.some(job => ACTIVE.has(job.status));
    box.className = `loop-runtime ${ready ? 'ready' : 'blocked'}`;
    box.innerHTML = `<span class="loop-runtime-light"></span><div><strong>${ready ? '运行环境已就绪' : runtime?.configured ? '配置存在，但启动检查未通过' : '尚未配置闭环运行环境'}</strong><small>${escapeHTML(runtime?.detail || (ready ? '模型、仿真、执行配置与证据输出检查已通过。' : '按左侧检查项完成配置；未就绪时不会创建任务。'))}</small></div>`;
    element('loop-profile').textContent = runtime?.profile || 'smolvla-libero-panda';
    start.disabled = !ready || active;
    start.title = active ? '已有闭环任务正在运行' : ready ? '启动已钉住的闭环 profile' : '先完成运行环境检查';

    element('loop-setup').hidden = ready;
    element('loop-checks').innerHTML = checks.length
      ? checks.map(check => `<div class="loop-check ${check.ok ? 'ok' : ''}"><i></i><strong>${escapeHTML(check.name || '检查项')}</strong><small>${escapeHTML(check.detail || (check.ok ? '已通过' : '未通过'))}</small></div>`).join('')
      : '<p class="empty">后端尚未返回检查明细。刷新后仍缺失时，请先查看服务日志。</p>';
    const setup = first(runtime?.setup_commands, runtime?.setup?.commands, runtime?.setup_steps, runtime?.setup?.steps);
    const values = Array.isArray(setup) ? setup : setup ? [setup] : [];
    element('loop-setup-steps').innerHTML = values.length
      ? `<pre class="loop-setup-commands">${escapeHTML(values.join('\n'))}</pre>`
      : '<pre class="loop-setup-commands">sentinel-evc loop-configure --python /path/vla/bin/python --source /path/repo --config /path/session.json --data-dir runs/workbench\n# 配置后从同一 workspace 重新启动 sentinel-evc serve</pre>';
  }

  function renderJobs() {
    const target = element('loop-job-list');
    target.innerHTML = jobs.length ? jobs.map(job => {
      const label = STATUS[job.status] || job.status || '未知';
      const title = job.summary?.task || job.task || job.profile || `闭环 ${String(job.id || '').slice(-8)}`;
      return `<button class="loop-job-item ${job.id === selectedId ? 'selected' : ''}" data-loop-job="${escapeHTML(job.id)}"><span>${escapeHTML(title)}</span><span class="state ${escapeHTML(job.status)}">${escapeHTML(label)}</span><small>${escapeHTML(timestamp(job.created_at))} · ${escapeHTML(job.stage || '等待阶段')}</small></button>`;
    }).join('') : '<p class="empty">还没有闭环任务。<br>运行环境就绪后可启动第一条真实 VLA 仿真。</p>';
    target.querySelectorAll('[data-loop-job]').forEach(button => {
      button.onclick = () => selectJob(button.dataset.loopJob, true);
    });
  }

  function statusClass(status) {
    if (['completed'].includes(status)) return 'success';
    if (['rejected'].includes(status)) return 'denied';
    if (['failed', 'interrupted'].includes(status)) return 'error';
    if (['stopped', 'stopping'].includes(status)) return 'warn';
    if (ACTIVE.has(status)) return 'active';
    return 'neutral';
  }

  function renderStage(job) {
    const stage = String(job?.stage || job?.status || 'queued').toLowerCase();
    const exact = STAGES.indexOf(stage);
    const reached = exact >= 0 ? exact : first(STAGE_ALIASES[stage], 0);
    element('loop-stage').querySelectorAll('span').forEach((item, index) => {
      item.classList.toggle('reached', index <= reached);
    });
  }

  function renderJob(job) {
    currentJob = job;
    const state = element('loop-job-state');
    element('loop-job-title').textContent = job.summary?.task || job.task || `${job.profile || 'SmolVLA'} 闭环`;
    element('loop-job-id').textContent = job.id || '—';
    state.className = `loop-state ${statusClass(job.status)}`;
    state.innerHTML = `<i></i><span>${escapeHTML(STATUS[job.status] || job.status || '未知')}</span>`;
    renderStage(job);
    element('loop-stop').disabled = !ACTIVE.has(job.status) || job.status === 'stopping';
    const error = errorText(job.error);
    element('loop-error').hidden = !error;
    element('loop-error').textContent = error;
    const evidenceReady = job.download_ready === true || job.evidence_ready === true || job.summary?.evidence_ready === true || job.verification?.ok === true;
    const download = element('loop-download');
    download.hidden = !evidenceReady;
    if (evidenceReady) download.href = `/api/loop-jobs/${encodeURIComponent(job.id)}/download`;
    element('loop-verification').textContent = job.verification ? (job.verification.ok ? 'PASS' : 'FAIL') : '—';
    element('loop-updated').textContent = `更新 ${timestamp(job.updated_at)}`;

    const operational = first(job.summary?.operational_completion, job.summary?.operational_completed);
    element('loop-operational').textContent = operational === true ? '正常完成' : operational === false ? '未正常完成' : job.status === 'completed' ? '正常完成' : TERMINAL.has(job.status) ? '未正常完成' : '进行中';
    const taskSuccess = first(job.summary?.task_success, job.task_success, job.summary?.success);
    element('loop-task-success').textContent = taskSuccess === true ? '成功' : taskSuccess === false ? '未成功' : '—';
    const results = Array.isArray(job.summary?.results) ? job.summary.results : [];
    const measuredElapsed = results.length && results.every(row => finite(row.elapsed_seconds))
      ? results.reduce((total, row) => total + Number(row.elapsed_seconds), 0) : null;
    element('loop-elapsed').textContent = duration(first(job.summary?.elapsed_seconds, job.elapsed_seconds, job.summary?.wall_seconds, measuredElapsed, secondsBetween(job.created_at, job.updated_at)));
    const measuredGate = results.length === 1 ? first(results[0]?.timing?.active_decision_path, results[0]?.timing?.gateway_to_writer) : null;
    const p50 = first(job.summary?.latency_ms?.p50, job.summary?.latency_p50_ms, job.latency?.p50_ms, measuredGate?.p50_ms);
    const p95 = first(job.summary?.latency_ms?.p95, job.summary?.latency_p95_ms, job.latency?.p95_ms, measuredGate?.p95_ms);
    element('loop-latency').textContent = finite(p50) || finite(p95) ? `${milliseconds(p50)} / ${milliseconds(p95)}` : results.length > 1 ? `见证据包 · ${results.length} episodes` : '—';
  }

  function decision(live) {
    const verdict = first(live?.verdict, live?.decision, live?.gate, live?.latest?.verdict, {});
    const raw = typeof verdict === 'string' ? verdict : first(verdict.allowed, verdict.decision, verdict.status, verdict.verdict, live?.allowed);
    const normalized = String(raw ?? '').toLowerCase();
    const allowed = raw === true || ['allow', 'allowed', 'approved', 'permit', 'permitted'].includes(normalized);
    const denied = raw === false || ['deny', 'denied', 'rejected', 'blocked'].includes(normalized);
    return {
      value: allowed ? 'allowed' : denied ? 'denied' : 'pending',
      label: allowed ? '允许写入' : denied ? '拒绝写入' : '等待反馈',
      reason: typeof verdict === 'object' ? first(verdict.reason, verdict.detail, verdict.code) : first(live?.reason, live?.latest?.reason),
    };
  }

  function renderLive(live) {
    currentLive = live || {};
    if (currentLive.status === 'running') {
      const state = element('loop-job-state');
      state.className = 'loop-state active';
      state.innerHTML = '<i></i><span>闭环运行中</span>';
      renderStage({stage: currentLive.stage || 'executing'});
    }
    const verdict = decision(currentLive);
    const badge = element('loop-verdict-badge');
    badge.className = `loop-verdict ${verdict.value}`;
    badge.textContent = verdict.label;
    element('loop-verdict-reason').textContent = verdict.reason || (verdict.value === 'pending' ? '门禁还没有返回允许或拒绝。' : '后端未附加原因。');
    const cursors = first(currentLive.cursors, currentLive.latest?.cursors, currentJob?.summary?.cursors, {});
    element('loop-submitted').textContent = first(cursors.submitted, cursors.submitted_cursor, 0);
    element('loop-accepted').textContent = first(cursors.accepted, cursors.accepted_cursor, 0);
    element('loop-observed').textContent = first(cursors.observed, cursors.observed_cursor, 0);
    element('loop-candidate-step').textContent = first(currentLive.observation_step, currentLive.step, currentLive.latest?.step, cursors.observed, '—');
    const candidate = first(currentLive.candidate_action, currentLive.candidate?.action, currentLive.latest?.candidate_action);
    const executed = first(currentLive.executed_action, currentLive.actual_action, currentLive.submitted_action, currentLive.actual?.action, currentLive.latest?.executed_action);
    element('loop-candidate-action').textContent = candidate === undefined ? '尚未生成' : compact(candidate);
    element('loop-executed-action').textContent = executed === undefined ? (verdict.value === 'denied' ? '未写入（门禁拒绝）' : '尚未写入') : compact(executed);
    const raw = first(currentLive.actual_state, currentLive.observation, currentLive.latest, currentLive);
    element('loop-raw').textContent = Object.keys(currentLive).length ? JSON.stringify({actual_state: raw, verdict: first(currentLive.verdict, currentLive.decision), cursors}, null, 2) : '—';
    const taskSuccess = first(currentLive.task_success, currentLive.actual?.task_success);
    const liveTerminal = ['complete', 'completed', 'stopped', 'rejected', 'failed', 'interrupted'].includes(currentLive.status);
    if (taskSuccess === true) element('loop-task-success').textContent = '成功';
    else if (taskSuccess === false) element('loop-task-success').textContent = liveTerminal ? '未成功' : '尚未判定';
    const elapsed = first(currentLive.elapsed_seconds, currentLive.timing?.elapsed_seconds);
    const liveElapsed = first(elapsed, secondsBetween(currentJob?.created_at, currentLive.updated_at));
    if (finite(liveElapsed)) element('loop-elapsed').textContent = duration(liveElapsed);
    const p50 = first(currentLive.latency_ms?.p50, currentLive.timing?.latency_p50_ms);
    const p95 = first(currentLive.latency_ms?.p95, currentLive.timing?.latency_p95_ms);
    if (finite(p50) || finite(p95)) element('loop-latency').textContent = `${milliseconds(p50)} / ${milliseconds(p95)}`;
  }

  function liveFrames(live) {
    const frames = first(live?.frames, live?.actual_frames, live?.episode?.frames, live?.trajectory?.frames);
    if (Array.isArray(frames)) return frames;
    const frame = first(live?.frame, live?.actual_frame, live?.actual, live?.latest?.frame);
    return frame ? [frame] : [];
  }

  async function modelFor(jobId, live) {
    const embedded = first(live?.model, live?.render_model);
    if (embedded) return embedded;
    if (models.has(jobId)) return models.get(jobId);
    if (missingModels.has(jobId)) return null;
    try {
      const value = await request(`/api/loop-jobs/${encodeURIComponent(jobId)}/model`);
      const model = value.model || value;
      models.set(jobId, model);
      return model;
    } catch (error) {
      if (currentJob && TERMINAL.has(currentJob.status)) missingModels.add(jobId);
      return null;
    }
  }

  async function renderViewer(job, live) {
    if (!viewer) return;
    const frames = liveFrames(live);
    const empty = element('loop-view-empty');
    if (!frames.length) {
      if (!viewerFrames.length || job.id !== selectedId) viewer.clear();
      empty.hidden = false;
      empty.querySelector('strong').textContent = '尚无实际反馈';
      empty.querySelector('small').textContent = '收到 MuJoCo body pose 后才绘制，不生成预览动作';
      return;
    }
    const model = await modelFor(job.id, live);
    if (!model) {
      empty.hidden = false;
      empty.querySelector('strong').textContent = '实际反馈已到达';
      empty.querySelector('small').textContent = '后端未提供可渲染的源模型；状态仍按原值显示';
      return;
    }
    const last = frames.at(-1) || {};
    const key = `${job.id}:${frames.length}:${first(last.step, last.index, last.observation_step, '')}`;
    if (viewerKey !== key) {
      try {
        viewer.setRecord(model, {frames});
        viewerKey = key;
        viewerFrames = frames;
      } catch (error) {
        empty.hidden = false;
        empty.querySelector('strong').textContent = '姿态无法绘制';
        empty.querySelector('small').textContent = error.message;
        return;
      }
    }
    empty.hidden = true;
    const timeline = element('loop-timeline');
    timeline.max = Math.max(0, viewerFrames.length - 1);
    timeline.value = timeline.max;
    timeline.disabled = viewerFrames.length < 2;
    viewer.draw(Number(timeline.value));
    element('loop-reset-view').disabled = false;
    element('loop-time').textContent = `观测 ${first(last.observation_step, last.step, frames.length - 1)} · ${frames.length} 帧`;
  }

  async function selectJob(id, userSelected = false) {
    if (!id || !request) return;
    selectedId = id;
    renderJobs();
    try {
      const [{job}, liveValue] = await Promise.all([
        request(`/api/loop-jobs/${encodeURIComponent(id)}`),
        request(`/api/loop-jobs/${encodeURIComponent(id)}/live`),
      ]);
      if (selectedId !== id) return;
      const live = liveValue.live || {};
      renderJob(job);
      renderLive(live);
      await renderViewer(job, live);
      if (userSelected) announce(`已打开闭环任务 ${id}。`);
      schedulePoll();
    } catch (error) {
      showError(`闭环任务读取失败：${error.message}`);
    }
  }

  function showError(message) {
    const target = element('loop-error');
    target.hidden = false;
    target.textContent = message;
    announce(message);
  }

  function schedulePoll() {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
    if (!currentJob || !ACTIVE.has(currentJob.status) || element('loop-page')?.hidden) return;
    pollTimer = setTimeout(async () => {
      pollTimer = null;
      try {
        await refresh(false);
      } catch (error) {
        showError(`闭环刷新失败：${error.message}`);
      }
    }, 1100);
  }

  async function refresh(includeRuntime = true) {
    if (!request || refreshing) return;
    refreshing = true;
    try {
      const responses = includeRuntime
        ? await Promise.all([request('/api/loop-runtime'), request('/api/loop-jobs')])
        : [null, await request('/api/loop-jobs')];
      if (responses[0]) runtime = responses[0].runtime || {};
      jobs = Array.isArray(responses[1].jobs) ? responses[1].jobs : [];
      renderRuntime();
      renderJobs();
      if (!selectedId || !jobs.some(job => job.id === selectedId)) {
        selectedId = jobs.find(job => ACTIVE.has(job.status))?.id || jobs[0]?.id || null;
      }
      if (selectedId) await selectJob(selectedId);
      else schedulePoll();
    } catch (error) {
      showError(`闭环服务不可用：${error.message}`);
      throw error;
    } finally {
      refreshing = false;
    }
  }

  async function start() {
    const button = element('loop-start');
    button.disabled = true;
    try {
      const value = await request('/api/loop-jobs', {});
      selectedId = value.job?.id;
      if (!selectedId) throw new Error('服务未返回任务编号。');
      announce('闭环任务已创建，正在准备已钉住的模型、仿真与证据环境。');
      await refresh();
    } catch (error) {
      showError(`闭环启动失败：${error.message}`);
      renderRuntime();
    }
  }

  async function stop() {
    if (!selectedId) return;
    const button = element('loop-stop');
    button.disabled = true;
    try {
      const value = await request(`/api/loop-jobs/${encodeURIComponent(selectedId)}/stop`, {});
      if (value.job) renderJob(value.job);
      announce('已请求停止后续动作；已进入的同步仿真步结束后保留最后反馈，再完成证据收尾。这不代表实机停止。');
      schedulePoll();
    } catch (error) {
      showError(`停止请求失败：${error.message}`);
    }
  }

  function deactivate() {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
  }

  function initializeLoopUI(options) {
    request = options?.api;
    announce = options?.notice || (() => {});
    if (typeof request !== 'function') throw new Error('闭环界面缺少 API 客户端。');
    if (!initialized) {
      initialized = true;
      try {
        viewer = new window.SentinelNativeViewer(element('loop-viewer'));
        viewer.clear();
      } catch (error) {
        element('loop-view-empty').querySelector('strong').textContent = '三维视图不可用';
        element('loop-view-empty').querySelector('small').textContent = error.message;
      }
      element('loop-start').onclick = start;
      element('loop-stop').onclick = stop;
      element('loop-refresh').onclick = () => refresh().catch(() => {});
      element('loop-reset-view').onclick = () => viewer?.reset();
      element('loop-timeline').oninput = event => {
        if (!viewerFrames.length) return;
        const index = Number(event.target.value);
        viewer?.draw(index);
        const frame = viewerFrames[index] || {};
        element('loop-time').textContent = `观测 ${first(frame.observation_step, frame.step, index)} · ${index + 1}/${viewerFrames.length} 帧`;
      };
    }
    return {refresh, deactivate};
  }

  function load() {
    if (!request) {
      const workbench = window.SentinelWorkbench;
      if (!workbench?.api) throw new Error('闭环界面缺少工作台 API 客户端。');
      initializeLoopUI({api: workbench.api, notice: workbench.notice});
    }
    return refresh();
  }

  window.initializeLoopUI = initializeLoopUI;
  window.refreshLoopUI = refresh;
  window.SentinelLoopUI = {initialize: initializeLoopUI, load, refresh, deactivate};
})();
