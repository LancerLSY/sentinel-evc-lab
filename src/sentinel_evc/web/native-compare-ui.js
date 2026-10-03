'use strict';

window.SentinelCompareUI = (() => {
  const el = id => document.getElementById(id);
  const esc = value => window.SentinelWorkbench.escapeHTML(value);
  const labels = {action:'最终动作', observation:'执行后反馈', cursor:'执行游标', outcome:'任务结果', authorization:'请求授权'};
  let initialized = false, generation = 0, report = null;

  function clear() {
    generation += 1; report = null;
    el('native-compare-result').replaceChildren();
    el('native-compare-download').hidden = true;
    el('native-compare-status').textContent = '选择参考运行和待检查运行，重新核验两份来源后比较。';
  }

  function updateRuns(runs) {
    if (!initialized) {
      initialized = true;
      el('native-compare-form').onsubmit = compare;
      for (const id of ['native-compare-left','native-compare-right']) el(id).onchange = clear;
      el('native-compare-download').onclick = download;
    }
    const valid = runs.filter(run => run.verification?.ok === true);
    for (const [side, mode] of [['left','baseline'],['right','active']]) {
      const select = el(`native-compare-${side}`), previous = select.value;
      select.innerHTML = valid.map(run => `<option value="${esc(run.id)}">${esc(run.id)} · ${esc(run.mode || 'VLA')}</option>`).join('');
      if (valid.some(run => run.id === previous)) select.value = previous;
      else select.value = (valid.find(run => run.mode === mode) || valid[side==='left'?0:1] || valid[0])?.id || '';
    }
    el('native-compare-submit').disabled = valid.length < 2;
    if (report && ![report.sources.left.run_id,report.sources.right.run_id].every(id => valid.some(run => run.id===id))) clear();
    if (valid.length < 2) el('native-compare-status').textContent = '先导入两份记录，并分别选择独立公钥。';
  }

  async function compare(event) {
    event.preventDefault(); clear();
    const left = el('native-compare-left').value, right = el('native-compare-right').value;
    if (!left || !right || left===right) { el('native-compare-status').textContent='请选择两份不同的运行记录。'; return; }
    const current = generation, button = el('native-compare-submit');
    button.disabled = true;
    el('native-compare-status').textContent = '正在重新核验两份签名来源，并按任务、初始状态、种子和指令对齐…';
    try {
      const value = await window.SentinelWorkbench.api(`/api/native-runs/${encodeURIComponent(left)}/compare/${encodeURIComponent(right)}`);
      if (current !== generation) return;
      report = value; render(); el('native-compare-download').hidden=false;
    } catch (error) {
      if (current===generation) el('native-compare-status').textContent = `比较未完成：${error.message}`;
    } finally { button.disabled=false; }
  }

  function detailsText(value) {
    return typeof value==='string' ? value : JSON.stringify(value, null, 2);
  }

  function explanation(d) {
    const changed=d.details?.state_leaf_differences || [];
    const paths=changed.map(item=>item.path);
    const camera=paths.length===1 && paths[0]==='pixels.image2';
    const feedback=camera?'第二路相机的反馈指纹不同；其余已记录状态指纹一致。':paths.length?`发生变化的反馈字段：${paths.join('、')}。`:'';
    const context=d.action_equal===true?'该步最终动作字节一致。':d.action_equal===false?'该步最终动作也不同。':'';
    return `${feedback} ${context}${d.previous_observation_equal===true?' 前一步反馈一致。':''}`.trim();
  }

  function sourceCard(side) {
    const source=report.sources[side];
    return `<div><strong>${side==='left'?'参考运行':'待检查运行'} · ${esc(source.run_id)}</strong><small>独立所选公钥 SHA-256</small><code>${esc(source.public_key_sha256)}</code><small>输入 manifest SHA-256</small><code>${esc(source.manifest_sha256)}</code></div>`;
  }

  function render() {
    const s=report.summary;
    const status={EQUIVALENT:'所比较的执行字段一致',DIFFERENCES:'已定位执行差异',INCOMPARABLE:'配置或样本不匹配，不能判为等价'}[s.status] || s.status;
    el('native-compare-status').textContent=`${status} · ${s.paired_episodes} 组配对 · ${s.compared_actions} 次动作`;
    const rows=report.episodes.filter(row => row.first_difference);
    const cards=[['动作差异',s.action_differences],['反馈差异',s.observation_differences],['游标 / 结果差异',`${s.cursor_differences} / ${s.outcome_differences}`],['授权模式变化样本',s.expected_authorization_differences]];
    el('native-compare-result').innerHTML=`
      <div class="compare-stats">${cards.map(([label,count])=>`<div><span>${esc(label)}</span><strong>${esc(count)}</strong></div>`).join('')}</div>
      <p class="compare-explanation">任务成功：参考 ${esc(s.left_successes)} / 待检查 ${esc(s.right_successes)}。随机许可编号和耗时不作为行为差异；预期授权模式变化单独计数。</p>
      ${report.eligibility.compatible?'':`<div class="compare-warning"><strong>只作跨配置或不完整样本诊断</strong><pre>${esc(detailsText(report.eligibility))}</pre></div>`}
      <div class="compare-differences">${rows.length?rows.map((row,index)=>{
        const d=row.first_difference,k=row.key;
        return `<article class="compare-difference"><div class="compare-difference-title"><strong>Task ${esc(k.task_id)} · 初始状态 ${esc(k.initial_state_index)}</strong><span>${esc(labels[d.category] || d.category)} · step ${esc(d.step)}</span></div>
          <p>${esc(k.instruction)}</p>${explanation(d)?`<p>${esc(explanation(d))}</p>`:''}<details><summary>查看差异字段和指纹</summary><pre>${esc(detailsText(d.details))}</pre></details>
          <div class="compare-evidence-actions">${['left','right'].map(side=>{
            const ref=d[`${side}_ref`];
            if(!ref)return '';
            const pose=ref.exact_pose?'该步保留了精确姿态':ref.pose_frame_index!==null?`该步姿态未保留；三维参考 step ${ref.pose_step}`:'该步未保留三维姿态';
            return `<button class="quiet-pill" data-compare-row="${index}" data-compare-side="${side}">${side==='left'?'查看参考记录':'查看待检查记录'} · step ${esc(ref.step)}</button><small>${esc(pose)}</small>`;
          }).join('')}</div></article>`;
      }).join(''):'<p class="compare-no-difference">没有发现非预期执行差异。等价范围以本报告列出的字段、样本和配置为准。</p>'}</div>
      <details class="compare-source"><summary>查看输入身份与核验范围</summary><div class="compare-source-grid">${sourceCard('left')}${sourceCard('right')}</div><p>派生报告可通过 CLI 重新核验输入并重算；报告本身不是新的签名执行记录。</p></details>`;
    el('native-compare-result').querySelectorAll('[data-compare-row]').forEach(button=>{
      button.onclick=async()=>{
        const ref=rows[Number(button.dataset.compareRow)].first_difference[`${button.dataset.compareSide}_ref`];
        button.disabled=true;
        try { await window.SentinelNativeUI.jumpToEvidence(ref); }
        catch(error) { window.SentinelWorkbench.notice(`记录定位失败：${error.message}`); }
        finally { button.disabled=false; }
      };
    });
  }

  function download() {
    if (!report) return;
    const link=document.createElement('a');
    link.href=`/api/native-runs/${encodeURIComponent(report.sources.left.run_id)}/compare/${encodeURIComponent(report.sources.right.run_id)}/report.json`;
    link.download='sentinel-execution-diff.json';link.click();
  }

  return {updateRuns};
})();
