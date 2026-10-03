'use strict';

window.SentinelLaunchUI = (() => {
  const el = id => document.getElementById(id);
  const esc = value => window.SentinelWorkbench.escapeHTML(value);
  const codes = {CAMERA_BINDING_CHANGED:'相机对应关系改变',STATE_BINDING_CHANGED:'状态顺序、单位或数值改变',CURSOR_CHANGED:'动作块位置改变',ACTION_SCHEMA_CHANGED:'动作格式改变',ACTION_BYTES_CHANGED:'最终动作改变',INPUT_MISMATCH:'两次检查的输入不同',SUITE_MISMATCH:'探针组不同',PROBE_ALIGNMENT_MISMATCH:'探针不齐全',INVALID_PROBE_PACK:'记录格式不完整',ADAPTER_DIGEST_CHANGED:'代码身份改变，探针行为一致'};
  let inputs = null, generation = 0, exportURLs = [];
  function clear() { generation++; inputs=null; exportURLs.forEach(url=>URL.revokeObjectURL(url)); exportURLs=[]; el('launch-export').hidden=true; el('launch-result').replaceChildren(); el('launch-status').textContent='输入已改变，请重新检查。'; }
  function diagnosis(report) {
    const value=report.diagnosis;
    if(!value?.patterns?.length)return '';
    const transforms={identity:'数值相同',sign:'符号相反',radians_to_degrees:'乘以约 57.296',degrees_to_radians:'乘以约 0.01745',minus_one_one_to_zero_one:'从 [-1,1] 转为 [0,1]',zero_one_to_minus_one_one:'从 [0,1] 转为 [-1,1]'};
    const lines=value.patterns.flatMap(pattern=>{
      if(pattern.kind==='camera_route')return pattern.mapping.map(item=>`候选 ${item.candidate_key} 收到的图像指纹，对应参考 ${item.reference_key}。检查相机路由。`);
      if(pattern.kind==='cursor_offset')return Object.entries(pattern.offsets).map(([key,offset])=>`${key==='action_index'?'动作索引':'动作块编号'}固定偏移 ${offset>0?'+':''}${offset}。检查队列选取位置。`);
      const label=pattern.kind==='state_transform'?'状态':'动作';
      return pattern.mapping.map(item=>`候选${label}分量 ${item.destination_index} 与参考分量 ${item.reference_index} ${item.transform==='scale_offset'?`近似满足 候选 = ${Number(item.scale.toPrecision(5))} × 参考 + ${Number(item.offset.toPrecision(5))}`:transforms[item.transform] || item.transform}。检查分量顺序和转换。`);
    });
    return `<article class="compare-difference"><strong>接线与转换线索 · ${esc(value.matched_probes)} 个探针</strong><ul>${lines.map(line=>`<li>${esc(line)}</li>`).join('')}</ul><small>这些数值关系在全部配对探针中出现。它们是排查线索，还需检查实际接入代码才能确定原因。修正后重新采集。</small></article>`;
  }
  function render(report) {
    const label={PASS:'通过：本组探针的接入行为一致',BLOCK:'阻止启动：接入行为发生改变',REVIEW:'暂停启动：需要补齐可比较的输入'}[report.verdict];
    el('launch-status').textContent=label;
    el('launch-result').innerHTML=`<div class="launch-verdict ${esc(report.verdict.toLowerCase())}"><strong>${esc(report.verdict)}</strong><span>检查 ${esc(report.scope.probe_count)} 个固定输入</span></div>${diagnosis(report)}${report.issues.map(issue=>`<article class="compare-difference"><strong>${esc(codes[issue.code] || issue.code)}</strong><p>探针 ${esc(issue.probe_id || '配置')} · ${esc(issue.field)}${issue.action_index!==undefined?` · 动作分量 ${esc(issue.action_index)}`:''}</p><div class="compare-source-grid"><div><small>参考值</small><pre>${esc(JSON.stringify(issue.reference,null,2))}</pre></div><div><small>待检查值</small><pre>${esc(JSON.stringify(issue.candidate,null,2))}</pre></div></div></article>`).join('')}<p class="compare-explanation">检查覆盖这组探针。导出的签名包可在无模型、无 GPU 的机器上重新计算本次判定。接入源和传感器真实性由采集方负责。</p>`;
  }
  async function check(event) {
    event.preventDefault(); clear(); const current=generation;
    const left=el('launch-reference').files[0],right=el('launch-candidate').files[0];
    if (!left || !right) return;
    if ([left,right].some(file=>file.size>8*1024*1024)) {el('launch-status').textContent='每个文件最多 8 MiB。';return;}
    const button=el('launch-submit');button.disabled=true;
    el('launch-status').textContent='正在核对同一组输入和实际接入输出…';
    try {
      const [reference_json,candidate_json]=await Promise.all([left.text(),right.text()]);
      const value={reference_json,candidate_json};
      const report=await window.SentinelWorkbench.api('/api/launch-check',value);
      if (current!==generation) return;
      inputs=value;render(report);el('launch-export').hidden=false;
    } catch(error) {if(current===generation) el('launch-status').textContent=`检查失败：${error.message}`;}
    finally {button.disabled=false;}
  }
  function downloadLink(encoded,name,type,label) {
    const bytes=Uint8Array.from(atob(encoded),char=>char.charCodeAt(0));
    const url=URL.createObjectURL(new Blob([bytes],{type}));
    exportURLs.push(url);
    const link=document.createElement('a');link.href=url;link.download=name;
    link.className='quiet-pill';link.textContent=label;return link;
  }
  async function download() {
    if(!inputs)return;const current=generation;el('launch-export').disabled=true;
    try {
      const value=await window.SentinelWorkbench.api('/api/launch-capsule',inputs);
      if(current!==generation)return;
      el('launch-downloads')?.remove();exportURLs.forEach(url=>URL.revokeObjectURL(url));exportURLs=[];
      const links=document.createElement('div');links.id='launch-downloads';links.className='compare-evidence-actions';
      links.append(downloadLink(value.capsule_base64,value.run_id+'.zip','application/zip','保存判定 ZIP'),downloadLink(value.public_key_base64,value.run_id+'.public','application/octet-stream','保存独立公钥'));
      el('launch-result').append(links);
      el('launch-status').textContent=`签名包已生成：${value.run_id}。请分别保存 ZIP 和公钥。解压后运行 launch-reproduce --bundle bundle --public-key PUBLIC --run-id ${value.run_id}`;
    } catch(error) {if(current===generation) el('launch-status').textContent=`导出失败：${error.message}`;}
    finally {el('launch-export').disabled=false;}
  }
  function init() {
    el('launch-form').onsubmit=check;el('launch-export').onclick=download;
    for(const id of ['launch-reference','launch-candidate'])el(id).onchange=clear;
  }
  document.addEventListener('DOMContentLoaded',init);
  return {render};
})();
