'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '∅').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const names = {
    unchanged:['Unchanged integration','接入配置未改动'],identity_only:['Code identity changed; behavior preserved','代码身份改变，接入行为相同'],mismatched_input:['Different input: review required','输入不同，需要重新采集'],
    camera_swap:['Two cameras swapped','两路相机对调'],camera_route_renamed:['Camera route renamed','相机名称改变'],state_index_swap:['State components swapped','状态分量顺序错位'],state_units_scaled:['State scale changed ×1,000','状态数值放大 1,000 倍'],
    action_sign:["Action direction inverted","动作方向反号"],chunk_offset:["Action index advanced","动作索引提前一位"],action_order_swap:['Action components swapped','动作分量顺序错位'],sign_inversion:['Action direction inverted','动作方向反号'],gripper_inversion:['Gripper convention inverted','夹爪方向反号'],chunk_off_by_one:['Next action selected too early','动作索引提前一位'],postprocess_scale:['Final action scale halved','最终动作缩小一半'],action_byte_change:['One final action bit changed','最终动作的一个比特改变']
  };
  const copy = {
    en:{nav:"Launch Gate",kicker:'LAUNCH GATE / BEFORE MOTION',title:'Find the integration error. See where to fix it.',lead:'Keep your VLA model. Before an update controls an arm, run fixed inputs through the actual integration. Compare the cameras and state consumed by the policy, the selected action, and the final bytes prepared for the writer.',stage:'Explore retained checks',case:'Integration change',probe:'Captured input',status:'Qualification',field:'First changed field',left:'Reference',right:'Candidate',cameras:'Camera → policy input',actions:'Final action values',actionScope:'Native model action units. Each row uses its own scale; center = 0. These are requests, not measured motion.',clues:'Where to look',clueScope:'The patterns recur across the paired inputs. Inspect the integration code to establish the cause, then recapture.',noClue:'No unambiguous routing or numeric pattern was found. Inspect the exact changed field.',pass:'Bindings, action cursor and final bytes match on this probe suite.',review:'Use the same input bytes and a complete paired suite before launch.',writer:'Downstream software calls',scope:'This explorer displays retained results. It runs no model or robot command in your browser. PASS covers the fixed probes; task performance and physical safety need separate validation.',download:'Download checks, signed capsules and public keys',guide:'Use Launch Gate in CLI or App',raw:'Exact retained report',chartData:'Full results and source identities',laneLocal:'Signed-record adapter replay',laneGpu:'Fresh GPU inference',metricFault:'injected integration configurations blocked',metricProbe:'fixed inputs in the selected suite',metricWriter:'downstream calls across blocked cases',metricSize:'camera-swap decision capsule · uncompressed',value1:'Catch errors before the first motion',value1b:'The check runs before env.step. You can inspect a camera swap or reversed action without moving the arm.',value2:'Get a specific repair clue',value2b:'Compare both values. Identify a camera route, swapped components, a sign change, a scale or an action-index offset.',value3:'Share a small decision package',value3b:'Another machine can verify the signature and recompute the verdict on a CPU. Model weights are unnecessary for this check.',position:'A different job from an authorization gate',positionBody:'RLSOK already binds approved model, processors, action contracts and runtime. KineGrant already supplies authorization and receipts. Launch Gate checks the observed VLA integration on fixed inputs and returns repair clues. The published authorization comparison remains a tie.',compare:'Pinned interfaces and actual comparison calls',recorded:'Recorded mapping replay: 10 task windows × 2 requests. Camera commitments are retained; vision inference is not rerun in this lane.'},
    zh:{nav:"启动检查",kicker:'LAUNCH GATE / 机械臂运动之前',title:'找出接入错误，指出具体改哪里。',lead:'继续用你现有的 VLA 模型。更新配置后，先把固定输入送进实际接入流程。检查模型实际收到的相机和状态、选中的动作，以及准备交给写入器的最终字节。',stage:'查看保存的检查结果',case:'接入配置变化',probe:'保存的输入',status:'检查结论',field:'第一个不同的字段',left:'参考配置',right:'待检查配置',cameras:'相机 → 模型输入',actions:'最终动作数值',actionScope:'单位为原生模型动作单位。每行单独缩放，中线为零。图中显示请求，不代表实际运动。',clues:'具体排查线索',clueScope:'这些关系在全部配对输入中出现。检查实际接入代码以确定原因，修正后重新采集。',noClue:'未发现唯一的路由或数值关系。请检查上方改变的具体字段。',pass:'在本组输入上，相机和状态绑定、动作索引与最终字节一致。',review:'先补齐完整的配对记录，并让两次检查使用相同输入。',writer:'下游软件调用次数',scope:'这个演示展示保存的结果。浏览器不运行模型，也不发送机械臂指令。通过只覆盖这组固定输入；任务表现与真机安全需要另行验证。',download:'下载检查数据、签名判定包与公钥',guide:'在 CLI 或 App 中使用',raw:'保存的完整判定',chartData:'全部结果与源码身份',laneLocal:'签名记录的接入回放',laneGpu:'真实 GPU 推理',metricFault:'注入的接入故障配置被拦住',metricProbe:'所选检查组中的固定输入',metricWriter:'被阻止案例的下游调用总数',metricSize:'相机对调判定包 · 未压缩',value1:'第一步动作前就查错',value1b:'在 env.step 之前检查。相机对调、动作反向，都可以先查看结果，不用让机械臂试错。',value2:'给出具体排查位置',value2b:'并排显示两个值，找出相机路由、分量错位、反号、缩放和动作索引偏移。',value3:'小文件就能重新核对',value3b:'另一台电脑只用 CPU 就能核验签名、重新计算判定。核对判定不需要模型权重。',position:'解决授权之外的接入问题',positionBody:'RLSOK 已经绑定获准的模型、处理器、动作契约和运行时；KineGrant 已经提供授权与收据。Launch Gate 用固定输入检查 VLA 实际接入行为，并给出排查线索。公开的普通授权对比仍为平局。',compare:'固定版本的接口与真实调用对照',recorded:'记录回放：10 个任务窗口，每个窗口 2 个请求。这一路保留相机指纹，不重新运行视觉推理。'}
  };
  let data, lane='recorded', chosen='camera_swap', probeIndex=0, locale='en';
  const tr = key => copy[locale][key];
  function clues(report) {
    return (report.diagnosis?.patterns || []).flatMap(pattern => {
      if(pattern.kind==='camera_route')return pattern.mapping.map(item=>locale==='zh'?`待检查的 ${item.candidate_key} 收到了参考 ${item.reference_key} 的图像。检查相机路由。`:`Candidate ${item.candidate_key} received the image bound to reference ${item.reference_key}. Check camera routing.`);
      if(pattern.kind==='cursor_offset')return Object.entries(pattern.offsets).map(([key,value])=>locale==='zh'?`${key} 固定偏移 ${value>0?'+':''}${value}。检查动作队列的选取位置。`:`${key} has a constant offset of ${value>0?'+':''}${value}. Check action selection.`);
      return pattern.mapping.map(item=>{
        const label=pattern.kind==='state_transform'?(locale==='zh'?'状态':'State'):(locale==='zh'?'动作':'Action');
        return locale==='zh'?`${label}分量 ${item.destination_index} = 参考分量 ${item.reference_index} × ${Number(item.scale.toPrecision(6))} + ${Number(item.offset.toPrecision(6))}。`:`${label} component ${item.destination_index} = reference component ${item.reference_index} × ${Number(item.scale.toPrecision(6))} + ${Number(item.offset.toPrecision(6))}.`;
      });
    });
  }
  function actionValues(probe){
    const raw=probe.action.bytes_hex.match(/../g).map(value=>parseInt(value,16));
    const view=new DataView(Uint8Array.from(raw).buffer),step=probe.action.dtype==='float64'?8:4;
    return Array.from({length:raw.length/step},(_,index)=>step===8?view.getFloat64(index*step,true):view.getFloat32(index*step,true));
  }
  const short=value=>value.startsWith('sha256:')?value.slice(7,19):value;
  function render(){
    if(!data)return;
    document.querySelectorAll('[data-launch-copy]').forEach(node=>node.textContent=tr(node.dataset.launchCopy));
    const suite=data[lane],cases=suite.cases;
    if(!cases.some(item=>item.case===chosen))chosen=cases[0].case;
    $('launch-case').innerHTML=cases.map(item=>`<option value="${escape(item.case)}" ${item.case===chosen?'selected':''}>${escape((names[item.case]||[item.case,item.case])[locale==='zh'?1:0])} · ${item.report.verdict}</option>`).join('');
    const item=cases.find(item=>item.case===chosen),ref=suite.reference.probes,candidate=item.probes;
    probeIndex=Math.min(probeIndex,ref.length-1);
    $('launch-probe').innerHTML=ref.map((probe,index)=>`<option value="${index}" ${index===probeIndex?'selected':''}>${escape(probe.id)}</option>`).join('');
    const a=ref[probeIndex],b=candidate[probeIndex];
    const report=item.report,issue=report.issues[0];
    $('launch-verdict').className='launch-status '+report.verdict.toLowerCase();
    $('launch-verdict').textContent=report.verdict;
    $('launch-calls').textContent=String(item.writer.downstream_calls);
    $('launch-field').textContent=issue ? issue.field + (issue.probe_id ? ' · '+issue.probe_id : '') : tr('pass');
    $('launch-left').textContent=JSON.stringify(issue?.reference??(report.verdict==='PASS'?'MATCH':null),null,2);
    $('launch-right').textContent=JSON.stringify(issue?.candidate??(report.verdict==='PASS'?'MATCH':null),null,2);
    const cameraKeys=[...new Set([...Object.keys(a.consumed.cameras),...Object.keys(b.consumed.cameras)])];
    $('launch-camera-map').innerHTML=cameraKeys.map(key=>{
      const left=a.consumed.cameras[key],right=b.consumed.cameras[key],source=Object.entries(a.consumed.cameras).find(([,hash])=>hash===right)?.[0];
      return `<div class="launch-route ${left===right?'same':'changed'}"><code>${escape(key)}</code><span>${escape(left?short(left):'∅')} <b aria-hidden="true">→</b> ${escape(right?short(right):'∅')}</span>${source&&left!==right?`<small>${escape(locale==='zh'?'来自参考输入：':'From reference input: ')}${escape(source)}</small>`:''}</div>`;
    }).join('');
    const valuesA=actionValues(a),valuesB=actionValues(b);
    const bar=(value,scale,label)=>{const width=Math.abs(value)/scale*50;return `<span class="${label}" style="width:${width}%;left:${value<0?50-width:50}%"></span>`;};
    $('launch-action-bars').innerHTML=valuesA.map((value,index)=>{const scale=Math.max(1e-12,Math.abs(value),Math.abs(valuesB[index]));return `<div class="launch-bar-row"><code>${index}</code><div class="launch-bar-pair">${bar(value,scale,'reference')}${bar(valuesB[index],scale,'candidate')}</div><span class="${value===valuesB[index]?'':'changed-value'}">${escape(value.toPrecision(6))}<br>${escape(valuesB[index].toPrecision(6))}</span></div>`;}).join('');
    const lines=clues(report);
    $('launch-clues').innerHTML=lines.length?`<ul>${lines.map(line=>`<li>${escape(line)}</li>`).join('')}</ul>`:`<p>${escape(tr(report.verdict==='PASS'?'pass':report.verdict==='REVIEW'?'review':'noClue'))}</p>`;
    $('launch-raw').textContent=JSON.stringify(report,null,2);
    $('launch-lane-note').textContent=lane==='recorded'?tr('recorded'):(locale==='zh'?suite.description_zh:suite.description_en);
    $('launch-lane').querySelectorAll('option').forEach(option=>option.textContent=tr(option.value==='recorded'?'laneLocal':'laneGpu'));
    $('launch-metric-faults').textContent=suite.acceptance.semantic_faults_blocked+'/'+suite.acceptance.semantic_fault_count;
    $('launch-metric-probes').textContent=ref.length;
    $('launch-metric-writer').textContent=suite.acceptance.blocked_downstream_calls;
    $('launch-metric-size').textContent=(suite.cases.find(c=>c.case==='camera_swap').capsule.bytes/1024).toFixed(1)+' KiB';
  }
  $('launch-case').onchange=event=>{chosen=event.target.value;probeIndex=0;render();};
  $('launch-probe').onchange=event=>{probeIndex=Number(event.target.value);render();};
  $('launch-lane').onchange=event=>{lane=event.target.value;probeIndex=0;render();};
  window.addEventListener('sentinel-language',event=>{locale=event.detail.locale;render();});
  locale=document.body.lang==='zh'?'zh':'en';
  fetch('assets/launch-gate.json').then(response=>{if(!response.ok)throw Error('data');return response.json();}).then(value=>{data=value;if(data.gpu){$('launch-lane').add(new Option(tr('laneGpu'),'gpu'));lane='gpu';$('launch-lane').value=lane;}render();}).catch(()=>{$('launch-field').textContent=locale==='zh'?'数据暂时无法加载，请下载完整结果。':'Data unavailable. Download the retained results.';});
})();
