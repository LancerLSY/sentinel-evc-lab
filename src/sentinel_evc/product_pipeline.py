"""One real numeric lifecycle shared by CLI, HTTP and persisted replay.

Wall time measures the host worker. Numeric time advances one dt only for each
actually observed command. Neither clock represents measured robot dynamics.
"""
from __future__ import annotations
import copy
import json
import threading
import time
from dataclasses import replace
from .authority import Authority
from .contracts import Context, Plan, Rejection, Snapshot, canonical_json, sha256_hex
from .delta_cert import CertificateStore, establish_root
from .events import EventLog
from .geometry import full_check
from .evidence import build_bundle, verify_bundle
from .executor import Executor, ExecutorState
from .sim_controller import SimController
from .numeric_world import make_numeric_case, generate_candidates, plant_step
from .prediction import build_default_numeric_artifacts, evaluate_candidates, slice_prediction
from .runstore import ACTIVE, TERMINAL

class ProductManager:
    def __init__(self,store,realtime=True):
        self.store=store
        self.realtime=realtime
        self._lock=threading.RLock()
        self._sessions={}
        self._closed=False

    def start(self,scenario):
        with self._lock:
            if self._closed:raise RuntimeError('服务已停止。')
            if any(s.thread.is_alive() or s.record['status'] in ACTIVE for s in self._sessions.values()):
                raise RuntimeError('已有活动运行，请先完成或停止它。')
            record=self.store.create(scenario)
            session=ProductSession(self,scenario,record)
            self._sessions[record['id']]=session
            completed=[key for key,value in self._sessions.items() if value.record['status'] in TERMINAL and not value.thread.is_alive()]
            for key in completed:self._sessions.pop(key)
            session.thread.start()
            return copy.deepcopy(record)

    def read(self,run_id):
        with self._lock:
            session=self._sessions.get(run_id)
            return session.snapshot() if session and (session.thread.is_alive() or (session.record.get('error') or {}).get('code')=='EVIDENCE_FINALIZATION_FAILED') else self.store.read(run_id)

    def events(self,run_id):
        with self._lock:
            session=self._sessions.get(run_id)
            return copy.deepcopy(session.log.events()) if session and session.thread.is_alive() else self.store.events(run_id)

    def stop(self,run_id):
        with self._lock:
            session=self._sessions.get(run_id)
            if not session:raise ValueError('run is not live')
            with session.condition:
                if session.record['status'] not in {'queued','preparing','running'}:raise ValueError('cannot stop')
                session.stop_requested=True
                session.record['status']='stopping'
                session.condition.notify_all()
            return session.snapshot()

    def resume(self,run_id,operator_approved):
        with self._lock:
            session=self._sessions.get(run_id)
            if not session or not operator_approved:raise ValueError('approval required')
            with session.condition:
                if session.record['status']!='stopped':raise ValueError('cannot resume')
                session.resume_requested=True
                session.record['status']='preparing'
                session.condition.notify_all()
            return session.snapshot()

    def close(self):
        with self._lock:
            self._closed=True
            sessions=list(self._sessions.values())
        for session in sessions:
            with session.condition:
                session.shutdown_requested=True
                session.stop_requested=True
                session.condition.notify_all()
        for session in sessions:
            session.thread.join(timeout=5)

class ProductSession:
    def __init__(self,manager,scenario,record):
        self.manager,self.store,self.scenario,self.record=manager,manager.store,scenario,record
        self.condition=threading.Condition(threading.RLock())
        self.log=EventLog(record['id'],schema_version='product-v1')
        self.stop_requested=False
        self.resume_requested=False
        self.shutdown_requested=False
        self.thread=threading.Thread(target=self.run,name='sentinel-'+record['id'][-8:],daemon=True)
        self.start_wall=time.monotonic()
        self.clock=time.monotonic_ns()
        self.sample_counter=0
        self.observed_index=0
        self.accepted_index=0
        self.position=(0.,0.,0.)
        self.velocity=0.
        self.result={'candidates':[],'selected':None,'observed_history':[],'steps':[],'cursors':{'submitted':0,'accepted':0,'observed':0},'model':{},'calibration':{},'wall_seconds':0.,'numeric_time':0.,'timing_scope':'observed-command dt; host wall time separate','prediction_scope':'fixed numeric candidate profile','verification':None}
        self.predictions=[]
        self.record['result']=self.result

    def snapshot(self):
        with self.condition:return copy.deepcopy(self.record)

    def persist(self):
        with self.condition:
            self.result['wall_seconds']=round(time.monotonic()-self.start_wall,6)
            self.result['numeric_time']=self.observed_index*.05
            self.store.save(self.record)
            self.store.save_asset(self.record['id'],'events.jsonl',self.log.to_jsonl().encode('utf-8'))
            self.store.save_asset(self.record['id'],'replay.json',self.result['steps'])

    def now(self):
        # In fast test mode a deterministic logical monotonic clock exercises leases.
        if self.manager.realtime:
            self.clock=max(self.clock+1,time.monotonic_ns())
        else:self.clock+=50_000_000
        return self.clock

    def snapshot_feedback(self,controller,now):
        self.sample_counter+=1
        feedback=controller.read_feedback(now)
        return Snapshot('obs-'+str(self.sample_counter),tuple(feedback['position']),now,supported=False)

    def observe(self,controller,case,executor):
        with self.condition:
            self._observe(controller,case,executor)

    def _observe(self,controller,case,executor):
        for item in controller.accepted[self.accepted_index:]:
            self.log.append('CONTROLLER_ACK',generation=item['gen'],action=list(item['action']))
        self.accepted_index=len(controller.accepted)
        for item in controller.observed[self.observed_index:]:
            position=tuple(item['action'])
            velocity=(position[0]-self.position[0])/.05
            acceleration=(velocity-self.velocity)/.05
            self.r,self.v=plant_step(self.r,self.v,acceleration,case.evaluator_params,.05)
            self.position,self.velocity=position,velocity
            self.observed_index+=1
            sample=json.loads(canonical_json({'step':self.observed_index,'time':self.observed_index*.05,'r':self.r,'v':self.v,'a':acceleration}))
            self.result['observed_history'].append(sample)
            self.result['steps'].append({'step':self.observed_index,'position':list(position),'cursors':dict(controller.cursors()),'state':executor.state,'payload_r':sample['r']})
            self.log.append('OBSERVED',**sample,position=list(position),generation=item['gen'])
            if self.root_prediction is not None:
                idx=self.observed_index-1
                if not self.root_prediction.lower[idx] <= self.r <= self.root_prediction.upper[idx]:
                    raise Rejection('PREDICTION_ENVELOPE','实际数值后果超出根预测包络')
        self.result['cursors']=controller.cursors()

    def advance(self,controller,case,executor,context):
        now=self.now()
        snapshot=self.snapshot_feedback(controller,now)
        executor.tick(now,snapshot,context)
        self.observe(controller,case,executor)
        self.persist()
        if self.manager.realtime:time.sleep(.05)

    def suffix(self,plan,offset):
        events=tuple((i-offset,event) for i,event in plan.gripper_events if i>=offset)
        return Plan(plan.plan_id+'-suffix-'+str(offset),plan.knots[offset:],plan.dt,events,plan.descriptor)

    def run(self):
        controller=executor=case=None
        final_status="failed"
        self.root_prediction=None
        try:
            with self.condition:self.record['status']='preparing'
            self.persist()
            case=make_numeric_case(self.scenario.seed)
            self.r,self.v=case.current_r,case.current_v
            model,calibration=build_default_numeric_artifacts(mode=self.scenario.prediction_mode)
            self.model,self.calibration=model,calibration
            self.result['model']={**model.summary(),'id':model.hash}
            self.result['calibration']={**calibration.summary(),'id':calibration.hash}
            plans=generate_candidates(self.scenario.displacement)
            def geometry_check(candidate):
                ok,margins,_=full_check(candidate,self.scenario.scene)
                return ok,min(margins)
            evaluation=evaluate_candidates(case.history,plans,model,calibration,self.now(),risk_limit=self.scenario.risk_limit,physical_check=geometry_check,ttl_ns=60_000_000_000)
            for row in evaluation.candidates:
                candidate=row.summary()
                candidate['duration']=float(candidate['duration'])
                self.result['candidates'].append(candidate)
            self.predictions=[row.prediction.summary() for row in evaluation.candidates if row.prediction is not None]
            for row in evaluation.candidates:
                self.log.append('PROPOSAL',candidate_id=row.plan.plan_id,plan_hash=row.plan.hash)
                self.log.append('PREDICTION',candidate_id=row.plan.plan_id,status=row.status,prediction_hash=row.prediction.hash if row.prediction else None,reason=row.reason)
            selected=evaluation.selected
            if selected is None:
                final_status='rejected'
                self.record['error']={'code':'NO_ALLOWED_CANDIDATE','message':'四个候选均未通过门控。'}
                return
            plan=selected.plan
            self.root_prediction=selected.prediction
            self.record['selected']=self.result['selected']=plan.plan_id
            controller=SimController(capacity=2,ack_delay_ticks=1,exec_delay_ticks=1,initial_position=plan.knots[0])
            certificate_store=CertificateStore()
            authority=Authority(certificate_store,events=self.log)
            executor=Executor(authority,controller,self.log)
            context=Context(scene_id=self.scenario.scene.scene_id)
            self.record['status']='running'
            self.persist()
            while self.observed_index < plan.horizon:
                if self.stop_requested:
                    executor.revoke('operator_stop' if not self.shutdown_requested else 'server_shutdown')
                    for _ in range(10):
                        self.advance(controller,case,executor,context)
                        if executor.poll_cancel() is True and controller.free_slots()==controller.capacity:break
                    else:raise Rejection('CANCEL_UNCONFIRMED','取消未确认或控制器未排空')
                    if self.shutdown_requested:raise Rejection('PROCESS_STOPPED','服务停止，运行已安全撤销')
                    with self.condition:
                        self.stop_requested=False
                        self.record['status']='stopped';self.persist()
                        while not self.resume_requested and not self.shutdown_requested:self.condition.wait(.2)
                    if self.shutdown_requested:raise Rejection('PROCESS_STOPPED','服务停止，运行未自动恢复')
                    context=replace(context,epoch=executor.generation,queue_rev=context.queue_rev+1,committed_prefix_hash=sha256_hex(plan.knots[:self.observed_index+1]))
                    now=self.now()
                    executor.try_recover(True,self.snapshot_feedback(controller,now),context,now)
                    self.resume_requested=False
                    if self.stop_requested:continue
                    self.record['status']='running'
                    if self.observed_index==plan.horizon:break
                offset=self.observed_index
                suffix=self.suffix(plan,offset)
                now=self.now()
                prediction=slice_prediction(self.root_prediction,plan,suffix,offset,now)
                self.predictions.append(prediction.summary())
                validation=establish_root(suffix,self.scenario.scene)
                if validation.verdict!='FULL':raise Rejection('GEOMETRY_VIOLATION','后缀几何复验失败')
                certificate_store.register(validation.certificate)
                authority.register_prediction(prediction)
                self.log.append('CERTIFICATE',**validation.certificate.summary())
                self.log.append('PREDICTION',prediction_hash=prediction.hash,root_prediction_hash=self.root_prediction.hash,suffix_offset=offset,plan_hash=suffix.hash)
                snapshot=self.snapshot_feedback(controller,now)
                count=min(4,suffix.horizon)
                lease=authority.prepare(suffix,validation.certificate,context,snapshot,now,prefix_len=count,ttl_ns=750_000_000,prediction=prediction,require_prediction=True)
                self.log.append('PREPARE',**lease.payload())
                executor.commit(lease,suffix,snapshot,context,now)
                target=offset+count
                for _ in range(20):
                    if self.stop_requested:break
                    self.advance(controller,case,executor,context)
                    if self.observed_index==target and controller.free_slots()==controller.capacity:break
                else:raise Rejection('EXECUTION_TIMEOUT','数值前缀未及时排空')
                if self.stop_requested:continue
                if self.observed_index!=target:raise Rejection('INCOMPLETE_PREFIX','数值前缀未完整执行')
                context=replace(context,queue_rev=context.queue_rev+1,committed_prefix_hash=sha256_hex(plan.knots[:self.observed_index+1]))
            final_status='completed'
        except Rejection as exc:
            final_status='failed'
            self.record['status']='stopping'
            self.record['error']={'code':exc.code,'message':exc.detail}
            if executor and controller:
                executor.revoke(exc.code)
                # Already accepted work may still execute. Always record its observed tail.
                for _ in range(10):
                    try:self.advance(controller,case,executor,Context(epoch=executor.generation))
                    except Rejection:pass
                    if controller.free_slots()==controller.capacity and controller.cancel_acked is True:break
                self.result['cursors']=controller.cursors()
        except Exception as exc:
            final_status='failed'
            self.record['status']='stopping'
            self.record['error']={'code':'INTERNAL_FAILURE','message':'运行失败；没有自动批准后续动作。'}
            if executor and controller:
                executor.revoke('internal_failure')
                for _ in range(10):
                    try:self.advance(controller,case,executor,Context(epoch=executor.generation))
                    except Rejection:pass
                    if controller.is_drained and controller.cancel_acked is True:break
            # Keep sanitized public diagnostics, with exception type only.
            self.log.append('BACKUP',reason=type(exc).__name__)
        finally:
            try:
                self.finalize(final_status)
            except Exception:
                # Motion is already complete or cancelled/drained. A failed export
                # never converts the run to success or holds the whole service active.
                with self.condition:
                    self.record['status']='failed'
                    self.record['error']={'code':'EVIDENCE_FINALIZATION_FAILED','message':'证据保存或核验失败，本次运行未获完成确认。'}
                    self.record['verification']={'ok':False,'message':'EVIDENCE_FINALIZATION_FAILED','trust':'self-contained demo key; external trust required'}
                try:self.store.save(self.record)
                except Exception:pass

    def finalize(self,final_status):
        self.result['status']=final_status
        self.result['error']=self.record['error']
        self.result['scope']=self.record['scope']
        self.log.append('OUTCOME',status=final_status,cursors=self.result['cursors'],error=self.record['error'])
        self.persist()
        assets={'scenario.json':self.scenario.summary(),'result.json':self.result,'model.json':self.result['model'],'calibration.json':self.result['calibration'],'predictions.json':self.predictions,'replay.json':self.result['steps']}
        for name,data in assets.items():self.store.save_asset(self.record['id'],name,data)
        encoded={name:json.dumps(data,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8') for name,data in assets.items()}
        bundle=build_bundle(self.log,str(self.store.directory(self.record['id'])),encoded)
        ok,message=verify_bundle(bundle['bundle_dir'],bundle['public_key'],self.record['id'])
        if not ok:raise ValueError('evidence verification failed')
        self.record['status']=final_status
        self.record['verification']={'ok':True,'message':message,'trust':'self-contained demo key; external trust required'}
        self.store.save(self.record)
