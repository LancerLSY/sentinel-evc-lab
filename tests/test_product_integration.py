import base64
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from sentinel_evc.scenario import Scenario
from sentinel_evc.runstore import RunStore
from sentinel_evc.product_pipeline import ProductManager, ProductSession
from sentinel_evc.server import LocalServer
from sentinel_evc.evidence import verify_bundle
from sentinel_evc.physics_jobs import PhysicsJobs

def wait_status(manager,run_id,states,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        record=manager.read(run_id)
        if record['status'] in states:return record
        time.sleep(.01)
    raise AssertionError('run did not reach expected state')


def test_real_serve_subprocess_nonce_sigterm_and_workspace_reopen(tmp_path):
    workspace = tmp_path / 'real-serve-workspace'
    nonce = '0123456789abcdef0123456789abcdef'
    environment = os.environ.copy()
    environment['SENTINEL_LAUNCH_NONCE'] = nonce
    process = subprocess.Popen(
        [sys.executable, '-m', 'sentinel_evc', 'serve', '--port', '0', '--data-dir', str(workspace)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
    )
    output = bytearray()
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    marker = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and marker is None:
            assert process.poll() is None, output.decode('utf-8', 'replace')
            if not selector.select(timeout=.2):
                continue
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                continue
            output.extend(chunk)
            for line in output.splitlines():
                if line.startswith(b'SENTINEL_READY '):
                    marker = json.loads(line.removeprefix(b'SENTINEL_READY '))
                    break
        assert marker is not None, output.decode('utf-8', 'replace')
        assert marker['nonce'] == nonce
        assert marker['url'].startswith('http://127.0.0.1:')
        with urlopen(marker['url'] + '/api/session', timeout=5) as response:
            assert response.status == 200
            assert json.load(response)['profile'] == 'numeric-simulator-product-v1'
        process.terminate()
        assert process.wait(timeout=15) == 0
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdout is not None:
            output.extend(process.stdout.read())
            process.stdout.close()

    store = RunStore(workspace)
    store.close()
    physics = PhysicsJobs(workspace / 'physics-jobs')
    physics.close()

def test_product_actual_observed_persisted_signed_and_replayed(tmp_path):
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario())
        record=wait_status(manager,run['id'],{'completed','failed','rejected'})
        manager._sessions[run['id']].thread.join()
        record=manager.read(run['id'])
        assert record['status']=='completed',record['error']
        result=record['result']
        assert len(result['candidates'])==4
        assert result['cursors']=={'submitted':40,'accepted':40,'observed':40}
        assert len(result['observed_history'])==len(result['steps'])==40
        events=manager.events(run['id'])
        observed=[e for e in events if e['type']=='OBSERVED']
        assert [e['payload']['r'] for e in observed]==[s['r'] for s in result['observed_history']]
        leases=[e for e in events if e['type']=='COMMIT']
        assert len(leases)==10 and all(e['payload']['prediction_hash'] for e in leases)
        path=store.directory(run['id'])
        assert json.loads((path/'replay.json').read_text())==result['steps']
        assert json.loads((path/'bundle/result.json').read_text())==result
        assert verify_bundle(str(path/'bundle'),str(path/'anchors/demo.public'),run['id'])[0]
        assert store.export(run['id']).is_file()
    finally:manager.close();store.close()
    store=RunStore(tmp_path)
    assert store.read(run['id'])['status']=='completed'
    assert store.read(run['id'])['result']==result
    store.close()

def test_product_stop_drain_approval_resume(tmp_path, monkeypatch):
    # This tests cancellation/recovery ordering, not host scheduling guarantees.
    # Keep actual pacing so the operator can interrupt, while using the same
    # deterministic lease clock as fast-mode tests on a busy shared CI runner.
    def logical_now(session):
        session.clock += 50_000_000
        return session.clock

    monkeypatch.setattr(ProductSession, 'now', logical_now)
    store=RunStore(tmp_path);manager=ProductManager(store)
    try:
        run=manager.start(Scenario())
        wait_status(manager,run['id'],{'running'})
        deadline=time.monotonic()+5
        while manager.read(run['id'])['result']['cursors']['submitted']<2:
            assert time.monotonic()<deadline
            time.sleep(.01)
        manager.stop(run['id'])
        stopped=wait_status(manager,run['id'],{'stopped','failed'})
        assert stopped['status']=='stopped',stopped['error']
        before=stopped['result']['cursors'].copy()
        time.sleep(.12)
        assert manager.read(run['id'])['result']['cursors']==before
        with pytest.raises(ValueError):manager.resume(run['id'],False)
        assert manager.resume(run['id'],True)['status']=='preparing'
        completed=wait_status(manager,run['id'],{'completed','failed'})
        assert completed['status']=='completed',completed['error']
        events=manager.events(run['id'])
        revoke=next(e for e in events if e['type']=='REVOKE')
        old=[e for e in events if e['seq']>revoke['seq'] and e['type']=='DISPATCH' and e['payload']['generation']<revoke['payload']['new_generation']]
        assert not old
        assert any(e['type']=='CANCEL_ACK' and e['payload']['confirmed'] for e in events)
    finally:manager.close();store.close()

def test_http_token_host_origin_and_input_boundaries(tmp_path):
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    server=LocalServer(('127.0.0.1',0),store,manager)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    def request(path,body=None,headers=None):
        raw=None if body is None else (body if isinstance(body,bytes) else json.dumps(body).encode())
        h={'Content-Type':'application/json'};h.update(headers or {})
        req=Request(base+path,data=raw,headers=h)
        try:
            with urlopen(req,timeout=5) as response:return response.status,json.load(response)
        except HTTPError as exc:return exc.code,json.load(exc)
    try:
        code,value=request('/api/session');assert code==200
        token=value['token']
        for headers in ({'Host':'attacker.example'},{'Origin':'https://attacker.example'},{'Sec-Fetch-Site':'cross-site'}):
            assert request('/api/session',headers=headers)[0]==403
        assert request('/api/runs',Scenario().summary())[0]==403
        headers={'X-Sentinel-Token':token}
        assert request('/api/runs',b'{"seed":1,"seed":2}',headers)[0]==400
        assert request('/api/runs',{'path':'/tmp/data'},headers)[0]==400
        assert request('/api/runs/../private')[0]==404
        code,created=request('/api/runs',Scenario().summary(),headers)
        assert code==201
        record=wait_status(manager,created['run']['id'],{'completed','failed'})
        manager._sessions[record['id']].thread.join()
        assert record['status']=='completed'
        assert request('/api/runs/'+record['id'])[1]['run']['result']['cursors']['observed']==40
        code,export=request('/api/runs/'+record['id']+'/export',{},headers)
        assert code==200 and export['download_url'].endswith('/download')
        import io,zipfile
        directory=store.directory(record['id'])
        (directory/'evidence.zip').write_bytes(b'corrupt cached zip')
        with urlopen(base+export['download_url'],timeout=5) as response:
            assert zipfile.is_zipfile(io.BytesIO(response.read()))
        (directory/'bundle/result.json').write_bytes(b'{}')
        code,error=request(export['download_url'])
        assert code==500 and error['error']['code']=='READ_FAILED'
    finally:
        server.shutdown();server.server_close();thread.join();manager.close();store.close()


def test_http_model_robot_and_viewer_entrypoints(tmp_path):
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    server=LocalServer(('127.0.0.1',0),store,manager)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    def request(path,body=None,token=None):
        raw=None if body is None else json.dumps(body).encode()
        headers={'Content-Type':'application/json'}
        if token is not None:headers['X-Sentinel-Token']=token
        req=Request(base+path,data=raw,headers=headers)
        try:
            with urlopen(req,timeout=5) as response:
                payload=response.read()
                if response.headers.get_content_type()=='application/json':
                    payload=json.loads(payload)
                return response.status,payload,response.headers.get_content_type()
        except HTTPError as exc:
            return exc.code,json.loads(exc.read()),exc.headers.get_content_type()
    try:
        code,session,_=request('/api/session');assert code==200
        token=session['token']
        code,viewer,content_type=request('/viewer.js')
        assert code==200 and content_type=='text/javascript' and b'SentinelViewer' in viewer

        obj=b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
        body={'name':'HTTP triangle','format':'obj','content_base64':base64.b64encode(obj).decode()}
        assert request('/api/assets',body)[0]==403
        assert request('/api/assets',{'name':'bad'},token)[0]==400
        code,created,_=request('/api/assets',body,token);assert code==201
        asset=created['asset'];asset_id=asset['id']
        assert request('/api/assets')[1]['assets'][0]['id']==asset_id
        assert request('/api/assets/'+asset_id)[1]['asset']['source_sha256']==asset['source_sha256']
        geometry=request('/api/assets/'+asset_id+'/geometry')[1]['geometry']
        assert geometry['vertices'][2]==[0.0,1.0,0.0] and geometry['triangles']==[[0,1,2]]
        check=request('/api/assets/'+asset_id+'/check',{},token)[1]['check']
        assert check['status']=='not_applicable' and check['physics_authorized'] is False
        assert request('/api/assets/asset-'+'0'*32+'/geometry')[0]==404

        robot_body={'name':'Protocol demo','driver':'mock'}
        assert request('/api/robots',robot_body)[0]==403
        code,created,_=request('/api/robots',robot_body,token);assert code==201
        profile=created['profile'];robot_id=profile['id']
        assert profile['hardware_motion'] is False and profile['credentials_stored'] is False
        diagnostic=request('/api/robots/'+robot_id+'/diagnose',{},token)[1]['diagnostic']
        assert diagnostic['status']=='demo_ready' and diagnostic['hardware_connected'] is False
        assert request('/api/robots')[1]['profiles'][0]['last_diagnostic']['status']=='demo_ready'
        assert request('/api/robots',{'name':'bad','driver':'mock','password':'secret'},token)[0]==400
        assert request('/api/robots/robot-'+'0'*32+'/diagnose',{},token)[0]==404
    finally:
        server.shutdown();server.server_close();thread.join();manager.close();store.close()


def test_local_server_does_not_create_physics_owner_before_bind(monkeypatch,tmp_path):
    import sentinel_evc.physics_jobs as jobs
    constructed=[]
    class ForbiddenPhysicsJobs:
        def __init__(self,root):constructed.append(root)
    def fail_bind(self,address,handler):
        raise OSError('injected bind failure')
    monkeypatch.setattr(jobs,'PhysicsJobs',ForbiddenPhysicsJobs)
    monkeypatch.setattr(ThreadingHTTPServer,'__init__',fail_bind)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        with pytest.raises(OSError,match='bind failure'):
            LocalServer(('127.0.0.1',0),store,manager)
        assert constructed==[]
    finally:manager.close();store.close()


def test_preparation_stop_never_dispatches_or_regresses_status(tmp_path,monkeypatch):
    import sentinel_evc.product_pipeline as pipeline
    entered=threading.Event();release=threading.Event()
    original=pipeline.build_default_numeric_artifacts
    def blocked(**kwargs):
        entered.set();assert release.wait(5)
        return original(**kwargs)
    monkeypatch.setattr(pipeline,'build_default_numeric_artifacts',blocked)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());assert entered.wait(5)
        assert manager.stop(run['id'])['status']=='stopping'
        assert manager.read(run['id'])['status']=='stopping'
        release.set();manager._sessions[run['id']].thread.join(10)
        record=manager.read(run['id'])
        assert record['status']=='failed' and record['error']['code']=='PROCESS_STOPPED'
        assert record['verification']['ok'] is True
        assert record['result']['cursors']=={'submitted':0,'accepted':0,'observed':0}
        assert not any(e['type']=='DISPATCH' for e in manager.events(run['id']))
        fresh=manager.start(Scenario());manager._sessions[fresh['id']].thread.join(10)
        assert manager.read(fresh['id'])['status']=='completed'
    finally:release.set();manager.close();store.close()


def test_shutdown_timeout_retains_storage_until_worker_exits(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event()
    original=ProductSession.finalize
    def blocked(self,status):
        entered.set();assert release.wait(5)
        return original(self,status)
    monkeypatch.setattr(ProductSession,'finalize',blocked)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());assert entered.wait(5)
        with pytest.raises(RuntimeError,match='ownership retained'):manager.close(timeout=.01)
        with pytest.raises(RuntimeError,match='already owned'):RunStore(tmp_path)
        with pytest.raises(RuntimeError):manager.start(Scenario())
        release.set();manager._sessions[run['id']].thread.join(10)
        manager.close()
        assert store.read(run['id'])['verification']['ok'] is True
    finally:release.set();manager.close();store.close()
    reopened=RunStore(tmp_path);reopened.close()


def test_stop_during_candidate_publication_is_preserved(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sentinel_evc.product_pipeline as pipeline
    entered=threading.Event();release=threading.Event()
    original=pipeline.evaluate_candidates
    class BlockedCandidates:
        def __init__(self,rows):self.rows=rows
        def __iter__(self):
            entered.set();assert release.wait(5)
            return iter(self.rows)
    def wrapped(*args,**kwargs):
        result=original(*args,**kwargs)
        return SimpleNamespace(candidates=BlockedCandidates(result.candidates),selected=result.selected)
    monkeypatch.setattr(pipeline,'evaluate_candidates',wrapped)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario(risk_limit=.001));assert entered.wait(5)
        assert manager.stop(run['id'])['status']=='stopping'
        release.set();manager._sessions[run['id']].thread.join(10)
        record=manager.read(run['id'])
        assert record['status']=='failed' and record['error']['code']=='PROCESS_STOPPED'
        assert record['result']['cursors']['submitted']==0
        assert record['verification']['ok'] is True
    finally:release.set();manager.close();store.close()


def test_stop_cannot_be_acknowledged_during_signed_finalization(tmp_path,monkeypatch):
    import sentinel_evc.product_pipeline as pipeline
    entered=threading.Event();release=threading.Event()
    original=pipeline.build_bundle
    def blocked(*args,**kwargs):
        entered.set();assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(pipeline,'build_bundle',blocked)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());assert entered.wait(5)
        with pytest.raises(ValueError,match='cannot stop'):manager.stop(run['id'])
        release.set();manager._sessions[run['id']].thread.join(10)
        assert manager.read(run['id'])['status']=='completed'
    finally:release.set();manager.close();store.close()


def test_serve_closes_manager_and_store_when_server_construction_fails(monkeypatch,tmp_path):
    import sentinel_evc.server as server_module
    import sentinel_evc.product_pipeline as pipeline
    closed=[]
    class FakeStore:
        def __init__(self,root):self.root=tmp_path
        def close(self):closed.append('store')
    class FakeManager:
        def __init__(self,store):pass
        def close(self):closed.append('manager')
    def fail_server(*args,**kwargs):raise OSError('injected construction failure')
    monkeypatch.setattr(server_module,'RunStore',FakeStore)
    monkeypatch.setattr(pipeline,'ProductManager',FakeManager)
    monkeypatch.setattr(server_module,'LocalServer',fail_server)
    with pytest.raises(OSError,match='construction failure'):
        server_module.serve(tmp_path,0)
    assert closed==['manager','store']


def test_http_reports_missing_physics_engine_and_rejects_malformed_dynamic_types(tmp_path,monkeypatch):
    import sentinel_evc.physics_jobs as jobs
    monkeypatch.setattr(jobs,'physics_engine_status',lambda:{
        'available':False,'version':None,'reason':'MuJoCo is not installed; install sentinel-evc-lab[physics].',
    })
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    server=LocalServer(('127.0.0.1',0),store,manager)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    def request(path,body=None,token=None):
        raw=None if body is None else json.dumps(body).encode()
        headers={'Content-Type':'application/json'}
        if token is not None:headers['X-Sentinel-Token']=token
        try:
            with urlopen(Request(base+path,data=raw,headers=headers),timeout=5) as response:
                return response.status,json.load(response)
        except HTTPError as exc:
            return exc.code,json.load(exc)
    try:
        code,session=request('/api/session');assert code==200
        assert session['physics_engine']=={
            'available':False,'version':None,'reason':'MuJoCo is not installed; install sentinel-evc-lab[physics].',
        }
        assert session['capabilities']['physics_jobs'] is False
        token=session['token']
        for friction in ('0.35',[0.35]):
            code,value=request('/api/physics-jobs',{'friction':friction},token)
            assert code==400 and value['error']['code']=='INPUT_SCHEMA'
        code,value=request('/api/physics-jobs',{},token)
        assert code==400 and 'not installed' in value['error']['message']
        code,value=request('/api/robots',{'name':'bad','driver':['mock']},token)
        assert code==400 and value['error']['code']=='INVALID_INPUT'
    finally:
        server.shutdown();server.server_close();thread.join();manager.close();store.close()

def test_all_rejected_keeps_denominator(tmp_path):
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario.parse({'risk_limit':.001}))
        record=wait_status(manager,run['id'],{'rejected','failed'})
        manager._sessions[run['id']].thread.join()
        assert record['status']=='rejected'
        assert len(record['result']['candidates'])==4
        assert record['result']['selected'] is None
        assert record['result']['cursors']['submitted']==0
    finally:manager.close();store.close()


def test_finalization_blocks_concurrent_start_and_export(tmp_path):
    import sentinel_evc.product_pipeline as pipeline
    original=pipeline.build_bundle
    entered=threading.Event();release=threading.Event()
    def delayed(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    pipeline.build_bundle=delayed
    try:
        run=manager.start(Scenario())
        assert entered.wait(5)
        assert manager.read(run['id'])['status'] not in {'completed','failed','rejected'}
        with pytest.raises(RuntimeError):manager.start(Scenario())
        with pytest.raises(ValueError):store.export(run['id'])
        release.set();manager._sessions[run['id']].thread.join(5)
        assert manager.read(run['id'])['status']=='completed'
        assert store.export(run['id']).is_file()
    finally:
        release.set();pipeline.build_bundle=original;manager.close();store.close()


def test_finalization_failure_does_not_wedge_service(tmp_path):
    import sentinel_evc.product_pipeline as pipeline
    original=pipeline.build_bundle
    def broken(*args,**kwargs):raise OSError('injected storage failure')
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        pipeline.build_bundle=broken
        run=manager.start(Scenario());manager._sessions[run['id']].thread.join(5)
        record=manager.read(run['id'])
        assert record['status']=='failed'
        assert record['error']['code']=='EVIDENCE_FINALIZATION_FAILED'
        assert record['verification']['ok'] is False
        pipeline.build_bundle=original
        second=manager.start(Scenario());manager._sessions[second['id']].thread.join(5)
        assert manager.read(second['id'])['status']=='completed'
    finally:pipeline.build_bundle=original;manager.close();store.close()

def test_new_stop_during_resume_is_not_lost(tmp_path):
    from sentinel_evc.executor import Executor
    original=Executor.try_recover
    entered=threading.Event();release=threading.Event()
    def delayed(self,*args,**kwargs):
        entered.set();assert release.wait(5)
        return original(self,*args,**kwargs)
    store=RunStore(tmp_path);manager=ProductManager(store)
    try:
        run=manager.start(Scenario());wait_status(manager,run['id'],{'running'})
        manager.stop(run['id']);wait_status(manager,run['id'],{'stopped'})
        before=len([e for e in manager.events(run['id']) if e['type']=='DISPATCH'])
        Executor.try_recover=delayed
        manager.resume(run['id'],True)
        assert entered.wait(5)
        manager.stop(run['id']);release.set()
        stopped=wait_status(manager,run['id'],{'stopped','failed'})
        assert stopped['status']=='stopped'
        assert len([e for e in manager.events(run['id']) if e['type']=='DISPATCH'])==before
    finally:
        release.set();Executor.try_recover=original;manager.close();store.close()


def test_server_close_waits_for_inflight_handler_before_workspace_release(tmp_path):
    entered, release = threading.Event(), threading.Event()
    store = RunStore(tmp_path); manager = ProductManager(store, realtime=False)
    server = LocalServer(('127.0.0.1', 0), store, manager)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    response = {}

    def blocked_import(name, format, content_base64):
        entered.set()
        assert release.wait(5)
        return {'id': 'asset-' + 'a' * 32, 'name': name, 'format': format,
                'physics_authorized': False}

    server.assets.import_asset = blocked_import
    base = 'http://127.0.0.1:' + str(server.server_address[1])

    def request():
        body = json.dumps({'name': 'barrier', 'format': 'obj', 'content_base64': 'eA=='}).encode()
        headers = {'Content-Type': 'application/json', 'X-Sentinel-Token': server.token}
        try:
            with urlopen(Request(base + '/api/assets', data=body, headers=headers), timeout=5) as result:
                response['status'] = result.status
                response['body'] = json.load(result)
        except BaseException as exc:
            response['error'] = exc

    request_thread = threading.Thread(target=request)
    request_thread.start()
    closer = None
    try:
        assert entered.wait(5)
        server.shutdown()
        closer = threading.Thread(target=server.server_close)
        closer.start()
        time.sleep(.05)
        assert closer.is_alive(), 'server_close released while a handler still used server resources'
        release.set()
        request_thread.join(5); closer.join(5); server_thread.join(5)
        assert not request_thread.is_alive() and not closer.is_alive() and not server_thread.is_alive()
        assert response.get('status') == 201, response
        assert response['body']['asset']['physics_authorized'] is False
        manager.close(); store.close()
        reopened = RunStore(tmp_path)
        reopened.close()
    finally:
        release.set()
        if closer is None:
            server.shutdown(); server.server_close()
        request_thread.join(5); server_thread.join(5)
        try: manager.close()
        except Exception: pass
        try: store.close()
        except Exception: pass
