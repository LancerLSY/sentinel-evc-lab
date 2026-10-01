import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from sentinel_evc.scenario import Scenario
from sentinel_evc.runstore import RunStore
from sentinel_evc.product_pipeline import ProductManager
from sentinel_evc.server import LocalServer
from sentinel_evc.evidence import verify_bundle

def wait_status(manager,run_id,states,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        record=manager.read(run_id)
        if record['status'] in states:return record
        time.sleep(.01)
    raise AssertionError('run did not reach expected state')

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

def test_product_stop_drain_approval_resume(tmp_path):
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
