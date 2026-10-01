import json
from pathlib import Path
import pytest
from sentinel_evc.scenario import Scenario, InputError, strict_json
from sentinel_evc.runstore import RunStore
from sentinel_evc.evidence import build_bundle, verify_bundle
from sentinel_evc.events import EventLog

def test_scenario_strict():
    for raw in (b'{"seed":1,"seed":2}',b'{"risk_limit":NaN}',b'\xff'):
        with pytest.raises(InputError): strict_json(raw)
    for obj in ({'path':'/tmp/x'},{'seed':True},{'displacement':float('inf')},{'scene':{'obstacles':[]}}):
        with pytest.raises(InputError): Scenario.parse(obj)
    assert Scenario.parse(Scenario().summary()) == Scenario()

def test_restart_marks_interrupted_and_preserves_replay(tmp_path):
    store = RunStore(tmp_path)
    record = store.create(Scenario())
    record['status'] = 'running'
    store.save(record)
    store.save_asset(record['id'],'replay.json',[{'step':1,'r':.02}])
    store.close()
    store = RunStore(tmp_path)
    assert store.read(record['id'])['status'] == 'failed'
    assert json.loads((store.directory(record['id'])/'replay.json').read_text()) == [{'step':1,'r':.02}]
    with pytest.raises(KeyError): store.read('../secrets')
    store.close()

def test_signed_product_assets_detect_tamper(tmp_path):
    log = EventLog('product-test')
    log.append('OUTCOME',outcome='complete')
    bundle = build_bundle(log,str(tmp_path),{'result.json':b'{"success":true}'})
    assert verify_bundle(bundle['bundle_dir'],bundle['public_key'],'product-test')[0]
    Path(bundle['bundle_dir'],'result.json').write_bytes(b'{"success":false}')
    assert not verify_bundle(bundle['bundle_dir'],bundle['public_key'],'product-test')[0]

def test_verifier_refuses_signed_unsafe_file_map(tmp_path):
    log=EventLog('unsafe');log.append('OUTCOME',outcome='ok')
    with pytest.raises(ValueError):build_bundle(log,str(tmp_path),{'../secret':b'x'})


def test_export_rejects_hostile_manifest_and_rechecks_visible_data(tmp_path):
    from sentinel_evc.product_pipeline import ProductManager
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    run=manager.start(Scenario());manager._sessions[run['id']].thread.join()
    try:
        path=store.directory(run['id'])
        record=json.loads((path/'run.json').read_text())
        record['result']['cursors']['observed']=999
        (path/'run.json').write_text(json.dumps(record))
        visible=manager.read(run['id'])
        assert visible['verification']['ok'] is False
        assert visible['result']['cursors']['observed']==40
        with pytest.raises(ValueError):store.export(run['id'])
        manifest=json.loads((path/'bundle/manifest.json').read_text())
        manifest['files']['../run.json']='sha256:'+'0'*64
        (path/'bundle/manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError):store.export(run['id'])
        assert not (path/'evidence.zip').exists()
    finally:manager.close();store.close()

def test_malformed_evidence_is_failure_not_exception(tmp_path):
    log=EventLog('malformed');log.append('OUTCOME',outcome='ok')
    bundle=build_bundle(log,str(tmp_path))
    Path(bundle['bundle_dir'],'events.jsonl').write_bytes(b'\xff')
    assert not verify_bundle(bundle['bundle_dir'],bundle['public_key'],'malformed')[0]
    Path(bundle['bundle_dir'],'manifest.json').write_text('[]')
    assert not verify_bundle(bundle['bundle_dir'],bundle['public_key'],'malformed')[0]


def test_only_one_service_owns_storage(tmp_path):
    first=RunStore(tmp_path)
    try:
        with pytest.raises(RuntimeError):RunStore(tmp_path)
    finally:first.close()
    second=RunStore(tmp_path);second.close()


def test_terminal_reads_and_lists_reverify_without_cached_verification(tmp_path):
    from sentinel_evc.product_pipeline import ProductManager
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());manager._sessions[run['id']].thread.join()
        path=store.directory(run['id'])/'run.json'
        record=json.loads(path.read_text());record.pop('verification')
        path.write_text(json.dumps(record))
        assert store.read(run['id'])['verification']['ok'] is True
        record['scope']='forged hardware success'
        path.write_text(json.dumps(record))
        visible=store.read(run['id'])
        assert visible['status']=='failed' and visible['verification']['ok'] is False
        assert visible['scope']=='numeric-simulator-product-v1'
        assert store.list()[0]['status']=='failed'
        with pytest.raises(ValueError):store.export(run['id'])
        record['status']='queued'
        path.write_text(json.dumps(record))
        assert store.read(run['id'])['status']=='failed'
    finally:manager.close();store.close()


@pytest.mark.parametrize('payload', ['[]','null','42','{"id":"wrong","status":"completed"}'])
def test_malformed_run_metadata_is_isolated_on_restart(tmp_path,payload):
    store=RunStore(tmp_path);run=store.create(Scenario());path=store.directory(run['id'])/'run.json'
    store.close();path.write_text(payload)
    store=RunStore(tmp_path)
    try:
        assert store.list()==[]
        with pytest.raises(KeyError):store.read(run['id'])
        assert store.create(Scenario())['status']=='queued'
    finally:store.close()


def test_failed_store_constructor_releases_workspace_lock(tmp_path):
    import sqlite3
    (tmp_path/'index.sqlite3').write_bytes(b'corrupt sqlite')
    with pytest.raises(sqlite3.DatabaseError):RunStore(tmp_path)
    (tmp_path/'index.sqlite3').unlink()
    store=RunStore(tmp_path);store.close()


def test_export_replaces_cache_and_ignores_legacy_temp_symlink(tmp_path):
    import zipfile
    from sentinel_evc.product_pipeline import ProductManager
    store=RunStore(tmp_path);manager=ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());manager._sessions[run['id']].thread.join()
        directory=store.directory(run['id'])
        victim=tmp_path/'outside.txt';victim.write_text('preserved')
        (directory/'evidence.tmp').symlink_to(victim)
        archive=store.export(run['id']);archive.write_bytes(b'corrupt cache')
        assert zipfile.is_zipfile(store.export(run['id']))
        assert victim.read_text()=='preserved'
    finally:manager.close();store.close()


@pytest.mark.parametrize('partial_manifest',[False,True])
def test_unsigned_finalization_failure_survives_restart(tmp_path,monkeypatch,partial_manifest):
    import sentinel_evc.product_pipeline as pipeline
    original=pipeline.build_bundle
    def fail(*args,**kwargs):
        if partial_manifest:
            bundle=Path(args[1])/'bundle';bundle.mkdir()
            (bundle/'manifest.json').write_text('{}')
        raise OSError('injected disk failure')
    monkeypatch.setattr(pipeline,'build_bundle',fail)
    store=RunStore(tmp_path);manager=pipeline.ProductManager(store,realtime=False)
    try:
        run=manager.start(Scenario());manager._sessions[run['id']].thread.join(10)
        assert manager.read(run['id'])['error']['code']=='EVIDENCE_FINALIZATION_FAILED'
    finally:manager.close();store.close()
    store=RunStore(tmp_path);manager=pipeline.ProductManager(store,realtime=False)
    try:
        record=store.read(run['id'])
        assert record['status']=='failed'
        assert record['error']['code']=='EVIDENCE_FINALIZATION_FAILED'
        assert record['verification']['ok'] is False
        with pytest.raises(ValueError):store.export(run['id'])
        monkeypatch.setattr(pipeline,'build_bundle',original)
        fresh=manager.start(Scenario());manager._sessions[fresh['id']].thread.join(10)
        assert manager.read(fresh['id'])['status']=='completed'
    finally:manager.close();store.close()
