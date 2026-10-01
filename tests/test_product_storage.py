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
