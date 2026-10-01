"""Atomic local assets with a rebuildable SQLite index and bounded public identifiers."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from .evidence import verify_bundle, _safe_name

RUN_ID = re.compile(r'^run-[0-9a-f]{32}$')
ACTIVE = {'queued','preparing','running','stopping','stopped'}
TERMINAL = {'completed','rejected','failed'}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

class _WorkspaceLock:
    """One local service owns a run directory, including across processes."""
    def __init__(self,path):
        if path.is_symlink():raise ValueError('symlink workspace lock refused')
        self.stream=open(path,'a+b')
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                if path.stat().st_size==0:self.stream.write(b'0');self.stream.flush()
                self.stream.seek(0);msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError('run directory is already owned by another local service') from exc
    def close(self):
        self.stream.close()

class RunStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self._lock = threading.RLock()
        self._process_lock = _WorkspaceLock(self.root / 'workspace.lock')
        self._db = sqlite3.connect(self.root / 'index.sqlite3',check_same_thread=False)
        self._db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, updated TEXT, summary TEXT)')
        self.rebuild()

    def directory(self, run_id):
        if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id):
            raise KeyError('run not found')
        path = self.root / run_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise KeyError('run not found')
        return path

    def _write(self, path, data):
        if path.is_symlink():
            raise ValueError('symlink storage refused')
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        with open(temp,'xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)

    def save_asset(self,run_id,name,value):
        if name not in {'scenario.json','result.json','model.json','calibration.json','predictions.json','replay.json','events.jsonl'}:
            raise ValueError('unsupported asset')
        data = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
        with self._lock:
            self._write(self.directory(run_id) / name,data)

    def create(self, scenario):
        with self._lock:
            run_id = 'run-' + uuid.uuid4().hex
            self.directory(run_id).mkdir()
            now = utc_now()
            record = {'id':run_id,'name':scenario.name,'status':'queued','scenario':scenario.summary(),'created_at':now,'updated_at':now,'scope':'numeric-simulator-product-v1','error':None,'selected':None,'result':None}
            self.save(record)
            self.save_asset(run_id,'scenario.json',scenario.summary())
            return record

    def save(self, record):
        with self._lock:
            record['updated_at'] = utc_now()
            self._write(self.directory(record['id']) / 'run.json',json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8'))
            summary = {k:record.get(k) for k in ('id','name','status','created_at','updated_at','error','selected','scope')}
            self._db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',(record['id'],record['updated_at'],json.dumps(summary,ensure_ascii=False)))
            self._db.commit()

    def read(self, run_id):
        with self._lock:
            try:
                record=json.loads((self.directory(run_id) / 'run.json').read_text('utf-8'))
                return self._verified_record(record)
            except (OSError,ValueError) as exc:
                raise KeyError('run not found') from exc

    def _verified_record(self,record):
        if record.get('status') not in TERMINAL or not record.get('verification'):
            return record
        directory=self.directory(record['id'])
        try:
            ok,message=verify_bundle(str(directory/'bundle'),str(directory/'anchors/demo.public'),record['id'])
            if not ok:raise ValueError('invalid bundle')
            result=json.loads((directory/'bundle/result.json').read_text('utf-8'))
            scenario=json.loads((directory/'bundle/scenario.json').read_text('utf-8'))
            visible_events=(directory/'events.jsonl').read_bytes()
            signed_events=(directory/'bundle/events.jsonl').read_bytes()
            identical=(record['result']==result and record['scenario']==scenario and visible_events==signed_events and record['status']==result['status'] and record['selected']==result['selected'] and record['name']==scenario['name'] and record['error']==result['error'] and record['scope']==result['scope'])
            record['result']=result
            record['scenario']=scenario
            record['name']=scenario['name']
            record['selected']=result['selected']
            record['error']=result['error']
            record['scope']=result['scope']
            record['verification']={'ok':identical,'message':message if identical else 'VISIBLE_RECORD_MISMATCH','trust':'self-contained demo key; external trust required'}
            if not identical:
                record['status']='failed'
                record['error']={'code':'EVIDENCE_MISMATCH','message':'运行记录与签名证据不一致；显示已签名数据。'}
        except (OSError,ValueError,KeyError):
            record['result']=None
            record['verification']={'ok':False,'message':'EVIDENCE_INVALID','trust':'self-contained demo key; external trust required'}
            record['status']='failed'
            record['error']={'code':'EVIDENCE_INVALID','message':'证据包未通过重新核验。'}
        return record

    def list(self):
        with self._lock:
            return [json.loads(row[0]) for row in self._db.execute('SELECT summary FROM runs ORDER BY updated DESC LIMIT 200')]

    def events(self, run_id):
        path = self.directory(run_id) / 'events.jsonl'
        with self._lock:
            record=self.read(run_id)
            if record.get('verification'):
                if not record['verification']['ok']:return []
                path=self.directory(run_id)/'bundle/events.jsonl'
            return [json.loads(line) for line in path.read_text('utf-8').splitlines()] if path.is_file() and not path.is_symlink() else []

    def rebuild(self):
        with self._lock:
            self._db.execute('DELETE FROM runs')
            self._db.commit()
            for path in self.root.iterdir():
                if RUN_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                    try:
                        record = self.read(path.name)
                        if record['status'] in ACTIVE:
                            record['status'] = 'failed'
                            record['error'] = {'code':'PROCESS_INTERRUPTED','message':'进程重启，运行已中断；请创建新运行。'}
                        self.save(record)
                    except (KeyError,ValueError):
                        continue

    def export(self,run_id):
        with self._lock:
            record = self.read(run_id)
            if record['status'] not in TERMINAL:
                raise ValueError('only terminal runs can be exported')
            directory = self.directory(run_id)
            files = ['bundle/manifest.json','bundle/manifest.sig','anchors/demo.public']
            if not record.get('verification',{}).get('ok'):
                raise ValueError('bundle must be finalized and verified')
            manifest = json.loads((directory/'bundle/manifest.json').read_text('utf-8'))
            if not isinstance(manifest.get('files'),dict) or any(not _safe_name(name) for name in manifest['files']):
                raise ValueError('unsafe manifest member')
            files += ['bundle/'+name for name in manifest['files']]
            archive = directory / 'evidence.zip'
            temp = directory / 'evidence.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for name in files:
                    path = directory/name
                    if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(directory.resolve()) is False:
                        raise ValueError('unsafe export path')
                    z.write(path,name)
            os.replace(temp,archive)
            return archive

    def close(self):
        with self._lock:
            self._db.close()
            self._process_lock.close()
