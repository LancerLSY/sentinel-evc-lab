"""Loopback-only single-user HTTP transport. Controller lifecycle stays in the worker."""
from __future__ import annotations
import json
import os
import re
import secrets
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit
from .scenario import Scenario, InputError, strict_json, TEMPLATES
from .runstore import RunStore

MAX_BODY = 12 * 1024 * 1024
MAX_NATIVE_BODY = 88 * 1024 * 1024
MAX_LAUNCH_BODY = 36 * 1024 * 1024

class LocalServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    request_timeout = 5.0
    def __init__(self, address, store, manager):
        if address[0] != '127.0.0.1':
            raise ValueError('only 127.0.0.1 is supported')
        from .assets import AssetStore
        from .robot_connectors import RobotRegistry
        from .physics_jobs import PhysicsJobs
        from .native_runs import NativeRunStore
        self.store, self.manager, self.token = store, manager, secrets.token_urlsafe(32)
        self.physics_jobs = None
        super().__init__(address, Handler)
        try:
            self.assets = AssetStore(store.root / "models")
            self.robots = RobotRegistry(store.root / "robots")
            self.physics_jobs = PhysicsJobs(store.root / "physics-jobs")
            self.native_runs = NativeRunStore(store.root / "native-runs")
            port = self.server_address[1]
            self.hosts = {'127.0.0.1:'+str(port),'localhost:'+str(port)}
            self.origins = {'http://'+host for host in self.hosts}
        except Exception:
            if self.physics_jobs is not None:
                self.physics_jobs.close()
                self.physics_jobs = None
            super().server_close()
            raise

    def get_request(self):
        request, client_address = super().get_request()
        try:
            request.settimeout(self.request_timeout)
        except BaseException:
            request.close()
            raise
        return request, client_address

    def server_close(self):
        try:
            if self.physics_jobs is not None:
                self.physics_jobs.close()
                self.physics_jobs = None
        finally:
            # ThreadingHTTPServer blocks here until every non-daemon request
            # handler has returned, so callers may then close manager/store.
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'SentinelLocal/0.2'
    def log_message(self,*args):
        pass

    def reply(self,status,value,content_type='application/json; charset=utf-8',download_name=None):
        data = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        if download_name:
            self.send_header('Content-Disposition',f'attachment; filename="{download_name}"')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def error(self,status,code,message):
        self.reply(status,{'error':{'code':code,'message':message}})

    def archive_reply(self, path):
        try:
            self.send_response(200)
            self.send_header('Content-Type','application/zip')
            self.send_header('Content-Length',str(path.stat().st_size))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Disposition','attachment; filename="sentinel-native-vla.zip"')
            self.end_headers()
            with path.open('rb') as handle:
                while chunk := handle.read(65536):
                    self.wfile.write(chunk)
        finally:
            path.unlink(missing_ok=True)

    def allowed(self,mutation=False):
        if self.headers.get('Host') not in self.server.hosts:
            self.error(403,'HOST_REJECTED','不支持此主机。');return False
        origin = self.headers.get('Origin')
        if origin and origin not in self.server.origins:
            self.error(403,'ORIGIN_REJECTED','跨来源请求已拒绝。');return False
        if self.headers.get('Sec-Fetch-Site') not in (None,'same-origin','none'):
            self.error(403,'ORIGIN_REJECTED','跨来源请求已拒绝。');return False
        if mutation and not secrets.compare_digest(self.headers.get('X-Sentinel-Token',''),self.server.token):
            self.error(403,'TOKEN_REQUIRED','请重新打开本地操作台。');return False
        return True

    def parts(self):
        parsed=urlsplit(self.path)
        if parsed.query or parsed.fragment or '%' in parsed.path or '//' in parsed.path:
            raise KeyError('unsupported path')
        return parsed.path.strip('/').split('/')

    def do_GET(self):
        if not self.allowed():return
        try:
            parts=self.parts()
            if parts == ['api','session']:
                physics_engine=self.server.physics_jobs.engine_status()
                return self.reply(200,{'token':self.server.token,'profile':'numeric-simulator-product-v1','templates':TEMPLATES,'physics_engine':physics_engine,'capabilities':{'stop':True,'resume':True,'export':True,'real_robot':False,'model_import':True,'robot_diagnostics':True,'physics_jobs':physics_engine['available'],'native_vla_recordings':True,'native_vla_compare':True}})
            if parts == ['api','native-runs']:
                return self.reply(200, {'runs': self.server.native_runs.list()})
            if len(parts) in (3,4,5,6) and parts[:2] == ['api','native-runs']:
                run_id = parts[2]
                if len(parts)==3:
                    return self.reply(200, {'run': self.server.native_runs.get(run_id)})
                if len(parts)==4 and parts[3]=='replay':
                    return self.reply(200, self.server.native_runs.replay(run_id))
                if len(parts)==4 and parts[3]=='model':
                    return self.reply(200, self.server.native_runs.asset(run_id,'viewer-model.json'),'application/json; charset=utf-8')
                if len(parts)==4 and parts[3]=='download':
                    return self.archive_reply(self.server.native_runs.export(run_id))
                if len(parts)==5 and parts[3]=='compare':
                    return self.reply(200,self.server.native_runs.compare(run_id,parts[4]))
                if len(parts)==6 and parts[3]=='compare' and parts[5]=='report.json':
                    report = self.server.native_runs.compare(run_id,parts[4])
                    data = (json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
                    return self.reply(200,data,download_name='sentinel-execution-diff.json')
                if len(parts)==5 and parts[3]=='files':
                    suffix=parts[4].rsplit('.',1)[-1].lower()
                    types={'json':'application/json; charset=utf-8','mp4':'video/mp4','png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg','webp':'image/webp'}
                    if suffix not in types:
                        raise KeyError('unsupported media')
                    return self.reply(200,self.server.native_runs.asset(run_id,parts[4]),types[suffix])
            if parts == ['api','assets']:
                return self.reply(200, {'assets': self.server.assets.list()})
            if len(parts) in (3,4) and parts[:2] == ['api','assets']:
                if len(parts)==3:
                    return self.reply(200, {'asset': self.server.assets.get(parts[2])})
                if parts[3]=='geometry':
                    return self.reply(200, {'geometry': self.server.assets.geometry(parts[2])})
            if parts == ['api','robots']:
                return self.reply(200, {'profiles': self.server.robots.list()})
            if parts == ['api','physics-jobs']:
                return self.reply(200, {'jobs': self.server.physics_jobs.list()})
            if len(parts) in (3,4) and parts[:2] == ['api','physics-jobs']:
                if len(parts)==3:
                    return self.reply(200, {'job': self.server.physics_jobs.get(parts[2])})
                if parts[3]=='trace':
                    return self.reply(200, self.server.physics_jobs.trace(parts[2]))
                if parts[3]=='download':
                    path=self.server.physics_jobs.export(parts[2])
                    return self.reply(200,path.read_bytes(),'application/zip')
            if parts == ['api','runs']:
                return self.reply(200,{'runs':self.server.store.list()})
            if parts == ['api','experiments']:
                from .experiments import experiment_registry
                experiments=[]
                for record in experiment_registry():
                    row=record.summary()
                    row.setdefault('title',row.get('name',row.get('id')))
                    row.setdefault('artifacts',row.get('expected_artifacts',[]))
                    experiments.append(row)
                return self.reply(200,{'experiments':experiments})
            if len(parts) in (3,4) and parts[:2] == ['api','runs']:
                run_id=parts[2]
                if len(parts)==3:return self.reply(200,{'run':self.server.manager.read(run_id)})
                if parts[3]=='events':return self.reply(200,{'events':self.server.manager.events(run_id)})
                if parts[3]=='download':
                    path=self.server.store.export(run_id)
                    return self.reply(200,path.read_bytes(),'application/zip')
            assets={'':'index.html','index.html':'index.html','app.js':'app.js','styles.css':'styles.css','viewer.js':'viewer.js','native-ui.js':'native-ui.js','native-viewer.js':'native-viewer.js','native-compare-ui.js':'native-compare-ui.js','launch-ui.js':'launch-ui.js'}
            name='/'.join(parts)
            if name in assets:
                resource=files('sentinel_evc').joinpath('web',assets[name])
                ctype='text/html; charset=utf-8' if assets[name].endswith('.html') else 'text/css; charset=utf-8' if assets[name].endswith('.css') else 'text/javascript; charset=utf-8'
                return self.reply(200,resource.read_bytes(),ctype)
            raise KeyError('route not found')
        except KeyError:
            self.error(404,'NOT_FOUND','运行或页面不存在。')
        except (OSError,ValueError,RuntimeError):
            self.error(500,'READ_FAILED','读取失败，请查看本地运行状态。')

    def do_POST(self):
        if not self.allowed(True):return
        try:
            parts=self.parts()
            length=self.headers.get('Content-Length','')
            body_limit = (MAX_NATIVE_BODY if parts == ['api','native-runs','import'] else
                          MAX_LAUNCH_BODY if parts in (['api','launch-check'],['api','launch-capsule']) else MAX_BODY)
            if self.headers.get('Transfer-Encoding') or not length.isdigit() or not 0 < int(length) <= body_limit:
                return self.error(413,'BODY_LIMIT','请求大小超出范围。')
            if self.headers.get('Content-Type','').split(';')[0].strip() != 'application/json':
                return self.error(415,'CONTENT_TYPE','需要 application/json。')
            self.connection.settimeout(5)
            body=strict_json(self.rfile.read(int(length)))
            if parts in (['api','launch-check'], ['api','launch-capsule']):
                from .launch_gate import MAX_PACK_BYTES, build_qualification_capsule, qualify
                if not isinstance(body,dict) or set(body) != {'reference_json','candidate_json'}:
                    raise InputError('请选择参考和待检查两份探针 JSON。')
                packs=[]
                for name in ('reference_json','candidate_json'):
                    value=body[name]
                    if not isinstance(value,str) or not 0 < len(value.encode('utf-8')) <= MAX_PACK_BYTES:
                        raise InputError('探针文件需要为不超过 8 MiB 的 JSON。')
                    packs.append(value.encode('utf-8'))
                if parts[-1] == 'launch-check':
                    return self.reply(200,qualify(*packs))
                import base64, io, tempfile, uuid, zipfile
                run_id='launch-'+uuid.uuid4().hex
                with tempfile.TemporaryDirectory(prefix='.launch-',dir=self.server.store.root) as temporary:
                    report,evidence=build_qualification_capsule(*packs,Path(temporary)/'decision',run_id)
                    archive=io.BytesIO()
                    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
                        for source in sorted(Path(evidence['bundle_dir']).iterdir()):
                            output.writestr('bundle/'+source.name,source.read_bytes())
                    public_key=Path(evidence['public_key']).read_bytes()
                return self.reply(200,{'report':report,'run_id':run_id,
                    'capsule_base64':base64.b64encode(archive.getvalue()).decode('ascii'),
                    'public_key_base64':base64.b64encode(public_key).decode('ascii'),
                    'trust':'new derived decision with demo key; select its public key separately'})
            if parts == ['api','native-runs','import']:
                return self.reply(201, {'run': self.server.native_runs.import_request(body)})
            if parts == ['api','assets']:
                if not isinstance(body,dict) or set(body) != {'name','format','content_base64'}:
                    raise InputError('需要 name、format、content_base64 模型参数。')
                return self.reply(201, {'asset': self.server.assets.import_asset(body['name'],body['format'],content_base64=body['content_base64'])})
            if len(parts)==4 and parts[:2]==['api','assets'] and parts[3]=='check' and body=={}:
                return self.reply(200, {'check': self.server.assets.check(parts[2])})
            if parts == ['api','robots']:
                if not isinstance(body,dict) or set(body)-{'name','driver','host','port'} or not {'name','driver'}.issubset(body):
                    raise InputError('需要 name、driver 及可选的 host、port。')
                return self.reply(201, {'profile': self.server.robots.create(**body)})
            if len(parts)==4 and parts[:2]==['api','robots'] and parts[3]=='diagnose' and body=={}:
                return self.reply(200, {'diagnostic': self.server.robots.diagnose(parts[2])})
            if parts == ['api','physics-jobs']:
                return self.reply(201, {'job': self.server.physics_jobs.start(body)})
            if len(parts)==4 and parts[:2]==['api','physics-jobs'] and parts[3]=='export' and body=={}:
                self.server.physics_jobs.export(parts[2])
                return self.reply(200, {'download_url':'/api/physics-jobs/'+parts[2]+'/download'})
            if parts == ['api','runs']:
                record=self.server.manager.start(Scenario.parse(body))
                return self.reply(201,{'run':record})
            if len(parts)==4 and parts[:2]==['api','runs']:
                run_id,action=parts[2:]
                if action=='stop' and body=={}:
                    return self.reply(200,{'run':self.server.manager.stop(run_id)})
                if action=='resume' and body=={'operator_approved':True}:
                    return self.reply(200,{'run':self.server.manager.resume(run_id,True)})
                if action=='export' and body=={}:
                    self.server.store.export(run_id)
                    return self.reply(200,{'download_url':'/api/runs/'+run_id+'/download'})
            raise InputError('unsupported action or body')
        except InputError as exc:
            self.error(400,'INPUT_SCHEMA',str(exc))
        except KeyError:
            self.error(404,'NOT_FOUND','运行不存在。')
        except RuntimeError as exc:
            self.error(409,'RUN_CONFLICT',str(exc))
        except ValueError as exc:
            self.error(400,'INVALID_INPUT',str(exc))
        except (OSError,TimeoutError):
            self.error(500,'WRITE_FAILED','保存失败，未批准新的动作。')

    def do_OPTIONS(self):
        self.error(405,'METHOD_NOT_ALLOWED','不支持跨来源访问。')


def serve(root='runs/workbench',port=8765, *, open_browser=False):
    from .product_pipeline import ProductManager
    import signal, threading
    store = manager = server = None
    previous_term = None
    try:
        store=RunStore(root)
        manager=ProductManager(store)
        server=LocalServer(('127.0.0.1',port),store,manager)
        print('Sentinel 本地操作台：http://127.0.0.1:'+str(server.server_address[1]),flush=True)
        launch_nonce=os.environ.get('SENTINEL_LAUNCH_NONCE','')
        if re.fullmatch(r'[0-9a-f]{32}',launch_nonce):
            print('SENTINEL_READY '+json.dumps({'nonce':launch_nonce,'url':'http://127.0.0.1:'+str(server.server_address[1])}),flush=True)
        print('数值工作台 / 三维物理 / 模型资产 / 机械臂诊断；Ctrl+C 停止服务。',flush=True)
        if threading.current_thread() is threading.main_thread():
            previous_term = signal.getsignal(signal.SIGTERM)
            def stop_service(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop_service)
        if open_browser:
            import webbrowser
            webbrowser.open('http://127.0.0.1:' + str(server.server_address[1]))
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        manager_closed=manager is None
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if manager is not None:
                    manager.close()
                    manager_closed=True
            finally:
                try:
                    # A timed-out worker still owns this workspace. Its thread
                    # keeps the manager/store alive until process exit or retry.
                    if store is not None and manager_closed:
                        store.close()
                finally:
                    if previous_term is not None:
                        signal.signal(signal.SIGTERM, previous_term)
