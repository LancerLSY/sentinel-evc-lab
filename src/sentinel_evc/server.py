"""Loopback-only single-user HTTP transport. Controller lifecycle stays in the worker."""
from __future__ import annotations
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit
from .scenario import Scenario, InputError, strict_json, TEMPLATES
from .runstore import RunStore

MAX_BODY = 65536

class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, store, manager):
        if address[0] != '127.0.0.1':
            raise ValueError('only 127.0.0.1 is supported')
        self.store, self.manager, self.token = store, manager, secrets.token_urlsafe(32)
        super().__init__(address, Handler)
        port = self.server_address[1]
        self.hosts = {'127.0.0.1:'+str(port),'localhost:'+str(port)}
        self.origins = {'http://'+host for host in self.hosts}

class Handler(BaseHTTPRequestHandler):
    server_version = 'SentinelLocal/0.2'
    def log_message(self,*args):
        pass

    def reply(self,status,value,content_type='application/json; charset=utf-8'):
        data = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def error(self,status,code,message):
        self.reply(status,{'error':{'code':code,'message':message}})

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
                return self.reply(200,{'token':self.server.token,'profile':'numeric-simulator-product-v1','templates':TEMPLATES,'capabilities':{'stop':True,'resume':True,'export':True,'real_robot':False}})
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
                    path=self.server.store.directory(run_id)/'evidence.zip'
                    if path.is_symlink() or not path.is_file():raise KeyError('no export')
                    return self.reply(200,path.read_bytes(),'application/zip')
            assets={'':'index.html','index.html':'index.html','app.js':'app.js','styles.css':'styles.css'}
            name='/'.join(parts)
            if name in assets:
                resource=files('sentinel_evc').joinpath('web',assets[name])
                ctype={'index.html':'text/html; charset=utf-8','styles.css':'text/css; charset=utf-8','app.js':'text/javascript; charset=utf-8'}[assets[name]]
                return self.reply(200,resource.read_bytes(),ctype)
            raise KeyError('route not found')
        except KeyError:
            self.error(404,'NOT_FOUND','运行或页面不存在。')
        except (OSError,ValueError):
            self.error(500,'READ_FAILED','读取失败，请查看本地运行状态。')

    def do_POST(self):
        if not self.allowed(True):return
        try:
            parts=self.parts()
            length=self.headers.get('Content-Length','')
            if self.headers.get('Transfer-Encoding') or not length.isdigit() or not 0 < int(length) <= MAX_BODY:
                return self.error(413,'BODY_LIMIT','请求大小超出范围。')
            if self.headers.get('Content-Type','').split(';')[0].strip() != 'application/json':
                return self.error(415,'CONTENT_TYPE','需要 application/json。')
            self.connection.settimeout(5)
            body=strict_json(self.rfile.read(int(length)))
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
        except ValueError:
            self.error(409,'INVALID_STATE','当前状态不支持此操作。')
        except (OSError,TimeoutError):
            self.error(500,'WRITE_FAILED','保存失败，未批准新的动作。')

    def do_OPTIONS(self):
        self.error(405,'METHOD_NOT_ALLOWED','不支持跨来源访问。')


def serve(root='runs/workbench',port=8765):
    from .product_pipeline import ProductManager
    store=RunStore(root)
    manager=ProductManager(store)
    server=LocalServer(('127.0.0.1',port),store,manager)
    print('Sentinel 本地操作台：http://127.0.0.1:'+str(server.server_address[1]),flush=True)
    print('数值模拟；Ctrl+C 停止服务。',flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.server_close()
        store.close()
