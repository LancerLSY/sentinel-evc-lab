"""Bounded, versioned public input. No controller commands enter through HTTP."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from .contracts import Scene, Sphere

class InputError(ValueError):
    pass

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise InputError('duplicate JSON field')
        out[key] = value
    return out

def strict_json(raw: bytes):
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(InputError('non-finite JSON number')))
    except (UnicodeError, ValueError) as exc:
        raise InputError('invalid JSON: ' + str(exc).split(':')[0]) from exc

def number(value, lo, hi, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise InputError(name + ' outside supported range')
    return float(value)

def vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise InputError(name + ' must contain three numbers')
    return tuple(number(x, -2, 2, name) for x in value)

def parse_scene(value):
    if value is None:
        return Scene('numeric-workspace-v1', (), (-.5,-.5,-.5), (1, .5,.5), .01,.005)
    if not isinstance(value, dict) or set(value) != {'obstacles','workspace','tool_radius','tracking_reserve'}:
        raise InputError('unsupported scene fields')
    workspace = value['workspace']
    if not isinstance(workspace, dict) or set(workspace) != {'lo','hi'}:
        raise InputError('invalid workspace')
    lo, hi = vector(workspace['lo'],'workspace.lo'), vector(workspace['hi'],'workspace.hi')
    if any(a >= b for a,b in zip(lo,hi)):
        raise InputError('workspace bounds must increase')
    obs = value['obstacles']
    if not isinstance(obs,list) or len(obs) > 16:
        raise InputError('at most 16 obstacles')
    spheres = []
    for item in obs:
        if not isinstance(item,dict) or set(item) != {'center','radius'}:
            raise InputError('invalid sphere')
        spheres.append(Sphere(vector(item['center'],'center'),number(item['radius'],.001,.5,'radius')))
    return Scene('numeric-custom-v1',tuple(spheres),lo,hi,number(value['tool_radius'],.001,.1,'tool_radius'),number(value['tracking_reserve'],.005,.05,'tracking_reserve'))

@dataclass(frozen=True)
class Scenario:
    name: str = '数值搬运'
    seed: int = 7
    displacement: float = .35
    risk_limit: float = .12
    prediction_mode: str = 'physical'
    scene_json: str = 'null'
    schema_version: str = 'product-v1'

    @classmethod
    def parse(cls, obj):
        if not isinstance(obj,dict) or set(obj) - {'schema_version','name','seed','displacement','risk_limit','prediction_mode','scene'}:
            raise InputError('unsupported scenario fields')
        if obj.get('schema_version','product-v1') != 'product-v1':
            raise InputError('unsupported schema version')
        name = obj.get('name','数值搬运')
        if not isinstance(name,str) or not 1 <= len(name.strip()) <= 80 or any(ord(x) < 32 for x in name):
            raise InputError('name must be 1–80 printable characters')
        seed = obj.get('seed',7)
        if type(seed) is not int or not 0 <= seed <= 1_000_000:
            raise InputError('seed must be an integer from 0 to 1000000')
        mode = obj.get('prediction_mode','physical')
        if mode not in ('physical','residual'):
            raise InputError('unsupported prediction mode')
        parse_scene(obj.get('scene'))
        return cls(name.strip(),seed,number(obj.get('displacement',.35),.01,.6,'displacement'),number(obj.get('risk_limit',.12),.001,.5,'risk_limit'),mode,json.dumps(obj.get('scene'),sort_keys=True,allow_nan=False))

    @property
    def scene(self):
        return parse_scene(json.loads(self.scene_json))

    def summary(self):
        return {'schema_version':self.schema_version,'name':self.name,'seed':self.seed,'displacement':self.displacement,'risk_limit':self.risk_limit,'prediction_mode':self.prediction_mode,'scene':json.loads(self.scene_json)}

TEMPLATES = [
    {'id':'transfer','name':'标准搬运','scenario':Scenario().summary()},
    {'id':'cautious','name':'严格后果门控','scenario':Scenario('严格后果门控',7,.35,.035).summary()},
    {'id':'residual','name':'残差基线','scenario':Scenario('残差基线',11,.35,.12,'residual').summary()},
]
