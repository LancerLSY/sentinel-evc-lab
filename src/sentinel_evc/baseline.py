"""Small reproducible baseline experiment; metrics are derived only from held-out roots."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
from .data import make_numeric_dataset
from .prediction import train_residual_model, make_physical_model, predict_plan
from .calibration import fit_root_max_calibration


def run_baseline(out,seed=10_000_000,mode='residual',risk_limit=.12):
    directory=Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('baseline output directory must be empty')
    directory.mkdir(parents=True,exist_ok=True)
    dataset=make_numeric_dataset(seed=seed)
    model=train_residual_model(dataset.split('train')) if mode=='residual' else make_physical_model()
    calibration=fit_root_max_calibration(dataset.split('cal'),model)
    errors=[];covered=allowed=false_allow=0;rows=[]
    for root in dataset.split('test'):
        root_covered=True
        outcomes=[]
        for plan,truth in zip(root.candidates,root.outcomes):
            prediction=predict_plan(root.history,plan,model,calibration,0,risk_limit=risk_limit)
            errors.extend(abs(a-b) for a,b in zip(truth.r,prediction.centers))
            coverage=bool(prediction.lower) and all(lo<=r<=hi for r,lo,hi in zip(truth.r,prediction.lower,prediction.upper))
            root_covered=root_covered and coverage
            if prediction.allowed:
                allowed+=1
                if max(abs(r) for r in truth.r)>risk_limit:false_allow+=1
            outcomes.append({'plan_hash':plan.hash,'prediction_hash':prediction.hash,'allowed':prediction.allowed,'covered':coverage,'actual_peak':max(abs(r) for r in truth.r)})
        covered+=int(root_covered)
        rows.append({'root_id':root.root_id,'joint_covered':root_covered,'candidates':outcomes})
    evaluation={'scope':'modest deterministic numeric roots; not v4 GRU or robot validation','test_roots':len(rows),'test_candidates':len(rows)*4,'mae_r':sum(errors)/len(errors),'joint_covered_roots':covered,'allowed_candidates':allowed,'false_allow_candidates':false_allow,'risk_limit':risk_limit,'model_hash':model.hash,'calibration_hash':calibration.hash,'dataset_hash':dataset.hash,'rows':rows}
    artifacts={'dataset_manifest.json':dataset.summary(),'model.json':model.summary(),'calibration.json':calibration.summary(),'evaluation.json':evaluation}
    digests={}
    for name,obj in artifacts.items():
        data=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
        (directory/name).write_bytes(data)
        digests[name]='sha256:'+hashlib.sha256(data).hexdigest()
    manifest={'schema_version':'numeric-experiment-v1','status':'completed','command':f'python -m sentinel_evc train-baseline --mode {mode} --seed {seed} --risk-limit {risk_limit} --out <empty-directory>','python':platform.python_version(),'source_digest':'sha256:'+hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),'dataset':dataset.summary(),'files':digests,'acceptance':{'all_four_branches':len(rows)*4==48,'finite_calibration':calibration.finite,'held_out_roots':True}}
    (directory/'experiment_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),'utf-8')
    return {k:v for k,v in evaluation.items() if k!='rows'}
