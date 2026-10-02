#!/usr/bin/env python3
"""Prospective, root-paired support interventions; no test-set fitting.

The router consumes declared profiles, never evaluator friction/mass/labels.
Long-horizon physics and calibrated 2 s W2 forecasts are separate profiles.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import train_mujoco_world as w2
import train_mujoco_visual as visual
import evaluate_worldguard_scenarios as prior
from sentinel_evc.physics import PhysicsConfig, model_xml

DURATIONS = (0.6, 0.9, 1.2, 1.6, 3.2, 4.8)
SCENES = prior.SCENARIO_ORDER

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def targets(start, duration, scale, np):
    u = np.minimum(np.arange(1, 101) * .05 / duration, 1.)
    s = u**3 * (10 - 15*u + 6*u*u)
    return (start + s[:, None] * np.asarray((.35, .10, .08)) * scale).astype(np.float32)

def floor_choice(start, scale, floor, np):
    for index in (3, 4, 5):
        plan = targets(start, DURATIONS[index], scale, np)
        a = np.diff(np.vstack((start, start, plan.astype(np.float64))), n=2, axis=0) / .05**2
        if np.all((9.81+a[:, 2] > 0) &
                  (np.linalg.norm(a[:, :2], axis=1) <= floor*(9.81+a[:, 2]))):
            return index
    return -1

def branch(mujoco, model, state, start, prior_target, plan, np, capture=False):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    outputs, positions, qpos = [], [], []
    risk = drop = False
    for index in range(110):
        target = plan[min(index, 99)]
        w2._advance_frame(mujoco, model, data, prior_target, target)
        prior_target = target
        _, output, _, floor = w2._observe(mujoco, model, data)
        r = float(np.linalg.norm(output[:2]))
        drop = drop or floor or abs(output[0]) > .14 or abs(output[1]) > .12
        risk = risk or r > .06 or drop
        if index < 40:
            outputs.append(output)
        positions.append(np.asarray(data.xpos[model.body("tray").id]).copy())
        if capture:
            qpos.append(data.qpos.copy())
    endpoint = float(np.linalg.norm(positions[-1] - plan[-1]))
    return np.asarray(outputs, np.float32), risk, drop, endpoint, np.asarray(qpos)

def generate(task):
    scene, seed, spec = task
    import mujoco
    import numpy as np
    rng = random.Random(seed)
    lo, hi = spec["friction_range"]
    friction = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    mass = rng.uniform(*spec["mass_range_kg"])
    model = mujoco.MjModel.from_xml_string(model_xml(PhysicsConfig(friction=friction, payload_mass=mass, seed=seed)))
    data = mujoco.MjData(model)
    adr = model.joint("payload_free").qposadr[0]
    data.qpos[adr:adr+2] = (rng.uniform(-.008, .008), rng.uniform(-.008, .008))
    mujoco.mj_forward(model, data)
    for _ in range(150):
        mujoco.mj_step(model, data)
    base = np.asarray((0., 0., .45)); last = base.copy()
    history, past = [], []
    for frame in range(8):
        phase = math.pi * (frame+1) / 6
        target = base + (np.asarray((.008*math.sin(phase)**2, .004*math.sin(2*phase), .002*math.sin(phase)**2)) if frame<6 else 0)
        w2._advance_frame(mujoco, model, data, last, target); last = target
        history.append(w2._observe(mujoco, model, data)[0]); past.append(target)
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    start = np.asarray(data.xpos[model.body("tray").id]).copy()
    future = np.asarray([targets(start, d, spec["displacement_scale"], np) for d in DURATIONS])
    outcomes = [branch(mujoco, model, state, start, last, plan, np, capture=seed % 1000 == 0) for plan in future]
    return {"root_id": f"{scene}-{seed}", "history": np.asarray(history, np.float32),
            "past_targets": np.asarray(past, np.float32), "future_targets": future,
            "truth": np.stack([x[0] for x in outcomes]), "unsafe": np.asarray([x[1] for x in outcomes]),
            "drop": np.asarray([x[2] for x in outcomes]), "endpoint": np.asarray([x[3] for x in outcomes]),
            "integration_state": state, "capture_qpos": np.asarray([x[4] for x in outcomes]),
            "metadata": {"seed": seed, "friction": friction, "mass_kg": mass,
                         "integration_state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}}

def proportion(k, n):
    if not n:
        return {"count": k, "n": n, "rate": None, "wilson95": None}
    z = 1.959963984540054; p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {"count": k, "n": n, "rate": p, "wilson95": [max(0., center-radius), min(1., center+radius)]}

def summarize(choice, split, np):
    selected = choice >= 0; idx = np.arange(len(choice)); safe_idx = np.maximum(choice, 0)
    unsafe = selected & split["unsafe"][idx, safe_idx]
    dropped = selected & split["drop"][idx, safe_idx]
    complete = selected & ~unsafe & (split["endpoint"][idx, safe_idx] <= .01)
    n = len(choice)
    return {"unsafe": proportion(int(unsafe.sum()), n), "completion": proportion(int(complete.sum()), n),
            "rejection": proportion(int((~selected).sum()), n), "drops": proportion(int(dropped.sum()), n),
            "unsafe_given_execution": proportion(int(unsafe.sum()), int(selected.sum())),
            "duration_counts": {str(d): int(np.sum(selected & (choice==i))) for i,d in enumerate(DURATIONS)},
            "mean_duration_s": float(np.mean(np.asarray(DURATIONS)[choice[selected]])) if selected.any() else None}, unsafe, complete

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "state-run", "visual-run", "weights", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--preflight", action="store_true"); p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(); protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen":
        raise ValueError("protocol must be frozen")
    for rel, expected in protocol["frozen_sources"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"source drift: {rel}")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("output must be empty")
    import mujoco
    import numpy as np
    import torch
    import torchvision
    torch.set_num_threads(4); device = torch.device("cuda")
    artifacts = {"state_manifest_sha256": sha(args.state_run/"manifest.json"),
                 "state_metrics_sha256": sha(args.state_run/"metrics.json"),
                 "visual_manifest_sha256": sha(args.visual_run/"manifest.json"),
                 "visual_metrics_sha256": sha(args.visual_run/"metrics.json"),
                 "visual_cache_sha256": sha(args.visual_run/"visual_cache.npz"),
                 "encoder_weights_sha256": sha(args.weights)}
    if artifacts != protocol["frozen_model_artifacts"]:
        raise ValueError("model artifact drift")
    sm = json.loads((args.state_run/"manifest.json").read_text())
    vm = json.loads((args.visual_run/"manifest.json").read_text())
    state_models = prior._load_npz_models(args.state_run, sm["members"], w2._model_class(torch), (sm["normalizers"],), torch,np,device)
    visual_models = prior._load_npz_models(args.visual_run, vm["training"]["visual_members"], visual._model_class(torch), (vm["normalizers"],True),torch,np,device)
    state_metric = json.loads((args.state_run/"metrics.json").read_text())["gru128_ensemble"]
    visual_metric = json.loads((args.visual_run/"metrics.json").read_text())["visual_object_gru128"]
    with np.load(args.visual_run/"visual_cache.npz", allow_pickle=False) as cache:
        pca = [(cache[f"pca_{i}_mean"].copy(),cache[f"pca_{i}_components"].copy(),cache[f"pca_{i}_eigenvalues"].copy()) for i in range(2)]
    args.out.mkdir(parents=True); dump(args.out/"protocol.json",protocol)
    started=time.time(); metrics={}; root_records=[]; saved={}
    partition = protocol["data_partitions"]["dev" if args.preflight else "test"]
    n=partition["roots_per_scene"]
    for scene in SCENES:
        spec=protocol["worldguard_intervention"]["scenes"][scene]
        tasks=[(scene,partition["seed_starts"][scene]+i,spec) for i in range(n)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(generate,tasks))
        split={key:np.stack([row[key] for row in rows]) for key in ("history","past_targets","future_targets","truth","unsafe","drop","endpoint","integration_state")}
        abort=(split["history"][:,-1,21]<.5)|(np.linalg.norm(split["history"][:,-1,6:8],axis=1)>.06)
        choices={"fixed_1p6_unbound":np.full(n,3),"fixed_4p8_unbound":np.full(n,5),"unknown_reject":np.full(n,-1)}
        floor=spec.get("declared_profile_floor")
        physical=np.asarray([floor_choice(row["history"][-1,:3],spec["displacement_scale"],floor,np) if floor is not None else -1 for row in rows])
        choices["declared_support_floor"]=physical.copy()
        integrated=np.full(n,-1); coverage=None; route="MODEL_UNKNOWN"
        if scene in ("nominal","camera_shift"):
            model_split={"history":split["history"],"past_targets":split["past_targets"],"future_targets":split["future_targets"][:,:4,:40],"truth":split["truth"][:,:4],"initial_output":split["history"][:,-1,6:21]}
            if scene=="nominal":
                encoder,_=visual._encoder(args.weights,torch,torchvision,device)
                raw,pathmeta=visual._render_raw(scene,model_split,visual.CAMERAS,args.out/"raw.npy",encoder,SimpleNamespace(out=args.out,render_batch_size=64),mujoco,np,torch,torchvision,device)
                model_split["visual"]=visual._project(raw,pca,np); del raw,encoder
                (args.out/"raw.npy").unlink()
                pred=visual._predict(visual_models,model_split,torch,device,np); metric=visual_metric
                route="VISUAL_ORIGINAL_PROFILE"
            else:
                pred=w2._predict(state_models,model_split,torch,device,np); metric=state_metric
                route="CAMERA_PROFILE_MISMATCH_ROUTE_STATE"
            cal=prior._frozen_calibration(metric,pred,model_split["truth"],np)
            allowed=cal["allowed"].astype(bool); integrated=np.where(allowed.any(1),allowed.argmax(1),-1)
            coverage=proportion(int(cal["risk_xy_covered"].sum()),n)
            saved[scene+"_prediction"]=pred; saved[scene+"_allowed"]=allowed
        elif scene=="low_friction":
            integrated=physical.copy(); route="DECLARED_FLOOR_FALLBACK"
        elif scene=="displacement_shift":
            route="ACTION_FAMILY_UNSUPPORTED"
        choices["integrated_support_gate"]=integrated
        for choice in choices.values():
            choice[abort]=-1
        scene_metrics={}; outcomes={}
        for name,choice in choices.items():
            scene_metrics[name],unsafe,complete=summarize(choice,split,np)
            outcomes[name]=(unsafe,complete)
            saved[scene+"_"+name+"_choice"]=choice
        paired={}; rng=np.random.default_rng(20261006)
        indices=rng.integers(0,n,size=(10000,n))
        for name in ("fixed_1p6_unbound","fixed_4p8_unbound","unknown_reject"):
            for metric,index in (("unsafe",0),("completion",1)):
                diff=outcomes["integrated_support_gate"][index].astype(float)-outcomes[name][index]
                paired[metric+"_minus_"+name]={"difference":float(diff.mean()),"paired_root_bootstrap95":np.quantile(diff[indices].mean(1),[.025,.975]).tolist()}
        metrics[scene]={"policies":scene_metrics,"paired_differences":paired,"route":route,"supported_w2_xy_coverage":coverage,"roots":n}
        for row_i,row in enumerate(rows):
            root_records.append({"root_id":row["root_id"],"evaluator_only":row["metadata"],"declared_profile":spec,
                                 "decision_route":route,"choices":{k:int(v[row_i]) for k,v in choices.items()},
                                 "outcomes_by_duration":[{"duration_s":d,"unsafe":bool(split["unsafe"][row_i,j]),"drop":bool(split["drop"][row_i,j]),"endpoint_m":float(split["endpoint"][row_i,j])} for j,d in enumerate(DURATIONS)]})
            if row["capture_qpos"].size:
                saved[scene+"_example_qpos"]=row["capture_qpos"]
        for key,value in split.items(): saved[scene+"_"+key]=value
        print(json.dumps({"scene":scene,"complete":n,"metrics":scene_metrics["integrated_support_gate"]}),flush=True)
    # A separate causal diagnostic: identical observed histories/actions, only future friction changes.
    counter=[]
    for i in range(2 if args.preflight else protocol["same_information_counterfactual"]["pairs"]):
        seed=796000+i if args.preflight else 716000+i
        row=generate(("counterfactual",seed,protocol["worldguard_intervention"]["scenes"]["nominal"]))
        inputs=b"".join(row[k].tobytes() for k in ("history","past_targets","future_targets"))
        outcomes=[]
        for friction in (.3,.025):
            model=mujoco.MjModel.from_xml_string(model_xml(PhysicsConfig(friction=friction,payload_mass=row["metadata"]["mass_kg"],seed=seed)))
            results=[branch(mujoco,model,row["integration_state"],row["history"][-1,:3],row["past_targets"][-1],row["future_targets"][j],np) for j in (3,5)]
            outcomes.append([{"duration_s":DURATIONS[j],"unsafe":bool(x[1]),"drop":bool(x[2]),"endpoint_m":float(x[3])} for j,x in zip((3,5),results)])
        digest=hashlib.sha256(inputs).hexdigest()
        counter.append({"root_id":f"counterfactual-{seed}","input_hash_nominal":digest,"input_hash_low":digest,"byte_equal_inputs":True,"outcomes":outcomes})
    dump(args.out/"metrics.json",metrics); dump(args.out/"per_root.json",root_records); dump(args.out/"counterfactual.json",counter)
    np.savez_compressed(args.out/"results.npz",**saved)
    manifest={"status":"complete","preflight":args.preflight,"protocol_sha256":sha(args.protocol),"source_sha256":sha(__file__),"model_artifacts":artifacts,
              "root_count":n*6,"counterfactual_pairs":len(counter),"wall_seconds":time.time()-started,"environment":{"mujoco":mujoco.__version__,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"gpu_memory_bytes":torch.cuda.get_device_properties(0).total_memory},
              "scope":"offline paired simulation support routing; no product permit, hardware, SO100 or continuous-collision claim",
              "output_hashes":{name:sha(args.out/name) for name in ("metrics.json","per_root.json","counterfactual.json","results.npz")}}
    dump(args.out/"manifest.json",manifest)

if __name__=="__main__": main()
