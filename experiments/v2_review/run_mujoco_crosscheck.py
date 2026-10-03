#!/usr/bin/env python3
"""Independent MuJoCo collision/FK cross-check of the naked UR5e profile."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/arm_kinematic')]
from ur5e_kin import UR5eModel, BatchedCertifiedChecker
from scenarios import SCENARIOS, OBSTACLE_RADIUS, make_case


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def independent_samples(plan, step):
    chunks=[np.asarray(plan[0])[None]]
    for left,right in zip(plan[:-1],plan[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(right-left))/step)))
        chunks.append(left[None]+(right-left)[None]*np.arange(1,n+1)[:,None]/n)
    out=np.concatenate(chunks)
    assert np.max(np.abs(np.diff(out,axis=0))) <= step+1e-14
    return out
def protocol(mjcf):
    return {'schema':'sentinel-ur5e-mujoco-crosscheck-v1','state':'frozen_before_outcomes',
      'seed':920031,'upstream_commit':'4d038b3feae26ec82b46a4d586379114012a8ac7',
      'mjcf_sha256':sha(mjcf),'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/arm_kinematic/ur5e_kin.py',ROOT/'experiments/arm_kinematic/arm_delta.py',ROOT/'experiments/arm_kinematic/scenarios.py']},
      'grid':{'historical_scenarios':list(SCENARIOS),'roots_per_scenario':30,'random_joint_configurations':2000,'dense_joint_step_rad':0.0005},
      'acceptance':{'fk_endpoint_max_error_m':1e-10,'certified_free_mujoco_contact':0},
      'model_transform':'remove visual mesh geoms/assets only; retain official collision geoms, bodies, joints, inertials and actuators; add two obstacle spheres',
      'scope':'bare UR5e capsule/cylinder self collision and two static spheres; kinematic FK/contact evaluation on independent dense samples, not MuJoCo continuous collision detection',
      'excluded':['VLA inference','dynamic tracking','floor/table','gripper/attached tools','arbitrary mesh obstacles','hardware safety']}


def compiled_xml(path):
    root=ET.parse(path).getroot()
    for node in root.iter():
        for child in list(node):
            if (child.tag=='geom' and child.get('mesh')) or child.tag=='mesh': node.remove(child)
    world=root.find('worldbody')
    for i,geom in enumerate(g for g in world.iter('geom') if g.get('class') in ('collision','eef_collision')):
        geom.set('name',f'audit_robot_{i}')
    for i in range(2):
        body=ET.SubElement(world,'body',name=f'audit_obstacle_{i}',mocap='true',pos='2 2 2')
        ET.SubElement(body,'geom',name=f'audit_sphere_{i}',type='sphere',size=str(OBSTACLE_RADIUS),rgba='.95 .25 .15 1')
    return ET.tostring(root,encoding='unicode')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--mjcf',required=True,type=Path)
    ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    expected=protocol(a.mjcf)
    if a.mode=='prepare':
        if a.protocol.exists(): raise FileExistsError(a.protocol)
        save(a.protocol,expected);print('Frozen',sha(a.protocol));return
    if json.loads(a.protocol.read_text()) != expected: raise RuntimeError('frozen inputs changed')
    import mujoco
    a.out.mkdir(parents=True,exist_ok=False)
    xml=compiled_xml(a.mjcf);(a.out/'collision_model.xml').write_text(xml)
    mm=mujoco.MjModel.from_xml_string(xml);dd=mujoco.MjData(mm);kin=UR5eModel(a.mjcf)
    robot_geoms=np.array([mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_GEOM,f'audit_robot_{i}') for i in range(9)])
    assert len(robot_geoms)==len(kin.capsules)==9
    sphere_geoms={mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_GEOM,f'audit_sphere_{i}') for i in range(2)}
    robot_set=set(robot_geoms.tolist());binding=[]
    for g,gi in enumerate(robot_geoms):
        c=kin.capsules[g];body=mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_BODY,int(mm.geom_bodyid[gi]))
        assert gi>=0 and body==kin.bodies[c.body].name
        wanted=mujoco.mjtGeom.mjGEOM_CYLINDER if c.kind=='cylinder' else mujoco.mjtGeom.mjGEOM_CAPSULE
        assert mm.geom_type[gi]==wanted and mm.geom_group[gi]==3
        assert mm.geom_size[gi,0]==c.radius and mm.geom_size[gi,1]==c.half
        binding.append({'geom':int(gi),'body':body,'kind':c.kind,'radius':c.radius,'half_length':c.half})
    rng=np.random.default_rng(expected['seed']); static=[];max_fk=0.;sphere_disagree=self_disagree=0
    far=np.array([[2.,2.,2.],[2.2,2.,2.]])
    def pose(q,obs):
        dd.qpos[:6]=q;dd.qvel[:]=0;dd.mocap_pos[:]=obs;mujoco.mj_forward(mm,dd)
    def contacts():
        return [j for j in range(dd.ncon) if (int(dd.contact[j].geom1) in robot_set and int(dd.contact[j].geom2) in robot_set|sphere_geoms) or (int(dd.contact[j].geom2) in robot_set and int(dd.contact[j].geom1) in sphere_geoms)]
    def contact(): return bool(contacts())
    for i in range(expected['grid']['random_joint_configurations']):
        q=rng.uniform(kin.low,kin.high);pose(q,far)
        A,B=kin.capsule_segments(q[None]);o,s=kin.clearances(q[None],far,OBSTACLE_RADIUS)
        proxy=bool(np.any(o<0) or np.any(s<0));mj=bool(contact());self_disagree+=proxy!=mj
        for g,gi in enumerate(robot_geoms):
            center=dd.geom_xpos[gi];axis=dd.geom_xmat[gi].reshape(3,3)[:,2];half=mm.geom_size[gi,1]
            max_fk=max(max_fk,float(np.abs(A[0,g]-(center-half*axis)).max()),float(np.abs(B[0,g]-(center+half*axis)).max()))
        static.append({'index':i,'q':q.tolist(),'capsule_contact':proxy,'mujoco_contact':mj,'mujoco_contacts':int(dd.ncon),
                       'mujoco_contact_but_capsule_free':mj and not proxy})
    save(a.out/'static_rows.json',static)
    rows=[];samples=free_count=free_contact=0;render_case=None
    for scenario in SCENARIOS:
        for root_id in range(expected['grid']['roots_per_scenario']):
            case=make_case(kin,scenario,root_id)
            for kind in ('parent','final'):
                plan=case[kind];obs=case[kind+'_obstacles']
                checker=BatchedCertifiedChecker(kin,obs,OBSTACLE_RADIUS);verdict=checker.check_plan(plan)
                Q=independent_samples(plan,expected['grid']['dense_joint_step_rad']);hit=False;first=None;count=0
                for q in Q:
                    pose(q,obs);count+=1
                    if contact():
                        hit=True
                        if first is None:
                            first={'q':q.tolist(),'contacts':[{'geom1':int(dd.contact[j].geom1),'geom2':int(dd.contact[j].geom2),'dist':float(dd.contact[j].dist)} for j in contacts()]}
                samples+=count;free_count+=bool(verdict['ok']);free_contact+=bool(verdict['ok'] and hit)
                rows.append({'scenario':scenario,'root':root_id,'kind':kind,'continuous_status':verdict['status'],
                    'continuous_ok':bool(verdict['ok']),'fk_evals':checker.fk_evals,'dense_samples':count,'mujoco_contact':hit,'first_contact':first})
                if scenario=='late_suffix' and kind=='final' and hit and render_case is None:
                    render_case={'scenario':scenario,'root':root_id,'plan':plan.tolist(),'obstacles':obs.tolist(),'first_contact':first}
        print(scenario,len(rows),'free_contact',free_contact,flush=True)
    save(a.out/'motion_rows.json',rows)
    if render_case: save(a.out/'contact_demo_case.json',render_case)
    out={'schema':'sentinel-ur5e-mujoco-crosscheck-result-v1','protocol_sha256':sha(a.protocol),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None},
        'model_xml_sha256':sha(a.out/'collision_model.xml'),'robot_collision_geoms':len(robot_geoms),'geom_binding':binding,
        'static':{'configurations':len(static),'max_fk_axis_endpoint_error_m':max_fk,'capsule_mujoco_label_disagreements':self_disagree,
                  'mujoco_contact_but_capsule_free':sum(x['mujoco_contact_but_capsule_free'] for x in static),'mujoco_contact_configs':sum(x['mujoco_contact'] for x in static)},
        'motion':{'paths':len(rows),'dense_configurations':samples,'certified_free_paths':free_count,'certified_free_mujoco_contact':free_contact,
                  'statuses':{s:sum(r['continuous_status']==s for r in rows) for s in ['free','collision','joint_limit','unknown']},
                  'mujoco_contact_paths':sum(r['mujoco_contact'] for r in rows)},
        'acceptance_pass':max_fk<=1e-10 and free_contact==0 and not any(x['mujoco_contact_but_capsule_free'] for x in static),
        'artifacts':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}}
    save(a.out/'result.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
