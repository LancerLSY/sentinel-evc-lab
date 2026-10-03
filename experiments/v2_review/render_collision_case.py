#!/usr/bin/env python3
"""Render the saved, rejected UR5e proposal with official visual meshes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'arm_kinematic'))
from ur5e_kin import UR5eModel
from scenarios import make_case

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--case',type=Path,required=True);ap.add_argument('--font',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(exist_ok=False,parents=True);case=json.loads(a.case.read_text());model=UR5eModel(a.assets/'ur5e.xml');c=make_case(model,case['scenario'],case['root'])
 assert np.array_equal(np.asarray(case['plan']),c['final']) and np.array_equal(np.asarray(case['obstacles']),c['final_obstacles'])
 xml=ET.parse(a.assets/'ur5e.xml').getroot();xml.find('compiler').set('meshdir',str((a.assets/'assets').resolve()))
 world=xml.find('worldbody');ET.SubElement(world,'geom',type='plane',size='2 2 .01',rgba='.16 .20 .26 1',contype='0',conaffinity='0')
 for i in range(2):
  b=ET.SubElement(world,'body',name=f'obs{i}',mocap='true');ET.SubElement(b,'geom',name=f'sphere{i}',type='sphere',size='.055',rgba='.98 .30 .18 1')
 visual=xml.find('visual')
 if visual is None: visual=ET.SubElement(xml,'visual')
 ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
 m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));d=mujoco.MjData(m)
 renderer=mujoco.Renderer(m,height=740,width=940);option=mujoco.MjvOption();option.geomgroup[3]=0
 camera=mujoco.MjvCamera();camera.lookat[:]=[0,-.08,.40];camera.distance=1.85;camera.azimuth=135;camera.elevation=-23
 fonts={s:ImageFont.truetype(a.font,s) for s in (24,30,36,46)}
 trails={name:model.geom_center(c[name],len(model.capsules)-1) for name in ('parent','final')}
 def marker(pos,radius,color):
  g=renderer.scene.geoms[renderer.scene.ngeom]
  mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius,0,0]),pos,np.eye(3).ravel(),np.asarray(color))
  renderer.scene.ngeom+=1
 def panel(name,obs,t):
  plan=c[name]
  u=min(t,1)*40;i=min(int(u),39);q=(1-(u-i))*plan[i]+(u-i)*plan[i+1]
  d.qpos[:6]=q;d.qvel[:]=0;d.mocap_pos[:]=obs;mujoco.mj_forward(m,d)
  renderer.update_scene(d,camera=camera,scene_option=option)
  hits=[j for j in range(d.ncon) if int(d.contact[j].geom1)!=int(d.contact[j].geom2)]
  for p in trails[name][:int(u)+1]:marker(p,.004,[.2,.95,.72,1] if name=='parent' else [1,.55,.18,1])
  for j in hits:marker(d.contact[j].pos,.013,[1,.12,.1,1])
  return renderer.render().copy(),q.tolist(),bool(hits)
 frames=300;fps=30;dest=a.out/'ur5e_changed_chunk.mp4'
 ff=imageio_ffmpeg.get_ffmpeg_exe();process=subprocess.Popen([ff,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=(a.out/'encoder.log').open('wb'))
 records=[]
 for f in range(frames):
  t=min(f/(fps*8),1);left,ql,hl=panel('parent',c['parent_obstacles'],t);right,qr,hr=panel('final',c['final_obstacles'],t)
  canvas=Image.new('RGB',(1920,1080),(12,18,28));draw=ImageDraw.Draw(canvas)
  draw.text((34,20),'一次改动，两条轨迹',font=fonts[46],fill=(244,247,252))
  draw.text((35,78),'One revised chunk | UR5e official meshes | saved kinematic case',font=fonts[24],fill=(151,172,199))
  canvas.paste(Image.fromarray(left),(20,170));canvas.paste(Image.fromarray(right),(960,170))
  draw.text((35,124),'原轨迹：验证通过 / Certified reference',font=fonts[30],fill=(62,211,177))
  draw.text((978,124),'修改后：拒绝放行 / Rejected proposal',font=fonts[30],fill=(255,141,111))
  draw.rectangle((958,170,961,910),fill=(72,86,107))
  draw.text((35,925),'轨迹采样无接触 / No sampled contact',font=fonts[30],fill=(62,211,177))
  draw.text((978,925),'发生碰撞 / CONTACT' if hr else '正在靠近障碍 / Approaching obstacle',font=fonts[30],fill=(255,87,80) if hr else (231,206,171))
  draw.text((35,986),'动作改了，就重新验证。右侧为被拒动作的碰撞重放。',font=fonts[30],fill=(239,242,247))
  draw.text((35,1030),'Saved motion/contact replay. Green/orange dots mark tool paths; red dots mark MuJoCo contacts.',font=fonts[24],fill=(142,162,185))
  process.stdin.write(np.asarray(canvas).tobytes());records.append({'frame':f,'time_s':f/fps,'path_fraction':t,'reference_q':ql,'proposal_q':qr,'reference_contact':hl,'proposal_contact':hr})
  if f in (150,225,260):canvas.save(a.out/f'frame-{f}.png')
 process.stdin.close();assert process.wait()==0;renderer.close()
 (a.out/'frames.json').write_text(json.dumps(records)+'\n')
 receipt={'schema':'sentinel-v2-case-media-v1','source_sha256':sha(__file__),'case_sha256':sha(a.case),'official_asset_receipt_sha256':sha(a.assets/'source-receipt.json'),'fps':fps,'frames':frames,'resolution':[1920,1080],
 'contact_frames':{'reference':sum(r['reference_contact'] for r in records),'proposal':sum(r['proposal_contact'] for r in records)},'scope':'counterfactual kinematic replay of late_suffix/root0; not controller execution','files':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}}
 (a.out/'media-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
