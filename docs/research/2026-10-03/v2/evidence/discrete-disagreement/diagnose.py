#!/usr/bin/env python3
"""Post-hoc diagnosis of the one frozen N=128, jitter=.01 disagreement."""

import argparse
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_samples(plan, step):
    chunks = [np.asarray(plan[0], dtype=float)[None]]
    for left, right in zip(plan[:-1], plan[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(right - left)) / step)))
        chunks.append(left[None] + (right - left)[None] * np.arange(1, n + 1)[:, None] / n)
    out = np.concatenate(chunks)
    assert np.max(np.abs(np.diff(out, axis=0))) <= step + 1e-14
    return out


def compiled_xml(path, obstacle_radius, obstacle_count):
    root = ET.parse(path).getroot()
    for node in root.iter():
        for child in list(node):
            if (child.tag == "geom" and child.get("mesh")) or child.tag == "mesh":
                node.remove(child)
    world = root.find("worldbody")
    collision = [g for g in world.iter("geom") if g.get("class") in ("collision", "eef_collision")]
    for i, geom in enumerate(collision):
        geom.set("name", f"audit_robot_{i}")
    for i in range(obstacle_count):
        body = ET.SubElement(world, "body", name=f"audit_obstacle_{i}", mocap="true", pos="2 2 2")
        ET.SubElement(body, "geom", name=f"audit_sphere_{i}", type="sphere",
                      size=str(obstacle_radius), rgba=".95 .25 .15 1")
    return ET.tostring(root, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dense-step", type=float, default=0.0005)
    args = ap.parse_args()
    sys.path.insert(0, str(args.arm_dir))
    from arm_delta import certify_full, discrete_batch
    from scenarios import OBSTACLE_RADIUS
    from ur5e_kin import UR5eModel

    args.out_dir.mkdir(parents=True, exist_ok=False)
    model = UR5eModel(args.mjcf)
    with np.load(args.trace) as z:
        candidates = z["candidates"]
        obstacles = z["obstacles"]
        reference = z["reference_ok"]
    mismatch = []
    comparisons = 0
    for root in range(len(candidates)):
        for cycle in range(len(candidates[root])):
            discrete, _ = discrete_batch(model, candidates[root, cycle], obstacles[root], OBSTACLE_RADIUS, 0.01)
            comparisons += len(discrete)
            for candidate in np.flatnonzero(discrete != reference[root, cycle]):
                mismatch.append((root, cycle, int(candidate), bool(discrete[candidate]),
                                 bool(reference[root, cycle, candidate])))
        print("scanned root", root, flush=True)
    if len(mismatch) != 1:
        raise RuntimeError(f"expected exactly one unique mismatch, found {mismatch}")
    root, cycle, candidate, discrete_ok, reference_ok = mismatch[0]
    plan = candidates[root, cycle, candidate]
    obs = obstacles[root]
    full = certify_full(model, plan, obs, OBSTACLE_RADIUS)
    if bool(full.ok) != reference_ok:
        raise RuntimeError("stored reference and fresh continuous checker differ")

    Q = independent_samples(plan, args.dense_step)
    min_obstacle = {"clearance_m": float("inf")}
    min_self = {"clearance_m": float("inf")}
    for start in range(0, len(Q), 4096):
        chunk = Q[start:start + 4096]
        co, cs = model.clearances(chunk, obs, OBSTACLE_RADIUS)
        io = np.unravel_index(int(np.argmin(co)), co.shape)
        if float(co[io]) < min_obstacle["clearance_m"]:
            min_obstacle = {"clearance_m": float(co[io]), "dense_index": start + int(io[0]),
                            "capsule": int(io[1]), "q": chunk[io[0]].tolist()}
        if cs.size:
            iss = np.unravel_index(int(np.argmin(cs)), cs.shape)
            if float(cs[iss]) < min_self["clearance_m"]:
                min_self = {"clearance_m": float(cs[iss]), "dense_index": start + int(iss[0]),
                            "pair": int(iss[1]), "q": chunk[iss[0]].tolist()}

    import mujoco
    xml = compiled_xml(args.mjcf, OBSTACLE_RADIUS, len(obs))
    xml_path = args.out_dir / "diagnostic_collision_model.xml"
    xml_path.write_text(xml)
    mm = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(mm)
    robot_geoms = np.array([mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM,
                                             f"audit_robot_{i}") for i in range(len(model.capsules))])
    sphere_geoms = {mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"audit_sphere_{i}") for i in range(len(obs))}
    robot_set = set(robot_geoms.tolist())
    bindings = []
    for index, geom_id in enumerate(robot_geoms):
        cap = model.capsules[index]
        body = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, int(mm.geom_bodyid[geom_id]))
        wanted = mujoco.mjtGeom.mjGEOM_CYLINDER if cap.kind == "cylinder" else mujoco.mjtGeom.mjGEOM_CAPSULE
        assert body == model.bodies[cap.body].name
        assert mm.geom_type[geom_id] == wanted and mm.geom_group[geom_id] == 3
        assert mm.geom_size[geom_id, 0] == cap.radius and mm.geom_size[geom_id, 1] == cap.half
        bindings.append({"geom": int(geom_id), "body": body, "kind": cap.kind,
                         "radius": cap.radius, "half_length": cap.half})

    contacts = []
    contact_sample_count = 0
    for dense_index, q in enumerate(Q):
        data.qpos[:6] = q
        data.qvel[:] = 0
        data.mocap_pos[:] = obs
        mujoco.mj_forward(mm, data)
        sample_contacts = []
        for j in range(data.ncon):
            g1, g2 = int(data.contact[j].geom1), int(data.contact[j].geom2)
            robot_robot = g1 in robot_set and g2 in robot_set
            robot_sphere = ((g1 in robot_set and g2 in sphere_geoms)
                            or (g2 in robot_set and g1 in sphere_geoms))
            if robot_robot or robot_sphere:
                sample_contacts.append({"geom1": g1, "geom2": g2,
                                        "distance_m": float(data.contact[j].dist)})
        if sample_contacts:
            contact_sample_count += 1
            if not contacts:
                contacts = [{"dense_index": dense_index, "q": q.tolist(),
                             "contacts": sample_contacts}]

    np.savez_compressed(args.out_dir / "trajectory_instance.npz", plan=np.asarray(plan, dtype="<f8"),
                        obstacles=np.asarray(obs, dtype="<f8"), root=np.int64(root),
                        cycle=np.int64(cycle), candidate=np.int64(candidate),
                        discrete_ok=np.bool_(discrete_ok), continuous_ok=np.bool_(reference_ok))
    result = {
        "schema": "sentinel-discrete-disagreement-posthoc-v1",
        "analysis_timing": "post-hoc diagnostic after the frozen performance outcomes",
        "scope": "one frozen N=128, jitter=0.01 candidate; no timing rerun",
        "trace_sha256": sha(args.trace),
        "mjcf_sha256": sha(args.mjcf),
        "source_sha256": {"diagnose.py": sha(__file__),
                          "arm_delta.py": sha(args.arm_dir / "arm_delta.py"),
                          "ur5e_kin.py": sha(args.arm_dir / "ur5e_kin.py")},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "mujoco": mujoco.__version__, "platform": platform.platform()},
        "unique_candidate_comparisons": comparisons,
        "mismatch_count": 1,
        "location": {"root": root, "cycle": cycle, "candidate": candidate},
        "direction": {"discrete_0.01_ok": discrete_ok, "continuous_reference_ok": reference_ok,
                      "fresh_continuous_ok": bool(full.ok), "fresh_continuous_status": full.status,
                      "fresh_continuous_fk_evals": full.fk_evals},
        "independent_dense_proxy_check": {"joint_step_rad": args.dense_step,
                                           "samples": len(Q),
                                           "min_obstacle": min_obstacle,
                                           "min_self": min_self,
                                           "sampled_collision": min(min_obstacle["clearance_m"],
                                                                    min_self["clearance_m"]) < 0},
        "independent_mujoco_dense_check": {
            "joint_step_rad": args.dense_step, "samples": len(Q),
            "robot_or_obstacle_contact_samples": contact_sample_count,
            "first_contact": contacts[0] if contacts else None,
            "model_binding": bindings,
            "boundary": "dense sampled mj_forward contact evaluation, not MuJoCo continuous collision detection"
        },
        "artifacts": {"trajectory_instance.npz": sha(args.out_dir / "trajectory_instance.npz"),
                      "diagnostic_collision_model.xml": sha(xml_path)}
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
