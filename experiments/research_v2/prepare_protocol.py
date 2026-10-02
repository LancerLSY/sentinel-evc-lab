"""Prepare immutable sources and distributions; execute before formal runs."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
old = json.loads((root / "docs/research/2026-10-02/worldguard-scenarios/protocol.json").read_text())
scenes = old["scenarios"]
for spec in scenes.values():
    spec.pop("seed_start")
scenes["low_friction"]["declared_profile_floor"] = .015
source_files = ["experiments/gpu/run_paper_interventions.py", "experiments/gpu/train_mujoco_world.py",
                "experiments/gpu/train_mujoco_visual.py", "experiments/gpu/evaluate_worldguard_scenarios.py",
                "src/sentinel_evc/physics.py", "experiments/arm/run_paper_cost.py",
                "experiments/arm/run_ur5e_guard.py", "experiments/arm/verify_ur5e_roots.py"]
protocol = {
    "schema": "sentinel-paper-v2-protocol-v1", "protocol_id": "paper-v2-20261002", "status": "frozen",
    "preregistration": {"external": False, "ordering": "commit before formal execution", "prior_results_informed_design": True},
    "frozen_sources": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},
    "frozen_model_artifacts": old["frozen_sources"],
    "data_partitions": {
        "train": "frozen original W2 only", "calibration": "frozen original W2 only",
        "dev": {"roots_per_scene": 2, "seed_starts": {s: 790000+1000*i for i,s in enumerate(old["scenario_order"])}},
        "test": {"roots_per_scene": 100, "seed_starts": {s: 710000+1000*i for i,s in enumerate(old["scenario_order"])}},
        "prior_test_seed_blocks": ["610000-615099", "620000-620099"],
        "independent_unit": "root; siblings, frames, policies and repeats remain paired"},
    "worldguard_intervention": {
        "scenes": scenes, "durations_s": [.6,.9,1.2,1.6,3.2,4.8], "evaluation_s":5., "posthold_s":.5,
        "history_frames":8, "frame_dt_s":.05, "w2_contract": "first four candidates, first 40 future frames only",
        "risk_xy_m":.06, "endpoint_tolerance_m":.01,
        "policies": ["fixed_1p6_unbound","fixed_4p8_unbound","declared_support_floor","unknown_reject","integrated_support_gate"],
        "routes": {"nominal":"frozen visual W2", "camera_shift":"frozen state W2", "low_friction":"declared floor .015 fallback", "other":"UNKNOWN"},
        "forbidden_decision_inputs": ["actual root friction", "actual root mass", "future labels", "test results"],
        "primary_metrics": ["unsafe/all roots", "completion/all roots", "rejection/all roots"],
        "completion": "no risk during 5.5s and endpoint error <=.01m; rejection is incomplete"},
    "same_information_counterfactual": {"pairs":100,"seed_start":716000,"preflight_seed_start":796000,
        "future_friction_branches":[.3,.025], "durations_s":[1.6,4.8],
        "required": "same observation/history/action bytes; only future friction changes"},
    "evc_cost": {"test_root_base":720000,"preflight_root_base":820000,"scene_stride":1000,
        "test_roots_per_scene":10,"preflight_roots_per_scene":2,"warmup":1,"timed_repeats":3,
        "obstacles":"sampled before plan without FK", "review":"1ms 0.005rad independent implementation",
        "primary":"marginal Cdelta vs Cfull", "amortized_K":[1,2,4,8],
        "equal_parent_cost":"Cparent/K + Cdelta versus Cparent/K + Cfull",
        "asset_commit":"4d038b3feae26ec82b46a4d586379114012a8ac7"},
    "statistics":{"binary":"Wilson95", "paired_bootstrap_replicates":10000,"bootstrap_seed":20261006,
                  "cost_bootstrap_seed":9102026,"multiplicity":"scene-stratified estimates; no unplanned significance claims"},
    "execution":{"hardware":"RTX 4090 D (24 GB)","workers":6,"torch_threads":4},
    "stop_conditions":["source/model hash mismatch","nonempty output","test seed overlaps dev/prior","test outcome changes policy"],
    "claim_limits":["offline paired MuJoCo support routing","not hardware","not external preregistration","not SO100 closedloop","not continuous safety proof"]}
out = root / "experiments/research_v2/protocol.json"
if out.exists():
    raise FileExistsError("protocol already exists; a scientific change requires a new version")
out.write_text(json.dumps(protocol, indent=2, sort_keys=True)+"\n")
print(hashlib.sha256(out.read_bytes()).hexdigest())
