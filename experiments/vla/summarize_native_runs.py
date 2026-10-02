#!/usr/bin/env python3
"""Compare independently verified native VLA runs without replaying a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sentinel_evc.evidence import verify_bundle
from sentinel_evc.scenario import strict_json


def read_run(directory: Path, key: Path) -> tuple[dict, dict, dict]:
    bundle = directory / "bundle"
    result = strict_json((bundle / "result.json").read_bytes())
    ok, detail = verify_bundle(str(bundle), str(key), result["run_id"])
    if not ok:
        raise ValueError(f"Evidence verification failed: {detail}")
    replay = strict_json((bundle / "replay.json").read_bytes())
    return result, replay, {
        "run_id": result["run_id"], "verified": True,
        "result_sha256": hashlib.sha256((bundle / "result.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        "public_key_sha256": hashlib.sha256(key.read_bytes()).hexdigest(),
    }


def wilson(successes: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0,center-radius), min(1,center+radius)]


def describe(values: list[float]) -> dict:
    values = sorted(values)
    return {"count": len(values), "mean": sum(values)/len(values) if values else None,
            **{name: values[max(0,math.ceil(q*len(values))-1)] if values else None
               for name,q in (("p50",.5),("p95",.95),("p99",.99))}}


def index_episodes(result: dict, replay: dict) -> dict:
    frames = {row["episode_id"]: row["frames"] for row in replay["episodes"]}
    output = {}
    for row in result["episodes"]:
        key = (row["task_id"],row["initial_state_index"],row["seed"])
        if key in output or row["episode_id"] not in frames:
            raise ValueError("Duplicate or unlinked episode")
        output[key] = (row, frames[row["episode_id"]])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("baseline","active","fault"):
        parser.add_argument("--"+mode, type=Path, required=True)
        parser.add_argument("--"+mode+"-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = {mode: read_run(getattr(args,mode), getattr(args,mode+"_key")) for mode in ("baseline","active","fault")}
    for mode,(result,_,_) in runs.items():
        if result["mode"] != mode or result["status"] != "complete":
            raise ValueError(f"{mode} run is not complete in the declared mode")
    baseline, active = (index_episodes(*runs[mode][:2]) for mode in ("baseline","active"))
    if set(baseline) != set(active):
        raise ValueError("Paired task/state/seed grid differs")
    pairs = []
    for key in sorted(baseline):
        left,left_frames = baseline[key]; right,right_frames = active[key]
        left_actions = [frame for frame in left_frames if frame.get("action") is not None]
        right_actions = [frame for frame in right_frames if frame.get("action") is not None]
        common = min(len(left_actions),len(right_actions))
        differences = [index for index in range(common) if left_actions[index]["action_bytes_hash"] != right_actions[index]["action_bytes_hash"]]
        maximum = max((abs(a-b) for x,y in zip(left_actions,right_actions) for a,b in zip(x["action"],y["action"])),default=0)
        pairs.append({"task_id":key[0],"initial_state_index":key[1],"seed":key[2],
                      "baseline_success":left["success"],"active_success":right["success"],
                      "baseline_steps":left["env_steps"],"active_steps":right["env_steps"],
                      "compared_actions":common,"different_action_bytes":len(differences),
                      "first_different_action":differences[0] if differences else None,
                      "common_prefix_max_abs_difference":maximum,
                      "whole_action_sequence_equal":not differences and len(left_actions)==len(right_actions),
                      "official_outcome_equal":left["success"]==right["success"]})
    task_rows = []
    for task_id in sorted({row["task_id"] for row in pairs}):
        rows = [row for row in pairs if row["task_id"]==task_id]
        task_rows.append({"task_id":task_id,"episodes_each":len(rows),
                          "baseline_successes":sum(row["baseline_success"] for row in rows),
                          "active_successes":sum(row["active_success"] for row in rows),
                          "action_identical_pairs":sum(row["whole_action_sequence_equal"] for row in rows)})
    result_fault = runs["fault"][0]
    fault_rows = []
    for name in sorted({row["fault"] for row in result_fault["fault_attempts"]}):
        rows = [row for row in result_fault["fault_attempts"] if row["fault"]==name]
        fault_rows.append({"fault":name,"attempts":len(rows),"blocked":sum(row["blocked"] for row in rows),
                           "native_writer_calls":sum(row["env_step_calls"] for row in rows),
                           "reasons":dict(Counter(row["reason"] for row in rows))})
    combined = []
    for episode in runs["active"][1]["episodes"]:
        for frame in episode["frames"]:
            timing = frame.get("latency_ms", {})
            authorization, admission = timing.get("gateway_authorize"), timing.get("gateway_submit_admission_before_writer")
            if authorization is not None and admission is not None:
                combined.append(authorization+admission)
    successes = {mode: runs[mode][0]["metrics"]["task_successes"] for mode in ("baseline","active")}
    identities = {mode: {key:value for key,value in runs[mode][0]["dependencies"]["files"].items() if key != "run/config"} for mode in ("baseline","active")}
    source_match = identities["baseline"] == identities["active"]
    identity_differences = sorted(key for key in set(identities["baseline"]) | set(identities["active"]) if identities["baseline"].get(key) != identities["active"].get(key))
    summary = {"schema":"sentinel-native-vla-comparison-v1","receipts":{mode:run[2] for mode,run in runs.items()},
               "paired_episodes":len(pairs),"dependency_identity_equal":source_match,
               "dependency_differences":identity_differences,"excluded_dependency_fields":["run/config: mode and run ID differ by design"],
               "task_successes":successes,"success_wilson95":{mode:wilson(n,len(pairs)) for mode,n in successes.items()},
               "action_identical_pairs":sum(row["whole_action_sequence_equal"] for row in pairs),
               "outcome_identical_pairs":sum(row["official_outcome_equal"] for row in pairs),
               "different_action_bytes":sum(row["different_action_bytes"] for row in pairs),
               "gateway_authorize_plus_admission_ms":describe(combined),
               "task_results":task_rows,"fault_results":fault_rows,"pairs":pairs,
               "claim_limits":["official checkpoint; task success is not Sentinel-trained-model accuracy",
                               "native request-integrity gateway only; no Panda collision or WorldGuard prediction",
                               "same-process writer instrumentation; not capability isolation against hostile local code",
                               "fault attempts do not count as task success or physical hazards",
                               "timing excludes env.step, inference, rendering and evidence serialization"]}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"comparison.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({key:summary[key] for key in ("paired_episodes","dependency_identity_equal","task_successes","action_identical_pairs","outcome_identical_pairs","different_action_bytes","gateway_authorize_plus_admission_ms")},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
