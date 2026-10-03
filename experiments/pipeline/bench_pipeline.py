#!/usr/bin/env python3
"""Stop-and-go vs pipelined one-time leases (pilot, not frozen).

For a 40-step plan and lease length K, reports
  ticks               control cycles until the 40th action is observed
  leases              one-time leases consumed
  auth_age_ms         age of the authorization (lease prepare instant) when each
                      step is written to the controller: mean / max
  idle_ticks          cycles in which the controller had a free slot but no
                      authorised step was available

Numbers are cycles of the reference SimController (dt = 50 ms, capacity 2 or 4),
not wall-clock latency of any robot.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
import run_tests as _rt  # noqa: E402

_rt._install_pytest_shim()
from test_executor_pipeline import DT, Rig  # noqa: E402


class TimedRig(Rig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prepared_at = {}

    def try_commit(self):
        before = set(self.lease_plans)
        ok = super().try_commit()
        for lid in set(self.lease_plans) - before:
            self.prepared_at[lid] = self.t
        return ok


def main(out):
    res = {}
    for cap in (2, 4):
        for K in (1, 2, 4, 8):
            row = {}
            for pipelined in (False, True):
                r = TimedRig(pipelined, K, max_ahead=2, cap=cap)
                n = idle = 0
                ages = []
                while len(r.ctrl.observed) < 40 and n < 1000:
                    free = r.ctrl.free_slots() > 0
                    sub_before = len(r.ctrl.submitted)
                    r.step()
                    n += 1
                    if len(r.ctrl.submitted) > sub_before:
                        lid = r.log.events()[-1]["payload"].get("lease_id")
                        ages.append((r.t - r.prepared_at[lid]) / 1e6)
                    elif free and len(r.ctrl.submitted) < 40:
                        idle += 1
                assert [o["action"] for o in r.ctrl.observed][:40] == [tuple(k) for k in r.plan.knots[1:41]]
                row["pipelined" if pipelined else "stop_and_go"] = {
                    "ticks": n, "leases": len(r.lease_plans), "idle_ticks": idle,
                    "auth_age_ms_mean": round(statistics.fmean(ages), 1), "auth_age_ms_max": round(max(ages), 1),
                }
            res[f"cap={cap},K={K}"] = row
            s, p = row["stop_and_go"], row["pipelined"]
            print(f"cap={cap} K={K}: stop-and-go {s['ticks']} ticks (idle {s['idle_ticks']}, age {s['auth_age_ms_mean']}/{s['auth_age_ms_max']} ms)"
                  f" | pipelined {p['ticks']} ticks (idle {p['idle_ticks']}, age {p['auth_age_ms_mean']}/{p['auth_age_ms_max']} ms)")
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
