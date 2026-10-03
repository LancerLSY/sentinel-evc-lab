#!/usr/bin/env python3
"""Execution-gating patterns vs 16 software-contract faults (pilot, not frozen).

Patterns
  P1 check-then-execute : the plan is verified once when it is queued; nothing is re-checked
                          at commit or dispatch (all runtime guards removed from EVC v2).
  P2 signed-token       : byte binding + one-time use + deadline at commit, but no context,
                          feedback, tracking, generation, completion-budget or recovery gates.
  EVC v1                : the repository's main branch (unchanged).
  EVC v2                : this branch (exact-byte binding, scene-content binding, completion
                          budget, revocation instant, idle-revoke context, atomic consumption).

A fault counts as BLOCKED only if no invalid authority reaches the controller (or, for the
recovery faults, the executor refuses to recover).  Faults are injected software conditions,
not physical hazards.  Each class is one deterministic constructed instance (not a rate).
Contexts carry scene_hash and revocations carry their instant, as the shipped v2 callers do.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

V2_SRC = Path(__file__).resolve().parents[2] / "src" / "sentinel_evc"
# main-branch checkout, e.g. `git worktree add ../evc-main main`
V1_SRC = Path(os.environ.get("SENTINEL_V1_SRC", str(Path(__file__).resolve().parents[3] / "evc-main" / "src" / "sentinel_evc")))
TMP = Path(tempfile.mkdtemp(prefix="evc_patterns_"))


def neutral(c):
    return f"if False and ({c}):"


RUNTIME_GUARDS = {
    "executor.py": [
        ("if lease.plan_hash != plan.hash:", neutral("lease.plan_hash != plan.hash")),
        ("if exact is not None and exact != plan.exact_hash:", neutral("exact is not None and exact != plan.exact_hash")),
        ("if getattr(live, field_name) != getattr(permitted, field_name):", neutral("getattr(live, field_name) != getattr(permitted, field_name)")),
        ("if math.dist(snapshot.position, plan.knots[0]) > START_TUBE:", neutral("math.dist(snapshot.position, plan.knots[0]) > START_TUBE")),
        ("if math.dist(snapshot.position, self._plan.knots[expected_index]) > START_TUBE:", neutral("math.dist(snapshot.position, self._plan.knots[expected_index]) > START_TUBE")),
        ("if not self._matches_controller_feedback(snapshot, now_ns):", neutral("not self._matches_controller_feedback(snapshot, now_ns)")),
        ("if live.epoch != permitted.epoch or live.epoch != generation:", neutral("live.epoch != permitted.epoch or live.epoch != generation")),
        ("if now_ns + len(self._pending) * int(self._plan.dt * 1e9) > self._lease.deadline_mono_ns:", neutral("now_ns + len(self._pending) * int(self._plan.dt * 1e9) > self._lease.deadline_mono_ns")),
        ("if self._controller.cancel_acked is not True:", neutral("self._controller.cancel_acked is not True")),
        ("if not self._controller.is_drained:", neutral("not self._controller.is_drained")),
        ("if snapshot.capture_mono_ns <= max(self._revoked_after_capture, self._revoked_at_ns):", neutral("snapshot.capture_mono_ns <= max(self._revoked_after_capture, self._revoked_at_ns)")),
        # recovery-context comparison (P1/P2 have no notion of a recovery contract)
        ("if getattr(live_context, field_name) != getattr(previous, field_name):", neutral("getattr(live_context, field_name) != getattr(previous, field_name)")),
        ("if live_context.queue_rev < previous.queue_rev:", neutral("live_context.queue_rev < previous.queue_rev")),
    ],
    "authority.py": [
        ("if lease.plan_hash != plan_hash:", neutral("lease.plan_hash != plan_hash")),
        ("if lease.lease_id in self._consumed:", neutral("lease.lease_id in self._consumed")),
        ("if now_ns > lease.deadline_mono_ns:", neutral("now_ns > lease.deadline_mono_ns")),
        ("if lease.authority_generation != self._generation:", neutral("lease.authority_generation != self._generation")),
        ("if context.epoch != self._generation:", neutral("context.epoch != self._generation")),
        ("if context.scene_hash is not None and cert.scene_hash != context.scene_hash:", neutral("context.scene_hash is not None and cert.scene_hash != context.scene_hash")),
    ],
}
KEEP_FOR_P2 = {"if lease.plan_hash != plan.hash:", "if exact is not None and exact != plan.exact_hash:",
               "if lease.plan_hash != plan_hash:", "if lease.lease_id in self._consumed:",
               "if now_ns > lease.deadline_mono_ns:"}

PATTERNS = {
    "P1_check_then_execute": (V2_SRC, RUNTIME_GUARDS),
    "P2_signed_token": (V2_SRC, {f: [(o, n) for o, n in reps if o not in KEEP_FOR_P2] for f, reps in RUNTIME_GUARDS.items()}),
    "EVC_v1_main": (V1_SRC, {}),
    "EVC_v2": (V2_SRC, {}),
}


def build(name, src, patches):
    pkg = "evcpat_" + name.lower()
    dst = TMP / pkg
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "web"))
    for f, reps in patches.items():
        p = dst / f
        t = p.read_text()
        for old, new in reps:
            if old in t:
                t = t.replace(old, new)
        p.write_text(t)
    sys.path.insert(0, str(TMP))
    return {m: importlib.import_module(f"{pkg}.{m}") for m in
            ("contracts", "scenarios", "delta_cert", "authority", "executor", "sim_controller", "events")}


FAULTS = ["action_replaced", "one_ulp_replacement", "late_action", "stale_feedback", "lease_expired",
          "deadline_tail", "lease_replay", "concurrent_replay", "scene_changed", "scene_content_changed_same_id",
          "queue_rev_changed", "stale_lease_after_recovery", "revoke_race", "cancel_unconfirmed",
          "pre_revoke_snapshot_recovery", "idle_revoke_context_change", "queue_changed_during_dispatch"]
T0, DT = 1_000_000_000, 50_000_000


def run(M, fault):
    C, Sc, D, A, E, S, Ev = (M[k] for k in ("contracts", "scenarios", "delta_cert", "authority", "executor", "sim_controller", "events"))
    Rej = C.Rejection
    has_scene_hash = "scene_hash" in C.Context.__dataclass_fields__
    scene = Sc.make_scene(0)
    p1, _ = Sc.make_parent_pair(0)
    store = D.CertificateStore(); auth = A.Authority(store)
    cert = D.establish_root(p1, scene).certificate; store.register(cert)

    def ctx(**kw):
        if has_scene_hash:
            kw.setdefault("scene_hash", scene.hash)
        else:
            kw.pop("scene_hash", None)
        kw.setdefault("scene_id", scene.scene_id)
        return C.Context(**kw)

    log = Ev.EventLog("pat", maxlen=100_000)
    class Stallable(S.SimController):
        """Reference controller whose feedback stream can stop updating (stuck driver thread)."""
        stalled_at = None

        def read_feedback(self, now_ns):
            f = super().read_feedback(now_ns)
            return f if self.stalled_at is None else dict(f, capture_mono_ns=self.stalled_at)

    ctrl = Stallable(capacity=2, drop_cancel_ack=(fault == "cancel_unconfirmed"), initial_position=p1.knots[0])
    ex = E.Executor(auth, ctrl, log)
    c0 = ctx()
    snap = C.Snapshot("obs-0", p1.knots[0], T0)
    seq = [0]

    def fb(t):
        seq[0] += 1
        f = ctrl.read_feedback(t)
        return C.Snapshot(f"obs-{seq[0]}", f["position"], t)

    def revoke(t):
        try:
            ex.revoke("fault", now_ns=t)
        except TypeError:
            ex.revoke("fault")

    def drive(n, t, c):
        for _ in range(n):
            t += DT
            ex.tick(t, fb(t), c)
        return t

    try:
        if fault == "deadline_tail":
            lease = auth.prepare(p1, cert, c0, snap, T0, prefix_len=4, ttl_ns=4 * DT)
            ex.commit(lease, p1, snap, c0, T0)
            t = T0 + 10_000_000
            times = []
            for _ in range(6):
                t += DT
                before = len(ctrl.submitted)
                try:
                    ex.tick(t, fb(t), c0)
                except Rej:
                    break
                if len(ctrl.submitted) > before:
                    times.append(t)
            late = [x for x in times if x + DT > lease.deadline_mono_ns]
            return "BLOCKED" if not late else f"ADMITTED({len(late)} steps finish after the deadline)"
        if fault in ("pre_revoke_snapshot_recovery", "idle_revoke_context_change", "stale_lease_after_recovery",
                     "revoke_race", "cancel_unconfirmed"):
            at_rest = fault in ("idle_revoke_context_change", "pre_revoke_snapshot_recovery")
            K = 1 if at_rest else 4
            lease = auth.prepare(p1, cert, c0, snap, T0, prefix_len=K)
            if fault == "stale_lease_after_recovery":
                spare = auth.prepare(p1, cert, c0, snap, T0, prefix_len=4)
            ex.commit(lease, p1, snap, c0, T0)
            t = drive(6 if at_rest else 1, T0, c0)      # at_rest: lease finished, arm stationary
            if fault == "pre_revoke_snapshot_recovery":
                ctrl.stalled_at = t + 1                    # feedback stream stalls just before the revoke
                early = C.Snapshot("early", ctrl.read_feedback(t + 1)["position"], t + 1)
            before = len(ctrl.submitted)
            revoke(t + 5)
            for i in range(5):
                # the stalled case advances only nanoseconds so the stalled sample stays younger than
                # the observation-age bound; only the revocation-instant rule can then reject it
                t += 1 if fault == "pre_revoke_snapshot_recovery" else DT
                try:
                    ex.tick(t, fb(t), c0)
                except Rej:
                    pass
            if fault == "revoke_race":
                return "BLOCKED" if len(ctrl.submitted) == before else "ADMITTED"
            ex.poll_cancel()
            rec_ctx = ctx(epoch=ex.generation, scene_id=("moved-scene" if fault == "idle_revoke_context_change" else scene.scene_id))
            if fault == "pre_revoke_snapshot_recovery":
                rec_snap, rec_now = early, t + 10          # recovery happens after the revoke instant
            else:
                rec_now = t + 2
                rec_snap = fb(rec_now)
            try:
                ex.try_recover(True, rec_snap, rec_ctx, rec_now)
            except Rej as e:
                if fault != "stale_lease_after_recovery":
                    return "BLOCKED:" + e.code
                return "SETUP_FAILED:" + e.code
            if fault == "stale_lease_after_recovery":
                s2 = fb(t + 3)
                ex.commit(spare, p1, s2, ctx(epoch=ex.generation), t + 3)
                return "ADMITTED"
            return "ADMITTED(recovered)"
        lease = auth.prepare(p1, cert, c0, snap, T0, prefix_len=4)
        if fault == "action_replaced":
            q, _ = Sc.perturb(p1, magnitude=0.001, seed=3, plan_id=p1.plan_id)
            ex.commit(lease, q, snap, c0, T0); drive(2, T0, c0); return "ADMITTED"
        if fault == "one_ulp_replacement":
            q = C.Plan(p1.plan_id, (p1.knots[0],) + tuple((math.nextafter(x, 1.0), y, z) for x, y, z in p1.knots[1:]), p1.dt)
            ex.commit(lease, q, snap, c0, T0); drive(2, T0, c0); return "ADMITTED"
        if fault == "late_action":
            ctrl.set_initial_position((p1.knots[0][0] + 0.02, p1.knots[0][1], p1.knots[0][2]))
            ex.commit(lease, p1, fb(T0), c0, T0); return "ADMITTED"
        if fault == "stale_feedback":
            ctrl.set_initial_position((p1.knots[0][0] + 0.002, p1.knots[0][1], p1.knots[0][2]))
            ex.commit(lease, p1, C.Snapshot("copy", p1.knots[0], T0), c0, T0); return "ADMITTED"
        if fault == "lease_expired":
            t = T0 + 600_000_000
            ex.commit(lease, p1, fb(t), c0, t); return "ADMITTED"
        if fault == "lease_replay":
            ex.commit(lease, p1, snap, c0, T0)
            ex2 = E.Executor(auth, S.SimController(initial_position=p1.knots[0]), log)
            ex2.commit(lease, p1, C.Snapshot("o2", p1.knots[0], T0), c0, T0); return "ADMITTED"
        if fault == "concurrent_replay":
            orig = auth.validate_available

            def slow(l):
                orig(l)
                time.sleep(0.05)              # widen the check-to-record window
            auth.validate_available = slow
            wins = []

            def go(i):
                c2 = S.SimController(initial_position=p1.knots[0])
                e2 = E.Executor(auth, c2, log)
                try:
                    e2.commit(lease, p1, C.Snapshot(f"c{i}", p1.knots[0], T0), c0, T0)
                    wins.append(i)
                except Rej:
                    pass
                except Exception:
                    pass
            th = [threading.Thread(target=go, args=(i,)) for i in range(2)]
            [x.start() for x in th]; [x.join() for x in th]
            return "BLOCKED" if len(wins) <= 1 else f"ADMITTED({len(wins)} commits of one lease)"
        if fault == "queue_changed_during_dispatch":
            ex.commit(lease, p1, snap, c0, T0)
            before = len(ctrl.submitted)
            t = T0 + DT
            ex.tick(t, fb(t), ctx(queue_rev=7))     # another writer changed the queue after commit
            return "BLOCKED" if len(ctrl.submitted) == before else "ADMITTED"
        if fault == "scene_changed":
            ex.commit(lease, p1, snap, ctx(scene_id="scene-9999"), T0); return "ADMITTED"
        if fault == "scene_content_changed_same_id":
            moved = Sc.make_scene(7)              # different obstacle positions, same scene_id label
            live = ctx()
            if has_scene_hash:
                live = C.Context(scene_id=scene.scene_id, scene_hash=moved.hash)
                lease = auth.prepare(p1, cert, live, snap, T0, prefix_len=4)
            ex.commit(lease, p1, snap, live, T0); return "ADMITTED"
        if fault == "queue_rev_changed":
            ex.commit(lease, p1, snap, ctx(queue_rev=7), T0); return "ADMITTED"
    except Rej as e:
        return "BLOCKED:" + e.code
    return "UNKNOWN"


def main():
    import types
    out = {}
    runs = []
    for name, (src, patches) in PATTERNS.items():
        M = build(name, src, patches)
        runs.append((name, M))
        if name == "EVC_v2":
            pipe = importlib.import_module("evcpat_evc_v2.executor_pipeline")
            runs.append(("EVC_v2_pipelined", dict(M, executor=types.SimpleNamespace(Executor=pipe.PipelinedExecutor))))
    for name, M in runs:
        out[name] = {f: run(M, f) for f in FAULTS}
        blocked = sum(v.startswith("BLOCKED") for v in out[name].values())
        print(f"{name:24s} blocked {blocked}/{len(FAULTS)}")
        for f, v in out[name].items():
            if not v.startswith("BLOCKED"):
                print(f"    {f:32s} {v}")
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
