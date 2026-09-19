"""三幕流水线。

三幕共用一条事件流、一个 run_id、一个签名包 —— 不是三个互相独立的脚本。
这一点是 demo 说服力的来源。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .authority import Authority
from .contracts import ErrorCode, LeaseContext, Rejection, Snapshot
from .delta_cert import CertificateStore, establish_root, validate_or_inherit
from .events import ZERO_HASH, EventLog
from .executor import Executor, ExecutorState
from .geometry import cross_validate, full_check
from .scenarios import make_parent_pair, make_scene, mix, perturb
from .sim_controller import SimController


class Clock:
    """单调时钟。用机器人主机单调时钟判断到期，UTC 只作审计标签。"""

    def __init__(self, start_ns: int = 0):
        self.now_ns = start_ns

    def advance(self, ns: int) -> int:
        self.now_ns += ns
        return self.now_ns


def _snapshot(obs_id: str, context: LeaseContext, position, observed_mono: int):
    return Snapshot(
        obs_id=obs_id,
        robot=context.robot,
        boot=context.boot,
        epoch=context.epoch,
        scene_id=context.scene_id,
        queue_rev=context.queue_rev,
        committed_prefix_hash=context.committed_prefix_hash,
        position=position,
        observed_mono=observed_mono,
    )


# ==================================================================== 第一幕


def act_one_geometry(cases: int, log: EventLog, cross_check: bool = True) -> dict:
    """变换让旧结论失效。

    对每个案例跑三条判定路径：
      a 只验父轨迹（对照组，代表当前普遍做法）
      b 最终全检（基线）
      c Δ-Cert + 必要全检（本项目）
    """
    stats = {
        "cases": cases,
        "path_a_wrong_release": 0,
        "path_a_total_violating": 0,
        "path_b_wrong_release": 0,
        "path_b_full_checks": 0,
        "path_c_wrong_release": 0,
        "path_c_full_checks": 0,
        "path_c_inherited": 0,
        "path_c_false_reject": 0,
        "safe_children_passed": 0,
        "safe_children_total": 0,
        "cross_check_disagreements": 0,
        "root_full_checks": 0,
    }
    samples = []

    for i in range(cases):
        seed = i
        scene = make_scene(seed % 50)
        p1, p2 = make_parent_pair(seed)

        # 建立父证书：一次完整几何检查
        root = establish_root(p1, scene)
        stats["root_full_checks"] += root.full_checks_used
        if root.verdict != "FULL":
            continue  # 父轨迹本身不合格的案例跳过
        parent_cert = root.certificate
        log.append(
            "CERTIFICATE",
            cert_id=parent_cert.cert_id,
            plan_hash=p1.hash,
            verdict=root.verdict,
            path=root.path,
            margins=list(root.margins),
            first_violation_segment=root.first_violation_segment,
            full_checks_used=root.full_checks_used,
            inherit_depth=root.inherit_depth,
            parent_cert_id=root.parent_cert_id,
        )

        # 一半案例用异侧混合（实际违规），一半用同侧扰动（实际安全）
        if i % 2 == 0:
            child, record = mix(p1, p2, plan_id=f"MIX-{i:06d}")
        else:
            child, record = perturb(p1, seed=seed, plan_id=f"NEAR-{i:06d}")

        parents = [p1.hash, p2.hash] if record.kind == "mix" else [p1.hash]
        log.append(
            "TRANSFORM",
            plan_hash=child.hash,
            method="mix" if record.kind == "mix" else "near",
            parent_plan_hashes=parents,
        )

        # 地面真值：独立判断这条子轨迹到底违不违规
        truly_ok, true_margins, first_viol = full_check(child, scene)

        if cross_check and i < 50:
            if not cross_validate(child, scene):
                stats["cross_check_disagreements"] += 1

        # ---- 路径 a：只验父轨迹就放行
        if not truly_ok:
            stats["path_a_total_violating"] += 1
            stats["path_a_wrong_release"] += 1  # 路径 a 放行了一条违规轨迹

        # ---- 路径 b：最终全检
        stats["path_b_full_checks"] += 1
        if not truly_ok:
            pass  # 正确拦下
        else:
            stats["safe_children_total"] += 1

        # ---- 路径 c：Δ-Cert + 必要全检
        v = validate_or_inherit(child, scene, p1, parent_cert, record)
        stats["path_c_full_checks"] += v.full_checks_used
        if v.verdict == "INHERITED":
            stats["path_c_inherited"] += 1
            if not truly_ok:
                stats["path_c_wrong_release"] += 1  # 严重错误：继承放行了违规轨迹
        elif v.verdict == "REJECTED":
            if truly_ok:
                stats["path_c_false_reject"] += 1  # 误拒
        elif v.verdict == "FULL":
            if not truly_ok:
                stats["path_c_wrong_release"] += 1

        if truly_ok and v.verdict in ("INHERITED", "FULL"):
            stats["safe_children_passed"] += 1

        if len(samples) < 8:
            samples.append({
                "case": i,
                "kind": record.kind,
                "truly_ok": truly_ok,
                "verdict": v.verdict,
                "min_margin": round(min(true_margins), 6),
                "first_violation_segment": first_viol,
                "parent_knots": [list(k) for k in p1.points],
                "child_knots": [list(k) for k in child.points],
                "obstacle": scene.obstacles[0].summary(),
            })

    if stats["path_b_full_checks"]:
        stats["full_check_reduction_pct"] = round(
            100.0 * (1 - stats["path_c_full_checks"] / stats["path_b_full_checks"]), 2
        )
    stats["samples"] = samples
    return stats


# ==================================================================== 第二幕

FAULTS = [
    "late_action",
    "lease_expired",
    "lease_replay",
    "scene_changed",
    "queue_rev_changed",
    "revoke_race",
    "cancel_unconfirmed",
]


def act_two_faults(log: EventLog) -> dict:
    """许可与撤销。七类故障各跑一遍。"""
    results = []

    for fault in FAULTS:
        results.append(_run_one_fault(fault, log))

    summary = {
        "faults_run": len(results),
        "blocked": sum(1 for r in results if r["blocked"]),
        "stale_gen_submissions_after_revoke": sum(
            r["stale_submissions_after_revoke"] for r in results
        ),
        "details": results,
    }
    return summary


def _run_one_fault(fault: str, log: EventLog) -> dict:
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    store = CertificateStore()
    authority = Authority(store)
    clock = Clock(1_000_000_000)

    root = establish_root(p1, scene)
    store.register(root.certificate)

    ctx = LeaseContext(
        robot="numeric-robot-0",
        boot=1,
        epoch=0,
        scene_id=scene.scene_id,
        queue_rev=0,
        committed_prefix_hash=ZERO_HASH,
    )
    snap = _snapshot("obs-0", ctx, p1.points[0], clock.now_ns)

    controller = SimController(
        capacity=2,
        ack_delay_ticks=1,
        exec_delay_ticks=1,
        drop_cancel_ack=(fault == "cancel_unconfirmed"),
    )
    ex = Executor(authority, controller, log)

    detail = {"fault": fault, "blocked": False, "code": None,
              "stale_submissions_after_revoke": 0, "cursors": {}}

    lease = None
    try:
        lease = authority.prepare(p1, root.certificate, ctx, snap, clock.now_ns,
                                  prefix_len=4, ttl_ns=500_000_000)
        log.append(
            "PREPARE",
            lease_id=lease.lease_id,
            plan_hash=p1.hash,
            cert_id=root.certificate.cert_id,
            prefix_len=lease.prefix_len,
            deadline_mono=lease.deadline_mono,
        )

        live_ctx = ctx
        commit_snap = snap
        now = clock.now_ns

        if fault == "lease_expired":
            now = clock.advance(2_000_000_000)  # 超过 ttl
        elif fault == "scene_changed":
            live_ctx = replace(ctx, scene_id="scene-9999")
        elif fault == "queue_rev_changed":
            live_ctx = replace(ctx, queue_rev=7)
        elif fault == "late_action":
            commit_snap = _snapshot("obs-late", ctx, (0.9, 0.9, 0.9), clock.now_ns)

        ex.commit(lease, p1, commit_snap, live_ctx, now)

        if fault == "lease_replay":
            # 同一许可第二次提交必须失败
            ex2 = Executor(authority, SimController(), log)
            ex2.commit(lease, p1, snap, ctx, clock.now_ns)

        # 正常推进两步
        for _ in range(3):
            clock.advance(50_000_000)
            ex.tick(clock.now_ns)

        if fault in ("revoke_race", "cancel_unconfirmed"):
            before = len(controller.submitted)
            gen_before = ex.generation
            ex.revoke(reason=fault)
            # 撤销之后继续推进，观察有没有新增旧代次提交
            for _ in range(5):
                clock.advance(50_000_000)
                ex.tick(clock.now_ns)
            after_stale = sum(
                1 for s in controller.submitted[before:] if s["gen"] < gen_before + 1
            )
            detail["stale_submissions_after_revoke"] = after_stale
            detail["submitted_before_revoke"] = before
            detail["observed_after_revoke"] = len(controller.observed)
            ex.poll_cancel()
            detail["blocked"] = after_stale == 0
            if fault == "cancel_unconfirmed":
                try:
                    ex.try_recover(operator_approved=True)
                    detail["blocked"] = False
                    detail["code"] = "RECOVERED_WITHOUT_ACK"  # 不该发生
                except Rejection as exc:
                    detail["blocked"] = True
                    detail["code"] = exc.code
        else:
            detail["blocked"] = False  # 该拒的没拒住

    except Rejection as exc:
        detail["blocked"] = True
        detail["code"] = exc.code
        log.append(
            "COMMIT",
            plan_hash=p1.hash,
            lease_id=lease.lease_id if lease else None,
            cert_id=root.certificate.cert_id,
            accepted=False,
            reason_code=exc.code,
        )

    for _ in range(4):
        controller.tick()
    for step_index, item in enumerate(controller.observed):
        log.append(
            "OBSERVED",
            plan_hash=p1.hash,
            lease_id=lease.lease_id if lease else None,
            cert_id=root.certificate.cert_id,
            step_index=step_index,
            position=list(item["action"]),
            observed=True,
        )
    detail["cursors"] = controller.cursors()
    return detail


# ==================================================================== 第三幕


def act_three_evidence(log: EventLog, out_dir: str) -> dict:
    from .evidence import build_bundle

    log.append(
        "OUTCOME",
        status="completed",
        submitted=len(log.by_type("DISPATCH")),
        accepted=len(log.by_type("CONTROLLER_ACK")),
        observed=len(log.by_type("OBSERVED")),
    )
    return build_bundle(log, out_dir)
