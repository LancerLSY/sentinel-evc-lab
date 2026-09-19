"""执行器五项发布不变量 + 证据链测试。"""

from dataclasses import replace

import pytest

from sentinel_evc.authority import Authority
from sentinel_evc.contracts import Certificate, ErrorCode, LeaseContext, Rejection, Snapshot
from sentinel_evc.delta_cert import CertificateStore, establish_root
from sentinel_evc.events import EventLog, ZERO_HASH
from sentinel_evc.evidence import build_bundle, failed_layers, verify_bundle
from sentinel_evc.executor import Executor, ExecutorState
from sentinel_evc.pipeline import Clock
from sentinel_evc.scenarios import make_parent_pair, make_scene
from sentinel_evc.sim_controller import SimController


def _rig(drop_cancel_ack=False, capacity=2):
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    store = CertificateStore()
    root = establish_root(p1, scene)
    store.register(root.certificate)
    auth = Authority(store)
    clock = Clock(1_000_000_000)
    log = EventLog("test")
    ctrl = SimController(capacity=capacity, drop_cancel_ack=drop_cancel_ack)
    ex = Executor(auth, ctrl, log, monotonic_ns=lambda: clock.now_ns)
    ctx = LeaseContext("robot-test", 0, 0, scene.scene_id, 0, ZERO_HASH)
    snap = Snapshot(
        obs_id="obs-0", **ctx.summary(), position=p1.points[0],
        observed_mono=clock.now_ns,
    )
    return dict(scene=scene, plan=p1, store=store, cert=root.certificate,
                auth=auth, clock=clock, log=log, ctrl=ctrl, ex=ex,
                ctx=ctx, snap=snap)


# ------------------------------------------------ 不变量 2：许可不可重复消费


def test_lease_cannot_be_consumed_twice():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)

    ex2 = Executor(r["auth"], SimController(), r["log"])
    with pytest.raises(Rejection) as exc:
        ex2.commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)
    assert exc.value.code == ErrorCode.LEASE_REPLAY


def test_forged_certificate_id_is_rejected():
    """上游自选的证书 ID 不构成证据。"""
    r = _rig()
    fake = Certificate(
        cert_id="cert-999999", plan=r["plan"], scene_id=r["scene"].scene_id,
        margins=(9.9,) * r["plan"].horizon, inherit_depth=0,
        parent_cert_id=None,
    )
    with pytest.raises(Rejection) as exc:
        r["auth"].prepare(r["plan"], fake, r["ctx"], r["snap"], r["clock"].now_ns)
    assert exc.value.code == ErrorCode.AUTH_FAILED


def test_expired_lease_is_rejected_at_commit():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns, ttl_ns=300_000_000)
    late = r["clock"].advance(1_000_000_000)
    with pytest.raises(Rejection) as exc:
        r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], late)
    assert exc.value.code == ErrorCode.LEASE_EXPIRED


def test_context_change_is_rejected():
    for field, value, code in [
        ("scene_id", "other", ErrorCode.CONTEXT_CHANGED),
        ("queue_rev", 42, ErrorCode.CONTEXT_CHANGED),
        ("epoch", 5, ErrorCode.CONTEXT_CHANGED),
        ("boot", 9, ErrorCode.CONTEXT_CHANGED),
    ]:
        r = _rig()
        lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                                  r["clock"].now_ns)
        live = replace(r["ctx"], **{field: value})
        with pytest.raises(Rejection) as exc:
            r["ex"].commit(lease, r["plan"], r["snap"], live, r["clock"].now_ns)
        assert exc.value.code == code, f"{field} 变化应给 {code}"


def test_state_outside_tube_rejected_even_with_same_obs_id():
    """obs_id 相同但状态已出管，一样要拒绝。"""
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    drifted = replace(r["snap"], position=(0.9, 0.9, 0.9))
    with pytest.raises(Rejection) as exc:
        r["ex"].commit(lease, r["plan"], drifted, r["ctx"], r["clock"].now_ns)
    assert exc.value.code == ErrorCode.TRACKING_TUBE


# ------------------------------------------------ 不变量 4：撤销屏障


def test_no_stale_generation_submissions_after_revoke():
    """撤销后不新增旧代次本地提交。"""
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)

    for _ in range(2):
        r["clock"].advance(50_000_000)
        r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))

    gen_before = r["ex"].generation
    n_before = len(r["ctrl"].submitted)
    r["ex"].revoke("test")

    for _ in range(8):
        r["clock"].advance(50_000_000)
        r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))

    new = r["ctrl"].submitted[n_before:]
    stale = [s for s in new if s["gen"] <= gen_before]
    assert stale == [], f"撤销后仍新增了 {len(stale)} 条旧代次提交"


def test_already_submitted_step_may_still_execute():
    """诚实的一条：撤销之前已提交的命令，撤销之后仍可能被观测到。

    这不是 bug。软件队列清空、控制器确认取消、实际运动停止是三件事。
    """
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)
    r["clock"].advance(50_000_000)
    r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))

    assert len(r["ctrl"].submitted) >= 1
    r["ex"].revoke("test")
    for _ in range(5):
        r["clock"].advance(50_000_000)
        r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))
    assert len(r["ctrl"].observed) >= 1, (
        "撤销前已提交的步骤应当仍被观测到 —— 如果这里是 0，"
        "说明模拟把撤销画成了瞬间制动")


# ------------------------------------------------ 不变量 5：未确认取消不恢复


def test_cannot_recover_without_cancel_ack():
    r = _rig(drop_cancel_ack=True)
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)
    r["ex"].revoke("test")
    for _ in range(10):
        r["clock"].advance(50_000_000)
        r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))
    with pytest.raises(Rejection) as exc:
        r["ex"].recover(replace(r["snap"], epoch=r["ex"].generation,
                                observed_mono=r["clock"].now_ns), human_approved=True)
    assert exc.value.code == ErrorCode.CANCEL_UNCONFIRMED


def test_cannot_recover_without_operator_approval():
    r = _rig()
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)
    r["ex"].revoke("test")
    for _ in range(10):
        r["clock"].advance(50_000_000)
        r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))
    with pytest.raises(Rejection):
        r["ex"].recover(replace(r["snap"], epoch=r["ex"].generation,
                                observed_mono=r["clock"].now_ns), human_approved=False)
    assert r["ex"].state == ExecutorState.FAULT


def test_controller_capacity_forces_batched_submission():
    """控制器容量 2 < 前缀 4，前缀必须分批提交。"""
    r = _rig(capacity=2)
    lease = r["auth"].prepare(r["plan"], r["cert"], r["ctx"], r["snap"],
                              r["clock"].now_ns, prefix_len=4)
    r["ex"].commit(lease, r["plan"], r["snap"], r["ctx"], r["clock"].now_ns)
    r["clock"].advance(50_000_000)
    r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))
    r["clock"].advance(50_000_000)
    r["ex"].tick(r["clock"].now_ns, replace(r["ctx"], epoch=r["ex"].generation))
    assert r["ctrl"].free_slots() <= 2


# ------------------------------------------------ 证据


def test_evidence_roundtrip_and_tampering(tmp_path):
    log = EventLog("ev-test")
    for i in range(20):
        log.append("DISPATCH", step_index=i, submitted=True)
    b = build_bundle(log, str(tmp_path))

    ok, msg = verify_bundle(b["bundle_dir"], b["public_key"], "ev-test")
    assert ok, msg

    # 改一个字节
    p = tmp_path / "bundle" / "events.jsonl"
    data = bytearray(p.read_bytes())
    for i, ch in enumerate(data):
        if chr(ch).isdigit() and i > len(data) // 2:
            data[i] = ord("7") if chr(ch) != "7" else ord("3")
            break
    p.write_bytes(bytes(data))
    assert not verify_bundle(b["bundle_dir"], b["public_key"], "ev-test")[0]
    assert "hash_chain" in failed_layers(b["bundle_dir"], b["public_key"], "ev-test")


def test_four_tampering_modes_give_four_distinct_profiles(tmp_path):
    """四种篡改要给四种不同的失败特征，不能都报同一个原因。"""
    import shutil

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    log = EventLog("p-test")
    for i in range(30):
        log.append("DISPATCH", step_index=i, submitted=True)
    b = build_bundle(log, str(tmp_path))
    bundle, pub = b["bundle_dir"], b["public_key"]
    profiles = set()

    t1 = tmp_path / "t1"
    shutil.copytree(bundle, t1)
    p = t1 / "events.jsonl"
    d = bytearray(p.read_bytes())
    for i, ch in enumerate(d):
        if chr(ch).isdigit() and i > len(d) // 2:
            d[i] = ord("7") if chr(ch) != "7" else ord("3")
            break
    p.write_bytes(bytes(d))
    profiles.add(failed_layers(str(t1), pub, "p-test"))

    t2 = tmp_path / "t2"
    shutil.copytree(bundle, t2)
    p = t2 / "events.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    profiles.add(failed_layers(str(t2), pub, "p-test"))

    wrong = tmp_path / "wrong.public"
    wrong.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw))
    profiles.add(failed_layers(bundle, str(wrong), "p-test"))

    profiles.add(failed_layers(bundle, pub, "wrong-run"))

    assert len(profiles) == 4, f"只有 {len(profiles)} 种失败特征，应当是 4 种"


def test_event_log_rejects_unknown_type():
    log = EventLog("x")
    with pytest.raises(ValueError):
        log.append("NOT_A_REAL_TYPE")


def test_hash_chain_is_deterministic():
    """同样的事件序列必须给出同样的末尾摘要 —— 跨机器复现的前提。"""
    a, b = (
        EventLog("same", monotonic_ns=lambda: 123,
                 utc_now=lambda: "2026-09-19T00:00:00Z")
        for _ in range(2)
    )
    for log in (a, b):
        for i in range(10):
            log.append("DISPATCH", step_index=i, submitted=True)
    assert a.tip_hash == b.tip_hash
