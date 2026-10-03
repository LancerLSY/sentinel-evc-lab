"""Δ-Cert v2: randomised soundness and decision-equivalence tests against the v1 analytic checker."""

import math
import random

import pytest

from sentinel_evc.contracts import Plan, Scene, Sphere
from sentinel_evc.delta_cert_v2 import full_v2, slice_certificate_v2, validate_or_inherit_v2
from sentinel_evc.geometry import full_check
from sentinel_evc.scene_index import IndexedChecker

TOL = 1e-12


def _scene(rng, m):
    obs = []
    for _ in range(m):
        obs.append(Sphere((rng.uniform(0.0, 0.8), rng.uniform(-0.35, 0.35), rng.uniform(0.05, 0.55)),
                          rng.uniform(0.01, 0.06)))
    return Scene(f"s{rng.random()}", tuple(obs), (0.0, -0.4, 0.0), (0.8, 0.4, 0.6), 0.02, 0.005)


def _path(rng, h):
    a = (rng.uniform(0.1, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.4))
    b = (rng.uniform(0.6, 0.7), rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.4))
    amp = (rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15))
    knots = []
    for i in range(h + 1):
        t = i / h
        knots.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]) + amp[0] * math.sin(math.pi * t),
                      a[2] + t * (b[2] - a[2]) + amp[1] * math.sin(math.pi * t)))
    return Plan(f"p{rng.random()}", tuple(knots), 0.05)


def _transform(rng, parent):
    kind = rng.choice(["perturb", "rtc_suffix", "shift", "blend", "repair"])
    ks = [list(k) for k in parent.knots]
    h = parent.horizon
    offset = 0
    mag = rng.choice([0.0005, 0.003, 0.01, 0.03, 0.08])
    if kind == "perturb":
        for i in range(1, h):
            for j in range(3):
                ks[i][j] += rng.uniform(-mag, mag)
    elif kind == "rtc_suffix":
        d = rng.randint(1, h - 1)
        tgt = [rng.uniform(-mag, mag) for _ in range(3)]
        for i in range(d, h + 1):
            w = (i - d) / max(1, h - d)
            for j in range(3):
                ks[i][j] += w * tgt[j]
    elif kind == "shift":
        offset = rng.randint(1, max(1, h // 4))
        ks = ks[offset:]
        last = ks[-1]
        for _ in range(offset):
            last = [last[0] + rng.uniform(-0.02, 0.02), last[1] + rng.uniform(-0.02, 0.02),
                    last[2] + rng.uniform(-0.02, 0.02)]
            ks.append(last)
        for i in range(len(ks) // 2, len(ks)):
            for j in range(3):
                ks[i][j] += rng.uniform(-mag, mag)
    elif kind == "blend":
        other = [[v + rng.uniform(-mag * 3, mag * 3) for v in k] for k in ks]
        w = rng.uniform(0.0, 1.0)
        ks = [[w * a + (1 - w) * b for a, b in zip(x, y)] for x, y in zip(ks, other)]
    elif kind == "repair":
        i = rng.randint(1, h - 1)
        for j in range(3):
            ks[i][j] += rng.uniform(-mag * 4, mag * 4)
    child = Plan(f"c{rng.random()}", tuple(tuple(k) for k in ks), parent.dt)
    return child, kind, offset


@pytest.mark.parametrize("seed", range(6))
def test_full_v2_matches_v1_decision_and_bounds(seed):
    rng = random.Random(seed)
    for _ in range(60):
        sc = _scene(rng, rng.choice([0, 1, 5, 40, 200]))
        plan = _path(rng, rng.choice([4, 16, 48]))
        ok1, m1, f1 = full_check(plan, sc)
        v2 = full_v2(plan, sc, IndexedChecker(sc, active_k=rng.choice([0, 1, 3])))
        assert v2.accepted == ok1
        assert all(a <= b + TOL for a, b in zip(v2.margins, m1))  # sound lower bounds
        if not ok1:
            assert v2.first_violation_segment == f1


@pytest.mark.parametrize("seed", range(10))
def test_inheritance_is_sound_and_decision_equivalent(seed):
    rng = random.Random(1000 + seed)
    n_inherited = n_mixed = n_rejected = 0
    for _ in range(80):
        sc = _scene(rng, rng.choice([1, 8, 60, 250]))
        chk = IndexedChecker(sc, far_cap=rng.choice([0.03, 0.1, 0.3]), active_k=rng.choice([0, 2, 4]))
        parent = _path(rng, rng.choice([8, 24, 48]))
        root = full_v2(parent, sc, chk)
        if not root.accepted:
            continue
        cert, cur = root.certificate, parent
        for _step in range(rng.randint(1, 12)):          # chains of revisions
            child, kind, offset = _transform(rng, cur)
            v = validate_or_inherit_v2(child, sc, cur, cert, kind, offset, chk)
            ok1, m1, _ = full_check(child, sc)
            assert v.accepted == ok1, (kind, offset)
            assert all(a <= b + TOL for a, b in zip(v.margins, m1)), kind
            if not v.accepted:
                n_rejected += 1
                break
            n_inherited += v.verdict == "INHERITED"
            n_mixed += v.verdict == "MIXED"
            cert, cur = v.certificate, child
    assert n_inherited + n_mixed > 0


def test_canonical_collision_does_not_trigger_identity_reuse():
    rng = random.Random(7)
    sc = _scene(rng, 5)
    p = _path(rng, 8)
    q = Plan("q", tuple((math.nextafter(x, 1.0), y, z) for x, y, z in p.knots), p.dt)  # one ulp
    assert p.hash == q.hash               # 12-significant-digit canonical digest collides
    assert p.exact_hash != q.exact_hash   # byte-exact digest does not
    root = full_v2(p, sc)
    v = validate_or_inherit_v2(q, sc, p, root.certificate, "perturb", 0)
    assert v.segments_reused == 0          # not treated as identical


def test_bit_identical_prefix_is_free_and_suffix_slice_exact():
    rng = random.Random(9)
    sc = _scene(rng, 50)
    p = _path(rng, 40)
    chk = IndexedChecker(sc)
    root = full_v2(p, sc, chk)
    assert root.accepted
    ks = [list(k) for k in p.knots]
    for i in range(35, 41):
        ks[i][2] += 0.001
    q = Plan("q", tuple(tuple(k) for k in ks), p.dt)
    v = validate_or_inherit_v2(q, sc, p, root.certificate, "rtc_suffix", 0, chk)
    assert v.accepted and v.segments_reused == 34
    s = Plan("s", p.knots[10:], p.dt)
    c = slice_certificate_v2(root.certificate, p, s, 10)
    assert c.margins == root.certificate.margins[10:]
    with pytest.raises(ValueError):
        slice_certificate_v2(root.certificate, p, Plan("x", q.knots[10:], p.dt), 10)


def test_scene_change_forces_full_path():
    rng = random.Random(11)
    tried = 0
    while True:
        tried += 1
        assert tried < 200
        sc = _scene(rng, 10)
        sc2 = _scene(rng, 10)
        p = _path(rng, 12)
        root = full_v2(p, sc)
        if root.accepted and sc2.hash != sc.hash:
            break
    v = validate_or_inherit_v2(p, sc2, p, root.certificate, "identity", 0)
    assert v.full_checks_used == 1 and v.reason_code == "scene_changed"
    assert v.accepted == full_check(p, sc2)[0]


def test_margins_within_ulps_of_zero_decide_exactly_like_v1():
    """v2 exact paths use v1's float expression; inherited bounds must clear INHERIT_EPS."""
    from sentinel_evc.contracts import Scene, Sphere
    from sentinel_evc.geometry import full_check
    rng = random.Random(1)
    checked = 0
    for trial in range(1500):
        tool = rng.choice([0.02, 0.01, 0.03, 0.015]); res = rng.choice([0.005, 0.003, 0.007, 0.0035])
        r = rng.uniform(0.005, 0.05)
        d = r + tool + res
        d = d + rng.randint(-3, 3) * math.ulp(d)
        sc = Scene("s", (Sphere((0.4, 0.0, 0.3 - d), r),), (0.0, -0.4, 0.0), (0.8, 0.4, 0.6), tool, res)
        child = Plan("c", ((0.1, 0.0, 0.3), (0.7, 0.0, 0.3)), 0.05)
        ok1 = full_check(child, sc)[0]
        assert full_v2(child, sc).accepted == ok1
        for gap in (0.0, 1e-12, 1e-10, 5e-10):
            parent = Plan("p", ((0.1, 0.0, 0.3 + gap), (0.7, 0.0, 0.3 + gap)), 0.05)
            pv = full_v2(parent, sc)
            if pv.certificate is None:
                continue
            v = validate_or_inherit_v2(child, sc, parent, pv.certificate, "perturb", 0)
            assert v.accepted == ok1, (trial, gap, v.verdict, ok1)
            checked += 1
    assert checked > 3000


def test_checker_built_for_another_scene_is_refused():
    from sentinel_evc.contracts import Scene, Sphere
    from sentinel_evc.scene_index import IndexedChecker
    ws_lo, ws_hi = (0.0, -0.4, 0.0), (0.8, 0.4, 0.6)
    empty = Scene("cell", (Sphere((0.7, 0.3, 0.5), 0.01),), ws_lo, ws_hi, 0.02, 0.005)
    blocked = Scene("cell", (Sphere((0.4, 0.0, 0.3), 0.05),), ws_lo, ws_hi, 0.02, 0.005)
    plan = Plan("p", tuple((0.1 + 0.6 * i / 8, 0.0, 0.3) for i in range(9)), 0.05)
    with pytest.raises(ValueError):
        full_v2(plan, blocked, IndexedChecker(empty))
    parent = Plan("p0", tuple((0.1 + 0.6 * i / 8, 0.001, 0.3) for i in range(9)), 0.05)
    pc = full_v2(parent, empty).certificate
    with pytest.raises(ValueError):
        validate_or_inherit_v2(plan, blocked, parent, pc, "perturb", 0, IndexedChecker(empty))


def test_store_certificate_admits_only_the_verified_bytes():
    from sentinel_evc.authority import Authority
    from sentinel_evc.contracts import Context, ErrorCode, Rejection, Snapshot
    from sentinel_evc.delta_cert import CertificateStore
    from sentinel_evc.delta_cert_v2 import to_store_certificate
    from sentinel_evc.scenarios import make_parent_pair, make_scene
    scene = make_scene(0)
    p, _ = make_parent_pair(0)
    store = CertificateStore()
    cert = to_store_certificate(full_v2(p, scene).certificate)
    store.register(cert)
    auth = Authority(store)
    ctx = Context(scene_id=scene.scene_id, scene_hash=scene.hash)
    snap = Snapshot("o0", p.knots[0], 10**9)
    auth.prepare(p, cert, ctx, snap, 10**9)                      # the verified bytes: issued
    q = Plan(p.plan_id, (p.knots[0],) + tuple((math.nextafter(x, 2.0), y, z) for x, y, z in p.knots[1:]), p.dt)
    assert q.hash == p.hash and q.exact_hash != p.exact_hash
    with pytest.raises(Rejection) as e:
        auth.prepare(q, cert, ctx, snap, 10**9)
    assert e.value.code == ErrorCode.CERTIFICATE_MISS
