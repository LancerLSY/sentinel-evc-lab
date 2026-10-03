"""Δ-Cert-arm: chain-aware certificate inheritance for joint-space plans.

Certificate of a plan = per segment, per collision capsule, a certified lower
bound on clearance to the obstacle set along the whole segment (continuous, not
sampled), plus the same for self-collision pairs.

Inheritance (Lemma A2): if child and parent segments correspond (same frame
grid, optional offset) and joint deviations at the two endpoints are
delta_k, delta_{k+1}, then for every capsule g

    lb_child[k, g] >= lb_parent[k, g] - sum_i rho[g, i] * max(|delta_k,i|, |delta_{k+1},i|)

where rho[g, i] bounds the distance from joint i's axis to capsule g and is
zero for joints that are not upstream of g (a wrist change never moves the
upper arm).  Self pairs use only the joints strictly between the two bodies.
Segments whose inherited bound is not > 0 are re-certified locally with the
batched continuous certifier.  Bit-identical segments are reused untouched.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field

import numpy as np

from ur5e_kin import ARM_NUMERIC_GUARD_M, BatchedCertifiedChecker, FKBudgetExceeded, densify, joint_ok


_ISSUER_KEY = secrets.token_bytes(32)


def scene_digest(obstacles, r_obs) -> str:
    a = np.ascontiguousarray(np.asarray(obstacles, dtype=np.float64))
    return hashlib.sha256(a.tobytes() + np.float64(r_obs).tobytes()).hexdigest()


def _put(h, label: str, data: bytes) -> None:
    name = label.encode("utf-8")
    h.update(len(name).to_bytes(4, "little")); h.update(name)
    h.update(len(data).to_bytes(8, "little")); h.update(data)


def _put_array(h, label: str, value, dtype) -> None:
    a = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    shape = np.asarray(a.shape, dtype="<i8")
    _put(h, label + ".shape", shape.tobytes())
    _put(h, label + ".data", a.tobytes())


def model_digest(model) -> str:
    """Digest the parsed geometry and every bound used by the certifier."""
    h = hashlib.sha256(b"sentinel-arm-model-v1\0")
    for i, body in enumerate(model.bodies):
        _put(h, f"body.{i}.name", body.name.encode("utf-8"))
        _put_array(h, f"body.{i}.identity", [body.parent, body.joint], "<i8")
        _put_array(h, f"body.{i}.pos", body.pos, "<f8")
        _put_array(h, f"body.{i}.rot", body.rot, "<f8")
        _put_array(h, f"body.{i}.axis", body.axis, "<f8")
    for i, capsule in enumerate(model.capsules):
        _put(h, f"capsule.{i}.kind", capsule.kind.encode("utf-8"))
        _put_array(h, f"capsule.{i}.body", [capsule.body], "<i8")
        _put_array(h, f"capsule.{i}.pos", capsule.pos, "<f8")
        _put_array(h, f"capsule.{i}.axis", capsule.axis_local, "<f8")
        _put_array(h, f"capsule.{i}.size", [capsule.half, capsule.radius], "<f8")
    _put_array(h, "joint.low", model.low, "<f8")
    _put_array(h, "joint.high", model.high, "<f8")
    _put_array(h, "home", model.home, "<f8")
    _put_array(h, "rho", model.rho, "<f8")
    _put_array(h, "self_pairs", model.pairs, "<i8")
    _put_array(h, "pair_rho", model.pair_rho, "<f8")
    return h.hexdigest()


def checker_profile_digest(max_depth=14, tight=np.inf, max_fk_evals=200_000) -> str:
    h = hashlib.sha256(b"sentinel-arm-checker-profile-v2\0")
    _put_array(h, "max_depth", [max_depth], "<i8")
    _put_array(h, "tight", [tight], "<f8")
    _put_array(h, "max_fk_evals", [max_fk_evals], "<i8")
    _put_array(h, "numeric_guard_m", [ARM_NUMERIC_GUARD_M], "<f8")
    _put(h, "include_self", b"true")
    return h.hexdigest()


def _frozen_f64(value) -> np.ndarray:
    """Own the values through immutable bytes so writeability cannot be re-enabled."""
    a = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return np.frombuffer(a.tobytes(), dtype="<f8").reshape(a.shape)


@dataclass(frozen=True)
class ArmCert:
    plan: np.ndarray       # (H+1, 6) exact knots the certificate covers
    scene: str
    lb_o: np.ndarray       # (H, G)
    lb_s: np.ndarray       # (H, P)
    depth: int = 0
    model: str | None = None
    profile: str | None = None
    content: str = field(default="", init=False)
    seal: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "plan", _frozen_f64(self.plan))
        object.__setattr__(self, "lb_o", _frozen_f64(self.lb_o))
        object.__setattr__(self, "lb_s", _frozen_f64(self.lb_s))
        object.__setattr__(self, "content", arm_cert_digest(self))


def arm_cert_digest(cert: ArmCert) -> str:
    h = hashlib.sha256(b"sentinel-arm-cert-v1\0")
    _put(h, "scene", cert.scene.encode("utf-8"))
    _put(h, "model", (cert.model or "").encode("ascii"))
    _put(h, "profile", (cert.profile or "").encode("ascii"))
    _put_array(h, "depth", [cert.depth], "<i8")
    _put_array(h, "plan", cert.plan, "<f8")
    _put_array(h, "lb_o", cert.lb_o, "<f8")
    _put_array(h, "lb_s", cert.lb_s, "<f8")
    return h.hexdigest()


def _issue_arm_cert(model, plan, scene, lb_o, lb_s, depth=0, max_depth=14, tight=np.inf,
                    max_fk_evals=200_000) -> ArmCert:
    cert = ArmCert(plan, scene, lb_o, lb_s, depth,
                   model_digest(model), checker_profile_digest(max_depth, tight, max_fk_evals))
    seal = hmac.new(_ISSUER_KEY, cert.content.encode("ascii"), hashlib.sha256).hexdigest()
    object.__setattr__(cert, "seal", seal)
    return cert


def _bound_to(model, cert: ArmCert, max_depth=14, tight=np.inf, max_fk_evals=200_000) -> bool:
    if cert is None or cert.model is None or cert.profile is None:
        return False
    try:
        h = len(cert.plan) - 1
        content = arm_cert_digest(cert)
        seal = hmac.new(_ISSUER_KEY, content.encode("ascii"), hashlib.sha256).hexdigest()
        return (
            cert.model == model_digest(model)
            and cert.profile == checker_profile_digest(max_depth, tight, max_fk_evals)
            and cert.content == content
            and hmac.compare_digest(cert.seal, seal)
            and cert.plan.ndim == 2 and cert.plan.shape[1] == 6
            and cert.lb_o.shape == (h, len(model.capsules))
            and cert.lb_s.shape == (h, model.pair_rho.shape[0])
        )
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass
class ArmVerdict:
    ok: bool
    status: str
    cert: ArmCert | None
    fk_evals: int
    reused: int = 0
    inherited: int = 0
    rechecked: int = 0


def certify_full(model, plan, obstacles, r_obs, max_depth=14, tight=np.inf,
                 max_fk_evals=200_000) -> ArmVerdict:
    chk = BatchedCertifiedChecker(model, obstacles, r_obs, max_depth=max_depth, tight=tight,
                                  max_fk_evals=max_fk_evals)
    r = chk.check_plan(plan)
    if not r["ok"]:
        return ArmVerdict(False, r["status"], None, chk.fk_evals, rechecked=len(plan) - 1)
    cert = _issue_arm_cert(model, plan, scene_digest(obstacles, r_obs),
                           r["seg_lb_obst"], r["seg_lb_self"], max_depth=max_depth, tight=tight,
                           max_fk_evals=max_fk_evals)
    return ArmVerdict(True, "free", cert, chk.fk_evals, rechecked=len(plan) - 1)


def inherit(model, parent_cert: ArmCert, child, obstacles, r_obs, offset=0, max_depth=14, tight=np.inf,
            max_fk_evals=200_000) -> ArmVerdict:
    child = np.asarray(child, dtype=float)
    if (parent_cert is None or parent_cert.scene != scene_digest(obstacles, r_obs)
            or not _bound_to(model, parent_cert, max_depth, tight, max_fk_evals)):
        return certify_full(model, child, obstacles, r_obs, max_depth, tight, max_fk_evals)
    if not joint_ok(model, child):                      # linear constraint: exact at knots
        return ArmVerdict(False, "joint_limit", None, 0)
    P = parent_cert.plan
    Hc = len(child) - 1
    Hp = len(P) - 1
    G, Pn = parent_cert.lb_o.shape[1], parent_cert.lb_s.shape[1]
    lb_o = np.empty((Hc, G))
    lb_s = np.empty((Hc, Pn))
    need = np.zeros(Hc, dtype=bool)
    reused = inherited = 0
    pk = np.arange(Hc) + offset
    has = pk + 1 <= Hp
    idx = np.where(has)[0]
    if len(idx):
        pa, pb = P[pk[idx]], P[pk[idx] + 1]
        ca, cb = child[idx], child[idx + 1]
        same = np.all(pa == ca, 1) & np.all(pb == cb, 1)
        e = np.maximum(np.abs(ca - pa), np.abs(cb - pb))           # (n, 6) per-joint deviation bound
        lo = parent_cert.lb_o[pk[idx]] - e @ model.rho.T - ARM_NUMERIC_GUARD_M
        ls = parent_cert.lb_s[pk[idx]] - e @ model.pair_rho.T - ARM_NUMERIC_GUARD_M
        ok = same | (np.all(lo > 0, 1) & (np.all(ls > 0, 1) if Pn else True))
        lb_o[idx] = np.where(same[:, None], parent_cert.lb_o[pk[idx]], lo)
        lb_s[idx] = np.where(same[:, None], parent_cert.lb_s[pk[idx]], ls)
        reused = int(same.sum())
        inherited = int((ok & ~same).sum())
        need[idx[~ok]] = True
    need[~has] = True
    fk = 0
    rechecked = int(need.sum())
    if rechecked:
        chk = BatchedCertifiedChecker(model, obstacles, r_obs, max_depth=max_depth, tight=tight,
                                      max_fk_evals=max_fk_evals)
        seg = np.where(need)[0]
        knots = np.unique(np.concatenate([seg, seg + 1]))
        try:
            o, s = chk._clear(child[knots])
        except FKBudgetExceeded:
            return ArmVerdict(False, "unknown", None, chk.fk_evals,
                              reused, inherited, rechecked)
        pos = {k: i for i, k in enumerate(knots)}
        ia = np.array([pos[k] for k in seg]); ib = np.array([pos[k + 1] for k in seg])
        st, slo, sls = chk.certify_segments(child[seg], child[seg + 1], o[ia], o[ib], s[ia], s[ib])
        fk = chk.fk_evals
        if np.any(st != 0):
            return ArmVerdict(False, "collision" if np.any(st == 1) else "unknown", None, fk,
                              reused, inherited, rechecked)
        lb_o[seg], lb_s[seg] = slo, sls
    cert = _issue_arm_cert(model, child, parent_cert.scene, lb_o, lb_s, parent_cert.depth + 1,
                           max_depth=max_depth, tight=tight, max_fk_evals=max_fk_evals)
    return ArmVerdict(True, "free", cert, fk, reused, inherited, rechecked)


# ============================================================ batched candidates
# A policy that samples N candidate chunks per cycle (best-of-N, flow-matching
# samples, MPC rollouts) needs N verdicts per cycle.  All candidates descend from
# the same committed plan, so they share one parent certificate.

def certify_full_batch(model, plans, obstacles, r_obs, max_depth=14, max_fk_evals=200_000,
                       return_meta=False):
    """Certify N plans in one batch. Returns (ok (N,), seg_lb_o (N,H,G), seg_lb_s (N,H,P), fk)."""
    plans = np.asarray(plans, dtype=float)
    N, H1, _ = plans.shape
    chk = BatchedCertifiedChecker(model, obstacles, r_obs, max_depth=max_depth,
                                  max_fk_evals=max_fk_evals)
    G, P = len(model.capsules), model.pair_rho.shape[0]
    try:
        o, s = chk._clear(plans.reshape(-1, 6))
    except FKBudgetExceeded:
        out = (np.zeros(N, dtype=bool), np.full((N, H1 - 1, G), np.inf),
               np.full((N, H1 - 1, P), np.inf), chk.fk_evals)
        meta = {"budget_exhausted": True, "unknown_candidates": N, "max_fk_evals": max_fk_evals}
        return out + (meta,) if return_meta else out
    G, P = o.shape[1], s.shape[1]
    o = o.reshape(N, H1, G); s = s.reshape(N, H1, P)
    st, lbo, lbs = chk.certify_segments(plans[:, :-1].reshape(-1, 6), plans[:, 1:].reshape(-1, 6),
                                        o[:, :-1].reshape(-1, G), o[:, 1:].reshape(-1, G),
                                        s[:, :-1].reshape(-1, P), s[:, 1:].reshape(-1, P))
    jok = np.array([joint_ok(model, p) for p in plans])
    ok = jok & np.all(st.reshape(N, H1 - 1) == 0, axis=1)
    out = (ok, lbo.reshape(N, H1 - 1, G), lbs.reshape(N, H1 - 1, P), chk.fk_evals)
    seg_status = st.reshape(N, H1 - 1)
    meta = {"budget_exhausted": chk.budget_exhausted,
            "unknown_candidates": int(np.any(seg_status == 2, axis=1).sum()) if chk.budget_exhausted else 0,
            "max_fk_evals": max_fk_evals}
    return out + (meta,) if return_meta else out


def inherit_batch(model, parent_cert: ArmCert, children, obstacles, r_obs, offset=0, max_depth=14,
                  max_fk_evals=200_000, return_meta=False):
    """Δ-Cert-arm for N children of one parent certificate, one batch.

    Returns (ok (N,), seg_lb_o (N,H,G), seg_lb_s (N,H,P), fk, (reused, inherited, rechecked)).
    Each child's verdict equals ``inherit(model, parent_cert, child, ...)``.
    """
    C = np.asarray(children, dtype=float)
    N, H1, _ = C.shape
    Hc = H1 - 1
    if (parent_cert is None or parent_cert.scene != scene_digest(obstacles, r_obs)
            or not _bound_to(model, parent_cert, max_depth, np.inf, max_fk_evals)):
        ok, lo, ls, fk, meta = certify_full_batch(model, C, obstacles, r_obs, max_depth,
                                                  max_fk_evals, return_meta=True)
        out = (ok, lo, ls, fk, (0, 0, N * Hc))
        return out + (meta,) if return_meta else out
    P = parent_cert.plan
    Hp = len(P) - 1
    G, Pn = parent_cert.lb_o.shape[1], parent_cert.lb_s.shape[1]
    lb_o = np.empty((N, Hc, G)); lb_s = np.empty((N, Hc, Pn))
    need = np.ones((N, Hc), dtype=bool)
    pk = np.arange(Hc) + offset
    idx = np.where(pk + 1 <= Hp)[0]
    reused = inherited = 0
    if len(idx):
        pa, pb = P[pk[idx]], P[pk[idx] + 1]                       # (n,6)
        ca, cb = C[:, idx], C[:, idx + 1]                          # (N,n,6)
        same = np.all(pa == ca, -1) & np.all(pb == cb, -1)         # (N,n)
        e = np.maximum(np.abs(ca - pa), np.abs(cb - pb))
        lo = parent_cert.lb_o[pk[idx]][None] - e @ model.rho.T - ARM_NUMERIC_GUARD_M
        ls = parent_cert.lb_s[pk[idx]][None] - e @ model.pair_rho.T - ARM_NUMERIC_GUARD_M
        ok = same | (np.all(lo > 0, -1) & (np.all(ls > 0, -1) if Pn else True))
        lb_o[:, idx] = np.where(same[..., None], parent_cert.lb_o[pk[idx]][None], lo)
        lb_s[:, idx] = np.where(same[..., None], parent_cert.lb_s[pk[idx]][None], ls)
        need[:, idx] = ~ok
        reused = int(same.sum()); inherited = int((ok & ~same).sum())
    fk = 0
    seg_ok = np.ones(N, dtype=bool)
    jok = np.array([joint_ok(model, c) for c in C])
    ni, si = np.nonzero(need)
    rechecked = len(ni)
    if rechecked:
        chk = BatchedCertifiedChecker(model, obstacles, r_obs, max_depth=max_depth,
                                      max_fk_evals=max_fk_evals)
        flat = C.reshape(-1, 6)
        ka, kb = ni * H1 + si, ni * H1 + si + 1
        uniq = np.unique(np.concatenate([ka, kb]))
        try:
            o, s = chk._clear(flat[uniq])
        except FKBudgetExceeded:
            unresolved = np.unique(ni)
            seg_ok[unresolved] = False
            out = (jok & seg_ok, lb_o, lb_s, chk.fk_evals, (reused, inherited, rechecked))
            meta = {"budget_exhausted": True, "unknown_candidates": int(len(unresolved)),
                    "max_fk_evals": max_fk_evals}
            return out + (meta,) if return_meta else out
        ia, ib = np.searchsorted(uniq, ka), np.searchsorted(uniq, kb)
        st, slo, sls = chk.certify_segments(flat[ka], flat[kb], o[ia], o[ib], s[ia], s[ib])
        fk = chk.fk_evals
        lb_o[ni, si], lb_s[ni, si] = slo, sls
        bad = np.unique(ni[st != 0])
        seg_ok[bad] = False
    out = (jok & seg_ok, lb_o, lb_s, fk, (reused, inherited, rechecked))
    meta = {"budget_exhausted": bool(rechecked and chk.budget_exhausted),
            "unknown_candidates": int(len(np.unique(ni[st == 2]))) if rechecked and chk.budget_exhausted else 0,
            "max_fk_evals": max_fk_evals}
    return out + (meta,) if return_meta else out


def discrete_batch(model, plans, obstacles, r_obs, step=0.01):
    """Common-practice sampled check of N plans in one clearance call."""
    Qs = [densify(p, step) for p in plans]
    counts = np.array([len(q) for q in Qs])
    o, s = model.clearances(np.concatenate(Qs, 0), obstacles, r_obs)
    bad = (o.min(1) < 0) | ((s.min(1) < 0) if s.shape[1] else False)
    owner = np.repeat(np.arange(len(plans)), counts)
    hit = np.zeros(len(plans), dtype=bool)
    hit[np.unique(owner[bad])] = True
    jok = np.array([joint_ok(model, p) for p in plans])
    return jok & ~hit, int(counts.sum())
