"""UR5e kinematic collision verifier built directly from the official MJCF.

No MuJoCo dependency: forward kinematics is reconstructed from the pinned
Menagerie ``ur5e.xml`` (body pos/quat, hinge axes, collision capsules), and
collision is evaluated with closed-form capsule-sphere and capsule-capsule
distances.  The EEF collision *cylinder* is enclosed by a capsule of the same
radius and half-length (conservative superset).

Three static checkers over a joint-space piecewise-linear plan:

* ``discrete``   -- sample every <= ``step`` rad (common practice, 0.01 rad in
                    the repository gate); can miss contact between samples.
* ``certified``  -- continuous certification with configuration-independent
                    joint-space Lipschitz motion bounds and adaptive bisection
                    (in the spirit of Schwarzer, Saha & Latombe); never misses a
                    contact of the capsule model.
* Δ-Cert-arm     -- see ``arm_delta.py``: per-segment, per-geometry certified
                    clearance ledger inherited through joint deviations.

Static geometry only: no dynamics, tracking or controller behaviour.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_NUMERIC_GUARD_M = 1e-9


class FKBudgetExceeded(RuntimeError):
    pass


def quat_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vec(s, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, dtype=float)


@dataclass
class Body:
    name: str
    parent: int
    pos: np.ndarray
    rot: np.ndarray
    joint: int = -1           # index into JOINTS, -1 if fixed
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Capsule:
    body: int
    pos: np.ndarray
    axis_local: np.ndarray    # unit z of the geom frame expressed in the body frame
    half: float
    radius: float
    kind: str


class UR5eModel:
    def __init__(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        self.xml_path = Path(xml_path)
        self.bodies: list[Body] = []
        self.capsules: list[Capsule] = []
        self.low = np.zeros(6)
        self.high = np.zeros(6)
        defaults = self._defaults(root)
        world = root.find("worldbody")
        for b in world.findall("body"):
            self._walk(b, -1, defaults, "ur5e")
        key = root.find("keyframe/key[@name='home']")
        self.home = _vec(key.get("qpos"), [0] * 6)
        nb = len(self.bodies)
        self.anc = [set() for _ in range(nb)]
        for i, b in enumerate(self.bodies):
            j, chain = i, set()
            while j >= 0:
                chain.add(j)
                j = self.bodies[j].parent
            self.anc[i] = chain
        self.joint_body = [next(i for i, b in enumerate(self.bodies) if b.joint == k) for k in range(6)]
        self._motion_bounds()
        self._self_pairs()

    # ---------------------------------------------------------------- parsing
    def _defaults(self, root):
        out = {}

        def walk(node, inherited):
            for d in node.findall("default"):
                cls = d.get("class")
                cur = {k: dict(v) for k, v in inherited.items()}
                for child in d:
                    if child.tag in ("joint", "geom", "general"):
                        cur.setdefault(child.tag, {}).update(child.attrib)
                out[cls] = cur
                walk(d, cur)
        walk(root.find("default"), {})
        return out

    def _walk(self, node, parent, defaults, cls):
        cls = node.get("childclass", cls)
        pos = _vec(node.get("pos"), [0, 0, 0])
        rot = quat_to_mat(_vec(node.get("quat"), [1, 0, 0, 0]))
        body = Body(node.get("name"), parent, pos, rot)
        if len(node.findall("joint")) > 1:
            raise ValueError(f"body {node.get('name')}: one hinge per body is assumed")
        jn = node.find("joint")
        if jn is not None:
            jcls = jn.get("class", cls)
            jd = dict(defaults.get(jcls, {}).get("joint", {}))
            jd.update(jn.attrib)
            # Lemma A1 and the FK below assume every hinge axis passes through its body origin
            if np.any(_vec(jd.get("pos"), [0, 0, 0]) != 0):
                raise ValueError(f"joint {jn.get('name')}: non-zero joint pos is not supported")
            if jd.get("type", "hinge") != "hinge":
                raise ValueError(f"joint {jn.get('name')}: only hinge joints are supported")
            body.joint = JOINTS.index(jn.get("name"))
            axis = _vec(jd.get("axis"), [0, 0, 1])
            body.axis = axis / np.linalg.norm(axis)
            lo, hi = (float(v) for v in jd["range"].split())
            self.low[body.joint], self.high[body.joint] = lo, hi
        idx = len(self.bodies)
        self.bodies.append(body)
        for g in node.findall("geom"):
            gcls = g.get("class", cls)
            gd = dict(defaults.get(gcls, {}).get("geom", {}))
            gd.update(g.attrib)
            if gd.get("group") != "3":
                continue
            kind = gd.get("type", "sphere")
            if kind not in ("capsule", "cylinder"):
                raise ValueError(f"unsupported collision geom {kind}")
            r, h = (float(v) for v in gd["size"].split())
            grot = quat_to_mat(_vec(gd.get("quat"), [1, 0, 0, 0]))
            self.capsules.append(Capsule(idx, _vec(gd.get("pos"), [0, 0, 0]), grot[:, 2].copy(), h, r, kind))
        for child in node.findall("body"):
            self._walk(child, idx, defaults, cls)

    # ------------------------------------------------- configuration-free bounds
    def _motion_bounds(self):
        """rho[g, i]: upper bound on the distance from joint i's axis point to any point of capsule g's axis."""
        G = len(self.capsules)
        self.rho = np.zeros((G, 6))
        for g, c in enumerate(self.capsules):
            for i in range(6):
                jb = self.joint_body[i]
                if jb not in self.anc[c.body]:
                    continue
                # chain of translations from joint body origin to the capsule body origin
                total, b = 0.0, c.body
                while b != jb:
                    total += float(np.linalg.norm(self.bodies[b].pos))
                    b = self.bodies[b].parent
                total += float(np.linalg.norm(c.pos)) + c.half
                self.rho[g, i] = np.nextafter(total, np.inf)

    def _self_pairs(self):
        """Capsule pairs MuJoCo would test: different bodies that are not parent/child."""
        pairs, rel = [], []
        for a in range(len(self.capsules)):
            for b in range(a + 1, len(self.capsules)):
                ba, bb = self.capsules[a].body, self.capsules[b].body
                if ba == bb or self.bodies[ba].parent == bb or self.bodies[bb].parent == ba:
                    continue
                pairs.append((a, b))
                # only joints strictly between the two bodies change their relative pose
                lo_b, hi_b, hi_g, lo_g = (ba, bb, b, a) if ba in self.anc[bb] else (bb, ba, a, b)
                w = np.zeros(6)
                for i in range(6):
                    jb = self.joint_body[i]
                    if jb in self.anc[hi_b] and jb not in self.anc[lo_b]:
                        w[i] = self.rho[hi_g, i]
                rel.append(w)
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        self.pair_rho = np.array(rel).reshape(-1, 6)

    # ------------------------------------------------------------- kinematics
    def capsule_segments(self, Q: np.ndarray):
        """Q: (B,6) -> endpoints A,B of each capsule axis, shape (B,G,3) each."""
        Q = np.atleast_2d(Q)
        B = Q.shape[0]
        nb = len(self.bodies)
        R = np.zeros((nb, B, 3, 3))
        T = np.zeros((nb, B, 3))
        for i, body in enumerate(self.bodies):
            if body.parent < 0:
                Rp, Tp = np.broadcast_to(np.eye(3), (B, 3, 3)), np.zeros((B, 3))
            else:
                Rp, Tp = R[body.parent], T[body.parent]
            Ri = Rp @ body.rot
            Ti = Tp + Rp @ body.pos
            if body.joint >= 0:
                th = Q[:, body.joint]
                a = body.axis
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
                Rj = np.eye(3) + s * K + (1 - c) * (K @ K)
                Ri = Ri @ Rj
            R[i], T[i] = Ri, Ti
        G = len(self.capsules)
        A = np.zeros((B, G, 3))
        Bp = np.zeros((B, G, 3))
        for g, c in enumerate(self.capsules):
            ctr = T[c.body] + R[c.body] @ c.pos
            ax = R[c.body] @ c.axis_local
            A[:, g] = ctr - c.half * ax
            Bp[:, g] = ctr + c.half * ax
        return A, Bp

    def geom_center(self, Q, g):
        A, Bp = self.capsule_segments(Q)
        return 0.5 * (A[:, g] + Bp[:, g])

    # -------------------------------------------------------------- distances
    def clearances(self, Q, obstacles, obstacle_radius):
        """Per-configuration clearances.

        Returns (obst (B,G), selfc (B,P)): signed distances (negative = contact).
        """
        A, Bp = self.capsule_segments(Q)
        radii = np.array([c.radius for c in self.capsules])
        obst = _seg_point_dist(A, Bp, obstacles) - radii[None, :, None] - obstacle_radius
        obst = obst.min(axis=2) if obstacles.shape[0] else np.full(A.shape[:2], np.inf)
        if len(self.pairs):
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            d = _seg_seg_dist(A[:, a], Bp[:, a], A[:, b], Bp[:, b])
            selfc = d - radii[a][None] - radii[b][None]
        else:
            selfc = np.zeros((A.shape[0], 0))
        return obst, selfc


def _seg_point_dist(A, B, P):
    """A,B: (N,G,3); P: (S,3) -> (N,G,S)."""
    d = B - A
    dd = np.einsum("ngk,ngk->ng", d, d)[..., None]
    ap = P[None, None, :, :] - A[:, :, None, :]
    t = np.einsum("ngsk,ngk->ngs", ap, d) / np.where(dd > 0, dd, 1.0)
    t = np.clip(t, 0.0, 1.0)
    closest = A[:, :, None, :] + t[..., None] * d[:, :, None, :]
    return np.linalg.norm(P[None, None] - closest, axis=-1)


def _seg_seg_dist(P0, P1, Q0, Q1):
    """Closest distance between segments, vectorised over leading dims."""
    d1, d2, r = P1 - P0, Q1 - Q0, P0 - Q0
    a = np.sum(d1 * d1, -1); e = np.sum(d2 * d2, -1); f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1); b = np.sum(d1 * d2, -1)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1), 0, 1), 0.0)
    t = (b * s + f) / np.where(e > 1e-12, e, 1)
    t_cl = np.clip(t, 0, 1)
    s = np.where(t < 0, np.clip(-c / np.where(a > 1e-12, a, 1), 0, 1), np.where(t > 1, np.clip((b - c) / np.where(a > 1e-12, a, 1), 0, 1), s))
    t = t_cl
    c1 = P0 + s[..., None] * d1
    c2 = Q0 + t[..., None] * d2
    return np.linalg.norm(c1 - c2, axis=-1)


# ==================================================================== checkers

def densify(plan, step, start=0):
    chunks = [plan[start:start + 1]]
    for i in range(max(1, start + 1), len(plan)):
        n = max(1, int(math.ceil(float(np.max(np.abs(plan[i] - plan[i - 1]))) / step)))
        chunks.append(np.linspace(plan[i - 1], plan[i], n + 1)[1:])
    return np.concatenate(chunks, 0)


def joint_ok(model, plan):
    return bool(np.all(plan >= model.low - 1e-9) and np.all(plan <= model.high + 1e-9))


def discrete_check(model, plan, obstacles, r_obs, step=0.01, start=0, include_self=True):
    """Common-practice sampled check. Returns dict(ok, reason, fk_evals)."""
    if not joint_ok(model, plan):
        return {"ok": False, "reason": "joint_limit", "fk_evals": 0}
    Q = densify(plan, step, start)
    o, s = model.clearances(Q, obstacles, r_obs)
    bad = (o.min(1) < 0) | ((s.min(1) < 0) if (include_self and s.shape[1]) else False)
    return {"ok": not bool(np.any(bad)), "reason": "collision" if np.any(bad) else "ok", "fk_evals": int(len(Q))}


class CertifiedChecker:
    """Continuous certification of joint-space linear segments.

    For a joint-space segment [qa, qb], capsule g's axis points move by at most
    M_g = sum_i rho[g,i] |qb_i - qa_i| (Lemma A1), so the clearance of g along the
    segment is >= (d_g(qa) + d_g(qb) - M_g) / 2.  Self pairs use relative bounds.
    Segments that cannot be certified are bisected up to ``max_depth``; a
    sampled configuration with negative clearance is a definite collision.
    """

    def __init__(self, model: UR5eModel, obstacles, r_obs, max_depth=14, include_self=True, tight=np.inf,
                 max_fk_evals=200_000):
        self.m, self.obs, self.r = model, np.asarray(obstacles, float).reshape(-1, 3), float(r_obs)
        self.max_depth, self.include_self = max_depth, include_self
        # certificate quality: a leaf is accepted only if its motion bound is <= tight (metres);
        # tight=inf gives the cheapest decision-only certification.
        self.tight = float(tight)
        self.max_fk_evals = int(max_fk_evals)
        self.fk_evals = 0
        self.budget_exhausted = False

    def _clear(self, Q):
        if self.fk_evals + len(Q) > self.max_fk_evals:
            self.budget_exhausted = True
            raise FKBudgetExceeded(f"FK budget {self.max_fk_evals} exhausted")
        self.fk_evals += len(Q)
        o, s = self.m.clearances(Q, self.obs, self.r)
        if not self.include_self:
            s = s[:, :0]
        return o, s

    def certify_segment(self, qa, qb, ca=None, cb=None):
        """Returns (status, lb_obst (G,), lb_self (P,)) where status in {free, collision, unknown}."""
        G = self.m.rho.shape[0]
        lb_o = np.full(G, np.inf)
        lb_s = np.full(self.m.pair_rho.shape[0] if self.include_self else 0, np.inf)
        try:
            if ca is None:
                ca = self._clear(qa[None])
            if cb is None:
                cb = self._clear(qb[None])
        except FKBudgetExceeded:
            return "unknown", lb_o, lb_s
        stack = [(qa, qb, ca, cb, 0)]
        while stack:
            a, b, (oa, sa), (ob, sb), depth = stack.pop()
            if np.any(oa < 0) or np.any(ob < 0) or (sa.size and (np.any(sa < 0) or np.any(sb < 0))):
                return "collision", lb_o, lb_s
            dq = np.abs(b - a)
            Mo = self.m.rho @ dq
            leaf_o = 0.5 * (oa[0] + ob[0] - Mo) - ARM_NUMERIC_GUARD_M
            ok = np.all(leaf_o > 0)
            if self.include_self and sa.size:
                Ms = self.m.pair_rho @ dq
                leaf_s = 0.5 * (sa[0] + sb[0] - Ms) - ARM_NUMERIC_GUARD_M
                ok = ok and np.all(leaf_s > 0)
            if ok:
                lb_o = np.minimum(lb_o, leaf_o)
                if self.include_self and sa.size:
                    lb_s = np.minimum(lb_s, leaf_s)
                continue
            if depth >= self.max_depth:
                return "unknown", lb_o, lb_s
            mid = 0.5 * (a + b)
            try:
                cm = self._clear(mid[None])
            except FKBudgetExceeded:
                return "unknown", lb_o, lb_s
            stack.append((mid, b, cm, (ob, sb), depth + 1))
            stack.append((a, mid, (oa, sa), cm, depth + 1))
        return "free", lb_o, lb_s

    def check_plan(self, plan):
        """Full certification. Returns dict(ok, status, seg_lb_obst (H,G), seg_lb_self (H,P), fk_evals)."""
        if not joint_ok(self.m, plan):
            return {"ok": False, "status": "joint_limit", "fk_evals": 0}
        start = self.fk_evals
        try:
            knots = self._clear(plan)
        except FKBudgetExceeded:
            return {"ok": False, "status": "unknown", "fk_evals": self.fk_evals - start}
        lbo, lbs = [], []
        for k in range(len(plan) - 1):
            ca = (knots[0][k:k + 1], knots[1][k:k + 1])
            cb = (knots[0][k + 1:k + 2], knots[1][k + 1:k + 2])
            st, lo, ls = self.certify_segment(plan[k], plan[k + 1], ca, cb)
            if st != "free":
                return {"ok": False, "status": st, "segment": k, "fk_evals": self.fk_evals - start}
            lbo.append(lo); lbs.append(ls)
        return {"ok": True, "status": "free", "seg_lb_obst": np.array(lbo), "seg_lb_self": np.array(lbs),
                "fk_evals": self.fk_evals - start}


class BatchedCertifiedChecker(CertifiedChecker):
    """Same certificate as CertifiedChecker, evaluated breadth-first in numpy batches.

    All knots are evaluated in one batch; every segment's first-level bound is
    computed at once; only uncertified intervals are bisected, one batched FK
    call per bisection level.  Per-segment lower bounds are the minimum over
    the certified leaves of that segment.
    """

    tight_depth = 8   # extra refinement for certificate quality never goes deeper than this

    def _leaf(self, oa, ob, sa, sb, dq, depth=0):
        Mo = dq @ self.m.rho.T
        lo = 0.5 * (oa + ob - Mo) - ARM_NUMERIC_GUARD_M
        ok = np.all(lo > 0, axis=1)
        if np.isfinite(self.tight) and depth < self.tight_depth:
            # certified leaves are refined further only to tighten the stored bound
            ok &= Mo.max(axis=1) <= self.tight
        if self.include_self and sa.shape[1]:
            ls = 0.5 * (sa + sb - dq @ self.m.pair_rho.T) - ARM_NUMERIC_GUARD_M
            ok &= np.all(ls > 0, axis=1)
        else:
            ls = np.zeros((len(dq), 0))
        return lo, ls, ok

    def certify_segments(self, qa, qb, oa, ob, sa, sb):
        """Vectorised certification of segments [qa_n, qb_n] with endpoint clearances.

        Returns (status array: 0 free / 1 collision / 2 unknown, lb_o (N,G), lb_s (N,P)).
        """
        N = len(qa)
        G = self.m.rho.shape[0]
        P = sa.shape[1]
        status = np.zeros(N, dtype=int)
        lb_o = np.full((N, G), np.inf)
        lb_s = np.full((N, P), np.inf)
        bad = np.any(oa < 0, 1) | np.any(ob < 0, 1)
        if P:
            bad |= np.any(sa < 0, 1) | np.any(sb < 0, 1)
        status[bad] = 1
        owner = np.arange(N)
        A, B, OA, OB, SA, SB = qa, qb, oa, ob, sa, sb
        live = ~bad
        A, B, OA, OB, SA, SB, owner = A[live], B[live], OA[live], OB[live], SA[live], SB[live], owner[live]
        depth = 0
        while len(owner):
            lo, ls, ok = self._leaf(OA, OB, SA, SB, np.abs(B - A), depth)
            if ok.any():
                np.minimum.at(lb_o, owner[ok], lo[ok])
                if P:
                    np.minimum.at(lb_s, owner[ok], ls[ok])
            rest = ~ok & (status[owner] == 0)
            if not rest.any():
                break
            if depth >= self.max_depth:
                status[np.unique(owner[rest])] = np.where(status[np.unique(owner[rest])] == 0, 2, status[np.unique(owner[rest])])
                break
            A, B, OA, OB, SA, SB, owner = A[rest], B[rest], OA[rest], OB[rest], SA[rest], SB[rest], owner[rest]
            M = 0.5 * (A + B)
            try:
                OM, SM = self._clear(M)
            except FKBudgetExceeded:
                unresolved = np.unique(owner)
                status[unresolved] = np.where(status[unresolved] == 0, 2, status[unresolved])
                break
            if not self.include_self:
                SM = SM[:, :0]
            hit = np.any(OM < 0, 1) | (np.any(SM < 0, 1) if P else False)
            if hit.any():
                status[np.unique(owner[hit])] = 1
            keep = status[owner] == 0
            A, B, OA, OB, SA, SB, OM, SM, M, owner = (x[keep] for x in (A, B, OA, OB, SA, SB, OM, SM, M, owner))
            A, B = np.concatenate([A, M]), np.concatenate([M, B])
            OA, OB = np.concatenate([OA, OM]), np.concatenate([OM, OB])
            SA, SB = np.concatenate([SA, SM]), np.concatenate([SM, SB])
            owner = np.concatenate([owner, owner])
            depth += 1
        return status, lb_o, lb_s

    def check_plan(self, plan):
        if not joint_ok(self.m, plan):
            return {"ok": False, "status": "joint_limit", "fk_evals": 0}
        start = self.fk_evals
        try:
            o, s = self._clear(plan)
        except FKBudgetExceeded:
            h = len(plan) - 1
            return {"ok": False, "status": "unknown", "seg_status": np.full(h, 2, dtype=int),
                    "seg_lb_obst": np.full((h, self.m.rho.shape[0]), np.inf),
                    "seg_lb_self": np.full((h, self.m.pair_rho.shape[0] if self.include_self else 0), np.inf),
                    "fk_evals": self.fk_evals - start}
        if not self.include_self:
            s = s[:, :0]
        st, lbo, lbs = self.certify_segments(plan[:-1], plan[1:], o[:-1], o[1:], s[:-1], s[1:])
        ok = bool(np.all(st == 0))
        out = {"ok": ok, "status": "free" if ok else ("collision" if np.any(st == 1) else "unknown"),
               "seg_status": st, "seg_lb_obst": lbo, "seg_lb_self": lbs, "fk_evals": self.fk_evals - start,
               "knot_obst": o, "knot_self": s}
        return out
