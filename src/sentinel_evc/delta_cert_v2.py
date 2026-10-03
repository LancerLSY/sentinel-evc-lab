"""Δ-Cert v2: segment-local, active-set certificate inheritance.

What changes relative to v1 (``delta_cert.py``)
-----------------------------------------------
1. **Segment-local fallback.**  v1 re-checks the *whole* plan as soon as one
   segment's bound is insufficient.  v2 re-checks only the failing segments.
2. **Active-set ledger with lazy exact refresh.**  Each segment keeps margins
   for the six box faces and the ``k`` nearest spheres, plus a sound lower bound
   for all other spheres.  On inheritance every entry first pays the Lipschitz
   loss ``L * e_k`` (O(1), no distance evaluation).  Only when that bound no
   longer certifies the segment are the active constraints re-evaluated exactly
   on the child segment (``k`` evaluations), so conservatism never accumulates
   on the constraints that actually bind; if the far remainder is exhausted the
   segment is re-checked exactly.
3. **Zero-deviation reuse.**  A segment whose two endpoints are bit-identical to
   its parent segment reuses the parent record with no arithmetic beyond the
   endpoint comparison (prefix-preserving transforms such as RTC are O(changed)).
4. **Offset correspondence.**  A child may be a shifted window of its parent
   (streaming execution drops executed steps and appends new ones).  Child
   segment k corresponds to parent segment k + offset; segments with no parent
   counterpart are checked exactly.
5. **Exact identity.**  R0 reuse and plan binding use ``Plan.exact_hash``
   (IEEE-754 bytes), not the 12-significant-digit canonical digest.

Soundness and decision comparison
--------------------------------
In real arithmetic, the segment bounds give the stated decision equivalence
for this sphere/box profile. The implementation uses float64, a 1e-9 m
inheritance guard, a bounded scene/plan scale, and a maximum inheritance depth.
Outside that profile it falls back to complete evaluation. A segment whose
bound is insufficient is evaluated with the full check's floating expressions.

The guard is an engineering margin, not a proved interval-arithmetic bound.
Randomised and adversarial comparisons support agreement on the recorded
inputs; they do not establish a theorem for all floating-point inputs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

from .contracts import Plan, Scene
from .scene_index import IndexedChecker, SegmentRecord, scene_scale

LIPSCHITZ = 1.0
PROOF_SCOPE_V2 = "numeric-sphere-box-L1-active-v2"
MAX_INHERIT_DEPTH_V2 = 64  # depth no longer degrades the binding constraints
INHERIT_EPS = 1e-9         # m; engineering guard for bounded float64 computations
MAX_COORD_M = 1e3          # outside this scale, use complete evaluation

# transform kinds whose semantics are fully captured by knot deviations
INHERITABLE_V2 = frozenset({"identity", "perturb", "rtc_suffix", "blend", "shift", "repair"})

_ids = itertools.count(1)


@dataclass(frozen=True)
class CertV2:
    cert_id: str
    plan_hash: str
    plan_exact: str
    scene_hash: str
    dt: float
    horizon: int
    segments: tuple  # tuple[SegmentRecord, ...]
    method: str  # FULL | INHERITED | MIXED
    parent_id: Optional[str] = None
    root_id: Optional[str] = None
    depth: int = 0
    proof_scope: str = PROOF_SCOPE_V2

    @property
    def margins(self) -> tuple:
        return tuple(s.lb for s in self.segments)


@dataclass
class VerdictV2:
    plan_hash: str
    verdict: str  # FULL | INHERITED | MIXED | REJECTED
    certificate: Optional[CertV2]
    margins: tuple
    first_violation_segment: Optional[int] = None
    reason_code: Optional[str] = None
    # cost accounting
    segments_reused: int = 0       # bit-identical, no work
    segments_inherited: int = 0    # bound + exact active set
    segments_rechecked: int = 0    # exact local fallback
    full_checks_used: int = 0      # whole-plan exact checks (fallback for deps failure)
    sphere_evaluations: int = 0

    @property
    def accepted(self) -> bool:
        return self.verdict != "REJECTED"


def _new_id() -> str:
    return f"cert2-{next(_ids):07d}"


def _checker_for(scene: Scene, checker: Optional[IndexedChecker]) -> IndexedChecker:
    if checker is None:
        return IndexedChecker(scene)
    if checker.scene.hash != scene.hash:
        raise ValueError("checker was built for another scene")
    return checker


def to_store_certificate(cert: "CertV2"):
    """Register-able v1 ``Certificate`` carrying the v2 margins and the exact byte binding.

    Authority consumes certificates through ``CertificateStore``; the adapter keeps the
    per-segment lower bounds, lineage and ``plan_exact`` so a lease can only be issued for
    the bytes that were verified.
    """
    from .contracts import Certificate
    return Certificate(cert.cert_id, cert.plan_hash, cert.scene_hash, cert.dt, cert.horizon,
                       cert.margins, "FULL" if cert.method == "FULL" else "INHERITED",
                       parent_id=cert.parent_id, root_id=cert.root_id, depth=cert.depth,
                       proof_scope=cert.proof_scope, plan_exact=cert.plan_exact)


def full_v2(plan: Plan, scene: Scene, checker: Optional[IndexedChecker] = None,
            reason: Optional[str] = None) -> VerdictV2:
    checker = _checker_for(scene, checker)
    before = checker.evaluations
    ok, recs, first = checker.full(plan)
    ev = checker.evaluations - before
    margins = tuple(r.lb for r in recs)
    if not ok:
        return VerdictV2(plan.hash, "REJECTED", None, margins, first, reason or "GEOMETRY_VIOLATION",
                         segments_rechecked=plan.horizon, full_checks_used=1, sphere_evaluations=ev)
    cert = CertV2(_new_id(), plan.hash, plan.exact_hash, scene.hash, plan.dt, plan.horizon,
                  recs, "FULL")
    return VerdictV2(plan.hash, "FULL", cert, margins, None, reason,
                     segments_rechecked=plan.horizon, full_checks_used=1, sphere_evaluations=ev)


def deps_v2(parent_cert: CertV2, parent: Plan, child: Plan, scene: Scene, kind: str,
            offset: int) -> Optional[str]:
    if kind not in INHERITABLE_V2:
        return "transform_not_registered"
    if parent_cert.plan_exact != parent.exact_hash:
        return "parent_plan_mismatch"
    if parent_cert.scene_hash != scene.hash:
        return "scene_changed"
    if parent_cert.proof_scope != PROOF_SCOPE_V2:
        return "proof_scope_changed"
    if parent_cert.dt != child.dt:
        return "dt_changed"
    if parent.descriptor != child.descriptor:
        return "descriptor_changed"
    if not (isinstance(offset, int) and 0 <= offset < parent.horizon):
        return "bad_offset"
    # gripper events must correspond under the offset (events beyond the parent are new)
    shifted = tuple((s - offset, e) for s, e in parent.gripper_events if s >= offset)
    child_overlap = tuple((s, e) for s, e in child.gripper_events if s + offset < parent.horizon)
    if shifted != child_overlap:
        return "gripper_events_changed"
    if parent_cert.depth + 1 > MAX_INHERIT_DEPTH_V2:
        return "max_depth_exceeded"
    scale = scene_scale(scene)
    if (scale > MAX_COORD_M
            or any(abs(c) > MAX_COORD_M for plan in (parent, child) for k in plan.knots for c in k)):
        return "coordinate_scale"          # outside the range the INHERIT_EPS rounding argument covers
    return None


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def validate_or_inherit_v2(child: Plan, scene: Scene, parent: Optional[Plan] = None,
                           parent_cert: Optional[CertV2] = None, kind: str = "unknown",
                           offset: int = 0, checker: Optional[IndexedChecker] = None) -> VerdictV2:
    checker = _checker_for(scene, checker)
    if parent is None or parent_cert is None:
        return full_v2(child, scene, checker)
    if not math.isfinite(checker.reach) or max(checker.far_cap, checker.reach) > MAX_COORD_M:
        return full_v2(child, scene, checker, "coordinate_scale")
    # R0: bit-identical plan
    if parent_cert.plan_exact == child.exact_hash and offset == 0:
        reason = deps_v2(parent_cert, parent, child, scene, "identity", 0)
        if reason is None:
            return VerdictV2(child.hash, "INHERITED", parent_cert, parent_cert.margins,
                             segments_reused=child.horizon)
        return full_v2(child, scene, checker, reason)
    reason = deps_v2(parent_cert, parent, child, scene, kind, offset)
    if reason is not None:
        return full_v2(child, scene, checker, reason)

    before = checker.evaluations
    pk, ck = parent.knots, child.knots
    segs = []
    reused = inherited = rechecked = 0
    first_violation = None
    for k in range(child.horizon):
        q0, q1 = ck[k], ck[k + 1]
        pk_idx = k + offset
        if pk_idx + 1 <= parent.horizon:
            p0, p1 = pk[pk_idx], pk[pk_idx + 1]
            prec: SegmentRecord = parent_cert.segments[pk_idx]
            if q0 == p0 and q1 == p1:                 # bit-identical segment
                segs.append(prec)
                reused += 1
                continue
            e = max(_dist(q0, p0), _dist(q1, p1))       # Lemma 1
            loss = LIPSCHITZ * e
            rest = prec.rest_lb - loss                  # Lemma 2 on the far remainder
            if rest >= INHERIT_EPS and prec.lb - loss >= INHERIT_EPS:
                # lazy path: the Lipschitz bound already certifies every constraint -> O(1), no evaluation;
                # active entries become lower bounds and are refreshed exactly only when they bind
                segs.append(SegmentRecord(tuple((cid, m - loss) for cid, m in prec.exact), rest))
                inherited += 1
                continue
            if rest >= INHERIT_EPS:
                exact = checker.margins_for(q0, q1, [cid for cid, _ in prec.exact])
                rec = SegmentRecord(exact, rest)
                if rec.lb >= 0.0:
                    segs.append(rec)
                    inherited += 1
                    continue
        # no counterpart, or the bound is insufficient: exact local check of this segment only
        rec = checker.check_segment(q0, q1)
        rechecked += 1
        segs.append(rec)
        if rec.lb < 0.0 and first_violation is None:
            first_violation = k
    ev = checker.evaluations - before
    margins = tuple(s.lb for s in segs)
    if first_violation is not None:
        return VerdictV2(child.hash, "REJECTED", None, margins, first_violation, "GEOMETRY_VIOLATION",
                         reused, inherited, rechecked, 0, ev)
    method = "INHERITED" if rechecked == 0 else "MIXED"
    cert = CertV2(_new_id(), child.hash, child.exact_hash, scene.hash, child.dt, child.horizon,
                  tuple(segs), method, parent_cert.cert_id,
                  parent_cert.root_id or parent_cert.cert_id, parent_cert.depth + 1)
    return VerdictV2(child.hash, method, cert, margins, None, None, reused, inherited, rechecked, 0, ev)


def slice_certificate_v2(cert: CertV2, plan: Plan, suffix: Plan, offset: int) -> CertV2:
    """Exact-suffix certificate (rule R3 for geometry): zero distance evaluations."""
    if cert.plan_exact != plan.exact_hash:
        raise ValueError("certificate does not cover plan")
    if suffix.knots != plan.knots[offset:] or suffix.dt != plan.dt or suffix.descriptor != plan.descriptor:
        raise ValueError("not an exact suffix")
    if suffix.gripper_events != tuple((s - offset, e) for s, e in plan.gripper_events if s >= offset):
        raise ValueError("gripper events do not correspond")
    return CertV2(_new_id(), suffix.hash, suffix.exact_hash, cert.scene_hash, suffix.dt, suffix.horizon,
                  cert.segments[offset:], "INHERITED", cert.cert_id, cert.root_id or cert.cert_id,
                  cert.depth + 1)
