"""Exact segment-wise geometry with a sweep-axis obstacle index (v2).

The index never changes a verdict: it only skips spheres that are provably
farther than a query radius, and reports a sound lower bound for them.
Margins of the spheres it does evaluate use the same closed-form formula as
``geometry.full_check``.

Standard library only, to keep the core runtime dependency on ``cryptography``.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional

from .contracts import Plan, Scene
from .geometry import segment_point_distance, validate_geometry_profile

# Box faces are always evaluated exactly; they are cheap and often binding.
BOX_IDS = ("x-", "x+", "y-", "y+", "z-", "z+")
NUMERIC_GUARD_M = 1e-9
MAX_INDEX_SCALE_M = 1e3


def scene_scale(scene: Scene) -> float:
    cached = scene.__dict__.get("_numeric_scale_cache")
    if cached is None:
        values = (*scene.ws_lo, *scene.ws_hi, scene.tool_radius, scene.tracking_reserve,
                  *(v for obstacle in scene.obstacles for v in (*obstacle.center, obstacle.radius)))
        cached = max(abs(v) for v in values)
        object.__setattr__(scene, "_numeric_scale_cache", cached)
    return cached


def box_margins(p0, p1, scene: Scene) -> tuple:
    """Exact per-face margins of a segment (linear constraints: endpoint extrema)."""
    slack = scene.tool_radius + scene.tracking_reserve
    out = []
    for axis in range(3):
        lo_v = min(p0[axis], p1[axis])
        hi_v = max(p0[axis], p1[axis])
        out.append(lo_v - scene.ws_lo[axis] - slack)
        out.append(scene.ws_hi[axis] - hi_v - slack)
    return tuple(out)


def sphere_margin(p0, p1, obstacle, scene: Scene) -> float:
    # Same floating-point expression, in the same order, as geometry.full_check, so the
    # exact paths of v2 reproduce v1's margins bit-for-bit (a regrouped slack can flip the
    # sign of a margin within one ulp of zero).
    return (segment_point_distance(p0, p1, obstacle.center) - obstacle.radius
            - scene.tool_radius - scene.tracking_reserve)


@dataclass(frozen=True)
class SweepIndex:
    """Obstacles sorted along the axis with the largest spread of centres.

    A segment query returns every sphere whose centre lies inside the segment's
    axis-aligned bounding interval inflated by ``reach`` along the sweep axis;
    every other sphere's centre is farther than ``reach`` from the segment.
    """

    scene_hash: str
    axis: int
    keys: tuple
    order: tuple
    max_radius: float

    @classmethod
    def build(cls, scene: Scene) -> "SweepIndex":
        obs = scene.obstacles
        if obs:
            spreads = []
            for a in range(3):
                vals = [o.center[a] for o in obs]
                spreads.append(max(vals) - min(vals))
            axis = max(range(3), key=lambda a: spreads[a])
        else:
            axis = 0
        order = tuple(sorted(range(len(obs)), key=lambda j: obs[j].center[axis]))
        keys = tuple(obs[j].center[axis] for j in order)
        rmax = max((o.radius for o in obs), default=0.0)
        return cls(scene.hash, axis, keys, order, rmax)

    def candidates(self, p0, p1, reach: float) -> tuple:
        lo = min(p0[self.axis], p1[self.axis]) - reach
        hi = max(p0[self.axis], p1[self.axis]) + reach
        i = bisect.bisect_left(self.keys, lo)
        j = bisect.bisect_right(self.keys, hi)
        return self.order[i:j]


@dataclass(frozen=True)
class SegmentRecord:
    """Per-segment certificate entry.

    ``exact``  : ((constraint_id, margin), ...) exact margins of the active set
                 (all six box faces plus the nearest spheres).
    ``rest_lb``: sound lower bound on the margin of every *other* sphere.
    """

    exact: tuple
    rest_lb: float

    @property
    def lb(self) -> float:
        m = self.rest_lb
        for _, v in self.exact:
            if v < m:
                m = v
        return m


class IndexedChecker:
    """Exact verifier used by the v2 full path and by local fallbacks.

    ``far_cap`` is the margin guaranteed for spheres outside the query reach.
    ``active_k`` is how many nearest spheres are kept with exact margins.
    """

    def __init__(self, scene: Scene, far_cap: float = 0.10, active_k: int = 2,
                 index: Optional[SweepIndex] = None):
        if (isinstance(far_cap, bool) or not isinstance(far_cap, (int, float))
                or not math.isfinite(far_cap) or far_cap <= 0
                or isinstance(active_k, bool) or not isinstance(active_k, int) or active_k < 0):
            raise ValueError("far_cap must be > 0 and active_k >= 0")
        self.scene = scene
        self.slack = scene.tool_radius + scene.tracking_reserve
        self.index = index or SweepIndex.build(scene)
        if self.index.scene_hash != scene.hash:
            raise ValueError("index built for another scene")
        self.far_cap = far_cap
        self.active_k = active_k
        # centre farther than reach from the segment  =>  margin >= far_cap
        self.reach = math.nextafter(far_cap + self.index.max_radius + self.slack + NUMERIC_GUARD_M, math.inf)
        self._skip_far = (math.isfinite(self.reach)
                          and max(scene_scale(scene), far_cap, self.reach) <= MAX_INDEX_SCALE_M)
        self.evaluations = 0  # exact sphere-segment distance evaluations (cost counter)

    def check_segment(self, p0, p1) -> SegmentRecord:
        obs = self.scene.obstacles
        exact = list(zip(BOX_IDS, box_margins(p0, p1, self.scene)))
        bounded_segment = all(abs(v) <= MAX_INDEX_SCALE_M for p in (p0, p1) for v in p)
        cand = (self.index.candidates(p0, p1, self.reach) if self._skip_far and bounded_segment
                else tuple(range(len(obs))))
        sph = []
        for j in cand:
            sph.append((sphere_margin(p0, p1, obs[j], self.scene), j))
        self.evaluations += len(cand)
        sph.sort()
        keep = sph[: self.active_k]
        rest = sph[self.active_k:]
        rest_lb = rest[0][0] if rest else math.inf
        # spheres not returned by the index are farther than `reach`
        if len(cand) < len(obs):
            rest_lb = min(rest_lb, self.far_cap)
        exact.extend((j, m) for m, j in keep)
        return SegmentRecord(tuple(exact), rest_lb)

    def margins_for(self, p0, p1, ids) -> tuple:
        """Exact margins of an explicit active set on a (new) segment."""
        box = None
        out = []
        n_sph = 0
        for cid in ids:
            if isinstance(cid, str):
                if box is None:
                    box = dict(zip(BOX_IDS, box_margins(p0, p1, self.scene)))
                out.append((cid, box[cid]))
            else:
                out.append((cid, sphere_margin(p0, p1, self.scene.obstacles[cid], self.scene)))
                n_sph += 1
        self.evaluations += n_sph
        return tuple(out)

    def full(self, plan: Plan):
        """Exact full verification; returns (ok, records, first_violation)."""
        validate_geometry_profile(plan)
        records = []
        first = None
        for k in range(plan.horizon):
            rec = self.check_segment(plan.knots[k], plan.knots[k + 1])
            records.append(rec)
            if first is None and rec.lb < 0.0:
                first = k
        return first is None, tuple(records), first
