"""Port of run_ur5e_guard.make_case using the kinematic model (probe = EEF collision geom centre)."""
from __future__ import annotations

import math

import numpy as np

SCENARIOS = ("clear", "late_suffix", "narrow_passage", "joint_limit", "tracking_disturbance", "environment_change")
OBSTACLE_RADIUS = 0.055
FAR = np.array([[2., 2., 2.], [2.2, 2., 2.]])


def interpolate(a, b, frames=41):
    x = np.linspace(0, 1, frames)[:, None]
    s = x * x * (3 - 2 * x)
    return a + s * (b - a)


def make_case(model, scenario, root, seed_base=710_000):
    rng = np.random.default_rng(seed_base + root)
    q0 = model.home.copy()
    goal = q0 + rng.uniform([-.55, -.35, -.45, -.35, -.3, -.4], [.55, .35, .45, .35, .3, .4])
    parent = interpolate(q0, goal)
    final = parent.copy()
    split = 20
    disturbance = 1.0
    parent_obs, final_obs = FAR.copy(), FAR.copy()
    if scenario in ("late_suffix", "environment_change"):
        delta = np.array([rng.choice([-.48, .48]), rng.choice([-.34, .34]), .24, 0, 0, 0])
        ramp = np.linspace(0, 1, len(final) - split)[:, None]
        final[split:] += ramp * delta
    elif scenario == "narrow_passage":
        bump = np.sin(np.linspace(0, math.pi, len(final) - split))[:, None]
        final[split:] += bump * np.array([[rng.choice([-.38, .38]), .22, -.18, 0, 0, 0]])
    elif scenario == "joint_limit":
        final[split:, 2] = np.linspace(final[split, 2], model.high[2] + .18, len(final) - split)
    elif scenario == "tracking_disturbance":
        final[split:] += np.linspace(0, 1, len(final) - split)[:, None] * np.array([[.75, -.55, .65, .6, -.5, .5]])
        disturbance = .16
    if scenario in ("late_suffix", "narrow_passage", "environment_change"):
        g = len(model.capsules) - 1
        pxyz = model.geom_center(parent, g)
        fxyz = model.geom_center(final, g)
        sep = np.linalg.norm(fxyz - pxyz, axis=1)
        idx = int(split + np.argmax(sep[split:]))
        cand = fxyz[idx]
        if scenario == "narrow_passage":
            tangent = fxyz[min(idx + 1, len(fxyz) - 1)] - fxyz[max(idx - 1, 0)]
            tangent /= np.linalg.norm(tangent) + 1e-12
            normal = np.cross(tangent, np.array([0., 0., 1.]))
            normal /= np.linalg.norm(normal) + 1e-12
            hw = .32 if root % 2 == 0 else .065
            final_obs = np.stack([cand + hw * normal, cand - hw * normal])
        else:
            final_obs[0] = cand
        if scenario != "environment_change":
            parent_obs = final_obs.copy()
    return {"root": root, "scenario": scenario, "parent": parent, "final": final, "split": split,
            "parent_obstacles": parent_obs, "final_obstacles": final_obs, "disturbance": disturbance,
            "transformed": scenario != "clear"}
