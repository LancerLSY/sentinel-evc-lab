"""Randomised soundness checks for the certified checker and Δ-Cert-arm (run directly)."""
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ur5e_kin import UR5eModel, BatchedCertifiedChecker, discrete_check
from scenarios import interpolate, OBSTACLE_RADIUS
from arm_delta import certify_full, inherit

# path to mujoco_menagerie/universal_robots_ur5e/ur5e.xml (Apache-2.0, not vendored here)
m = UR5eModel(Path(os.environ['UR5E_MJCF']))


def obstacles_near(rng, plan, n):
    eef = m.geom_center(plan, len(m.capsules) - 1)
    pts = eef[rng.integers(0, len(eef), n)]
    return pts + rng.normal(0, 0.08, pts.shape)


def dense_min(plan, obs, start=0):
    from ur5e_kin import densify
    Q = densify(plan, 0.0005)
    o, s = m.clearances(Q, obs, OBSTACLE_RADIUS)
    return o, s, Q


def check_lb_sound(plan, cert, obs):
    """every certified per-segment per-capsule bound <= true clearance at dense samples inside the segment"""
    for k in range(len(plan) - 1):
        lam = np.linspace(0, 1, 60)[:, None]
        Q = plan[k] + lam * (plan[k + 1] - plan[k])
        o, s = m.clearances(Q, obs, OBSTACLE_RADIUS)
        assert np.all(o.min(0) >= cert.lb_o[k] - 1e-9), (k, o.min(0) - cert.lb_o[k])
        if s.shape[1]:
            assert np.all(s.min(0) >= cert.lb_s[k] - 1e-9)


rng = np.random.default_rng(0)
n_acc = n_rej = n_inh = 0
for trial in range(120):
    goal = m.home + rng.uniform(-0.6, 0.6, 6)
    parent = interpolate(m.home, goal, 41)
    obs = obstacles_near(rng, parent, rng.integers(1, 5))
    v = certify_full(m, parent, obs, OBSTACLE_RADIUS)
    o, s, _ = dense_min(parent, obs)
    dense_free = bool(o.min() >= 0 and (s.size == 0 or s.min() >= 0))
    if v.ok:
        assert dense_free, 'certified-free plan has a dense-sampled contact'
        check_lb_sound(parent, v.cert, obs)
    else:
        continue
    cert, cur = v.cert, parent
    for step in range(rng.integers(1, 10)):
        kind = rng.choice(['perturb', 'suffix', 'shift'])
        child = cur.copy(); offset = 0
        mag = rng.choice([0.001, 0.01, 0.05, 0.15])
        if kind == 'perturb':
            child[1:] += rng.normal(0, mag, child[1:].shape)
        elif kind == 'suffix':
            d = rng.integers(1, 40)
            w = np.clip((np.arange(41) - d) / max(1, 40 - d), 0, 1)[:, None]
            child += w * rng.normal(0, mag, 6)
        else:
            offset = int(rng.integers(1, 5))
            child = np.concatenate([cur[offset:], np.repeat(cur[-1:], offset, 0) + rng.normal(0, 0.01, (offset, 6))])
        r = inherit(m, cert, child, obs, OBSTACLE_RADIUS, offset)
        o, s, _ = dense_min(child, obs)
        dense_free = bool(o.min() >= 0 and (s.size == 0 or s.min() >= 0)) and bool(np.all(child >= m.low) and np.all(child <= m.high))
        full = certify_full(m, child, obs, OBSTACLE_RADIUS)
        if r.ok:
            assert dense_free, ('inherited plan has contact', kind, mag)
            check_lb_sound(child, r.cert, obs)
            n_acc += 1; n_inh += r.inherited
            assert full.ok or full.status == 'unknown'
            cert, cur = r.cert, child
        else:
            n_rej += 1
            if r.status == 'collision':
                # a contact found on a re-certified segment is found by the from-scratch certifier too
                assert full.status == 'collision', (r.status, full.status)
            assert not full.ok or r.status == 'unknown'
            break
print('PASS accepted', n_acc, 'rejected', n_rej, 'inherited segments', n_inh)


def test_batch_inheritance_equals_single_child_inheritance():
    import numpy as np
    from arm_delta import certify_full, inherit, inherit_batch
    from run_arm_pilot import GOAL_LO, place_obstacles
    from scenarios import OBSTACLE_RADIUS, interpolate
    rng = np.random.default_rng(11)
    checked = 0
    for root in range(6):
        base = interpolate(m.home, m.home + rng.uniform(GOAL_LO, -GOAL_LO), 41)
        obs = place_obstacles(m, base, rng)
        cert = certify_full(m, base, obs, OBSTACLE_RADIUS).cert
        for offset in (0, 1, 3):
            prop = np.concatenate([base[offset:], np.repeat(base[-1:], offset, 0)])
            kids = prop[None] + np.cumsum(rng.normal(0, rng.choice([0.002, 0.02, 0.06]), (6, 41, 6)), 1) * 0.3
            kids[:, :2] = prop[None, :2]
            ok, lo, ls, fk, _ = inherit_batch(m, cert, kids, obs, OBSTACLE_RADIUS, offset)
            for i in range(len(kids)):
                v = inherit(m, cert, kids[i], obs, OBSTACLE_RADIUS, offset)
                assert v.ok == bool(ok[i])
                if v.ok:
                    assert np.allclose(v.cert.lb_o, lo[i]) and np.allclose(v.cert.lb_s, ls[i])
                checked += 1
    assert checked == 108


test_batch_inheritance_equals_single_child_inheritance()
print('PASS batch == single inheritance on 108 children')
