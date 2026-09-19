"""G1：固定构造集上的独立密集采样对照，不代表任意连续轨迹证明。"""

import numpy as np
import pytest

from sentinel_evc import geometry
from sentinel_evc.scenarios import make_parent_pair, make_scene, mix, perturb


def _sampled_allowed(points):
    knots = np.asarray(points, dtype=np.float64)
    t = np.linspace(0.0, 1.0, 1001)[None, :, None]
    samples = (1.0 - t) * knots[:-1, None, :] + t * knots[1:, None, :]

    # 直接逐点计算固定 profile，独立于生产侧解析线段最近点和盒极值公式。
    reserve = 0.02 + 0.005
    sphere_margin = np.linalg.norm(samples - (0.4, 0.0, 0.3), axis=2) - 0.05 - reserve
    lower_margin = samples - (0.0, -0.4, 0.0) - reserve
    upper_margin = (0.8, 0.4, 0.6) - samples - reserve
    return bool(
        np.all(sphere_margin > 0.0)
        and np.all(lower_margin > 0.0)
        and np.all(upper_margin > 0.0)
    )


@pytest.fixture(scope="module")
def compared_cases():
    results = []
    scene = make_scene(1234)
    for index in range(1000):
        seed = 1234 + index
        upper, lower = make_parent_pair(seed)
        if index < 500:
            kind = "mix"
            child, _ = mix(upper, lower, weight=0.5)
        else:
            kind = "near"
            child, _ = perturb(upper, magnitude=0.003, seed=seed)
        sampled = _sampled_allowed(child.points)
        analytic = geometry.full_check(child, scene)[0]
        results.append((seed, kind, sampled, analytic))
    return results


def test_all_1000_cases_match_independent_1001_point_sampling(compared_cases):
    disagreements = [
        (seed, kind) for seed, kind, sampled, analytic in compared_cases
        if sampled != analytic
    ]
    assert disagreements == []


def test_all_500_mix_cases_violate_in_independent_sampling(compared_cases):
    allowed = [sampled for _, kind, sampled, _ in compared_cases if kind == "mix"]
    assert allowed == [False] * 500


def test_all_500_near_cases_pass_in_independent_sampling(compared_cases):
    allowed = [sampled for _, kind, sampled, _ in compared_cases if kind == "near"]
    assert allowed == [True] * 500
