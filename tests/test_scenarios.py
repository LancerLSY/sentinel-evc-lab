"""批准的固定数值 profile 与有种子轨迹构造。"""

import math
import random

import pytest

from sentinel_evc.geometry import full_check
from sentinel_evc.scenarios import make_parent_pair, make_scene, mix, perturb


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_scene_geometry_is_fixed(seed):
    scene = make_scene(seed)
    assert scene.ws_lo == (0.0, -0.4, 0.0)
    assert scene.ws_hi == (0.8, 0.4, 0.6)
    assert len(scene.obstacles) == 1
    assert scene.obstacles[0].center == (0.4, 0.0, 0.3)
    assert scene.obstacles[0].radius == 0.05
    assert scene.tool_radius == 0.02
    assert scene.tracking_reserve == 0.005


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_parent_arcs_follow_seeded_profile(seed):
    p1, p2 = make_parent_pair(seed)
    amplitude = random.Random(seed).uniform(0.10, 0.14)
    assert p1 == make_parent_pair(seed)[0]
    assert p1 != make_parent_pair(seed + 1)[0]
    assert p1.dt == p2.dt == 0.05
    assert p1.horizon == p2.horizon == 16
    for i, (upper, lower) in enumerate(zip(p1.points, p2.points)):
        t = i / 16
        expected_y = amplitude * math.sin(math.pi * t)
        assert upper == pytest.approx((0.1 + 0.6 * t, expected_y, 0.3))
        assert lower == pytest.approx((0.1 + 0.6 * t, -expected_y, 0.3))


def test_near_moves_only_internal_yz_and_repeats_with_seed():
    parent, _ = make_parent_pair(1234)
    child, record = perturb(parent, seed=1234)
    assert child == perturb(parent, seed=1234)[0]
    assert child != perturb(parent, seed=1235)[0]
    assert child.points[0] == parent.points[0]
    assert child.points[-1] == parent.points[-1]
    assert dict(record.parameters)["magnitude"] == 0.003
    for source, target in zip(parent.points[1:-1], child.points[1:-1]):
        assert target[0] == source[0]
        assert abs(target[1] - source[1]) <= 0.003
        assert abs(target[2] - source[2]) <= 0.003


def test_constructed_mix_and_near_cases_have_expected_geometry():
    scene = make_scene()
    for seed in range(500):
        p1, p2 = make_parent_pair(seed)
        mixed, record = mix(p1, p2)
        near, _ = perturb(p1, seed=seed)
        assert dict(record.parameters)["weight"] == 0.5
        assert all(point[1] == 0.0 for point in mixed.points)
        assert full_check(p1, scene)[0]
        assert full_check(p2, scene)[0]
        assert not full_check(mixed, scene)[0]
        assert full_check(near, scene)[0]
