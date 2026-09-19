"""几何与 Δ-Cert 的正确性测试。"""

from sentinel_evc.contracts import Plan, Scene, Sphere, TransformRecord
from sentinel_evc.delta_cert import (
    LIPSCHITZ,
    deviation_bounds,
    establish_root,
    validate_or_inherit,
)
from sentinel_evc.geometry import cross_validate, full_check, segment_point_distance
from sentinel_evc.scenarios import make_parent_pair, make_scene, mix, perturb


def _scene():
    return Scene(
        scene_id="t",
        obstacles=(Sphere(center=(0.5, 0.0, 0.0), radius=0.1),),
        ws_lo=(-10.0, -10.0, -10.0),
        ws_hi=(10.0, 10.0, 10.0),
        tool_radius=0.0,
        tracking_reserve=0.0,
    )


def test_midpoint_penetration():
    """两端点都在障碍外，线段中部穿过球心 —— 必须判为违规。

    这是最典型的漏检。只查端点的实现会放行这条轨迹。
    """
    plan = Plan(
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        dt=0.05,
        gripper_events=(),
        controller_profile="test",
        task_phase="test",
    )
    ok, margins, first = full_check(plan, _scene())
    assert not ok, "线段穿过球心却被放行 —— 说明检查只看了端点"
    assert first == 0
    assert margins[0] < 0


def test_segment_distance_clamps_to_endpoints():
    """最近点参数必须夹在 [0,1]，否则会算到线段延长线上。"""
    d = segment_point_distance((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    assert abs(d - 4.0) < 1e-12


def test_two_independent_implementations_agree():
    """G1 门禁：解析实现与采样实现在构造集上零分歧。"""
    for seed in range(40):
        scene = make_scene(seed % 50)
        p1, p2 = make_parent_pair(seed)
        child, _ = mix(p1, p2)
        assert cross_validate(p1, scene), f"seed={seed} 父轨迹判定分歧"
        assert cross_validate(child, scene), f"seed={seed} 子轨迹判定分歧"


def test_mix_of_two_valid_trajectories_violates():
    """项目的核心主张：两条分别通过的轨迹，混合之后穿过障碍。"""
    scene = make_scene(0)
    p1, p2 = make_parent_pair(0)
    assert full_check(p1, scene)[0]
    assert full_check(p2, scene)[0]
    child, _ = mix(p1, p2)
    assert not full_check(child, scene)[0], "混合轨迹应当违规，否则场景构造有问题"


def test_same_side_perturbation_is_inherited():
    """必要的负对照：正常的同侧微调应该能被继承，不能一变就拒。"""
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    root = establish_root(p1, scene)
    child, record = perturb(p1, seed=1)
    v = validate_or_inherit(child, scene, p1, root.certificate, record)
    assert v.verdict == "INHERITED"
    assert v.full_checks_used == 0


def test_mix_cannot_be_inherited_and_is_rejected():
    scene = make_scene(0)
    p1, p2 = make_parent_pair(0)
    root = establish_root(p1, scene)
    child, record = mix(p1, p2)
    v = validate_or_inherit(child, scene, p1, root.certificate, record)
    assert v.verdict == "REJECTED"
    assert v.full_checks_used == 1, "拒绝之前必须真的做过完整检查"


def test_margin_deducts_from_remaining_not_original():
    """链式继承时余量必须从剩余量扣，不是每次回到最初值。

    这个错误会让第二次、第三次变换偷偷放行本该拒绝的轨迹。
    """
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    root = establish_root(p1, scene)

    c1, r1 = perturb(p1, magnitude=0.006, seed=1, plan_id="A")
    v1 = validate_or_inherit(c1, scene, p1, root.certificate, r1)
    assert v1.verdict == "INHERITED"

    c2, r2 = perturb(c1, magnitude=0.006, seed=2, plan_id="B")
    v2 = validate_or_inherit(c2, scene, c1, v1.certificate, r2)
    assert v2.verdict == "INHERITED"

    e1 = deviation_bounds(p1, c1)
    e2 = deviation_bounds(c1, c2)
    for k in range(p1.horizon):
        expected = root.certificate.margins[k] - LIPSCHITZ * e1[k] - LIPSCHITZ * e2[k]
        assert abs(v2.certificate.margins[k] - expected) < 1e-12, (
            f"第 {k} 段余量没有从剩余量累计扣减")
    assert v2.certificate.inherit_depth == 2


def test_inherited_margin_is_conservative_lower_bound():
    """继承给出的必须是保守下界：不高于真实余量。"""
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    root = establish_root(p1, scene)
    child, record = perturb(p1, seed=3)
    v = validate_or_inherit(child, scene, p1, root.certificate, record)
    _, true_margins, _ = full_check(child, scene)
    for got, truth in zip(v.certificate.margins, true_margins):
        assert got <= truth + 1e-12, "继承余量高于真实余量 —— 界不保守，是严重错误"


def test_unregistered_transform_forces_full_check():
    """未登记的变换类型不能走继承，必须完整重验。"""
    scene = make_scene(0)
    p1, _ = make_parent_pair(0)
    root = establish_root(p1, scene)
    child, _ = perturb(p1, seed=4)
    bad = TransformRecord("x", "unknown", p1.hash, child.hash)
    v = validate_or_inherit(child, scene, p1, root.certificate, bad)
    assert v.full_checks_used == 1
    assert v.path == "full"


def test_scene_change_invalidates_certificate():
    """场景变了，旧证书不能继承。"""
    p1, _ = make_parent_pair(0)
    root = establish_root(p1, make_scene(0))
    child, record = perturb(p1, seed=5)
    other = make_scene(7)
    v = validate_or_inherit(child, other, p1, root.certificate, record)
    assert v.full_checks_used == 1
    assert v.path == "full"
