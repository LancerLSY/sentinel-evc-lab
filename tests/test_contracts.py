"""冻结合同的字节向量与输入别名回归。"""

import dataclasses
import math
import os
import subprocess
import sys

import pytest

from sentinel_evc.contracts import (
    MAX_INHERIT_DEPTH, Certificate, Lease, LeaseContext, Plan, Scene, Snapshot,
    Sphere, TransformRecord, canonical_json, sha256_hex, strict_json_loads,
)


def make_plan(**changes):
    values = dict(points=((0.1, 0.0, 0.3), (0.2, 0.1, 0.3)), dt=0.05,
                  gripper_events=(), controller_profile="numeric", task_phase="transfer")
    return Plan(**(values | changes))


def make_context():
    return LeaseContext("robot", 1, 0, "scene", 0, "sha256:" + "0" * 64)


def test_canonical_fixed_bytes_and_digest():
    assert canonical_json({"b": 1.0, "a": [True, 1, None, -0.0]}) == (
        b'{"a":[true,1,null,0.0000000000000000e+00],"b":1.0000000000000000e+00}'
    )
    assert sha256_hex({}) == "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert sha256_hex("abc") == "sha256:6cc43f858fbb763301637b5af970e2a46b46f461f27e5a0f41e009c59b827b25"


@pytest.mark.parametrize(("value", "expected"), [
    (0.0, b"0.0000000000000000e+00"), (-0.0, b"0.0000000000000000e+00"),
    (1.0, b"1.0000000000000000e+00"), (0.1, b"1.0000000000000001e-01"),
    (sys.float_info.max, b"1.7976931348623157e+308"),
    (math.ulp(0.0), b"4.9406564584124654e-324"),
    (1.0000000000000002, b"1.0000000000000002e+00"),
    (2.0 ** -17, b"7.6293945312500000e-06"),
    (0.0000102519989013671875, b"1.0251998901367188e-05"),
    (0.0000107288360595703125, b"1.0728836059570312e-05"),
])
def test_float_vectors(value, expected):
    assert canonical_json(value) == expected


def test_unicode_sorting_escaping_and_no_normalization():
    assert canonical_json({"😀": "中", "\ue000": "e\u0301", "a": '"\\\n'}) == (
        '{"a":"\\\"\\\\\\n","\ue000":"e\u0301","😀":"中"}'.encode("utf-8")
    )
    assert canonical_json("é") != canonical_json("e\u0301")
    assert canonical_json([1, 1.0, True, False]) == b"[1,1.0000000000000000e+00,true,false]"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                  "\ud800", {"\udfff": 0}, {1: "x"}, {True: "x"}])
def test_canonical_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        canonical_json(value)


@pytest.mark.parametrize("source", ['{"x":1,"x":2}', '{"x":{"a":0,"a":1}}',
                                   'NaN', 'Infinity', '-Infinity', '1e999',
                                   '"\\ud800"', '{"\\udfff":0}', b'\xef\xbb\xbf{}',
                                   '{"a":0,"\\u0061":1}'])
def test_strict_json_rejects_ambiguous_or_invalid_input(source):
    with pytest.raises(ValueError):
        strict_json_loads(source)


def test_strict_json_preserves_numeric_types_and_unicode():
    values = strict_json_loads(b'[1,1.0,true,"\\ud83d\\ude00"]')
    assert [type(value) for value in values] == [int, float, bool, str]
    assert values[-1] == "😀"


def test_canonical_is_stable_in_fresh_processes():
    program = ('from sentinel_evc.contracts import canonical_json,sha256_hex; '
               'v={k:v for k,v in {(\"z\",0.1),(\"a\",-0.0)}}; '
               'print(canonical_json(v).hex()); print(sha256_hex(v))')
    outputs = [subprocess.check_output([sys.executable, "-c", program],
               env=os.environ | {"PYTHONHASHSEED": seed}) for seed in ("1", "93")]
    expected = canonical_json({"a": 0.0, "z": 0.1}).hex().encode()
    assert outputs[0] == outputs[1]
    assert outputs[0].splitlines()[0] == expected


def test_domain_fields_and_aliases():
    points, gripper = [[0.1, 0.0, 0.3], [0.2, 0.1, 0.3]], ["hold"]
    plan = make_plan(points=points, gripper_events=gripper)
    margins = [0.02]
    cert = Certificate("cert", plan, "scene", margins, 0, None)
    original = plan.hash
    points[0][0] = 99.0
    gripper.append("open")
    margins[0] = 99.0
    assert plan.hash == original and plan.gripper_events == ("hold",)
    assert cert.margins == (0.02,) and cert.plan_hash == original
    assert tuple(field.name for field in dataclasses.fields(plan)) == (
        "points", "dt", "gripper_events", "controller_profile", "task_phase")
    assert tuple(field.name for field in dataclasses.fields(cert)) == (
        "cert_id", "plan", "scene_id", "margins", "inherit_depth", "parent_cert_id")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cert.plan.dt = 0.1
    with pytest.raises(TypeError):
        cert.plan.points[0][0] = 99.0


@pytest.mark.parametrize("changes", [
    {"points": ((0.11, 0.0, 0.3), (0.2, 0.1, 0.3))}, {"dt": 0.06},
    {"gripper_events": ("hold",)}, {"controller_profile": "another"}, {"task_phase": "next"},
])
def test_plan_hash_covers_each_field(changes):
    assert make_plan(**changes).hash != make_plan().hash


@pytest.mark.parametrize("changes", [{"points": [(0, 0, 0)]}, {"points": [(0, 0), (0, 0)]},
    {"points": [(0, 0, 0), (0, 0, float("inf"))]}, {"dt": 0}, {"dt": -1}, {"dt": float("nan")}])
def test_plan_rejects_invalid_invariants(changes):
    with pytest.raises(ValueError):
        make_plan(**changes)


@pytest.mark.parametrize(("depth", "parent"), [(0, "p"), (1, None), (-1, None), (5, "p"), (1, "")])
def test_certificate_depth_relationship(depth, parent):
    with pytest.raises(ValueError):
        Certificate("cert", make_plan(), "scene", (0.1,), depth, parent)


def test_certificate_invariants_and_max_depth():
    assert MAX_INHERIT_DEPTH == 4
    assert Certificate("cert", make_plan(), "scene", (0.1,), 4, "parent").inherit_depth == 4
    for cert_id, scene_id, margins in [("", "scene", (0.1,)), ("c", "", (0.1,)), ("c", "s", ())]:
        with pytest.raises(ValueError):
            Certificate(cert_id, make_plan(), scene_id, margins, 0, None)


def test_lease_snapshot_fields_signing_and_frozen_objects():
    context = make_context()
    lease = Lease("lease", make_plan().hash, context, "cert", 1, 1.0, "abc")
    assert set(lease.summary()) == {"lease_id", "final_hash", "context", "cert_id", "prefix_len", "deadline_mono", "hmac"}
    assert set(context.summary()) == {"robot", "boot", "epoch", "scene_id", "queue_rev", "committed_prefix_hash"}
    assert lease.signing_bytes() == canonical_json(lease.payload())
    assert b'"hmac"' not in lease.signing_bytes()
    assert dataclasses.replace(lease, hmac="def").signing_bytes() == lease.signing_bytes()
    position = [0.1, 0.2, 0.3]
    snapshot = Snapshot("obs", **context.summary(), position=position, observed_mono=0.5)
    assert set(snapshot.summary()) == {"obs_id", "robot", "boot", "epoch", "scene_id",
                                      "queue_rev", "committed_prefix_hash", "position", "observed_mono"}
    position[0] = 99
    assert snapshot.position == (0.1, 0.2, 0.3) and snapshot.context == context
    for obj in (make_plan(), Certificate("c", make_plan(), "s", (0.1,), 0, None), context, lease, snapshot):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, dataclasses.fields(obj)[0].name, None)


def test_preserved_domain_helpers_copy_nested_inputs():
    center, lo, hi = [0, 0, 0], [0, 0, 0], [1, 1, 1]
    sphere = Sphere(center, 0.1)
    obstacles = [sphere]
    scene = Scene("s", obstacles, lo, hi)
    params = {"weight": [0.5]}
    transform = TransformRecord("t", "mix", "p", "c", params)
    center[0] = lo[0] = hi[0] = 99
    obstacles.clear()
    params["weight"].append(0.8)
    assert sphere.center == (0, 0, 0)
    assert scene.ws_lo == (0, 0, 0) and scene.ws_hi == (1, 1, 1)
    assert scene.obstacles == (sphere,)
    assert dict(transform.parameters)["weight"] == (0.5,)


def test_nonempty_ids_and_opaque_string_dependencies():
    context = make_context()
    for create in (
        lambda: dataclasses.replace(context, robot=""),
        lambda: dataclasses.replace(context, scene_id=""),
        lambda: Lease("", make_plan().hash, context, "cert", 1, 1.0),
        lambda: Lease("lease", make_plan().hash, context, "", 1, 1.0),
        lambda: Snapshot("", **context.summary(), position=(0, 0, 0), observed_mono=0.0),
        lambda: make_plan(gripper_events=[["open"]]),
        lambda: make_plan(controller_profile=[]),
        lambda: make_plan(task_phase=[]),
    ):
        with pytest.raises(ValueError):
            create()
