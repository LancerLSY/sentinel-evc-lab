import base64
import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import pytest

from sentinel_evc.assets import AssetStore


OBJ = b"""# triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""


def test_obj_import_persists_hash_and_real_triangle_preview(tmp_path):
    store = AssetStore(tmp_path)
    asset = store.import_asset("tool", "OBJ", content_base64=base64.b64encode(OBJ).decode())
    assert asset["source_sha256"] == "sha256:" + hashlib.sha256(OBJ).hexdigest()
    assert asset["physics_authorized"] is False
    geometry = store.geometry(asset["id"])
    assert geometry["vertices"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert geometry["triangles"] == [[0, 1, 2]]
    assert store.get(asset["id"])["id"] == asset["id"]
    assert [row["id"] for row in store.list()] == [asset["id"]]
    assert store.check(asset["id"])["status"] == "not_applicable"
    assert store.get(asset["id"])["last_check"]["status"] == "not_applicable"


def test_binary_stl_import_uses_all_triangle_vertices(tmp_path):
    header = b"sentinel".ljust(80, b"\0")
    facet = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    store = AssetStore(tmp_path)
    asset = store.import_asset("fixture", "stl", content=header + struct.pack("<I", 1) + facet)
    geometry = store.geometry(asset["id"])
    assert geometry["triangle_count"] == 1
    assert geometry["vertices"][1] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("xml", [
    b'<!DOCTYPE mujoco [<!ENTITY x SYSTEM "file:///etc/passwd">]><mujoco/>',
    b'<mujoco><include file="/tmp/model.xml"/></mujoco>',
    b'<mujoco><asset><mesh file="http://example.invalid/a.stl"/></asset></mujoco>',
    b'<mujoco><extension><plugin plugin="x"/></extension></mujoco>',
    b'<mujoco><size memory="999999G"/></mujoco>',
    b'<mujoco><worldbody><replicate count="100000"><geom type="sphere" size="1"/></replicate></worldbody></mujoco>',
    b'<mujoco><worldbody><composite type="grid" count="1000 1000 1"/></worldbody></mujoco>',
    b'<mujoco><worldbody><flexcomp count="1000 1000 1"/></worldbody></mujoco>',
    b'<mujoco><option iterations="1000000"/></mujoco>',
    b'<mujoco><worldbody><geom type="box" size="1e200 1 1"/></worldbody></mujoco>',
])
def test_xml_import_refuses_external_assets_entities_and_plugins(tmp_path, xml):
    with pytest.raises(ValueError):
        AssetStore(tmp_path).import_asset("unsafe", "mjcf", content=xml)


def test_urdf_primitive_preview_is_explicitly_not_authorization(tmp_path):
    xml = b'''<robot name="one"><link name="base"><visual><origin xyz="1 2 3"/>
      <geometry><cylinder radius="0.2" length="0.8"/></geometry></visual></link></robot>'''
    store = AssetStore(tmp_path)
    asset = store.import_asset("arm", "urdf", content=xml)
    geometry = store.geometry(asset["id"])
    assert geometry["primitives"][0]["dimensions"] == [0.2, 0.8]
    assert geometry["primitives"][0]["position"] == [1.0, 2.0, 3.0]
    assert geometry["physics_authorized"] is False


def test_declared_mjcf_primitive_has_a_complete_identity_rotation(tmp_path):
    store = AssetStore(tmp_path)
    asset = store.import_asset(
        "primitive", "mjcf",
        content=b'<mujoco><worldbody><geom type="box" size="1 2 3"/></worldbody></mujoco>',
    )
    rotation = store.geometry(asset["id"])["primitives"][0]["rotation"]
    assert rotation == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_xml_structure_counts_are_bounded_before_engine_compile(tmp_path):
    geoms = b"".join(b'<geom type="sphere" size="0.1"/>' for _ in range(1001))
    with pytest.raises(ValueError, match="structure"):
        AssetStore(tmp_path).import_asset("expanded", "mjcf", content=b"<mujoco><worldbody>" + geoms + b"</worldbody></mujoco>")


def test_mujoco_check_compiles_short_settles_and_replaces_preview(tmp_path, monkeypatch):
    xml = b'<mujoco><worldbody><geom type="box" size="1 2 3"/></worldbody></mujoco>'
    store = AssetStore(tmp_path)
    asset = store.import_asset("scene", "mjcf", content=xml)

    class Model:
        nbody, ngeom, njnt, nu = 1, 1, 0, 0
        geom_type = [1]
        geom_size = [[1.0, 2.0, 3.0]]

        @classmethod
        def from_xml_string(cls, value):
            assert value == xml.decode()
            return cls()

    class Data:
        def __init__(self, model):
            self.qpos, self.qvel = [], []
            self.geom_xpos = [[0.1, 0.2, 0.3]]
            self.geom_xmat = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

    fake = SimpleNamespace(
        __version__="3.14-test",
        MjModel=Model,
        MjData=Data,
        mjtGeom=SimpleNamespace(mjGEOM_BOX=1, mjGEOM_SPHERE=2, mjGEOM_CYLINDER=3, mjGEOM_CAPSULE=4),
        mj_forward=lambda model, data: None,
        mj_step=lambda model, data: None,
    )
    monkeypatch.setattr("sentinel_evc.assets.importlib.util.find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    report = store.check(asset["id"])
    assert report["status"] == "passed", report
    assert report["settle_steps"] == 5
    assert report["physics_authorized"] is False
    compiled = store.geometry(asset["id"])
    assert compiled["source"] == "mujoco-compiled"
    assert compiled["primitives"][0]["dimensions"] == [2.0, 4.0, 6.0]
    assert json.loads((tmp_path / asset["id"] / "check.json").read_text())["status"] == "passed"

    class WarningData(Data):
        def __init__(self, model):
            super().__init__(model)
            self.warning = [SimpleNamespace(number=1, lastinfo=0)]

    fake.MjData = WarningData
    warned = store.check(asset["id"])
    assert warned["status"] == "failed"
    assert "warnings" in warned["error"]["message"]


def test_asset_inputs_are_bounded_and_identifiers_do_not_traverse(tmp_path):
    store = AssetStore(tmp_path)
    with pytest.raises(ValueError):
        store.import_asset("x", "obj", content=b"", content_base64="")
    with pytest.raises(ValueError):
        store.import_asset("x", "fbx", content=b"x")
    with pytest.raises(KeyError):
        store.get("../asset-secret")
    linked = tmp_path / "linked-assets"
    linked.symlink_to(tmp_path / "actual-assets", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        AssetStore(linked)


def test_model_check_rejects_source_changed_after_import(tmp_path, monkeypatch):
    store = AssetStore(tmp_path)
    asset = store.import_asset("scene", "mjcf", content=b'<mujoco><worldbody/></mujoco>')
    (tmp_path / asset["id"] / "source.mjcf").write_bytes(b'<mujoco model="changed"/>')
    monkeypatch.setattr("sentinel_evc.assets.importlib.util.find_spec", lambda name: object())
    with pytest.raises(ValueError, match="import digest"):
        store.check(asset["id"])


@pytest.mark.parametrize("metadata", [[], 7, {"format": "obj"}, {"id": "wrong", "format": "obj"}])
def test_corrupt_asset_metadata_isolated_while_valid_asset_remains_visible(tmp_path, metadata):
    store = AssetStore(tmp_path)
    valid = store.import_asset("valid", "obj", content=OBJ)
    corrupt_id = "asset-" + "a" * 32
    corrupt = tmp_path / corrupt_id
    corrupt.mkdir()
    (corrupt / "asset.json").write_text(json.dumps(metadata), "utf-8")
    with pytest.raises(KeyError):
        store.get(corrupt_id)
    assert [record["id"] for record in store.list()] == [valid["id"]]


def test_nested_asset_metadata_cannot_enable_physics_or_drop_list_response(tmp_path):
    store = AssetStore(tmp_path)
    valid = store.import_asset("valid", "obj", content=OBJ)
    path = tmp_path / valid["id"] / "asset.json"
    record = json.loads(path.read_text("utf-8"))
    record["capabilities"] = []
    path.write_text(json.dumps(record), "utf-8")
    with pytest.raises(KeyError):
        store.get(valid["id"])
    assert store.list() == []
