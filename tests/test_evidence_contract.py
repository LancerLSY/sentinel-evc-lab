"""证据字节与独立校验回归；格式 fixture 不代表运行成绩。"""

import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from sentinel_evc import evidence
from sentinel_evc.events import EventLog


FIXTURE = Path(__file__).parent / "fixtures" / "sample_events.jsonl"


def signed_fixture(out, raw=None, **overrides):
    bundle = out / "bundle"
    bundle.mkdir(parents=True)
    raw = FIXTURE.read_bytes() if raw is None else raw
    lines = raw.split(b"\n")[:-1]
    manifest = {
        "run_id": "format-fixture", "event_count": len(lines),
        "tip_hash": "sha256:" + sha256(lines[-1] if lines else b"").hexdigest(),
        "files": {"events.jsonl": "sha256:" + sha256(raw).hexdigest()},
    }
    manifest.update(overrides)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    key = Ed25519PrivateKey.generate()
    (bundle / "events.jsonl").write_bytes(raw)
    (bundle / "manifest.json").write_bytes(manifest_bytes)
    (bundle / "manifest.sig").write_bytes(key.sign(manifest_bytes))
    public = out / "fixture.public"
    public.write_bytes(key.public_key().public_bytes_raw())
    return bundle, public


def test_builder_emits_only_minimal_bundle_and_raw_keys(tmp_path):
    log = EventLog("test")
    log.append("PROPOSAL", role="parent")
    result = evidence.build_bundle(log, str(tmp_path))
    bundle = Path(result["bundle_dir"])
    assert {p.name for p in bundle.iterdir()} == {"events.jsonl", "manifest.json", "manifest.sig"}
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    assert set(manifest) == {"run_id", "event_count", "tip_hash", "files"}
    assert set(manifest["files"]) == {"events.jsonl"}
    assert len((bundle / "manifest.sig").read_bytes()) == 64
    assert len(Path(result["public_key"]).read_bytes()) == 32
    assert Path(result["public_key"]).parent != bundle
    assert evidence.verify_bundle(str(bundle), result["public_key"], "test")[0]


def test_builder_rejects_empty_and_already_drained_stream(tmp_path):
    log = EventLog("test")
    with pytest.raises(ValueError):
        evidence.build_bundle(log, str(tmp_path / "empty"))
    log.append("PROPOSAL", role="parent")
    log.drain()
    log.append("PROPOSAL", role="child")
    with pytest.raises(ValueError):
        evidence.build_bundle(log, str(tmp_path / "partial"))


def test_independent_fixed_fixture_roundtrip(tmp_path):
    bundle, public = signed_fixture(tmp_path)
    assert evidence.verify_bundle(str(bundle), str(public), "format-fixture")[0]
    assert all(value is True for value in evidence.verify_layers(str(bundle), str(public), "format-fixture").values())


def test_four_tampers_have_distinct_ordered_failure_profiles(tmp_path):
    bundle, public = signed_fixture(tmp_path / "original")
    changed = tmp_path / "changed"
    shutil.copytree(bundle, changed)
    path = changed / "events.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"deadline_mono":1.5000000000000000e+00', b'"deadline_mono":1.6000000000000000e+00'))
    cut = tmp_path / "cut"
    shutil.copytree(bundle, cut)
    (cut / "events.jsonl").write_bytes(b"\n".join(FIXTURE.read_bytes().split(b"\n")[:-4]) + b"\n")
    wrong = tmp_path / "wrong.public"
    wrong.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes_raw())
    profiles = [
        evidence.failed_layers(str(changed), str(public), "format-fixture"),
        evidence.failed_layers(str(cut), str(public), "format-fixture"),
        evidence.failed_layers(str(bundle), str(wrong), "format-fixture"),
        evidence.failed_layers(str(bundle), str(public), "wrong-run"),
    ]
    assert [profile[0] for profile in profiles] == ["hash_chain", "event_count", "signature", "run_id"]
    assert "tip_hash" in profiles[1]
    assert len(set(profiles)) == 4


@pytest.mark.parametrize("transform", [
    lambda raw: raw.replace(b"\n", b"\r\n"),
    lambda raw: raw[:-1],
    lambda raw: raw + b"\n",
    lambda raw: b"\xef\xbb\xbf" + raw,
    lambda raw: raw.replace(b'"seq":0,', b'"seq":0,"seq":0,', 1),
    lambda raw: raw.replace(b'"seq":0,', b'"seq":NaN,', 1),
    lambda raw: raw.replace(b'"seq":0,', b'"seq":Infinity,', 1),
    lambda raw: raw.replace(b'"seq":0,', b'"seq":1e999,', 1),
    lambda raw: raw.replace(b'"role":"parent"', b'"role":"\\ud800"', 1),
    lambda raw: raw.replace(b'"role":"parent"', b'"role":"\xff"', 1),
])
def test_invalid_json_or_framing_fails_even_with_valid_signature(tmp_path, transform):
    bundle, public = signed_fixture(tmp_path, transform(FIXTURE.read_bytes()))
    result = evidence.verify_layers(str(bundle), str(public), "format-fixture")
    assert result["hash_chain"] is not True
    assert not evidence.verify_bundle(str(bundle), str(public), "format-fixture")[0]


def test_empty_signed_stream_is_rejected(tmp_path):
    bundle, public = signed_fixture(tmp_path, b"")
    assert not evidence.verify_bundle(str(bundle), str(public), "format-fixture")[0]


def test_unicode_line_separators_do_not_split_jsonl_records(tmp_path):
    event = {
        "seq": 0, "ts_mono_ns": 0, "ts_utc": "2026-09-19T00:00:00Z",
        "run_id": "format-fixture", "type": "REVOKE", "plan_hash": None,
        "lease_id": None, "cert_id": None,
        "payload": {"reason": "a\u2028b\u2029c", "old_epoch": 0, "new_epoch": 1},
        "prev_hash": "sha256:" + "0" * 64,
    }
    raw = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    bundle, public = signed_fixture(tmp_path, raw)
    assert evidence.verify_bundle(str(bundle), str(public), "format-fixture")[0]


def test_builder_does_not_overwrite_bundle(tmp_path):
    log = EventLog("test")
    log.append("PROPOSAL", role="parent")
    result = evidence.build_bundle(log, str(tmp_path))
    signature = Path(result["bundle_dir"], "manifest.sig").read_bytes()
    with pytest.raises(FileExistsError):
        evidence.build_bundle(log, str(tmp_path))
    assert Path(result["bundle_dir"], "manifest.sig").read_bytes() == signature


@pytest.mark.parametrize("seq,payload", [
    (3, {"dropped_count": 2, "first_dropped_seq": 0, "last_dropped_seq": 2}),
    (3, {"dropped_count": 3, "first_dropped_seq": 1, "last_dropped_seq": 3}),
    (True, {"dropped_count": 1, "first_dropped_seq": 0, "last_dropped_seq": 0}),
])
def test_signed_gap_must_match_reserved_sequences(tmp_path, seq, payload):
    event = json.loads(FIXTURE.read_bytes().split(b"\n")[0])
    event.update(seq=seq, type="LOG_GAP", payload=payload)
    raw = json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    bundle, public = signed_fixture(tmp_path, raw)
    assert evidence.failed_layers(str(bundle), str(public), "format-fixture")[0] == "hash_chain"


def test_duplicate_manifest_keys_are_rejected_before_signature(tmp_path):
    bundle, public = signed_fixture(tmp_path)
    path = bundle / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b'"run_id":', b'"run_id":"format-fixture","run_id":'))
    assert evidence.failed_layers(str(bundle), str(public), "format-fixture")[0] == "run_id"


def test_event_run_id_is_checked_independently_of_manifest(tmp_path):
    raw = FIXTURE.read_bytes().replace(b'"run_id":"format-fixture"', b'"run_id":"different"', 1)
    bundle, public = signed_fixture(tmp_path, raw)
    assert evidence.failed_layers(str(bundle), str(public), "format-fixture")[0] == "run_id"


def test_manifest_file_set_is_exact(tmp_path):
    raw = FIXTURE.read_bytes()
    bundle, public = signed_fixture(tmp_path, files={"events.jsonl": "sha256:" + sha256(raw).hexdigest(), "extra": "ignored"})
    assert "file_digest" in evidence.failed_layers(str(bundle), str(public), "format-fixture")


def test_independent_canonical_golden_and_no_protocol_imports():
    assert evidence._canonical_json({"中": -0.0, "a": [True, 1, 0.1]}) == b'{"a":[true,1,1.0000000000000001e-01],"\xe4\xb8\xad":0.0000000000000000e+00}'
    tree = ast.parse(Path(evidence.__file__).read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_bundle":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                assert child.module not in {"contracts", "events", "executor", "authority"}


def test_missing_files_or_bad_public_key_are_reported(tmp_path):
    bundle, public = signed_fixture(tmp_path)
    public.write_bytes(b"bad")
    assert evidence.failed_layers(str(bundle), str(public), "format-fixture") == ("signature",)
    (bundle / "manifest.sig").unlink()
    assert evidence.failed_layers(str(bundle), str(public), "format-fixture") == ("files_present",)
