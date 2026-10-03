"""Finite, signed launch qualification for a VLA adapter.

Qualification compares a reference and candidate adapter on the same bounded
probe suite.  A PASS covers only those exact recorded probes.  It is not a
task-success, collision, dynamics, physical-stop, or functional-safety proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contracts import canonical_json
from .events import EventLog
from .evidence import build_bundle, verify_bundle
from .native_gateway import NativeGatewayDenied
from .scenario import strict_json


PACK_SCHEMA = "sentinel-vla-probe-pack-v1"
REPORT_SCHEMA = "sentinel-vla-launch-qualification-v1"
REPRODUCTION_SCHEMA = "sentinel-vla-launch-reproduction-v1"
MAX_PACK_BYTES = 8 * 1024 * 1024
MAX_PROBES = 128
MAX_COMPONENTS = 64
MAX_CAMERAS = 8
_DIGEST_PREFIX = "sha256:"


def _exact_json(value: Any) -> bytes:
    """Deterministic JSON that preserves Python's exact float round trip."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest_bytes(raw: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(raw).hexdigest()


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(_DIGEST_PREFIX):
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _text(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a non-empty printable string of at most {maximum} characters")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any, name: str) -> int | float:
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} fields must be exactly {sorted(expected)}")
    return value


def _array_parts(array: Any) -> tuple[str, tuple[int, ...], bytes]:
    value = array
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    if not hasattr(value, "shape") or not hasattr(value, "dtype") or not hasattr(value, "tobytes"):
        raise ValueError("tensor must expose shape, dtype and tobytes")
    shape = tuple(int(dimension) for dimension in value.shape)
    if len(shape) > 8 or any(dimension < 0 for dimension in shape):
        raise ValueError("tensor shape is outside the supported identity envelope")
    try:
        raw = value.tobytes(order="C")
    except TypeError:
        raw = value.tobytes()
    raw = raw if isinstance(raw, bytes) else bytes(raw)
    if len(raw) > MAX_PACK_BYTES:
        raise ValueError("tensor bytes exceed 8 MiB")
    return str(value.dtype), shape, raw


def capture_tensor_identity(tensor: Any) -> str:
    """Return the native dtype/NUL/shape/NUL/C-order-bytes identity.

    The helper deliberately imports neither NumPy nor Torch.  Those objects
    stay owned by the policy environment.
    """
    dtype, shape, raw = _array_parts(tensor)
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json(list(shape)))
    digest.update(b"\0")
    digest.update(raw)
    return _DIGEST_PREFIX + digest.hexdigest()


def capture_probe(*, probe_id: str, input_hash: str, cameras: dict[str, Any],
                  state_names: list[str], state_units: list[str], state_values: list[int | float],
                  chunk_id: int, action_index: int, action: Any) -> dict[str, Any]:
    """Capture one JSON-compatible probe without importing an array library.

    Camera values may be actual tensor-like objects or already captured sha256
    identities.  The action must be the actual final tensor/array.
    """
    if not isinstance(cameras, dict):
        raise ValueError("cameras must be a policy-key mapping")
    dtype, shape, raw = _array_parts(action)
    dtype_object = getattr(action, "dtype", None)
    byte_order = getattr(dtype_object, "byteorder", None)
    if dtype not in ("float32", "float64"):
        raise ValueError("action dtype must be float32 or float64")
    if sys.byteorder != "little" or byte_order not in (None, "=", "<"):
        raise ValueError("action capture requires little-endian float storage")
    probe = {
        "id": probe_id,
        "input_hash": input_hash,
        "consumed": {
            "cameras": {key: value if _is_digest(value) else capture_tensor_identity(value)
                        for key, value in cameras.items()},
            "state": {"names": state_names, "units": state_units, "values": state_values},
        },
        "cursor": {"chunk_id": chunk_id, "action_index": action_index},
        "action": {"dtype": dtype, "shape": list(shape), "byte_order": "little",
                   "bytes_hex": raw.hex()},
    }
    return _validate_probe(probe, 0)


def _action_identity(dtype: str, shape: list[int], raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json(shape))
    digest.update(b"\0")
    digest.update(raw)
    return _DIGEST_PREFIX + digest.hexdigest()


def _validate_action(value: Any, name: str) -> dict[str, Any]:
    value = _exact_keys(value, {"dtype", "shape", "byte_order", "bytes_hex"}, name)
    dtype = value["dtype"]
    if dtype not in ("float32", "float64"):
        raise ValueError(f"{name}.dtype must be float32 or float64")
    if value["byte_order"] != "little":
        raise ValueError(f"{name}.byte_order must be little")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) not in (1, 2):
        raise ValueError(f"{name}.shape must be [N] or [1,N]")
    if len(shape) == 2 and shape[0] != 1:
        raise ValueError(f"{name}.shape must be [N] or [1,N]")
    count = shape[-1] if shape else 0
    if type(count) is not int or not 1 <= count <= MAX_COMPONENTS or any(type(x) is not int for x in shape):
        raise ValueError(f"{name}.shape has an invalid component count")
    encoded = value["bytes_hex"]
    if not isinstance(encoded, str) or len(encoded) > MAX_COMPONENTS * 16:
        raise ValueError(f"{name}.bytes_hex is invalid")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValueError(f"{name}.bytes_hex is invalid") from exc
    width, code = (4, "f") if dtype == "float32" else (8, "d")
    if len(raw) != count * width:
        raise ValueError(f"{name}.bytes_hex length does not match dtype and shape")
    values = struct.unpack("<" + code * count, raw)
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} contains a non-finite action")
    return {"dtype": dtype, "shape": shape, "byte_order": "little",
            "bytes_hex": encoded.lower()}


def _validate_probe(value: Any, index: int) -> dict[str, Any]:
    name = f"probes[{index}]"
    value = _exact_keys(value, {"id", "input_hash", "consumed", "cursor", "action"}, name)
    probe_id = _text(value["id"], name + ".id", 128)
    if not _is_digest(value["input_hash"]):
        raise ValueError(name + ".input_hash must be a sha256 digest")
    consumed = _exact_keys(value["consumed"], {"cameras", "state"}, name + ".consumed")
    cameras = consumed["cameras"]
    if not isinstance(cameras, dict) or len(cameras) > MAX_CAMERAS:
        raise ValueError(name + ".consumed.cameras must contain at most 8 bindings")
    checked_cameras: dict[str, str] = {}
    for key, digest in cameras.items():
        checked_cameras[_text(key, name + ".camera key", 128)] = digest
        if not _is_digest(digest):
            raise ValueError(name + ".camera identity must be a sha256 digest")
    state = _exact_keys(consumed["state"], {"names", "units", "values"}, name + ".consumed.state")
    names, units, values = state["names"], state["units"], state["values"]
    if not all(isinstance(items, list) for items in (names, units, values)):
        raise ValueError(name + ".state arrays must be lists")
    if not 0 <= len(names) <= MAX_COMPONENTS or len(names) != len(units) or len(names) != len(values):
        raise ValueError(name + ".state arrays must have the same 0..64 length")
    checked_names = [_text(item, name + ".state name", 128) for item in names]
    if len(checked_names) != len(set(checked_names)):
        raise ValueError(name + ".state names must be unique")
    checked_units = [_text(item, name + ".state unit", 64) for item in units]
    checked_values = [_finite(item, name + ".state value") for item in values]
    cursor = _exact_keys(value["cursor"], {"chunk_id", "action_index"}, name + ".cursor")
    checked_cursor = {
        "chunk_id": _integer(cursor["chunk_id"], name + ".cursor.chunk_id"),
        "action_index": _integer(cursor["action_index"], name + ".cursor.action_index"),
    }
    return {
        "id": probe_id,
        "input_hash": value["input_hash"],
        "consumed": {"cameras": checked_cameras, "state": {
            "names": checked_names, "units": checked_units, "values": checked_values}},
        "cursor": checked_cursor,
        "action": _validate_action(value["action"], name + ".action"),
    }


def _validate_pack(value: Any) -> dict[str, Any]:
    value = _exact_keys(value, {"schema", "suite_id", "adapter_digest", "provenance", "probes"}, "probe pack")
    if value["schema"] != PACK_SCHEMA:
        raise ValueError("unsupported probe pack schema")
    suite_id = _text(value["suite_id"], "suite_id", 256)
    if not _is_digest(value["adapter_digest"]):
        raise ValueError("adapter_digest must be a sha256 digest")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or len(_exact_json(provenance)) > 64 * 1024:
        raise ValueError("provenance must be a JSON object no larger than 64 KiB")
    probes = value["probes"]
    if not isinstance(probes, list) or not 1 <= len(probes) <= MAX_PROBES:
        raise ValueError("probes must contain 1..128 entries")
    checked = [_validate_probe(probe, index) for index, probe in enumerate(probes)]
    ids = [probe["id"] for probe in checked]
    if len(ids) != len(set(ids)):
        raise ValueError("probe ids must be unique")
    result = {"schema": PACK_SCHEMA, "suite_id": suite_id,
              "adapter_digest": value["adapter_digest"],
              "provenance": provenance, "probes": checked}
    if len(_exact_json(result)) > MAX_PACK_BYTES:
        raise ValueError("probe pack exceeds 8 MiB")
    return result


def _pack_bytes(source: bytes | bytearray | memoryview | dict[str, Any]) -> bytes:
    if isinstance(source, dict):
        raw = _exact_json(source)
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    else:
        raise ValueError("probe pack must be JSON bytes or an object")
    if len(raw) > MAX_PACK_BYTES:
        raise ValueError("probe pack exceeds 8 MiB")
    return raw


def freeze_probe_pack(source: bytes | bytearray | memoryview | dict[str, Any]) -> bytes:
    """Validate and return the canonical immutable probe-pack bytes."""
    raw = _pack_bytes(source)
    return _exact_json(_validate_pack(strict_json(raw)))


def _source_identity(raw: bytes, pack: dict[str, Any] | None) -> dict[str, Any]:
    return {"pack_sha256": _digest_bytes(raw),
            "suite_id": pack.get("suite_id") if pack else None,
            "adapter_digest": pack.get("adapter_digest") if pack else None}


def _issue(code: str, field_name: str, remediation: str, *, probe_id: str | None = None,
           reference: Any = None, candidate: Any = None, action_index: int | None = None) -> dict[str, Any]:
    value = {"code": code, "field": field_name, "reference": reference,
             "candidate": candidate, "remediation": remediation}
    if probe_id is not None:
        value["probe_id"] = probe_id
    if action_index is not None:
        value["action_index"] = action_index
    return value


def _first_behavior_difference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    for left, right in zip(reference["probes"], candidate["probes"]):
        probe_id = left["id"]
        left_cameras, right_cameras = left["consumed"]["cameras"], right["consumed"]["cameras"]
        for key in sorted(set(left_cameras) | set(right_cameras)):
            if left_cameras.get(key) != right_cameras.get(key):
                return _issue("CAMERA_BINDING_CHANGED", "consumed.cameras." + key,
                              "restore the reference policy-key-to-tensor binding or qualify a new suite",
                              probe_id=probe_id, reference=left_cameras.get(key), candidate=right_cameras.get(key))
        for field_name in ("names", "units", "values"):
            left_values = left["consumed"]["state"][field_name]
            right_values = right["consumed"]["state"][field_name]
            if len(left_values) != len(right_values) or any(
                    _exact_json(left_item) != _exact_json(right_item)
                    for left_item, right_item in zip(left_values, right_values)):
                mismatch = next((i for i, pair in enumerate(zip(left_values, right_values))
                                 if _exact_json(pair[0]) != _exact_json(pair[1])),
                                min(len(left_values), len(right_values)))
                return _issue("STATE_BINDING_CHANGED", f"consumed.state.{field_name}[{mismatch}]",
                              "restore the reference state order, units and values or qualify a new suite",
                              probe_id=probe_id,
                              reference=left_values[mismatch] if mismatch < len(left_values) else None,
                              candidate=right_values[mismatch] if mismatch < len(right_values) else None)
        for field_name in ("chunk_id", "action_index"):
            if left["cursor"][field_name] != right["cursor"][field_name]:
                return _issue("CURSOR_CHANGED", "cursor." + field_name,
                              "restore chunk selection and cursor semantics before launch",
                              probe_id=probe_id, reference=left["cursor"][field_name],
                              candidate=right["cursor"][field_name])
        left_action, right_action = left["action"], right["action"]
        for field_name in ("dtype", "shape"):
            if left_action[field_name] != right_action[field_name]:
                return _issue("ACTION_SCHEMA_CHANGED", "action." + field_name,
                              "restore the reference output schema or qualify a new suite",
                              probe_id=probe_id, reference=left_action[field_name], candidate=right_action[field_name])
        left_raw = bytes.fromhex(left_action["bytes_hex"])
        right_raw = bytes.fromhex(right_action["bytes_hex"])
        if left_raw != right_raw:
            width, code = (4, "f") if left_action["dtype"] == "float32" else (8, "d")
            action_index = next(index for index in range(len(left_raw) // width)
                                if left_raw[index * width:(index + 1) * width] != right_raw[index * width:(index + 1) * width])
            left_value = struct.unpack("<" + code, left_raw[action_index * width:(action_index + 1) * width])[0]
            right_value = struct.unpack("<" + code, right_raw[action_index * width:(action_index + 1) * width])[0]
            return _issue("ACTION_BYTES_CHANGED", "action.bytes",
                          "inspect normalization, chunk selection and postprocessing before launch",
                          probe_id=probe_id, action_index=action_index,
                          reference={"value": left_value, "identity": _action_identity(left_action["dtype"], left_action["shape"], left_raw)},
                          candidate={"value": right_value, "identity": _action_identity(right_action["dtype"], right_action["shape"], right_raw)})
    return None


def qualify(reference: bytes | bytearray | memoryview | dict[str, Any],
            candidate: bytes | bytearray | memoryview | dict[str, Any]) -> dict[str, Any]:
    """Compare two bounded probe packs and return PASS, BLOCK, or REVIEW."""
    raw_values: list[bytes] = []
    packs: list[dict[str, Any] | None] = []
    errors: list[str | None] = []
    for label, source in (("reference", reference), ("candidate", candidate)):
        try:
            raw = _pack_bytes(source)
            pack = _validate_pack(strict_json(raw))
            error = None
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            try:
                raw = _pack_bytes(source)
            except (ValueError, TypeError):
                raw = b""
            pack, error = None, f"{label}: {exc}"
        raw_values.append(raw)
        packs.append(pack)
        errors.append(error)
    sources = {"reference": _source_identity(raw_values[0], packs[0]),
               "candidate": _source_identity(raw_values[1], packs[1])}
    scope = {"probe_count": len(packs[0]["probes"]) if packs[0] else 0,
             "claim": "exact captured I/O fields on this finite recorded probe suite",
             "evidence": "newly signed derived decision, not an original execution signature",
             "excludes": ["task success", "collision or dynamics validation", "physical stop", "functional safety"]}
    issues: list[dict[str, Any]] = []
    if errors[0] or errors[1]:
        for label, error in zip(("reference", "candidate"), errors):
            if error:
                issues.append(_issue("INVALID_PROBE_PACK", label,
                                     "repair the schema and recapture the bounded probe pack",
                                     candidate=error))
        verdict, summary = "REVIEW", "Launch needs review because a probe pack is missing or invalid."
    else:
        left, right = packs[0], packs[1]
        assert left is not None and right is not None
        left_ids = [probe["id"] for probe in left["probes"]]
        right_ids = [probe["id"] for probe in right["probes"]]
        if left["suite_id"] != right["suite_id"]:
            issues.append(_issue("SUITE_MISMATCH", "suite_id",
                                 "capture both adapters with the same frozen suite",
                                 reference=left["suite_id"], candidate=right["suite_id"]))
        elif left_ids != right_ids:
            issues.append(_issue("PROBE_ALIGNMENT_MISMATCH", "probes[].id",
                                 "capture the same ordered probe ids before comparing outputs",
                                 reference=left_ids, candidate=right_ids))
        else:
            for l_probe, r_probe in zip(left["probes"], right["probes"]):
                if l_probe["input_hash"] != r_probe["input_hash"]:
                    issues.append(_issue("INPUT_MISMATCH", "input_hash",
                                         "rerun both adapters on the identical recorded input",
                                         probe_id=l_probe["id"], reference=l_probe["input_hash"],
                                         candidate=r_probe["input_hash"]))
                    break
        if issues:
            verdict, summary = "REVIEW", "Launch needs review because the candidate was not measured on the aligned reference inputs."
        else:
            difference = _first_behavior_difference(left, right)
            if difference:
                issues.append(difference)
                verdict, summary = "BLOCK", "Launch blocked at the first changed consumed input, cursor, or final action."
            else:
                verdict = "PASS"
                changed = left["adapter_digest"] != right["adapter_digest"]
                summary = ("Adapter digest changed, but captured bindings, cursor and final action match on every aligned probe."
                           if changed else "Captured bindings, cursor and final action match on every aligned probe.")
                if changed:
                    issues.append(_issue("ADAPTER_DIGEST_CHANGED", "adapter_digest",
                                         "retain this report with the candidate release; PASS remains limited to these probes",
                                         reference=left["adapter_digest"], candidate=right["adapter_digest"]))
    report = {"schema": REPORT_SCHEMA, "verdict": verdict, "summary": summary,
              "issues": issues, "sources": sources, "scope": scope}
    if verdict == "BLOCK":
        from .launch_diagnosis import diagnose_bindings
        report["diagnosis"] = diagnose_bindings(packs[0], packs[1])
    return report


def build_qualification_capsule(reference: bytes, candidate: bytes, out_dir: str | Path,
                                run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a newly signed derived decision capsule from exact pack bytes."""
    if not isinstance(reference, bytes) or not isinstance(candidate, bytes):
        raise ValueError("capsule inputs must be immutable bytes")
    out = Path(out_dir)
    if out.exists() and (out.is_symlink() or not out.is_dir() or any(out.iterdir())):
        raise ValueError("qualification output must be a new or empty directory")
    report = qualify(reference, candidate)
    log = EventLog(run_id=run_id, schema_version="product-v1")
    log.append("PREPARE", operation="launch_qualification",
               reference_sha256=_digest_bytes(reference), candidate_sha256=_digest_bytes(candidate))
    log.append("OUTCOME", operation="launch_qualification", verdict=report["verdict"],
               report_sha256=_digest_bytes(_exact_json(report)))
    bundle = build_bundle(log, str(out), artifacts={
        "reference.json": reference,
        "candidate.json": candidate,
        "qualification.json": _exact_json(report),
    })
    return report, bundle


def _read_regular_snapshot(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(path.name + " is not a readable regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError(path.name + " exceeds its bound or is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError(path.name + " exceeds its bound")
        return raw
    finally:
        os.close(descriptor)


def reproduce_qualification_capsule(bundle_dir: str | Path, public_key: str | Path,
                                    run_id: str) -> dict[str, Any]:
    """Verify signed bytes, requalify them, and compare the stored decision."""
    base = {"schema": REPRODUCTION_SCHEMA, "run_id": run_id}
    bundle = Path(bundle_dir)
    try:
        limits = {"manifest.json": 1024 * 1024, "manifest.sig": 64,
                  "events.jsonl": 1024 * 1024, "reference.json": MAX_PACK_BYTES,
                  "candidate.json": MAX_PACK_BYTES, "qualification.json": 1024 * 1024}
        snapshot = {name: _read_regular_snapshot(bundle / name, maximum)
                    for name, maximum in limits.items()}
        key_snapshot = _read_regular_snapshot(Path(public_key), 32)
        if len(key_snapshot) != 32:
            raise ValueError("public key must contain exactly 32 bytes")
        manifest = strict_json(snapshot["manifest.json"])
        expected = {"events.jsonl", "reference.json", "candidate.json", "qualification.json"}
        if not isinstance(manifest, dict) or set(manifest.get("files", {})) != expected:
            raise ValueError("signed capsule file set is not the launch qualification schema")
        # verify_bundle remains the independent verifier, but it receives a
        # private directory made from the exact bytes used below.  The source
        # path cannot change between verification and qualification.
        with tempfile.TemporaryDirectory(prefix="sentinel-launch-reproduce-") as temporary:
            fixed = Path(temporary)
            fixed_bundle = fixed / "bundle"
            fixed_bundle.mkdir()
            for name, raw in snapshot.items():
                (fixed_bundle / name).write_bytes(raw)
            fixed_key = fixed / "selected.public"
            fixed_key.write_bytes(key_snapshot)
            ok, reason = verify_bundle(str(fixed_bundle), str(fixed_key), run_id)
        if not ok:
            return {**base, "reproduction": "FAIL", "launch_verdict": None, "reason": reason}
        reference = snapshot["reference.json"]
        candidate = snapshot["candidate.json"]
        stored = strict_json(snapshot["qualification.json"])
        if not isinstance(stored, dict):
            raise ValueError("stored qualification must be an object")
        reproduced = qualify(reference, candidate)
        # Repair hints can improve independently of the exact verdict. Their
        # original bytes remain authenticated by the bundle, but are not part
        # of the authoritative qualification algorithm.
        stored_decision = {key: value for key, value in stored.items() if key != "diagnosis"}
        reproduced_decision = {key: value for key, value in reproduced.items() if key != "diagnosis"}
        if _exact_json(stored_decision) != _exact_json(reproduced_decision):
            return {**base, "reproduction": "FAIL", "launch_verdict": reproduced["verdict"],
                    "reason": "stored authoritative qualification does not match requalification of the signed probe bytes"}
    except (OSError, ValueError, TypeError) as exc:
        return {**base, "reproduction": "FAIL", "launch_verdict": None,
                "reason": "capsule unreadable: " + str(exc)}
    return {**base, "reproduction": "PASS", "launch_verdict": reproduced["verdict"],
            "diagnosis_reproduction": "MATCH" if _exact_json(stored.get("diagnosis")) == _exact_json(reproduced.get("diagnosis")) else "UPDATED_HINTS",
            "reason": "signed probe bytes reproduce the stored authoritative qualification exactly; original repair hints remain signature-verified"}


def export_capsule(reference: bytes, candidate: bytes, out_dir: str | Path,
                   run_id: str) -> dict[str, Any]:
    """Export a capsule and return the existing ``build_bundle`` metadata."""
    return build_qualification_capsule(reference, candidate, out_dir, run_id)[1]


def reproduce_capsule(bundle_dir: str | Path, public_key: str | Path,
                      run_id: str) -> dict[str, Any]:
    """Stable short name for :func:`reproduce_qualification_capsule`."""
    return reproduce_qualification_capsule(bundle_dir, public_key, run_id)


@dataclass(frozen=True)
class QualifiedNativeWriter:
    """Trusted in-process writer gated by a finite probe qualification.

    The object is intended as the writer passed to ``NativeGateway.submit``.
    It checks the immutable packs and current adapter digest, acknowledges
    ``entered()`` and checks the digest again immediately before its single
    downstream call.  ``current_adapter_digest`` must be an O(1) read of an
    already loaded identity, not a per-request model-file hash.  Concurrent
    adapter hot-swap requires the loader to hold a shared external lock across
    the complete ``NativeGateway.submit``/writer call.  This class does not
    provide atomic hot-swap, process isolation, or physical safety.
    """

    reference_pack: bytes
    candidate_pack: bytes
    current_adapter_digest: Callable[[], str]
    downstream_writer: Callable[[Any], Any]
    _candidate_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_pack, bytes) or not isinstance(self.candidate_pack, bytes):
            raise ValueError("qualification packs must be immutable bytes")
        if not callable(self.current_adapter_digest) or not callable(self.downstream_writer):
            raise ValueError("adapter digest callback and downstream writer must be callable")
        report = qualify(self.reference_pack, self.candidate_pack)
        if report["verdict"] != "PASS":
            raise NativeGatewayDenied("LAUNCH_QUALIFICATION_" + report["verdict"], report["summary"])
        candidate = _validate_pack(strict_json(self.candidate_pack))
        object.__setattr__(self, "_candidate_digest", candidate["adapter_digest"])
        self._check_current_digest()

    def _check_current_digest(self) -> None:
        try:
            current = self.current_adapter_digest()
        except Exception as exc:
            raise NativeGatewayDenied("ADAPTER_DIGEST_UNAVAILABLE", "could not read current adapter digest") from exc
        if not _is_digest(current):
            raise NativeGatewayDenied("ADAPTER_DIGEST_UNAVAILABLE",
                                      "current adapter digest is missing or invalid")
        if current != self._candidate_digest:
            raise NativeGatewayDenied("ADAPTER_DIGEST_CHANGED",
                                      "current adapter bytes differ from the qualified candidate")

    def __call__(self, request: Any, entered: Callable[[], None]) -> Any:
        self._check_current_digest()
        if not callable(entered):
            raise NativeGatewayDenied("WRITER_ENTRY_MISSING", "gateway entry acknowledgement is required")
        entered()
        self._check_current_digest()
        return self.downstream_writer(request)


__all__ = [
    "PACK_SCHEMA", "REPORT_SCHEMA", "QualifiedNativeWriter",
    "build_qualification_capsule", "capture_probe", "capture_tensor_identity", "export_capsule",
    "freeze_probe_pack", "qualify", "reproduce_capsule", "reproduce_qualification_capsule",
]
