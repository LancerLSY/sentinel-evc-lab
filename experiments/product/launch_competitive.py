#!/usr/bin/env python3
"""Run the same Launch Gate packs through pinned authorization products.

This experiment does not score an intended, newly authorized request as a
safety failure.  It asks a narrower product question: after reference and
camera-swapped adapters are each approved as new intent, do the installed
authorization APIs accept their final actions, and do they themselves emit a
reference-versus-candidate VLA semantic qualification?
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


KINEGRANT_COMMIT = "3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f"
RLSOK_COMMIT = "5df8ce9a77349bf1e82e3b1f60aa0cd55138e034"
SCHEMA = "sentinel-launch-competitive-evidence-v1"
PACK_LIMIT = 8 * 1024 * 1024
RESULT_LIMIT = 16 * 1024 * 1024
ARCHIVE_LIMIT = 128 * 1024 * 1024


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"non-finite JSON value in {label}: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def _snapshot(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"snapshot source is not a regular file: {path}")
        if details.st_size > limit:
            raise ValueError(f"snapshot source exceeds {limit} bytes: {path}")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise ValueError(f"snapshot source exceeds {limit} bytes: {path}")
        return raw
    finally:
        os.close(descriptor)


def _parse_pack(raw: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label)
    if not isinstance(value, dict):
        raise ValueError(f"launch pack must be an object: {label}")
    if value.get("schema") != "sentinel-vla-probe-pack-v1" or len(value.get("probes", [])) != 20:
        raise ValueError(f"unexpected launch pack: {label}")
    return value


def _read_pack(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = _snapshot(path, PACK_LIMIT)
    return raw, _parse_pack(raw, str(path))


def _git_head(path: Path) -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()
    for arguments in (["git", "diff", "--quiet", "HEAD", "--"],
                      ["git", "diff", "--cached", "--quiet", "HEAD", "--"]):
        if subprocess.run(arguments, cwd=path, check=False).returncode:
            raise RuntimeError(f"tracked competitor source is dirty: {path}")
    return head


def _git_archive(path: Path, expected: str, destination: Path) -> dict[str, Any]:
    head = _git_head(path)
    if head != expected:
        raise RuntimeError(f"competitor checkout is not pinned at {expected}: {path}")
    process = subprocess.run(["git", "archive", "--format=tar", expected], cwd=path,
                             check=True, capture_output=True)
    archive = process.stdout
    if len(archive) > ARCHIVE_LIMIT:
        raise RuntimeError(f"competitor Git archive exceeds {ARCHIVE_LIMIT} bytes: {path}")
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"unsafe path in competitor Git archive: {member.name}")
            if member.isdev() or member.isfifo():
                raise RuntimeError(f"unsupported member in competitor Git archive: {member.name}")
        bundle.extractall(destination, filter="data")
    return {
        "commit": head,
        "git_archive_sha256": _sha(archive),
        "git_archive_bytes": len(archive),
        "source_mode": "git archive of pinned commit; untracked checkout files excluded",
    }


def _source_hashes(root: Path, names: list[str]) -> dict[str, str]:
    return {name: _sha((root / name).read_bytes()) for name in names}


def _tree_receipt(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            kind, content = b"L", os.readlink(path).encode("utf-8")
        elif path.is_file():
            kind, content = b"F", path.read_bytes()
        elif path.is_dir():
            continue
        else:
            raise RuntimeError(f"unsupported dependency tree entry: {path}")
        digest.update(kind + b"\0" + relative + b"\0" + content + b"\0")
        count += 1
    return {"tree_sha256": "sha256:" + digest.hexdigest(), "entries": count}


def _validate_selected_case(launch: dict[str, Any], launch_raw: bytes,
                            reference: dict[str, Any], reference_raw: bytes,
                            candidate: dict[str, Any], candidate_raw: bytes) -> dict[str, Any]:
    cases = launch.get("cases")
    if not isinstance(cases, list):
        raise ValueError("launch result cases must be a list")
    selected = [row for row in cases if isinstance(row, dict) and row.get("case") == "camera_swap"]
    if len(selected) != 1:
        raise ValueError("launch result must contain exactly one camera_swap case")
    row = selected[0]
    report = row.get("report")
    if not isinstance(report, dict) or report.get("verdict") != "BLOCK":
        raise ValueError("retained Sentinel camera-swap result is not BLOCK")
    issues = report.get("issues")
    if not isinstance(issues, list) or not issues or issues[0].get("code") != "CAMERA_BINDING_CHANGED":
        raise ValueError("retained Sentinel camera-swap result lacks the expected first issue")
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("camera-swap report has no source identities")
    expected = (("reference", reference, reference_raw), ("candidate", candidate, candidate_raw))
    for role, pack, raw in expected:
        identity = sources.get(role)
        if not isinstance(identity, dict):
            raise ValueError(f"camera-swap report has no {role} source identity")
        if identity.get("pack_sha256") != _sha(raw):
            raise ValueError(f"camera-swap report {role} hash does not match snapshotted bytes")
        if identity.get("suite_id") != pack.get("suite_id"):
            raise ValueError(f"camera-swap report {role} suite does not match snapshotted bytes")
    if reference.get("suite_id") != candidate.get("suite_id"):
        raise ValueError("reference and candidate suite identifiers differ")
    left = reference["probes"]
    right = candidate["probes"]
    if [probe.get("id") for probe in left] != [probe.get("id") for probe in right]:
        raise ValueError("reference and candidate ordered probe identifiers differ")
    if [probe.get("input_hash") for probe in left] != [probe.get("input_hash") for probe in right]:
        raise ValueError("reference and candidate input hashes differ")
    camera_changed = False
    for reference_probe, candidate_probe in zip(left, right, strict=True):
        if reference_probe["consumed"]["cameras"] != candidate_probe["consumed"]["cameras"]:
            camera_changed = True
        for field in ("state",):
            if reference_probe["consumed"][field] != candidate_probe["consumed"][field]:
                raise ValueError(f"camera-swap case also changes consumed.{field}")
        for field in ("cursor", "action"):
            if reference_probe[field] != candidate_probe[field]:
                raise ValueError(f"camera-swap case also changes {field}")
    if not camera_changed:
        raise ValueError("camera-swap candidate does not change camera bindings")
    return {
        "row": row,
        "launch_result_sha256": _sha(launch_raw),
        "reference_pack_sha256": _sha(reference_raw),
        "candidate_pack_sha256": _sha(candidate_raw),
    }


def _python_dependency_receipt(python: str) -> dict[str, Any]:
    program = r'''import hashlib, importlib.metadata as m, json, os
distribution=m.distribution("cryptography")
metadata=distribution._path/"METADATA"
module=distribution.locate_file("cryptography/__init__.py")
def sha(path): return "sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()
tree=hashlib.sha256(); count=0
for entry in sorted(distribution.files, key=lambda item:str(item)):
 path=distribution.locate_file(entry)
 if not path.is_file(): continue
 raw=path.read_bytes(); tree.update(b"F\0"+str(entry).encode()+b"\0"+raw+b"\0"); count+=1
print(json.dumps({"name":"cryptography","version":distribution.version,
"module_sha256":sha(module),"metadata_sha256":sha(metadata),
"distribution_tree_sha256":"sha256:"+tree.hexdigest(),"distribution_files":count}))'''
    process = subprocess.run([python, "-c", program], check=True,
                             capture_output=True, text=True)
    return _strict_json(process.stdout.encode("utf-8"), "cryptography receipt")


def _node_dependency_receipt(node_modules: Path) -> dict[str, Any]:
    names = [".package-lock.json", "typescript/package.json", "zod/package.json",
             "@types/node/package.json"]
    for name in names:
        if not (node_modules / name).is_file():
            raise RuntimeError(f"required installed dependency receipt is missing: {name}")
    return {"root": str(node_modules.resolve()), "files": _source_hashes(node_modules, names),
            "package_trees": {
                name: _tree_receipt(node_modules / name)
                for name in ("typescript", "zod", "@types/node")
            }}


def _run_kinegrant(reference: Path, candidate: Path, source: Path, output: Path) -> None:
    from kinegrant.capability import CapabilityIssuer
    from kinegrant.crypto import Ed25519KeyPair
    from kinegrant.gate import ActionGate, SQLiteReplayStore
    from kinegrant.gatekeeper import Gatekeeper
    from kinegrant.models import ActionRequest, PolicyRule
    from kinegrant.policy import PolicyEngine
    from kinegrant.receipt import ReceiptLog, verify_receipt_chain
    from kinegrant.revocation import RevocationList
    from kinegrant.sequence import ActionJournal, SequencePolicy

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sentinel-launch-kinegrant-") as temporary:
        scratch = Path(temporary)
        for pack_path in (reference, candidate):
            raw, pack = _read_pack(pack_path)
            pack_kind = "reference" if pack_path == reference else "camera_swap"
            for index, probe in enumerate(pack["probes"]):
                request = ActionRequest(
                    request_id=f"urn:sentinel:launch:{pack_kind}:{probe['id']}",
                    agent="urn:sentinel:vla-adapter",
                    target="urn:sentinel:libero-panda",
                    action="kg.action.move",
                    purpose="approved-launch-competitive-probe",
                    context={
                        "pack_kind": pack_kind,
                        "pack_sha256": _sha(raw),
                        "adapter_digest": pack["adapter_digest"],
                        "probe_id": probe["id"],
                        "input_hash": probe["input_hash"],
                        "consumed": probe["consumed"],
                        "cursor": probe["cursor"],
                        "final_action": probe["action"],
                    },
                )
                authority = Ed25519KeyPair.generate()
                rule = PolicyRule(
                    policy_id=f"urn:sentinel:launch-policy:{pack_kind}:{index}",
                    issuer=authority.kid,
                    target=request.target,
                    effect="allow",
                    actions=(request.action,), subjects=(request.agent,),
                    purposes=(request.purpose,),
                    constraints={"required_context": request.context, "min_approval_tier": 1},
                    obligations=("emitActionReceipt",),
                )
                engine = PolicyEngine([rule], trusted_policy_issuers={authority.kid},
                                      require_known_actions=True)
                decision = engine.evaluate(request)
                capability = CapabilityIssuer(authority).issue_scoped(
                    request, decision, ttl_seconds=10,
                    approval_tier=decision.required_approval_tier, wire_version="1.0")
                receipts = ReceiptLog(Ed25519KeyPair.generate())
                revocations = RevocationList()
                gatekeeper = Gatekeeper(
                    gate=ActionGate(
                        trusted_issuers={authority.kid},
                        replay_store=SQLiteReplayStore(scratch / f"{pack_kind}-{index}.sqlite"),
                        revocation_list=revocations),
                    sequence=SequencePolicy(()), journal=ActionJournal(),
                    receipt_log=receipts, revocation_list=revocations)
                dispatches: list[str] = []
                outcome = gatekeeper.execute(
                    capability, request,
                    lambda _verified: dispatches.append(probe["action"]["bytes_hex"]),
                    evidence_hash=_sha(raw),
                    obligation_results=[{"obligation": "emitActionReceipt", "status": "satisfied"}],
                )
                rows.append({
                    "pack": pack_kind, "probe_id": probe["id"],
                    "decision": "ALLOW" if outcome.allowed else "DENY",
                    "reason": outcome.reason or outcome.stage,
                    "dispatches": len(dispatches), "request_digest": request.digest,
                    "receipt_chain_valid": verify_receipt_chain(
                        receipts.entries, trusted_executors={receipts.executor_key.kid}),
                })
    output.write_bytes(_json_bytes({
        "product": "KineGrant", "version": "2.65.5", "commit": KINEGRANT_COMMIT,
        "api": ["ActionRequest", "PolicyEngine.evaluate", "CapabilityIssuer.issue_scoped",
                "Gatekeeper.execute", "ReceiptLog"],
        "calls": len(rows), "allowed": sum(row["decision"] == "ALLOW" for row in rows),
        "dispatches": sum(row["dispatches"] for row in rows),
        "receipt_chains_valid": all(row["receipt_chain_valid"] for row in rows),
        "data_seen": "full probe fields embedded in ActionRequest.context and bound by request digest",
        "observed_vla_semantic_comparison_api_called": False,
        "source_files": _source_hashes(source, [
            "src/kinegrant/models.py", "src/kinegrant/policy.py",
            "src/kinegrant/capability.py", "src/kinegrant/gate.py",
            "src/kinegrant/gatekeeper.py",
        ]),
        "rows": rows,
    }))


RLSOK_HARNESS = r"""
import { readFileSync, writeFileSync } from 'node:fs';
import { canonicalJson, sha256, type ExecutionEvidence } from '../../packages/core/evidence';
import { executablePolicyHash, executablePolicySpecSchema, diffExecutablePolicies, type ExecutablePolicySpec } from '../../packages/core/exec-spec';
import { configurationDigest, executionConfigurationV1Schema } from '../../packages/core/execution-configuration';
import { ReleaseExecutionGate } from '../../packages/core/execution-gate';
import type { ReleaseRecord } from '../../packages/core/release-policy';
import { runtimeAttestationSchema } from '../../packages/core/runtime-attestation';

const H=(c:string)=>c.repeat(64); const DEVICE='launch-arm'; const CONTROLLER='launch-controller';
type Pack=any;
function releaseFor(pack:Pack, label:string, now:string):ExecutablePolicySpec {
 const config=executionConfigurationV1Schema.parse({schemaVersion:1,deviceIdentity:DEVICE,robotIdentity:H('e'),rosDistro:'jazzy',rmwImplementation:'rmw_fastrtps_cpp',jointState:{topic:'/joint_states',messageType:'sensor_msgs/msg/JointState'},controller:{name:CONTROLLER,followJointTrajectoryAction:'/launch/follow_joint_trajectory',actionType:'control_msgs/action/FollowJointTrajectory'},jointOrder:Array.from({length:7},(_,i)=>`joint_${i}`),adapter:{identity:pack.adapter_digest,version:label},observedAt:now});
 return executablePolicySpecSchema.parse({apiVersion:'realitywarden.io/v1alpha1',kind:'ExecutablePolicy',metadata:{name:`launch-${label}`,releaseId:`launch-${label}`,createdAt:now},model:{artifact:`launch/${label}`,sha256:pack.provenance.source_manifest_sha256.slice(7),framework:'custom',policyType:'recorded-final-action',codeRevision:pack.adapter_digest},actionContract:{representation:'program',dimension:1,jointOrder:[],units:{position:'none',velocity:'none'},normalizerSha256:H('b'),preprocessorSha256:pack.adapter_digest.slice(7),postprocessorSha256:H('d')},robot:{profileId:'libero-panda',profileSha256:H('e'),urdfSha256:H('f'),controllerType:CONTROLLER,controllerConfigSha256:H('1')},runtimePolicy:{policySha256:H('2'),maxStateAgeMs:1000,maxConfigurationAgeMs:60000,maxAttestationAgeMs:1000,requiredCapabilities:['controller.available','state.fresh'],failClosed:true},executionConfiguration:config,approvedConfigurationDigest:configurationDigest(config),evidence:{scenarioPackId:pack.suite_id,testReportSha256:H('3'),status:'approved',approvedBy:'competitive-harness',approvedAt:now},deployment:{allowedDeviceIds:[DEVICE],mode:'released',expiresAt:'2099-01-01T00:00:00.000Z'}});
}
function record(spec:ExecutablePolicySpec):ReleaseRecord { const h=executablePolicyHash(spec); return {releaseId:spec.metadata.releaseId,state:'released',executablePolicyHash:h,approvedIdentityHash:h,approvedConfigurationDigest:spec.approvedConfigurationDigest,approvedBy:'competitive-harness',approvedAt:spec.evidence.approvedAt}; }
async function runPack(pack:Pack,label:string){ const rows:any[]=[]; let dispatches=0; const now=new Date(); const spec=releaseFor(pack,label,now.toISOString()); const config=spec.executionConfiguration!; const releaseRecord=record(spec);
 for(const probe of pack.probes){ let mono=10000; const action={bytesHex:probe.action.bytes_hex,dtype:probe.action.dtype,shape:probe.action.shape}; const state={inputHash:probe.input_hash,cameras:probe.consumed.cameras,state:probe.consumed.state,cursor:probe.cursor}; const runtime=runtimeAttestationSchema.parse({schemaVersion:1,source:{identity:'launch-monitor',kind:'external-monitor',version:'1'},observedAt:now.toISOString(),continuityToken:`${label}:${probe.input_hash}`,availableCapabilities:['controller.available','state.fresh']}); const evidence:ExecutionEvidence[]=[]; const gate=new ReleaseExecutionGate<any,any,{accepted:true}>({async dispatch(){dispatches+=1;return {accepted:true};}},{append(entry){evidence.push(entry);}},async()=>({allowed:true,reason:'approved_new_intent',matchedRuleIds:['approved_probe']}),(v)=>sha256(canonicalJson(v)),async()=>releaseRecord,async()=>config,async()=>runtime,()=>mono); const request={release:spec,releaseRecord,deviceId:DEVICE,proposalId:`${label}:${probe.id}`,action,actionHash:sha256(canonicalJson(action)),state,stateObservedAt:now.toISOString(),controllerIdentity:CONTROLLER,executionConfiguration:config,runtimeAttestation:runtime,now}; const decision=await gate.evaluate(request); if(decision.status!=='allowed'){rows.push({probe_id:probe.id,decision:'DENY',reason:decision.reason});continue;} await gate.execute(decision.authorizedRequest); rows.push({probe_id:probe.id,decision:'ALLOW',reason:decision.reason,evidence_entries:evidence.length}); }
 return {spec,rows,dispatches}; }
async function main(){const reference=JSON.parse(readFileSync(process.env.LAUNCH_REFERENCE!,'utf8'));const candidate=JSON.parse(readFileSync(process.env.LAUNCH_CANDIDATE!,'utf8'));const left=await runPack(reference,'reference');const right=await runPack(candidate,'camera-swap');const diff=diffExecutablePolicies(left.spec,right.spec);const rows=[...left.rows.map((r:any)=>({...r,pack:'reference'})),...right.rows.map((r:any)=>({...r,pack:'camera_swap'}))];writeFileSync(process.env.RLSOK_OUTPUT!,JSON.stringify({product:'RLSOK',version:'1.5.12',commit:'5df8ce9a77349bf1e82e3b1f60aa0cd55138e034',api:['diffExecutablePolicies','ReleaseExecutionGate.evaluate','ReleaseExecutionGate.execute','runtimeAttestationSchema'],calls:rows.length,allowed:rows.filter((r:any)=>r.decision==='ALLOW').length,dispatches:left.dispatches+right.dispatches,declared_release_diff:diff,data_seen:'exact final action and generic state snapshot including cameras/state/cursor; release binds model, normalizer, preprocessor, postprocessor, action contract, config and runtime attestation',observed_vla_semantic_comparison_api_called:false,action_schema_accommodation:'RLSOK has no opaque/raw seven-float SmolVLA action contract. The harness uses its program contract only to instantiate the generic release gate; native action semantics are not validated.',rows},null,2)+'\n');}
void main();
"""


def _run_rlsok(reference: Path, candidate: Path, source: Path,
               node_modules: Path, node: str, output: Path) -> dict[str, Any]:
    if not (node_modules / "typescript/bin/tsc").is_file():
        raise RuntimeError("installed RLSOK comparison node_modules is unavailable")
    os.symlink(node_modules, source / "node_modules", target_is_directory=True)
    harness = source / "tests/competitive-boundary/launchCompetitive.test.ts"
    harness.parent.mkdir(parents=True, exist_ok=True)
    harness.write_text(RLSOK_HARNESS, encoding="utf-8")
    environment = {**os.environ, "LAUNCH_REFERENCE": str(reference.resolve()),
                   "LAUNCH_CANDIDATE": str(candidate.resolve()),
                   "RLSOK_OUTPUT": str(output.resolve())}
    process = subprocess.run([node, "scripts/run-rlsok.cjs", "--test",
                              "tests/competitive-boundary/launchCompetitive.test.ts"],
                             cwd=source, env=environment, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError("RLSOK harness failed: " + process.stderr[-2000:])
    return {"stdout": process.stdout.strip(), "stderr": process.stderr.strip(),
            "generated_harness_sha256": _sha(RLSOK_HARNESS.encode("utf-8"))}


def _orchestrate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("output already exists")
    reference = args.launch_result / "packs/reference.json"
    candidate = args.launch_result / "packs/camera_swap.json"
    launch_result = args.launch_result / "result.json"
    runner_raw = _snapshot(Path(__file__), RESULT_LIMIT)
    launch_raw = _snapshot(launch_result, RESULT_LIMIT)
    reference_raw = _snapshot(reference, PACK_LIMIT)
    candidate_raw = _snapshot(candidate, PACK_LIMIT)
    launch = _strict_json(launch_raw, str(launch_result))
    if not isinstance(launch, dict):
        raise ValueError("launch result must be an object")
    reference_value = _parse_pack(reference_raw, str(reference))
    candidate_value = _parse_pack(candidate_raw, str(candidate))
    selected = _validate_selected_case(launch, launch_raw, reference_value, reference_raw,
                                       candidate_value, candidate_raw)
    camera_row = selected["row"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-launch-competitive-") as temporary:
        temporary_path = Path(temporary)
        snapshots = temporary_path / "inputs"
        snapshots.mkdir(mode=0o700)
        private_reference = snapshots / "reference.json"
        private_candidate = snapshots / "camera_swap.json"
        private_reference.write_bytes(reference_raw)
        private_candidate.write_bytes(candidate_raw)
        private_runner = temporary_path / "launch_competitive.py"
        private_runner.write_bytes(runner_raw)
        kine_source = temporary_path / "kinegrant-source"
        rlsok_source = temporary_path / "rlsok-source"
        kine_archive = _git_archive(args.kinegrant_repo, KINEGRANT_COMMIT, kine_source)
        rlsok_archive = _git_archive(args.rlsok_repo, RLSOK_COMMIT, rlsok_source)
        kinegrant_result = temporary_path / "kinegrant.json"
        kine_env = {**os.environ, "PYTHONPATH": str(kine_source / "src")}
        kine = subprocess.run([
            args.python, str(private_runner), "_kinegrant",
            "--reference", str(private_reference), "--candidate", str(private_candidate),
            "--source", str(kine_source), "--output", str(kinegrant_result),
        ], env=kine_env, capture_output=True, text=True)
        if kine.returncode:
            raise RuntimeError("KineGrant harness failed: " + kine.stderr[-2000:])
        rlsok_result = temporary_path / "rlsok.json"
        node_receipt = _run_rlsok(private_reference, private_candidate, rlsok_source,
                                  args.node_modules, args.node, rlsok_result)
        kine_value = _strict_json(_snapshot(kinegrant_result, RESULT_LIMIT), "KineGrant result")
        rlsok_value = _strict_json(_snapshot(rlsok_result, RESULT_LIMIT), "RLSOK result")
        node_version = subprocess.run([args.node, "--version"], check=True,
                                      capture_output=True, text=True).stdout.strip()
        result = {
            "schema": SCHEMA,
            "runner_sha256": _sha(runner_raw),
            "source": {
                "launch_result": str(launch_result),
                "launch_result_sha256": selected["launch_result_sha256"],
                "reference_pack_sha256": selected["reference_pack_sha256"],
                "camera_swap_pack_sha256": selected["candidate_pack_sha256"],
                "suite_id": reference_value["suite_id"],
                "probe_ids": [probe["id"] for probe in reference_value["probes"]],
                "input_hashes": [probe["input_hash"] for probe in reference_value["probes"]],
                "probe_count_per_pack": len(reference_value["probes"]),
                "snapshot_contract": "bounded O_NOFOLLOW regular-file snapshots; exact bytes supplied to both products",
            },
            "sentinel_observation": {
                "verdict": camera_row["report"]["verdict"],
                "first_issue": camera_row["report"]["issues"][0],
                "downstream_calls": camera_row["writer"]["downstream_calls"],
            },
            "kinegrant": {**kine_value, "source_snapshot": kine_archive,
                           "dependency": _python_dependency_receipt(args.python),
                           "pyproject_sha256": _sha((kine_source / "pyproject.toml").read_bytes())},
            "rlsok": {**rlsok_value, "node": node_version,
                      "source_snapshot": rlsok_archive,
                      "dependency": _node_dependency_receipt(args.node_modules),
                      "package_lock_sha256": _sha((rlsok_source / "package-lock.json").read_bytes()),
                      "source_files": _source_hashes(rlsok_source, [
                          "packages/core/exec-spec.ts", "packages/core/execution-gate.ts",
                          "packages/core/execution-configuration.ts",
                          "packages/core/runtime-attestation.ts",
                      ]),
                      "generated_harness_sha256": node_receipt["generated_harness_sha256"],
                      "controlled_additions": ["bound node_modules symlink", "generated TypeScript harness"],
                      "runner_stdout": node_receipt["stdout"]},
            "interpretation": {
                "authorized_new_intent": "Both products accepted both packs when each request/release was independently approved.",
                "not_a_failure": "These accepts are correct authorization outcomes, not safety misses.",
                "product_difference": "Sentinel additionally compares the installed adapter's captured VLA bindings against the retained reference and emits the first mismatch plus a signed offline reproducer.",
                "falsifier": "A custom KineGrant policy or RLSOK ActionPolicy can compare the same retained fields and deny the camera swap; this experiment claims a packaged workflow difference, not exclusive expressiveness.",
            },
        }
        args.output.write_bytes(_json_bytes(result))
        print(json.dumps({"output": str(args.output), "sha256": _sha(args.output.read_bytes()),
                          "kinegrant_allowed": kine_value["allowed"],
                          "rlsok_allowed": rlsok_value["allowed"],
                          "sentinel_verdict": camera_row["report"]["verdict"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--launch-result", type=Path, required=True)
    run.add_argument("--kinegrant-repo", type=Path, required=True)
    run.add_argument("--rlsok-repo", type=Path, required=True)
    run.add_argument("--node-modules", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--node", default="node")
    run.add_argument("--output", type=Path, required=True)
    internal = sub.add_parser("_kinegrant")
    internal.add_argument("--reference", type=Path, required=True)
    internal.add_argument("--candidate", type=Path, required=True)
    internal.add_argument("--source", type=Path, required=True)
    internal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "_kinegrant":
        _run_kinegrant(args.reference, args.candidate, args.source, args.output)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
