"""Strict, non-interactive SSH transport for the self-contained physics experiment.

Authentication is delegated to the user's existing OpenSSH configuration, agent,
keys, and known_hosts database.  This module never discovers credentials, accepts
passwords, or weakens host-key checking.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

from .contracts import canonical_json
from .evidence import verify_bundle


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_FILES = 20_000
MAX_FILE_BYTES = 96 * 1024 * 1024
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PYTHON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRUST_SCOPE = (
    "SSH known-host authentication authenticated collection at collection time; "
    "the unsigned local receipt requires preserved custody and is not a transferable attestation; "
    "experiment keys are self-contained demo integrity keys, not customer PKI"
)
_SSH_OPTIONS = (
    "-o", "StrictHostKeyChecking=yes",
    "-o", "BatchMode=yes",
    "-o", "PasswordAuthentication=no",
    "-o", "KbdInteractiveAuthentication=no",
    "-o", "NumberOfPasswordPrompts=0",
    "-o", "ConnectTimeout=15",
    "-T",
)


class RemoteExperimentError(RuntimeError):
    pass


def _strict_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def _validate_inputs(host, out, seed, friction, render, python):
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        raise ValueError("host must be a bounded OpenSSH host alias")
    if not isinstance(python, str) or not _PYTHON.fullmatch(python) or Path(python).name != python:
        raise ValueError("python must be a bounded executable basename")
    if type(seed) is not int or not 0 <= seed < 1_000_000:
        raise ValueError("seed must be an integer in [0,1000000)")
    if isinstance(friction, bool) or not isinstance(friction, (int, float)):
        raise ValueError("friction must be finite")
    friction = float(friction)
    if not math.isfinite(friction) or not 0.01 <= friction <= 1.0:
        raise ValueError("friction must be in [0.01,1.0]")
    if type(render) is not bool:
        raise ValueError("render must be bool")
    output = Path(out).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("remote experiment output directory must be empty")
    return output, friction


def _run(args, *, input_bytes=None, timeout=30):
    return subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _host_pin(host: str) -> dict:
    config = _run(["ssh", "-G", host], timeout=15)
    if config.returncode != 0:
        raise RemoteExperimentError("unable to resolve the OpenSSH host alias")
    values = {}
    for raw in config.stdout.decode("utf-8", "replace").splitlines():
        key, _, value = raw.partition(" ")
        if key in {"hostname", "port", "hostkeyalias", "userknownhostsfile"} and value:
            values[key] = value.strip()
    hostname = values.get("hostname")
    port = values.get("port", "22")
    if not hostname or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise RemoteExperimentError("OpenSSH resolved an invalid host or port")
    lookup = values.get("hostkeyalias") or (hostname if port == "22" else f"[{hostname}]:{port}")
    known_outputs = []
    known_files = values.get("userknownhostsfile", "").split()
    commands = (["ssh-keygen", "-F", lookup, "-f", str(Path(name).expanduser())]
                for name in known_files if name.lower() != "none")
    for command in commands:
        found = _run(command, timeout=15)
        if found.returncode == 0 and found.stdout.strip():
            known_outputs.append(found.stdout)
    if not known_outputs:
        known = _run(["ssh-keygen", "-F", lookup], timeout=15)
        if known.returncode == 0 and known.stdout.strip():
            known_outputs.append(known.stdout)
    if not known_outputs:
        raise RemoteExperimentError("strict host-key checking requires an existing known_hosts entry")
    fingerprints = _run(["ssh-keygen", "-lf", "-"], input_bytes=b"".join(known_outputs), timeout=15)
    if fingerprints.returncode != 0:
        raise RemoteExperimentError("unable to fingerprint the existing known_hosts entry")
    found = sorted(
        {part for line in fingerprints.stdout.decode("utf-8", "replace").splitlines()
         for part in line.split() if part.startswith("SHA256:")}
    )
    if not found:
        raise RemoteExperimentError("known_hosts entry has no usable fingerprint")
    return {"host_alias": host, "resolved_host": hostname, "port": int(port),
            "known_host_fingerprints": found}


def _source_files(root: Path):
    allowed = []
    for base, suffixes in ((root / "src", {".py", ".html", ".css", ".js", ".md"}),
                           (root / "tests", {".py"})):
        if not base.is_dir() or base.is_symlink():
            raise RemoteExperimentError(f"missing public source directory: {base.name}")
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink() or not path.is_file() or path.suffix not in suffixes:
                continue
            allowed.append(path)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RemoteExperimentError(f"missing public source file: {name}")
        allowed.append(path)
    return tuple(sorted(set(allowed)))


def _source_archive(root: Path) -> tuple[bytes, str]:
    files = _source_files(root)
    total = sum(path.stat().st_size for path in files)
    if len(files) > MAX_FILES or total > MAX_EXTRACTED_BYTES:
        raise RemoteExperimentError("public source package exceeds transport bounds")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            relative = path.relative_to(root)
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    data = stream.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise RemoteExperimentError("compressed public source package exceeds transport bounds")
    return data, hashlib.sha256(data).hexdigest()


def _expected_source_manifest_sha256(root: Path) -> str:
    source_paths = list((root / "src" / "sentinel_evc").rglob("*.py"))
    source_paths += [root / "pyproject.toml"]
    source_paths += list((root / "tests").glob("test_*.py"))
    if any(not path.is_file() or path.is_symlink() for path in source_paths):
        raise RemoteExperimentError("source manifest contains a missing or unsafe file")
    manifest = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_paths)
    }
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


_REMOTE_BOOTSTRAP = r'''
import hashlib, json, os, pathlib, subprocess, sys, tarfile

MAX_ARCHIVE=256*1024*1024; MAX_TOTAL=256*1024*1024; MAX_FILE=96*1024*1024; MAX_FILES=20000
job_id, seed, friction, render, expected_package = sys.argv[1:]
if not job_id.startswith('job-') or len(job_id) != 36: raise SystemExit('invalid job id')
root = pathlib.Path.home()/'.local'/'share'/'sentinel-evc'/'ssh-jobs'/job_id
root.mkdir(parents=True, exist_ok=False)
source=root/'source'; output=root/'result'; logs=root/'logs'; source.mkdir(); logs.mkdir()
package=sys.stdin.buffer.read(MAX_ARCHIVE+1)
if len(package)>MAX_ARCHIVE or hashlib.sha256(package).hexdigest()!=expected_package: raise SystemExit('invalid source package')
(root/'source.tar.gz').write_bytes(package)
total=count=0
with tarfile.open(fileobj=__import__('io').BytesIO(package), mode='r:gz') as tf:
  for member in tf.getmembers():
    parts=pathlib.PurePosixPath(member.name).parts
    if member.name.startswith('/') or not parts or '..' in parts or member.issym() or member.islnk() or not (member.isdir() or member.isfile()): raise SystemExit('unsafe source archive')
    count+=1; total+=member.size
    if count>MAX_FILES or member.size>MAX_FILE or total>MAX_TOTAL: raise SystemExit('source archive exceeds bounds')
    target=source.joinpath(*parts)
    if member.isdir(): target.mkdir(parents=True,exist_ok=True); continue
    target.parent.mkdir(parents=True,exist_ok=True)
    incoming=tf.extractfile(member)
    with target.open('xb') as out: out.write(incoming.read())

venv=root/'venv'; subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
py=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'
if render=='1': env['MUJOCO_GL']='egl'
def call(name,args):
  with (logs/name).open('wb') as log:
    return subprocess.run(args,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
if call('install.log',[str(py),'-m','pip','install','.[physics,test]'])!=0: raise SystemExit('dependency installation failed')
env['PYTHONPATH']=str(source/'src')
if call('tests.log',[str(py),'-m','pytest','-q'])!=0: raise SystemExit('remote tests failed')
cmd=[str(py),'-m','sentinel_evc','physics','--out',str(output),'--seed',seed,'--friction',friction]
if render=='1': cmd.append('--render')
experiment_rc=call('experiment.log',cmd)
complete_path=output/'COMPLETE.json'
if experiment_rc not in (0,3) or not complete_path.is_file(): raise SystemExit('physics experiment incomplete')

verify_program="""
import pathlib, sys
from sentinel_evc.ssh_experiment import _verify_result_tree
_verify_result_tree(pathlib.Path(sys.argv[1]))
"""
if call('verification.log',[str(py),'-c',verify_program,str(output)])!=0: raise SystemExit('experiment verification failed')
complete=json.loads(complete_path.read_text('utf-8'))

transport=output/'transport-logs'; transport.mkdir()
for log in logs.iterdir(): (transport/log.name).write_bytes(log.read_bytes())
archive=root/'result.tar.gz'; total=count=0
with tarfile.open(archive,'w:gz') as tf:
  for path in sorted(output.rglob('*')):
    if path.is_symlink(): raise SystemExit('symlink result refused')
    if not path.is_file(): continue
    size=path.stat().st_size; total+=size; count+=1
    if count>MAX_FILES or size>MAX_FILE or total>MAX_TOTAL: raise SystemExit('result exceeds bounds')
    tf.add(path,arcname='result/'+path.relative_to(output).as_posix(),recursive=False)
if archive.stat().st_size>MAX_ARCHIVE: raise SystemExit('compressed result exceeds bounds')
marker={'schema':'sentinel-ssh-complete-v1','job_id':job_id,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_size':archive.stat().st_size,
        'package_sha256':expected_package,'experiment_id':complete['experiment_id'],'index_tip':complete['index_tip'],'index_public_key_sha256':complete['public_key_sha256'],
        'source_manifest_sha256':complete['source_manifest_sha256'],'experiment_exit_code':experiment_rc}
print(json.dumps(marker,sort_keys=True,separators=(',',':')))
'''


def _remote_command(python: str, job_id: str, seed: int, friction: float,
                    render: bool, package_sha256: str) -> str:
    encoded = base64.b64encode(_REMOTE_BOOTSTRAP.encode("utf-8")).decode("ascii")
    program = f"import base64;exec(compile(base64.b64decode({encoded!r}),'<sentinel-ssh-bootstrap>','exec'))"
    args = (python, "-c", program, job_id, str(seed), format(friction, ".12g"),
            "1" if render else "0", package_sha256)
    return " ".join(shlex.quote(value) for value in args)


def _run_remote_job(host, command, package, timeout):
    result = _run(["ssh", *_SSH_OPTIONS, host, command], input_bytes=package, timeout=timeout)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[-4000:]
        raise RemoteExperimentError("remote physics job failed: " + detail)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RemoteExperimentError("remote job did not return one completion marker")
    marker = _strict_json(lines[0])
    if not isinstance(marker, dict) or marker.get("schema") != "sentinel-ssh-complete-v1":
        raise RemoteExperimentError("invalid remote completion marker")
    return marker


def _fetch_remote_archive(host: str, python: str, job_id: str, timeout: int) -> bytes:
    relative = f".local/share/sentinel-evc/ssh-jobs/{job_id}/result.tar.gz"
    program = ("import pathlib,sys; p=pathlib.Path.home()/" + repr(relative) +
               "; n=p.stat().st_size; "
               f"assert 0<n<={MAX_ARCHIVE_BYTES}; sys.stdout.buffer.write(p.read_bytes())")
    command = " ".join(shlex.quote(value) for value in (python, "-c", program))
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["ssh", *_SSH_OPTIONS, host, command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        assert process.stdout is not None
        chunks, size = [], 0
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    raise RemoteExperimentError("remote result download timed out")
                ready = selector.select(min(remaining, 1.0))
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = process.stdout.read1(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    process.kill()
                    raise RemoteExperimentError("remote result archive exceeds transfer bound")
                chunks.append(chunk)
            remaining = deadline - time.monotonic()
            return_code = process.wait(timeout=max(remaining, 0.001))
        except Exception:
            process.kill()
            process.wait()
            raise
        finally:
            selector.close()
        if return_code != 0:
            errors.seek(0)
            raise RemoteExperimentError(
                "remote result download failed: " + errors.read(4000).decode("utf-8", "replace")
            )
    return b"".join(chunks)


def _safe_extract_result(data: bytes, destination: Path) -> None:
    if not data or len(data) > MAX_ARCHIVE_BYTES:
        raise RemoteExperimentError("invalid result archive size")
    total = count = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or not path.parts or path.parts[0] != "result"
                    or ".." in path.parts or member.issym() or member.islnk()
                    or not (member.isdir() or member.isfile())):
                raise RemoteExperimentError("unsafe result archive member")
            count += 1
            total += member.size
            if count > MAX_FILES or member.size > MAX_FILE_BYTES or total > MAX_EXTRACTED_BYTES:
                raise RemoteExperimentError("result archive exceeds extraction bounds")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if not relative.parts:
                continue
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise RemoteExperimentError("missing result archive payload")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_trial_ids(seed: int) -> set[str]:
    expected = {
        f"physics-{seed}-integrated",
        f"physics-{seed}-revoke",
        f"physics-{seed}-geometry_mutation",
        f"physics-{seed}-friction_mutation",
        f"physics-{seed}-replay",
    }
    expected.update(
        f"physics-{seed}-{step}us-{duration}"
        for step in (2000, 1000, 500, 250, 125)
        for duration in (6, 9, 12, 16)
    )
    return expected


def _tree_digest(root: Path) -> str:
    rows, total, count = [], 0, 0
    for path in sorted(root.rglob("*")):
        if path.name == "REMOTE_RECEIPT.json":
            continue
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RemoteExperimentError("unsafe local result member")
        if path.is_file():
            size = path.stat().st_size
            total += size
            count += 1
            if count > MAX_FILES or size > MAX_FILE_BYTES or total > MAX_EXTRACTED_BYTES:
                raise RemoteExperimentError("local result exceeds verification bounds")
            rows.append({"path": path.relative_to(root).as_posix(), "size": size, "sha256": _sha256(path)})
    return hashlib.sha256(canonical_json(rows)).hexdigest()


def _verify_result_tree(directory: Path, receipt: dict | None = None) -> dict:
    complete = _strict_json((directory / "COMPLETE.json").read_bytes())
    index = directory / "experiment-index"
    bundle = index / "bundle"
    public = index / "anchors" / "demo.public"
    required = ("experiment_id", "index_tip", "public_key_sha256", "summary_sha256", "source_manifest_sha256")
    if not isinstance(complete, dict) or any(not isinstance(complete.get(key), str) for key in required):
        raise RemoteExperimentError("invalid experiment completion marker")
    if not _DIGEST.fullmatch(complete["index_tip"]):
        raise RemoteExperimentError("invalid experiment index tip")
    if not _HEX64.fullmatch(complete["public_key_sha256"]):
        raise RemoteExperimentError("invalid experiment public-key digest")
    if _sha256(public) != complete["public_key_sha256"]:
        raise RemoteExperimentError("experiment index public key changed")
    okay, message = verify_bundle(
        str(bundle), str(public), complete["experiment_id"], expected_tip=complete["index_tip"]
    )
    if not okay:
        raise RemoteExperimentError("experiment index failed verification: " + message)
    index_manifest = _strict_json((bundle / "manifest.json").read_bytes())
    required_index_files = {"events.jsonl", "summary.json", "completion.json", "source_manifest.json",
                            "environment.json", "trial_anchors.json", "job_config.json"}
    if not required_index_files.issubset(set(index_manifest.get("files", {}))):
        raise RemoteExperimentError("experiment index omits required signed artifacts")

    signed_completion = _strict_json((bundle / "completion.json").read_bytes())
    summary_bytes = (bundle / "summary.json").read_bytes()
    source_bytes = (bundle / "source_manifest.json").read_bytes()
    if (_sha256(bundle / "summary.json") != signed_completion.get("summary_sha256")
            or _sha256(bundle / "source_manifest.json") != signed_completion.get("source_manifest_sha256")
            or signed_completion.get("experiment_id") != complete["experiment_id"]):
        raise RemoteExperimentError("signed completion does not bind summary/source")
    if (signed_completion["summary_sha256"] != complete["summary_sha256"]
            or signed_completion["source_manifest_sha256"] != complete["source_manifest_sha256"]):
        raise RemoteExperimentError("completion marker disagrees with signed index")
    root_summary = directory / "summary.json"
    if (not root_summary.is_file() or root_summary.is_symlink()
            or root_summary.read_bytes() != summary_bytes
            or _sha256(root_summary) != signed_completion["summary_sha256"]):
        raise RemoteExperimentError("top-level summary differs from signed summary")
    summary = _strict_json(summary_bytes)
    source_manifest = _strict_json(source_bytes)
    job_config = _strict_json((bundle / "job_config.json").read_bytes())
    anchors = _strict_json((bundle / "trial_anchors.json").read_bytes())
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise RemoteExperimentError("signed source manifest is empty")
    if not isinstance(job_config, dict) or not isinstance(anchors, dict):
        raise RemoteExperimentError("signed experiment config/trial index is incomplete")
    seed = job_config.get("seed")
    if type(seed) is not int or set(anchors) != _expected_trial_ids(seed):
        raise RemoteExperimentError("signed experiment has an incomplete or unexpected trial set")

    for run_id, anchor in anchors.items():
        if not isinstance(run_id, str) or not _SAFE_COMPONENT.fullmatch(run_id) or not isinstance(anchor, dict):
            raise RemoteExperimentError("unsafe trial index entry")
        trial = directory / run_id
        trial_bundle = trial / "bundle"
        trial_public = trial / "anchors" / "demo.public"
        if (not _HEX64.fullmatch(str(anchor.get("public_key_sha256", "")))
                or not _HEX64.fullmatch(str(anchor.get("manifest_sha256", "")))
                or not _DIGEST.fullmatch(str(anchor.get("tip_hash", "")))
                or anchor.get("run_id") != run_id):
            raise RemoteExperimentError("invalid trial anchor")
        if (not trial_public.is_file() or trial_public.is_symlink()
                or not (trial_bundle / "manifest.json").is_file()
                or (trial_bundle / "manifest.json").is_symlink()):
            raise RemoteExperimentError("trial bundle or public key is missing or unsafe")
        if (_sha256(trial_public) != anchor["public_key_sha256"]
                or _sha256(trial_bundle / "manifest.json") != anchor["manifest_sha256"]):
            raise RemoteExperimentError("trial anchor digest mismatch")
        trial_manifest = _strict_json((trial_bundle / "manifest.json").read_bytes())
        required_trial_files = {"events.jsonl", "result.json", "root.json", "plan.json",
                                "physics.xml", "trace.json", "config.json"}
        if not required_trial_files.issubset(set(trial_manifest.get("files", {}))):
            raise RemoteExperimentError("trial bundle omits required signed artifacts")
        valid, detail = verify_bundle(
            str(trial_bundle), str(trial_public), run_id, expected_tip=anchor["tip_hash"]
        )
        if not valid:
            raise RemoteExperimentError(f"trial {run_id} failed verification: {detail}")

    if receipt is not None:
        marker = receipt.get("remote_marker")
        if not isinstance(marker, dict):
            raise RemoteExperimentError("receipt has no remote completion marker")
        bindings = {
            "experiment_id": complete["experiment_id"],
            "index_tip": complete["index_tip"],
            "index_public_key_sha256": complete["public_key_sha256"],
            "source_manifest_sha256": complete["source_manifest_sha256"],
        }
        if any(marker.get(key) != value for key, value in bindings.items()):
            raise RemoteExperimentError("receipt does not bind the signed experiment")
        expected_config = receipt.get("requested_config")
        if not isinstance(expected_config, dict) or any(job_config.get(key) != expected_config.get(key) for key in ("seed", "friction", "render")):
            raise RemoteExperimentError("signed job config differs from requested config")
        if signed_completion["source_manifest_sha256"] != receipt.get("expected_source_manifest_sha256"):
            raise RemoteExperimentError("signed source manifest differs from the submitted source tree")
    return {"summary": summary, "job_config": job_config, "trial_count": len(anchors), "message": message}


def verify_physics_experiment_result(out) -> dict:
    """Verify a downloaded result against its SSH-channel receipt and all signed trials."""
    directory = Path(out).expanduser().resolve()
    receipt_path = directory / "REMOTE_RECEIPT.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RemoteExperimentError("no valid remote receipt")
    receipt = _strict_json(receipt_path.read_bytes())
    if not isinstance(receipt, dict) or receipt.get("schema") != "sentinel-ssh-receipt-v1":
        raise RemoteExperimentError("invalid remote receipt")
    host = receipt.get("host")
    if (not isinstance(host, dict) or not _HOST.fullmatch(str(host.get("host_alias", "")))
            or not isinstance(host.get("known_host_fingerprints"), list)
            or not host["known_host_fingerprints"]):
        raise RemoteExperimentError("invalid known-host receipt")
    if _tree_digest(directory) != receipt.get("result_tree_sha256"):
        raise RemoteExperimentError("downloaded result tree changed after receipt creation")
    verified = _verify_result_tree(directory, receipt)
    return {
        "ok": True,
        "experiment_id": receipt["remote_marker"]["experiment_id"],
        "trial_count": verified["trial_count"],
        "acceptance": verified["summary"].get("acceptance", {}),
        "infrastructure_gates_pass": bool(verified["summary"].get("infrastructure_gates_pass")),
        "host_alias": receipt["host"]["host_alias"],
        "known_host_fingerprints": list(receipt["host"]["known_host_fingerprints"]),
        "trust": _TRUST_SCOPE,
    }


def run_remote_physics(
    host,
    out,
    *,
    seed=7,
    friction=0.35,
    render=False,
    python="python3",
    timeout=7200,
) -> dict:
    """Run and retrieve one isolated remote physics experiment over strict OpenSSH."""
    output, friction = _validate_inputs(host, out, seed, friction, render, python)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 60 <= timeout <= 86_400:
        raise ValueError("timeout must be an integer from 60 to 86400 seconds")
    pin = _host_pin(host)
    root = Path(__file__).resolve().parents[2]
    package, package_sha256 = _source_archive(root)
    expected_source_manifest_sha256 = _expected_source_manifest_sha256(root)
    job_id = "job-" + uuid.uuid4().hex
    command = _remote_command(python, job_id, seed, friction, render, package_sha256)
    marker = _run_remote_job(host, command, package, timeout)
    expected_marker = {
        "job_id": job_id,
        "package_sha256": package_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
    }
    if any(marker.get(key) != value for key, value in expected_marker.items()):
        raise RemoteExperimentError("remote completion marker is not bound to this job/package")
    if (not _HEX64.fullmatch(str(marker.get("archive_sha256", "")))
            or type(marker.get("archive_size")) is not int
            or not 0 < marker["archive_size"] <= MAX_ARCHIVE_BYTES):
        raise RemoteExperimentError("remote completion marker has invalid archive metadata")
    archive = _fetch_remote_archive(host, python, job_id, timeout)
    if len(archive) != marker["archive_size"] or hashlib.sha256(archive).hexdigest() != marker["archive_sha256"]:
        raise RemoteExperimentError("downloaded archive does not match the remote completion marker")

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / ("." + output.name + "." + job_id + ".tmp")
    if staging.exists():
        raise RemoteExperimentError("local staging path already exists")
    try:
        _safe_extract_result(archive, staging)
        _verify_result_tree(staging)
        receipt = {
            "schema": "sentinel-ssh-receipt-v1",
            "host": pin,
            "remote_marker": marker,
            "requested_config": {"seed": seed, "friction": friction, "render": render},
            "expected_source_manifest_sha256": expected_source_manifest_sha256,
            "python_basename": python,
            "ssh_policy": {"strict_host_key_checking": True, "batch_mode": True, "password_authentication": False},
            "trust_scope": _TRUST_SCOPE,
            "result_tree_sha256": _tree_digest(staging),
        }
        _verify_result_tree(staging, receipt)
        (staging / "REMOTE_RECEIPT.json").write_bytes(canonical_json(receipt))
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_physics_experiment_result(output)
