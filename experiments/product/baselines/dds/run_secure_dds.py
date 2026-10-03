#!/usr/bin/env python3
"""Build experiment credentials and run a real CycloneDDS Security transport probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
TMP = Path("/private/tmp/sentinel-dds-20261003")
OPENSSL = TMP / "openssl-install/bin/openssl"
OPENSSL_LIB = TMP / "openssl-install/lib"
CYCLONE = TMP / "cyclonedds-install"
ENDPOINT = TMP / "app-build/sentinel_dds_endpoint"
DOMAIN_ID = 173
ALLOWED_TOPIC = "SentinelAction"
FORBIDDEN_TOPIC = "SentinelForbidden"


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 60,
        check: bool = True, stdout_path: Path | None = None,
        stderr_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if command and Path(command[0]) == OPENSSL:
        env = dict(os.environ if env is None else env)
        env["OPENSSL_CONF"] = str(TMP / "openssl-3.5.8/apps/openssl.cnf")
        env["DYLD_LIBRARY_PATH"] = str(OPENSSL_LIB)
    start = time.monotonic_ns()
    stdout = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr = stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.PIPE
    try:
        completed = subprocess.run(
            command, text=True, stdout=stdout, stderr=stderr, env=env,
            timeout=timeout, check=False,
        )
    finally:
        if stdout_path:
            stdout.close()
        if stderr_path:
            stderr.close()
    completed.wall_ns = time.monotonic_ns() - start  # type: ignore[attr-defined]
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str, mode: int = 0o644) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def subject_for(cert: Path) -> str:
    completed = run([str(OPENSSL), "x509", "-in", str(cert), "-noout", "-subject",
                     "-nameopt", "RFC2253"])
    return completed.stdout.strip().removeprefix("subject=")


def create_identity(name: str, common_name: str, ca_cert: Path, ca_key: Path) -> dict[str, Path]:
    key = CONFIG / f"{name}.key.pem"
    csr = CONFIG / f"{name}.csr.pem"
    cert = CONFIG / f"{name}.cert.pem"
    serial = CONFIG / f"{ca_cert.stem}.srl"
    run([str(OPENSSL), "genrsa", "-out", str(key), "2048"])
    run([str(OPENSSL), "req", "-new", "-key", str(key), "-out", str(csr),
         "-subj", f"/C=CN/O=Sentinel EVC/OU=DDS Boundary/CN={common_name}"])
    command = [str(OPENSSL), "x509", "-req", "-CA", str(ca_cert), "-CAkey", str(ca_key),
               "-days", "3650", "-in", str(csr), "-out", str(cert)]
    if serial.exists():
        command += ["-CAserial", str(serial)]
    else:
        command += ["-CAcreateserial", "-CAserial", str(serial)]
    run(command)
    key.chmod(0o600)
    return {"key": key, "cert": cert}


def permissions_xml(name: str, subject: str, publishes: list[str], subscribes: list[str]) -> str:
    publish = "".join(f"<topic>{topic}</topic>" for topic in publishes)
    subscribe = "".join(f"<topic>{topic}</topic>" for topic in subscribes)
    sections = []
    if publish:
        sections.append(f"<publish><topics>{publish}</topics><partitions><partition>*</partition></partitions></publish>")
    if subscribe:
        sections.append(f"<subscribe><topics>{subscribe}</topics><partitions><partition>*</partition></partitions></subscribe>")
    return f'''<?xml version="1.0" encoding="utf-8"?>
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="https://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_permissions.xsd">
 <permissions><grant name="{name}"><subject_name>{subject}</subject_name>
 <validity><not_before>2026-01-01T00:00:00</not_before><not_after>2036-01-01T00:00:00</not_after></validity>
 <allow_rule><domains><id>{DOMAIN_ID}</id></domains>{''.join(sections)}</allow_rule>
 <default>DENY</default></grant></permissions>
</dds>
'''


def config_xml(name: str, identity_ca: Path, identity: dict[str, Path], permissions: Path) -> str:
    auth = CYCLONE / "lib/libdds_security_auth.dylib"
    access = CYCLONE / "lib/libdds_security_ac.dylib"
    crypto = CYCLONE / "lib/libdds_security_crypto.dylib"
    trace = RAW / f"cyclonedds-{name}.${{CYCLONEDDS_PID}}.log"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config"><Domain id="{DOMAIN_ID}">
 <General><Interfaces><NetworkInterface name="lo0" multicast="true"/></Interfaces><AllowMulticast>true</AllowMulticast></General>
 <Tracing><Verbosity>finest</Verbosity><OutputFile>{trace}</OutputFile></Tracing>
 <Security>
  <Authentication><Library initFunction="init_authentication" finalizeFunction="finalize_authentication" path="{auth}"/>
   <IdentityCA>file:{identity_ca}</IdentityCA><IdentityCertificate>file:{identity['cert']}</IdentityCertificate><PrivateKey>file:{identity['key']}</PrivateKey></Authentication>
  <Cryptographic><Library initFunction="init_crypto" finalizeFunction="finalize_crypto" path="{crypto}"/></Cryptographic>
  <AccessControl><Library initFunction="init_access_control" finalizeFunction="finalize_access_control" path="{access}"/>
   <PermissionsCA>file:{CONFIG / 'permissions-ca.cert.pem'}</PermissionsCA><Governance>file:{CONFIG / 'governance.p7s'}</Governance><Permissions>file:{permissions}</Permissions></AccessControl>
 </Security>
</Domain></CycloneDDS>
'''


def bootstrap() -> dict[str, object]:
    for directory in (CONFIG, RAW, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    if not OPENSSL.exists() or not ENDPOINT.exists():
        raise RuntimeError("isolated OpenSSL/CycloneDDS endpoint build is missing")

    id_key = CONFIG / "identity-ca.key.pem"
    id_cert = CONFIG / "identity-ca.cert.pem"
    rogue_key = CONFIG / "rogue-identity-ca.key.pem"
    rogue_cert = CONFIG / "rogue-identity-ca.cert.pem"
    perm_key = CONFIG / "permissions-ca.key.pem"
    perm_cert = CONFIG / "permissions-ca.cert.pem"
    for key, cert, cn in (
        (id_key, id_cert, "Sentinel Identity CA"),
        (rogue_key, rogue_cert, "Untrusted Identity CA"),
        (perm_key, perm_cert, "Sentinel Permissions CA"),
    ):
        run([str(OPENSSL), "genrsa", "-out", str(key), "2048"])
        run([str(OPENSSL), "req", "-x509", "-key", str(key), "-out", str(cert),
             "-days", "3650", "-subj", f"/C=CN/O=Sentinel EVC/OU=DDS Boundary/CN={cn}"])
        key.chmod(0o600)

    identities = {
        "receiver": create_identity("receiver", "Sentinel Receiver", id_cert, id_key),
        "allowed-publisher": create_identity("allowed-publisher", "Allowed Publisher", id_cert, id_key),
        "denied-publisher": create_identity("denied-publisher", "Denied Publisher", id_cert, id_key),
        "rogue-publisher": create_identity("rogue-publisher", "Rogue Publisher", rogue_cert, rogue_key),
    }
    governance = CONFIG / "governance.xml"
    write(governance, f'''<?xml version="1.0" encoding="utf-8"?>
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="https://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_governance.xsd">
 <domain_access_rules><domain_rule><domains><id>{DOMAIN_ID}</id></domains>
 <allow_unauthenticated_participants>false</allow_unauthenticated_participants><enable_join_access_control>true</enable_join_access_control>
 <discovery_protection_kind>ENCRYPT</discovery_protection_kind><liveliness_protection_kind>ENCRYPT</liveliness_protection_kind><rtps_protection_kind>ENCRYPT</rtps_protection_kind>
 <topic_access_rules><topic_rule><topic_expression>*</topic_expression><enable_discovery_protection>true</enable_discovery_protection>
 <enable_liveliness_protection>true</enable_liveliness_protection><enable_read_access_control>true</enable_read_access_control><enable_write_access_control>true</enable_write_access_control>
 <metadata_protection_kind>ENCRYPT</metadata_protection_kind><data_protection_kind>ENCRYPT</data_protection_kind></topic_rule></topic_access_rules>
 </domain_rule></domain_access_rules></dds>
''')
    run([str(OPENSSL), "smime", "-sign", "-in", str(governance), "-text",
         "-out", str(CONFIG / "governance.p7s"), "-signer", str(perm_cert), "-inkey", str(perm_key)])

    grants = {
        "receiver": ([], [ALLOWED_TOPIC]),
        "allowed-publisher": ([ALLOWED_TOPIC], []),
        "denied-publisher": (["OtherTopic"], []),
        "rogue-publisher": ([ALLOWED_TOPIC], []),
    }
    for name, identity in identities.items():
        publishes, subscribes = grants[name]
        xml = CONFIG / f"permissions-{name}.xml"
        signed = CONFIG / f"permissions-{name}.p7s"
        write(xml, permissions_xml(name, subject_for(identity["cert"]), publishes, subscribes))
        run([str(OPENSSL), "smime", "-sign", "-in", str(xml), "-text", "-out", str(signed),
             "-signer", str(perm_cert), "-inkey", str(perm_key)])
        identity_ca = rogue_cert if name == "rogue-publisher" else id_cert
        write(CONFIG / f"cyclonedds-{name}.xml", config_xml(name, identity_ca, identity, signed))

    public_artifacts = sorted(p for p in CONFIG.iterdir() if not ("key" in p.name or p.suffix == ".csr"))
    manifest = {
        "domain_id": DOMAIN_ID,
        "allowed_topic": ALLOWED_TOPIC,
        "cyclonedds_version": "11.0.1",
        "cyclonedds_commit": "e54e991f75a3e67f8e628da3171122e36ea5b872",
        "openssl_version": run([str(OPENSSL), "version"]).stdout.strip(),
        "platform": platform.platform(),
        "endpoint_sha256": sha256(ENDPOINT),
        "artifacts": {str(p.relative_to(ROOT)): sha256(p) for p in public_artifacts},
    }
    write(ROOT / "config-manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def runtime_env(role: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CYCLONEDDS_URI"] = str(CONFIG / f"cyclonedds-{role}.xml")
    env["DYLD_LIBRARY_PATH"] = f"{CYCLONE / 'lib'}:{OPENSSL_LIB}"
    return env


def parse_received(path: Path) -> list[dict[str, object]]:
    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("RECEIVED\t"):
            continue
        _, index, monotonic, payload = raw_line.split("\t", 3)
        encoded = payload.encode("utf-8")
        case_id = None
        try:
            case_id = json.loads(payload).get("case_id")
        except (json.JSONDecodeError, AttributeError):
            pass
        records.append({"index": int(index), "receive_monotonic_ns": int(monotonic),
                        "case_id": case_id, "payload": payload,
                        "payload_sha256": hashlib.sha256(encoded).hexdigest()})
    return records


def transport(ndjson: Path, label: str, timeout_ms: int) -> dict[str, object]:
    lines = [line for line in ndjson.read_text(encoding="utf-8").splitlines() if line]
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)
    subscriber_out, subscriber_err = out_dir / "subscriber.stdout.log", out_dir / "subscriber.stderr.log"
    publisher_out, publisher_err = out_dir / "publisher.stdout.log", out_dir / "publisher.stderr.log"
    sub_stdout = subscriber_out.open("w", encoding="utf-8")
    sub_stderr = subscriber_err.open("w", encoding="utf-8")
    started = time.monotonic_ns()
    subscriber = subprocess.Popen(
        [str(ENDPOINT), "subscribe", ALLOWED_TOPIC, str(len(lines)), str(timeout_ms)],
        text=True, stdout=sub_stdout, stderr=sub_stderr, env=runtime_env("receiver"),
    )
    time.sleep(0.5)
    publisher = run([str(ENDPOINT), "publish", ALLOWED_TOPIC, str(ndjson), str(timeout_ms)],
                    env=runtime_env("allowed-publisher"), timeout=max(30, timeout_ms // 1000 + 10),
                    check=False, stdout_path=publisher_out, stderr_path=publisher_err)
    try:
        subscriber_rc = subscriber.wait(timeout=max(30, timeout_ms // 1000 + 10))
    except subprocess.TimeoutExpired:
        subscriber.terminate()
        subscriber_rc = subscriber.wait(timeout=5)
    finally:
        sub_stdout.close()
        sub_stderr.close()
    received = parse_received(subscriber_out)
    result = {
        "label": label,
        "input": str(ndjson.resolve()),
        "input_sha256": sha256(ndjson),
        "input_nonempty_lines": len(lines),
        "publisher_exit_code": publisher.returncode,
        "subscriber_exit_code": subscriber_rc,
        "wall_ns": time.monotonic_ns() - started,
        "received_count": len(received),
        "exact_ordered_payload_match": [r["payload"] for r in received] == lines,
        "received": received,
        "raw_logs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in (subscriber_out, subscriber_err, publisher_out, publisher_err)},
    }
    write(out_dir / "result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def negative(role: str, topic: str, label: str, timeout_ms: int = 3000) -> dict[str, object]:
    sample = RESULTS / f"{label}-input.ndjson"
    write(sample, '{"case_id":"negative-control","action_hex":"00","dtype":"float32","shape":[1]}\n')
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout, stderr = out_dir / "publisher.stdout.log", out_dir / "publisher.stderr.log"
    completed = run([str(ENDPOINT), "publish", topic, str(sample), str(timeout_ms)],
                    env=runtime_env(role), timeout=15, check=False,
                    stdout_path=stdout, stderr_path=stderr)
    result = {"label": label, "role": role, "topic": topic,
              "exit_code": completed.returncode, "wall_ns": completed.wall_ns,
              "stdout": stdout.read_text(encoding="utf-8"),
              "stderr": stderr.read_text(encoding="utf-8"),
              "raw_logs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                           for p in (stdout, stderr)}}
    write(out_dir / "result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def peer_negative(role: str, label: str, timeout_ms: int = 5000) -> dict[str, object]:
    sample = RESULTS / f"{label}-input.ndjson"
    write(sample, '{"case_id":"untrusted-peer-control","action_hex":"00","dtype":"float32","shape":[1]}\n')
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_out, sub_err = out_dir / "subscriber.stdout.log", out_dir / "subscriber.stderr.log"
    pub_out, pub_err = out_dir / "publisher.stdout.log", out_dir / "publisher.stderr.log"
    sub_stdout = sub_out.open("w", encoding="utf-8")
    sub_stderr = sub_err.open("w", encoding="utf-8")
    started = time.monotonic_ns()
    subscriber = subprocess.Popen(
        [str(ENDPOINT), "subscribe", ALLOWED_TOPIC, "1", str(timeout_ms)],
        text=True, stdout=sub_stdout, stderr=sub_stderr, env=runtime_env("receiver"),
    )
    time.sleep(0.5)
    publisher = run([str(ENDPOINT), "publish", ALLOWED_TOPIC, str(sample), str(timeout_ms)],
                    env=runtime_env(role), timeout=15, check=False,
                    stdout_path=pub_out, stderr_path=pub_err)
    subscriber_rc = subscriber.wait(timeout=15)
    sub_stdout.close()
    sub_stderr.close()
    received = parse_received(sub_out)
    result = {
        "label": label, "role": role, "topic": ALLOWED_TOPIC,
        "publisher_exit_code": publisher.returncode, "subscriber_exit_code": subscriber_rc,
        "received_count": len(received), "wall_ns": time.monotonic_ns() - started,
        "publisher_stdout": pub_out.read_text(encoding="utf-8"),
        "publisher_stderr": pub_err.read_text(encoding="utf-8"),
        "subscriber_stdout": sub_out.read_text(encoding="utf-8"),
        "subscriber_stderr": sub_err.read_text(encoding="utf-8"),
        "raw_logs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in (sub_out, sub_err, pub_out, pub_err)},
    }
    write(out_dir / "result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    tx = sub.add_parser("transport")
    tx.add_argument("ndjson", type=Path)
    tx.add_argument("--label", required=True)
    tx.add_argument("--timeout-ms", type=int, default=30000)
    neg = sub.add_parser("negative")
    neg.add_argument("--role", choices=["allowed-publisher", "denied-publisher", "rogue-publisher"], required=True)
    neg.add_argument("--topic", default=ALLOWED_TOPIC)
    neg.add_argument("--label", required=True)
    peer = sub.add_parser("peer-negative")
    peer.add_argument("--role", choices=["rogue-publisher"], required=True)
    peer.add_argument("--label", required=True)
    peer.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.command == "bootstrap":
        print(json.dumps(bootstrap(), indent=2, sort_keys=True))
    elif args.command == "transport":
        print(json.dumps(transport(args.ndjson, args.label, args.timeout_ms), indent=2, sort_keys=True))
    elif args.command == "negative":
        print(json.dumps(negative(args.role, args.topic, args.label), indent=2, sort_keys=True))
    else:
        print(json.dumps(peer_negative(args.role, args.label, args.timeout_ms), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
