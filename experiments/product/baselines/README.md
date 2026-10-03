# Installed authorization comparison adapters

These are optional experiment dependencies. The product gateway does not import
KineGrant, RLSOK, ROSClaw, CycloneDDS or NumPy. The
[protocol and results](../../../docs/research/2026-10-03/execution-boundary/RESULTS.md)
separate actual secure transport, application authorization and exploratory
interface probes.

## KineGrant

Clone `https://github.com/zoahdev/kinegrant-protocol`, check out
`3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f`, and install that checkout in an
isolated Python 3.11+ environment. Run its official `kinegrant-demo` first.
Create separate output directories and run the exact retained adapter:

```bash
python /path/to/sentinel/experiments/product/baselines/kinegrant.py \
  --payloads /path/to/pack/freeze-v4/payloads.ndjson \
  --output /path/to/new-formal-directory/results.ndjson

python /path/to/sentinel/experiments/product/baselines/kinegrant.py \
  --payloads /path/to/pack/freeze-v4/payloads.ndjson \
  --output /path/to/new-exploratory-directory/results.ndjson \
  --entry-races --entry-races-only
```

The adapter records its historical v3 parent-protocol identity. The pack's
v3/v4 inputs are byte-identical; v4 only clarifies main dispatch counters.
The 70-case formal artifact and 40 exploratory interface probes remain separate.

## RLSOK

Clone `https://github.com/realitywarden/rlsok`, check out
`5df8ce9a77349bf1e82e3b1f60aa0cd55138e034`, and install its lockfile dependencies
in an isolated Node 22 environment. The retained TypeScript experiment uses the
upstream runner and relative imports. Copy `rlsok.ts` into that checkout at
`tests/competitive-boundary/frozenHarness.test.ts`, then run from its root:

```bash
FROZEN_PAYLOADS=/path/to/pack/freeze-v4/payloads.ndjson \
RLSOK_RESULTS=/path/to/new-results.ndjson \
RLSOK_PROBES=/path/to/new-entry-probes.json \
node scripts/run-rlsok.cjs --test tests/competitive-boundary/frozenHarness.test.ts
```

This is an experimental case runner using upstream tooling; it adds no product
unit tests. The archive includes upstream qualification output, adapter config,
source identities and exact commands. ROSClaw is a qualified API/source neighbor,
without a 70-case score.

## Secure DDS

`dds/src` contains the actual C/IDL publisher/subscriber used in the experiment.
The retained setup runner is macOS-specific and expects isolated installs beneath
`/private/tmp/sentinel-dds-20261003`: `openssl-install`, `cyclonedds-install` and
`app-build`. Build OpenSSL 3.5.8 and CycloneDDS at commit
`e54e991f75a3e67f8e628da3171122e36ea5b872` with `ENABLE_SECURITY=YES`,
`ENABLE_SSL=YES`, and Release configuration; link the plugins to that OpenSSL.
Build the CMake endpoint against that CycloneDDS install.

The pack retains the actual build record, frozen security protocol, public
certificate/configuration hashes, raw native traces and all positive/negative
controls. See the [official Security build and setup](https://cyclonedds.io/docs/cyclonedds/latest/security.html).
Run `python dds/run_secure_dds.py --help` for bootstrap, transport and peer/topic
controls. Bootstrap generates new experiment identities and private keys locally;
`dds/.gitignore` excludes them. Existing release data contains no private keys.

## Attribution

KineGrant and RLSOK are Apache-2.0, ROSClaw is MIT; retained license texts are
under `licenses/`. CycloneDDS is obtained from its upstream source under its
published Eclipse distribution license. These adapters exercise public APIs
and do not vendor or replace the upstream authorization implementations.
