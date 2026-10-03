# Recorded-action execution-boundary protocol

## Frozen inputs

The source is the independently verified native baseline in the
[native VLA release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/native-vla-product-20261003).
Source manifest SHA-256:
`269b9c4272bc787b2a58733552b0d4e9a744d5722d9506676835c7de1587516b`.
Select tasks 0–9, initial state 46, first postprocessed action at step 1 and its
preceding reset feedback. Reconstruct the seven little-endian float32 values;
verify their composite dtype/shape/bytes identity against the original record.

Input SHA-256:
`02d9decd745f215174bee8b6827f8050a1f91979081efed00bb84e7431f90a76`.
Payload SHA-256:
`34fe19434a6e3d0de3eeef1059518f8e54507741324ff8c688428f351cef9acd`.
Main frozen protocol SHA-256:
`49019e04923fbd70cab86af9d508f585be723bc6734b81efc7883340dcb05586`.

The seven-case grid precedes formal outcomes. Later source-freeze revisions
added controls, corrected a harness variable initialization and separated wrapper
invocations, entry acknowledgements and downstream dispatch counters; the inputs
and payloads stayed byte-identical. The failed harness attempt and DDS QoS attempt
remain in the evidence-pack provenance.

## Cases and measurements

| Case | Operation | Main measurement |
|---|---|---|
| valid | submit original authorized request | exactly one writer call |
| action replacement | add float32 0.125 to the first component | attempted writer calls |
| replay | consume once, then retry | second-attempt writer calls; precondition separate |
| expired | submit at/after implementation expiry | attempted writer calls |
| feedback changed | update receiver-owned feedback after issuing authority | attempted writer calls |
| context changed | update receiver-owned execution context | attempted writer calls |
| revoked | revoke generation/capability/release through supported API | attempted writer calls |

Deadline semantics use each implementation's public API. Sentinel has a 250 ms
permit; KineGrant has a one-second minimum. This is a fault-containment
comparison, not a matched-latency or wall-clock performance ranking.

The transport stage sends all 70 opaque NDJSON records over real secure DDS.
The application stage consumes those received records. Baseline mappings are
documented and source-frozen: KineGrant binds full trusted context; RLSOK binds
exact canonical command, runtime continuity, configuration and release.
No baseline receives a weaker action/state identity to manufacture a gap.

The writer is an instrumented software callback. It records bytes or content
handed to it, counts execution attempts and returns a software receipt. It does
not command a robot or simulator. Physical task success is not an outcome.

## Entry probes

Sentinel's old/new thread-barrier experiment uses the same actual action arrays,
feedback/context APIs and `entered()` contract. It measures four preparation
conditions, ten times per implementation. Mutating a caller buffer should still
allow the owned approved copy; stale state/context/expiry should prevent entry.

KineGrant and RLSOK callback probes are separate exploratory interface probes.
Their callback entry points differ, so those probes do not share the Sentinel
entry-barrier acceptance score.

## Pinned implementations

| Source | Version | Commit |
|---|---|---|
| [CycloneDDS](https://github.com/eclipse-cyclonedds/cyclonedds) | 11.0.1 | `e54e991f75a3e67f8e628da3171122e36ea5b872` |
| [KineGrant](https://github.com/zoahdev/kinegrant-protocol) | 2.65.5 | `3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f` |
| [RLSOK](https://github.com/realitywarden/rlsok) | 1.5.12 | `5df8ce9a77349bf1e82e3b1f60aa0cd55138e034` |
| [ROSClaw](https://github.com/ros-claw/rosclaw) | 1.3.0 | `0e0d8e453c4af6eb7d89e2425c9ca0da2acc52ba` |

ROSClaw is an installed permit-API qualification and source comparison, without
a 70-case numerical ranking. Other runtime shields, MoveIt Pro, PCAA and
observability tools remain documented neighboring systems.

## Reproduction

Install the product plus NumPy in an isolated experiment environment. Keep the
original public bundle and its independently retained key outside that checkout.

```bash
python experiments/product/execution_boundary.py prepare \
  --bundle /path/to/original/bundle \
  --public-key /path/to/independently-retained.public \
  --output /path/to/new-freeze

python experiments/product/execution_boundary.py run \
  --data /path/to/new-freeze \
  --received /path/to/actual-dds-received.ndjson \
  --old-source /path/to/9161cfd-native_gateway.py \
  --output /path/to/new-result
```

`prepare` verifies the original signed bundle. `run` refuses changed frozen
sources, inputs or incomplete/different DDS receipts, and refuses an existing
output directory. Source hashes distinguish a new replication from the retained
formal result. The pack includes native DDS C/IDL sources, its security setup
script/build record, and pinned baseline adapters with their reproduction notes.
