# Recorded-action adapter launch qualification

## Customer problem and decision

A VLA deployment can still start after a camera key, state-vector order,
unit convention, action order, gripper convention, chunk cursor or final
postprocessing step changes.  A model file checksum or container digest does
not show whether those interfaces still produce the same request.

This protocol captures what a deployment adapter actually consumed and emitted
on a bounded set of signed recordings.  A candidate release receives:

- `PASS` when every recorded input is aligned and every consumed binding,
  cursor and final float32 action is identical;
- `REVIEW` when the two releases were not measured on the same input suite;
- `BLOCK` at the first changed mapping or final action.

`QualifiedNativeWriter` applies that result at the existing trusted writer
boundary.  It checks the current adapter digest again immediately before the
gateway entry acknowledgement and downstream software call.
`current_adapter_digest` must read the already loaded deployment identity in
O(1). A loader that permits concurrent hot swapping must hold its own shared
lock across the complete writer call; the writer does not provide that lock.

## Frozen source and probe contract

The source is the independently retained public key and signed bundle
`native-formal-baseline-v3`.  `prepare` snapshots the manifest, signature,
independently retained public key and every file declared by the manifest into
a bounded private temporary bundle.  It verifies the signature and exact
digest of every snapshotted member, then verifies all seven bundle layers on
that same private snapshot.  No later read from the mutable source path
supplies probe values or verification evidence.  It selects tasks 0–9, fixed
initial state 46, and
steps 0 and 1: 10 recorded windows and 20 final requests.

For every step, the reader requires one each of:

1. `policy_input`, including the two actual camera tensor identity commitments,
   actual eight-component state and observation hash;
2. `selected_normalized_action`, including actual chunk and action cursor;
3. `official_postprocessed_action`, including the exact final float32 values.

The reader reconstructs both normalized and final little-endian float32 action
bytes and refuses an identity mismatch.  The frozen pack schema is
`sentinel-vla-probe-pack-v1`:

```text
schema, suite_id, adapter_digest, provenance,
probes[] = {
  id, input_hash,
  consumed.cameras[policy_key] = tensor_identity_sha256,
  consumed.state = {names, units, values},
  cursor = {chunk_id, action_index},
  action = {dtype=float32, shape=[1,7], byte_order=little, bytes_hex}
}
```

State values use strict JSON shortest-roundtrip finite-float encoding.  The
schema rejects duplicate state component names; action byte order is explicit
so the same hex cannot be decoded with a host-dependent convention.

The trace retains camera tensor commitments, not raw pixels.  This is an
adapter-mapping replay.  It does not rerun vision or model inference.

## Adapter hook

`RecordedActionAdapter.observe(record)` is the experiment adapter.  A real
runner uses the same output contract after its own preprocessing, chunk
selection and official postprocessing:

```python
probe = capture_probe(
    probe_id=case_id,
    input_hash=observation_hash,
    cameras=policy_camera_tensors,
    state_names=state_names,
    state_units=state_units,
    state_values=state_values,
    chunk_id=chunk_id,
    action_index=action_index,
    action=official_postprocessed_action,
)
```

The production hook must capture the tensor objects actually passed to the
policy and the final array actually passed to the native gateway.  Supplying a
configuration declaration in place of those observed values is outside this
contract.

## Frozen cases

The candidate packs are produced by executing the adapter configuration.  The
experiment does not edit qualification results directly.

| Case | Expected result |
|---|---|
| unchanged adapter | PASS |
| different adapter digest, same observed behavior | PASS, limited to these probes |
| different recorded input hash | REVIEW |
| same-shape camera tensors swapped | BLOCK |
| camera policy route renamed | BLOCK |
| state indices swapped | BLOCK |
| state unit changed from m to mm and value scaled | BLOCK |
| action components reordered | BLOCK |
| action component sign inverted | BLOCK |
| gripper convention inverted | BLOCK |
| chunk action index advanced by one | BLOCK |
| postprocessing scale changed | BLOCK |
| one final float32 action bit changed | BLOCK |

Every semantic fault must change the first probe.  This prevents a nominal
fault from silently hitting a zero or an unused field.

## Acceptance and measurements

The run is accepted only when:

- unchanged and identity-only candidates pass;
- the input mismatch returns review;
- all ten semantic faults block;
- blocked cases make zero downstream software-writer calls;
- a candidate that passed but changes its live adapter digest before entry is
  denied with zero entry acknowledgements and zero downstream calls;
- every signed derived capsule verifies and reproduces its stored decision.

The result retains 200 actual `perf_counter_ns` qualification measurements per
case, the reference and candidate pack sizes, each capsule size, the original
signed trace/result/replay byte count, source commit, source-file digests and
measurement host.  Timing covers in-process qualification only; it excludes
model inference, environment stepping and network transport.

## Reproduction

Use an empty output directory at each phase:

```bash
python experiments/product/launch_qualification.py prepare \
  --bundle /path/to/native-formal-baseline-v3/bundle \
  --public-key /path/to/native-v3-baseline.public \
  --output /path/to/launch-prepared

python experiments/product/launch_qualification.py run \
  --data /path/to/launch-prepared \
  --bundle /path/to/native-formal-baseline-v3/bundle \
  --public-key /path/to/native-v3-baseline.public \
  --output /path/to/launch-result
```

`run` takes and authenticates a fresh source snapshot and requires its signed
identities to equal the prepared snapshot.  It also refuses changed experiment
or launch-gate source after `prepare`.  Each case is exported as a newly signed
derived capsule containing the exact reference pack, candidate pack and
qualification report.

## Scope

This protocol measures deterministic adapter compatibility and the software
writer decision for 20 retained requests.  It does not measure task success,
collision clearance, dynamics, a physical stop or robot hardware.  RLSOK's
published release/configuration and canonical-command binding is a neighboring
deployment control.  This experiment measures an additional finite behavioral
probe comparison and its writer deny; it is not an overall product ranking.

Repair hints match observed camera routing, component transforms and cursor offsets across all paired probes. Their numeric tolerance is separate from the exact qualification verdict. Apply a correction in the actual adapter and recapture probes before approval.
