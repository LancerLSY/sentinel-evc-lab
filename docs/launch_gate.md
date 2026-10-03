# Sentinel Launch Gate

Before an updated adapter controls an arm, run the same asymmetric inputs through the reference and candidate. Sentinel checks the named camera bindings, ordered state values and units, selected action chunk/index, and the exact final native request. It points to the first changed field and reports observed routing, sign, scale and cursor patterns.

## Use from CLI or App

```bash
sentinel-evc launch-check --reference reference.json --candidate candidate.json --out runs/launch-check
sentinel-evc launch-reproduce --bundle runs/launch-check/bundle --public-key runs/launch-check/anchors/demo.public --run-id launch-check
```

`launch-check` returns 0 for PASS and 3 for BLOCK or REVIEW. `launch-reproduce` returns 0 when the signed input bytes reproduce the stored verdict, including an original BLOCK. Its `launch_verdict` stays separate from reproduction success.

In the App or local browser workbench, choose **AI execution → Launch Gate**. Select two probe JSON files. Inspect the first changed field and the mapping clues. Export the capsule ZIP and separate public key. A colleague can reproduce the verdict on a CPU with the core package. Model weights and full camera videos are unnecessary for verdict reproduction.

## Capture actual integration outputs

`capture_probe()` accepts the actual camera tensors consumed by the policy and the final tensor prepared for the writer. Torch/NumPy stay in the existing model environment. The core imports neither. Camera data is stored as a dtype/shape/bytes commitment; final actions retain their exact little-endian bytes. Capture ordered state names, units and values at the consumed-input boundary, not from an expected configuration copied into the output.

The pack schema is `sentinel-vla-probe-pack-v1`. It includes a frozen suite ID, adapter digest, provenance and 1–128 probes. Each probe binds its source input hash to consumed cameras/state, the selected cursor and final action. Packs reject duplicate JSON keys, duplicate state names, non-finite action/state values, unsupported byte order and inconsistent dimensions. Pack/report JSON preserves floating-point values.

Use fixed, differing camera inputs and varying state/action values. Identical cameras or constant signals cannot expose every permutation. Both adapters must receive the same input bytes and reset state/random seeds consistently. Intentional policy changes require a newly reviewed reference. PASS covers the retained probes.

## Keep the result at the writer entry

```python
from sentinel_evc.launch_gate import QualifiedNativeWriter

writer = QualifiedNativeWriter(
    reference_bytes,
    candidate_bytes,
    lambda: loaded_adapter_identity,  # O(1), already loaded identity
    environment.step,                # one-argument downstream writer
)
gateway.submit(permit, final_action, snapshot=snapshot, context=context, writer=writer)
```

The wrapper refuses BLOCK/REVIEW packs and an adapter identity that changed since qualification. It checks the identity before and after the native gateway's `entered()` callback. The loader must hold a shared external lock across the complete `submit`/writer call when supporting concurrent hot swapping. Reading the identity must not hash model weights on each action. The gateway retains its existing one-use, feedback, context, deadline and owned-action checks.

## Interpret the result

- **PASS:** all captured bindings, state facts, cursors and final action bytes match on the fixed probes. Identity-only refactors can pass.
- **BLOCK:** a captured input interpretation, cursor or final action changed. Inspect the first field and repair hints. Correct the actual adapter and recapture the probes.
- **REVIEW:** inputs, suite, probe IDs or required schema do not align. Supply comparable complete records before launch.

A repair hint describes a relationship across all paired probes. Numeric transform matching uses a stated tolerance. It never weakens the exact verdict or automatically transforms an action.

The capsule is a newly signed derived decision with a demo integrity key. The receiver selects its public key independently. Verification snapshots bounded regular files once, checks the existing independent seven-layer verifier on those bytes, and recomputes every authoritative report field from the same bytes. Repair hints remain signature-verified; their heuristic can improve in a newer version without changing the authoritative decision. Reproduction reports `UPDATED_HINTS` when that happens. This checks recorded integration behavior; task success, sensor authenticity, collision clearance and physical stopping require their own validation.

## Evidence and competitive position

The [fixed experiment protocol](research/2026-10-03/launch-gate/PROTOCOL.md) separates a signed-record mapping replay from fresh GPU policy inference. Each fault is applied in the adapter before capture. Source identities, all outcomes, software-writer counters, capsule size and qualification timing are retained.

LeRobot supplies model deployment, processors and feature contracts. RLSOK binds approved releases, configuration, runtime and action contracts. KineGrant supplies authorization and portable receipts. Sentinel adds the packaged VLA integration check and executable diagnosis described above. The pinned comparison audit credits those capabilities. Qualification is complementary to model training and motion validation.
