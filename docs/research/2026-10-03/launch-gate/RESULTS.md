# VLA integration checks before motion

Sentinel Launch Gate checks an installed VLA integration against a reviewed reference on fixed inputs. It compares the named camera tensors and ordered state values passed to the policy, the selected action cursor, and the exact final action bytes. A changed binding blocks the trusted software writer and produces a specific field difference and observed repair hints.

## Measured outcomes

| Experiment | Inputs | Outcome | Blocked downstream calls |
|---|---:|---|---:|
| Signed-record adapter replay | 20 requests from 10 task windows | 10/10 injected integration configurations BLOCK | 0 |
| Fresh official SmolVLA GPU inference | 10 LIBERO tasks, initial state 46 | 6/6 injected integration configurations BLOCK | 0 |
| Unchanged integration | Same respective input suites | PASS; one diagnostic software call per passing case | — |
| Identity-only recorded adapter change | 20 requests | PASS; observed behavior preserved | — |
| Different source input | Recorded suite | REVIEW | 0 |
| Loaded adapter identity changes after qualification | Recorded suite | DENIED before entry | 0 |

The GPU experiment uses **RTX 4090 D (24 GB)**. It performs 40 fresh inference paths: reference, unchanged repeat, swapped cameras and swapped state components for each of ten tasks. Output-order, sign, gripper and cursor faults are applied at their declared adapter stages to the official model output. The unchanged repeat matches exact captured values and action bytes. Every GPU fault has zero entry acknowledgements and zero downstream software calls. The runner makes **zero `env.step` calls**.

The recorded experiment reruns a mapping adapter over independently verified signed native records. It retains camera commitments and exact action values; it does not rerun vision inference. The two experiments have separate protocols and result files.

## What the diagnosis shows

- **Swapped cameras:** the candidate's `observation.images.camera1` receives the tensor bound to reference `camera2`, and vice versa, on all ten GPU inputs.
- **Normalized state swap:** candidate component 0 follows reference component 1 with scale `1.4461643742` and offset `0.7711530316`; component 1 follows reference component 0 with scale `0.6914839616` and offset `-0.5332408861`. Maximum residuals are `5.70e-8` and `1.17e-7` across all ten inputs.
- **Output order:** action components 0 and 1 correspond to the opposite reference components.
- **Direction/gripper:** the changed action component equals the negative of the reference component.
- **Cursor:** the selected action index has a constant `+1` offset.

Numeric hints require a unique varying source relation across every paired input, with finite values and a declared `2e-6` relative matching tolerance. Exact qualification remains byte/value based. A hint gives a place to inspect; check the actual integration code to establish the cause, then recapture after repair. Sentinel does not apply a guessed correction to a robot action.

## Cost and portable evidence

On the retained local CPU host, 200 qualification measurements per recorded case give an unchanged-case mean of **1.436 ms**, P95 **1.484 ms**. Camera-swap qualification averages **0.859 ms**, P95 **0.900 ms**. These measurements cover in-process qualification; model inference, signing, network transfer and environment stepping are excluded.

GPU inference-path means are **360.98 ms** reference, **315.18 ms** unchanged, **312.95 ms** camera swap and **314.44 ms** state swap. The reference series includes initial warmup and had a maximum of **788.15 ms**. Timing includes the policy preprocessor, GPU `select_action`, official postprocessor and CUDA synchronization. It is not a cross-product speed benchmark.

The GPU camera-swap signed capsule is **26,172 bytes (25.6 KiB)** before compression. It contains the exact reference/candidate probe records, the qualification report, events and signed manifest. It retains camera commitments rather than raw image pixels. A CPU machine can verify the selected key and recompute every authoritative decision field without model weights. All **13 recorded capsules and 7 GPU capsules** independently reproduce; current repair hints also match. Evolving hints remain signature-verified and are reported separately from authoritative decision reproduction.

## Competitive position

[The pinned comparison](COMPETITIVE.md) runs real KineGrant 2.65.5 and RLSOK 1.5.12 authorization APIs on the same retained reference and camera-swap action bytes. Each permits 40/40 properly authorized requests; this is their intended behavior. RLSOK also correctly invalidates approval for changed release fields. The native SmolVLA action semantics are explicitly outside the RLSOK program-schema accommodation used in this harness.

Sentinel adds an installed workflow for comparing observed VLA integration behavior with a retained reference before launch, locating changed fields, enforcing its decision and sharing a signed CPU reproducer. The distinction concerns this packaged workflow. Custom integration can add behavioral probes to the neighboring products. The earlier ordinary authorization comparison remains a tie at 60/60 denied faults for Sentinel, KineGrant and RLSOK.

## Reproduce the retained decision

Download [the evidence archive](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/launch-gate-20261003/sentinel-launch-gate-evidence.zip), check its published SHA-256, and unpack it. Keep the selected public key separate from the bundle.

```bash
sentinel-evc launch-reproduce \
  --bundle sentinel-launch-gate-evidence/gpu/capsules/camera_swap/bundle \
  --public-key sentinel-launch-gate-evidence/gpu/capsules/camera_swap/anchors/demo.public \
  --run-id gpu-launch-camera-swap
```

Expected: `reproduction=PASS`, `launch_verdict=BLOCK`, `diagnosis_reproduction=MATCH`. Reproduction success authenticates and recomputes the retained BLOCK; it does not grant launch permission.

Fresh GPU execution uses the [retained GPU protocol and commands](GPU.md). The [recorded protocol](PROTOCOL.md) describes source snapshot authentication, mapping adapters, positive controls and fault configurations. [Integration usage](../../../../launch_gate.md) shows the writer hook and CLI/App entry points.

## Native App workflow

A locally built macOS App imported the retained GPU reference and camera-swap
probe files, reported BLOCK with both camera-routing hints, and saved the signed
ZIP and independent public key through native file dialogs. CLI reproduction of
the actual saved ZIP returned PASS / BLOCK / MATCH. The archive retains the saved
bundle, public key, receipt and interface screenshot under `app/`. This check
reuses captured GPU probes; it does not rerun model inference or move hardware.

## Source identities and verification

| Retained artifact | SHA-256 |
|---|---|
| Recorded result | `8eee92c3524fa773379b2925e20d2889367f80f22e1ac775b79e660a3d8dfaa7` |
| GPU result | `7334fd8310d84f83a642c4a27d190ee0fb987a3d893f2040b481a581f3b60910` |
| Frozen GPU protocol | `a7e4df281a29fa93cdb5a1da78f348dc3e84e05711ada983c22e4cef34460b3b` |
| Competitive result | `21f518797356605182aaf7e8e1f8bb0d50e676150bb058f3e6edda5e4d1d1da2` |

GPU rows directly bind reference/candidate pack hashes and signed capsule manifest hashes. The protocol binds the runner, model/config/processor files, task/state files, runtime versions and first-party qualifier sources. The competitive run binds immutable input snapshots and pinned Git archives. The release index includes every retained evidence-file hash and the exact public source commit.

Independent review approved the release with no unresolved findings. Existing regression: **211 passed, 0 failed, 9 skipped**. The 1,000-case geometry baseline retains 500/500 safe passes and zero final-check false allows; seven fault cases, 2,031 signed events and four distinct tamper failures match the standing release baseline.

PASS covers the retained finite probes. This launch check evaluates integration behavior and the trusted software writer. Task success, collision clearance, sensor authenticity, physical stopping and hardware safety require their own validation. Concurrent adapter replacement requires the loader's shared external lock across the complete gateway/writer call and an O(1) loaded-identity read.
