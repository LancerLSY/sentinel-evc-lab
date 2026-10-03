# Sentinel VLA GPU launch-probe evidence

This directory contains the retained result of a finite, no-motion launch
qualification run on an NVIDIA GeForce RTX 4090 D. It covers ten LIBERO
Spatial tasks at fixed initial-state index 46. The runner reset each official
environment, ran the official environment and policy preprocessors, executed
SmolVLA on the GPU, and ran the official policy postprocessor. It never called
`env.step`.

## Result

- Repository result: `gpu-result.json`; evidence archive: `gpu/result.json`
- Result SHA-256: `7334fd8310d84f83a642c4a27d190ee0fb987a3d893f2040b481a581f3b60910`
- Repository protocol: `gpu-protocol.json`; evidence archive: `gpu/protocol.json`
- Protocol SHA-256: `a7e4df281a29fa93cdb5a1da78f348dc3e84e05711ada983c22e4cef34460b3b`
- The protocol in the evidence archive is byte-identical to the frozen protocol.
- Every case row directly binds the exact reference pack SHA-256, candidate
  pack SHA-256, and signed capsule manifest SHA-256 retained under `gpu/` in the evidence archive.
- The unchanged candidate passed and completed one writer entry and one
  downstream software call.
- Camera swap, state-index swap, action-order swap, action-sign inversion,
  gripper inversion, and chunk-offset cases all blocked before writer entry:
  six of six blocked, zero blocked downstream calls.
- All seven signed capsules reproduce with `reproduction=PASS` and
  `diagnosis_reproduction=MATCH` under the retained source identities.
- Across all ten probes, the state-swap diagnosis identifies the observed
  cross-component relationships `destination[0] <- reference[1]`
  (`scale=1.4461643742`, `offset=0.7711530316`, maximum residual `5.70e-8`)
  and `destination[1] <- reference[0]` (`scale=0.6914839616`,
  `offset=-0.5332408861`, maximum residual `1.17e-7`). This is an observed
  repair hint; it is not a cause attribution or an automatically applied
  correction.

Mean measured inference-path time, including the official policy preprocessor,
GPU `select_action`, postprocessor, and CUDA synchronization, was 360.98 ms for
the reference pass, 315.18 ms for the unchanged repeat, 312.95 ms for the
camera-swap probes, and 314.44 ms for the state-swap probes. The reference series
includes initial warmup and had a maximum of 788.15 ms. These are measurements from
this host and finite probe suite, not a throughput or task-success benchmark.

## Captured boundary

Each reference probe binds the exact `observation.images.camera1`,
`observation.images.camera2`, and normalized `observation.state` tensor objects
returned by the official policy preprocessor and passed as the batch to
`policy.select_action`. It separately retains an upstream observation identity
for the two environment camera tensors and physical-unit state bytes. It binds
the official postprocessor's final little-endian float32 `[1, 7]` action.

This is the public `select_action` call boundary. SmolVLA may perform additional
internal tensor transforms inside `select_action`; this evidence does not claim
to capture private intermediate VLM activations. It proves the observed adapter
input/output contract and writer decision for these ten frozen inputs.

## Frozen runtime

- `lerobot==0.6.1`
- `hf-libero==0.1.4`
- `robosuite==1.4.0`
- `mujoco==3.8.1`
- `num2words==0.5.14`
- `numpy==2.2.6`
- `torch==2.8.0+cu128`
- `transformers==5.5.4`
- CUDA 12.8, Python 3.12.3
- SmolVLA checkpoint revision `31d453f7edd78c839a8bbc39744a292686daf0de`
- SmolVLA weight SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`
- SmolVLM2 backbone revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- SmolVLM2 weight SHA-256 `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`
- LIBERO assets revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`

The protocol also freezes the runner, model configs and processors, task BDDL
files, fixed initial-state files, and every first-party source file used by the
launch qualifier.

## Reproduction

On a host with the frozen files and runtime installed, first freeze the protocol
before producing outcomes:

```bash
LIBERO_CONFIG_PATH=/path/to/libero-config \
PYTHONPATH=src python experiments/vla/run_launch_probes.py prepare \
  --output /empty/path/protocol.json
```

Then run real inference without environment stepping:

```bash
LIBERO_CONFIG_PATH=/path/to/libero-config \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python experiments/vla/run_launch_probes.py run \
  --protocol /empty/path/protocol.json \
  --checkpoint /path/to/smolvla_libero-31d453f \
  --backbone /path/to/SmolVLM2-500M-Video-Instruct \
  --output /empty/path/result
```

The runner forces `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Output paths must not already exist. A changed
runner, source file, runtime version, model file, task source, or protocol is
rejected before inference.

To reproduce every signed decision from this retained copy:

```bash
PYTHONPATH=src python - <<'PY'
import hashlib
import json
from pathlib import Path
from sentinel_evc.launch_gate import reproduce_qualification_capsule

root = Path("sentinel-launch-gate-evidence/gpu")
index = json.loads((root / "result.json").read_text())
digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
for row in index["cases"]:
    case = row["case"]
    capsule = root / "capsules" / case
    assert row["sources"]["reference"]["pack_sha256"] == digest(root / "packs/reference.json")
    assert row["sources"]["candidate"]["pack_sha256"] == digest(root / f"packs/{case}.json")
    assert row["capsule_manifest_sha256"] == digest(capsule / "bundle/manifest.json")
    report = reproduce_qualification_capsule(
        capsule / "bundle",
        capsule / "anchors/demo.public",
        "gpu-launch-" + case.replace("_", "-"),
    )
    print(case, "HASHES_MATCH", report["reproduction"], report["diagnosis_reproduction"])
PY
```

The retained `run-v6.log` ends with harmless robosuite EGL destructor warnings
after successful result creation. The process exited zero, the result gates
passed, and the signed capsules independently reproduce.

## Retained source identities

The published GPU result is the fresh 40-inference run. Its index binds every
pack and capsule manifest directly. Qualification remains byte-exact. The paired
recorded-action replay is published as `recorded-result.json` in this repository
and `recorded/result.json` in the evidence archive, with SHA-256
`8eee92c3524fa773379b2925e20d2889367f80f22e1ac775b79e660a3d8dfaa7`.

The evidence archive retains this GPU run's `run-v6.log`. Paths in the frozen
protocol describe the original execution host; rerunning on a new host requires
preparing a new protocol with the exact installed source, runtime and model files.
