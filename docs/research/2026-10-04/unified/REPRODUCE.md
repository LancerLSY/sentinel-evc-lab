# Reproduce the unified experiment

## Read the retained results

Extract each archive into a separate directory. The archives include the complete
run records and exact source snapshots. Use the source snapshot for its stage.
The repository runner contains the later A3 full-state restoration correction. Historical stages require their archived source snapshots.

For a historical A1 execution, keep the repository experiment directory separate:

```bash
cd retained/A1/A1-source
MUJOCO_GL=egl PYTHONPATH=src python run_unified_libero.py --config /path/to/config-A1.json
```

`panda_geometry.py` in that directory is the A1 checker. The corresponding A2
directory is `retained/A2/A2-source`. Changing paths or deadlines changes the
configuration hash. Save that change in the new run receipt.

```bash
mkdir -p retained/A1
tar -xzf docs/research/2026-10-04/unified/evidence/A1-run.tar.gz -C retained/A1
python experiments/vla/summarize_unified_libero.py \
  --run-dir retained/A1/formal-A1-20261004 --output-dir summary-A1
```

Summary generation uses the Python standard library. It does not run a model or
simulator. The CSV retains every formal episode, including crashes and denials.

## Prepare an optional GPU environment

The experiment runtime is separate from the lightweight product installation.
The A3 native-copy helper requires Linux and exactly MuJoCo 3.3.7. It uses the bundled official C API through Python ctypes. Recorded versions: Python 3.12, torch 2.8.0+cu128, numpy 2.2.6, MuJoCo 3.3.7,
robosuite 1.4.0, hf-libero 0.1.4, lerobot 0.6.1, transformers 5.5.4,
huggingface-hub 1.33.0. Rendering uses imageio, imageio-ffmpeg and Pillow.
Plot generation also uses matplotlib.

Download the pinned inputs:

- `lerobot/smolvla_libero`, revision `31d453f7edd78c839a8bbc39744a292686daf0de`.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` tokenizer/configuration,
  revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- `lerobot/libero-assets`, revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.

The SmolVLA checkpoint contains the complete frozen backbone weights. Its
`model.safetensors` SHA-256 is
`9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.

Configure LIBERO's asset, BDDL and initial-state paths. Confirm that all 50 spatial
initial states exist. The exact software inventory is recorded in the manifests.

## Execute a fresh grid

Copy the selected stage configuration. Adjust checkpoint/backbone/output paths
and the stop deadline for your machine. Preserve task, state, seed, aggregation,
contact-policy and tolerance settings when comparing results. Save the revised
config hash before running.

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python experiments/vla/run_unified_libero.py --config /path/to/config.json
```

Use a new output directory. The runner refuses to overwrite a previous run.
The one-step engineering preflight precedes formal episodes and is excluded from
task-success counts. A nonzero process result and a failed manifest are expected
when tracking aborts occur. Retain them.

## Produce a live demonstration

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python experiments/vla/capture_unified_libero.py \
  --config /path/to/config-A2.json --output-dir /path/to/new-live-demo
```

The helper runs exactly task 0/state 40/delta EVC with the fixed seed. It renders
after authorized writes and excludes forecast frames. Keep this demonstration
outside formal statistics. The video uses the simulator's 20 Hz motion clock.
Its metadata records the actual wall-clock computation time.

`replay_unified_libero.py` independently attempts action replay and checks saved
substep trajectories. It aborts video generation when the declared tolerance is
exceeded. The retained failed A1 replay illustrates why a saved-action replay
must be checked before it is presented as an exact reproduction.
