# Reproduce the v2 evidence

Use Python 3.12.3, NumPy 2.3.2 and MuJoCo 3.8.1 to match the recorded experiment environment. Core installation still requires only `cryptography`. Install experiment dependencies in a separate environment.

```bash
python -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install -e . cryptography==46.0.3 numpy==2.3.2 mujoco==3.8.1 pytest==8.4.2
```

The delivery archive contains `source/`, `assets/`, `results/`, `reviews/`, the three original inputs and patches. Run the following commands from `source/`. Use new output directories. Existing evidence must remain unchanged.

## Check retained summaries without new timing

```bash
python experiments/v2_review/analyze_counterbalanced_arm.py --inputs ../results/fair-arm/repeat0.json ../results/fair-arm/repeat1.json ../results/fair-arm/repeat2.json --amendment ../results/fair-arm/analysis-amendment.pre-summary.json --out ../recomputed-main.json
python experiments/v2_review/analyze_counterbalanced_arm.py --inputs ../results/fair-arm-n4/repeat0.json ../results/fair-arm-n4/repeat1.json ../results/fair-arm-n4/repeat2.json --amendment ../results/fair-arm-n4/analysis-amendment.pre-summary.json --out ../recomputed-n4.json
```

Compare these files with `results/fair-arm/summary.root-cluster.json` and its N4 counterpart. Both summaries reproduced byte-for-byte locally. The separate analysis script implements the corrected root-cluster unit. The original runner's `summarize` subcommand retains its historical root-process analysis and must not supply the reported confidence intervals.

## Repeat timing on frozen candidate inputs

The full archive supplies all ten hashed NPZ trace files. Copy each retained `protocol.frozen.json` into its corresponding `traces/` directory before a new run.

```bash
cp ../results/fair-arm/protocol.frozen.json ../results/fair-arm/traces/
cp ../results/fair-arm-n4/protocol.frozen.json ../results/fair-arm-n4/traces/
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-2 python experiments/v2_review/run_counterbalanced_arm.py run --mjcf ../assets/ur5e.xml --trace-dir ../results/fair-arm/traces --out ../new-main-repeat0.json --repeat-index 0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-2 python experiments/v2_review/run_counterbalanced_arm_n4.py run --mjcf ../assets/ur5e.xml --trace-dir ../results/fair-arm-n4/traces --out ../new-n4-repeat0.json --repeat-index 0
```

Repeat with indices 1 and 2 in separate Python processes. These processes are technical repeats of the same roots. Use the root-cluster analyzer above with the new output paths. `taskset` is a Linux command. A new hardware timing result is a new experiment, not a reconstruction of the historical wall-clock measurements.

The Git evidence directory contains timing outputs and trace hashes. The full delivery archive contains the large frozen inputs. For a fresh input study, use the corresponding runner's `prepare` subcommand with `--mjcf`, `--trace-dir`, `--protocol`, `--roots 12`, `--cycles 60` and `--warmup 20`. Freeze the inputs before timing.

## Native permission faults

```bash
python experiments/v2_review/run_native_fault_drill.py --output-dir ../new-native-fault-drill
```

This command writes its frozen protocol before running eight classes × 200 cases. The aggregate acceptance criteria and separate concurrency probes are in its summary. Its JSONL matches the supplied server record byte-for-byte.

## Independent MuJoCo contacts

```bash
python experiments/v2_review/run_mujoco_crosscheck.py run --mjcf ../assets/ur5e.xml --protocol ../results/mujoco-crosscheck-protocol.json --out ../new-mujoco-crosscheck
```

This uses the recorded protocol and verifies its source/model bindings. It evaluates static contact with `mj_forward`. It does not run dynamics or MuJoCo continuous collision detection.

## Numeric product and signed bundle

```bash
python -m sentinel_evc run --help
python -m sentinel_evc verify --bundle ../results/product-integration/v2/bundle --public-key ../results/product-integration/v2/anchors/demo.public --run-id run-1937670cf89444b490aa7296ad537114
python -m pytest -q
```

The run ID above is read from the retained v2 manifest. The public key is the supplied demo trust anchor. Customer trust requires an independently selected public key. The complete existing suite on the experiment host produced 262 passes and no skips. This audit reused the supplied patch's tests and repository checks. It did not write new regression cases.

## Render the saved case

Install `pillow` and `imageio-ffmpeg==0.6.0` in the experiment environment. Select a local font that contains the Chinese caption glyphs. The archive includes official UR5e assets and their BSD-3-Clause license. It excludes font files.

```bash
MUJOCO_GL=egl python experiments/v2_review/render_collision_case.py --assets ../assets/render-assets --case ../results/mujoco-crosscheck/contact_demo_case.json --font /absolute/path/to/chinese-font.ttf --out ../new-collision-video
```

The saved video is a counterfactual kinematic replay. `frames.json` records each frame's joints and contacts. Typography and encoder outputs can differ on another font or runtime. The model and trajectory inputs remain explicitly bound.
