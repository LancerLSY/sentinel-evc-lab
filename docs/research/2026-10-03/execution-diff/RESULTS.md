# Execution acceptance with recorded evidence

Sentinel adds a verified execution-diff workflow to the CLI and local App:
select two independent integrity keys, align the executions, identify the first
unexpected change, and open the source frame in recorded 3D. The derived report
can be downloaded, recomputed offline and used as an explicit regression gate.

## Actual product result

The complete formal-v3 engineering grid contains 40 paired episodes, 5,460
feedback frames and 5,420 final actions per side. The new comparison found:

| Field | Result |
|---|---:|
| Final-action hashes and recorded action values | 0 differences |
| Post-step feedback | **1 difference** |
| Episode-relative submitted/accepted/observed cursors | 0 differences |
| Step outcomes and terminal results | 0 differences |
| Unexpected authorization changes | 0 differences |
| Expected baseline-to-active authorization changes | 40 episodes / 5,420 frames |
| Task success | 33/40 on both sides |

The only feedback difference is **task 0, initial state 48, seed 44048, step 4**.
Only the `pixels.image2` fingerprint changes. The preceding feedback and current
final action agree. This locates a camera-feedback difference without assigning
its cause to the policy, sensor, renderer or simulator.

Both inputs retain step-4 actions and fingerprints, but their preceding saved
body pose is step 0. The App and project-page comparison label this explicitly;
the timeline also opens subsequent original saved poses. No interpolation or
new dynamics is presented as recorded execution.

## Installed Rerun comparison

[Rerun SDK/CLI 0.38.1](https://pypi.org/project/rerun-sdk/0.38.1/) was installed
in a separate experiment environment. The adapter exports the same complete
action, feedback, state-fingerprint, normalized-cursor and outcome facts into
RRD, with one columnar chunk per episode and an execution-step timeline.

The actual command was:

```bash
rerun rrd compare --unordered --ignore-timeline log_tick baseline-projection.rrd active-projection.rrd
```

| Observed workflow | Actual output |
|---|---|
| Rerun, identical-file control | exit 0 |
| Rerun, baseline vs active | exit 1; one unmatched episode chunk per side, at task 0/state 48 |
| Sentinel, same source records | feedback category, step 4, camera 2, equal preceding feedback/action; references to both recorded frames |
| Sentinel, action/cursor/outcome/authorization regression gate | PASS |
| Sentinel, feedback regression gate | FAIL, preserving the observed difference |
| Downloaded-report recomputation | PASS; modified summary rejected |

This is a measured diagnosis advantage over this installed CLI workflow: Sentinel
returns the VLA-specific first changed step and field, separates expected
permission changes and attaches independently verified source/frame identities.
Rerun correctly detects the inequality. Its other interfaces and custom tooling
may support more detailed analysis; this comparison does not rank all Rerun or
Foxglove workflows or claim an exclusive industry capability.

A fresh Sentinel CLI comparison took **3.596 seconds**, including verification of
306,530,091 bytes of signed input assets, JSON decoding, model validation and
diagnosis. This is one local engineering measurement, not a percentile or an
inference-speed claim. [Acceptance receipt](sentinel-acceptance.json).

Rerun conversion and comparison timings are retained separately in
[the machine result](rerun-comparison.json). Sentinel includes native signature
verification and model/record validation, so the two wall-clock workloads are
not a speed benchmark. The original inference runs used **RTX 4090 D (24 GB)**;
the new offline diagnosis ran on a local macOS arm64 CPU.

## Reproduce the product workflow

Import the two [original bundles and separately retained keys](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/native-vla-product-20261003)
with `native-import`, then:

```bash
sentinel-evc native-compare --data-dir runs/workbench \
  --left native-formal-baseline-v3 --right native-formal-active-v3 \
  --out execution-diff.json
sentinel-evc native-compare --data-dir runs/workbench \
  --left native-formal-baseline-v3 --right native-formal-active-v3 \
  --verify-report execution-diff.json
sentinel-evc native-compare --data-dir runs/workbench \
  --left native-formal-baseline-v3 --right native-formal-active-v3 \
  --fail-on action,cursor,outcome,authorization
```

Select **AI execution → execution differences** in the App to use the same API.
Click either source reference to load the corresponding episode and frame.
Default CLI use is diagnostic; `--fail-on` chooses the regression categories.
Configuration mismatches, incomplete facts and unmatched samples cannot pass
as equivalent. Report verification accepts JSON formatting differences but
recomputes all facts and preserves boolean types.

The experiment-only [RRD adapter](../../../../experiments/product/compare_recordings.py)
requires `rerun-sdk==0.38.1`. Run it with `--help` for bundle, Rerun executable
and output arguments. Raw command outputs are retained beside this report.
[Protocol and interpretation](../../../native_run_comparison.md).

## Product positioning

For teams already running a VLA policy, Sentinel provides an execution acceptance
boundary: authorize the exact final request once, observe the actual writer,
retain a portable signed record, then compare changes at the recorded execution
boundary. This supports integration review, regression decisions and customer
incident reproduction without rerunning GPU inference.

The current product is a local developer preview. Its physical robot entry is
read-only, and the native Panda profile does not establish collision safety.
The diff feature improves execution diagnosis; it does not increase policy
success or repair the seven retained task failures.
