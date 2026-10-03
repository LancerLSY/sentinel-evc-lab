# Verified execution differences

## Product task

Compare two independently verified VLA recordings, identify the first execution
difference and open its recorded evidence. This supports policy-integration
regression review and failure diagnosis without rerunning a GPU model.

The comparison joins each result to its own replay by episode ID, then pairs
episodes by task, initial-state index, seed and instruction. It compares reset
feedback, exact final-action identity, authorization semantics, episode-relative
execution cursors, post-step observations and outcomes. Random permit IDs,
run IDs and timing measurements are not behavior differences. Expected
baseline-to-active authorization changes are reported separately.

Model/profile/backend/configuration differences make the result a comparison
across configurations; they cannot receive an equivalence verdict. Missing,
duplicate or unmatched evidence cannot establish equivalence either.

A difference is an observation, not a proven cause. Equal action bytes followed
by different camera fingerprints locate a feedback difference; they do not
identify a sensor, rendering or physics defect. Policy-input causality requires
additional input-trace analysis.

## Recorded frame navigation

The diagnostic links to both source episode/frame references. A frame without
recorded body poses is marked explicitly. A nearby saved pose is a reference,
with its own step number; it is never labelled the exact divergence pose.

The report is a deterministic derived JSON document, not a newly signed
execution record. Its input manifests, selected-key fingerprints and file
identities are retained. Offline verification rechecks both input bundles and
recomputes the report. Signature integrity does not establish sensor truth or
physical safety.

## Acceptance and comparison protocol

Freeze this protocol before running the new product workflow. Use the published
formal-v3 baseline and active bundles with their separately retained public keys.
The 40-cell dataset is already known and audited: it is an engineering acceptance
sample, not a new blinded scientific evaluation. Retain every matched/unmatched
cell and all differences; do not select only visually attractive examples.

Acceptance must reproduce the complete paired action comparison and correctly
separate expected authorization changes from feedback differences. Measure one
fresh CLI comparison including input verification, decoding and diagnostic
computation; report input sizes and hardware. Run offline recomputation and a
modified-report rejection, retaining commands and actual exit codes.

For an external comparison, pin Rerun SDK/CLI 0.38.1 in a separate experiment
environment. Export the same action, feedback, outcome and episode-relative
cursor facts to two structurally identical RRD files, using an execution-step
timeline. Compare them using the official `rerun rrd compare` command; retain
version, settings, output, exit code and elapsed time. Confirm an identical-file
control. Conversion time is separate from comparison time. SDK/RRD dependencies
are experiment-only and are not Sentinel runtime dependencies.

Rerun's generic equality check and Sentinel's signed-source diagnostic serve
different tasks. Report the observed output from each. Do not infer that a
competitor cannot support equivalent functionality through another workflow,
claim a speedup from different workloads, or invent human diagnosis times.

Official references: [Rerun CLI](https://rerun.io/docs/reference/cli),
[custom data](https://rerun.io/docs/howto/logging-and-ingestion/custom-data),
[Foxglove comparison](https://docs.foxglove.dev/docs/visualization/comparison-mode),
[LeRobot visualization](https://huggingface.co/docs/lerobot/using_dataset_tools).
