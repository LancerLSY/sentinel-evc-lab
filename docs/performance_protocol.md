# Local performance protocol

This measures the current CPU reference profiles. It does not measure a VLA checkpoint,
physical robot, authenticated SSH host, UI frame rate or production throughput.

Run the benchmark on a clean checkout with optional physics dependencies installed.
Keep warmup and measured repetitions separate; execute workloads sequentially, without
running the test suite or a second benchmark at the same time. Preserve every sample,
the workload configuration, software environment and measured source identity. Host
wall time uses `perf_counter`; simulation time is reported separately. Filesystem and
import caches are not forcibly flushed, so this is a warmed local workload.

## Workloads

- **Geometry comparison:** same constructed safe/unsafe cases, same parent proof and
  transform workload. One path fully checks every final child; the other attempts
  inheritance and pays for failed fallback. The measured total includes parent
  establishment, transforms, certificate/event work and evidence finalization. Verify
  decision agreement and safe/unsafe counts before comparing timings. Root-proof and
  fallback costs must accompany child full-check counts. This is a scoped geometric
  comparison, not a speedup for the complete robot-policy system.
- **Numeric product:** standard seed-7 physical profile, four candidates, real wall-clock
  pacing, 40 actual simulated observations, runtime authorization and persisted signed
  evidence. Check completed status, all three cursors and evidence integrity. Report ZIP
  export and independent verification costs explicitly. A 2-second observed-command
  horizon is not a 2-second host execution deadline.
- **Physics:** full headless MuJoCo fixed tray/payload experiment with five resolutions,
  25 signed trials, signed aggregate completion and complete returned-tree verification.
  Retain the `.6s` convergence failure and slip result. Faster completion does not turn
  either scientific result into a pass.

## Reporting

P50/P95/P99 are nearest-rank order statistics of directly measured total samples. Never
add per-stage percentiles to construct a total percentile. For small repetition counts,
P95/P99 may both equal the maximum; those values do not estimate a population tail or
provide confidence intervals. Report sample count, minimum and maximum beside them.
Performance comparisons apply only to these fixed workloads and hardware. Failure or
incomplete evidence is a failed measurement, not an excluded slow sample.

Raw sample and manifest digests make the published summary checkable. Benchmark timing
metadata is a local measurement record, not a remotely attested timing source. Signed
trial evidence uses the explicitly supplied demonstration public keys.
