# Unified SmolVLA–Panda–EVC experiment

This experiment measures the complete simulator path: a native Panda VLA predicts
an action chunk, a fixed temporal aggregation changes the chunk, the same Panda
geometry checker evaluates the original and final chunks, and an EVC gateway
authorizes the bytes received by the simulator writer. Each branch continues
from its own resulting observations. Source and configuration hashes are saved
before the formal run. Engineering preflight results are excluded.

## Fixed inputs

- Policy: `lerobot/smolvla_libero`, revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`.
- Checkpoint SHA-256:
  `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
- Backbone configuration/tokenizer: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
  revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The VLM weights come from
  the complete SmolVLA checkpoint; no backbone weights are randomly substituted.
- Assets: `lerobot/libero-assets`, revision
  `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
- Actual GPU: RTX 4090 D, 24 GB. MuJoCo 3.3.7, robosuite 1.4.0,
  hf-libero 0.1.4, LeRobot 0.6.1.
- LIBERO spatial tasks 0, 4, 5; initial states 40, 41; seeds 2026100400,
  2026100401. Six matched roots, three branches each, at most 280 control steps.
- Excluded engineering preflight: task 0, state 49, seed 2026100399.
- Predict 50 actions, execute at most 10, then obtain a new observation.
  From cycle two, final actions are `0.5 * new[0:10] + 0.5 * previous[10:20]`.
  Inputs are actual model outputs. There are no injected unsafe suffixes,
  relocated obstacles, or invented joint-limit/tracking hazards.

Temporal aggregation is an explicitly selected deployment variant. It is not the
default SmolVLA executor or an RTC implementation. Temporal ensembling is also
used in the [official ACT implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py),
but this study's two-chunk rule is specified above and evaluated on its own terms.

## Paired branches

1. `parent_only`: certify the current raw chunk and apply that decision to the
   final aggregated chunk. Save any mismatch between certified and written bytes.
2. `full_final`: certify the final aggregated chunk with a complete check.
3. `delta_evc`: certify the raw parent, attempt inherited clearance for the final
   chunk, and perform a complete check when inheritance cannot establish clearance.

All branches use the same native action limits and gateway. A rejected action
terminates its episode without issuing a replacement movement. Rejections,
crashes, timeouts and missed tasks remain in the denominator. Branches have the
same seeds and initial states; trajectories can diverge after different decisions.

## Geometry and contact policy

The checker uses the compiled Panda collision geometries, declared joint limits,
MuJoCo masks, explicit contact pairs and body exclusions. Intended gripper contact
with `akita_black_bowl_1` is allowed using an explicit list of geometry ID/name
pairs. Table, cabinet and the other bowl receive no blanket exception. The exact
resolved pair list and model/profile hashes are retained.

Certificates establish clearance of the represented joint-linear arm paths
against the non-arm scene held at the chunk-root snapshot. The scene is refreshed
when a new chunk is evaluated; it is not independently certified at every write.
Moving fingers and carried
objects are outside that continuous certificate. The experiment separately
records actual simulator contacts and forecast-to-execution differences. It does
not infer physical safety from simulator clearance, endpoint agreement or task
completion. Internal physics-substep records, when available, are retained with
their sampling interval; they do not erase this scope boundary.

## Outcomes and evidence

- Task success per matched root, using the LIBERO success criterion.
- Confirmed unwanted simulator contacts, including transient contacts when
  physics-substep instrumentation is available.
- Raw parent certified / final confirmed collision, separately from final
  unknown, invalid and unproved decisions. An unknown is not labeled a collision.
- Full versus incremental decisions on identical final candidates; inheritance,
  fallback and query counts. Finite-resolution clearance disagreements are a
  diagnostic, not an independent false-stop oracle.
- Actual inclusive cycle latency, inference and geometry costs, permit/write
  latency. Selected-path component sums are estimates, not observed deployment
  latency or an end-to-end speedup.
- Observation, model output, transformed action, scene, geometry certificate,
  permit and bytes hashed inside the actual simulator writer.
- Complete controller/integration/RNG restoration checks for reversible forecasts,
  plus predicted and executed arm states. Drift invalidates continued admission.

All six roots are listed before formal results are examined. Each episode flushes
its records and candidate arrays. The stop deadline is 05:45 UTC on 2026-10-04,
before the GPU shutdown; incomplete roots are reported explicitly. With six roots,
the result is a bounded integration study. It cannot estimate field incident
frequency or establish a general superiority claim.
