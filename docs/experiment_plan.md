# Experiments left for subsequent increments

The [2026-10-02 GPU report](gpu_training_results.md) now records completed numeric
GRU, MuJoCo object, visual object and real-data joint training, plus completed actual
SmolVLA fine-tuning and paired held-out evaluation. [The separate experiment scripts](../experiments/gpu/README.md)
provide those reproducible model runs. The built-in registry below still describes
product integration prerequisites; its pending state does not erase completed
GPU training.

`python -m sentinel_evc experiments` lists prerequisites, entrypoints and acceptance expectations. Selecting one with `--experiment-id` prints `PREREQUISITES_REQUIRED` and exits 3; it does not fabricate a result or silently launch unconfigured hardware.

1. **v4 GRU product binding**: strict numerical GPU training, root bootstrap, independent calibration/test and no-action comparison are complete. The default product still uses its standard-library physical/ridge profile. Bind imported GRU weights, its normalization/action family and calibration to the prediction registry before enabling it in that path.
2. **Real VLA shadow adapter**: select checkpoint/upstream commit, licenses, camera/action descriptor and normalization. Record ten fixed-seed episodes with raw proposal, transformed final action, timestamps and digests; no driver authority in the model process.
3. **MuJoCo/vision**: freeze model/assets, physics parameters and renderer/camera layouts; evaluate link/load geometry and sensor uncertainty; save all difficult/failing episodes, paired candidate roots and layout generalization splits.
4. **Device controller**: characterize submit/accepted/observed, queue depth, cancel accepted versus confirmed, accepted-tail execution, measured braking/load behavior and support sensing. Attach capability table and measured limits before an adapter can authorize motion.
5. **End-to-end cost**: [local CPU reference-profile measurements](performance_results.md) now include paired parent/transform/inheritance/fallback/signing costs, full numeric forecast/controller lifecycles and complete MuJoCo evidence. The generic registry item remains pending for configured VLA/device/remote workloads. Warmup/repetition/hardware/source recorded; P50/P95/P99 from end-to-end timestamps, not sums of marginal percentiles.
6. **Distribution shifts / real support**: material/object/load/camera changes, failed sensor or stale scene, support false positives and failed releases. Recalibrate changed model/rule/profile, preserve original negative results.
7. **External evidence trust**: long-lived signer outside runtime, known public key/run ID and independently retained chain anchor; rotation/restart and custody protocol.

Integration items without direct results receive no invented metrics. The GPU
report and compact JSON retain actual model evaluations and all negative results.
The baseline CLI writes modest experiment manifests with source/environment,
root data/model/calibration IDs, acceptance fields and output digests; sample
counts accompany each result.

The first concrete MuJoCo experiment and SSH protocol are now implemented in [physics_ssh_design.md](physics_ssh_design.md). Run `physics --out <empty-directory>` with optional physics dependencies installed. The five-resolution, 25-trial experiment preserves a slipping selected branch and a currently failed fast terminal-pose convergence gate. Real robot/vision/VLA experiments above remain prerequisites, and a live SSH reproduction requires a configured authenticated target. Exit 3 with a signed COMPLETE/index means a completed negative experiment; exit 2 or an absent COMPLETE marker means incomplete execution.
