# Experiments left for subsequent increments

`python -m sentinel_evc experiments` lists prerequisites, entrypoints and acceptance expectations. Selecting one with `--experiment-id` prints `PREREQUISITES_REQUIRED` and exits 3; it does not fabricate a result or silently launch unconfigured hardware.

1. **v4 GRU training**: freeze a root dataset with all sibling branches together; fit normalization on train; choose architecture/scales on dev; freeze weights/rule; independent cal and final test. Save dataset hashes, weights, model card, root joint coverage, MAE, action shuffle and no-action ablations. Current standard-library ridge weights are a different baseline.
2. **Real VLA shadow adapter**: select checkpoint/upstream commit, licenses, camera/action descriptor and normalization. Record ten fixed-seed episodes with raw proposal, transformed final action, timestamps and digests; no driver authority in the model process.
3. **MuJoCo/vision**: freeze model/assets, physics parameters and renderer/camera layouts; evaluate link/load geometry and sensor uncertainty; save all difficult/failing episodes, paired candidate roots and layout generalization splits.
4. **Device controller**: characterize submit/accepted/observed, queue depth, cancel accepted versus confirmed, accepted-tail execution, measured braking/load behavior and support sensing. Attach capability table and measured limits before an adapter can authorize motion.
5. **End-to-end cost**: include root proof, candidate transforms, inheritance/fallback, forecast, signing and controller feedback. Warmup/repetition/hardware/source recorded; P50/P95/P99 from end-to-end timestamps, not sums of marginal percentiles.
6. **Distribution shifts / real support**: material/object/load/camera changes, failed sensor or stale scene, support false positives and failed releases. Recalibrate changed model/rule/profile, preserve original negative results.
7. **External evidence trust**: long-lived signer outside runtime, known public key/run ID and independently retained chain anchor; rotation/restart and custody protocol.

No metrics are assigned to these pending items. The baseline CLI writes actual modest experiment manifests with source/environment, root data/model/calibration IDs, acceptance fields and output digests. Its small sample count must accompany any reported metric.
