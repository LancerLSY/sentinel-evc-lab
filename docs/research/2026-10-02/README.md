# Research evidence and downloads

The [simulation research release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/simulation-research-20261002)
provides source, retained data and results, the complete pinned SO100 dataset,
official UR5e assets, and trajectory-derived media. Large data and weights stay
in release assets; the compact JSON evidence here remains directly readable.

| Study | Evidence | Scope |
|---|---|---|
| WorldGuard, six new scenarios | [metrics](worldguard-scenarios/metrics.json), [protocol](worldguard-scenarios/protocol.json), [manifest](worldguard-scenarios/manifest.json), [erratum](worldguard-scenarios/provenance_supplemental_erratum.json) | 600 fresh roots; low-friction and camera failures remain unresolved |
| Known-profile physical fallback | [metrics](low-friction-fallback/metrics.json), [protocol](low-friction-fallback/protocol.json), [manifest](low-friction-fallback/manifest.json), [erratum](low-friction-fallback/provenance_supplemental_erratum.json) | 100 additional roots; separate 5 s profile with an assumed friction floor |
| Constructed UR5e cases | [original manifest](ur5e/original_manifest.json), [original per-root outcomes](ur5e/original_per_root.json), [official upstream verification](ur5e/upstream_verification.json) | Deterministic mechanism examples; original evaluator reuse limitations are preserved |
| Independent WorldGuard/fallback review | [review record](independent_review.json) | Bitwise reproduction and metric checks; broad WorldGuard safety claim remains blocked |
| SmolVLA loading | [model card](../../huggingface/smolvla_model_card.md), [loading verification](../../huggingface/smolvla_loading_verification.json) | Generic loader matches the original constructor bitwise on one development example |
| Figures | [source](../../../experiments/gpu/plot_research_results.py), [input/output hashes](figure_manifest.json) | Saved-metrics plots; no new experiment or seed selection |

The erroneous `frozen_at_utc` labels in the two original protocols are retained
with their original hashes. Supplemental errata explain the available file
ordering evidence. No externally timestamped preregistration is claimed.
Bootstrap and Wilson intervals describe sampled roots; they are not universal
robot safety guarantees. Actual measured hardware is RTX 4090 D. RTX 5090 is
an unmeasured target configuration.

Frozen upstream neural weights and SmolVLA optimizer/RNG snapshots are omitted
from public packages. The model card identifies every required upstream file
by pinned revision and SHA-256. Dataset/model licenses and individual file
hashes accompany the corresponding archives. The private design document and
authentication credentials are excluded.

The [repaired arm run](ur5e/v3/manifest.json), [separate higher-resolution replay](ur5e/v3/review/verification_manifest.json), [reviewed metrics](ur5e/v3/review/reviewed_metrics.json) and [independent review](ur5e/v3/independent_review.json) preserve all 180 constructed cases. Binding fixes and higher-resolution replay do not create a new statistical holdout.

## Published artifacts

All seven release assets were uploaded and their server SHA-256 digests and sizes
matched the local artifacts. Anonymous requests followed the public download
redirects and returned HTTP 200 with the expected content lengths. These records
cover publication and access; the scientific limitations above still apply.

- [Asset names, sizes and SHA-256](release_artifact_index.json)
- [GitHub release and asset identifiers](github_release_receipt.json)
- [Archive contents and per-file verification](release_archive_verification.json)
- [Anonymous download checks](public_download_verification.json)

The source ZIP records commit `17607e766c1e8c7500dab4acf181554a855058d3`.
Later changes on `main` add publication receipts and navigation corrections;
they do not change the archived experiments or their results.
