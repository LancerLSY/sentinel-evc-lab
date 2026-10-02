# Reproduce the prospective studies

Read [PROTOCOL.md](../../docs/research/2026-10-02/paper-v2/PROTOCOL.md),
[BROAD_FALLBACK_PROTOCOL.md](../../docs/research/2026-10-02/paper-v2/BROAD_FALLBACK_PROTOCOL.md)
and the [results](../../docs/research/2026-10-02/paper-v2/RESULTS.md) first.
The original cost source is frozen at `0ae1c20`, support routing at `ce56e1e`,
and broad physical stress at `797c12d`. Later report/render changes do not alter
those runners. Reproduction produces new timing values rather than promising
bitwise-identical wall time.

## Data and dependencies

The [paper-validation release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/paper-validation-20261002)
contains complete root-level outcomes, initial states, candidate tensors/plans,
protocols, manifests, source snapshots, plots and media. The existing
[GPU release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/gpu-experiments-20261002)
provides the frozen W2 model/calibration files and visual cache required by the
routing study. Use the package paths below after extraction, or substitute the
corresponding directories from a retained GPU run.

These are isolated research environments, not product-core installation commands.
The contact studies used MuJoCo 3.14.0, Python 3.12.3, torch 2.8.0+cu128,
NumPy 2.2.6 and the actual RTX 4090 D (24 GB) host. UR5e physics is CPU work;
rendering uses EGL. Use the pinned official Menagerie asset tree identified by
the run manifest; the portable model archive retains its upstream license.

## Formal contact and cost runs

```bash
python experiments/gpu/run_paper_interventions.py \
  --protocol experiments/research_v2/intervention_protocol.json \
  --state-run /path/to/w2-mujoco-v2 \
  --visual-run /path/to/w2-mujoco-visual-v1 \
  --weights /path/to/resnet18-f37072fd.pth \
  --out runs/reproduction-interventions --workers 6

python experiments/arm/run_paper_cost.py \
  --protocol experiments/research_v2/protocol.json \
  --asset-dir /path/to/universal_robots_ur5e \
  --out runs/reproduction-cost

python experiments/gpu/run_broad_fallback.py \
  --protocol experiments/research_v2/broad_fallback_protocol.json \
  --preflight --out runs/reproduction-broad-preflight --workers 6

python experiments/gpu/run_broad_fallback.py \
  --protocol experiments/research_v2/broad_fallback_protocol.json \
  --preflight-evidence runs/reproduction-broad-preflight/manifest.json \
  --source-commit 797c12d3af04ee4335803c0ea5bd695c9c74d67e \
  --out runs/reproduction-broad --workers 6
```

Outputs must use new directories. The frozen formal seeds are retained for exact
input reproduction; changed scientific conditions require a separately frozen
protocol and disjoint evaluation roots. Do not recalibrate on these final roots.

## Figures and video

```bash
python experiments/research_v2/plot_results.py \
  --interventions runs/reproduction-interventions \
  --cost runs/reproduction-cost --broad runs/reproduction-broad \
  --out runs/reproduction-figures
```

The two 1080p films bind their actual source tensors and rendered output hashes
in the [UR5e manifest](../../docs/research/2026-10-02/paper-v2/video/video_manifest.json)
and [contact manifest](../../docs/research/2026-10-02/paper-v2/friction-video/video_manifest.json).
They render stored simulator states rather than generating hypothetical images.
The UR5e red trajectory is a labelled rejected counterfactual; it is not dispatched.
The contact film is a tray/payload scene, not an articulated robot.

The separate native SmolVLA LIBERO evaluation has its own pinned environment and
[protocol](../../docs/research/2026-10-02/paper-v2/LIBERO_PROTOCOL.md).
Its official checkpoint belongs to LIBERO's arm; the published SO100 overlay is
not mapped to it. Action-path observation alone does not measure intervention benefit.
