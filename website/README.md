# Sentinel EVC project homepage

Public homepage: https://lansiyao.com/research/sentinel-vla/project/

A static, bilingual research homepage with the recorded UR5e demonstration,
full-mesh 3D trajectory replay, source-backed comparison charts, local installation
commands and experiment downloads. It needs no JavaScript packages or build step.

## Local preview

From the repository root:

```bash
python3 -m http.server 8877 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8877. Serve over HTTP so the replay and chart JSON can load.

## Data and interpretation

- `assets/evidence.json` contains chart values, denominators and six hashed source
  identities pinned to the public repository commit `a6656f852e763a676292310db16fe2c3c7eae374`.
- `assets/ur5e-trajectory-replay.json` preserves all 41 saved frames of roots 10000
  and 20000. Joint and body positions come from the recorded joint trajectories
  forwarded through the pinned MuJoCo model. `assets/ur5e-viewer-model.json`
  contains all 20 original visual meshes (121,783 triangles) and 82 saved-frame
  body rotations from that same model. Exact-equal vertices are welded without
  removing triangles; visual normals are derived. The native WebGL viewer has
  camera presets, frame stepping, speed control and a Wrist 3 path. It does not
  integrate dynamics or interpolate poses. Rejected motion is a counterfactual;
  review labels describe the entire trajectory, with no invented contact times.
  The two cases were selected after review. Geometry, exporter, model and data
  hashes are retained in the model asset and `assets/ur5e-viewer-manifest.json`.
  `assets/LICENSE-UR5E-BSD-3-Clause.txt` retains the model's BSD license.
- Video, poster, workbench screenshot and fallback figure derive from the retained
  project artifacts. Replay metadata includes the model, NPZ, review and media
  identities. Large research archives remain in the linked GitHub Release.
- `SHA256SUMS` records the bundled asset digests. Run `shasum -a 256 -c SHA256SUMS`
  from this directory to verify them.

The UR5e results are constructed-case comparisons; the SmolVLA result measures
held-out recorded-action reconstruction. WorldGuard plots retain the camera and
low-friction failures. The separate low-friction fallback is a MuJoCo research
profile. GPU measurements identify the actual **RTX 4090 D (24 GB)**.
The Hugging Face button links the published
[SmolVLA overlay repository](https://huggingface.co/LancerLSY/sentinel-smolvla-so100).
The GitHub Release retains a mirror; the Hub publication receipt verifies the
21 uploaded payload files against the selected local bundle.

## Editing and publication

Edit `index.html`, `styles.css`, `app.js` and `replay-viewer.js`. Chinese strings are in `app.js`; English
strings are in the HTML. Replace the disabled arXiv resource and software citation
only after the paper URL and bibliographic metadata are available. Keep chart data
bound to retained source files and refresh asset hashes after changes.

The public project page is hosted within the existing personal website at
`/research/sentinel-vla/project/`, linked from `/research/sentinel-vla/`.
This directory remains a portable static snapshot. Deployment configuration and
short-lived source credentials are managed outside this repository. To host independently, serve this directory on any static
hosting provider. Code uses the repository's MIT license; upstream UR5e model
assets retain their BSD-3-Clause attribution in this directory and the linked
demonstration archive.

The academic-page layout reference is [FINGR](https://www.lyt0112.com/projects/FINGR).
Sentinel text, code, figures, data and media are specific to this project.

## Rebuild viewer geometry

With the retained demonstration/model files extracted under their source paths:

```bash
python experiments/arm/export_ur5e_viewer.py --repo-root . --output-dir /tmp/ur5e-viewer
```

This deterministic export requires MuJoCo 3.14.0 and NumPy; it runs `mj_forward`
only. Source paths can be overridden with `--model`, `--npz`, `--reviews` and
`--media`. Copy the geometry and license into `website/assets/`; rename the
export's `manifest.json` to `ur5e-viewer-manifest.json` and refresh `SHA256SUMS`.
No JavaScript dependencies are required.
