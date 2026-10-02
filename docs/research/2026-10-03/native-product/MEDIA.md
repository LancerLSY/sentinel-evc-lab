# Watch the action that actually crossed the boundary

The [interactive project page](https://lansiyao.com/research/sentinel-vla/project/)
uses each task's compiled MuJoCo visual meshes and retained body poses. Select
a task, scrub the time line, and compare the seven-dimensional native request,
one-use permit, submitted/accepted/observed run cursors and actual task outcome.
The dual-camera video is a separate player for the same selected trajectory.

| Selection | Task | Actual actions | Outcome | Video |
|---|---|---:|---|---|
| Predeclared task 0 / state 46 | Bowl between plate and ramekin → plate | 77 | Success | [Dual-camera replay](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/native-vla-product-20261003/companion-task00-state46.mp4) |
| Predeclared task 4 / state 46 | Bowl in the cabinet's top drawer → plate | 131 | Success | [Dual-camera replay](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/native-vla-product-20261003/companion-task04-state46.mp4) |
| Predeclared task 5 / state 46 | Bowl on the ramekin → plate | 83 | Success | [Dual-camera replay](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/native-vla-product-20261003/companion-task05-state46.mp4) |
| First ordered failure, task 0 / state 47 | Bowl between plate and ramekin → plate | 280 | Failed at step cap | [Post-hoc diagnostic](https://github.com/LancerLSY/sentinel-evc-lab/releases/download/native-vla-product-20261003/companion-task00-state47.mp4) |

The first three cells were declared before the formal run. The failure is a
post-hoc diagnostic of an already counted failure; neither rendering nor
diagnosis adds benchmark samples. All seven formal failures remain in
[RESULTS.md](RESULTS.md).

## What each view shows

The task-specific mesh view supports orbit, zoom, play/pause and step selection.
Its pose is tied to the selected recorded step. Native actions are grouped as
translation request, rotation request and gripper request; these are the raw
official float32 requests before the existing controller's downstream mapping.
The three cursors count the entire run, not just the displayed episode.

Each 1440×876 H.264 movie contains two upright 720×720 presentation renders,
one external camera and one wrist camera, at the simulator's 20 fps playback
clock. Captions show the source authorization, shortened permit identity, raw
action, outcome and audit status. This clock does not establish wall-clock
real-time inference performance. Video controls and interactive 3D controls
operate independently.

The movie generator reconstructs the exact frozen environment and initial
state, then replays the retained official actions without policy inference.
It checks **every action, every signed 360×360 observation hash, every outcome,
the compiled model identity and every retained pose** before completing a
companion. Presentation renders are captured after those verified transitions;
no extra physics steps are introduced. Four companions report **zero mismatches**.
All four movies were fully decoded and their frame counts, dimensions, codec,
frame rate and hashes were checked.

The three declared episodes retain a full pose at every step. The failed
episode retains 29 poses over 281 frames; its additional displayed poses come
from the audited reconstruction. Those additional poses are explicitly derived
data, not new signed formal measurements. Original actions, permits, feedback
hashes and outcomes remain the authority. The browser checks the published
transport hashes; independent signature verification requires importing the
original ZIP with a separately selected public key.

## Read the failure

The task definition identifies `akita_black_bowl_1` as the target and
`(On akita_black_bowl_1 plate_1)` as the placement goal. In task 0 / state 47,
the target has contact with one gripper finger at steps 36–42 and 80–84, but
no recorded contact with the plate. Its maximum lift is **9.69 mm**, maximum
displacement **9.95 mm**, and reward remains zero over all **280** actions.
This trajectory supports **contact without sustained grasp, lift or placement**.
It does not isolate policy, perception, friction or controller behavior as a
unique cause. [Measurements and task-definition provenance](failure-task00-state47.json).

## Source and reuse

- Formal runner source: `2ede56bd655554ad03b9dc93731ca1660aae9010`.
- Presentation generator: `a3071f9d7c0b88e60c878c7b350054e34510a9c7`,
  [render_native_companion.py](../../../../experiments/vla/render_native_companion.py).
- [Media source/asset manifest](media-manifest.json),
  [decode and orientation receipt](media-validation.json),
  [independent identity audit](media-root-audit.json).
- [Signed source ZIP and separate key](ARTIFACTS.json),
  [complete release](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/native-vla-product-20261003).

Meshes and task assets originate from LIBERO/robosuite. The delivery retains
their [upstream notices and source links](upstream-notices/license_sources.json);
project code licensing does not relicense upstream assets. Fault authorization
records have no scene poses and are presented as the actual rejection table,
with no fabricated fault motion.
