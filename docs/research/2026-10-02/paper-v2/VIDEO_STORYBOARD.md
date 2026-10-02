# UR5e paper video storyboard — frozen before rendering

## Purpose

The video explains one mechanism: a validation result for a parent trajectory cannot authorize a changed final trajectory. It uses the verified full-mesh MuJoCo Menagerie UR5e model and stored high-resolution joint states. It is an explanatory replay, not a new experiment and not a statistical sample.

The video must make four facts visible without relying on narration:

1. what task the robot is attempting;
2. what changed after the parent check;
3. what the parent-only branch would do;
4. what the final Sentinel gate permits the executor to send.

## Fixed cases and order

The renderer selects exact root IDs. Selection does not depend on a render-time search.

| Chapter | Root | Scenario | Frozen review | Story role |
|---|---:|---|---|---|
| 1 | `0` | `clear` | safe | Negative control: an unchanged safe plan is allowed and executes. |
| 2 | `10000` | `late_suffix` | collision | The final suffix changed after the parent check; parent-only would permit the counterfactual collision trajectory, while the final gate rejects it. |
| 3 | `50000` | `environment_change` | collision; context changed | The obstacle/context change invalidates reuse; the final gate rejects the candidate and the executor holds. |

These cases explain the mechanism. Aggregate numbers shown in the outro come from all 180 independently reviewed roots. The three displayed IDs must never be described as an unbiased sample.

## Visual grammar

- Output: 1920×1080, 20 fps, approximately 20 seconds.
- Motion: 41 stored states, 50 ms between states, displayed at 1×.
- Left pane: stored independently reviewed final-plan trajectory. On rejected cases it is labelled **counterfactual — not dispatched**.
- Right pane: Sentinel final-gate behavior from the same initial state. Allowed controls replay the stored trajectory. Rejected candidates remain at the initial state because no candidate motion is dispatched.
- Red means the independent replay has reached an observed obstacle-contact state.
- Blue means the final gate rejected the candidate before dispatch.
- Green means the exact plan was allowed and executed in this simulation evidence replay.
- The phase badge changes from **shared prefix** to **transformed suffix** at frame 20.
- Every motion frame shows task, change, gate decision, review reason, measured gate wall time and simulated trajectory time.

Chinese labels are included for the case title, task, change, phase, decision, review reason and aggregate result when a CJK font exists on the render host or is supplied with `--font-path`. English labels remain complete on hosts without a CJK font.

## Chapter timing

1. Intro card: what left and right mean; explicitly says MuJoCo simulation and frozen evidence.
2. Safe control card and split-screen motion.
3. Changed-suffix card and split-screen motion; poster is taken at the first visible obstacle contact, when available.
4. Scene-change card and split-screen motion; context invalidation is named on the card and lower explanation band.
5. Aggregate outro: parent-only false allows, full-gate observed false allows, and the scope boundary.

Title and hold frames are visibly distinct from motion. There is no interpolation that could create unrecorded robot states.

## Evidence and claim boundary

Inputs are verified against `full-v3/review/verification_manifest.json` before rendering:

- `reviewed_per_root.json` SHA-256;
- `highres_replay_trajectories.npz` SHA-256;
- portable UR5e model tree SHA-256.

The output manifest records input hashes, exact root IDs, output hashes, frame count, duration, environment and visualization semantics.

The video may say that, within the 180 constructed MuJoCo roots and the reviewed constraint family, the full final gate had 0 observed false allows among 139 reviewed-unsafe roots. It may not say that Sentinel guarantees safety, provides a continuous collision certificate, validates real hardware, or controls UR5e with SmolVLA. Static sampled checks and high-resolution dynamic review remain distinct.

## Render command

Run on the prepared GPU host after the paper protocol is frozen:

```bash
MUJOCO_GL=egl python experiments/arm/render_paper_video.py \
  --run runs/arm-20261002/full-v3 \
  --portable-model runs/arm-20261002/media-v4/model/portable_ur5e \
  --font-path /path/to/SourceHanSerifSC-Regular.otf \
  --out runs/arm-20261002/paper-video-v1
```

The output directory must not already contain the named video, poster or manifest. The renderer encodes frames directly with PyAV/libx264 and does not create an unbounded frame directory.
