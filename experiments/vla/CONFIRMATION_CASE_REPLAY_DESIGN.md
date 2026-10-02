# Confirmation case replay design

## Scope and fixed selection

This is a post-hoc exact-action diagnostic of the completed full-suite
confirmation. It does not call the policy, change the frozen evaluation, or
add benchmark episodes.

- Failure: task 4, fixed state 30, seed 44021. This is the first failed task-4
  cell in the declared state order. The source outcome is 280 steps, reward 0,
  no termination, no truncation, and `is_success=false`.
- Success: task 4, fixed state 32, seed 44023. This is the first successful
  task-4 cell after that failure. The source outcome is 126 steps, reward 1,
  termination, no truncation, and `is_success=true`.
- Instruction: pick up the black bowl in the top drawer of the wooden cabinet
  and place it on the plate.

Selection is fixed before replay. No alternative state may replace either
case after observing the replay.

## Source and environment binding

The replay requires confirmation manifest SHA-256
`ec87b22d166dd167c3549d3e976bfe29d10c2de128aeef2218ceb67bdcd7efac`
and trace SHA-256
`a78454f9c5e138e056cd61d9243c7ac7fff1b6855df161d6e8903d1cb38792ab`.
It also verifies the frozen confirmation runner/protocol receipts, checkpoint
files, full asset tree, task source hashes, MuJoCo 3.3.7, reset state and seed,
the first processed state/camera hashes, every stored `env.step` action, and
every stored official outcome. Any mismatch stops capture.

The formal environment declaration contains all ten task IDs. Replay projects
that frozen list to task 4 only to avoid constructing nine unused environments.
After JSON canonicalization, every other camera, control, timing, feature, and
reset field must remain identical; the manifest records this sole projection.

## Retained measurements

Each case retains the native 360x360 RGB render at the initial reset and after
every recorded action, encoded at the evaluation rate of 20 Hz. The rendered
scene contains the simulated robot mesh and task geometry. Each action record
retains:

- the exact source action digest and official reward, termination, truncation,
  and success label;
- target-bowl, plate, and observed EEF positions;
- target-bowl lift relative to its initial center height;
- Euclidean bowl-center-to-EEF and bowl-center-to-plate distances;
- the uniquely resolved top-drawer joint position and displacement from reset;
- MuJoCo contact pairs involving the target bowl, reported as named simulator
  body pairs and counts. A geometric distance is never labeled as contact.

Body and joint inventories are retained. Capture stops if it cannot uniquely
resolve the target black-bowl body, plate body, or top-drawer joint from the
actual model names.

## Media and interpretation

Raw videos preserve one frame per physical replay state. A bilingual paired
film may extend the shorter success display to the failed case's 280-action
duration by holding its terminal frame. Every held frame is labeled
`DISPLAY HOLD — no action or simulation step` and is excluded from trajectory
statistics. This presentation hold is not a replayed action.

The comparison is outcome-conditioned and descriptive. It may show where the
recorded trajectories differ, but it cannot identify a causal mechanism,
estimate a success probability, or establish policy, Sentinel, or real-robot
performance.
