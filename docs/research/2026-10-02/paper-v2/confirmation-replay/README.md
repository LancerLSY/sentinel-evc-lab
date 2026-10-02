# Confirmation replay: retained strict failure

The post-hoc task-4 replay did not complete. It replayed all 280 stored actions
for state 30 after matching the retained instruction, processed robot state,
and both processed camera hashes. The state-32 success illustration then
matched the instruction and processed robot state but failed both processed
camera hash checks. The frozen runner stopped before any state-32 action and
before writing a complete manifest or paired film.

[`failed_status.json`](failed_status.json) retains the exact expected and
diagnostic-reset camera digests, source/runtime/asset bindings, and hashes for
the partial state-30 native RGB artifacts. The failed runner retained the
camera-gate boolean but did not print its observed digests; the listed observed
digests therefore come from a separate no-action reset with the same frozen
state, seed, runtime, and assets, and are labeled accordingly. The original
formal action trace remains
at `runs/paper-v2/libero-confirmation-final-100-v1/action_trace.jsonl` with
SHA-256 `a78454f9c5e138e056cd61d9243c7ac7fff1b6855df161d6e8903d1cb38792ab`.

This is a replay compatibility failure, not a new benchmark result. It adds no
sample to the 100-episode confirmation, changes no denominator, and provides no
success/failure causal contrast. The partial state-30 video may be inspected as
a recorded-action failure illustration only. A success comparison must not be
shown or claimed from this attempt.
