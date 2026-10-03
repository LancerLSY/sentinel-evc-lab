# Launch Gate competitive evidence

## Result

The measured product difference is a packaged **pre-motion VLA behavioral
qualification**, not stronger generic authorization.

The same retained reference pack and camera-swapped candidate pack were passed
through installed KineGrant 2.65.5 and RLSOK 1.5.12 public APIs. Each final
action was presented as a new, explicitly approved intent. Both products
correctly accepted all 40 calls: 20 reference probes and 20 candidate probes.
Sentinel compared the two packs as a release qualification, returned `BLOCK`,
identified the first changed policy camera binding at
`task00-state46-step00`, and made zero downstream writer calls.

These competitor accepts are not safety failures. They answer “is this newly
approved request allowed?” Sentinel Launch Gate additionally answers “does this
installed adapter still consume and emit the same VLA fields as the reviewed
adapter?”

| Actual installed call | Input supplied | Measured result |
|---|---|---|
| KineGrant 2.65.5 `ActionRequest` → `PolicyEngine.evaluate` → `CapabilityIssuer.issue_scoped` → `Gatekeeper.execute` | Full camera/state/cursor/action probe embedded in the request context; a fresh exact-context policy and capability for every request | 40/40 allowed, 40 dispatches, all receipt chains valid |
| RLSOK 1.5.12 `ReleaseExecutionGate.evaluate/execute` | Exact final-action bytes in a generic request; state snapshot containing camera bindings, ordered state and cursor; separately approved reference and candidate executable policies/configurations | 40/40 allowed, 40 dispatches; native SmolVLA action semantics were not validated |
| Sentinel Launch Gate | Retained reference pack plus candidate pack | `BLOCK`, first camera-key binding mismatch, 0 downstream calls, signed offline reproduction |

## What RLSOK already does

RLSOK is a strong counterexample to any claim that release identity or
configuration binding is unique to Sentinel. The pinned
`diffExecutablePolicies()` call detected the camera-swap candidate's declared
change as:

```text
preprocessor
action contract
execution configuration
approved configuration digest
```

It returned `invalidatesApproval: true`. This is a declared-release comparison,
not a comparison of the observed VLA camera/state/cursor fields. The experiment then constructed a
separately approved candidate release, as an operator would after reviewing a
change. RLSOK bound its model, normalizer, preprocessor, postprocessor, action
contract, controller/configuration, exact final action, generic state snapshot
and runtime attestation. The release gate correctly dispatched the newly
approved requests.

RLSOK 1.5.12 has no opaque/raw contract for the recorded seven-float SmolVLA
action. The harness therefore used its `program` contract as a schema
accommodation while placing the exact action bytes in the generic gate request.
This run validates the generic release-gate path and its data carriage; it does
not validate native relative-Cartesian/gripper action semantics.

The inspected runtime-attestation schema contains source identity/kind/version,
observation time, continuity token and capability names. The execution gate can
also pass the generic state object to a custom `ActionPolicy`. The installed
surface tested here did not itself compare the candidate's observed VLA camera
mapping, state schema and chunk cursor against a retained reference pack or
emit a minimized signed VLA mismatch capsule.

Pinned source:
[`5df8ce9a77349bf1e82e3b1f60aa0cd55138e034`](https://github.com/realitywarden/rlsok/tree/5df8ce9a77349bf1e82e3b1f60aa0cd55138e034).
Relevant APIs:
[`exec-spec.ts`](https://github.com/realitywarden/rlsok/blob/5df8ce9a77349bf1e82e3b1f60aa0cd55138e034/packages/core/exec-spec.ts),
[`execution-gate.ts`](https://github.com/realitywarden/rlsok/blob/5df8ce9a77349bf1e82e3b1f60aa0cd55138e034/packages/core/execution-gate.ts),
and
[`runtime-attestation.ts`](https://github.com/realitywarden/rlsok/blob/5df8ce9a77349bf1e82e3b1f60aa0cd55138e034/packages/core/runtime-attestation.ts).

## What KineGrant already does

KineGrant is a strong counterexample to claims about unique exact-request
authorization, portable receipts or arbitrary context binding. The experiment
placed the complete retained probe in `ActionRequest.context`; the official
request digest bound those fields. A default-deny policy required the complete
context, and every request received a new scoped capability before
`Gatekeeper.execute()`.

This proves KineGrant can carry and authorize the same facts. It does not prove
that KineGrant cannot implement the Sentinel comparison. A customer can write a
policy that compares the supplied context with a retained reference and denies
the changed mapping. The measured product distinction is that Sentinel ships
the VLA-specific capture, alignment, first-difference diagnosis and signed
offline reproducer as one workflow.

Pinned source:
[`3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f`](https://github.com/zoahdev/kinegrant-protocol/tree/3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f).
Relevant APIs:
[`models.py`](https://github.com/zoahdev/kinegrant-protocol/blob/3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f/src/kinegrant/models.py),
[`policy.py`](https://github.com/zoahdev/kinegrant-protocol/blob/3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f/src/kinegrant/policy.py),
and
[`gatekeeper.py`](https://github.com/zoahdev/kinegrant-protocol/blob/3e4df23bcd2092acd5ea6c9b8b9c6994edd2793f/src/kinegrant/gatekeeper.py).

## Differentiated product position

Sentinel ships this workflow:

> Sentinel provides an installed VLA release check that captures the actual
> camera bindings, ordered state fields and units, chunk cursor and final
> action; compares them with a retained reviewed adapter before motion; points
> to the first mismatch; and signs a small decision that reproduces offline.

This is useful when a release remains internally valid but its actual adapter
wiring changed. A release hash says which release is present. An authorization
gate says whether the current request is permitted. Launch Gate supplies a
reviewed behavioral reference and checks the installed adapter against it.

The falsifier is concrete: if KineGrant or RLSOK ships an official workflow
that captures these observed VLA fields, aligns a retained reference and
candidate, returns the first semantic mismatch, gates the writer and exports a
signed minimal reproducer without bespoke integration code, this workflow
distinction no longer holds.

## Reproduction and retained evidence

Command:

```bash
python experiments/product/launch_competitive.py run \
  --launch-result /path/to/local-launch-gate-final-v6-result \
  --kinegrant-repo /path/to/kinegrant-protocol \
  --rlsok-repo /path/to/rlsok \
  --node-modules /path/to/pinned/rlsok/node_modules \
  --output /new/path/result.json
```

Retained local result:

```text
work/product-advantage-20261003/launch-competitive-v1/result.json
SHA-256 21f518797356605182aaf7e8e1f8bb0d50e676150bb058f3e6edda5e4d1d1da2
launch result  sha256:ffb59d50ec8cd5f8229dabe5e700ab7d9f48da60561ab38750a501aa51c51036
reference pack sha256:d0481fdf198b16cc78c9ba48522e24c5d087b0b950f436648b26729b583f48bf
camera-swap   sha256:8a7cb6fb7f1a8e38df3bcd95f43a19769e30ead3997135616b8b9584d46a4b1f
Node           v24.19.0
```

The runner snapshots the result and both packs once through bounded,
`O_NOFOLLOW` regular-file reads. It proves the selected report's source hashes,
suite, ordered probe IDs and input hashes match those exact bytes, then gives
both competitors only private copies of the snapshots. Competitor code is
extracted from `git archive` at each pinned commit, so untracked checkout files
are excluded. The evidence records both archive hashes, the full installed
KineGrant `cryptography` distribution tree, the RLSOK lockfile and full
installed TypeScript, Zod and Node type package trees, and the generated
harness hash.

The run records every decision and request/probe identity. It runs no
new model inference, simulator, robot, GPU or physical motion. The camera swap
comes from the signed recorded-action adapter integration; raw pixels are not
present. The earlier ordinary authorization comparison remains unchanged:
KineGrant and RLSOK each passed the retained 60/60 authorization-fault grid.
