# Bounded raw-request contract: retained failures

The initial native request profile admitted six motion values in `[-1,1]` and
the gripper value in `[-1.1,1.1]`. It preserved the official postprocessor bytes,
but applied the upstream controller's downstream clipping range too early.
The three original signed bundles were independently imported and verified;
[the audit](V2_NEGATIVE_AUDIT.json) records their result, manifest, key and archive
hashes. Source/configuration identities remain in [freeze-v2](freeze-v2/).

| Lane | Planned | Captured | Actual accepted actions | Observed result |
|---|---:|---:|---:|---|
| Direct baseline | 50 | 50 | 5,659 | 47 successes, 3 task failures, zero crashes |
| Active authorization | 50 | 2 | 107 | task 0/state 41 stopped on `ACTION_OUT_OF_PROFILE` |
| Authorization faults | 10 | 2 | 178 | task 1/state 45 stopped on `ACTION_OUT_OF_PROFILE`; 12/12 recorded invalid attempts blocked |

Forty baseline requests in nine task/state cells exceeded the original six-value
motion envelope. The first active denial was `delta_z=-1.0298006534576416`;
the fault lane's denial was `delta_z=-1.0027008056640625`. These were unchanged
requests from the official policy, and the native controller normally clips them
inside `env.step`. Every accepted request matched its postprocessor output
exactly. The early stops are interface compatibility failures, not GPU failures.
The recorder's `crashed=true` field denotes an episode exception here, specifically
`NativeGatewayDenied`.

The active lane left 48 planned episodes unexecuted; the fault lane left eight.
Their partial task fractions do not estimate completed-grid success rates.
The 12 invalid authorization attempts invoked the instrumented forbidden writer
zero times; the planned 60-attempt fault endpoint was not completed.

## Media provenance

The baseline recorder assigned identical video content to three different
predeclared task/state filenames. Their shared SHA-256 is
`bacdeedf4b4ab695c3d24836d68505f70e0dfa3073c6367cc1956fd79af69c81`.
Those ambiguous recordings are retained in their original signed bundle but
excluded from task-specific playback. Recorded action/pose/outcome data are
reported separately. The App disables videos whose verified content hash is
associated with distinct task/state identities.

## Revised evaluation

A separate [raw-request engineering protocol](RAW_REQUEST_PROTOCOL.md) uses the
complete finite float32 representation domain at the named pinned native API,
with fixed shape and dtype. It leaves controller clipping in the upstream
controller and introduces no raw-request numerical modification. The new grid
is disjoint from this v2 grid, with four explicitly disclosed prior diagnostic
cells. These original failures are not overwritten or retrospectively rescored.
