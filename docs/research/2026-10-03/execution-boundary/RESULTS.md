# Execution-boundary product comparison

Sentinel now owns the approved action buffer and checks live feedback, context,
generation and timing at the writer's entry acknowledgement. In the 40 recorded
concurrency probes, the preceding implementation permitted 40 invalid writes;
the updated implementation permitted zero. The caller-buffer cases still
execute the approved copy, while the three stale-entry conditions stop dispatch.

The closest installed authorization systems also pass the ordinary fault grid.
KineGrant and RLSOK each block all 60 invalid attempts, as does Sentinel. These
results establish a shared capability. Sentinel's product contribution is its
native VLA integration: exact postprocessor bytes, live entry acknowledgement,
separate execution cursors, and signed records that open directly in task-specific
3D replay and execution-difference diagnosis.

## Common-input authorization results

Ten recorded float32 actions, one from each LIBERO task, were reconstructed from
the independently verified native baseline. Each was used in seven fixed cases:
valid, action substitution, replay, expiry, feedback change, context change, and
revoked generation. Each application configuration received the same 70 records.

| Configuration | Valid requests executed | Invalid attempts blocked | Writer calls in the 60 invalid attempts |
|---|---:|---:|---:|
| Secure CycloneDDS transport, without an application gate | 10/10 | 0/60 | 60 |
| Same transport + schema reference | 10/10 | 0/60 | 60 |
| Same transport + HMAC/action/nonce/TTL reference | 10/10 | 30/60 | 30 |
| Installed KineGrant 2.65.5 | 10/10 | 60/60 | 0 |
| Installed RLSOK 1.5.12 | 10/10 | 60/60 | 0 |
| Updated Sentinel native gateway | 10/10 | 60/60 | 0 |

DDS provides participant authentication, permissions and encrypted delivery.
It does not interpret these application faults. The schema and HMAC lanes are
declared experiment reference configurations, rather than installed competing
robot products. All replay cases count the first legitimate execution separately
from the second attempted use.

KineGrant was given exact command bytes, dtype, shape and receiver-owned state
through its public request-context API, with full required-context policy,
persistent nonce consumption and revocation. All 50 nonempty official receipt
chains in its formal and exploratory runs verified. RLSOK used its canonical
action content, runtime continuity, configuration and release-refresh APIs.
RLSOK's Evidence entries are audit records, not signed physical acknowledgements.
These strongest supported mappings preserve both baselines' successful checks.

## Concurrent writer preparation

The Sentinel comparison uses actual threads and an entry barrier after initial
admission but before the trusted writer calls `entered()`. There are ten cases
per condition and per implementation, for 80 rows total.

| Change during preparation | Original: invalid writes | Updated: invalid writes | Updated behavior |
|---|---:|---:|---|
| Caller modifies the original array | 10/10 | 0/10 | 10 approved copies executed |
| New feedback arrives | 10/10 | 0/10 | 10 denials, no dispatch |
| Execution context changes | 10/10 | 0/10 | 10 denials, no dispatch |
| Actual preparation delay exceeds 250 ms permit | 10/10 | 0/10 | 10 denials, no dispatch |

Original source: `9161cfd62625e5658b04d477d743ae6351c408b5`. Updated source is
identified by the frozen source hashes in `protocol.json`. Deadline races wait
0.30 seconds on the actual monotonic clock. Ordinary expiry rows use injected
clock values and are reported separately.

Both revisions invoke the preparation wrapper in all 40 cases. In this table,
an invalid write means a downstream software dispatch after preparation,
rather than invoking that wrapper. The result retains separate
`callback_invocations`, `entry_acknowledgements` and `downstream_dispatches`.

KineGrant's additional 40 exploratory callbacks changed external buffers/state
or delayed execution after its native authorization. The callbacks ran; expiry
subsequently prevented receipt creation. RLSOK preserved the original command
when the caller changed its mutable object. Its separate live-store probe
dispatched against the snapshots returned by trusted refresh hooks after that
store changed. These APIs define different entry points. **The exploratory
callbacks are not a common-contract ranking or evidence of a general competitor
defect.** They explain the trusted adapter work that Sentinel's explicit
acknowledgement incorporates. Baselines can implement an equivalent adapter.

## Actual secure transport and controls

CycloneDDS 11.0.1 was built with Authentication, AccessControl and Cryptographic
plugins, linked to an isolated OpenSSL 3.5.8. Separate publisher/subscriber
processes authenticated and delivered 70/70 records byte-for-byte, in order.
Input and receipt SHA-256 both equal
`34fe19434a6e3d0de3eeef1059518f8e54507741324ff8c688428f351cef9acd`.

Forbidden-topic and no-ACL controls returned `-13 / Not Allowed By Security`.
An untrusted-CA peer did not authenticate and delivered zero records. An initial
batch using history depth one delivered 3/70 and timed out; that failure is
retained. The qualified run changed only reader/writer history to depth 128.
The application stage then probes the exact received records with local state
and clock fixtures; it is not an end-to-end robot network deployment.

Sixty additional application controls cover wrong shapes and modified
authenticators. They are derived from received source actions and explicitly
marked as local controls rather than new DDS messages.

## Evidence and reproduction

[Frozen protocol](PROTOCOL.md) · [Machine-readable comparison](comparison.json) ·
[Evidence pack](https://github.com/LancerLSY/sentinel-evc-lab/releases/tag/execution-boundary-20261003) ·
[Experiment entrypoint](../../../../experiments/product/execution_boundary.py) ·
[Integration contract](../../../native_vla_gateway.md).

The pack retains inputs, source hashes, gateway events, signed main results,
baseline adapters and individual outcomes, official qualification output,
transport controls and the history-depth failure. Source identity is
`SHA256(dtype || NUL || canonical_json(shape) || NUL || raw_C_order_bytes)`;
it is intentionally different from SHA-256 of raw bytes alone. A separate
verification confirmed all ten identities, both event-chain sets and the main
bundle under the retained demo public key.

These are recorded-action software-writer experiments on macOS. They add no
new policy inference, GPU task success, physical motion or collision-safety
measurement. The earlier 0.802 ms gateway mean belongs to its original revision
and is not the updated gateway's latency. Signatures establish integrity under
the selected key. Trusted gateway, array implementation and writer remain the
process boundary.

## Product claim

**Approve the final VLA request, preserve its bytes through writer preparation,
and make the execution record independently inspectable.** The measured
improvement is zero invalid writes in the four published concurrency conditions,
with all ten valid requests admitted. Signed native replay and first-difference
diagnosis provide the customer workflow around that boundary.

Global exclusivity is not established: KineGrant and RLSOK implement equivalent
ordinary authorization checks, and custom adapters can reproduce the entry
contract. Product differentiation rests on the delivered integration and its
reproducible evidence, rather than an assertion that others cannot implement it.
