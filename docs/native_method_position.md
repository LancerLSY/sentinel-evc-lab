# Final-request authorization and replay

![Native policy, exact request, permit, writer and actual feedback](media/native-request-flow.svg)

Sentinel's native integration makes one final policy request the unit of
authorization. The official postprocessor's array shape, dtype and bytes are
bound to the actual preceding feedback, task/model/profile context, queue
revision, environment step and revocation generation. The resulting permit
has a deadline and can be consumed once, immediately before the captured
environment writer is entered.

This addresses a concrete failure: a request can be correct when prepared but
be replaced, replayed, delayed, or submitted after its feedback/context has
changed. Logging the original proposal alone cannot establish which request
crossed the write boundary. Sentinel retains the raw chunk, selected action,
postprocessor output, authorized request and real environment transition, with
separate submitted, accepted and observed cursors.

## Relationship to existing tools

| Tool | Documented role | Sentinel's integration focus |
|---|---|---|
| [LeRobot](https://huggingface.co/docs/lerobot/main/inference) | model/robot deployment, including [asynchronous inference](https://huggingface.co/docs/lerobot/async) | reuse the official processors and native action bytes at the environment boundary |
| [MoveIt Pro ExecutePolicy](https://docs.picknik.ai/api/classes/moveit-pro/behaviors/executepolicy/) | action-chunk blending, joint constraints, robot/world collision checks and controller dispatch | an explicit expiring, one-use final-request authorization record; the Panda profile does not replace these physical checks |
| [MoveIt Pro MCP](https://docs.picknik.ai/how_to/programmatic_sdks/mcp_server/) | deployment/session context and scoped execution/cancellation interfaces | per-request binding to feedback, exact action bytes and local queue/generation state |
| [Foxglove](https://docs.foxglove.dev/docs) and [Rerun](https://rerun.io/docs/howto/logging-and-ingestion/shared-recordings) | recorded-data visualization and shared recordings | signed import/export plus task-specific mesh replay of the same request/decision/transition timeline |

These are complementary integration roles. The
[installed authorization comparison](research/2026-10-03/execution-boundary/RESULTS.md)
also covers KineGrant 2.65.5 and RLSOK 1.5.12. Both admit 10/10 valid requests
and block 60/60 ordinary faults, matching Sentinel on that grid. RLSOK also
preserves its private command copy when the caller mutates its object.

Sentinel's delivered integration accepts the native postprocessor array, owns
its storage, and rechecks live execution facts at an explicit writer-entry
acknowledgement. Its signed native records connect that request to task-specific
mesh replay and first-difference diagnosis. Separate preparation probes explain
these interfaces; they do not establish a universal exclusivity or speed ranking.

## What the method makes measurable

- **Transparent passage:** compare every actual postprocessed request with the
  array passed to the environment, and compare separately inferred baseline and
  active episodes on identical task/state/seed cells.
- **Authorization failures:** inject six invalid-permit/request conditions at
  the real gateway boundary; count denied attempts and forbidden-writer calls
  separately from task success.
- **Cost:** measure authorization and admission before the writer, excluding
  policy inference, simulation and rendering.
- **Reproducibility:** independently choose a public key, verify the retained
  bundle, and inspect actual task geometry, body poses and source identities.

The [raw-request protocol](research/2026-10-03/native-product/RAW_REQUEST_PROTOCOL.md) defines the
grid and boundaries before outcomes are observed. The [native contract](native_vla_gateway.md)
defines permit and revocation semantics.

The current Panda profile provides in-process request integrity. Physical
collision/dynamics/WorldGuard validation, enforced hardware driver isolation
and braking remain separate engineering work. An entered synchronous simulator
step may finish after revocation. A signed record establishes integrity under
the selected key; it does not establish sensor truth or physical robot safety.
