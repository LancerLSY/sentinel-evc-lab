# Integrated CPU rerun

This rerun uses the final integrated working tree after the scene-hash, index-boundary and contract hardening changes.

- Numeric W1: all three alternative methods agree with v1 full check on 1,000/1,000 cases.
- Numeric W2: all three methods agree on 4,320/4,320 cycles; 210 rejected cycles and all reused/inherited/rechecked counts are identical to the earlier reproduction.
- M>=16 median speed-up over the faster full checker: 3.30–13.70x.
- M>=16 P95 speed-up: 2.02–8.14x.
- M=1 median ratio: 0.84–0.99x, so the integrated implementation has no low-obstacle speed advantage.
- W1 v2 delta is 51.8% slower than v1 full when parent/index/scene costs are included. This one-shot negative result is stronger than in the earlier local run.
- Pipeline JSON is byte-identical to the supplied pilot: K=1 remains 41 ticks and 50 ms authorization age; K>1 remains 41 ticks with older authorization.

Exact source hashes, commands and output hashes are in `manifest.json`. Timings are environment-sensitive; structural decisions and pipeline values are deterministic in this protocol.
