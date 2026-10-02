# Experimental design and evidence units

Every formal lane fixes its implementation, inputs, grid, comparator and endpoints before execution. Earlier outcomes motivate later designs and remain development evidence for those choices. Git ordering establishes a prospective freeze; it is not external preregistration. The protocols below retain the individual freeze points, excluded compatibility episodes and complete outcome records.

| Scientific question | Independent evaluation unit / grid | Comparator or treatment | Main endpoint and failure accounting | Frozen design |
|---|---|---|---|---|
| Does support-aware routing retain safety and utility? | 600 new roots; six scenes × 100 | Frozen router, fixed 1.6 s, fixed 4.8 s, physical floor, reject-all controls | Risk, completion and rejection over 5.5 s; every rejected root is incomplete | [Routing](PROTOCOL.md), `ce56e1e` |
| Can an unseen future physical change alter outcomes? | 100 new matched pairs | Identical permitted inputs/actions/state; change only future contact friction | Paired unsafe-label disagreement for 1.6 s and 4.8 s | [Counterfactual](PROTOCOL.md), `ce56e1e` |
| Where does the physical fallback hold or fail? | 324 roots; 27 parameter cells × 12 | Fixed 1.6 s, fixed 4.8 s, declared-floor screening | Risk, drops, transport completion and duration; eight corners + four interior points per cell | [Physical stress](BROAD_FALLBACK_PROTOCOL.md), `797c12d` |
| Does final-plan reuse preserve decisions and reduce cost? | 60 fresh UR5e roots; six predefined strata | Parent-only, full, bound incremental/fallback, reject transforms | Independently reviewed false allow/reject, paired marginal cost, equally charged parent cost and P95 | [UR5e cost](PROTOCOL.md), `0ae1c20` |
| How does the official native policy perform in the initial configuration? | 100 rollouts; ten LIBERO-Spatial tasks × states 0–9 | Fixed official checkpoint, MuJoCo 3.8.1, execute 50 / predict 50 | Official success, per-task results, crashes and capped failures; no replacement | [Native policy](LIBERO_PROTOCOL.md), `c3b9f9f` |
| Does more frequent observation address the severe failure? | 20 matched task/state pairs / 40 rollouts; tasks 5 and 0 × states 10–19 | Execute 50 versus 10; unchanged MuJoCo 3.8.1, paired RNG and first observations/actions | Paired official success and traced runtime; task 5 primary, task 0 control | [Horizon](REPLAN_PROTOCOL.md), `d1b4b1c` |
| Does the simulator version alter initial conditions before action? | 20 matched pairs / 40 resets; tasks 5 and 0 × states 10–19 | MuJoCo 3.8.1 versus isolated 3.3.7; no policy or action | Object position/orientation and camera differences; no task-success inference | [Reset audit](RESET_AUDIT_PROTOCOL.md), `c7bfa7d` |
| Does the complete backend version affect actual policy outcomes? | 20 matched pairs / 40 rollouts; tasks 5 and 0 × states 20–29 | MuJoCo 3.8.1 versus isolated 3.3.7; execute 10 / predict 50 in both | Paired official success; crashes remain in denominator; no reset-only causal attribution | [Backend](BACKEND_PROTOCOL.md), `eebfb7a` |
| What is full-suite performance under the declared compatibility configuration? | 100 new rollouts; all ten tasks × states 30–39 | Fixed official checkpoint, isolated MuJoCo 3.3.7, execute 10 / predict 50 | Overall and every-task official success, raw step outcomes and crashes; fixed 100-cell denominator | [Confirmation](CONFIRMATION_PROTOCOL.md), `9b1a48d` |

## Analysis and retained failures

Candidate siblings, video frames, repeated timing measurements and neural action chunks do not add independent samples. Bootstrap analyses resample the stated root or task/state pair and preserve within-unit pairing. Wilson and bootstrap 95% intervals describe these fixed or stratified datasets; they do not qualify a continuous physical domain or a real deployment population. Endpoints are reported without selective significance claims or a post-hoc multiplicity-adjusted discovery claim.

The original 58/100 grid and the fresh confirmation grid use different states, backend and execution horizon. Their percentages are separately reported configurations, not a paired estimate of improvement. The native policy is the pinned official LIBERO/Panda checkpoint; the repository's published SO100 overlay has a separate offline action-reconstruction evaluation. Observe-only recording provides no estimate of Sentinel intervention efficacy.

The stored 3.2 s sibling analysis and the outcome-selected failure video replay are explicitly post-hoc diagnostics. They neither tune the evaluated policies nor add episodes to formal denominators. Successful videos do not replace failed evaluations. Every output is linked to source/input/output SHA-256 values, with raw roots, reset receipts, traces and negative outcomes in the experiment archive.

## Hardware and reproducibility

GPU: **RTX 4090 D (24 GB)**. Contact and UR5e studies use MuJoCo 3.14.0; native-policy lanes pin their own complete simulator versions. The confirmation uses a fresh process with isolated MuJoCo 3.3.7, leaving the 3.8.1 environment intact. No hardware robot motion is measured.

[Complete results](RESULTS.md) · [中文结果](RESULTS.zh-CN.md) · [Physical-study commands](../../../../experiments/research_v2/README.md) · [Native-policy commands](../../../../experiments/vla/README.md)

The optional paired backend replay renderer is distributed as source only. No executed paired-film capture is included in this release; the four published films have separate verified media manifests.
