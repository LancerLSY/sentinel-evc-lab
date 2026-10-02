# Sentinel EVC: prospective experiment protocol

The machine-readable sources are [the original cost protocol](../../../../experiments/research_v2/protocol.json) and [the amended intervention protocol](../../../../experiments/research_v2/intervention_protocol.json). They were committed before their respective formal executions (cost: `0ae1c20`; interventions: `ce56e1e`). The intervention amendment fixes descriptor routing and independently asserted branch equality after review, before any formal intervention root ran. It preserves distributions, policies, thresholds and sample sizes. This establishes ordering of this new study; it is not external preregistration. Prior results motivated these interventions and are explicitly development evidence. None of the new formal roots is used to train, choose, normalize or calibrate a model.

## Scientific questions

The previous audit found 100/100 unsafe selections in hidden low friction and 7/100 under shifted cameras. Every original low-friction candidate was unsafe. Enlarging a predictor cannot create a safe candidate or recover information absent from observations. We test a minimal routing repair and preserve its utility cost.

1. **Support routing:** does a frozen visual model in its nominal contract, a camera-independent state model under camera shift, and a declared-friction physical fallback reduce unsafe execution while retaining task completion?
2. **Observability:** can identical observed histories and actions lead to different outcomes when only future friction changes?
3. **Final-plan validity:** do full and incrementally reused verification agree under independently sampled obstacle fields?
4. **Cost:** does static prefix reuse save marginal wall time when dynamic replay remains complete? A slower result is an admissible outcome.

## WorldGuard design

Six scenes contain 100 independent roots each: nominal, low friction, low mass, high mass, displaced goal, and shifted camera. Their distributions and seeds are fixed in JSON. The statistical unit is the root; six candidate siblings, 110 time frames, policy decisions, and repeated measurements within a root are paired observations.

Training, development scales, PCA, ensemble checkpoints, and calibration quantiles remain exactly the previously frozen W2 artifacts. A separate implementation preflight uses two roots per scene, from distinct 790000-series blocks. It verifies APIs and file shapes only. A scientific policy or threshold change after viewing a final outcome requires a new protocol and unused seed blocks.

Each root saves complete `mjSTATE_INTEGRATION`, observed history, and candidate targets. Six siblings (0.6, 0.9, 1.2, 1.6, 3.2, 4.8 s) are replayed from that state for 5 s followed by a 0.5 s hold. W2 only predicts the original first four actions over its trained 2 s horizon. The long-horizon physical fallback is a separate policy; it is never submitted to W2 outside its forecast contract.

| Policy | Allowed information and behavior |
|---|---|
| Fixed 1.6 s | common observed-support abort, otherwise execute; cross-profile stress baseline |
| Fixed 4.8 s | common abort, otherwise execute; explicit utility/cost comparator |
| Declared physical floor | if a profile supplies `mu_min=.015`, select the shortest 1.6/3.2/4.8 s plan satisfying the acceleration screen; otherwise UNKNOWN |
| Reject all | zero-task-utility control |
| Integrated support gate | nominal: visual W2; camera mismatch: state W2; declared low friction: physical floor; unsupported mass/goal: UNKNOWN |

The floor screen checks `norm(a_xy) <= mu_min*(g+a_z)` and `g+a_z>0` at every target step. It is a conservative target screening heuristic, conditional on the declared friction floor. Servo dynamics and contact errors remain measured outcomes. It is not an online friction estimator or a physical guarantee.

Support bounds are **declared configuration evidence**, not inferred ground truth or learned OOD detection. Policies never receive the evaluator's actual friction, mass, future trajectory, risk or drop labels. Nominal profile declarations can be wrong in deployment: a separate counterfactual audit makes this failure explicit. A camera digest mismatch routes to state W2 only because that branch has observed object state and unchanged nominal physics/actions; such object-state sensing is not assumed available on an arbitrary real robot.

## Outcomes and uncertainty

Co-primary outcomes per scene are:

- unsafe executions / all roots;
- completed tasks / all roots, counting every rejection as incomplete;
- rejected roots / all roots.

Unsafe means any relative XY excursion beyond 0.06 m, payload floor contact, or carried-object drop envelope violation during the full 5.5 s evaluation. Completion additionally requires final tray target error at most 0.01 m. This is **tray transport completion**, not robot PickPlace success. Selected-only safety, drop count, duration, and 2 s W2 joint XY coverage are secondary outcomes.

Use two-sided Wilson 95% intervals for binary rates. Zero observations among 100 roots leaves a 3.70% Wilson upper bound. Policy differences use 10,000 paired-root bootstrap resamples, seed 20261006. Report every policy and scene, including all-reject and failed cases. No unplanned significance claim, cherry-picked metric, or universal zero-risk claim is permitted.

## Same-information counterfactual

One hundred new roots, seeds 716000–716099, each freeze one nominal pre-action state and observed history. Clone it into future contact friction 0.3 and 0.025, with otherwise identical mass, state and target bytes. Replay 1.6 and 4.8 s targets in each branch. Hash the permitted inputs, assert equality, and retain both branch outcomes.

This deliberately changes **future** contact friction after observation. It is a causal mechanism diagnostic of an unobserved future/environment change; it is not evidence that fixed friction cannot ever be identified from informative prior motion, and does not estimate real-world friction-change frequency.

## UR5e validity and fair cost

The official MuJoCo Menagerie UR5e model is pinned to upstream commit `4d038b3feae26ec82b46a4d586379114012a8ac7`. Sixty roots use six predeclared strata and independent obstacle/plan random generators. Obstacles are sampled before any trajectory, unlike the earlier trajectory-constructed fixtures. These are still stratified simulator scenes, not field incident prevalence.

The strata preserve code identifiers: clear, suffix perturbation, smooth suffix bump (`narrow_passage` legacy identifier), joint-limit violation, tracking disturbance, environment change. A label does not force an unsafe outcome. Separate 1 ms / 0.005 rad review scans label final plans; review is excluded from gate timing. Full and incremental gates use their original 2 ms / 0.01 rad resolution, so disagreements can be observed and retained.

Parent reuse binds exact prefix, integration state, environment, model/assets, time/control settings, tracking threshold, and validator source digest. Failed reuse triggers a complete final-plan check. Dynamic replay always begins at frame zero.

Each method receives one excluded warmup and three stored repeats with cyclic order rotation. Report parent validation plus record construction `C_parent`, final full validation `C_full`, and record check plus suffix/full-fallback validation `C_delta`. Primary timing compares `C_delta` against `C_full` after a parent record exists. Equal upstream costs are shown for **both** methods: `C_parent/K+C_full` and `C_parent/K+C_delta`, for K=1/2/4/8. These are analytical amortization scenarios, not measured repeated production use.

Report per-root medians, cross-root median/P95, paired cost differences and ratios with 10,000-root-bootstrap intervals. Report false allow among independently unsafe roots, false reject among safe roots, full/incremental agreement, fallbacks and scan/rollout work. Keep negative timing and conservative false rejections.

## Reproducibility and interpretation

Protocol, runner sources, trained model manifests, calibration metrics, feature cache and encoder are SHA-256 bound. Output directories must be empty. Preserve per-root outcomes, tensors, timing repeats, sources, assets and full manifests. GPU hardware is RTX 4090 D (24 GB); MuJoCo physics is CPU work.

This study evaluates bounded simulation routing and verification. UR5e is not controlled by the SO100 overlay. Genuine VLA closed-loop experiments use a separately matched official checkpoint/environment and separate protocol. No hardware motion, continuous-collision proof, safety certification, or sensor-truth claim follows from these results.

References: [official LeRobot LIBERO interface](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx), [SO arm calibration migration](https://github.com/huggingface/lerobot/blob/main/docs/source/backwardcomp.mdx), [conformal prediction under covariate shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).
