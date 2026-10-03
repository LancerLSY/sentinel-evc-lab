## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-03
- Verification Status: VERIFIED
- Version Label: sentinel_evc_v2_cpu_reproduction_v1

## Validation Report

- **Source**: Claude v2 patch pilot artifacts
- **Overall Confidence**: CAUTION
- **Scope**: CPU-only reproduction of D1, D2 and D4. This report does not validate the UR5e/MuJoCo arm experiment or a real VLA closed loop.

### Executive finding

The supplied CPU evidence is reproducible at the level that matters most for correctness. The 17-class contract matrix and all eight pipeline configurations reproduce byte-for-byte. The numeric benchmark reproduces every generated scenario outcome, rejection count, segment-accounting count and decision: 4,320 streaming cycles plus 1,000 one-shot candidates show no disagreement with the v1 full-check reference for any of the three alternative decision paths.

The performance direction reproduces, but the exact multipliers do not. On this Apple M4 host, v2 delta is 3.44–14.14x faster than the faster full checker for M>=16 on pooled medians, versus 2.94–11.98x in the supplied pilot. In the one-shot W1 workload it remains slower than v1 full check, but by 4.7% here rather than 26% in the supplied pilot. These are environment-sensitive timings and must not be presented as hardware-independent constants.

### Provenance

| Item | Identity |
|---|---|
| Patched code | `bdf9d49e1444ba57db69afe6e42b5c5e154614f1` |
| Patched `src/sentinel_evc` tree | `a6f20cf966ca53d174aa1c7d046b08bcbac8b433` |
| v1 baseline | public main `45cc88347ea812a84c5021fc793387dce6bea295` |
| Baseline `src/sentinel_evc` tree | `9e78747b664c0b155e33b4661498c5198c5723d1` |
| Host | Mac mini Mac16,10, Apple M4, 10 cores, 16 GB |
| OS/runtime | macOS 27.0 build 26A428, arm64; Python 3.12.14 |
| Timing controls | Numeric runs executed without the other experiment processes; CPU affinity was not pinned |

Exact commands, script hashes and output hashes are in `results/cpu/run_manifest.json`. Product source files were not modified; `git diff --exit-code` passed after the runs.

### Reproducibility verdicts

| Experiment | Determinism class | Original | Re-run | Verdict |
|---|---|---:|---:|---|
| D1 contract matrix | Deterministic | five modes x 17 classes | same JSON bytes and SHA-256 | REPRODUCIBLE |
| D4 pipeline model | Deterministic | 8 configurations | same JSON bytes and SHA-256 | REPRODUCIBLE |
| D2 W1 decisions | Deterministic structure, environment-sensitive timing | 1,000/1,000 agreement for each of 3 methods | 1,000/1,000 for each | REPRODUCIBLE |
| D2 W2 structure and decisions | Deterministic structure, environment-sensitive timing | 108 runs, 4,320 cycles, 210 rejects | all fields identical except timings | REPRODUCIBLE |
| D2 timing multipliers | Environment-sensitive | M>=16 median 2.94–11.98x | 3.44–14.14x | PARTIALLY_REPRODUCIBLE |

The W2 semantic comparison included `(H, M, jitter, repetition, cycles, rejected, agreement, reused/inherited/rechecked segments)` for all 108 runs; it matched exactly. Raw timing arrays were intentionally excluded from exact comparison.

### D1: contract fault matrix

The re-run exactly reproduces:

| Pattern | Constructed classes blocked |
|---|---:|
| Check-then-execute ablation | 1/17 |
| Signed-token ablation | 6/17 |
| EVC v1 main | 11/17 |
| EVC v2 | 17/17 |
| EVC v2 pipelined | 17/17 |

This supports a precise product claim: in these 17 deterministic software-contract constructions, the shipped v2 implementation blocks all injected invalid-authority paths, including the six classes missed by v1. It does not establish a population-level fault interception rate and does not compare against named external products. P1 and P2 are source-level ablations of this repository. The patcher also does not assert that every intended source substitution occurred, so future source edits could silently make these ablations less faithful unless the harness gains a replacement-count check.

### D2: numeric delta certificate

#### Decision equivalence

| Workload | Candidates/cycles | v1 delta agreement | v2 full agreement | v2 delta agreement |
|---|---:|---:|---:|---:|
| W1 one-shot | 1,000 | 1,000 | 1,000 | 1,000 |
| W2 streaming | 4,320 | 4,320 | 4,320 | 4,320 |
| Added M=2/3 boundary probe | 1,440 | 1,440 | 1,440 | 1,440 |

The supplied experiment therefore reproduces 5,320 original candidate/cycle decisions with zero disagreement per alternative checker, plus 1,440 new boundary-probe cycles with zero disagreement. This is evidence for these generated numeric profiles, not a proof of all floating-point inputs.

#### Timing

| Measure | Supplied pilot | Apple M4 re-run |
|---|---:|---:|
| M>=16 median speed-up vs faster full checker | 2.94–11.98x | 3.44–14.14x |
| M>=16 P95 speed-up | 2.36–9.94x | 2.06–8.21x |
| M=1 median speed-up | 0.56–0.81x | 0.92–1.05x |
| W1 v2 delta vs v1 full | 26% slower | 4.7% slower |

The negative W1 direction reproduces, but its magnitude does not. The statement "M=1 is slower" is not stable on this host: cells range from 0.92x to 1.05x. The robust claim is that M=1 is around the crossover and provides no dependable speed advantage.

The added M=2/3 probe used the supplied `w2` implementation unchanged: H in {16,40,128}, jitter in {0.0005,0.003}, three fixed-seed repetitions and 40 cycles per cell.

| Obstacles | Median speed-up range | P95 speed-up range | Interpretation on this host |
|---|---:|---:|---|
| M=2 | 1.19–1.42x | 0.97–1.47x | median improves; one high-jitter tail cell is slightly slower |
| M=3 | 1.33–1.84x | 0.93–1.80x | median improves; one high-jitter tail cell is slower |

This moves the median crossover to between M=1 and M=2 on this Apple M4 run, but tail latency still depends on jitter and fallback frequency. A product selector based only on obstacle count would be too coarse; the measured fallback/rejection behavior and the target hardware matter.

The timing protocol remains a pilot: three repetitions, no CPU pinning, no warm-up protocol, no confidence interval and many cells. Use cell-level medians/P95 and ranges; do not publish one universal speed-up number.

### D4: pipelined one-time leases

The eight rows reproduce byte-for-byte. The strongest supported configuration is `pipeline + K=1` in the reference simulator:

- 40 actions finish in 41 control ticks, versus 80/60/50/45 ticks for stop-and-go K=1/2/4/8.
- It has 0 idle ticks and 50 ms mean/max authorization age.
- Larger pipelined leases still finish in 41 ticks but increase authorization age up to 535 ms mean and 800 ms max at K=8.

Thus the useful product choice is specifically `pipeline + K=1`. The generic sentence "pipelining preserves 50 ms freshness" is false for K>1. The experiment is a discrete SimController tick model; it is not robot wall-clock evidence and does not measure certificate-computation overlap.

### Warnings

| Type | Detail | Affected claim |
|---|---|---|
| Comparator scope | P1/P2 are repository ablations, not installed third-party products | Competitor/exclusivity language |
| Sample meaning | One deterministic case per fault class | Any percentage, rate or confidence interval for D1 |
| Timing portability | Exact multipliers changed substantially on Apple M4 | Universal speed-up claims |
| Selection | W2 searches for a free path and does not report failed construction attempts | Workload representativeness |
| Tail behavior | M=2/3 medians improve while some high-jitter P95 cells do not | Hard crossover rule |
| Model boundary | Pipeline benchmark is a tick simulator | Real robot throughput/freshness |
| Evidence boundary | CPU run does not cover D3 or real VLA V6 | Whole-system completion claim |

### Fallacy Scan

- **Coverage**: 11/11 statistical and methodological fallacy types checked.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | Aggregating across horizon and jitter can hide cell reversals; report ranges and cells, especially P95. |
| Ecological fallacy | NOTE | No group-to-individual inference is made; unit is a constructed case or control cycle. |
| Berkson's paradox | CAUTION | W2 conditions on finding a free path, which can select easier geometry; failed search attempts are not recorded. |
| Collider bias | NOTE | No adjusted observational model or control-variable conditioning is used. |
| Base-rate neglect | CAUTION | Equal deterministic fault classes have no relation to real deployment prevalence; 17/17 is not predictive value. |
| Regression to the mean | NOTE | No pre/post selection on extreme scores is used. |
| Survivorship bias | CAUTION | `w2` drops a cell if no free path is found. All expected rows appeared here, but construction-level rejection counts remain unreported. |
| Look-elsewhere effect | CAUTION | Many cells and latency summaries are inspected without a confirmatory correction; avoid highlighting only the largest multiplier. |
| Garden of forking paths | CAUTION | The pilot is explicitly non-frozen and not preregistered; the new M=2/3 probe is exploratory. |
| Correlation != causation | CAUTION | The ablations causally test this code under injected faults, but cannot establish effects for external products or physical safety. |
| Reverse causality | NOTE | Deterministic interventions precede measured outcomes; no directional observational claim is used. |

### Bottom-line claim boundary

CPU evidence now supports these three statements:

1. The v2 implementation closes six specific contract classes missed by the main-branch v1 implementation in the supplied 17-class deterministic matrix.
2. Its numeric delta checker matches the v1 full-check decision on all 5,320 supplied cases/cycles and all 1,440 added crossover cycles; performance improves materially as obstacle count grows, while one-shot and low-obstacle cases do not have a stable advantage.
3. In the reference controller model, one-step chained leases remove stop-and-go gaps without increasing the 50 ms authorization age.

It does not yet support "only we can do this," a named-competitor superiority claim, real-robot throughput, or end-to-end VLA task-success improvement.
