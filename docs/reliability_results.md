# Alignment and reliability validation — 2026-10-01

This report accompanies the [v4 F01–F12 alignment review](design_alignment.md).
The release covers the local numeric product and fixed MuJoCo contact fixture.
Real VLA, GRU, visual WorldGuard, authenticated remote reproduction and hardware
motion qualification remain pending.

## Reproduction protocol

Install the existing optional dependencies and use new, empty output directories:

```bash
python -m pip install -e ".[test,physics]"
python -m pytest -q
python run_tests.py
python tools/validate_scenarios.py --out runs/reliability-scenarios
python -m sentinel_evc physics --out runs/physics-low-friction --seed 7 --friction .10
python -m sentinel_evc physics --out runs/physics-nominal --seed 11 --friction .35
python -m sentinel_evc physics --out runs/physics-high-friction --seed 23 --friction .70
```

A physics exit code of 3 means the complete evidence retained a failed scientific
gate. Check `COMPLETE.json` and independently verify the local index plus all 25
trial bundles with the frozen implementation's result-tree verifier:

```bash
python - <<'PY'
from pathlib import Path
from sentinel_evc.ssh_experiment import _verify_result_tree
for name in ("physics-low-friction", "physics-nominal", "physics-high-friction"):
    result = _verify_result_tree(Path("runs") / name)
    assert result["trial_count"] == 25
    print(name, result["trial_count"], "verified")
PY
```

This internal verifier is the version-pinned local verification path used for
the measurements and product physics reads. The public `verify-physics --out`
command verifies `remote-physics` outputs and additionally requires a valid
`REMOTE_RECEIPT.json`; it must not be used to claim remote identity for local runs.
Preserve
execution/evidence/scientific outcomes separately; do not modify thresholds after
observing the results.

The additional numeric matrix uses logical command time, not robot wall time.
It covers seeds 0/7/23, stricter risk refusal, conservative residual refusal,
translated-library-plan equivalence, obstacle blockage, an unsupported displacement,
restart read/export, tamper refusal and a later independent valid run. Translation
is a library boundary check; the current UI does not expose arbitrary start poses.

Physics variants change seed and friction within the same tray fixture. Each
experiment retains four movement durations at five timesteps, the integrated
selection, revoke, geometry mutation, friction mutation and replay: 25 signed
trials and a signed aggregate. This does not establish arbitrary task coverage.

## Repairs exercised

- Numeric terminal/detail/list verification cannot be bypassed by deleting a
  cached verification field or changing displayed status. Download ZIPs regenerate
  from freshly verified source; corrupt caches are repaired and corrupt signed
  sources are refused. Unique exclusive export temporaries avoid following a
  legacy temporary-file symlink.
- Invalid run/physics/asset/robot metadata is isolated. Constructor failures
  release newly acquired locks. Unfinalized interrupted/finalization failures
  remain non-success records, cannot export and allow a later independent run.
- Preparation stop requests are checked around model/evaluation/publication.
  Finalization has an internal phase that refuses new stop acknowledgments.
  Requests accepted before that phase remain effective in the terminal outcome.
- Numeric/physics workers retain workspace ownership when shutdown times out.
  HTTP handlers complete before shared store teardown.
- Registered prediction hash, plan, deadline and decision bindings are captured;
  later field mutation denies preparation and dispatch.
- Public source copy/SSH packaging refuses root/nested links and special files.
  Only the two audited public benchmark/scenario tools are optionally included.
  The scenario source digest covers its own producer and the public-source manifest.
- Native App startup requires a structured readiness marker and matching random
  launch nonce. Arbitrary log URLs cannot select the backend. Graceful termination
  has a 15-second budget before forced process reaping.
- Pytest discovery is restricted to repository `tests/`, preventing old extracted
  packages under `runs/` from colliding with current test modules.

## Measurement record

The measured code snapshot is [`8c7c840f2f4b2d93f5434943b4cdb77e4762e347`](https://github.com/LancerLSY/sentinel-evc-lab/commit/8c7c840f2f4b2d93f5434943b4cdb77e4762e347).
This report is a subsequent documentation commit. Environment: macOS arm64,
CPython 3.12.14, MuJoCo 3.14.0. The frozen public-source manifest SHA-256 is
`d3a429dcf26bc321617eb19cffc2872b5e46b480994dc1f4ed6a1b38263a36e7`.
The separate physics source manifest, actual configuration, signed-index tips and
artifact digests are retained in the [compact measurement files](reliability/2026-10-01-m4/validation.json)
and [checksums](reliability/2026-10-01-m4/checksums.json).

| Validation | Actual result |
|---|---|
| Full pytest with real MuJoCo and loopback HTTP | 220 passed, 0 failed, 0 skipped |
| Fallback runner with the same optional dependencies | 220 passed, 0 failed, 0 skipped |
| Actual extracted public source archive | 219 passed; only its own recursive rehearsal deselected |
| Additional numeric scenario matrix | 9/9 expected outcomes matched |
| Extended physics experiments | 3 experiments × 25 signed trials; all 75 independently verified |
| Python syntax / Swift typecheck | 54 Python files parsed; native Swift typecheck passed |
| Existing demo/sample/RESULTS regression | 1,000 cases; 2,031 verified events; 7 faults; 4 distinct tamper refusals; all 16 RESULTS rows matched |

These counts describe this suite, not accumulated historical counts. The archive
rehearsal exercises local packaging; no authenticated SSH experiment completed.
No separate Python type checker or linter was installed; syntax checks, runtime
regressions and two independent review lanes were used. The review verdict is
APPROVE for code and WATCH for the scoped architecture; full v4 remains blocked.

### Additional physical scenarios

Each row uses the same predeclared nine acceptance gates, including scientific
convergence and the nominal slip/stable contrast. Gate limits were not relaxed.

| Seed | Friction | Complete evidence | Scientific gates | Selected .6s outcome / max offset | 1.6s outcome / max offset |
|---|---:|---|---:|---|---|
| 7 | .10 | 25/25 verified | 7/9 | drop / 250.3 mm | stable / 9.2 mm |
| 11 | .35 | 25/25 verified | 8/9 | slip / 77.1 mm | stable / 2.4 mm |
| 23 | .70 | 25/25 verified | 7/9 | stable / 23.2 mm | stable / 4.2 mm |

All three failed strict step convergence for the .6s branch; the low-friction
case also failed .9s convergence. `paired_negative` failed at .10 because the
fast branch dropped, and at .70 because it stayed stable. That contrast gate
describes the original benchmark hypothesis, not infrastructure integrity or
successful task behavior. None of the experiments had engine warnings. All
ordinary candidate commands completed with 40/40/40 cursors, while geometry and
friction mutations were refused before dispatch.

Each revoke injection submitted zero new old-generation commands after REVOKE,
observed one accepted-tail command, confirmed cancel/drain, and continued physical
integration for .548 seconds after the request. Payload motion during that tail
and hold was about 17.2/10.0/8.8 mm, respectively. The full matrix, branch outcomes,
convergence differences and revoke traces are summarized in
[physics-results.json](reliability/2026-10-01-m4/physics-results.json).

The geometry-only selector chooses the .6s candidate before sibling future labels
are computed. The drop/slip results show why this profile cannot be promoted to
general consequence-aware three-dimensional execution. A trained, calibrated
final-action 3D predictor remains an acceptance gate. Slow sibling outcomes are
evaluator comparisons, not hindsight inputs to selection.

The numeric [case records](reliability/2026-10-01-m4/numeric-cases.jsonl),
[summary](reliability/2026-10-01-m4/numeric-summary.json) and
[source/environment](reliability/2026-10-01-m4/numeric-metadata.json) preserve the
expected and actual refusal/completion outcomes separately. The source manifest
includes the scenario producer, tests, build metadata and public resources.

Full signed trial trees remain in ignored local `runs/`; compact public files are
summaries with digests, not a replacement for those signed bundles. Reproduce on
the frozen code commit to regenerate complete evidence. Included demo keys
provide integrity under chosen local custody, not external attestation.
