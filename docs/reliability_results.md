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
gate. Check `COMPLETE.json` and independently verify the index plus all 25 trial
bundles with `sentinel_evc verify-physics --out <experiment-dir>`. Preserve
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

The code snapshot is frozen before the extended validation. The following report
commit adds compact measured results and checksums; full signed run trees remain
under ignored `runs/` locally. Included demo keys provide integrity checks under
chosen local custody, not external attestation.
