# Product foundation validation

Validation history and the current CLI/desktop increment, exercised on macOS with Python 3.12.14. The v4 GRU, real VLA, visual world model and robot trials remain in [experiment_plan.md](experiment_plan.md).

## Main merge and performance validation (2026-10-01)

PR #3 is merged. With the new reproducible benchmark, the full suite passes **167 tests**.
The measured clean main commit `7a16586` passed CI. [Actual performance results](performance_results.md)
include 20 paired geometry repetitions, 10 complete realtime numeric runs and three
25-trial MuJoCo experiments, plus retained warmups. All completed evidence verified;
the physics profile retained its scientific failure. Delta was slower in the simple
geometry workload. Timing stage scope is explicit, and the raw sample digest was
independently checked before publication.

## CLI, desktop, model and visual workbench validation (2026-10-01)

The v0.3 increment adds managed installation, an AppKit/WebKit macOS App, bounded OBJ/STL/MJCF/URDF assets, read-only robot profiles and persisted local physics jobs. Both entrypoints use the same engine and signed evidence verifier. The App is locally built and ad-hoc signed; distribution signing/notarization and a bundled standalone Python runtime remain pending.

The browser exercise imported the repository's two-link MJCF model, displayed its actual geometry, compiled it with MuJoCo 3.14.0, and completed five settle steps without warnings. The check persisted and remained visible after re-selection. Model inspection leaves `physics_authorized: false`. OBJ import and mock robot diagnostics were also exercised through the installed CLI; browser mock diagnostics explicitly returned `hardware_connected: false` and no commands. No real robot was contacted.

A UI-created physics job completed all 25 signed trials. Its recorded 1,260-frame integrated trace played in the three-dimensional viewport, and the browser downloaded a 262-entry ZIP with no corrupt entries. Independent verification validated all 25 trials. The UI simultaneously displayed completed execution, evidence PASS, scientific acceptance incomplete, and the actual `slip` outcome. It did not turn the failed .6s convergence gate into a success.

The actual native App created a numerical run with 40 submitted/accepted/observed commands, selected the 1.6s branch and displayed evidence PASS. Closing its window removed both the App and its owned Python server process. Native download conversion and standard menu contracts are covered by automated checks. The final NSSavePanel save action was not exercised because the user was actively operating the App window. Browser ZIP download was verified separately.

Desktop and 390px layouts were visually inspected; the narrow model stage remained usable with no page-width overflow. Navigation, empty states, engine-unavailable states, actual geometry, execution charts, recorded playback and export are backed by local data. Completed physics jobs stop polling; re-selection/manual refresh re-verifies evidence. Invalid completed evidence becomes a persisted failure, and no-engine installs disable physics creation with a reason.

![Actual signed physics workbench](screenshots/physics-workbench.jpg)

Final v0.3 checks: **163 passed** with actual MuJoCo available, and the fallback runner **163 passed, 0 failed, 0 skipped**. The earlier 68- and 97-test counts below describe the preceding increments. Tests include core/no-engine states, hostile model expansion/resource requests, malformed API types, failed construction cleanup, App workspace/runtime cache isolation, persisted evidence invalidation, interactive installation and native menu/download lifecycle contracts. CI additionally installs a fresh core environment and checks packaged viewer/Swift/wizard resources.

The interactive installer was exercised in a real terminal through `./tools/install.sh`: select the core profile, choose a new destination, decline App creation, complete all five real installation stages, and display the SSH command without connecting. The generated managed CLI starts successfully. Automated checks also cover the full profile, App choice, existing-directory refusal, invalid input, cancellation/EOF, dependency failure cleanup with retained logs, post-install launch failure, and noninteractive automation compatibility. PowerShell wiring is provided but has not been exercised on a Windows host in this local validation.

The final managed `all` installation includes independent MuJoCo 3.14.0 and an arm64 App with macOS 12.0 deployment target. Its published source snapshot exactly matches the current `src/`, `tests/`, `pyproject.toml`, `README.md` and `LICENSE`; source SHA-256 is `d492ce3179ac731c5ac3fd17d8942309fd3d6a5628b78f70dafdbd77ed8203db`, frozen at `2026-10-01T07:49:46.712826+00:00`. Deep strict ad-hoc signature and plist validation passed. Installation guards refuse destinations inside copied source trees before doing any work, and post-publication progress failures cannot misreport a completed installation.

## Historical automated checks

`python -m pytest -q`: **68 passed**. Tests cover final-plan/certificate bindings, finite immutable contracts, independent root calibration, unknown action families, exact forecast suffixes, lease replay/generation/context/feedback checks, cancel-confirm/drain/recovery, gripper release, sticky log gaps, persistence and signed replay, malformed evidence, export traversal, local HTTP boundaries, workspace ownership, recovery races and evidence-finalization failures.

The fallback `python run_tests.py` runs the same suite without pytest installed. Packaging includes the local HTML/CSS/JS assets in the wheel. CI also runs a product CLI simulation, a residual training artifact smoke and the existing demo/sample checks on Python 3.10 and 3.12. Remote CI status is available from the repository Actions page.

## Historical regression

The existing 1,000-case demo still reports 500 unsafe parent-only releases, zero unsafe final-full or delta releases, and 500/500 accepted safe negative controls. Final-full validation uses 1,000 calls; delta plus fallback uses 500 (excluding parent certificate creation). Seven existing fault injections retain their expected refusal reasons and zero new old-generation submissions after revocation. The generated 2,031-event evidence passes all seven verification layers. An independent geometry cross-check covered 50 cases with zero disagreements; it did not independently cross-check all 1,000 cases.

The original upstream `sample_run` bytes were independently verified by the new verifier: 2,031 events, all seven layers PASS. Existing sample files, old schema contracts and historical RESULTS.md are preserved. Four tamper profiles fail with the expected distinct signatures; changing run_id now also fails the chain and tip identity checks.

## Browser exercise with actual data

The local workbench was exercised through its browser UI: create the standard physical scenario, inspect four candidate outcomes, request stop after six observed commands, drain the accepted seventh command, approve resume, finish all 40 commands, move the replay slider, filter DISPATCH events and generate the downloadable signed ZIP. Final cursor counts were submitted=accepted=observed=40, authority generation=1 and evidence verification PASS. Browser console error count was zero.

The four candidate durations were 0.6/0.9/1.2/1.6 seconds. The first three exceeded the 0.12 consequence threshold; the 1.6-second branch had an upper peak about 0.104 and was selected. Geometry margins were about 0.485. These are calculated scenario outcomes, not UI placeholders.

## Baseline evidence and remaining limits

The two retained [baseline artifact sets](../examples/numeric_baseline/README.md) contain 12 held-out roots and 48 branches each. Both covered all 12 roots but refused every branch at 0.12. This demonstrates a conservative negative result; zero false allows with zero allows provides no availability or accident-rate evidence. The ridge residual mean error was slightly worse than the physical baseline on this split.

Independent release review found no remaining blocking defects after fixes to finalization, persisted evidence consistency and stop/resume races. This supports releasing the foundation increment within its numerical profile. Physical stop dynamics, large-sample calibration, unseen action profiles, training selection, real controller behavior and full v4 experiments remain unvalidated. Host elapsed time and numerical command time are separate; numerical time freezes while waiting for operator recovery.

## Physics/SSH increment validation

The new profile is documented in [physics_ssh_design.md](physics_ssh_design.md). Local Python 3.12.14 / MuJoCo 3.14.0 validation now covers actual three-dimensional contact, five timestep resolutions at fixed 2ms gateway cadence, complete-state pairing/replay, geometry/friction mutation refusal, accepted-tail draining with ongoing dynamics, signed experiment/source/configuration bindings, hostile summary/index tampering and strict SSH archive/transport boundaries. Both pytest and the fallback runner execute the full 97-case suite; the fallback implements restoring `monkeypatch.setattr`, skip and parametrization. With no optional MuJoCo installed, nine physics cases skip explicitly.

The complete local experiment contains 25 independently signed trials and one signed aggregate index. Eight of nine gates passed. Strict convergence failed for the .6s branch: final-position difference about 1.64mm and orientation difference about .095rad at .25/.125ms exceed fixed 1mm/.01rad limits. The .9/1.2/1.6s branches passed. In the 2ms integrated branch the selected .6s Plan slips (~79mm maximum payload offset), whereas the 1.6s branch is stable (~4.5mm). These are actual fixture outcomes, not a safety guarantee or learned 3D WorldGuard validation.

The final source retains the historical 1,000-case demo, all seven injection results, 2,031-event verification and four distinct tamper refusals. The original `sample_run` and `RESULTS.md` remain unchanged. CI adds a separate optional headless MuJoCo job and uploads completed evidence even when a scientific gate legitimately returns exit 3; CI success does not make that gate true.

No live SSH job has been completed in this increment. Tests exercise transport with mocked SSH, plus real local MuJoCo experiments. Live reproduction remains gated by a user-selected authenticated host. The launcher itself uses existing keys/agent and known_hosts; it never reads password comments. An unsigned local transfer receipt requires trusted custody and does not constitute transferable remote attestation.

A clean local bootstrap rehearsal also completed successfully: stdlib-only parent Python, fresh private venv, wheel/dependency installation, all 97 remote-style tests, 25 real physics trials, venv-based full evidence verification and bounded result archive. The experiment returned 3 with complete negative evidence. This tests the actual job program end to end without an SSH connection; it does not replace authenticated host reproduction.
