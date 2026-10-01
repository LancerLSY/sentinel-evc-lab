# Product foundation validation

Validated on macOS with Python 3.12.14. This is the local numerical product increment; the v4 GRU, real VLA, visual world model and robot trials remain in [experiment_plan.md](experiment_plan.md).

## Automated checks

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
