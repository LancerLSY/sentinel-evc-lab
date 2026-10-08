# Managed closed-loop integration: validation record

Date: 2026-10-08. Scope: the CLI/App job manager and its connection to the existing unified SmolVLA/Panda runner. This record establishes local integration behavior; it adds no GPU task-success result.

| Check | Observed result |
|---|---|
| Existing repository checks | 262 passed, 0 failed, 0 skipped, including the available optional MuJoCo checks |
| Existing product integration and storage checks | 33 passed |
| Core demo, 1,000 fixed cases | 500 safe and 500 unsafe cases; final and incremental gates released 0 unsafe cases; 7/7 contract faults blocked; independent seven-layer verification passed |
| Evidence mutation demo | All four mutations rejected with distinct failure profiles |
| Local HTTP entry | Readiness and job-list endpoints returned 200; unconfigured start and browser-supplied executable parameters were rejected |
| Operator readiness with the retained model files | Checkpoint and backbone file hashes matched; Linux/CUDA/ML dependencies were unavailable locally, so readiness remained false and launch stayed disabled |
| Lifecycle and evidence probes | Stop-before-writer, process ownership, signed terminal snapshots, tamper rejection and retained frame history checked during independent review |
| Native macOS preview | Swift build, plist and local ad-hoc signature checks passed; the launched App backend served the new loop page from a temporary local source workspace |
| Python/JavaScript syntax and diff hygiene | Passed |

The native preview requires its managed Python/source environment. This check does not establish a portable, notarized installer.

The existing headless MuJoCo fixture also ran. It retained its scientific negative result: the integrated payload slipped and the convergence acceptance gate failed (command exit 3). This fixture uses a different local physics profile and is not a new Panda/VLA measurement. A rendered attempt failed on the local CoreGraphics connection; no video was produced from that attempt.

## Remaining acceptance

The new managed entry still needs an actual run on its declared Linux/CUDA/MuJoCo 3.3.7 environment. The previously supplied GPU service closed the SSH connection during handshake, so no new GPU run was completed in this validation. The older frozen A6 results and the excluded native-policy control remain unchanged.

Run the [configured product session](managed_closed_loop.md), retain the raw writer events and signed export, and report task outcome, denials, contacts and measured latency together. Physical robot motion needs a device-specific writer and measured stop behavior; the present hardware entry remains read-only.
