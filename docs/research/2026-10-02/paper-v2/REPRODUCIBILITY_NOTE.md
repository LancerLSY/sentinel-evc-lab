# Portable assets for diagnostic replay

The original native-policy studies retain their frozen protocols, action traces,
official outcomes and historical asset identity: 955 files, 188,829,695 bytes,
SHA-256 `f04f72afb7503afb071f9de7c6734b0eca147f87549df0a9080cc987afb17781`.
That identity describes the complete installed directory, including Hugging Face
download metadata and interrupted downloads. It is not a portable model-content
manifest. The original directory was not archived before its server became
unavailable; downloading the same pinned repository cannot reconstruct its
download timestamps or partial files.

The underlying asset source is still the fixed official dataset revision
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
Reconstructed diagnostic replays use a separately frozen content receipt. It
lists the actual scene, mesh and texture files required by the selected tasks,
their sizes, computed SHA-256 values and official LFS SHA-256 or Git blob SHA-1
identities. Every physical file in the reconstructed asset directory must be
covered, and every receipt entry must exist and match. Download caches and
incomplete downloads are excluded from this content identity.

This changes only the replay's asset-directory binding. It does not claim to
recover the historical 955-file directory. Acceptance still requires the pinned
software and checkpoint files, all task-source hashes, fixed state and seed,
first processed robot state and both camera hashes, every recorded action, and
every official reward, termination, truncation and success outcome. A mismatch
stops capture. Missing model dependencies fail during environment construction.
The new content receipt and implementation are frozen before capture.

These are post-hoc action replays without policy inference or additional
benchmark episodes. They do not revise the original success rates, establish
Sentinel intervention efficacy or demonstrate physical robot performance.
The historical formal runners continue to require their original directory
identity; the portable replay procedure does not silently change those protocols.
