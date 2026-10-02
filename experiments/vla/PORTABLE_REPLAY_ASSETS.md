# Portable assets for post-hoc LIBERO replay

The historical frozen asset receipt covers a mixed site-packages tree created
by an interrupted Hub snapshot, targeted follow-up downloads, and local Hub
metadata. Its retained aggregate hash does not identify the original path list,
so a newly downloaded complete repository cannot honestly claim to reproduce
that whole tree.

Replay therefore accepts a separately frozen
`sentinel-libero-portable-assets-v1` manifest for revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` of the
`lerobot/libero-assets` dataset. The manifest itself is bound by a required
SHA-256 argument. Each relative file entry records bytes, computed SHA-256, and
an official fixed-revision digest (`sha256` or `git-blob-sha1`). Paths are
normalized and may not traverse the root or use symbolic links.

The frozen [`libero_portable_assets.json`](libero_portable_assets.json) replay
manifest has SHA-256
`e743224025f6966a2aa7c9cda51dc0c9aa3b800d68d576c12ec8e88d4f0877af`.
It declares 239 files totaling 189,311,115 bytes; their canonical reconstructed
content SHA-256 is
`c7fef2c780dbab2bbdaf375a2a8bbac171341718988f01cd5309bbb4d692e015`.

At capture time every declared file must exist and match all digests. Every
regular file under the configured LIBERO asset root must be declared, except
Hub cache paths under `.cache` and incomplete download files ending in
`.incomplete`. The canonical content identity is SHA-256 over sorted UTF-8
records `relative_path NUL sha256 NUL bytes newline`.

Capture manifests retain both identities: the original frozen aggregate receipt
as historical provenance and the reconstructed canonical content used for the
replay. This gate is limited to excluded, exact-action post-hoc replay. It does
not authorize a new benchmark run or assert that the reconstructed directory is
the original mixed whole tree.
