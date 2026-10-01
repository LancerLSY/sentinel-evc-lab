# Local workbench assets

No CDN or build step. `sentinel-evc serve` serves these packaged assets on loopback.
All displayed measurements come from the HTTP run record, persisted replay and event stream.
Scenario JSON is sent verbatim so duplicate keys are rejected by the server.
Numeric time and host wall time have separate labels. The bundled public key is a demo trust source.

The workbench has five product surfaces: numeric execution, recorded MuJoCo
physics, bounded model import/preview, read-only robot diagnostics, and the
experiment roadmap. `viewer.js` is an offline Canvas renderer. It draws only
server-returned geometry and recorded positions; an empty result remains an
explicit empty state.
