# Mechanism research notes

The engineering chain is final action → constrained proof → bounded one-use permit → current-state commit → controller feedback → revocation barrier → signed replay. Consequence predictions impose an additional gate after fixed physical validation. The product scope and remaining experiments are mapped in `implementation_matrix.md`.

Relevant primary references and their implementation implications:

- [CPython HTTP server](https://docs.python.org/3/library/http.server.html): simple local transport; this basic product binds loopback and validates Host/Origin/token, rather than treating a standard-library server as internet deployment infrastructure.
- [Python monotonic clock](https://docs.python.org/3/library/time.html#time.monotonic_ns): permits expire on the local host monotonic clock; UTC is an audit label, and persisted monotonic deadlines cannot authorize after process restart.
- [Conformal prediction tutorial](https://arxiv.org/abs/2107.07511): the finite-sample rank is `ceil((n+1)*(1-alpha))`; rank above n means an unbounded envelope. The sample unit here is a root maximum over all prescribed candidates/times; it is not four independent samples per root or a guarantee of conditional accident risk.
- [NIST EdDSA standard](https://csrc.nist.gov/pubs/fips/186-5/final) and [cryptography Ed25519 API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/): signatures establish integrity relative to a selected key. The included demo public key is not an external trust anchor or evidence that sensors are honest.
- [ROS 2 actions](https://design.ros2.org/articles/actions.html): cancellation is a lifecycle protocol, so request/acceptance/confirmation and outstanding accepted work need distinct records. The simulated model is not a ROS controller adapter.
- [OpenVLA upstream](https://github.com/openvla/openvla) and [OpenPI upstream](https://github.com/Physical-Intelligence/openpi): actual checkpoint, action normalization and upstream versions are prerequisite integrations. Scripted numeric candidates do not establish a VLA result.

The first release favors strict observable bindings and reproducible tests. Every future action adapter, repair, candidate-generation or model change must create/recheck the final digest and calibration scope before obtaining a new permit.
