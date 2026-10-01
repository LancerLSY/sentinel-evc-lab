"""Truthful registry for experiments that still require external prerequisites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    title: str
    status: str
    prerequisites: tuple[str, ...]
    command: str
    acceptance: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    metrics: tuple = ()

    def __post_init__(self) -> None:
        if any(not isinstance(value, tuple) for value in (
            self.prerequisites,
            self.acceptance,
            self.expected_artifacts,
            self.metrics,
        )):
            raise ValueError("experiment collections must be immutable tuples")

    def summary(self) -> dict:
        return {
            "id": self.experiment_id,
            "title": self.title,
            "status": self.status,
            "prerequisites": list(self.prerequisites),
            "command": self.command,
            "acceptance": list(self.acceptance),
            "expected_artifacts": list(self.expected_artifacts),
            "metrics": list(self.metrics),
        }


def experiment_registry() -> tuple[ExperimentRecord, ...]:
    return (
        ExperimentRecord(
            "real-vla-shadow",
            "Real VLA record-only shadow integration",
            "pending",
            ("frozen upstream commit", "actual checkpoint", "reviewed action descriptor"),
            "python -m sentinel_evc experiments --experiment-id real-vla-shadow",
            ("ten fixed-seed transparent replays", "raw and final action digests recorded"),
            ("manifest.json", "raw_actions.jsonl", "final_actions.jsonl"),
        ),
        ExperimentRecord(
            "gru-residual-world",
            "v4 GRU ResidualWorld training",
            "pending",
            ("optional ML environment", "frozen root dataset", "independent calibration roots"),
            "python -m sentinel_evc experiments --experiment-id gru-residual-world",
            ("held-out root evaluation", "no hidden simulator fields in model input"),
            ("model.json", "calibration.json", "evaluation.json"),
        ),
        ExperimentRecord(
            "physical-robot-controller",
            "Robot controller cancellation and feedback trial",
            "pending",
            ("device access", "driver capability table", "approved test cell"),
            "python -m sentinel_evc experiments --experiment-id physical-robot-controller",
            ("measured cancel semantics", "feedback and stop bounds recorded"),
            ("controller_manifest.json", "trial_events.jsonl", "measurements.json"),
        ),
        ExperimentRecord(
            "end-to-end-cost-benchmark",
            "End-to-end validation cost benchmark",
            "pending",
            ("frozen hardware", "frozen scenarios", "warmup and repetition protocol"),
            "python -m sentinel_evc experiments --experiment-id end-to-end-cost-benchmark",
            ("parent, transform, inherit, fallback and signing costs included", "P50/P95 reported"),
            ("benchmark_manifest.json", "samples.jsonl", "summary.json"),
        ),
    )
