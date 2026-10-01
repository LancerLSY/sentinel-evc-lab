"""Root-grouped numeric data generation and split validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import Plan, sha256_hex
from .numeric_world import NumericHistory, NumericOutcome, generate_candidates, make_numeric_case, rollout_candidate


VALID_SPLITS = ("train", "dev", "cal", "test")


@dataclass(frozen=True)
class NumericRoot:
    root_id: str
    split: str
    seed: int
    history: NumericHistory
    candidates: tuple[Plan, ...]
    outcomes: tuple[NumericOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or not isinstance(self.outcomes, tuple):
            raise ValueError("root candidates/outcomes must be immutable tuples")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if len(self.candidates) != 4 or len(self.outcomes) != 4:
            raise ValueError("each root must retain all four candidate branches")
        if any(outcome.root_id != self.root_id for outcome in self.outcomes):
            raise ValueError("outcome root mismatch")
        if tuple(plan.hash for plan in self.candidates) != tuple(
            outcome.plan_hash for outcome in self.outcomes
        ):
            raise ValueError("candidate/outcome binding mismatch")

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary())

    def summary(self) -> dict:
        return {
            "root_id": self.root_id,
            "split": self.split,
            "seed": self.seed,
            "history_hash": self.history.hash,
            "candidate_hashes": [plan.hash for plan in self.candidates],
            "outcome_hashes": [
                sha256_hex({"plan_hash": out.plan_hash, "r": out.r, "v": out.v})
                for out in self.outcomes
            ],
        }


@dataclass(frozen=True)
class NumericDataset:
    roots: tuple[NumericRoot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.roots, tuple):
            raise ValueError("dataset roots must be an immutable tuple")
        validate_root_splits(self.roots)

    def split(self, name: str) -> tuple[NumericRoot, ...]:
        if name not in VALID_SPLITS:
            raise ValueError(f"unknown split: {name}")
        return tuple(root for root in self.roots if root.split == name)

    @property
    def hash(self) -> str:
        return sha256_hex([root.summary() for root in self.roots])

    def summary(self) -> dict:
        return {
            "dataset_hash": self.hash,
            "counts": {name: len(self.split(name)) for name in VALID_SPLITS},
            "root_ids": [root.root_id for root in self.roots],
        }


def validate_root_splits(examples: Iterable[NumericRoot]) -> None:
    """Reject sibling/root leakage across train/dev/cal/test."""

    seen = {}
    for example in examples:
        prior = seen.get(example.root_id)
        if prior is not None:
            if prior != example.split:
                raise ValueError(
                    f"root {example.root_id} appears in both {prior} and {example.split}"
                )
            raise ValueError(f"duplicate root {example.root_id} in {example.split}")
        seen[example.root_id] = example.split


def _make_root(seed: int, split: str) -> NumericRoot:
    case = make_numeric_case(seed)
    candidates = generate_candidates()
    outcomes = tuple(rollout_candidate(case, plan) for plan in candidates)
    return NumericRoot(case.root_id, split, seed, case.history, candidates, outcomes)


def make_numeric_dataset(
    train_roots: int = 32,
    dev_roots: int = 8,
    cal_roots: int = 19,
    test_roots: int = 12,
    seed: int = 0,
) -> NumericDataset:
    """Generate disjoint deterministic root groups; siblings never cross a split."""

    counts = (train_roots, dev_roots, cal_roots, test_roots)
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("split counts must be non-negative integers")
    roots = []
    next_seed = seed
    for split, count in zip(VALID_SPLITS, counts):
        for _ in range(count):
            roots.append(_make_root(next_seed, split))
            next_seed += 1
    return NumericDataset(tuple(roots))
