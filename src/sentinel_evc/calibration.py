"""Independent root-level conformal calibration for numeric predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import sha256_hex
from .prediction import DEFAULT_CANDIDATE_RULE_HASH, ResidualModel, predict_trajectory


@dataclass(frozen=True)
class CalibrationRef:
    alpha: float
    q: float | None
    rank: int
    root_count: int
    model_hash: str
    candidate_rule_hash: str
    score_definition: str
    root_manifest_hash: str
    scale_digest: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be finite and in (0,1)")
        if self.q is not None and (not math.isfinite(self.q) or self.q < 0):
            raise ValueError("q must be a finite non-negative value or None")
        if self.rank < 1 or self.root_count < 1:
            raise ValueError("calibration rank/root_count must be positive")
        if not all(isinstance(value, str) and value for value in (
            self.model_hash,
            self.candidate_rule_hash,
            self.score_definition,
            self.root_manifest_hash,
            self.scale_digest,
        )):
            raise ValueError("calibration bindings must be non-empty strings")

    @property
    def hash(self) -> str:
        return sha256_hex(self.summary(include_hash=False))

    @property
    def calibration_id(self) -> str:
        return self.hash

    @property
    def finite(self) -> bool:
        return self.q is not None and math.isfinite(self.q)

    def summary(self, include_hash: bool = True) -> dict:
        result = {
            "alpha": self.alpha,
            "q": self.q,
            "rank": self.rank,
            "root_count": self.root_count,
            "model_hash": self.model_hash,
            "candidate_rule_hash": self.candidate_rule_hash,
            "score_definition": self.score_definition,
            "root_manifest_hash": self.root_manifest_hash,
            "scale_digest": self.scale_digest,
            "finite": self.finite,
        }
        if include_hash:
            result["calibration_hash"] = self.hash
        return result


def fit_root_max_calibration(
    examples,
    model: ResidualModel,
    alpha: float = 0.1,
    candidate_rule_hash: str = DEFAULT_CANDIDATE_RULE_HASH,
) -> CalibrationRef:
    """Use one max normalized error per independent calibration root."""

    roots = tuple(examples)
    if not roots or any(root.split != "cal" for root in roots):
        raise ValueError("calibration accepts independent cal roots only")
    from .data import validate_root_splits

    validate_root_splits(roots)
    overlap = set(model.train_root_ids).intersection(root.root_id for root in roots)
    if overlap:
        raise ValueError("calibration roots overlap model training roots: " + sorted(overlap)[0])
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    scores = []
    for root in roots:
        root_score = 0.0
        for plan, outcome in zip(root.candidates, root.outcomes):
            centers, scales = predict_trajectory(root.history, plan, model)
            if len(centers) != len(outcome.r):
                raise ValueError("prediction/outcome horizon mismatch")
            for truth, center, scale in zip(outcome.r, centers, scales):
                if not all(math.isfinite(value) for value in (truth, center, scale)) or scale <= 0:
                    raise ValueError("non-finite calibration score")
                root_score = max(root_score, abs(truth - center) / scale)
        scores.append(root_score)
    scores.sort()
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    q = scores[rank - 1] if rank <= n else None
    manifest_hash = sha256_hex(
        [{"root_id": root.root_id, "hash": root.hash} for root in sorted(roots, key=lambda item: item.root_id)]
    )
    return CalibrationRef(
        alpha=alpha,
        q=q,
        rank=rank,
        root_count=n,
        model_hash=model.hash,
        candidate_rule_hash=candidate_rule_hash,
        score_definition="root-max-over-candidate-time-absolute-r-error-over-sigma-v1",
        root_manifest_hash=manifest_hash,
        scale_digest=sha256_hex({"candidate_sigmas": model.candidate_sigmas}),
    )
