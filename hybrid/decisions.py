"""Decision result structures (doc §22, rev. 3: premask_mass added)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def normalized_entropy(probs: dict[str, float]) -> float:
    """Entropy in [0, 1], normalized by log2(n_candidates)."""
    n = len(probs)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs.values() if p > 0.0)
    return h / math.log2(n)


def with_temperature(res: "DecisionResult", T: float) -> "DecisionResult":
    """Re-derive probabilities/confidence/margin/entropy from raw scores at
    temperature T. Monotone ⇒ selection unchanged; the *threshold mapping*
    changes, which is the point: gates must run on the calibrated
    distribution (Phase 3 finding: raw gating escalated 90% vs calibrated
    ~28%). Mutates and returns res."""
    if T == 1.0 or not res.raw_scores:
        return res
    m = max(res.raw_scores.values())
    exps = {k: math.exp((v - m) / T) for k, v in res.raw_scores.items()}
    z = sum(exps.values())
    res.probabilities = {k: v / z for k, v in exps.items()}
    res.confidence = max(res.probabilities.values())
    ranked = sorted(res.probabilities.values(), reverse=True)
    res.margin = ranked[0] - ranked[1] if len(ranked) >= 2 else 1.0
    res.entropy = normalized_entropy(res.probabilities)
    return res


@dataclass
class DecisionResult:
    selected: str
    confidence: float

    probabilities: dict[str, float]
    # Candidate -> score before the final softmax (log-likelihood or logit).
    raw_scores: dict[str, float] = field(default_factory=dict)

    margin: float = 0.0
    entropy: float = 0.0

    scoring_method: str = ""
    # e.g. "d2:alpha=1.0:pmi_lambda=0.0" — normalization/correction switches are
    # part of the method identity and must not be averaged away in reports.
    normalization: str = ""

    # Fraction of raw (unmasked) probability mass captured by the candidate
    # set at the read position. For multi-token candidates this is the sum of
    # first-token masses (documented approximation — see premask.py).
    premask_mass: float = 1.0

    prefill_ms: float = 0.0
    scoring_ms: float = 0.0
    total_ms: float = 0.0

    escalated: bool = False

    def __post_init__(self) -> None:
        if not self.entropy:
            self.entropy = normalized_entropy(self.probabilities)
        if self.probabilities and not self.margin:
            ranked = sorted(self.probabilities.values(), reverse=True)
            if len(ranked) >= 2:
                self.margin = ranked[0] - ranked[1]
