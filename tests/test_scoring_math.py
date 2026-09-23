"""Math-only tests for decisions/scoring — run anywhere, no GPU needed."""

import math

import pytest
import torch

from hybrid.decisions import DecisionResult, normalized_entropy
from hybrid.scoring.premask import PREMASK_VALID_THRESHOLD, is_valid, premask_mass


def test_normalized_entropy_bounds():
    assert normalized_entropy({"a": 1.0}) == 0.0
    assert normalized_entropy({"a": 1.0, "b": 0.0}) == 0.0
    # uniform over n candidates ⇒ 1.0
    for n in (2, 5, 10):
        probs = {f"c{i}": 1 / n for i in range(n)}
        assert abs(normalized_entropy(probs) - 1.0) < 1e-9


def test_margin_from_probabilities():
    r = DecisionResult(selected="a", confidence=0.6, probabilities={"a": 0.6, "b": 0.3, "c": 0.1})
    assert r.margin == pytest.approx(0.3)


def test_premask_mass_exact():
    # logits with known softmax: [0, 0, 0] → 1/3 each; candidates 0 and 2 → 2/3
    logits = torch.zeros(3)
    assert premask_mass(logits, [0, 2]) == pytest.approx(2 / 3)


def test_premask_mass_large_vocab():
    logits = torch.zeros(248_320)
    ids = [0, 1, 2]
    assert premask_mass(logits, ids) == pytest.approx(3 / 248_320)
    assert not is_valid(3 / 248_320)  # below the validity threshold
    assert is_valid(0.5)


def test_d2_alpha_normalization_math():
    # candidate of 2 tokens, raw -4.0: alpha=1 → -2.0; alpha=0 → -4.0
    raw, n = -4.0, 2
    assert raw / n**1.0 == pytest.approx(-2.0)
    assert raw / n**0.0 == pytest.approx(-4.0)


def test_softmax_shift_invariance():
    scores = torch.tensor([-1.0, -2.0, -3.0])
    p1 = torch.softmax(scores, dim=0)
    p2 = torch.softmax(scores + 100.0, dim=0)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_premask_threshold_value():
    assert 0.0 < PREMASK_VALID_THRESHOLD < 1.0
