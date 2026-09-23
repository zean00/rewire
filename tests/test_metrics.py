"""Math-only tests: ECE + choice extraction + shuffling."""

import json

import pytest

from eval.cells import extract_choice_idx, match_gold_text, shuffled_choices
from hybrid.calibration.metrics import expected_calibration_error


def test_ece_perfect_calibration():
    # within each bin, average confidence equals empirical accuracy
    conf = [0.05] * 20 + [0.95] * 20
    corr = [True] + [False] * 19 + [True] * 19 + [False]  # 0.05 and 0.95 accurate
    r = expected_calibration_error(conf, corr)
    assert r["ece"] < 1e-9


def test_ece_worst_case():
    # always confident but always wrong
    conf = [0.99] * 10
    corr = [False] * 10
    r = expected_calibration_error(conf, corr)
    assert r["ece"] == pytest.approx(0.99)


def test_ece_bin_edge_and_bounds():
    r = expected_calibration_error([1.0, 0.0], [True, False])  # right-edge bin
    assert r["ece"] == pytest.approx(0.0)
    assert r["bins"][-1]["lo"] == pytest.approx(0.9)


def test_extract_choice_text_and_index():
    choices = ["Sydney", "Canberra", "Melbourne", "Perth"]
    assert extract_choice_idx("The capital of Australia is Canberra.", choices) == 1
    assert extract_choice_idx("2. Canberra", choices) == 1
    assert extract_choice_idx("I think it's number 3", choices) == 2
    assert extract_choice_idx("No idea, sorry.", choices) is None


def test_extract_choice_first_mention_wins():
    choices = ["Sydney", "Canberra"]
    # "Sydney" is mentioned first; the convention returns the first mention
    assert extract_choice_idx("Not Sydney, but Canberra.", choices) == 0


def test_match_gold_text():
    assert match_gold_text("It is the Great Wall of China.", "Great Wall")
    assert not match_gold_text("Nope.", "Great Wall")


def test_shuffled_choices_is_deterministic_and_complete():
    item = json.loads('{"id": "x1", "choices": ["a", "b", "c", "d"], "answer_idx": 2}')
    s1 = shuffled_choices(item, 99)
    s2 = shuffled_choices(item, 99)
    assert s1 == s2
    assert sorted(s1) == ["a", "b", "c", "d"]
