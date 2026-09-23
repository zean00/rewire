"""Math-only tests for the runtime controller's accept/escalate logic."""

import pytest

from hybrid.decisions import DecisionResult
from hybrid.runtime import RuntimeConfig, should_accept


def res(conf, margin=0.5, premask=0.5):
    return DecisionResult(selected="a", confidence=conf,
                          probabilities={"a": conf, "b": 1 - conf},
                          margin=margin, premask_mass=premask)


CFG = RuntimeConfig(confidence_accept=0.90, confidence_low=0.65,
                    margin_min=0.10, premask_gate=0.05)


def test_high_confidence_accepted():
    assert should_accept(res(0.95), CFG)


def test_low_confidence_escalates():
    assert not should_accept(res(0.5), CFG)


def test_small_margin_escalates_even_at_high_confidence():
    assert not should_accept(res(0.95, margin=0.05), CFG)


def test_premask_gate_overrides_confidence():
    assert not should_accept(res(0.99, premask=0.01), CFG)


def test_from_yaml_reads_placeholders():
    cfg = RuntimeConfig.from_yaml("configs/thresholds.yaml")
    assert 0 < cfg.confidence_low < cfg.confidence_accept <= 1.0
    assert cfg.premask_gate > 0
