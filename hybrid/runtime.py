"""Hybrid runtime controller (doc §7, Phase 3 deliverable).

Deterministic software controller implementing the escalation loop:

    DECIDE -> accept if calibrated confidence >= thresholds
           -> else THINK(budget) -> DECIDE again -> accept best available

Thresholds come from configs/thresholds.yaml (fit in Phase 3, not assumed).
This is the minimal "adaptive inference controller" the PoC's headline
comparison (S8/Q12) runs through: hybrid_runtime vs A (never-think) vs
G (think-always).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from .decide import decide
from .decisions import DecisionResult
from .engine import LoadedModel
from .template import pin_after_think, question_messages


@dataclass
class RuntimeConfig:
    confidence_accept: float = 0.90
    confidence_low: float = 0.65
    margin_min: float = 0.10
    think_budget: int = 256
    generate_budget: int = 768
    method: str = "d2"
    alpha: float = 1.0
    pmi_lambda: float = 0.0
    premask_gate: float = 0.05
    temperature: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        raw = yaml.safe_load(open(path))
        t = raw.get("thresholds", raw)
        m = raw.get("model", {})
        g = m.get("generation", {})
        return cls(
            confidence_accept=t.get("confidence_accept", 0.90),
            confidence_low=t.get("confidence_low", 0.65),
            margin_min=t.get("margin_min", 0.10),
            think_budget=g.get("max_new_tokens_think", 256),
            generate_budget=g.get("max_new_tokens_think", 768),
            method=raw.get("scoring", {}).get("method", "d2"),
            alpha=raw.get("scoring", {}).get("d2_alpha", 1.0),
            premask_gate=raw.get("scoring", {}).get("premask_valid_threshold", 0.05),
            temperature=raw.get("scoring", {}).get("temperature", 1.0),
        )


@dataclass
class TurnResult:
    final: DecisionResult
    escalated: bool
    think_ran: bool
    premask_valid: bool
    n_decisions: int
    events: list = field(default_factory=list)


def should_accept(res: DecisionResult, cfg: RuntimeConfig) -> bool:
    if res.premask_mass < cfg.premask_gate:
        return False  # invalid measurement — escalate rather than trust
    if res.margin < cfg.margin_min:
        return False
    return res.confidence >= cfg.confidence_accept


def hybrid_decide_turn(
    loaded: LoadedModel, question: str, candidates: list[str], cfg: RuntimeConfig,
    escalation: str = "redecide",
) -> TurnResult:
    """One DECIDE→(escalate) turn.

    escalation="redecide": THINK then re-score candidates at the post-think
    read point (doc Experiment E). Phase 2's filler control found NO net
    benefit from this mechanism (filler_control.md) — kept for the
    preregistered comparison.
    escalation="generate": THINK-generation produces the answer (options
    shown, native thinking). This is confidence-gated mode selection — the
    pivot mechanism after kill criterion 2 fired; supported by the regime
    table (think-always wins only on computation-shaped domains).
    """
    events: list[dict] = []

    from .decisions import with_temperature

    # anchor=True: the format anchor lifts pre-mask mass to ~0.9 (Phase 1
    # finding) — without it the premask validity gate fires on most ARC items
    # and the runtime escalates everything. Production DECIDE config.
    first = decide(loaded, question, candidates, method=cfg.method, alpha=cfg.alpha,
                   pmi_lambda=cfg.pmi_lambda, anchor=True)
    first = with_temperature(first, cfg.temperature)
    events.append({"type": "decide", "confidence": first.confidence,
                   "premask_mass": first.premask_mass, "selected": first.selected})

    if should_accept(first, cfg):
        return TurnResult(final=first, escalated=False, think_ran=False,
                          premask_valid=first.premask_mass >= cfg.premask_gate,
                          n_decisions=1, events=events)

    if escalation == "redecide":
        msgs = question_messages(question, candidates)
        post = pin_after_think(loaded.tokenizer, loaded.model, msgs,
                               max_think_tokens=cfg.think_budget)
        second = decide(loaded, question, candidates, method=cfg.method, alpha=cfg.alpha,
                        pmi_lambda=cfg.pmi_lambda, read_mode="postthink",
                        max_think_tokens=cfg.think_budget, anchor=True)
        second = with_temperature(second, cfg.temperature)
        events.append({"type": "think", "budget": cfg.think_budget})
        events.append({"type": "decide", "confidence": second.confidence,
                       "premask_mass": second.premask_mass, "selected": second.selected})
        final = second if second.confidence >= first.confidence else first
        return TurnResult(final=final, escalated=True, think_ran=True,
                          premask_valid=second.premask_mass >= cfg.premask_gate,
                          n_decisions=2, events=events)

    if escalation == "generate":
        from .decisions import DecisionResult
        from .engine import generate_answer
        from .extract import extract_choice_idx

        msgs = question_messages(question, candidates)
        answer, meta = generate_answer(loaded, msgs, max_new_tokens=cfg.generate_budget,
                                       enable_thinking=True)
        idx = extract_choice_idx(answer, candidates)
        selected = candidates[idx] if idx is not None else first.selected
        final = DecisionResult(
            selected=selected,
            confidence=first.confidence,  # gate confidence; generation carries no distribution
            probabilities=first.probabilities,
            scoring_method="decide_escalate_generate",
            premask_mass=first.premask_mass,
        )
        events.append({"type": "think_generate", "gen_tokens": meta.get("think_tokens_approx", 0),
                       "parsed": idx is not None, "selected": selected})
        return TurnResult(final=final, escalated=True, think_ran=True,
                          premask_valid=first.premask_mass >= cfg.premask_gate,
                          n_decisions=1, events=events)

    raise ValueError(f"unknown escalation mode: {escalation}")
