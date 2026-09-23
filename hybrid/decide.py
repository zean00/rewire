"""DECIDE API (doc §21) — thin wrapper over pinned template + scorers.

Phase 1 deliverable. The wrapper exists so callers cannot bypass the pinned
read point: every decision goes through here.
"""

from __future__ import annotations

import os

from .decisions import DecisionResult
from .engine import LoadedModel
from .scoring import score_first_token, score_sequence_logprob, score_sequence_logprob_batched
from .template import pin_after_think, pin_prompt, question_messages


class DecideError(ValueError):
    pass


def decide(
    loaded: LoadedModel,
    question: str,
    candidates: list[str],
    method: str = "d2",
    alpha: float = 1.0,
    pmi_lambda: float = 0.0,
    leading_space: bool = False,
    read_mode: str = "pinned_nothink",
    max_think_tokens: int = 256,
    anchor: bool = False,
    batched: bool | None = None,
    cache_mode: str = "auto",
) -> DecisionResult:
    """Score a bounded candidate set. method: "d1" | "d2".

    read_mode: "pinned_nothink" (default, all measurements) or "postthink"
    (sensitivity checks only — see template.py). anchor=True adds the
    format-anchor line (Phase 1 experiment).

    batched: D2-only. True ⇒ shared-prefix scheduling (one prompt prefill +
    per-candidate continuation scoring; same math, ~3-5x faster). None ⇒
    HYBRID_BATCHED_D2 env toggle, so runtime re-runs can flip the path
    without call-site edits. Records self-describe via
    DecisionResult.scoring_method ("d2_sequence_logprob_batched(mode)").
    eval/batch_check.py is the parity gate — parity before latency claims.
    """
    if len(set(candidates)) != len(candidates):
        raise DecideError("duplicate candidates — DECIDE requires a distinct set (§16.2 duplicate-semantic control handles this explicitly)")
    if len(candidates) < 2:
        raise DecideError("need at least 2 candidates")

    if batched is None:
        batched = os.environ.get("HYBRID_BATCHED_D2", "") == "1"

    messages = question_messages(question, candidates, anchor=anchor)
    if read_mode == "pinned_nothink":
        pinned = pin_prompt(loaded.tokenizer, messages)
    elif read_mode == "postthink":
        pinned = pin_after_think(loaded.tokenizer, loaded.model, messages, max_think_tokens=max_think_tokens)
    else:
        raise DecideError(f"unknown read_mode: {read_mode}")

    if method == "d1":
        res, _ = score_first_token(loaded.model, loaded.tokenizer, pinned, candidates, leading_space=leading_space)
        return res
    if method == "d2":
        if batched:
            if read_mode != "pinned_nothink":
                raise DecideError("batched D2 supports read_mode='pinned_nothink' only")
            return score_sequence_logprob_batched(
                loaded.model, loaded.tokenizer, pinned, candidates,
                alpha=alpha, pmi_lambda=pmi_lambda, leading_space=leading_space,
                cache_mode=cache_mode,
            )
        return score_sequence_logprob(
            loaded.model, loaded.tokenizer, pinned, candidates,
            alpha=alpha, pmi_lambda=pmi_lambda, leading_space=leading_space,
        )
    raise DecideError(f"unknown method: {method}")
