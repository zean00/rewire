"""D1 — restricted first-token logits (doc §10.2).

One forward pass; softmax over the gathered candidate-token logits (not the
full 248K vocab). Pre-mask mass recorded per doc §9.1.
"""

from __future__ import annotations

import time

import torch

from ..decisions import DecisionResult
from ..template import PinnedPrompt, question_messages
from . import premask


def candidate_first_tokens(tokenizer, candidates: list[str], leading_space: bool = False) -> dict[str, list[int]]:
    """Candidate label -> list of single candidate token ids.

    One token per candidate required; candidates whose text does not tokenize
    to a single id are reported to the caller (D1's documented limitation).
    Tokenization variant (leading space) is fixed config, not per-call choice —
    mixing variants would corrupt calibration.
    """
    out: dict[str, list[int]] = {}
    for c in candidates:
        text = (" " + c) if leading_space else c
        ids = tokenizer.encode(text, add_special_tokens=False)
        out[c] = ids
    return out


@torch.no_grad()
def score_first_token(
    model,
    tokenizer,
    pinned: PinnedPrompt,
    candidates: list[str],
    leading_space: bool = False,
) -> tuple[DecisionResult, dict[str, list[int]]]:
    """Returns (result, tokenization_map). Candidates that do not map to a
    single token get their first token — recorded in the map for the report."""
    tok_map = candidate_first_tokens(tokenizer, candidates, leading_space=leading_space)
    first_ids = [ids[0] for ids in tok_map.values()]

    t0 = time.perf_counter()
    out = model(input_ids=pinned.input_ids.to(model.device))
    logits_row = out.logits[0, -1, :]
    forward_ms = (time.perf_counter() - t0) * 1000.0

    cand_ids = torch.tensor(first_ids, device=logits_row.device, dtype=torch.long)
    masked = logits_row[cand_ids]
    probs = torch.softmax(masked.float(), dim=-1)

    mass = premask.premask_mass(logits_row, first_ids)

    probs_d = {c: float(p) for c, p in zip(candidates, probs.tolist())}
    selected = max(probs_d, key=probs_d.get)
    res = DecisionResult(
        selected=selected,
        confidence=probs_d[selected],
        probabilities=probs_d,
        raw_scores={c: float(s) for c, s in zip(candidates, masked.float().tolist())},
        scoring_method="d1_restricted_logits",
        normalization="none",
        premask_mass=mass,
        scoring_ms=forward_ms,
        total_ms=forward_ms,
    )
    return res, tok_map
