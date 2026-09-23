"""D2 — full candidate sequence log-likelihood (doc §10.2 rev. 3).

score(c) = sum_i log P(t_i | context, t_<i)

Normalization (method identity, recorded in DecisionResult.normalization):
  alpha=1.0  mean log-prob
  alpha=0.0  raw sum
  alpha=0.5  geometric middle ground (empirical)

Correction (optional, from Phase 1 per rev. 3):
  PMI/DC (Holtzman et al. 2021): subtract lambda * logP(c | content-free ctx)
"""

from __future__ import annotations

import copy
import os
import time

import torch

from ..decisions import DecisionResult
from ..template import PinnedPrompt, question_messages
from .premask import premask_mass

CONTENT_FREE_INPUT = "N/A"

# Set once per process by the batched scorer's reuse probe: "shared" (prefix
# cache rewinds correctly after each suffix forward), "copy" (deepcopy the
# prefix cache per candidate), or "batched" (tiled batched suffix forwards).
_CACHE_REUSE_MODE: str | None = None


@torch.no_grad()
def _candidate_logprob(model, prompt_ids: torch.Tensor, cand_ids: torch.Tensor) -> float:
    """Teacher-forced log-prob of cand_ids continuing prompt_ids."""
    # Device-normalize both sides: pinned prompts may live on CPU (default
    # read point) or CUDA (post-think prompts).
    ids = torch.cat(
        [prompt_ids.to(model.device), cand_ids.to(model.device)], dim=1
    )
    logits = model(input_ids=ids).logits.float()
    # position of first candidate token predicts from last prompt position
    start = prompt_ids.shape[1] - 1
    relevant = logits[0, start : start + cand_ids.shape[1], :]
    logprobs = torch.log_softmax(relevant, dim=-1)
    tgt = cand_ids.to(model.device).squeeze(0)
    return float(logprobs.gather(-1, tgt.unsqueeze(-1)).sum().item())


def _encode_candidates(tokenizer, candidates: list[str], leading_space: bool) -> dict[str, torch.Tensor]:
    out = {}
    for c in candidates:
        text = (" " + c) if leading_space else c
        ids = tokenizer.encode(text, add_special_tokens=False)
        out[c] = torch.tensor([ids], dtype=torch.long)
    return out


@torch.no_grad()
def score_sequence_logprob(
    model,
    tokenizer,
    pinned: PinnedPrompt,
    candidates: list[str],
    alpha: float = 1.0,
    pmi_lambda: float = 0.0,
    leading_space: bool = False,
) -> DecisionResult:
    cand_ids = _encode_candidates(tokenizer, candidates, leading_space)

    t0 = time.perf_counter()
    raw: dict[str, float] = {}
    first_tokens: list[int] = []
    for c in candidates:
        raw[c] = _candidate_logprob(model, pinned.input_ids, cand_ids[c])
        first_tokens.append(int(cand_ids[c][0, 0].item()))

    if pmi_lambda > 0.0:
        cf_messages = question_messages(CONTENT_FREE_INPUT, candidates)
        from ..template import pin_prompt

        cf_pinned = pin_prompt(tokenizer, cf_messages)
        for c in candidates:
            raw[c] -= pmi_lambda * _candidate_logprob(model, cf_pinned.input_ids, cand_ids[c])

    norm = {c: s / (len(cand_ids[c][0]) ** alpha) for c, s in raw.items()}
    scores = torch.tensor([norm[c] for c in candidates], dtype=torch.float32)
    probs = torch.softmax(scores, dim=-1)
    forward_ms = (time.perf_counter() - t0) * 1000.0

    # pre-mask mass: sum of first-token masses at the real read position (approx for multi-token)
    with torch.no_grad():
        logits_row = model(input_ids=pinned.input_ids.to(model.device)).logits[0, -1, :]
    mass = premask_mass(logits_row, first_tokens)

    probs_d = {c: float(p) for c, p in zip(candidates, probs.tolist())}
    selected = max(probs_d, key=probs_d.get)
    normalization = f"alpha={alpha}:pmi_lambda={pmi_lambda}"
    return DecisionResult(
        selected=selected,
        confidence=probs_d[selected],
        probabilities=probs_d,
        raw_scores=raw,
        scoring_method="d2_sequence_logprob",
        normalization=normalization,
        premask_mass=mass,
        prefill_ms=0.0,
        scoring_ms=forward_ms,
        total_ms=forward_ms,
    )


# --- Batched D2 (L1) ---------------------------------------------------------
#
# The prompt (including the full Options block) is IDENTICAL for every
# candidate; only the scored continuation differs. The sequential path above
# therefore pays one full prefill per candidate. The batched path pays it
# once: a single prefix forward with use_cache=True (which also yields the
# last-position logits the pre-mask mass needs — the sequential path spends
# a whole extra forward on those), then per-candidate continuation scoring
# against the shared cache.
#
# Strategy is decided by evidence, not assumption, because the hybrid
# Gated DeltaNet layers' cache behavior under reuse/rewind/tile is a measured
# quantity (L1 parity gate, eval/batch_check.py):
#   shared  — one cache, suffix forward per candidate, crop back to prefix
#             length after each; pads never precede real tokens, so recurrent
#             state at real positions is only ever fed real tokens.
#   copy    — deepcopy of the pristine prefix cache per candidate (fallback
#             if neither crop-rewind nor tiling validates).
#   batched — tile the cache to B rows, right-pad continuations, one forward
#             per chunk; production default when the probe validates the tile.


def _clear_stale_rope_deltas(model) -> None:
    """Drop rope_deltas left behind by generate() so text-only cached forwards
    use the model's own position inference (the path the parity gate
    validated). With rope_deltas set, qwen3_5's stale-delta branch derives
    positions from the FULL attention-mask length for any cached forward that
    carries a mask — crash or silent corruption on our tiled forwards."""
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "rope_deltas", None) is not None:
        inner.rope_deltas = None


def _cont_position_ids(rd, P: int, S: int, B: int, device) -> torch.Tensor:
    """Continuation positions after an image-conditioned prefill: text arange
    P..P+S-1 plus the stored multimodal rope delta — the same continuation the
    wrapper's incremental-generation hook builds. Passed explicitly because
    with rope_deltas set the model's own derivation uses the full mask length
    instead of the S new tokens."""
    pos = torch.arange(P, P + S, device=device).unsqueeze(0).expand(B, S)
    return pos + rd.reshape(-1, 1).to(device).expand(B, S)


@torch.no_grad()
def _prefix_state(model, prompt_ids: torch.Tensor, prefill_kwargs: dict | None = None):
    """One prefill of the shared prompt.

    Returns (cache, first_token_logprobs, raw_last_row):
      - cache: prefix KV/state for suffix forwards
      - first_token_logprobs: log_softmax of the last-position row — scores
        each candidate's FIRST token (this was the parity-run bug: raw logits
        leaked in as if they were log-probs, +20-nat errors)
      - raw_last_row: the un-softmaxed row, exactly what premask_mass expects

    prefill_kwargs carries multimodal payloads (pixel_values, image_grid_thw)
    for image-conditioned prefills; the suffix forwards stay text-only.
    """
    ids = prompt_ids.to(model.device)
    out = model(input_ids=ids, use_cache=True, **(prefill_kwargs or {}))
    row = out.logits[0, -1, :].float()
    return out.past_key_values, row.log_softmax(dim=-1), row


@torch.no_grad()
def _suffix_logprob_on_cache(model, cache, first_lp: torch.Tensor, cand_ids: torch.Tensor,
                             position_ids: torch.Tensor | None = None) -> float:
    """Logprob of cand_ids continuing the cached prefix (single row, no pads).

    position_ids ([1, S]) is passed only for image-conditioned prefixes (see
    _cont_position_ids); text-only prefixes rely on the model's own position
    inference, which is the path the parity gate validated."""
    ids = cand_ids.to(model.device)
    extra = {} if position_ids is None else {"position_ids": position_ids.to(model.device)}
    out = model(input_ids=ids, past_key_values=cache, use_cache=False, **extra)
    lg = out.logits[0].float()
    s = first_lp[ids[0, 0]]
    if ids.shape[1] > 1:
        lp = torch.log_softmax(lg[: ids.shape[1] - 1], dim=-1)
        s = s + lp.gather(-1, ids[0, 1:].unsqueeze(-1)).sum()
    return float(s)


@torch.no_grad()
def _batched_chunk_scores(model, tiled_cache, first_lp: torch.Tensor, chunk: list,
                          position_ids: torch.Tensor | None = None) -> list[float]:
    """Score a chunk of candidate continuations against a B-row tiled cache.

    Right-padded continuations: pads sit AFTER each row's real tokens, and the
    recurrent (GDN) state is updated strictly causally — a pad consumed after a
    row's tokens cannot corrupt that row's earlier scores. Rows are gathered
    independently, so per-row scoring equals the single-row forward.
    """
    B, S = len(chunk), max(int(c.shape[1]) for c in chunk)
    P = int(tiled_cache.get_seq_length())
    inp = torch.zeros((B, S), dtype=torch.long)
    mask = torch.zeros((B, P + S), dtype=torch.long, device=model.device)
    mask[:, :P] = 1
    for i, c in enumerate(chunk):
        inp[i, : c.shape[1]] = c[0]
        mask[i, P : P + c.shape[1]] = 1
    extra = {} if position_ids is None else {"position_ids": position_ids.to(model.device)}
    out = model(
        input_ids=inp.to(model.device), attention_mask=mask,
        past_key_values=tiled_cache, use_cache=False, **extra,
    )
    lg = out.logits.float()
    totals = []
    for i, c in enumerate(chunk):
        L = int(c.shape[1])
        tgt = c[0].to(model.device)
        s = first_lp[tgt[0]]
        if L > 1:
            lp = torch.log_softmax(lg[i, : L - 1], dim=-1)
            s = s + lp.gather(-1, tgt[1:].unsqueeze(-1)).sum()
        totals.append(float(s))
    return totals


@torch.no_grad()
def _probe_cache_reuse(model, prompt_ids: torch.Tensor, cand_ids: torch.Tensor) -> str:
    """One-time check that the shared-cache path scores the same thing.

    Three independent conditions, in promotion order:
      1. Conditioning: suffix logprob on the pristine cache vs the full
         forward (sequential reference) must agree within the observed
         bf16 shape-noise floor (~0.35 nats; measured, not assumed —
         different kernel schedules, same math).
      2. Rewind: after crop() back to the prefix, re-scoring the SAME
         candidate must reproduce lp1 BIT-EXACTLY (deterministic kernels,
         identical inputs) — any lossy GDN-state restore shows up here
         regardless of the noise floor. Pass -> 'shared'.
      3. Tiling (when rewind is lossy): tile a FRESH pristine prefix to 4
         identical candidate rows; the tile is correct iff all rows score
         identically (deterministic) AND match the untiled suffix score.
         Pass -> 'batched' (production default). Fail -> 'copy'.
    """
    _clear_stale_rope_deltas(model)
    ref = _candidate_logprob(model, prompt_ids, cand_ids)
    cache, first_lp, _raw_row = _prefix_state(model, prompt_ids)
    P = int(prompt_ids.shape[1])
    act = getattr(cache, "activate_past_recording", None)
    if act is not None:
        act()
    elif hasattr(cache, "layers") and hasattr(cache.layers[0], "activate_past_recording"):
        for l in cache.layers:
            l.activate_past_recording()
    lp1 = _suffix_logprob_on_cache(model, cache, first_lp, cand_ids)
    if abs(lp1 - ref) > 0.35:
        return "copy"
    rewind = getattr(cache, "crop", None)
    if rewind is not None:
        try:
            rewind(P)
            lp2 = _suffix_logprob_on_cache(model, cache, first_lp, cand_ids)
            if abs(lp2 - lp1) <= 1e-6:
                return "shared"
        except Exception:
            pass
    # rewind is lossy on this architecture — validate the tiled batch path on
    # a fresh pristine prefix (the cache above was mutated by the rewind test)
    cache2, first_lp2, _ = _prefix_state(model, prompt_ids)
    try:
        tiled = _tile_cache(cache2, 4)
        rows = _batched_chunk_scores(model, tiled, first_lp2, [cand_ids] * 4)
    except Exception:
        return "copy"
    lp_copy = _suffix_logprob_on_cache(
        model, copy.deepcopy(cache2), first_lp2, cand_ids)
    if any(abs(r - rows[0]) > 1e-6 for r in rows[1:]):
        return "copy"
    if abs(rows[0] - lp1) > 0.35 or abs(rows[0] - lp_copy) > 0.35:
        return "copy"
    return "batched"


def _tile_cache(cache, repeats: int):
    """Tile a batch-1 prefix cache to `repeats` rows (batched suffix mode).

    Uses the library's own beam-search reorder path: index_select with an
    all-zeros index is an exact tile, and layer.reorder_cache covers both
    full-attention layers (keys/values) and the linear-attention state DICTS
    (conv_states / recurrent_states) that a hand-rolled tensor walk misses.
    The pristine cache is deepcopied first — reorder rebinds state in place,
    and the pristine must survive for the next chunk (and for copy mode).
    """
    tiled = copy.deepcopy(cache)
    beam_idx = torch.zeros(repeats, dtype=torch.long)
    if hasattr(tiled, "layers"):
        for layer in tiled.layers:
            layer.reorder_cache(beam_idx)
    else:  # legacy flat caches
        tiled.reorder_cache(beam_idx)
    return tiled


@torch.no_grad()
def _score_all_batched(model, prompt_ids, cand_list, mode, batch_size,
                       prefill_kwargs: dict | None = None):
    """Returns (per-candidate total logprobs, raw last-position logits row)."""
    image_prefill = bool(prefill_kwargs) and any(
        k in prefill_kwargs for k in ("pixel_values", "image_grid_thw")
    )
    if image_prefill:
        cache, first_lp, raw_row = _prefix_state(model, prompt_ids, prefill_kwargs)
        # the multimodal prefill computed mrope and stored rope_deltas;
        # continuations must carry explicit positions (text arange + delta),
        # otherwise the stale-delta branch derives full-mask-length positions
        rd = getattr(getattr(model, "model", None), "rope_deltas", None)
    else:
        _clear_stale_rope_deltas(model)
        cache, first_lp, raw_row = _prefix_state(model, prompt_ids)
        rd = None
    P = int(prompt_ids.shape[1])
    totals: list[float] = []
    if mode == "batched":
        for i0 in range(0, len(cand_list), batch_size):
            chunk = cand_list[i0 : i0 + batch_size]
            tiled = _tile_cache(cache, len(chunk))
            pos = None
            if rd is not None:
                pos = _cont_position_ids(
                    rd, P, max(int(c.shape[1]) for c in chunk), len(chunk), model.device)
            totals += _batched_chunk_scores(model, tiled, first_lp, chunk, pos)
        return totals, raw_row
    if mode == "copy":
        # the pristine cache must never see a suffix forward: score every
        # candidate against its own deepcopy of the untouched prefix state
        for c in cand_list:
            pos = _cont_position_ids(rd, P, int(c.shape[1]), 1, model.device) if rd is not None else None
            totals.append(
                _suffix_logprob_on_cache(model, copy.deepcopy(cache), first_lp, c, pos)
            )
        return totals, raw_row
    # shared: score on the live cache, crop back to the prefix after each —
    # validated once per process by _probe_cache_reuse before we get here.
    # Linear-attention layers only roll back if per-token state recording is
    # on (the probe verified the rewind is bit-exact with it activated).
    act = getattr(cache, "activate_past_recording", None)
    if act is not None:
        act()
    elif hasattr(cache, "layers") and hasattr(cache.layers[0], "activate_past_recording"):
        for l in cache.layers:
            l.activate_past_recording()
    for c in cand_list:
        pos = _cont_position_ids(rd, P, int(c.shape[1]), 1, model.device) if rd is not None else None
        totals.append(_suffix_logprob_on_cache(model, cache, first_lp, c, pos))
        cache.crop(-int(c.shape[1]))  # v5: negative int = remove last N tokens
    return totals, raw_row


def _get_cache_mode(model, prompt_ids, cand_ids, requested: str) -> str:
    global _CACHE_REUSE_MODE
    if requested != "auto":
        return requested
    if _CACHE_REUSE_MODE is None:
        _CACHE_REUSE_MODE = _probe_cache_reuse(model, prompt_ids, cand_ids)
    return _CACHE_REUSE_MODE


@torch.no_grad()
def score_sequence_logprob_batched(
    model,
    tokenizer,
    pinned: PinnedPrompt,
    candidates: list[str],
    alpha: float = 1.0,
    pmi_lambda: float = 0.0,
    leading_space: bool = False,
    batch_size: int = 8,
    cache_mode: str = "auto",
    prefill_kwargs: dict | None = None,
) -> DecisionResult:
    """D2 with one shared prefix prefill instead of one per candidate.

    Mathematically the same read as score_sequence_logprob (each candidate's
    continuation is scored given exactly the same context); only the forward
    scheduling differs. eval/batch_check.py is the parity gate — do not trust
    latency numbers before it passes.
    """
    cand_ids = _encode_candidates(tokenizer, candidates, leading_space)
    cand_list = [cand_ids[c] for c in candidates]

    t0 = time.perf_counter()
    mode = _get_cache_mode(model, pinned.input_ids, cand_list[0], cache_mode)
    raw_list, raw_row = _score_all_batched(
        model, pinned.input_ids, cand_list, mode, batch_size,
        prefill_kwargs=prefill_kwargs,
    )
    scoring_ms = (time.perf_counter() - t0) * 1000.0

    raw = {c: s for c, s in zip(candidates, raw_list)}
    first_tokens = [int(cand_ids[c][0, 0].item()) for c in candidates]

    if pmi_lambda > 0.0:
        cf_messages = question_messages(CONTENT_FREE_INPUT, candidates)
        from ..template import pin_prompt

        cf_pinned = pin_prompt(tokenizer, cf_messages)
        cf_raw, _ = _score_all_batched(
            model, cf_pinned.input_ids, cand_list, mode, batch_size
        )
        for c, s in zip(candidates, cf_raw):
            raw[c] -= pmi_lambda * s

    norm = {c: s / (len(cand_ids[c][0]) ** alpha) for c, s in raw.items()}
    scores = torch.tensor([norm[c] for c in candidates], dtype=torch.float32)
    probs = torch.softmax(scores, dim=-1)

    # pre-mask mass: free from the prefix forward's last-position logits
    # (the sequential path spends a whole extra prompt forward on this row)
    mass = premask_mass(raw_row, first_tokens)

    probs_d = {c: float(p) for c, p in zip(candidates, probs.tolist())}
    selected = max(probs_d, key=probs_d.get)
    return DecisionResult(
        selected=selected,
        confidence=probs_d[selected],
        probabilities=probs_d,
        raw_scores=raw,
        scoring_method=f"d2_sequence_logprob_batched({mode})",
        normalization=f"alpha={alpha}:pmi_lambda={pmi_lambda}",
        premask_mass=mass,
        prefill_ms=0.0,
        scoring_ms=scoring_ms,
        total_ms=scoring_ms,
    )
