"""One-step diagnostic for the batched-D2 suffix-forward mismatch.

v2: each variant gets a FRESH prefix forward (v2 of the probe had later
variants reading a cache already mutated by an earlier variant's suffix
forward), and tests the linear-attention layers' activate_past_recording +
crop rewind path.

  ref      — full forward over [prompt + cand] (the sequential computation)
  pristine — suffix forward against the untouched prefix cache
  copy     — same against a deepcopy of a pristine prefix cache
  rewind   — activate_past_recording (if present), score, crop, rescore,
             bit-equality check
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from hybrid.actions import candidate_lines, context_text, load_web_steps  # noqa: E402
from hybrid.engine import load_model  # noqa: E402
from hybrid.scoring.sequence_logprob import _candidate_logprob, _encode_candidates, _prefix_state  # noqa: E402
from hybrid.template import pin_prompt, question_messages  # noqa: E402

DATASET = Path.home() / "hybrid-qwen" / "eval" / "datasets" / "webreplay_v1" / "webreplay_v1.jsonl"


@torch.no_grad()
def token_logprobs(model, prompt_ids, cand_ids):
    """Per-token logprobs, full-forward reference."""
    ids = torch.cat([prompt_ids.to(model.device), cand_ids.to(model.device)], dim=1)
    logits = model(input_ids=ids).logits.float()
    start = prompt_ids.shape[1] - 1
    lp = torch.log_softmax(logits[0, start : start + cand_ids.shape[1]], dim=-1)
    tgt = cand_ids.to(model.device).squeeze(0)
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def suffix_per_token(model, cache, first_lp, cand_ids):
    ids = cand_ids.to(model.device)
    out = model(input_ids=ids, past_key_values=cache, use_cache=False)
    lg = out.logits[0].float()
    per = [first_lp[ids[0, 0]]]
    if ids.shape[1] > 1:
        lp = torch.log_softmax(lg[: ids.shape[1] - 1], dim=-1)
        per.extend(list(lp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)))
    return torch.stack(per)


def main():
    steps = load_web_steps(DATASET)
    step = next(s for s in steps if 12 <= len(s.candidates) <= 16)
    lines = candidate_lines(step)
    print(f"step {step.id}  n_candidates={len(lines)}")

    loaded = load_model()
    model, tok = loaded.model, loaded.tokenizer

    msgs = question_messages(context_text(step), lines, anchor=True)
    pinned = pin_prompt(tok, msgs)
    cand = lines[len(lines) // 2]
    cand_ids = _encode_candidates(tok, [cand], False)[cand]
    P = int(pinned.input_ids.shape[1])
    L = int(cand_ids.shape[1])
    print(f"prompt_len={P}  cand_len={L}  cand={cand[:60]!r}")

    ref = _candidate_logprob(model, pinned.input_ids, cand_ids)
    per_ref = token_logprobs(model, pinned.input_ids, cand_ids)
    print(f"ref full-forward total = {ref:.4f}")

    def report(name, per):
        d = (per - per_ref).abs()
        tot = float(per.sum())
        print(f"{name}: total={tot:.4f}  |d_total|={abs(tot - ref):.4f}  "
              f"max_token_d={float(d.max()):.4f}")
        for i in torch.argsort(d, descending=True)[:3]:
            i = int(i)
            print(f"   token {i}: ref={float(per_ref[i]):.4f} got={float(per[i]):.4f}")

    # pristine: fresh prefix, score directly on it
    cache, first_lp, _raw = _prefix_state(model, pinned.input_ids)
    report("pristine", suffix_per_token(model, cache, first_lp, cand_ids))
    del cache

    # copy: fresh prefix, deepcopy, score on the copy
    cache, first_lp, _raw = _prefix_state(model, pinned.input_ids)
    report("copy    ", suffix_per_token(model, copy.deepcopy(cache), first_lp, cand_ids))
    del cache

    # rewind: fresh prefix, activate per-token state recording if the
    # linear-attention layers support it, score, crop, rescore
    cache, first_lp, _raw = _prefix_state(model, pinned.input_ids)
    activated = False
    if hasattr(cache, "activate_past_recording"):
        cache.activate_past_recording()
        activated = True
    elif hasattr(cache, "layers") and hasattr(cache.layers[0], "activate_past_recording"):
        for l in cache.layers:
            l.activate_past_recording()
        activated = True
    print(f"activate_past_recording invoked: {activated}")
    lp1 = suffix_per_token(model, cache, first_lp, cand_ids)
    try:
        cache.crop(-L)
        lp2 = suffix_per_token(model, cache, first_lp, cand_ids)
        cache.crop(-L)
        lp3 = suffix_per_token(model, cache, first_lp, cand_ids)
        print(f"rewind: lp1={float(lp1.sum()):.6f} lp2={float(lp2.sum()):.6f} "
              f"lp3={float(lp3.sum()):.6f} bit_eq(1,2)={bool(torch.equal(lp1, lp2))} "
              f"bit_eq(1,3)={bool(torch.equal(lp1, lp3))}")
    except Exception as e:  # noqa: BLE001
        print(f"rewind: crop raised {e!r}")


if __name__ == "__main__":
    main()
