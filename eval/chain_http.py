#!/usr/bin/env python3
"""Decision reads for the decide chain over llama-server HTTP.

The chain-on-llama.cpp port: same d2 math as hybrid.scoring
(sequence logprob, alpha normalization, premask mass, temperature
re-tempering via the shared DecisionResult), different transport.

d2 per candidate: ONE grammar-forced generation whose GBNF grammar is
exactly the candidate string; the per-token logprobs the server reports
are the model's RAW distribution (verified 2026-09-21 on Gemma 12B: a
forced '[' reads -24.24 while the true top-1 '<|channel>' reads -0.00 —
grammar-masked logprobs would read ~0), so the sum IS the teacher-forced
sequence logprob. Alpha normalization uses the GENERATED token count:
grammar piece boundaries can differ from a free /tokenize pass.

Read-point pinning, the llama.cpp way (both verified live on Gemma 12B):
  decision reads   chat_template_kwargs {enable_thinking: false} — the
                   closed-channel prefill puts the read position on the
                   content channel (op words land there: CLICK -0.45 vs
                   SELECT -11.36; without it the position is thinking-
                   channel-dominated and no op word is visible)
  generation       reasoning_budget_tokens: 0 — the verified generation
                   knob; the two knobs must NOT be combined (earlier
                   finding: combined enforcement silently breaks)

hybrid.decisions is imported for the result structures so the runtime's
with_temperature gate code runs unchanged; it is torch-free. Everything
else here is stdlib-only.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hybrid.decisions import DecisionResult  # noqa: E402

READ_KNOB = {"chat_template_kwargs": {"enable_thinking": False}}
GEN_KNOB = {"reasoning_budget_tokens": 0}

# Fixed decision wording — mirror of hybrid.template.question_messages
# (torch-importing module; the strings are the source of truth, the mirror
# is asserted equal by the tests).
DECISION_INSTRUCTION = "Answer with exactly one of the listed options."
DECISION_ANCHOR = "Reply with the option text only, nothing else."


def gbnf_escape(s: str) -> str:
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def endpoint(base_url, path):
    """Join base and path without doubling the version prefix (accepts both
    http://host:8998 and http://host:8998/v1 — the same normalizer the
    onboarding script needed; a doubled /v1/v1 404s on real servers)."""
    base = base_url.rstrip("/")
    if path.startswith("/v1/") and base.endswith("/v1"):
        path = path[len("/v1"):]
    return base + path


def post(base_url, path, body, timeout):
    req = urllib.request.Request(
        endpoint(base_url, path), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def decision_messages(question: str, candidates: list[str],
                      anchor: bool = True) -> list[dict]:
    choice_lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    content = f"{question}\n\nOptions:\n{choice_lines}\n\n{DECISION_INSTRUCTION}"
    if anchor:
        content += f" {DECISION_ANCHOR}"
    return [{"role": "user", "content": content}]


def forced_score(base_url, model, messages, candidate, timeout):
    """One grammar-forced generation; returns (seq_logprob, n_tokens,
    first_piece) or None on hard failure."""
    grammar = f'root ::= "{gbnf_escape(candidate)}"'
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": len(candidate) + 8, "logprobs": True,
            "top_logprobs": 20, "grammar": grammar, **READ_KNOB}
    try:
        payload, _ = post(base_url, "/v1/chat/completions", body, timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    except Exception:
        return None
    try:
        toks = payload["choices"][0]["logprobs"]["content"] or []
    except Exception:
        return None
    if not toks:
        return None
    return (sum(t.get("logprob") or 0.0 for t in toks), len(toks),
            toks[0].get("token", ""))


def premask_http(base_url, model, messages, first_pieces, timeout):
    """Sum of DISTINCT first-piece masses at the read position (the same
    de-duplication as hybrid.scoring.premask; pieces outside the top-20
    contribute 0 — a documented truncation, flagged via the mass itself)."""
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
            **READ_KNOB}
    try:
        payload, _ = post(base_url, "/v1/chat/completions", body, timeout)
        c0 = ((payload["choices"][0].get("logprobs") or {})
              .get("content") or [{}])[0]
        top = {t.get("token"): t.get("logprob")
               for t in (c0.get("top_logprobs") or [])}
    except Exception:
        return 0.0
    return sum(math.exp(top[p]) for p in set(first_pieces)
               if top.get(p) is not None)


def decide_http(base_url, model, question, candidates, method="d2",
                alpha=1.0, anchor=True, timeout=120, workers=4):
    """d2 sequence-logprob decision over HTTP; DecisionResult-compatible.

    Returns None if any candidate's forced read fails — the caller decides
    how to degrade (the proxy's guarded wrapper hands the turn to the
    model, the chain's own low-confidence path). method is accepted for
    call-site parity; only d2 travels over HTTP (the chain's reads are
    all d2).
    """
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate candidates")
    if len(candidates) < 2:
        raise ValueError("need at least 2 candidates")
    t0 = time.perf_counter()
    msgs = decision_messages(question, candidates, anchor=anchor)
    with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as ex:
        scores = list(ex.map(
            lambda c: forced_score(base_url, model, msgs, c, timeout),
            candidates))
    if any(s is None for s in scores):
        return None
    raw = {c: s[0] for c, s in zip(candidates, scores)}
    norm = {c: raw[c] / (scores[i][1] ** alpha)
            for i, c in enumerate(candidates)}
    m = max(norm.values())
    exps = {c: math.exp(v - m) for c, v in norm.items()}
    z = sum(exps.values())
    probs = {c: e / z for c, e in exps.items()}
    selected = max(probs, key=probs.get)
    mass = premask_http(base_url, model, msgs,
                        [s[2] for s in scores], timeout)
    ms = (time.perf_counter() - t0) * 1000.0
    return DecisionResult(
        selected=selected, confidence=probs[selected],
        probabilities=probs, raw_scores=raw,
        scoring_method="d2_sequence_logprob_http",
        normalization=f"alpha={alpha}:pmi_lambda=0.0",
        premask_mass=mass, scoring_ms=ms, total_ms=ms)


def think_value_http(base_url, model, content, timeout=120,
                     max_new_tokens=24):
    """THINK stage over HTTP (mirror of run_chain.think_value's generate:
    ~24 tokens, no think, exact text back)."""
    body = {"model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": max_new_tokens, **GEN_KNOB}
    try:
        payload, _ = post(base_url, "/v1/chat/completions", body, timeout)
        text = payload["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""
    return text.strip().strip('"').strip()
