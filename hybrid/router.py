"""Mode router — the runtime's main loop (three output contracts, per-step).

The three modes, each measured (see eval/reports/headline.md, webreplay.md):

  DECIDE      pure classification over a supplied candidate set, zero decode
              steps. Fastest mode; beats vanilla no-think generation on both
              MCQ (headline: 0.856 vs 0.764 at equal latency) and web element
              selection (Gate B: 0.320 vs 0.258 elem_acc). Output = the choice.
  TOOL_CALL   DECIDE tool -> DECIDE target (confidence-gated) -> THINK value ->
              assembled tool call {tool, target, value}. The structured-output
              mode: in the live runtime the assembly is a schema-constrained
              decode, structurally valid by construction. Output = tool call.
  TEXT        free-text generation with optional THINK. The only mode where
              generation IS the answer (final user-facing replies). Output =
              text.

Mode selection is a HARNESS-level decision based on affordances, not a learned
router: the harness knows what it can offer the model this step (a candidate
set / a tool schema / an open question), and offering a candidate set *is*
choosing DECIDE mode. This matters because the measured within-mode routing
signal is only moderate (web margin AUC 0.699) — good enough to gate
escalation inside a mode, not good enough to pick modes.

Within a mode, the calibrated gate (Gate) decides escalation; both gate fits
are domain-specific and shipped in configs (webgate.json for web steps,
thresholds.yaml for MCQ). The module is the open-weights counterpart of the
hosted server's "auto" routing, with auditable events on every turn.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .decide import decide
from .decisions import DecisionResult, with_temperature
from .engine import LoadedModel, generate_answer


@dataclass
class Gate:
    """Calibrated within-mode escalation thresholds (domain-fitted)."""

    accept_conf: float = 0.90
    temperature: float = 1.0
    premask_gate: float = 0.05

    @classmethod
    def from_json(cls, path: str | Path) -> "Gate":
        d = json.loads(Path(path).read_text())
        return cls(accept_conf=d["accept_conf"], temperature=d["temperature"])

    def escalates(self, res: DecisionResult) -> bool:
        t = with_temperature(res, self.temperature)
        if res.premask_mass < self.premask_gate:
            return True  # invalid measurement — never trust
        return t.confidence < self.accept_conf


@dataclass
class DecideRequest:
    """Affordance: an enumerable candidate set. Output = the chosen line."""

    question: str
    candidates: list[str]
    gate: Gate
    strategy: str = "reread"      # "reread" (information escalation, web) or
                                  # "redecide_think" (think-then-re-score, MCQ)
    reread: Callable[[list[str]], tuple[str, list[str]]] | None = None
    # reread(top_candidate_lines) -> (question_suffix, replacement_candidate_lines)


@dataclass
class ToolCallRequest:
    """Affordance: a tool schema. Output = {tool, target, value}.

    The three stages map onto the measured chain (webreplay H3_web):
      1. DECIDE the tool (small candidate set, one line per tool).
      2. DECIDE the target argument (larger set, op-conditioned, gated).
      3. THINK the free-form value, only if the schema needs one (~24 tokens).
    Callables keep the class web-agnostic; the web demo supplies them.
    """

    tool_question: str
    tool_candidates: list[str]                        # index i names tool i
    target_question: Callable[[str], str]             # tool -> question text
    target_candidates: Callable[[str], list[str]]     # tool -> candidate lines
    gate: Gate
    target_reread: Callable[[str, list[str]], tuple[str, list[str]]] | None = None
    tool_reread: Callable[[str], tuple[str, list[str]]] | None = None
    # chosen target line -> (question with the element in view, op candidate
    # lines). The H3B fix: op-first is only a prior for the target read; the
    # final tool is decided with the element visible (webreplay op_acc
    # 0.507 -> 0.843).
    value_needed: Callable[[str], bool] | None = None     # tool -> schema wants a value
    value_question: Callable[[str, str], str] | None = None  # (tool, chosen target) -> prompt
    target_to_ref: Callable[[str], Any] | None = None     # chosen line -> structured ref
    value_max_tokens: int = 24
    beam_ops: bool = False
    # beam_ops: score the target under EVERY offered tool and let the
    # element-conditioned op re-read arbitrate (log-linear product of the
    # three stage likelihoods). Motivated by the measured op-first failure
    # mode: committing to the op before any element is seen loses when the
    # op read is a near-tie (live s4: SELECT .354 / CLICK .326 / TYPE .319,
    # gold TYPE). Default False = the replay-measured H3B chain.


@dataclass
class TextRequest:
    """Affordance: an open question. Output = text.

    think: False = answer directly, True = forced THINK first, None = native
    default. Mode selection itself is the compute gate here — with no candidate
    set there is no distribution, so the gate lives in the harness decision to
    route this step to TEXT at all.
    """

    messages: list[dict]
    think: bool | None = None
    max_new_tokens: int = 768


@dataclass
class TurnOutcome:
    mode: str
    choice: str | None = None
    tool_call: dict | None = None
    text: str | None = None
    confidence: float | None = None
    escalated: bool = False
    think_ran: bool = False
    n_decisions: int = 0
    latency_ms: float = 0.0
    events: list = field(default_factory=list)

    @property
    def output(self) -> Any:
        return {"decide": self.choice, "tool_call": self.tool_call, "text": self.text}[self.mode]


def route(loaded: LoadedModel, req: DecideRequest | ToolCallRequest | TextRequest) -> TurnOutcome:
    """One runtime turn: dispatch on the harness-declared mode."""
    t0 = time.perf_counter()
    if isinstance(req, DecideRequest):
        out = _decide_turn(loaded, req)
    elif isinstance(req, ToolCallRequest):
        out = _tool_call_turn(loaded, req)
    elif isinstance(req, TextRequest):
        out = _text_turn(loaded, req)
    else:
        raise ValueError(f"unsupported request type: {type(req).__name__}")
    out.latency_ms = (time.perf_counter() - t0) * 1000
    return out


def _decide_turn(loaded: LoadedModel, req: DecideRequest) -> TurnOutcome:
    first = decide(loaded, req.question, req.candidates, method="d2", alpha=0.5, anchor=True)
    out = TurnOutcome(mode="decide", choice=first.selected, confidence=first.confidence,
                      n_decisions=1, escalated=req.gate.escalates(first))
    out.events.append({"stage": "decide", "selected": first.selected,
                       "confidence": first.confidence, "premask_mass": first.premask_mass})
    if not out.escalated:
        return out

    if req.strategy == "redecide_think":
        from .template import pin_after_think, question_messages
        msgs = question_messages(req.question, req.candidates)
        pin_after_think(loaded.tokenizer, loaded.model, msgs, max_think_tokens=256)
        second = decide(loaded, req.question, req.candidates, method="d2", alpha=0.5,
                        read_mode="postthink", max_think_tokens=256, anchor=True)
        out.think_ran = True
        out.events.append({"stage": "think_redecide"})
    elif req.strategy == "reread":
        if req.reread is None:
            raise ValueError("strategy 'reread' requires DecideRequest.reread")
        top = _top_lines(first, req.candidates, 3)
        suffix, reread_cands = req.reread(top)
        second = decide(loaded, req.question + suffix, reread_cands,
                        method="d2", alpha=0.5, anchor=True)
        out.events.append({"stage": "reread", "n_candidates": len(reread_cands)})
    else:
        raise ValueError(f"unknown decide strategy: {req.strategy}")

    out.events.append({"stage": "decide", "selected": second.selected,
                       "confidence": second.confidence})
    out.n_decisions = 2
    if second.confidence >= first.confidence:
        out.choice, out.confidence = second.selected, second.confidence
    return out


def _clean_generation(text: str) -> str:
    """User-facing output: cut at the first special token (the engine decodes
    with skip_special_tokens=False), then strip quotes/whitespace."""
    return re.split(r"<\|[^|>]*\|>", text)[0].strip().strip('"').strip()


def _beam_tool_call(loaded: LoadedModel, req: ToolCallRequest, out: TurnOutcome,
                    tool_res: DecisionResult) -> TurnOutcome:
    """Beam over the op factorization (req.beam_ops=True).

    The op-first chain commits to the op before any element is seen and loses
    when that read is a near-tie (live s4: SELECT .354 / CLICK .326 / TYPE
    .319, gold TYPE). Here every op gets its own op-conditioned target read,
    and the element-conditioned op re-read — the strongest op signal (replay
    op 0.507 → 0.843) — arbitrates: pairs (branch op, element) are scored by
    the log-linear product  log P(op) + log P(element|op) + log P(op'|element)
    and the winner's re-read op is the final tool. Replaces the per-branch
    rich re-read, so the gate does not apply in this mode."""
    prior = [tool_res.probabilities.get(c, 0.0) for c in req.tool_candidates]
    branches = []
    for i in range(len(req.tool_candidates)):
        tq, tc = req.target_question(i), req.target_candidates(i)
        res = decide(loaded, tq, tc, method="d2", alpha=0.5, anchor=True)
        out.n_decisions += 1
        ref = req.target_to_ref(res.selected) if req.target_to_ref else res.selected
        branches.append({"tool": i, "line": res.selected, "ref": ref, "conf": res.confidence})
        out.events.append({"stage": "beam_target", "tool": i, "selected": res.selected,
                           "confidence": res.confidence})

    rereads: dict = {}
    for b in branches:
        key = repr(b["ref"])
        if key in rereads or req.tool_reread is None:
            continue
        rq, rc = req.tool_reread(b["line"])
        rr = decide(loaded, rq, rc, method="d2", alpha=1.0, anchor=True)
        out.n_decisions += 1
        ri = rc.index(rr.selected) if rr.selected in rc else None
        rereads[key] = (ri, rr.confidence)
        out.events.append({"stage": "beam_reread", "element": b["ref"],
                           "tool": ri, "confidence": rr.confidence})

    best = None
    for b in branches:
        ri, rconf = rereads.get(repr(b["ref"]), (b["tool"], 1.0))
        if ri is None or prior[b["tool"]] <= 0 or b["conf"] <= 0 or rconf <= 0:
            continue
        score = math.log(prior[b["tool"]]) + math.log(b["conf"]) + math.log(rconf)
        if best is None or score > best[0]:
            best = (score, b, ri)
    if best is None:  # degenerate priors/rereads: fall back to the best branch as-is
        b = max(branches, key=lambda x: x["conf"])
        best = (math.log(max(b["conf"], 1e-12)), b, b["tool"])
    score, b, final_tool = best
    out.confidence = math.exp(score)
    out.events.append({"stage": "beam_argmax", "tool": final_tool,
                       "element": b["ref"], "score": round(score, 3)})

    value = None
    if req.value_needed and req.value_needed(final_tool) and req.value_question is not None:
        prompt = req.value_question(final_tool, b["line"])
        value, _ = generate_answer(loaded, [{"role": "user", "content": prompt}],
                                   max_new_tokens=req.value_max_tokens, enable_thinking=False)
        value = _clean_generation(value)
        out.think_ran = True
        out.events.append({"stage": "think_value", "n_tokens": req.value_max_tokens})

    ref = req.target_to_ref(b["line"]) if req.target_to_ref else b["line"]
    out.tool_call = {"tool": final_tool, "target": ref, "value": value}
    return out


def _tool_call_turn(loaded: LoadedModel, req: ToolCallRequest) -> TurnOutcome:
    # Stage 1 — DECIDE the tool (schema constraint: exactly one of the offered tools).
    tool_res = decide(loaded, req.tool_question, req.tool_candidates,
                      method="d2", alpha=1.0, anchor=True)
    tool = req.tool_candidates.index(tool_res.selected)
    out = TurnOutcome(mode="tool_call", n_decisions=1)
    out.events.append({"stage": "decide_tool", "tool": tool,
                       "confidence": tool_res.confidence})

    if req.beam_ops:
        return _beam_tool_call(loaded, req, out, tool_res)

    # Stage 2 — DECIDE the target argument, confidence-gated.
    tq, tc = req.target_question(tool), req.target_candidates(tool)
    res = decide(loaded, tq, tc, method="d2", alpha=0.5, anchor=True)
    out.n_decisions += 1
    out.confidence = res.confidence
    out.escalated = req.gate.escalates(res)
    out.events.append({"stage": "decide_target", "selected": res.selected,
                       "confidence": res.confidence, "escalated": out.escalated})
    if out.escalated and req.target_reread is not None:
        top = _top_lines(res, tc, 3)
        suffix, reread_cands = req.target_reread(tool, top)
        second = decide(loaded, tq + suffix, reread_cands, method="d2", alpha=0.5, anchor=True)
        out.n_decisions += 1
        out.events.append({"stage": "reread_target", "selected": second.selected,
                           "confidence": second.confidence})
        if second.confidence >= res.confidence:
            res = second
    chosen = res.selected

    # Stage 2b — re-DECIDE the tool with the chosen target in view. Op-first
    # alone sees no element and mispredicts TYPE/SELECT (it predicts SELECT
    # for most gold CLICK items); element identity is the strongest op signal.
    if req.tool_reread is not None:
        rq, rc = req.tool_reread(chosen)
        op2 = decide(loaded, rq, rc, method="d2", alpha=1.0, anchor=True)
        if op2.selected in rc:
            tool = rc.index(op2.selected)
        out.n_decisions += 1
        out.events.append({"stage": "reread_tool", "tool": tool,
                           "confidence": op2.confidence})

    # Stage 3 — THINK the free-form value, only when the schema needs one.
    value = None
    if req.value_needed and req.value_needed(tool) and req.value_question is not None:
        prompt = req.value_question(tool, chosen)
        value, _ = generate_answer(loaded, [{"role": "user", "content": prompt}],
                                   max_new_tokens=req.value_max_tokens, enable_thinking=False)
        # engine decodes with skip_special_tokens=False; cut at the first
        # special token — the value is the content before it
        value = _clean_generation(value)
        out.think_ran = True
        out.events.append({"stage": "think_value", "n_tokens": req.value_max_tokens})

    ref = req.target_to_ref(chosen) if req.target_to_ref else chosen
    out.tool_call = {"tool": tool, "target": ref, "value": value}
    return out


def _text_turn(loaded: LoadedModel, req: TextRequest) -> TurnOutcome:
    text, meta = generate_answer(loaded, req.messages,
                                 max_new_tokens=req.max_new_tokens,
                                 enable_thinking=req.think)
    # user-facing output: cut the trailing special tokens the raw decode keeps
    text = _clean_generation(text)
    return TurnOutcome(mode="text", text=text, think_ran=bool(req.think),
                       events=[{"stage": "generate", "think": req.think,
                                "meta": {k: meta[k] for k in meta if isinstance(meta[k], (int, float, str, bool))}}])


def _top_lines(res: DecisionResult, candidates: list[str], k: int) -> list[str]:
    order = sorted(res.raw_scores, key=res.raw_scores.get, reverse=True)
    return [c for c in order[:k] if c in candidates]


if __name__ == "__main__":
    # Plumbing self-test with a stubbed backbone (no GPU): validates the three
    # mode dispatches, the gate, and the TurnOutcome output contract.
    import random

    from .decisions import DecisionResult

    rng = random.Random(0)

    def fake_decide(loaded, question, candidates, **kw):
        i = rng.randrange(len(candidates))
        conf = 0.99 if "obvious" in question else rng.uniform(0.05, 0.4)
        return DecisionResult(selected=candidates[i], confidence=conf,
                              probabilities={}, scoring_method="stub", premask_mass=0.9,
                              raw_scores={c: 1.0 / (j + 1) for j, c in enumerate(candidates)})

    def fake_generate(loaded, messages, max_new_tokens=256, enable_thinking=None):
        return "Generated answer.", {}

    # python -m hybrid.router runs this block as __main__, a different module
    # object than "hybrid.router" — patch this namespace's globals directly.
    g = globals()
    g["decide"] = fake_decide
    g["generate_answer"] = fake_generate

    gate = Gate(accept_conf=0.5, temperature=1.0)
    loaded = object()

    # DECIDE, confident -> accepted, no escalation
    o = route(loaded, DecideRequest("obvious: pick", ["a", "b"], gate))
    assert o.mode == "decide" and o.output in ("a", "b") and not o.escalated and o.n_decisions == 1

    # DECIDE, low confidence + reread strategy -> escalates
    calls = []
    o = route(loaded, DecideRequest("hard: pick", ["a", "b", "c"], gate,
                                    reread=lambda top: (calls.append(top) or (" again", top[:2]))))
    assert o.escalated and o.n_decisions == 2 and len(calls[0]) == 3 and o.output in ("a", "b", "c")

    # TOOL_CALL: tool decide -> gated target -> value think only when needed
    o = route(loaded, ToolCallRequest(
        tool_question="which tool?", tool_candidates=["TYPE — enter text", "CLICK — press"],
        target_question=lambda t: f"target for {t}?",
        target_candidates=lambda t: ["[1] input", "[2] button"],
        gate=Gate(accept_conf=0.99, temperature=1.0),
        target_reread=lambda t, top: (" focus", top),
        value_needed=lambda t: t == 0,
        value_question=lambda t, target: "what text?",
        target_to_ref=lambda line: line.split("]")[0].strip("[]"),
    ))
    assert o.mode == "tool_call" and set(o.tool_call) == {"tool", "target", "value"}
    assert o.escalated and o.n_decisions == 3
    assert (o.tool_call["value"] is None) == (o.tool_call["tool"] == 1)

    # TOOL_CALL with the H3B op re-read: the tool may flip once the element is
    # in view; value_needed then keys off the FINAL tool.
    seen = []
    o = route(loaded, ToolCallRequest(
        tool_question="which tool?", tool_candidates=["TYPE — enter text", "CLICK — press"],
        target_question=lambda t: f"target for {t}?",
        target_candidates=lambda t: ["[1] input", "[2] button"],
        gate=Gate(accept_conf=0.99, temperature=1.0),
        target_reread=lambda t, top: (" focus", top),
        tool_reread=lambda chosen: (seen.append(chosen) or
                                    (f"chosen {chosen}", ["TYPE — enter text", "CLICK — press"])),
        value_needed=lambda t: t == 0,
        value_question=lambda t, target: "what text?",
    ))
    assert o.mode == "tool_call" and len(seen) == 1 and o.n_decisions == 4
    assert (o.tool_call["value"] is None) == (o.tool_call["tool"] == 1)

    # TOOL_CALL beam-over-ops: targets scored under every tool, the
    # element-conditioned re-read arbitrates, value keys off the FINAL tool.
    o = route(loaded, ToolCallRequest(
        tool_question="which tool?", tool_candidates=["TYPE — enter text", "CLICK — press"],
        target_question=lambda t: f"target for {t}?",
        target_candidates=lambda t: ["[1] input", "[2] button"],
        gate=Gate(accept_conf=0.99, temperature=1.0),
        tool_reread=lambda chosen: ("chosen", ["TYPE — enter text", "CLICK — press"]),
        value_needed=lambda t: t == 0,
        value_question=lambda t, target: "what text?",
        beam_ops=True,
    ))
    assert o.mode == "tool_call" and o.tool_call["target"] in ("[1] input", "[2] button")
    n_rereads = len({e["element"] for e in o.events if e["stage"] == "beam_reread"})
    assert o.n_decisions == 1 + 2 + n_rereads and any(e["stage"] == "beam_argmax" for e in o.events)
    assert (o.tool_call["value"] is None) == (o.tool_call["tool"] == 1)
    print("router self-test OK —",
          {"decide": o.mode, "out": o.tool_call, "events": len(o.events)})
