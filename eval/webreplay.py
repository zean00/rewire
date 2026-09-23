"""Offline web-replay comparison (browser-use Gate B).

Arms, all on identical replayed steps (Mind2Web cleaned_html candidates):

  D2E   DECIDE over element lines (anchored D2) + 3-way op read
  H_web D2E gate -> THINK -> re-DECIDE (hybrid_decide_turn, reusing the MCQ
        thresholds + T=0.5 calibration unchanged — recalibration is step #5)
  C_web vanilla no-think generation, options shown
  G_web vanilla think-always generation (native default), options shown;
        subsampled (--g-n) because think-always costs ~17-30 s/step

  v2: D2E2 (op-first -> op-conditioned element read), H2_web (op-first ->
  confidence gate -> rich re-read of top-3, NO think), gate fitted by
  eval.webgate on the cal split (configs/webgate.json).

  v3 (tool-call chain, structured-output mode): H3_web = DECIDE op -> DECIDE
  element (gated) -> op re-read with element in view -> THINK value
  (TYPE/SELECT) -> assembled tool call.
  C3_web / G3_web = vanilla one-shot JSON tool call, no-think / think-always.
  OPFIX_web = op re-read over D2E2's chosen elements (op-stage ablation).
  Metrics add value_acc (exact, stage-conditional) and action_acc
  (op AND element AND value).

Primary metric: element accuracy (Mind2Web convention). Joint = element AND op.
Vanilla replies are parsed number-first, then by candidate-text match. Records
are appended per item (checkpoint/resume by id, same convention as the MCQ
runner after the fmedian crash lesson).

Usage:
  python -m eval.webreplay --arms D2E C_web --n 60        # smoke
  python -m eval.webreplay                                 # full run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from hybrid.actions import OPS, candidate_lines, context_text, load_web_steps
from hybrid.decide import decide
from hybrid.engine import generate_answer, load_model
from hybrid.runtime import RuntimeConfig, hybrid_decide_turn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"
DATASET = ROOT / "eval/datasets/webreplay_v1/webreplay_v1.jsonl"


def parse_choice(reply: str, lines: list[str]) -> int | None:
    """Index of the chosen element line; number first, then text match."""
    for tok in re.findall(r"\[?(\d+)\]?", reply):
        k = int(tok)
        if 1 <= k <= len(lines):
            return k - 1
    low = reply.lower()
    for i, line in enumerate(lines):
        for chunk in re.findall(r"'([^']+)'", line):
            if chunk.lower() in low:
                return i
    return None


def parse_op(reply: str) -> str | None:
    up = reply.upper()
    for op in OPS:
        if op in up:
            return op
    return None


def parse_tool_call(reply: str, lines: list[str]):
    """Parse a one-shot JSON tool call (v3 vanilla arms): (op, idx, value) or None."""
    s = re.sub(r"<\|[^|>]*\|>", "", reply)  # trailing special tokens from decode
    s = re.sub(r"```(?:json)?|```", "", s).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    op = d.get("op") if d.get("op") in OPS else None
    try:
        k = int(d.get("element"))
    except (TypeError, ValueError):
        k = None
    idx = k - 1 if k is not None and 1 <= k <= len(lines) else None
    v = d.get("value")
    return op, idx, (str(v).strip() if v is not None else None)


def _norm_val(s: str) -> str:
    return " ".join(str(s).split()).lower().rstrip(".")


def _value_fields(op_pred, gold_op: str, gold_value, value_pred):
    """Value scoring. value_scored = op decided correctly AND gold has a value —
    the stage-conditional metric (value quality given correct upstream decisions).
    Returns (value_exact, value_loose, value_scored)."""
    gold_val = (gold_value or "").strip()
    if not (op_pred == gold_op and gold_op in ("TYPE", "SELECT") and gold_val):
        return True, True, False
    p, g = _norm_val(value_pred or ""), _norm_val(gold_val)
    exact = p == g
    loose = exact or (len(g) > 1 and g in p) or (len(p) > 1 and p in g)
    return exact, loose, True


def gold_idx(step, lines) -> int:
    for i, c in enumerate(step.candidates):
        if c["bid"] == step.gold_bid:
            return i
    raise ValueError(f"gold bid missing from candidates: {step.id}")


@torch.no_grad()
def run_d2e(loaded, steps, rpath: Path, done: set):
    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        res = decide(loaded, context_text(step), lines, method="d2",
                     alpha=0.5, anchor=True)
        el_ms = (time.perf_counter() - t0) * 1000
        sel = res.selected
        op_lines = [f"{op} — {'press/activate' if op == 'CLICK' else 'enter a value into' if op == 'TYPE' else 'choose an option from'} the selected element"
                    for op in OPS]
        op_res = decide(loaded, context_text(step) + f"\n\nThe chosen element is: {sel}",
                        op_lines, method="d2", alpha=1.0, anchor=True)
        op_pred = OPS[op_lines.index(op_res.selected)] if op_res.selected in op_lines else None
        rec = {
            "arm": "D2E", "id": step.id, "op": step.op,
            "correct": sel == lines[gold_idx(step, lines)],
            "op_pred": op_pred, "op_correct": op_pred == step.op,
            "confidence": res.confidence, "premask_mass": res.premask_mass,
            "latency_ms": el_ms + op_res.total_ms,
            "n_candidates": len(lines), "n_candidates_total": step.n_candidates_total,
            "selected_bid": step.candidates[lines.index(sel)]["bid"] if sel in lines else None,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [D2E] {k+1}/{len(steps)}", flush=True)


@torch.no_grad()
def run_vanilla(loaded, steps, rpath: Path, done: set, arm: str, think: bool, max_new: int):
    """Vanilla prompts are built here, not via question_messages: the web
    instruction asks for 'number + operation' so the op metric is measurable
    from the reply (the smoke test showed number-only replies make op_acc
    structurally 0)."""

    def messages(step, lines):
        content = (context_text(step) + "\n\nOptions:\n" + "\n".join(lines) +
                   "\n\nReply with the option number and the operation "
                   "(for example '3 CLICK') and nothing else.")
        return [{"role": "user", "content": content}]

    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        answer, _ = generate_answer(loaded, messages(step, lines), max_new_tokens=max_new,
                                    enable_thinking=think)
        ms = (time.perf_counter() - t0) * 1000
        idx = parse_choice(answer, lines)
        rec = {
            "arm": arm, "id": step.id, "op": step.op,
            "correct": idx is not None and step.candidates[idx]["bid"] == step.gold_bid,
            "op_pred": parse_op(answer), "op_correct": parse_op(answer) == step.op,
            "confidence": None, "premask_mass": None,
            "latency_ms": ms, "n_candidates": len(lines),
            "n_candidates_total": step.n_candidates_total,
            "selected_bid": step.candidates[idx]["bid"] if idx is not None else None,
            "parsed": idx is not None,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [{arm}] {k+1}/{len(steps)}", flush=True)


@torch.no_grad()
def run_hweb(loaded, steps, cfg, rpath: Path, done: set):
    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        turn = hybrid_decide_turn(loaded, context_text(step), lines, cfg,
                                  escalation="redecide")
        ms = (time.perf_counter() - t0) * 1000
        sel = turn.final.selected
        rec = {
            "arm": "H_web", "id": step.id, "op": step.op,
            "correct": sel == lines[gold_idx(step, lines)],
            "op_pred": None, "op_correct": False,
            "confidence": turn.final.confidence, "premask_mass": turn.final.premask_mass,
            "latency_ms": ms, "escalated": turn.escalated,
            "n_candidates": len(lines), "n_candidates_total": step.n_candidates_total,
            "selected_bid": step.candidates[lines.index(sel)]["bid"] if sel in lines else None,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [H_web] {k+1}/{len(steps)}", flush=True)


def _append(rpath: Path, rec: dict):
    with open(rpath, "a") as f:
        f.write(json.dumps(rec) + "\n")


OP_DESC = {
    "CLICK": "CLICK — press or activate the target element (buttons, links, icons)",
    "TYPE": "TYPE — enter text into the target element (input fields, search boxes)",
    "SELECT": "SELECT — choose an option from the target element (dropdowns, comboboxes)",
}


def _op_first(loaded, step):
    """Op-first factorization: decide the operation before the element."""
    op_lines = [OP_DESC[op] for op in OPS]
    op_res = decide(loaded, context_text(step), op_lines, method="d2",
                    alpha=1.0, anchor=True)
    op = OPS[op_lines.index(op_res.selected)] if op_res.selected in op_lines else None
    return op, op_res


# v1's element-conditioned op read (the measured-good 3-way read, op_acc 0.490):
# element identity is the strongest op signal — the op-first read, which sees
# no element, predicts SELECT for 61% of gold CLICK items.
OP_LINES_ELEM = [
    f"{op} — {'press/activate' if op == 'CLICK' else 'enter a value into' if op == 'TYPE' else 'choose an option from'} the selected element"
    for op in OPS
]


def _op_elem_read(loaded, step, sel_line: str, lines: list[str] | None = None) -> str | None:
    """3-way op read with the chosen element in view.

    lines=None: element line only (v1's read, op_acc 0.490). lines=list: the
    full option block is also visible — vanilla one-shot gets op_acc ~0.85
    precisely because it sees the element lines when it names the operation;
    this levels that context advantage (H3B)."""
    ctx = context_text(step)
    if lines:
        ctx += "\n\nOptions:\n" + "\n".join(lines)
    ctx += f"\n\nThe chosen element is: {sel_line}"
    op_res = decide(loaded, ctx, OP_LINES_ELEM, method="d2", alpha=1.0, anchor=True)
    return OPS[OP_LINES_ELEM.index(op_res.selected)] if op_res.selected in OP_LINES_ELEM else None


@torch.no_grad()
def run_d2e2(loaded, steps, rpath: Path, done: set, split: str):
    """v2 decide arm: op-first read, then op-conditioned element read.

    Records margin (top1-top2) so the gate-fit script can compare routing
    signals (confidence vs margin) on the cal split.
    """
    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        op, op_res = _op_first(loaded, step)
        res = decide(loaded, context_text(step, op), lines, method="d2",
                     alpha=0.5, anchor=True)
        ms = (time.perf_counter() - t0) * 1000
        probs = sorted(res.probabilities.values(), reverse=True)
        margin = probs[0] - probs[1] if len(probs) > 1 else 1.0
        rec = {
            "arm": "D2E2", "id": step.id, "op": step.op, "split": split,
            "correct": res.selected == lines[gold_idx(step, lines)],
            "op_pred": op, "op_correct": op == step.op, "op_conf": op_res.confidence,
            "confidence": res.confidence, "margin": margin,
            "premask_mass": res.premask_mass,
            "latency_ms": ms, "n_candidates": len(lines),
            "n_candidates_total": step.n_candidates_total,
            "selected_bid": step.candidates[lines.index(res.selected)]["bid"],
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [D2E2] {k+1}/{len(steps)}", flush=True)


@torch.no_grad()
def run_h2web(loaded, steps, rpath: Path, done: set, split: str, gate: dict):
    """v2 hybrid: op-first decide -> confidence gate -> rich re-read of top-3.

    No THINK anywhere: Gate B showed think->re-DECIDE is harmful on web steps
    (0.188 tail vs 0.271 plain D2E). The escalation swaps reasoning for
    information — top-3 candidates re-scored with subtree/section context.
    Final = the more confident of (first read, re-read).
    """
    from hybrid.actions import rich_blocks
    from hybrid.decisions import with_temperature

    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        op, op_res = _op_first(loaded, step)
        res = decide(loaded, context_text(step, op), lines, method="d2",
                     alpha=0.5, anchor=True)
        tempered = with_temperature(res, gate["temperature"])
        escalated = tempered.confidence < gate["accept_conf"]
        if escalated:
            order = sorted(res.raw_scores, key=res.raw_scores.get, reverse=True)
            bids = [step.candidates[lines.index(l)]["bid"] for l in order[:3]]
            blocks = rich_blocks(step, bids)
            reread = decide(loaded,
                            context_text(step, op) + "\n\nFocus on these options:\n" + "\n".join(blocks),
                            blocks, method="d2", alpha=0.5, anchor=True)
            if reread.confidence >= res.confidence:
                sel_bid = bids[blocks.index(reread.selected)]
                conf = reread.confidence
            else:
                sel_bid = step.candidates[lines.index(res.selected)]["bid"]
                conf = res.confidence
        else:
            sel_bid = step.candidates[lines.index(res.selected)]["bid"]
            conf = res.confidence
        ms = (time.perf_counter() - t0) * 1000
        rec = {
            "arm": "H2_web", "id": step.id, "op": step.op, "split": split,
            "correct": sel_bid == step.gold_bid,
            "op_pred": op, "op_correct": op == step.op,
            "confidence": conf, "escalated": escalated,
            "latency_ms": ms, "n_candidates": len(lines),
            "n_candidates_total": step.n_candidates_total,
            "selected_bid": sel_bid,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [H2_web] {k+1}/{len(steps)}", flush=True)


def _think_value(loaded, step, op: str, sel_line: str) -> str:
    """THINK stage of the v3 chain: generate the argument value.
    Small bounded generation (~24 tokens) — the value itself, not a rationale."""
    ask = ("What exact text should be typed into it? Reply with the text and nothing else."
           if op == "TYPE" else
           "Which option should be selected? Reply with the option's value and nothing else.")
    content = context_text(step, op) + f"\n\nChosen element: {sel_line}\n\n{ask}"
    answer, _ = generate_answer(loaded, [{"role": "user", "content": content}],
                                max_new_tokens=24, enable_thinking=False)
    # engine decodes with skip_special_tokens=False; regex parsers upstream
    # tolerate the trailing special tokens, raw text cannot.
    answer = re.split(r"<\|[^|>]*\|>", answer)[0].strip()
    return answer.strip('"').strip()


@torch.no_grad()
def run_h3web(loaded, steps, rpath: Path, done: set, split: str, gate: dict,
              arm: str = "H3_web", op_options: bool = False):
    """v3 tool-call chain — the runtime's structured-output mode, measured:
    DECIDE op (op-first prior) -> DECIDE element (confidence-gated rich re-read)
    -> DECIDE op again with the element in view -> THINK value (TYPE/SELECT,
    ~24 tokens) -> output = assembled tool call {op, element, value}. In the
    live runtime the assembly is a schema-constrained decode, so it is
    structurally valid by construction; here the harness assembles it (same
    guarantee, zero decode cost).

    arm="H3B_web", op_options=True: the final op read also sees the element
    lines (the context vanilla one-shot has — H3's measured weakness)."""
    from hybrid.actions import rich_blocks
    from hybrid.decisions import with_temperature

    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        op, op_res = _op_first(loaded, step)
        res = decide(loaded, context_text(step, op), lines, method="d2",
                     alpha=0.5, anchor=True)
        tempered = with_temperature(res, gate["temperature"])
        escalated = tempered.confidence < gate["accept_conf"]
        if escalated:
            order = sorted(res.raw_scores, key=res.raw_scores.get, reverse=True)
            bids = [step.candidates[lines.index(l)]["bid"] for l in order[:3]]
            blocks = rich_blocks(step, bids)
            reread = decide(loaded,
                            context_text(step, op) + "\n\nFocus on these options:\n" + "\n".join(blocks),
                            blocks, method="d2", alpha=0.5, anchor=True)
            if reread.confidence >= res.confidence:
                sel_bid = bids[blocks.index(reread.selected)]
                conf = reread.confidence
            else:
                sel_bid = step.candidates[lines.index(res.selected)]["bid"]
                conf = res.confidence
        else:
            sel_bid = step.candidates[lines.index(res.selected)]["bid"]
            conf = res.confidence
        sel_line = lines[next(i for i, c in enumerate(step.candidates) if c["bid"] == sel_bid)]
        # op re-read with the chosen element in view: op-first alone is a weak
        # prior used to condition the element read; the final tool name is
        # decided with the target visible.
        op_final = (_op_elem_read(loaded, step, sel_line, lines if op_options else None) or op)
        value_pred = (_think_value(loaded, step, op_final, sel_line)
                      if op_final in ("TYPE", "SELECT") and (step.value or "").strip() else None)
        ms = (time.perf_counter() - t0) * 1000
        v_exact, v_loose, v_scored = _value_fields(op_final, step.op, step.value, value_pred)
        rec = {
            "arm": arm, "id": step.id, "op": step.op, "split": split,
            "correct": sel_bid == step.gold_bid,
            "op_pred": op_final, "op_correct": op_final == step.op,
            "op_first_pred": op,
            "confidence": conf, "escalated": escalated,
            "value_pred": value_pred, "value_exact": v_exact,
            "value_loose": v_loose, "value_scored": v_scored,
            "action_correct": (op_final == step.op and sel_bid == step.gold_bid
                               and (not v_scored or v_exact)),
            "tool_call": {"op": op_final, "element": sel_bid, "value": value_pred},
            "latency_ms": ms, "n_candidates": len(lines),
            "n_candidates_total": step.n_candidates_total,
            "selected_bid": sel_bid, "parsed": True,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [{arm}] {k+1}/{len(steps)}", flush=True)


def _json_messages(step, lines):
    content = (context_text(step) + "\n\nOptions:\n" + "\n".join(lines) +
               '\n\nRespond with a single JSON tool call and nothing else, in this exact format:\n'
               '{"op": "CLICK", "element": 3, "value": null}\n'
               'where "op" is one of CLICK, TYPE, SELECT; "element" is the option number; '
               '"value" is the text to type or the option to select (null for CLICK).')
    return [{"role": "user", "content": content}]


@torch.no_grad()
def run_c3web(loaded, steps, rpath: Path, done: set, split: str,
              arm: str, think: bool, max_new: int):
    """v3 vanilla one-shot: a single generation must produce the whole tool call
    (op + element + value as JSON). C3_web = no-think, G3_web = think-always."""
    for k, step in enumerate(steps):
        if step.id in done:
            continue
        lines = candidate_lines(step)
        t0 = time.perf_counter()
        answer, _ = generate_answer(loaded, _json_messages(step, lines),
                                    max_new_tokens=max_new, enable_thinking=think)
        ms = (time.perf_counter() - t0) * 1000
        tc = parse_tool_call(answer, lines)
        if tc is None:
            op, idx, value_pred = None, None, None
        else:
            op, idx, value_pred = tc
        sel_bid = step.candidates[idx]["bid"] if idx is not None else None
        v_exact, v_loose, v_scored = _value_fields(op, step.op, step.value, value_pred)
        rec = {
            "arm": arm, "id": step.id, "op": step.op, "split": split,
            "correct": sel_bid == step.gold_bid,
            "op_pred": op, "op_correct": op == step.op,
            "confidence": None, "escalated": False,
            "value_pred": value_pred, "value_exact": v_exact,
            "value_loose": v_loose, "value_scored": v_scored,
            "action_correct": (op == step.op and sel_bid == step.gold_bid
                               and (not v_scored or v_exact)),
            "latency_ms": ms, "n_candidates": len(lines),
            "n_candidates_total": step.n_candidates_total,
            "selected_bid": sel_bid, "parsed": tc is not None,
        }
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [{arm}] {k+1}/{len(steps)}", flush=True)


@torch.no_grad()
def run_opfix(loaded, steps, rpath: Path, done: set, split: str, base_records: list):
    """Op repair for the D2E2 configuration: re-read the operation with D2E2's
    chosen element in view (records the element result unchanged, so it is a
    pure op-stage ablation — the cheapest fix for op-first's SELECT bias)."""
    by_id = {r["id"]: r for r in base_records}
    for k, step in enumerate(steps):
        if step.id in done or step.id not in by_id:
            continue
        base = by_id[step.id]
        lines = candidate_lines(step)
        sel_idx = next((i for i, c in enumerate(step.candidates)
                        if c["bid"] == base["selected_bid"]), None)
        if sel_idx is None:
            continue
        op_pred = _op_elem_read(loaded, step, lines[sel_idx])
        rec = {**base, "arm": "OPFIX_web",
               "op_pred": op_pred, "op_correct": op_pred == step.op}
        _append(rpath, rec); done.add(step.id)
        if (k + 1) % 25 == 0:
            print(f"  [OPFIX_web] {k+1}/{len(steps)}", flush=True)


def fmt_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def report(by_arm: dict, path: Path, split: str = "test"):
    def rows_for(rows):
        return [r for r in rows if r.get("split", "test") == split or split == "all"]

    lines = [f"# Web Replay — Gate B (element selection on replayed Mind2Web steps, split={split})", ""]
    arms = {arm: rows_for(rows) for arm, rows in by_arm.items()}
    lines += [fmt_row(["arm", "n", "elem_acc", "op_acc", "value_acc", "action_acc",
                       "parse_fail", "p50 s", "escalation"]),
              fmt_row(["---"] * 9)]
    for arm, rows in arms.items():
        n = len(rows)
        if not n:
            continue
        acc = sum(r["correct"] for r in rows) / n
        opa = sum(r["op_correct"] for r in rows) / n
        vs = [r for r in rows if r.get("value_scored")]
        va = f"{sum(r['value_exact'] for r in vs) / len(vs):.3f}" if vs else "—"
        has_act = any("action_correct" in r for r in rows)
        act = f"{sum(r['action_correct'] for r in rows) / n:.3f}" if has_act else "—"
        pf = sum(1 for r in rows if r.get("parsed") is False)
        p50 = sorted(r["latency_ms"] for r in rows)[n // 2] / 1000
        esc = (sum(r["escalated"] for r in rows) / n) if rows and "escalated" in rows[0] else 0.0
        lines.append(fmt_row([arm, n, f"{acc:.3f}", f"{opa:.3f}", va, act,
                              pf, f"{p50:.2f}", f"{esc:.0%}"]))
    lines += ["", "## Per-operation element accuracy", "",
              fmt_row(["arm"] + list(OPS)), fmt_row(["---"] * (len(OPS) + 1))]
    for arm, rows in arms.items():
        row = [arm]
        for op in OPS:
            rs = [r for r in rows if r["op"] == op]
            row.append(f"{sum(r['correct'] for r in rs) / len(rs):.3f}" if rs else "—")
        lines.append(fmt_row(row))
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["D2E", "C_web", "H_web", "G_web"])
    ap.add_argument("--split", default="test", choices=["all", "cal", "test"])
    ap.add_argument("--n", type=int, default=0, help="cap items (smoke)")
    ap.add_argument("--g-n", type=int, default=200, help="G_web subsample")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report-split", default="test")
    args = ap.parse_args()

    torch.manual_seed(0)
    steps_all = load_web_steps(DATASET)
    steps = (steps_all[::2] if args.split == "cal"
             else steps_all[1::2] if args.split == "test" else steps_all)
    if args.n:
        steps = steps[: args.n]
    print(f"steps: {len(steps)} ({args.split})", flush=True)

    rpath = REPORTS / "webreplay_records.jsonl"
    by_arm: dict[str, list] = {}
    done: dict[str, set] = {}
    if rpath.exists():
        for line in rpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_arm.setdefault(r["arm"], []).append(r)
                done.setdefault(r["arm"], set()).add(r["id"])
        for arm in args.arms:  # report-only also needs full arm lists
            by_arm.setdefault(arm, [])
            done.setdefault(arm, set())
    if args.report_only:
        report(by_arm, REPORTS / "webreplay.md", args.report_split)
        return 0

    loaded = load_model()
    cfg = RuntimeConfig.from_yaml(str(ROOT / "configs/thresholds.yaml"))
    gate_path = ROOT / "configs/webgate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None

    if "D2E2" in args.arms:
        run_d2e2(loaded, steps, rpath, done.setdefault("D2E2", set()), args.split)
    if "C_web" in args.arms:
        run_vanilla(loaded, steps, rpath, done.setdefault("C_web", set()),
                    arm="C_web", think=False, max_new=768)
    if "H_web" in args.arms:
        run_hweb(loaded, steps, cfg, rpath, done.setdefault("H_web", set()))
    if "G_web" in args.arms:
        run_vanilla(loaded, steps[: args.g_n], rpath, done.setdefault("G_web", set()),
                    arm="G_web", think=True, max_new=768)
    if "H2_web" in args.arms:
        if gate is None:
            print("H2_web requires configs/webgate.json (run eval.webgate first)", flush=True)
            return 1
        run_h2web(loaded, steps, rpath, done.setdefault("H2_web", set()), args.split, gate)
    if "H3_web" in args.arms:
        if gate is None:
            print("H3_web requires configs/webgate.json (run eval.webgate first)", flush=True)
            return 1
        run_h3web(loaded, steps, rpath, done.setdefault("H3_web", set()), args.split, gate)
    if "H3B_web" in args.arms:
        if gate is None:
            print("H3B_web requires configs/webgate.json (run eval.webgate first)", flush=True)
            return 1
        run_h3web(loaded, steps, rpath, done.setdefault("H3B_web", set()), args.split, gate,
                  arm="H3B_web", op_options=True)
    if "C3_web" in args.arms:
        run_c3web(loaded, steps, rpath, done.setdefault("C3_web", set()), args.split,
                  arm="C3_web", think=False, max_new=96)
    if "G3_web" in args.arms:
        run_c3web(loaded, steps[: args.g_n], rpath, done.setdefault("G3_web", set()), args.split,
                  arm="G3_web", think=True, max_new=768)
    if "OPFIX_web" in args.arms:
        base = [json.loads(l) for l in rpath.read_text().splitlines()
                if l.strip() and json.loads(l)["arm"] == "D2E2"]
        run_opfix(loaded, steps, rpath, done.setdefault("OPFIX_web", set()), args.split, base)

    for arm in set(args.arms) | set(by_arm):  # reload full arm histories for the report
        by_arm[arm] = [json.loads(l) for l in rpath.read_text().splitlines()
                       if l.strip() and json.loads(l)["arm"] == arm]
    report(by_arm, REPORTS / "webreplay.md", args.report_split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
