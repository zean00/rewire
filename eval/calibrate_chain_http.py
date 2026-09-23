#!/usr/bin/env python3
"""Fit the web escalation gate for the chain-on-llama.cpp port.

Mirror of eval/webgate.py (the documented fitter that produced
configs/webgate.json for the 4B) with one change: the D2E2 reads are taken
over llama-server HTTP via eval.chain_http, so the fitted constants
describe the backend that will run them at serving time. The fitter rules
are unchanged:

  temperature  grid over T minimizing ECE of tempered confidences
  accept_conf  smallest threshold whose accepted-subset accuracy is at
               least max(0.45, 1.15 x cal overall), coverage >= 15%
               (falls back to the 70th-percentile quantile otherwise)
  answer_gate  accept_conf - 0.20, floored at 0.30 (the 4B's fitted pair
               was 0.70 / 0.50 — same 0.20 gap, same role: the ANSWER-
               forced path demands a solid target but less than the
               action path)

webreplay_v1 CAL SPLIT ONLY (even lines; the odd-line test split is never
read — by construction). Items are bounded to <= 30 candidates for cost;
the sample, seed and skip counts are printed for audit.

Usage (on the GPU host, against the serving llama-server):
  python3 eval/calibrate_chain_http.py --base-url http://127.0.0.1:8998 \
      --model gemma-4-12b-it [--n 120] [--workers 4]
Writes configs/webgate_remote.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid.actions import OPS, WebStep, candidate_lines, context_text  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
import chain_http  # noqa: E402

OP_LINES_ELEM = [
    f"{op} — {'press/activate' if op == 'CLICK' else 'enter a value into' if op == 'TYPE' else 'choose an option from'} the selected element"
    for op in OPS
]


def auc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n) \
        + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def ece(confidences, corrects, n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    total = len(confidences)
    return sum(len(b) / total * abs(sum(o for _, o in b) / len(b) - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def cal_steps(path, max_cand: int = 30):
    """webreplay_v1 cal split ONLY: even lines, as WebSteps."""
    steps, n_test = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % 2:  # odd lines are the test split — never read
                n_test += 1
                continue
            d = json.loads(line)
            if len(d.get("candidates") or []) > max_cand:
                continue
            steps.append(WebStep(
                id=d["id"], task=d["task"], history=d["history"],
                op=d["op"], value=d.get("value"), candidates=d["candidates"],
                gold_bid=d["gold_bid"],
                n_candidates_total=d.get("n_candidates_total", 0)))
    return steps, n_test


def gold_idx(step: WebStep, lines: list[str]):
    for i, c in enumerate(step.candidates):
        if c["bid"] == step.gold_bid:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cal-set",
                    default=str(ROOT / "eval/datasets/webreplay_v1/webreplay_v1.jsonl"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(ROOT / "configs/webgate_remote.json"))
    args = ap.parse_args()

    steps, n_test_skipped = cal_steps(args.cal_set)
    rng = random.Random(args.seed)
    rng.shuffle(steps)
    steps = steps[: args.n]
    print(f"cal steps: {len(steps)} (test-split lines never opened: {n_test_skipped})",
          flush=True)

    recs, skipped = [], {"dup": 0, "tiny": 0, "fail": 0, "nogold": 0}
    t_start = time.time()
    for k, step in enumerate(steps):
        lines = candidate_lines(step)
        if len(set(lines)) != len(lines):
            skipped["dup"] += 1
            continue
        if len(lines) < 2:
            skipped["tiny"] += 1
            continue
        # op-first factorization (D2E2): 3-way op read, then op-conditioned
        # element read, both d2 anchored — the reads the runtime gates on
        try:
            op_res = chain_http.decide_http(
                args.base_url, args.model, context_text(step), OP_LINES_ELEM,
                alpha=1.0, anchor=True, timeout=args.timeout,
                workers=args.workers)
            if op_res is None:
                skipped["fail"] += 1
                continue
            op = op_res.selected.split(" — ")[0]
            res = chain_http.decide_http(
                args.base_url, args.model, context_text(step, op), lines,
                alpha=0.5, anchor=True, timeout=args.timeout,
                workers=args.workers)
        except Exception as e:
            print(f"  [{k}] read error {e!r}", flush=True)
            skipped["fail"] += 1
            continue
        if res is None:
            skipped["fail"] += 1
            continue
        gi = gold_idx(step, lines)
        if gi is None:
            skipped["nogold"] += 1
            continue
        recs.append({
            "id": step.id, "op": step.op, "op_pred": op,
            "op_correct": op == step.op,
            "correct": res.selected == lines[gi],
            "confidence": res.confidence, "margin": res.margin,
            "premask_mass": res.premask_mass, "n_cand": len(lines),
            "split": "cal", "arm": "D2E2-http"})
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(steps)} steps | {len(recs)} records "
                  f"| {time.time() - t_start:.0f}s", flush=True)

    if len(recs) < 40:
        print(f"need >= 40 records, have {len(recs)}; skipped={skipped}")
        return 1
    labels = [r["correct"] for r in recs]
    overall = sum(labels) / len(labels)
    op_acc = sum(r["op_correct"] for r in recs) / len(recs)
    print(f"\nrecords n={len(recs)} (skipped={skipped})")
    print(f"overall elem_acc {overall:.3f} | op_acc {op_acc:.3f}")
    print(f"AUC confidence: {auc([r['confidence'] for r in recs], labels):.3f}")
    print(f"AUC margin:     {auc([r['margin'] for r in recs], labels):.3f}")
    print(f"AUC premask:    {auc([r['premask_mass'] for r in recs], labels):.3f}")

    temperature = min((t for t in (0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)),
                      key=lambda t: ece([r["confidence"] ** (1.0 / t)
                                         for r in recs], labels))
    print(f"fitted temperature: {temperature}")

    thresholds = sorted({round(r["confidence"], 3) for r in recs})
    best = None
    curve = []
    for th in thresholds:
        acc_sub = [r["correct"] for r in recs if r["confidence"] >= th]
        cov = len(acc_sub) / len(recs)
        acc = sum(acc_sub) / len(acc_sub) if acc_sub else 0.0
        curve.append((th, cov, acc))
        if cov >= 0.15 and acc >= max(0.45, 1.15 * overall) and best is None:
            best = th
    if best is None:
        qs = sorted(r["confidence"] for r in recs)
        best = qs[int(len(qs) * 0.70)]
        print("no threshold reached the accuracy bar; "
              "falling back to 30%-coverage quantile")
    print(f"accept_conf: {best:.3f}")
    answer_gate = max(0.30, round(best - 0.20, 3))
    print("coverage/accuracy curve (theta, coverage, accepted_acc):")
    for th, cov, acc in curve[:: max(1, len(curve) // 15)]:
        print(f"  {th:.3f}  {cov:.0%}  {acc:.3f}")

    gate = {"accept_conf": best, "temperature": temperature,
            "answer_gate": answer_gate,
            "fitted_for": {"model": args.model, "base_url": args.base_url,
                           "cal_split_only": True, "n": len(recs),
                           "op_acc": round(op_acc, 3)},
            "diagnostics": {"cal_overall": overall,
                            "auc_conf": auc([r["confidence"] for r in recs], labels),
                            "auc_margin": auc([r["margin"] for r in recs], labels),
                            "auc_premask": auc([r["premask_mass"] for r in recs], labels)},
            "records": str(ROOT / "eval/reports/webreplay_records_http.jsonl")}
    Path(args.out).write_text(json.dumps(gate, indent=1))
    out_rec = ROOT / "eval/reports/webreplay_records_http.jsonl"
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    out_rec.write_text("\n".join(json.dumps(r) for r in recs))
    print(f"wrote {args.out}")
    print(f"wrote {out_rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
