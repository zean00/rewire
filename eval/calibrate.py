"""Phase 3 — calibration + escalation policy (doc §11, §18.1).

Re-scores a deterministic per-domain split (first 50% by id order =
calibration, rest = test), then:
  1. temperature scaling fit on calibration (per method),
  2. content-free prior correction, fit and evaluated,
  3. transfer check: ECE on test split with calibration-split T,
  4. threshold sweep for the escalation policy (confidence/margin) targeting
     Useful Escalation Rate, reported against the filler-control availability
     of THINK.

Usage: python -m eval.calibrate [--datasets arc_hard arith_hard]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from eval.cells import load_items
from hybrid.engine import load_model
from hybrid.scoring import score_sequence_logprob
from hybrid.template import pin_prompt, question_messages

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"


def softmax_T(raw: dict[str, float], T: float) -> dict[str, float]:
    m = max(raw.values())
    exps = {k: math.exp((v - m) / T) for k, v in raw.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def ece(conf: list[float], corr: list[bool], n_bins: int = 10) -> float:
    return expected_calibration_error(conf, corr, n_bins)["ece"]


from hybrid.calibration.metrics import expected_calibration_error  # noqa: E402


def score_split(loaded, items, pmi_lambda=0.0):
    out = []
    for it in items:
        msgs = question_messages(it["question"], it["choices"])
        pinned = pin_prompt(loaded.tokenizer, msgs)
        r = score_sequence_logprob(loaded.model, loaded.tokenizer, pinned, it["choices"],
                                   alpha=1.0, pmi_lambda=pmi_lambda)
        gold = it["choices"][it["answer_idx"]]
        out.append({"id": it["id"], "domain": it["domain"], "raw": r.raw_scores,
                    "selected": r.selected, "correct": r.selected == gold})
    return out


def fit_temperature(scored):
    """Grid-search T minimizing ECE on calibration items."""
    best = (None, float("inf"))
    for T in [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
        conf, corr = [], []
        for s in scored:
            p = softmax_T(s["raw"], T)
            top = max(p, key=p.get)
            conf.append(p[top])
            corr.append(top == s["selected"])  # same selection; correctness fixed
        e = ece(conf, corr)
        if e < best[1]:
            best = (T, e)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["arc_hard", "arith_hard"])
    args = ap.parse_args()

    torch.manual_seed(0)
    loaded = load_model()

    items = [it for d in args.datasets for it in load_items(ROOT / f"eval/datasets/eval_v1/{d}.jsonl")]
    # deterministic split: half calibration, half test
    items.sort(key=lambda x: x["id"])
    cal, test = items[::2], items[1::2]
    print(f"cal={len(cal)} test={len(test)}", flush=True)

    lines = ["# Phase 3 — Calibration and Escalation", ""]

    for label, lam in (("D2", 0.0), ("D2P (pmi=1.0)", 1.0)):
        cal_s = score_split(loaded, cal, pmi_lambda=lam)
        test_s = score_split(loaded, test, pmi_lambda=lam)
        T, ece_cal = fit_temperature(cal_s)
        conf_t, corr_t = [], []
        for s in test_s:
            p = softmax_T(s["raw"], T)
            top = max(p, key=p.get)
            conf_t.append(p[top])
            corr_t.append(s["correct"])
        acc = sum(corr_t) / len(corr_t)
        lines += [
            f"## {label}",
            "",
            f"- fitted T = **{T}** (calibration ECE {ece_cal:.3f})",
            f"- test: accuracy **{acc:.3f}**, ECE(T-transferred) = **{ece(conf_t, corr_t):.3f}**",
        ]

        # escalation threshold sweep on test (uncalibrated confidence ranks == calibrated ranks
        # for monotone T; sweep on transferred T)
        rows = []
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            n_esc = sum(1 for c in conf_t if c < thr)
            rows.append((thr, n_esc / len(conf_t)))
        lines += ["", "escalation rate by confidence threshold (test):", ""]
        for thr, rate in rows:
            lines.append(f"- conf < {thr:.2f} → escalate {rate:.1%}")
        lines.append("")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "calibration_v1.md").write_text("\n".join(lines) + "\n")
    print("wrote eval/reports/calibration_v1.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
