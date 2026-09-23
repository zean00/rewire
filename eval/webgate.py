"""Fit the web escalation gate from cal-split D2E2 records (eval.webgate).

Gate B showed the MCQ gate does not transfer to web steps (conf AUC 0.648,
premask AUC 0.493). This fits web-specific values on the calibration split:

  temperature  grid over T minimizing ECE of tempered confidences
  accept_conf  smallest threshold whose accepted-subset accuracy is at least
               max(0.45, 1.15 x cal overall) subject to coverage >= 15%
               (falls back to the 30%-coverage quantile if never reached)

Also prints the signal comparison (confidence vs margin vs premask AUC) and
the coverage/accuracy curve so the choice is auditable. Writes
configs/webgate.json for eval.webreplay --arms H2_web.

Usage: python -m eval.webgate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"


def auc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def ece(confidences, corrects, n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    total = len(confidences)
    return sum(len(b) / total * abs(sum(o for _, o in b) / len(b) - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def main() -> int:
    recs = [json.loads(l) for l in (REPORTS / "webreplay_records.jsonl").read_text().splitlines()
            if l.strip()]
    cal = [r for r in recs if r.get("arm") == "D2E2" and r.get("split") == "cal"]
    if len(cal) < 100:
        print(f"need cal-split D2E2 records (have {len(cal)}); run eval.webreplay --arms D2E2 --split cal first")
        return 1
    labels = [r["correct"] for r in cal]
    overall = sum(labels) / len(labels)
    print(f"cal n={len(cal)}, overall elem_acc {overall:.3f}")

    print(f"AUC confidence: {auc([r['confidence'] for r in cal], labels):.3f}")
    print(f"AUC margin:     {auc([r['margin'] for r in cal], labels):.3f}")
    print(f"AUC premask:    {auc([r['premask_mass'] for r in cal], labels):.3f}")

    temperature = min((t for t in (0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)),
                      key=lambda t: ece([r["confidence"] ** (1.0 / t) for r in cal], labels))
    print(f"fitted temperature: {temperature}")

    thresholds = sorted({round(r["confidence"], 3) for r in cal})
    best = None
    curve = []
    for th in thresholds:
        acc_sub = [r["correct"] for r in cal if r["confidence"] >= th]
        cov = len(acc_sub) / len(cal)
        acc = sum(acc_sub) / len(acc_sub) if acc_sub else 0.0
        curve.append((th, cov, acc))
        if cov >= 0.15 and acc >= max(0.45, 1.15 * overall) and best is None:
            best = th
    if best is None:
        qs = sorted(r["confidence"] for r in cal)
        best = qs[int(len(qs) * 0.70)]  # 30% coverage fallback
        print(f"no threshold reached the accuracy bar; falling back to 30%-coverage quantile")
    print(f"accept_conf: {best:.3f}")
    print("coverage/accuracy curve (theta, coverage, accepted_acc):")
    for th, cov, acc in curve[:: max(1, len(curve) // 15)]:
        print(f"  {th:.3f}  {cov:.0%}  {acc:.3f}")

    gate = {"accept_conf": best, "temperature": temperature,
            "diagnostics": {"cal_overall": overall,
                            "auc_conf": auc([r["confidence"] for r in cal], labels),
                            "auc_margin": auc([r["margin"] for r in cal], labels),
                            "auc_premask": auc([r["premask_mass"] for r in cal], labels)}}
    out = ROOT / "configs/webgate.json"
    out.write_text(json.dumps(gate, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
