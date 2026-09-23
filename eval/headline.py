"""Headline comparison (S8 / Q12): hybrid escalation vs never-think (A) vs
think-always (G) on accuracy-per-latency across the difficulty spectrum.

A and G arms are REUSED from the Phase-1 baseline table records (identical
configs measured on the same machine — re-running them would cost ~2.5 h for
no new information). This script runs only the hybrid arms:

  H-redecide  DECIDE -> escalate(conf<accept) -> THINK -> re-DECIDE  (doc Exp E;
              prior: filler control found no net benefit — preregistered arm)
  H-generate  DECIDE -> escalate -> THINK-generation answers  (post-filler
              pivot: confidence-gated mode selection)

Usage: python -m eval.headline
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from eval.cells import load_items
from hybrid.engine import load_model
from hybrid.runtime import RuntimeConfig, hybrid_decide_turn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"
ARMS = ["H_redecide", "H_generate"]


def fmt_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def load_checkpoint() -> tuple[dict, dict]:
    """Returns (hybrid_records_by_arm, baseline_records_by_cell)."""
    hybrid: dict[str, list[dict]] = {}
    path = REPORTS / "headline_records.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                hybrid.setdefault(r["arm"], []).append(r)

    base: dict[str, dict[str, dict]] = {}
    bpath = REPORTS / "baseline_table_v1_records.jsonl"
    for line in bpath.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            base.setdefault(r["cell"], {})[r["id"]] = r
    return hybrid, base


def main() -> int:
    torch.manual_seed(0)
    cfg = RuntimeConfig.from_yaml(str(ROOT / "configs/thresholds.yaml"))
    datasets = ["arc_easy", "arc_hard", "arith_easy", "arith_hard", "intent"]
    items = [it for d in datasets for it in load_items(ROOT / f"eval/datasets/eval_v1/{d}.jsonl")]
    hybrid, base = load_checkpoint()
    print(f"items: {len(items)}; checkpointed hybrid arms: {sorted(hybrid)}", flush=True)

    loaded = load_model()
    REPORTS.mkdir(parents=True, exist_ok=True)

    # H_generate first: it is the post-filler-pivot mechanism (the science);
    # H_redecide is the preregistered negative-control arm. Checkpointing
    # makes the order irrelevant to correctness.
    for arm, escalation in (("H_generate", "generate"), ("H_redecide", "redecide")):
        if arm in hybrid and len(hybrid[arm]) >= len(items):
            print(f"arm {arm}: complete checkpoint, skipping", flush=True)
            continue
        print(f"arm {arm} ({escalation}) ...", flush=True)
        done_ids = {r["id"] for r in hybrid.get(arm, [])}
        with open(REPORTS / "headline_records.jsonl", "a") as f:
            for k, it in enumerate(items):
                if it["id"] in done_ids:
                    continue
                t0 = time.perf_counter()
                turn = hybrid_decide_turn(loaded, it["question"], it["choices"], cfg,
                                          escalation=escalation)
                ms = (time.perf_counter() - t0) * 1000
                rec = {
                    "arm": arm, "domain": it["domain"], "id": it["id"],
                    "correct": turn.final.selected == it["choices"][it["answer_idx"]],
                    "latency_ms": ms, "escalated": turn.escalated,
                    "confidence": turn.final.confidence,
                    "selected": turn.final.selected,
                    "method": turn.final.scoring_method,
                }
                f.write(json.dumps(rec) + "\n")
                hybrid.setdefault(arm, []).append(rec)
                if (k + 1) % 25 == 0:
                    print(f"  [{arm}] {k + 1}/{len(items)}", flush=True)

    # ---------- report ----------
    def stats(rows):
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        p50 = sorted(r["latency_ms"] for r in rows)[n // 2] / 1000.0
        return acc, p50, acc / p50

    a_rows = list(base["A"].values())
    g_rows = list(base["G"].values())
    c_rows = list(base["C"].values())
    lines = ["# Headline Comparison (S8/Q12)", "",
             f"gate: D2 confidence < {cfg.confidence_accept} (margin < {cfg.margin_min} also escalates; "
             f"premask < {cfg.premask_gate} forces escalation); think budget {cfg.think_budget}/"
             f"{cfg.generate_budget}. A and G reused from baseline_table records (same machine/config).",
             "CAA = accuracy / p50 latency (s).", ""]
    lines += [fmt_row(["arm", "n", "accuracy", "p50 s", "CAA", "escalation rate"]),
              fmt_row(["---"] * 6)]
    for name, rows, esc_key in (("A never-think", a_rows, None), ("C no-think generate", c_rows, None),
                                ("G think-always", g_rows, None),
                                ("H-redecide", hybrid["H_redecide"], "escalated"),
                                ("H-generate", hybrid["H_generate"], "escalated")):
        acc, p50, caa = stats(rows)
        er = (sum(r[esc_key] for r in rows) / len(rows)) if esc_key else 0.0
        lines.append(fmt_row([name, len(rows), f"{acc:.3f}", f"{p50:.2f}", f"{caa:.2f}", f"{er:.0%}"]))

    lines += ["", "## Per-domain accuracy", "",
              fmt_row(["arm"] + datasets), fmt_row(["---"] * (len(datasets) + 1))]
    for name, rows in (("A", a_rows), ("C", c_rows), ("G", g_rows),
                       ("H-redecide", hybrid["H_redecide"]), ("H-generate", hybrid["H_generate"])):
        row = [name]
        for d in datasets:
            rs = [r for r in rows if r["domain"] == d]
            row.append(f"{sum(r['correct'] for r in rs) / len(rs):.3f}" if rs else "—")
        lines.append(fmt_row(row))

    (REPORTS / "headline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
