"""Baseline table v1 (IMPLEMENTATION_PLAN.md Step 2, doc §16).

Cells on all five eval_v1 domains (~500 items):
  A   no-options, no-think generation          (baseline A)
  C   options shown, no-think generation       (baseline C)
  G   options shown, native default (think-always)  (baseline G — full n)
  B   options shown, forced think              (subsample — G covers think-always)
  D1  first-token restricted logits
  D2  sequence log-likelihood, alpha=1.0
  D2P D2 + PMI/DC correction (lambda=1.0, content-free "N/A")

Extras (ARC-Easy only): D2 alpha∈{0,0.5} normalization sensitivity,
D2-anchored format anchor, order-shuffle control, premask-mass gate stats.

Writes eval/reports/baseline_table_v1.md and per-item records to
eval/reports/baseline_table_v1_records.jsonl (Phase 3 reuses these — no re-run
needed for calibration/threshold fitting).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from eval.cells import load_items, run_decision_cell, run_generative_cell
from hybrid.calibration.metrics import expected_calibration_error
from hybrid.engine import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval/datasets/eval_v1"
REPORTS = ROOT / "eval/reports"

DOMAINS = ["arc_easy", "arc_hard", "arith_easy", "arith_hard", "intent"]
B_SUBSAMPLE_PER_DOMAIN = 6  # forced-think is the expensive cell


def fmt_row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def summarize(records: list[dict]) -> dict:
    n = len(records)
    lat = sorted(r["latency_ms"] for r in records)
    toks = [r.get("gen_tokens", 0) for r in records]
    conf = [r["confidence"] for r in records if "confidence" in r]
    corr = [r["correct"] for r in records if "confidence" in r]
    out = {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "parsed": sum(r["parsed"] for r in records) / n if n else 0.0,
        "p50_ms": lat[len(lat) // 2] if lat else 0.0,
        "p95_ms": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else 0.0,
        "mean_tokens": statistics.fmean(toks) if toks else 0.0,
    }
    if conf:
        out["ece"] = expected_calibration_error(conf, corr)["ece"]
        if "premask_mass" in records[0]:
            out["premask_p50"] = statistics.median([r["premask_mass"] for r in records])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from checkpointed records; run no cells")
    args = ap.parse_args()

    torch.manual_seed(0)
    items_by_domain = {d: load_items(DATA / f"{d}.jsonl") for d in DOMAINS}
    all_items = [it for d in DOMAINS for it in items_by_domain[d]]
    print(f"total items: {len(all_items)}", flush=True)

    loaded = None if args.report_only else load_model()
    all_records: list[dict] = []

    # Checkpoint/resume: records append per cell; completed cells are skipped
    # on restart. A crash never costs more than the cell that was running.
    RECORDS = REPORTS / "baseline_table_v1_records.jsonl"
    REPORTS.mkdir(parents=True, exist_ok=True)
    done_cells: dict[str, list[dict]] = {}
    if RECORDS.exists():
        for line in RECORDS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("domain") == "arc_challenge":
                    r["domain"] = "arc_hard"  # normalize legacy records
                done_cells.setdefault(r["cell"], []).append(r)
        print(f"resuming; cells already done: {sorted(done_cells)}", flush=True)

    def run(name, fn, *a, **kw):
        if name in done_cells:
            recs = done_cells[name]
            all_records.extend(recs)
            print(f"cell {name}: SKIPPED (checkpoint, n={len(recs)})", flush=True)
            return recs
        if args.report_only:
            raise SystemExit(f"--report-only but cell {name} has no checkpointed records")
        print(f"cell {name} ...", flush=True)
        recs = fn(*a, **kw)
        with open(RECORDS, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        all_records.extend(recs)
        s = summarize(recs)
        print(f"cell {name}: acc={s['accuracy']:.3f} p50={s['p50_ms']:.0f}ms", flush=True)
        return recs

    # Decision cells first (cheap, and the new science); generative cells after.
    run("D1", run_decision_cell, loaded, all_items, "D1", method="d1")
    run("D2", run_decision_cell, loaded, all_items, "D2", method="d2", alpha=1.0)
    run("D2P", run_decision_cell, loaded, all_items, "D2P", method="d2", alpha=1.0, pmi_lambda=1.0)

    # Generative baselines
    run("A", run_generative_cell, loaded, all_items, "A",
        show_options=False, enable_thinking=False, max_new_tokens=200)
    run("C", run_generative_cell, loaded, all_items, "C",
        show_options=True, enable_thinking=False, max_new_tokens=200)
    run("G", run_generative_cell, loaded, all_items, "G",
        show_options=True, enable_thinking=None, max_new_tokens=512)
    b_items = [it for d in DOMAINS for it in items_by_domain[d][:B_SUBSAMPLE_PER_DOMAIN]]
    run("B", run_generative_cell, loaded, b_items, "B",
        show_options=True, enable_thinking=True, max_new_tokens=768)

    # --- ARC-Easy extras ----------------------------------------------------
    arc = items_by_domain["arc_easy"]
    d2_arc = [r for r in all_records if r["cell"] == "D2" and r["domain"] == "arc_easy"]
    alpha_rows = []
    for a in (0.0, 0.5, 1.0):
        recs = run(f"D2_alpha{a}", run_decision_cell, loaded, arc, f"D2_alpha{a}", method="d2", alpha=a)
        alpha_rows.append((a, summarize(recs)["accuracy"]))
    anchored = run("D2_anchored", run_decision_cell, loaded, arc, "D2_anchored",
                   method="d2", alpha=1.0, anchor=True)
    reshuffled = run("D2_reshuffle", run_decision_cell, loaded, arc, "D2_reshuffle",
                     method="d2", alpha=1.0, choice_seed=99)

    # order-shuffle stability: per-item selected text in D2 vs D2_reshuffle,
    # over the intersection of items (reshuffle ran on ARC-Easy only)
    sel_main = {r["id"]: r["selected"] for r in all_records if r["cell"] == "D2"}
    sel_re = {r["id"]: r["selected"] for r in reshuffled}
    common = [i for i in sel_re if i in sel_main]
    stable = sum(1 for i in common if sel_main[i] == sel_re[i])

    # per-domain accuracy for headline cells
    lines = ["# Baseline Table v1", "",
             "Domains: arc_easy (100), arc_hard (100), arith_easy (100), arith_hard (100), intent (100). "
             "Generative cells greedy; D cells pinned no-think read point; PMI lambda=1.0 content-free 'N/A'.", ""]
    lines += ["## Main table", "",
              fmt_row(["cell", "n", "accuracy", "p50 ms", "p95 ms", "mean tokens", "ECE", "premask p50"]),
              fmt_row(["---"] * 8)]
    order = ["A", "C", "G", "B", "D1", "D2", "D2P"]
    by_cell: dict[str, list[dict]] = {}
    for r in all_records:
        by_cell.setdefault(r["cell"], []).append(r)
    for cell in order:
        s = summarize(by_cell[cell])
        lines.append(fmt_row([cell, s["n"], f"{s['accuracy']:.3f}", f"{s['p50_ms']:.0f}",
                              f"{s['p95_ms']:.0f}", f"{s['mean_tokens']:.0f}",
                              f"{s.get('ece', float('nan')):.3f}" if "ece" in s else "—",
                              f"{s.get('premask_p50', float('nan')):.4f}" if s.get("premask_p50") is not None else "—"]))

    lines += ["", "## Per-domain accuracy", "",
              fmt_row(["cell"] + DOMAINS), fmt_row(["---"] * (len(DOMAINS) + 1))]
    for cell in order:
        accs = []
        for d in DOMAINS:
            rs = [r for r in by_cell[cell] if r["domain"] == d]
            accs.append(f"{sum(r['correct'] for r in rs) / len(rs):.3f}" if rs else "—")
        lines.append(fmt_row([cell] + accs))

    lines += ["", "## ARC-Easy extras", "",
              fmt_row(["experiment", "result"]), fmt_row(["---", "---"]),
              fmt_row(["order-shuffle selection stability (D2 vs reshuffled, ARC-Easy)", f"{stable}/{len(common)}"]),
              fmt_row(["premask p50: D2(arc_easy) vs D2-anchored(arc_easy)",
                       f"{summarize(d2_arc)['premask_p50']:.4f} vs {summarize(anchored)['premask_p50']:.4f}"]),
              fmt_row(["accuracy: D2(arc_easy) vs D2-anchored(arc_easy)",
                       f"{summarize(d2_arc)['accuracy']:.3f} vs {summarize(anchored)['accuracy']:.3f}"])]
    lines += ["", "### D2 length-normalization sensitivity (ARC-Easy)", "",
              fmt_row(["alpha", "accuracy"]), fmt_row(["---", "---"])]
    for a, acc in alpha_rows:
        lines.append(fmt_row([a, f"{acc:.3f}"]))
    lines += ["", f"B cell note: forced-think runs on a {len(b_items)}-item subsample "
                  f"({B_SUBSAMPLE_PER_DOMAIN}/domain); G covers think-always at full n (Phase 0 measured G≡B behaviorally).", ""]

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "baseline_table_v1.md").write_text("\n".join(lines) + "\n")
    with open(REPORTS / "baseline_table_v1_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print("wrote reports", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
