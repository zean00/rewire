"""S-experiment: can the decision-read mechanism itself pick the OUTPUT MODE?

Motivated by the question "why is mode selection not decided automatically by
the decision-read runtime?" — this arm measures a d2_over_modes router: the
same D2 sequence-scoring read used for op/element/value, pointed at three
output-contract candidates, and compared against deployable alternatives on
 BOTH mode accuracy and simulated action accuracy.

Preregistered (before any run):
- Gold mode (deterministic from the gold action, not fitted):
    DECIDE     iff gold op == CLICK   (a pure choice suffices; execute as click)
    TOOL_CALL  iff gold op in {TYPE, SELECT}  (a value must be produced)
    TEXT       — no replay step has a free-form gold; TEXT is a distractor
                 candidate only (its wins are scored as automatic failures).
- Misroute cost asymmetry (the point under test): DECIDE-routed on a
  gold-TYPE/SELECT step is a GUARANTEED action loss (a click cannot express
  "type"); TOOL_CALL-routed on a gold-CLICK step costs only latency (the chain
  handles clicks fine); TEXT-routed anywhere is a guaranteed loss.
- Predictions:
    P1  full-context mode read (elements in view, H3B op-re-read rendering)
        lands near the op-read's accuracy (~0.84); the bare variant (task +
        history only) is worse and systematically biased — the op-first read
        without elements predicts SELECT for 61% of gold-CLICK steps.
    P2  action-level, d2_over_modes <= always-TOOL_CALL (affordance routing):
        its wins would be latency, its misroutes are guaranteed losses.
    P3  form_present (any input/textarea/select candidate — deployable DOM
        signal) routes its DECIDE branch only on form-free pages, so it never
        loses action accuracy vs always-TOOL_CALL.
    P4  TEXT essentially never wins on agentic steps.

Execution splice (no new inference): TOOL_CALL-routed steps take the measured
H3B_web per-item action_correct (the chain IS the runtime's TOOL_CALL mode);
DECIDE-routed steps take the D2E per-item unconditioned element decide
(D2E's element read is exactly router DECIDE mode), executed as a click — a
success iff gold op is CLICK and the element pick was right.

Usage:
  python eval/mode_probe.py --split test            # run reads + analyze
  python eval/mode_probe.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from hybrid.actions import candidate_lines, context_text, load_web_steps
from hybrid.decide import decide
from hybrid.engine import load_model

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"
DATASET = ROOT / "eval/datasets/webreplay_v1/webreplay_v1.jsonl"
OUTDIR = REPORTS / "mode_probe"
RPATH = OUTDIR / "mode_probe_records.jsonl"

# The three output contracts, rendered like the op candidates (OP_DESC style).
# Scored as the D2 candidate set; the prompt's own Options block carries the
# page elements so the read sees everything affordance routing sees.
MODE_LINES = [
    "DECIDE — answer by choosing one of the listed page elements (a pure selection, no value needed)",
    "TOOL_CALL — answer with a structured action: operation, target element, and value",
    "TEXT — answer in free-form natural language (no element, no action)",
]
MODES = ("DECIDE", "TOOL_CALL", "TEXT")
FORM_TAGS = {"input", "textarea", "select"}


def gold_mode(op: str) -> str:
    return "DECIDE" if op == "CLICK" else "TOOL_CALL"


def form_present(step) -> bool:
    return any((c.get("tag") or "") in FORM_TAGS for c in step.candidates)


@torch.no_grad()
def run_reads(steps, rpath: Path, done: set):
    for k, step in enumerate(steps):
        for variant, ctx in (
            ("full", context_text(step)
             + "\n\nOptions:\n" + "\n".join(candidate_lines(step))
             + "\n\nWhich output contract should this turn use?"),
            ("bare", context_text(step)
             + "\n\nWhich output contract should this turn use?"),
        ):
            key = (variant, step.id)
            if key in done:
                continue
            res = decide(loaded_global[0], ctx, MODE_LINES, method="d2",
                         alpha=1.0, anchor=True, batched=True)
            probs = {m: float(res.probabilities.get(line, 0.0))
                     for m, line in zip(MODES, MODE_LINES)}
            pred = MODES[MODE_LINES.index(res.selected)] if res.selected in MODE_LINES else None
            top2 = sorted(probs.values(), reverse=True)
            rec = {
                "arm": "MODE", "variant": variant, "id": step.id, "op": step.op,
                "gold_mode": gold_mode(step.op), "form_present": form_present(step),
                "mode_pred": pred, "probs": probs,
                "margin": top2[0] - top2[1] if len(top2) >= 2 else None,
                "confidence": res.confidence, "premask_mass": res.premask_mass,
                "latency_ms": res.total_ms,
            }
            with open(rpath, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(key)
        if (k + 1) % 25 == 0:
            print(f"  [MODE] {k+1}/{len(steps)}", flush=True)


loaded_global = []


def load_records():
    recs = [json.loads(l) for l in RPATH.read_text().splitlines() if l.strip()]
    base = [json.loads(l) for l in (REPORTS / "webreplay_records.jsonl").read_text().splitlines() if l.strip()]
    h3b = {r["id"]: r for r in base if r["arm"] == "H3B_web" and r.get("split") == "test"}
    d2e = {r["id"]: r for r in base if r["arm"] == "D2E"}
    return recs, h3b, d2e


def p50(xs):
    return statistics.median(xs) if xs else float("nan")


def analyze(n_steps: int):
    recs, h3b, d2e = load_records()
    test_ids = set(h3b)
    full = {r["id"]: r for r in recs if r["variant"] == "full"}
    bare = {r["id"]: r for r in recs if r["variant"] == "bare"}
    cov_f = len(test_ids & set(full)) / max(len(test_ids), 1)
    cov_b = len(test_ids & set(bare)) / max(len(test_ids), 1)
    cov_d = len(test_ids & set(d2e)) / max(len(test_ids), 1)
    print(f"coverage: full {cov_f:.3f}  bare {cov_b:.3f}  D2E {cov_d:.3f}  (n_test={len(test_ids)})")

    # ---- gold-mode distribution -------------------------------------------------
    golds = [r["gold_mode"] for r in full.values()]
    n_dec = sum(g == "DECIDE" for g in golds)
    n_tool = sum(g == "TOOL_CALL" for g in golds)
    print(f"gold modes: DECIDE {n_dec} ({n_dec/len(golds):.3f})  TOOL_CALL {n_tool} ({n_tool/len(golds):.3f})  TEXT 0")

    # ---- mode accuracy + confusion for both variants ----------------------------
    def mode_stats(by_id, name):
        rows = [by_id[i] for i in test_ids if i in by_id]
        acc = sum(r["mode_pred"] == r["gold_mode"] for r in rows) / len(rows)
        conf = {(g, p): 0 for g in MODES[:2] for p in MODES}
        for r in rows:
            conf[(r["gold_mode"], r["mode_pred"])] += 1
        text_wins = sum(r["mode_pred"] == "TEXT" for r in rows)
        lat = p50([r["latency_ms"] for r in rows])
        pm = sum(r["premask_mass"] >= 0.35 for r in rows) / len(rows)
        print(f"\n[{name}] mode acc {acc:.3f}  p50 {lat:.0f} ms  premask_ok {pm:.3f}  TEXT wins {text_wins}")
        for g in MODES[:2]:
            row = {p: conf[(g, p)] for p in MODES}
            print(f"  gold {g:9s} -> DECIDE {row['DECIDE']:4d}  TOOL_CALL {row['TOOL_CALL']:4d}  TEXT {row['TEXT']:4d}"
                  f"   (miss {row['DECIDE']+row['TOOL_CALL']+row['TEXT']-conf[(g,g)]:4d})")
        return acc, rows

    acc_f, rows_f = mode_stats(full, "d2_over_modes · full context (elements in view)")
    acc_b, _ = mode_stats(bare, "d2_over_modes · bare context (task+history only)")

    # agreement between the two variants and with the op-first read
    both = [i for i in test_ids if i in full and i in bare]
    agree = sum(full[i]["mode_pred"] == bare[i]["mode_pred"] for i in both) / len(both)
    opf = [h3b[i].get("op_first_pred") for i in test_ids]
    opf_mode_acc = sum((o == "CLICK") == (h3b[i]["op"] == "CLICK")
                       for i, o in zip(test_ids, opf)) / len(test_ids)
    print(f"\nfull vs bare agreement: {agree:.3f}")
    print(f"mode-from-op-first-read accuracy: {opf_mode_acc:.3f} (zero extra cost — the chain already runs this read)")

    # ---- deployable routers: simulated action accuracy ---------------------------
    # execution: TOOL_CALL -> H3B chain action_correct; DECIDE -> D2E element
    # decide executed as a click (success iff gold CLICK and element right);
    # TEXT -> automatic failure.
    def simulate(router, name):
        n_tc = n_dc = n_tx = succ_tc = succ_dc = 0
        for i in test_ids:
            m = router(i)
            if m == "TOOL_CALL":
                n_tc += 1
                succ_tc += bool(h3b[i]["action_correct"])
            elif m == "DECIDE":
                n_dc += 1
                r = d2e.get(i)
                ok = bool(r and r["correct"]) and h3b[i]["op"] == "CLICK"
                succ_dc += ok
            else:
                n_tx += 1
        acc = (succ_tc + succ_dc) / len(test_ids)
        print(f"  {name:34s} mode-branch sizes TC/DC/TX {n_tc:4d}/{n_dc:4d}/{n_tx:3d}  "
              f"action acc {acc:.3f}  (chain succ {succ_tc}/{n_tc}, decide succ {succ_dc}/{n_dc})")
        return acc

    print("\nsimulated action accuracy by router (test n=%d):" % len(test_ids))
    a_status = simulate(lambda i: "TOOL_CALL", "always TOOL_CALL (status quo)")
    a_full = simulate(lambda i: full[i]["mode_pred"] if i in full else "TOOL_CALL",
                      "d2_over_modes · full context")
    a_bare = simulate(lambda i: bare[i]["mode_pred"] if i in bare else "TOOL_CALL",
                      "d2_over_modes · bare context")
    a_opf = simulate(lambda i: "DECIDE" if h3b[i].get("op_first_pred") == "CLICK" else "TOOL_CALL",
                     "mode from op-first read (derived)")
    a_form = simulate(lambda i: "DECIDE" if not form_present_gold(i) else "TOOL_CALL",
                      "form_present affordance (DOM)")
    a_gold = simulate(lambda i: gold_mode(h3b[i]["op"]),
                      "perfect mode router (upper bound)")

    print(f"\ndeltas vs status quo: full {a_full-a_status:+.3f}  bare {a_bare-a_status:+.3f}  "
          f"op-derived {a_opf-a_status:+.3f}  form_present {a_form-a_status:+.3f}  perfect {a_gold-a_status:+.3f}")

    # where the full-context read's misses concentrate
    errs = [r for r in rows_f if r["mode_pred"] != r["gold_mode"]]
    by_op = {}
    for r in errs:
        by_op[r["op"]] = by_op.get(r["op"], 0) + 1
    fp_err = sum(r["form_present"] for r in errs)
    print(f"full-context errors: {len(errs)}  by gold op {by_op}  on form_present pages {fp_err}")
    cm = [sorted(r["probs"].values(), reverse=True)[0] - sorted(r["probs"].values(), reverse=True)[1]
          for r in rows_f if r["mode_pred"] != r["gold_mode"]]
    print(f"error margins: p50 {p50(cm):.3f}  (confident wrong routes vs near-ties)")


def form_present_gold(i):
    # form_present needs the step, not the record — rebuilt from the dataset
    return FORM_PRESENT_BY_ID[i]


FORM_PRESENT_BY_ID = {}


def main() -> int:
    global loaded_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "cal", "all"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    steps_all = load_web_steps(DATASET)
    # form_present is a deployable DOM signal — index it for every step so
    # --analyze-only and smoke slices can still join against full test ids.
    for s in steps_all:
        FORM_PRESENT_BY_ID[s.id] = form_present(s)
    steps = (steps_all[::2] if args.split == "cal"
             else steps_all[1::2] if args.split == "test" else steps_all)
    if args.n:
        steps = steps[: args.n]
    print(f"steps: {len(steps)} ({args.split})", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(len(steps))
        return 0

    done = set()
    if RPATH.exists():
        for line in RPATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["id"]))
    print(f"steps: {len(steps)} ({args.split}); already done: {len(done)//2}", flush=True)

    if len(done) // 2 < len(steps):
        loaded_global.append(load_model())
        run_reads(steps, RPATH, done)
        del loaded_global[:]

    analyze(len(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
