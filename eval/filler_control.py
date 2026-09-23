"""Phase 2 — neutral-filler THINK control (doc §16.2, S5).

For every item in the hard domains, three arms, all ending in an identical
pinned D2 decision at the same read point:

  base    DECIDE(question, candidates)
  think   THINK on the question (budget B) -> freeze think block -> DECIDE
  filler  DECIDE after inserting a content-free filler paragraph (~matched
          token length) into the context

The claim under test (H4/S5): real reasoning content, not extra context
tokens, causes decision improvement. Verdict: Δacc(think) − Δacc(filler).

Run after baseline_table (GPU-serial):  python -m eval.filler_control
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import torch

from eval.cells import load_items
from hybrid.engine import load_model
from hybrid.scoring import score_sequence_logprob
from hybrid.template import pin_after_think, pin_prompt, question_messages

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"

FILLER = (
    "Weather patterns in temperate coastal regions vary gradually across the "
    "year, with morning fog common near large bodies of water. Local markets "
    "often open early, and public transit schedules shift slightly in summer. "
    "Municipal libraries keep extended hours on weekdays, and community "
    "centers host evening classes in art and language. Foot traffic in "
    "commercial districts peaks around lunchtime, when cafes and bakeries "
    "record their highest sales. Parks see steady use on dry afternoons."
)

DOMAINS = ["arc_hard", "arith_hard"]
THINK_BUDGET = 256


def decide_on(loaded, messages, choices):
    pinned = pin_prompt(loaded.tokenizer, messages)
    return score_sequence_logprob(loaded.model, loaded.tokenizer, pinned, choices, alpha=1.0)


def main() -> int:
    torch.manual_seed(0)
    items = [it for d in DOMAINS for it in load_items(ROOT / f"eval/datasets/eval_v1/{d}.jsonl")]
    print(f"items: {len(items)}", flush=True)
    loaded = load_model()

    filler_tokens = len(loaded.tokenizer.encode(FILLER, add_special_tokens=False))
    rng = random.Random(7)
    records = []

    for k, it in enumerate(items):
        gold = it["choices"][it["answer_idx"]]
        msgs = question_messages(it["question"], it["choices"])
        rec = {"domain": it["domain"], "id": it["id"]}

        r_base = decide_on(loaded, msgs, it["choices"])
        rec["base_correct"] = r_base.selected == gold
        rec["base_conf"] = r_base.confidence

        # think arm: real reasoning about the question, frozen into the prompt
        post = pin_after_think(loaded.tokenizer, loaded.model, msgs, max_think_tokens=THINK_BUDGET)
        r_think = score_sequence_logprob(loaded.model, loaded.tokenizer, post, it["choices"], alpha=1.0)
        rec["think_correct"] = r_think.selected == gold
        rec["think_conf"] = r_think.confidence
        from hybrid.template import apply_template

        base_text, _ = apply_template(loaded.tokenizer, msgs, enable_thinking=True)
        rec["think_text_tokens"] = len(loaded.tokenizer.encode(post.text, add_special_tokens=False)) - len(
            loaded.tokenizer.encode(base_text, add_special_tokens=False))

        # filler arm: unrelated text inserted BEFORE the options, identical
        # single-turn message structure to base (structure held constant)
        filler = FILLER if filler_tokens <= THINK_BUDGET else FILLER[:400]
        r_fill = decide_on(
            loaded,
            question_messages(it["question"] + "\n\nContext note: " + filler, it["choices"]),
            it["choices"],
        )
        rec["filler_correct"] = r_fill.selected == gold
        rec["filler_conf"] = r_fill.confidence

        records.append(rec)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)

    # verdict
    def acc(arm):
        return sum(r[arm] for r in records) / len(records)

    def mean_conf(arm):
        return statistics.fmean(r[arm] for r in records)

    d_think = acc("think_correct") - acc("base_correct")
    d_fill = acc("filler_correct") - acc("base_correct")
    lines = [
        "# Neutral-Filler THINK Control (Phase 2)",
        "",
        f"domains: {DOMAINS}, n={len(items)}, think budget={THINK_BUDGET}, decision method: D2 alpha=1.0 pinned",
        "",
        "| arm | accuracy | mean confidence |",
        "|---|---|---|",
        f"| base | {acc('base_correct'):.3f} | {mean_conf('base_conf'):.3f} |",
        f"| think | {acc('think_correct'):.3f} | {mean_conf('think_conf'):.3f} |",
        f"| filler | {acc('filler_correct'):.3f} | {mean_conf('filler_conf'):.3f} |",
        "",
        f"- Δacc(think) = **{d_think:+.3f}**",
        f"- Δacc(filler) = **{d_fill:+.3f}**",
        f"- differential = **{d_think - d_fill:+.3f}** "
        + ("→ reasoning content causally helps (S5 supported)"
           if d_think - d_fill > 0.02 else
           "→ no evidence reasoning content (vs extra tokens) improves decisions (S5 NOT supported)"),
        f"- filler tokens: {filler_tokens}",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "filler_control.md").write_text("\n".join(lines) + "\n")
    with open(REPORTS / "filler_control_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("\n".join(lines[-6:]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
