"""Phase 0 validity report (IMPLEMENTATION_PLAN.md, Step 1).

Produces eval/reports/phase0_validity.md containing:
  1. environment + load-path record;
  2. pinned-read-point determinism check;
  3. scoring-position sensitivity (pinned vs post-think), 10 items;
  4. pre-mask mass distribution over the toy set (D2, and D1 where applicable);
  5. baselines A / B / G smoke cells (accuracy + latency + tokens).

Run on the GPU host:
  python -m eval.phase0_report [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml

from hybrid.engine import generate_answer, load_model
from hybrid.scoring.premask import PREMASK_VALID_THRESHOLD
from hybrid.scoring.restricted_logits import candidate_first_tokens, score_first_token
from hybrid.scoring.sequence_logprob import score_sequence_logprob
from hybrid.template import pin_after_think, pin_prompt, question_messages

ROOT = Path(__file__).resolve().parents[1]


def load_items(path: Path, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


def extract_choice(answer: str, choices: list[str]) -> int | None:
    """Map a generated answer to a choice index.

    The decision prompt lists numbered text options and asks for one of them,
    so models typically answer with option text ("Canberra"), possibly with an
    index ("2. Canberra"). Match the first choice text mentioned in the answer
    (conversational answers can mention several — first mention is the
    standard convention); fall back to a bare index number.
    """
    a = answer.lower()
    positions = [(a.find(c.lower()), i) for i, c in enumerate(choices) if c.lower() in a]
    hits = [p for p in positions if p[0] >= 0]
    if hits:
        return min(hits)[1]
    m = re.search(r"\b([1-9])\b", a)
    if m and int(m.group(1)) <= len(choices):
        return int(m.group(1)) - 1
    return None


def fmt_row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sensitivity-items", type=int, default=10)
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip the expensive post-think generation cells (e.g. when re-running baselines only)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
    items = load_items(ROOT / "eval/datasets/toy_mcqa.jsonl", args.limit)

    lines: list[str] = ["# Phase 0 — Validity Report", ""]

    # --- environment ---------------------------------------------------------
    import transformers

    loaded = load_model(cfg["model"]["id"])
    torch.cuda.reset_peak_memory_stats()
    lines += [
        "## Environment",
        "",
        fmt_row(["key", "value"]),
        fmt_row(["---", "---"]),
        fmt_row(["torch", torch.__version__]),
        fmt_row(["cuda available", torch.cuda.is_available()]),
        fmt_row(["device capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a"]),
        fmt_row(["transformers", transformers.__version__]),
        fmt_row(["model", cfg["model"]["id"]]),
        fmt_row(["load path (HF class)", loaded.load_path]),
        fmt_row(["dtype", loaded.dtype]),
    ]
    print("\n".join(lines[-10:]), flush=True)

    # --- determinism ----------------------------------------------------------
    item0 = items[0]
    pinned0 = pin_prompt(loaded.tokenizer, question_messages(item0["question"], item0["choices"]))
    with torch.no_grad():
        l1 = loaded.model(input_ids=pinned0.input_ids.to(loaded.model.device)).logits[0, -1, :]
        l2 = loaded.model(input_ids=pinned0.input_ids.to(loaded.model.device)).logits[0, -1, :]
    det_maxdiff = float((l1 - l2).abs().max().item())
    lines += [
        "",
        "## 1. Pinned read-point determinism",
        "",
        f"- two identical forwards, max |logit diff| = **{det_maxdiff:.3e}** "
        + ("(deterministic ✔)" if det_maxdiff < 1e-3 else "(NON-DETERMINISTIC — investigate before any measurement)"),
        f"- read_mode: `{pinned0.read_mode}`, template_kwargs: `{pinned0.template_kwargs}`",
        "",
        "Pinned prompt (item 0) verbatim:",
        "",
        "```text",
        pinned0.text.strip()[:2000],
        "```",
    ]
    print(f"determinism maxdiff={det_maxdiff:.2e}", flush=True)

    # --- pre-mask mass (D2 over all items) ------------------------------------
    mass_rows = []
    d1_rows = []
    tok_map_all = candidate_first_tokens(loaded.tokenizer, ["A", "B", "C", "D"])
    for it in items:
        pinned = pin_prompt(loaded.tokenizer, question_messages(it["question"], it["choices"]))
        r2 = score_sequence_logprob(
            loaded.model, loaded.tokenizer, pinned, it["choices"], alpha=cfg["scoring"]["d2_alpha"]
        )
        gold = it["choices"][it["answer_idx"]]
        mass_rows.append((it["id"], r2.premask_mass, r2.selected == gold))
        r1, _ = score_first_token(loaded.model, loaded.tokenizer, pinned, it["choices"])
        d1_rows.append((it["id"], r1.premask_mass, r1.selected == gold))

    def mass_table(rows, name):
        vals = sorted(m for _, m, _ in rows)
        n_valid = sum(1 for m in vals if m >= PREMASK_VALID_THRESHOLD)
        acc = sum(1 for _, m, ok in rows if ok) / len(rows)
        return [
            f"### {name}",
            "",
            fmt_row(["metric", "value"]),
            fmt_row(["---", "---"]),
            fmt_row(["n", len(rows)]),
            fmt_row(["premask min / p50 / max", f"{vals[0]:.4f} / {vals[len(vals)//2]:.4f} / {vals[-1]:.4f}"]),
            fmt_row([f"n with mass ≥ {PREMASK_VALID_THRESHOLD}", f"{n_valid}/{len(rows)}"]),
            fmt_row(["top-1 accuracy (uncorrected)", f"{acc:.3f}"]),
            "",
        ]

    lines += ["## 2. Pre-mask candidate mass", ""] + mass_table(mass_rows, "D2 (alpha=%.1f)" % cfg["scoring"]["d2_alpha"]) + mass_table(d1_rows, "D1 (first-token)")
    print(f"premask d2 p50={sorted(m for _, m, _ in mass_rows)[len(mass_rows)//2]:.3f}", flush=True)

    # --- scoring-position sensitivity ------------------------------------------
    agree_n, sens_n = 0, 0
    if not args.skip_sensitivity:
        lines += ["## 3. Scoring-position sensitivity (pinned vs post-think)", "",
                fmt_row(["item", "pinned selected", "post-think selected", "pinned p_top", "post-think p_top", "agree"]),
                fmt_row(["---"] * 6)]
        sens_n = min(args.sensitivity_items, len(items))
        for it in items[:sens_n]:
            pinned = pin_prompt(loaded.tokenizer, question_messages(it["question"], it["choices"]))
            r_pin = score_sequence_logprob(loaded.model, loaded.tokenizer, pinned, it["choices"])
            post = pin_after_think(loaded.tokenizer, loaded.model, question_messages(it["question"], it["choices"]), max_think_tokens=256)
            r_post = score_sequence_logprob(loaded.model, loaded.tokenizer, post, it["choices"])
            agree = r_pin.selected == r_post.selected
            agree_n += int(agree)
            lines.append(fmt_row([it["id"], r_pin.selected[:24], r_post.selected[:24],
                                  f"{r_pin.confidence:.3f}", f"{r_post.confidence:.3f}", "✔" if agree else "✘"]))
            print(f"sens {it['id']}: pinned={r_pin.selected[:20]} post={r_post.selected[:20]} agree={agree}", flush=True)
        lines += ["", f"agreement: **{agree_n}/{sens_n}**", ""]

    # --- baselines A / B / G ----------------------------------------------------
    lines += ["## 4. Baselines A / B / G (smoke cells)", "",
              fmt_row(["baseline", "accuracy", "unparsed", "mean gen tokens", "mean latency ms"]),
              fmt_row(["---", "---", "---", "---", "---"])]
    import time as _t

    for name, kw in (("A (no think, max 256)", {"enable_thinking": False, "max_new_tokens": 256}),
                     ("B (forced think, max 768)", {"enable_thinking": True, "max_new_tokens": 768}),
                     ("G (native default)", {"enable_thinking": None, "max_new_tokens": 512})):
        correct = 0
        unparsed = 0
        toks, lats = [], []
        for it in items:
            msgs = question_messages(it["question"], it["choices"])
            t0 = _t.perf_counter()
            answer, meta = generate_answer(loaded, msgs, **kw)
            lats.append((_t.perf_counter() - t0) * 1000)
            toks.append(meta.get("think_tokens_approx", 0) + len(loaded.tokenizer.encode(answer, add_special_tokens=False)))
            pred = extract_choice(answer, it["choices"])
            correct += int(pred == it["answer_idx"])
            unparsed += int(pred is None or answer == "")
        lines.append(fmt_row([name, f"{correct}/{len(items)}", f"unparsed {unparsed}", f"{sum(toks)/len(toks):.0f}", f"{sum(lats)/len(lats):.0f}"]))
        print(f"baseline {name}: {correct}/{len(items)} unparsed={unparsed}", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    lines += ["", f"peak VRAM allocated: **{peak_gb:.2f} GB**", ""]

    out = ROOT / "eval/reports/phase0_validity.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
