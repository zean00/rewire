"""Reusable eval cells: generative baselines (A/B/C/G) and decision cells (D1/D2)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from hybrid.decide import decide
from hybrid.engine import generate_answer
from hybrid.extract import extract_choice_idx, match_gold_text
from hybrid.template import plain_question, question_messages


def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for it in items:
        # normalize domain label: dataset builder tags ARC-Challenge items
        # "arc_challenge"; the experiment grid calls the domain "arc_hard"
        if it.get("domain") == "arc_challenge":
            it["domain"] = "arc_hard"
    return items


def shuffled_choices(item: dict, seed: int) -> list[str]:
    rng = random.Random(seed + hash(item["id"]) % 100000)
    order = list(range(len(item["choices"])))
    rng.shuffle(order)
    return [item["choices"][j] for j in order]


def run_generative_cell(
    loaded, items: list[dict], name: str, *, show_options: bool,
    enable_thinking: bool | None, max_new_tokens: int,
) -> list[dict]:
    records = []
    for k, it in enumerate(items):
        msgs = question_messages(it["question"], it["choices"]) if show_options else plain_question(it["question"])
        t0 = time.perf_counter()
        answer, meta = generate_answer(loaded, msgs, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        pred = extract_choice_idx(answer, it["choices"]) if show_options else (
            it["answer_idx"] if match_gold_text(answer, it["choices"][it["answer_idx"]]) else None
        )
        n_tok = meta.get("think_tokens_approx", 0) + len(loaded.tokenizer.encode(answer, add_special_tokens=False))
        records.append({
            "cell": name, "domain": it["domain"], "id": it["id"], "correct": pred == it["answer_idx"],
            "parsed": pred is not None, "gen_tokens": n_tok, "latency_ms": lat_ms,
            "has_think": meta.get("has_think", False),
        })
        if (k + 1) % 25 == 0:
            print(f"  [{name}] {k + 1}/{len(items)}", flush=True)
    return records


def run_decision_cell(
    loaded, items: list[dict], name: str, *, method: str = "d2", alpha: float = 1.0,
    pmi_lambda: float = 0.0, anchor: bool = False, choice_seed: int | None = None,
) -> list[dict]:
    records = []
    for k, it in enumerate(items):
        choices = shuffled_choices(it, choice_seed) if choice_seed is not None else it["choices"]
        res = decide(
            loaded, it["question"], choices, method=method, alpha=alpha,
            pmi_lambda=pmi_lambda, anchor=anchor,
        )
        pred = choices.index(res.selected)
        gold_text = it["choices"][it["answer_idx"]]
        records.append({
            "cell": name, "domain": it["domain"], "id": it["id"],
            "correct": res.selected == gold_text, "parsed": True,
            "selected": res.selected, "confidence": res.confidence,
            "premask_mass": res.premask_mass, "latency_ms": res.total_ms,
            "margin": res.margin, "entropy": res.entropy,
        })
        if (k + 1) % 100 == 0:
            print(f"  [{name}] {k + 1}/{len(items)}", flush=True)
    return records
