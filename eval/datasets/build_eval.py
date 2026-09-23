"""Build eval set v1 (IMPLEMENTATION_PLAN.md 2.4).

Domains (each item: {id, domain, question, choices, answer_idx}):

  arc_easy / arc_hard   Allen AI ARC (real MCQA, easy/challenge splits) — the
                        difficulty axis. Sampled with fixed seed.
  arith_easy / arith_hard  Generated arithmetic, gold by construction. Hard =
                        multi-step with plausible distractors; thinking should
                        pay off here.
  intent                Generated §15.1 action-selection; gold by construction
                        from explicit situation semantics.

Writes eval/datasets/eval_v1/*.jsonl + MANIFEST.json (provenance, seeds).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/datasets/eval_v1"
SEED = 17


def write_jsonl(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} items")


def sample_arc(config: str, n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", config, split="test")
    ds = ds.shuffle(seed=SEED)
    rows = []
    for ex in ds:
        keys = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        if ex["answerKey"] not in keys or len(texts) < 3:
            continue
        rows.append({
            "id": f"{config.lower().replace('-', '_')}_{ex['id']}",
            "domain": config.lower().replace("-", "_"),
            "question": ex["question"].strip(),
            "choices": [t.strip() for t in texts],
            "answer_idx": keys.index(ex["answerKey"]),
        })
        if len(rows) >= n:
            break
    return rows


def gen_arithmetic() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    easy, hard = [], []

    def distractors(val: int, pool: list[int]) -> list[int]:
        cands = [val + rng.choice([-1, 1, -10, 10, 2, -2]) for _ in range(8)]
        cands += pool
        out: list[int] = []
        for c in cands:
            if c != val and c not in out:
                out.append(c)
            if len(out) == 3:
                break
        while len(out) < 3:
            c = val + rng.randint(-20, 20)
            if c != val and c not in out:
                out.append(c)
        return out

    for i in range(100):
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        if rng.random() < 0.5:
            q, val = f"What is {a} + {b}?", a + b
        else:
            q, val = f"What is {a} - {b}?", a - b
        ds_ = distractors(val, [a, b, a + b, a - b, val + 100])
        choices = [str(val)] + [str(d) for d in ds_]
        rng.shuffle(choices)
        easy.append({"id": f"arith_easy_{i:03d}", "domain": "arith_easy", "question": q,
                     "choices": choices, "answer_idx": choices.index(str(val))})

    for i in range(100):
        a, b, c, d = (rng.randint(12, 39), rng.randint(3, 9), rng.randint(100, 499), rng.randint(10, 99))
        val = a * b + c - d
        q = f"What is ({a} × {b}) + {c} − {d}?"
        wrong = [val + rng.choice([-b, b, -10, 10, 100]), val - a * b + rng.randint(1, 9), rng.randint(min(val - 200, 1), val + 200)]
        ds_ = distractors(val, wrong)
        choices = [str(val)] + [str(x) for x in ds_]
        rng.shuffle(choices)
        hard.append({"id": f"arith_hard_{i:03d}", "domain": "arith_hard", "question": q,
                     "choices": choices, "answer_idx": choices.index(str(val))})
    return easy, hard


def gen_intent() -> list[dict]:
    """§15.1-style action selection. Gold by construction: the situation text
    states exactly which condition holds."""
    rng = random.Random(SEED + 1)
    candidates = ["answer from context", "inspect files", "search the web", "ask the user"]
    topics = ["sales report", "photo album", "contract draft", "recipe", "meeting notes",
              "budget spreadsheet", "travel itinerary", "code review feedback", "resume", "invoice"]
    rows = []
    templates = [
        # (situation template, gold candidate) — the resource condition is explicit.
        ("The user asks you to summarize the {topic}. The {topic} is attached to this conversation in full.", 0),
        ("The user asks you to summarize the {topic}. The {topic} is not attached, but local file tools are available and the {topic} exists on disk.", 1),
        ("The user asks for this week's exchange rate for the {topic} purchase, and the conversation contains no such data.", 2),
        ("The user asks you to update the {topic} but refers to it only as 'the thing we discussed yesterday', with no other identifying detail in the conversation.", 3),
        ("The user asks whether the {topic} contains any overdue items, and the {topic} is attached in full.", 0),
        ("The user asks you to fix a broken link inside the {topic}; the {topic} is not attached but the files are available locally.", 1),
        ("The user asks about a regulatory change affecting the {topic} that happened after your knowledge cutoff.", 2),
        ("The user asks you to choose between two versions of the {topic} but never says what the versions are or where to find them.", 3),
    ]
    for i in range(100):
        tpl, gold = templates[i % len(templates)]
        topic = rng.choice(topics)
        q = tpl.format(topic=topic)
        order = list(range(4))
        rng.shuffle(order)
        choices = [candidates[j] for j in order]
        rows.append({"id": f"intent_{i:03d}", "domain": "intent", "question": q,
                     "choices": choices, "answer_idx": order.index(gold)})
    return rows


def main() -> None:
    write_jsonl("arc_easy", sample_arc("ARC-Easy", 100))
    write_jsonl("arc_hard", sample_arc("ARC-Challenge", 100))
    easy, hard = gen_arithmetic()
    write_jsonl("arith_easy", easy)
    write_jsonl("arith_hard", hard)
    write_jsonl("intent", gen_intent())
    manifest = {
        "seed": SEED,
        "source": "allenai/ai2_arc (test split, sampled) + generated arithmetic/intent (gold by construction)",
        "files": ["arc_easy", "arc_hard", "arith_easy", "arith_hard", "intent"],
        "note": "visual MC deferred to Phase 6; calibration split carved per-domain at Phase 3",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    main()
