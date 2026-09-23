"""Answer extraction utilities (shared by eval cells and the runtime)."""

from __future__ import annotations

import re


def match_gold_text(answer: str, gold: str) -> bool:
    return gold.strip().lower() in answer.strip().lower()


def extract_choice_idx(answer: str, choices: list[str]) -> int | None:
    """Map a generated answer to a choice index.

    The decision prompt lists numbered text options and asks for one of them,
    so models typically answer with option text ("Canberra"), possibly with an
    index ("2. Canberra"). Match the first choice text mentioned in the answer
    (conversational answers can mention several — first mention is the
    standard convention); fall back to a bare index number.
    """
    a = answer.lower()
    hits = [(a.find(c.lower()), i) for i, c in enumerate(choices) if c.lower() in a]
    real = [h for h in hits if h[0] >= 0]
    if real:
        return min(real)[1]
    m = re.search(r"\b([1-9])\b", a)
    if m and int(m.group(1)) <= len(choices):
        return int(m.group(1)) - 1
    return None
