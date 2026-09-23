"""Build the web-replay eval set (webreplay_v1) from Mind2Web (osunlp/Mind2Web, HF).

Streams the train split, keeps the first --n-tasks tasks, and emits one item per
annotated action whose target element survives HTML cleaning:

  {
    "id": "m2w_<annotation_id>_<action_uid>",
    "task": str,                       # confirmed_task
    "history": [str, ...],             # action_reprs before this step
    "op": "CLICK" | "TYPE" | "SELECT",
    "value": str | null,               # typed/selected value (TYPE/SELECT only)
    "candidates": [{"bid", "tag", "text", "attrs"}...],   # gold always included
    "gold_bid": str,
    "n_candidates_total": int,         # candidates on the page before capping
  }

Candidate text is extracted from cleaned_html via backend_node_id. Candidates
are capped at --max-candidates (gold first, then dataset order) — reported in
n_candidates_total so the capping is never hidden. Split is deterministic:
items[::2] = cal, items[1::2] = test (same convention as eval_v1).

Usage:
  python -m eval.datasets.build_webreplay --n-tasks 250 --max-candidates 20
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

TEXT_KEYS = ("aria-label", "title", "placeholder", "alt", "name", "id", "href")


def _shorten(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s[: n - 1] + "…" if len(s) > n else s


def _element_map(soup: BeautifulSoup) -> dict[str, dict]:
    """backend_node_id -> rendering fields for every node that carries one.

    Text policy (first non-empty wins): direct text -> one-level-down child
    texts. Containers thereby stop dumping whole page sections (the smoke test
    showed 'main' candidates swallowing the page and creating duplicate lines).
    parent_text + DOM path give context to text-free elements (svg icons), whose
    identity lives in their surroundings, not themselves.
    """
    out: dict[str, dict] = {}
    for el in soup.find_all(True):
        bid = el.get("backend_node_id")
        if bid is None:
            continue
        own = " ".join(s.strip() for s in el.find_all(string=True, recursive=False) if s.strip())
        if not own:
            own = " ".join(
                " ".join(s.strip() for s in ch.find_all(string=True, recursive=False) if s.strip())
                for ch in el.find_all(recursive=False)
            ).strip()
        text = _shorten(own or "", 80)
        parent = el.parent
        ptext = _shorten(" ".join(s.strip() for s in parent.find_all(string=True, recursive=False) if s.strip()) if parent else "", 40)
        gp = parent.parent if parent is not None else None
        gptext = _shorten(" ".join(s.strip() for s in gp.find_all(string=True, recursive=False) if s.strip()) if gp else "", 40)
        sub = _shorten(el.get_text(" ", strip=True) or "", 200)
        path = [p.name for p in el.parents if p.name][:4]
        attrs = {k: _shorten(str(el.get(k)), 40) for k in TEXT_KEYS if el.get(k)}
        out[str(bid)] = {
            "tag": el.name, "text": text, "attrs": attrs,
            "path": path, "parent_text": ptext, "gp_text": gptext, "sub": sub,
        }
    return out


def _dedupe(items: list[dict]) -> list[dict]:
    """Drop candidates that render identically to an earlier one (gold kept)."""
    for it in items:
        seen: set = set()
        kept = []
        for c in it["candidates"]:
            key = (c["tag"], c["text"], tuple(c.get("path") or ()), c.get("parent_text"))
            if key in seen and c["bid"] != it["gold_bid"]:
                continue
            seen.add(key)
            kept.append(c)
        it["candidates"] = kept
    return items


def build_items(tasks, max_candidates: int, max_text: int) -> list[dict]:
    items: list[dict] = []
    n_empty_gold = 0
    for task in tasks:
        ann = task["annotation_id"]
        for act in task["actions"]:
            pos = act.get("pos_candidates") or []
            if not pos:
                n_empty_gold += 1
                continue
            soup = BeautifulSoup(act["cleaned_html"], "lxml")
            emap = _element_map(soup)
            gold_bid = str(pos[0]["backend_node_id"])
            if gold_bid not in emap:
                n_empty_gold += 1
                continue

            negs = [str(c["backend_node_id"]) for c in act.get("neg_candidates") or []]
            keep_bids = [gold_bid] + [b for b in negs if b != gold_bid and b in emap]
            keep_bids = keep_bids[:max_candidates]
            random.Random(ann).shuffle(keep_bids)  # de-bias positional order

            op = act["operation"]["op"]
            items.append({
                "id": f"m2w_{ann}_{act['action_uid']}",
                "task": task["confirmed_task"],
                "history": task["action_reprs"][:-1],
                "op": op,
                "value": act["operation"].get("value"),
                "candidates": [emap[b] | {"bid": b} for b in keep_bids],
                "gold_bid": gold_bid,
                "n_candidates_total": len(negs) + len(pos),
            })
    print(f"skipped {n_empty_gold} actions (empty/missing gold after cleaning)", flush=True)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=250)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--max-text", type=int, default=120)
    ap.add_argument("--out", default="eval/datasets/webreplay_v1")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    tasks = list(itertools.islice(iter(ds), args.n_tasks))
    print(f"fetched {len(tasks)} tasks", flush=True)

    items = build_items(tasks, args.max_candidates, args.max_text)
    items = _dedupe(items)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "webreplay_v1.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_cal = len(list(items[::2]))
    (out / "MANIFEST.json").write_text(json.dumps({
        "source": "osunlp/Mind2Web (train, first n via streaming)",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(tasks), "n_items": len(items), "n_cal": n_cal,
        "n_test": len(items) - n_cal, "max_candidates": args.max_candidates,
        "ops": {op: sum(1 for it in items if it["op"] == op) for op in ("CLICK", "TYPE", "SELECT")},
        "candidate_counts": {"p50": sorted(len(it["candidates"]) for it in items)[len(items) // 2],
                             "max": max(len(it["candidates"]) for it in items)},
        "note": "candidates capped (gold always kept); order shuffled per item with seed=annotation_id",
    }, indent=1))
    print(f"wrote {len(items)} items -> {out}/webreplay_v1.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    import itertools
    raise SystemExit(main())
