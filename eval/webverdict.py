"""Gate B v2/v3 verdict analysis: matched-item McNemar comparisons on the
test split (records in eval/reports/webreplay_records.jsonl).

Usage: python -m eval.webverdict [--split test]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECS = ROOT / "eval/reports/webreplay_records.jsonl"


def load(split: str) -> dict[str, dict[str, dict]]:
    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in RECS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split", "test") != split:
            continue
        by_arm[r["arm"]][r["id"]] = r
    return by_arm


def mcnemar(a: dict[str, dict], b: dict[str, dict], key="correct"):
    """Returns (a_acc, b_acc, n, b_only_wins, a_only_wins, chi2, p)."""
    ids = sorted(set(a) & set(b))
    aw = bw = 0
    for i in ids:
        abool, bbool = a[i][key], b[i][key]
        if abool and not bbool:
            aw += 1
        elif bbool and not abool:
            bw += 1
    n = len(ids)
    if aw + bw == 0:
        acc_a = sum(x[key] for x in a.values()) / len(a) if a else 0.0
        acc_b = sum(x[key] for x in b.values()) / len(b) if b else 0.0
        return acc_a, acc_b, n, bw, aw, 0.0, 1.0
    chi2 = (abs(aw - bw) - 1) ** 2 / (aw + bw)
    p = math.erfc(math.sqrt(chi2 / 2))  # chi2_1 survival = erfc(sqrt(x/2))
    acc_a = sum(x[key] for x in a.values()) / len(a) if a else 0.0
    acc_b = sum(x[key] for x in b.values()) / len(b) if b else 0.0
    return acc_a, acc_b, n, bw, aw, chi2, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    arms = load(args.split)

    print(f"split={args.split}")
    for arm in sorted(arms):
        rows = list(arms[arm].values())
        n = len(rows)
        elem = sum(r["correct"] for r in rows) / n
        opa = sum(r["op_correct"] for r in rows) / n
        act = (sum(r["action_correct"] for r in rows) / n
               if any("action_correct" in r for r in rows) else None)
        vs = [r for r in rows if r.get("value_scored")]
        va = sum(r["value_exact"] for r in vs) / len(vs) if vs else None
        vloose = sum(r["value_loose"] for r in vs) / len(vs) if vs else None
        esc = (sum(r["escalated"] for r in rows) / n
               if any("escalated" in r for r in rows) else None)
        pf = sum(1 for r in rows if r.get("parsed") is False)
        p50 = sorted(r["latency_ms"] for r in rows)[n // 2] / 1000
        extra = f" action={act:.3f}" if act is not None else ""
        extra += f" value={va:.3f}/{vloose:.3f}(n={len(vs)})" if va is not None else ""
        extra += f" esc={esc:.0%}" if esc is not None else ""
        print(f"  {arm:7s} n={n} elem={elem:.3f} op={opa:.3f}{extra} "
              f"parse_fail={pf} p50={p50:.2f}s")

    def cmp(name, a_arm, b_arm, key="correct", pa=None, pb=None):
        if a_arm not in arms or b_arm not in arms:
            return
        a = arms[a_arm] if pa is None else {i: r for i, r in arms[a_arm].items() if pa(r)}
        b = arms[b_arm] if pb is None else {i: r for i, r in arms[b_arm].items() if pb(r)}
        if not a or not b or not (set(a) & set(b)):
            print(f"  {name}: skipped (empty subset)")
            return
        acc_a, acc_b, n, bw, aw, chi2, p = mcnemar(a, b, key)
        print(f"  {name}: {acc_b:.3f} vs {acc_a:.3f} (n={n}) "
              f"[{b_arm} wins {bw}, {a_arm} wins {aw}] chi2={chi2:.1f} p={p:.2g}")

    print("\nmatched comparisons (elem_acc):")
    cmp("D2E2  vs D2E  (op-first effect)", "D2E", "D2E2")
    cmp("D2E2  vs C_web (decide vs vanilla)", "C_web", "D2E2")
    cmp("H2    vs D2E2 same items (escalation effect)", "D2E2", "H2_web")
    cmp("H2    vs C_web", "C_web", "H2_web")
    if "H2_web" in arms:
        esc_ids = {i for i, r in arms["H2_web"].items() if r.get("escalated")}
        cmp("H2 escalated tail vs D2E2 same ids", "D2E2", "H2_web",
            pa=lambda r: r["id"] in esc_ids, pb=lambda r: r["id"] in esc_ids)

    if "H3_web" in arms or "C3_web" in arms:
        print("\nv3 tool-call comparisons:")
        cmp("H3    vs C3   (action_acc)", "C3_web", "H3_web", key="action_correct")
        cmp("H3    vs G3   (action_acc, G3 subsample)", "G3_web", "H3_web", key="action_correct")
        cmp("H3    vs C3   (elem_acc)", "C3_web", "H3_web")
        cmp("H3    vs C3   (value_exact, scored items)",
            "C3_web", "H3_web", key="value_exact",
            pa=lambda r: r.get("value_scored"), pb=lambda r: r.get("value_scored"))
        cmp("H3    vs C3   (value_loose, scored items)",
            "C3_web", "H3_web", key="value_loose",
            pa=lambda r: r.get("value_scored"), pb=lambda r: r.get("value_scored"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
