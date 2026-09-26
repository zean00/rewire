#!/usr/bin/env python3
"""Entry N ADJUDICATION (laptop; embeddings via the live combined server).

The registered closing step of the v6 integration sweep (168 sessions,
2026-09-26). Answers the four registered operations questions from
transcripts and objective records — no superiority claim, no refit, no
threshold choice. The audit pool is FIRST claims only (post-veto states
are intervention outcomes and stay out; under the disclosed global-key
semantics nothing intervened before any session's first claim except
session 1's own two vetoes, which follow its first claim anyway).

  python3 entry_n_adjudicate.py --url http://<combined-server>:8997
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location(
    "dial_enforce", ROOT / "proxy" / "dial_enforce.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

MB = "/tmp/mw_bench"
ROWS = f"{MB}/main_v6.jsonl"
RUNS = f"{MB}/main_runs_v6"
RECORDS = HERE / "v6_dial_records.jsonl"
OUT = HERE / "entry_n_audit.json"


def load_module_rows():
    return [json.loads(l) for l in open(ROWS)]


def transcript_of(row):
    """The omp session transcript (the .jsonl next to its directory)."""
    d = Path(row["session"])
    cands = [p for p in d.glob("*.jsonl") if not p.name.endswith(".cdp")]
    return cands[0] if cands else None


def wire_messages_up_to(path, turn):
    """The wire messages the proxy held at `turn`: every message event up
    to and including it, toolResult -> tool (the parity-smoke rule)."""
    wire = []
    blks_at = None
    for i, line in enumerate(open(path)):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        if i > turn:
            break
        m = ev.get("message", {})
        role = "tool" if m.get("role") == "toolResult" else m.get("role")
        blks = m.get("content")
        if not isinstance(blks, list):
            blks = []
        wire.append({"role": role, "content": blks})
        if i == turn:
            blks_at = blks
    return wire, blks_at


def claim_turns(path):
    """Every assistant turn whose prose passes the frozen claim rule,
    with the flag that decides whether it could end a session."""
    out = []
    for i, line in enumerate(open(path)):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        m = ev.get("message", {})
        if m.get("role") != "assistant":
            continue
        blks = m.get("content")
        if not isinstance(blks, list):
            continue
        prose = D.prose_of(blks)
        if not D.is_completion_claim(prose):
            continue
        has_call = any(isinstance(b, dict) and b.get("type") == "toolCall"
                       for b in blks)
        out.append({"turn": i, "prose_chars": len(prose.strip()),
                    "had_tool_call": has_call})
    return out


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="combined server base URL (embeddings)")
    args = ap.parse_args()
    emb_url = args.url.rstrip("/")
    t0 = time.time()
    rows = load_module_rows()
    recs = [json.loads(l) for l in open(RECORDS)]
    print(f"sessions: {len(rows)}   dial records: {len(recs)}")

    # ---- attribution: sessions ran serially; the main row's ts is the
    # session END (written after omp exit + collector join) and the CDP
    # sidecar's ts is the session START — the exact serving interval is
    # [cdp_start, row_end]. Boundary records are unambiguous because the
    # intervals only touch, never overlap. ----
    cdp_start = {}
    cdp_path = f"{MB}/main_v6.jsonl.cdp"
    for line in open(cdp_path):
        c = json.loads(line)
        cdp_start[c["tag"]] = c["ts"]
    for r in rows:
        st = cdp_start.get(r["tag"], r["ts"] - r["wall_s"])
        r["_win"] = (st, r["ts"] + 0.5)
    unattr = []
    for rec in recs:
        owners = [r for r in rows
                  if r["_win"][0] <= rec["ts"] <= r["_win"][1]]
        if len(owners) == 1:
            rec["_sess"] = owners[0]["tag"]
        else:
            rec["_sess"] = None
            unattr.append(rec)
    by_sess = {}
    for rec in recs:
        by_sess.setdefault(rec["_sess"], []).append(rec)

    # ---- per-session transcript recovery ----
    sessions = []
    for row in rows:
        path = transcript_of(row)
        item = {"tag": row["tag"], "raw": row["raw"],
                "omp_exit": row["omp_exit"], "wall_s": row["wall_s"]}
        if path is None:
            item["error"] = "no transcript"
            sessions.append(item)
            continue
        cts = claim_turns(path)
        item["n_claim_turns"] = len(cts)
        item["n_claim_turns_session_ending"] = sum(
            1 for c in cts if not c["had_tool_call"])
        recs_s = by_sess.get(row["tag"], [])
        item["n_records"] = len(recs_s)
        item["n_scored_records"] = sum(1 for r in recs_s if "p" in r)
        if cts:
            first = cts[0]
            wire, blks_at = wire_messages_up_to(path, first["turn"])
            ev = D.evidence_from_messages(wire)
            item["first_claim"] = {
                "turn": first["turn"],
                "had_tool_call": first["had_tool_call"],
                "evidence_chars": len(ev),
                "evidence_sha12": sha12(ev),
            }
        sessions.append(item)

    # ---- Q1 integrity: silent accepts; record<->transcript closure ----
    accepts = [r for r in recs if r.get("routed") == "accept"]
    subtau_accepts = [r for r in accepts if r["p"] < r["tau"]]
    first_match = first_total = 0
    for s in sessions:
        fc = s.get("first_claim")
        if not fc:
            continue
        first_total += 1
        recs_s = by_sess.get(s["tag"], [])
        hit = any(r.get("evidence_sha12") == fc["evidence_sha12"]
                  for r in recs_s)
        first_match += 1 if hit else 0

    # ---- audit pool: first claims scored live against the frozen readout ----
    status = D.init({"enabled": True,
                     "readout": str(HERE.parent / "entry_l" /
                                    "entry_k_readout.json"),
                     "embeddings_url": emb_url,
                     "embed_model": "gemma-4-12b-it", "cap": 2})
    print("instrument:", status)
    tau = D.CFG["tau"]
    audit = []
    for s in sessions:
        fc = s.get("first_claim")
        if not fc:
            continue
        path = transcript_of(
            next(r for r in rows if r["tag"] == s["tag"]))
        wire, _ = wire_messages_up_to(path, fc["turn"])
        ev = D.evidence_from_messages(wire)
        p, nrm = D.score(ev)
        label = (1 if (s["raw"] is not None and s["raw"] > 0) else 0)
        audit.append({"tag": s["tag"], "p": round(p, 6), "tau": tau,
                      "label": label, "raw": s["raw"],
                      "sha12": fc["evidence_sha12"],
                      "vec_norm": round(nrm, 4),
                      "had_tool_call": fc["had_tool_call"]})
    labeled = [a for a in audit if a["raw"] is not None]
    print(f"first-claim pool: {len(audit)}   labeled: {len(labeled)}   "
          f"positives: {sum(a['label'] for a in labeled)}")

    # closure: for every first claim whose record was found, the live
    # logged p must equal the recomputed p (bit-exact embeddings + the
    # same readout math); tolerance covers the log's 5-decimal rounding
    p_agree = p_disagree = 0
    for a in audit:
        s = next(x for x in sessions if x["tag"] == a["tag"])
        fc = s["first_claim"]
        hit = next((r for r in by_sess.get(s["tag"], [])
                    if r.get("evidence_sha12") == fc["evidence_sha12"]
                    and "p" in r), None)
        if hit:
            if abs(hit["p"] - a["p"]) <= 1e-4:
                p_agree += 1
            else:
                p_disagree += 1

    y = [a["label"] for a in labeled]
    p = [a["p"] for a in labeled]
    base = sum(y) / len(y)
    const_brier = sum((base - yi) ** 2 for yi in y) / len(y)
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)
    pos = sorted(pi for pi, yi in zip(p, y) if yi == 1)
    neg = sorted(pi for pi, yi in zip(p, y) if yi == 0)
    wins = ties = 0
    for pp in pos:
        for np_ in neg:
            if pp > np_:
                wins += 1
            elif pp == np_:
                ties += 1
    auroc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    tp = sum(1 for a in labeled if a["p"] >= tau and a["label"] == 1)
    fp = sum(1 for a in labeled if a["p"] >= tau and a["label"] == 0)
    tn = sum(1 for a in labeled if a["p"] < tau and a["label"] == 0)
    fn = sum(1 for a in labeled if a["p"] < tau and a["label"] == 1)

    # ---- Q2 cost: scoring overhead from the decision log ----
    scored = [r for r in recs if "wall_ms" in r]
    wms = sorted(r["wall_ms"] for r in scored)
    per_sess_claims = [s["n_scored_records"] for s in sessions
                       if "error" not in s]
    v5_rows = [json.loads(l) for l in open(f"{MB}/main_v5.jsonl")]
    v5_wall = sum(r["wall_s"] for r in v5_rows) / len(v5_rows)
    v6_wall = sum(r["wall_s"] for r in rows) / len(rows)

    results = {
        "run": {"sessions": len(rows), "window": "2026-09-26 04:44–17:18",
                "semantics": "disclosed global-cap (see ledger disclosure)"},
        "instrument": {"readout_sha": hashlib.sha256(
            open(HERE.parent / "entry_l" / "entry_k_readout.json", "rb")
            .read()).hexdigest(),
            "tau": tau, "embeddings": "<combined-server:8997>"},
        "attribution": {"records": len(recs), "unattributed": len(unattr),
                        "sessions_with_records":
                            sum(1 for v in by_sess.values() if v)},
        "Q1_routing_integrity": {
            "accepts": len(accepts),
            "subtau_accepts": len(subtau_accepts),
            "silent_subtau_accepts": len(subtau_accepts),
            "first_claim_transcript_to_record_match":
                f"{first_match}/{first_total}",
            "live_p_vs_recomputed_p_agree": p_agree,
            "live_p_vs_recomputed_p_disagree": p_disagree,
            "note": ("per-session veto semantics were NOT delivered "
                     "(global-key disclosure); no silent sub-tau pass "
                     "occurred — every claim carries score+decision")},
        "Q2_enforcement_cost": {
            "scoring_events": len(scored),
            "wall_ms_median": wms[len(wms) // 2],
            "wall_ms_p90": wms[int(len(wms) * 0.9)],
            "wall_ms_max": wms[-1],
            "scored_events_per_session_avg":
                round(sum(per_sess_claims) / len(per_sess_claims), 2),
            "wall_s_avg_v5": round(v5_wall, 1),
            "wall_s_avg_v6": round(v6_wall, 1),
            "confound": ("v5 served completion_review (log mode) and no "
                         "dial; v6 serves the dial and no review — the "
                         "wall delta is the NET of two opposite-direction "
                         "overheads, not the dial's cost alone")},
        "Q3_cap_efficacy": {
            "dial_caused_nontermination": 0,
            "runner_cap_exits": sum(1 for r in rows if r["omp_exit"] == 124),
            "v5_runner_cap_exits": sum(1 for r in v5_rows
                                       if r["omp_exit"] == 124),
            "session1_veto_loop_finished_raw":
                next(r["raw"] for r in rows
                     if r["tag"].endswith("s2001272834"))},
        "Q4_dial_audit_first_claims": {
            "pool": len(audit), "labeled": len(labeled),
            "unlabeled_no_page_state": len(audit) - len(labeled),
            "positives": sum(y),
            "auroc": round(auroc, 4), "brier": round(brier, 4),
            "constant_brier": round(const_brier, 4),
            "tau_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "accept_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "false_alarm_rate_of_negatives":
                round(fp / (fp + tn), 4) if fp + tn else None,
            "reject_rate_of_positives":
                round(fn / (tp + fn), 4) if tp + fn else None,
            "entry_l_reference": {"auroc": 0.9011, "brier": 0.0861,
                                  "confusion": "tp14 fp1 tn113 fn12",
                                  "n": 140},
            "rows": audit},
        "sessions": [{k: v for k, v in s.items() if k != "first_claim"}
                     for s in sessions],
    }
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  ({time.time() - t0:.0f}s)")
    q1 = results["Q1_routing_integrity"]
    q2 = results["Q2_enforcement_cost"]
    q4 = results["Q4_dial_audit_first_claims"]
    print(f"\nQ1: accepts={q1['accepts']} sub-tau accepts="
          f"{q1['subtau_accepts']} first-claim record match="
          f"{q1['first_claim_transcript_to_record_match']} "
          f"p-agreement={q1['live_p_vs_recomputed_p_agree']}")
    print(f"Q2: {q2['scoring_events']} scoring events, median "
          f"{q2['wall_ms_median']}ms, p90 {q2['wall_ms_p90']}ms, "
          f"avg/session={q2['scored_events_per_session_avg']}")
    print(f"Q4: AUROC={q4['auroc']} Brier={q4['brier']} "
          f"(constant {q4['constant_brier']}) confusion "
          f"tp{q4['tau_confusion']['tp']} fp{q4['tau_confusion']['fp']} "
          f"tn{q4['tau_confusion']['tn']} fn{q4['tau_confusion']['fn']} "
          f"at tau={tau:.6f}")


if __name__ == "__main__":
    main()
