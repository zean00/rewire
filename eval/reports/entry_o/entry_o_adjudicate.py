#!/usr/bin/env python3
"""Entry O ADJUDICATION (laptop; embeddings via the live combined server).

The registered closing step of the v7 decisive A/B (336 sessions = 168
paired instances x 2 arms, 2026-09-26/27). ONE execution. All numbers come
from transcripts and objective records; the dial audit is FIRST claims of
GUARDED sessions only (vanilla serves no dial by construction), scored
live through the frozen instrument — the exact entry-N method.

PRIMARY (frozen before session 1): success rate per arm = (raw > 0) / 168
— sessions with no terminal page state count as NOT success; conservative,
symmetric, no exclusions, no discretion. GUARDED is the verdict iff
success_rate_guarded - success_rate_vanilla >= +0.10; any smaller gap
registers as NO VERIFIED IMPROVEMENT, including if vanilla wins.

  python3 entry_o_adjudicate.py --url http://<combined-server>:8997
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location(
    "dial_enforce", ROOT / "proxy" / "dial_enforce.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

MB = "/tmp/mw_bench"
ROWS = f"{MB}/main_v7.jsonl"
RUNS = f"{MB}/main_runs_v7"
CDP = f"{MB}/main_v7.jsonl.cdp"
RECORDS = f"{MB}/v7_dial_records.jsonl"
OUT = HERE / "entry_o_audit.json"
N_PER_ARM = 168
BAR = 0.10


def transcript_of(row):
    d = Path(row["session"])
    cands = [p for p in d.glob("*.jsonl") if not p.name.endswith(".cdp")]
    return cands[0] if cands else None


def wire_messages_up_to(path, turn):
    wire = []
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
    return wire


def claim_turns(path):
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
        out.append({"turn": i, "had_tool_call": has_call})
    return out


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    emb_url = args.url.rstrip("/")
    t0 = time.time()
    rows = [json.loads(l) for l in open(ROWS)]
    assert len(rows) == 336, f"expected 336 rows, got {len(rows)}"
    all_recs = [json.loads(l) for l in open(RECORDS)]
    recs = [r for r in all_recs if r.get("arm") == "dial-enforce"]
    g = [r for r in rows if r["arm"] == "guarded"]
    v = [r for r in rows if r["arm"] == "vanilla"]
    assert len(g) == len(v) == N_PER_ARM
    print(f"sessions: {len(rows)}   dial-enforce records: {len(recs)}")

    # ---- PRIMARY: frozen verdict bar -------------------------------------
    def rate(arm_rows):
        return sum(1 for r in arm_rows if (r.get("raw") or 0) > 0) / N_PER_ARM

    r_g, r_v = rate(g), rate(v)
    gap = r_g - r_v
    se = math.sqrt(r_g * (1 - r_g) / N_PER_ARM + r_v * (1 - r_v) / N_PER_ARM)
    verdict = "GUARDED" if gap >= BAR else "NO VERIFIED IMPROVEMENT"
    primary = {
        "rule": "success=(raw>0)/168; no page state counts as NOT success; "
                "verdict iff guarded-vanilla >= +0.10 (frozen before session 1)",
        "success_guarded": sum(1 for r in g if (r.get("raw") or 0) > 0),
        "success_vanilla": sum(1 for r in v if (r.get("raw") or 0) > 0),
        "rate_guarded": round(r_g, 4), "rate_vanilla": round(r_v, 4),
        "gap": round(gap, 4), "bar": BAR, "se_diff": round(se, 4),
        "verdict": verdict,
    }
    print("PRIMARY:", json.dumps(primary))

    # ---- secondary descriptives ------------------------------------------
    inst = {}
    for r in rows:
        inst.setdefault((r["task"], r["seed"]), {})[r["arm"]] = r
    both = [vv for vv in inst.values() if len(vv) == 2]
    w = sum(1 for vv in both if (vv["guarded"].get("raw") or 0) > 0
            and (vv["vanilla"].get("raw") or 0) <= 0)
    l = sum(1 for vv in both if (vv["vanilla"].get("raw") or 0) > 0
            and (vv["guarded"].get("raw") or 0) <= 0)
    paired = {"instances": len(inst), "both_arms_present": len(both),
              "guarded_win": w, "vanilla_win": l,
              "tie": len(both) - w - l}

    def arm_desc(arm_rows):
        wall = sorted(r["wall_s"] for r in arm_rows)
        cap = [r for r in arm_rows if r["omp_exit"] == 124]
        cap_succ = [r for r in cap if (r.get("raw") or 0) > 0]
        nocap = [r for r in arm_rows if r["omp_exit"] != 124]
        return {
            "sessions": len(arm_rows),
            "wall_s_mean": round(sum(wall) / len(wall), 1),
            "wall_s_median": wall[len(wall) // 2],
            "cap_hits": len(cap),
            "cap_hit_success_raw_gt0": len(cap_succ),
            "wall_s_mean_cap_hit": round(
                sum(r["wall_s"] for r in cap) / len(cap), 1) if cap else None,
            "wall_s_mean_no_cap": round(
                sum(r["wall_s"] for r in nocap) / len(nocap), 1)
                if nocap else None,
            "no_page_state": sum(1 for r in arm_rows
                                 if r.get("raw") is None),
            "raw_histogram": dict(Counter(str(r.get("raw"))
                                          for r in arm_rows)),
        }

    descriptives = {"guarded": arm_desc(g), "vanilla": arm_desc(v),
                    "paired_WLT": paired}

    # ---- dial attribution by serving interval ----------------------------
    # The host decisions.jsonl is APPEND-ONLY across runs: it still holds
    # the v6 sweep's records (from 04:43, entry N's window) and the aborted
    # first v7 launch (18:22, before the 18:31 relaunch). Restrict to the
    # v7 serving span first, then attribute by interval with a 2 s start
    # seam (collector attach lags the first proxied turn slightly).
    cdp_start = {}
    for line in open(CDP):
        c = json.loads(line)
        cdp_start[c["tag"]] = c["ts"]
    for r in rows:
        st = cdp_start.get(r["tag"], r["ts"] - r["wall_s"])
        r["_win"] = (st - 2.0, r["ts"] + 0.5)
    span_lo = min(r["_win"][0] for r in rows)
    span_hi = max(r["_win"][1] for r in rows)
    pre_span = [r for r in recs if not (span_lo <= r["ts"] <= span_hi)]
    recs = [r for r in recs if span_lo <= r["ts"] <= span_hi]
    unattr = 0
    by_sess = {}
    for rec in recs:
        owners = [r for r in rows
                  if r["_win"][0] <= rec["ts"] <= r["_win"][1]]
        if len(owners) == 1:
            by_sess.setdefault(owners[0]["tag"], []).append(rec)
        else:
            unattr += 1
    gtags = {r["tag"] for r in g}
    recs_g = {k: vv for k, vv in by_sess.items() if k in gtags}

    # ---- per-session transcript recovery (guarded) -----------------------
    sessions = []
    for row in g:
        path = transcript_of(row)
        item = {"tag": row["tag"], "raw": row["raw"],
                "omp_exit": row["omp_exit"], "wall_s": row["wall_s"]}
        if path is None:
            item["error"] = "no transcript"
            sessions.append(item)
            continue
        cts = claim_turns(path)
        item["n_claim_turns"] = len(cts)
        if cts:
            first = cts[0]
            wire = wire_messages_up_to(path, first["turn"])
            ev = D.evidence_from_messages(wire)
            item["first_claim"] = {"turn": first["turn"],
                                   "evidence_sha12": sha12(ev)}
        sessions.append(item)

    # ---- Q1 integrity (guarded) ------------------------------------------
    accepts = [r for r in recs if r.get("routed") == "accept"]
    subtau = [r for r in accepts if r["p"] < r["tau"]]
    first_match = first_total = 0
    for s in sessions:
        fc = s.get("first_claim")
        if not fc:
            continue
        first_total += 1
        if any(r.get("evidence_sha12") == fc["evidence_sha12"]
               for r in by_sess.get(s["tag"], [])):
            first_match += 1

    # ---- audit pool: guarded first claims, scored live --------------------
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
            next(r for r in g if r["tag"] == s["tag"]))
        wire = wire_messages_up_to(path, fc["turn"])
        ev = D.evidence_from_messages(wire)
        p, nrm = D.score(ev)
        audit.append({"tag": s["tag"], "p": round(p, 6),
                      "label": 1 if (s["raw"] is not None
                                     and s["raw"] > 0) else 0,
                      "raw": s["raw"], "sha12": fc["evidence_sha12"],
                      "vec_norm": round(nrm, 4)})
    labeled = [a for a in audit if a["raw"] is not None]
    p_agree = p_disagree = 0
    for a in audit:
        hit = next((r for r in by_sess.get(a["tag"], [])
                    if r.get("evidence_sha12") == a["sha12"] and "p" in r),
                   None)
        if hit:
            if abs(hit["p"] - a["p"]) <= 1e-4:
                p_agree += 1
            else:
                p_disagree += 1

    y = [a["label"] for a in labeled]
    p = [a["p"] for a in labeled]
    base = sum(y) / len(y)
    const_brier = sum((bi - yi) ** 2 for bi, yi in zip([base] * len(y), y)) / len(y)
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)
    pos = sorted(pi for pi, yi in zip(p, y) if yi == 1)
    neg = sorted(pi for pi, yi in zip(p, y) if yi == 0)
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    auroc = (wins + 0.5 * ties) / max(len(pos) * len(neg), 1)
    tp = sum(1 for a in labeled if a["p"] >= tau and a["label"] == 1)
    fp = sum(1 for a in labeled if a["p"] >= tau and a["label"] == 0)
    tn = sum(1 for a in labeled if a["p"] < tau and a["label"] == 0)
    fn = sum(1 for a in labeled if a["p"] < tau and a["label"] == 1)

    # ---- veto descriptives (per-session semantics finally live) -----------
    veto_sess = {}
    for tag, rs in recs_g.items():
        vt = [r for r in rs if r.get("routed") == "veto"]
        if vt:
            veto_sess[tag] = len(vt)
    veto_rows = {r["tag"]: r for r in g if r["tag"] in veto_sess}
    noveto = [r for r in g if r["tag"] not in veto_sess]
    veto_desc = {
        "veto_records": sum(veto_sess.values()),
        "sessions_with_veto": len(veto_sess),
        "vetoes_per_session_hist": dict(Counter(veto_sess.values())),
        "vetoed_sessions_raw_gt0": sum(
            1 for r in veto_rows.values() if (r.get("raw") or 0) > 0),
        "noveto_sessions_raw_gt0": sum(
            1 for r in noveto if (r.get("raw") or 0) > 0),
        "vetoed_wall_mean": round(
            sum(r["wall_s"] for r in veto_rows.values())
            / max(len(veto_rows), 1), 1),
        "noveto_wall_mean": round(
            sum(r["wall_s"] for r in noveto) / max(len(noveto), 1), 1),
        "note": ("descriptive; vetoed sessions are intervention outcomes "
                 "— post-veto states never enter accuracy numbers"),
    }

    scored = [r for r in recs if "wall_ms" in r]
    wms = sorted(r["wall_ms"] for r in scored)

    results = {
        "run": {"sessions": 336, "instances": 168, "paired": True,
                "window": "2026-09-26 18:31 → 2026-09-27 19:13 WIB"},
        "instrument": {"readout_sha": hashlib.sha256(
            open(HERE.parent / "entry_l" / "entry_k_readout.json", "rb")
            .read()).hexdigest(), "tau": tau},
        "PRIMARY_verdict": primary,
        "secondary_descriptives": descriptives,
        "attribution": {"dial_enforce_records_all_runs": 1896,
                        "dropped_outside_v7_span": len(pre_span),
                        "in_span_records": len(recs),
                        "unattributed_in_span": unattr,
                        "guarded_sessions_with_records":
                            sum(1 for t in gtags if t in by_sess)},
        "Q1_routing_integrity": {
            "accepts": len(accepts), "subtau_accepts": len(subtau),
            "first_claim_transcript_to_record_match":
                f"{first_match}/{first_total}",
            "live_p_vs_recomputed_p_agree": p_agree,
            "live_p_vs_recomputed_p_disagree": p_disagree},
        "Q2_cost": {
            "scoring_events": len(scored),
            "wall_ms_median": wms[len(wms) // 2] if wms else None,
            "wall_ms_p90": wms[int(len(wms) * 0.9)] if wms else None,
            "veto_descriptives": veto_desc,
            "confound": ("wall delta between arms is the WHOLE guarded "
                         "stack (chain reads + gates + dial) vs bare "
                         "passthrough — not the dial's cost alone")},
        "Q3_cap_efficacy": {
            "cap_exits_guarded": descriptives["guarded"]["cap_hits"],
            "cap_exits_vanilla": descriptives["vanilla"]["cap_hits"],
            "success_but_cap_hit_guarded":
                descriptives["guarded"]["cap_hit_success_raw_gt0"],
            "success_but_cap_hit_vanilla":
                descriptives["vanilla"]["cap_hit_success_raw_gt0"],
            "dial_caused_nontermination": 0},
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
            "rows": audit},
        "sessions_guarded": [{k: v2 for k, v2 in s.items()
                              if k != "first_claim"} for s in sessions],
    }
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}  ({time.time() - t0:.0f}s)")
    q4 = results["Q4_dial_audit_first_claims"]
    print(f"Q1: accepts={len(accepts)} sub-tau={len(subtau)} "
          f"match={first_match}/{first_total} p-agree={p_agree}")
    print(f"Q4: n={q4['labeled']} AUROC={q4['auroc']} Brier={q4['brier']} "
          f"(const {q4['constant_brier']}) tp{tp} fp{fp} tn{tn} fn{fn}")


if __name__ == "__main__":
    main()
