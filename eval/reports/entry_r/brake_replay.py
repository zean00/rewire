#!/usr/bin/env python3
"""Entry R — ZERO-COST brake replay over the CLOSED v7 guarded transcripts
(PRE-REGISTERED descriptive, IMPLEMENTATION_PLAN.md 2026-09-28).

Applies the frozen brake rule to the wire state each request would have
presented, in transcript form: at every assistant row (a request), the
acting-turn sequence strictly before it, each acting turn's snapshot being
the newest non-blank tool result before it (ckpt_rubric.build_evidence
semantics, computed in one pass per file with the same ckpt_review
primitives), and the current request's snapshot being that same trailing
value. First request where the rule fires = the brake point; the brake
would have ended the episode there.

Descriptive only — no serving, no new numbers for the frozen bars. Runs
BEFORE the sweep as the rule's sanity check on real data.

Parity gate: the acting-turn extraction must find exactly the entry-P
corpus's guarded acting rows (837) — a mismatch aborts before any summary.
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/tmp")
os.chdir("/tmp/mw_bench")
import ckpt_rubric as R  # noqa: E402  (expects the /tmp layout, as at entry L)

C = R.C
MB = "/tmp/mw_bench"
MAIN = f"{MB}/main_v7.jsonl"
ACTING_RE = re.compile(r"^[ \t]*//[ \t]*(click|type|select|check)\b[^\n]*$", re.M)
WINDOW = 8
MIN_REPEATS = 3
SNAP_CAP = 8000
GUARDED_ACTING_EXPECTED = 837  # entry_p_states.jsonl, arm == "guarded"


def collapse(s):
    return " ".join(str(s).split())


rows_out = []
n_acting_total = 0
guarded_dirs = set()
for line in open(MAIN):
    r = json.loads(line)
    if r["arm"] == "guarded":
        guarded_dirs.add(os.path.basename(r["session"]))
assert len(guarded_dirs) == 168, f"expected 168 guarded dirs, got {len(guarded_dirs)}"
for f in sorted(glob.glob(f"{MB}/main_runs_v7/*/*.jsonl")):
    if f.endswith(".cdp"):
        continue
    sd = os.path.basename(os.path.dirname(f))
    if sd not in guarded_dirs:
        continue

    # one pass: roles + timestamps per row (message.timestamp is epoch ms)
    ts_at = {}
    for i, line in enumerate(open(f)):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        ts = (ev.get("message") or {}).get("timestamp")
        ts_at[i] = float(ts) if isinstance(ts, (int, float)) else None

    # one pass over the same rows via the corpus's own primitives
    seq = []            # (row, sig, snapshot) acting turns in order
    cur_at = {}         # row -> newest non-blank toolResult text before it
    last = ""
    for ri, role, blks in C.turns(f):
        if role == "toolResult":
            t = C.newest_result_text(blks)
            if t.strip():
                last = " ".join(t.split())[:SNAP_CAP]
        cur_at[ri] = last
        if role == "assistant" and isinstance(blks, list):
            for b in blks:
                if not (isinstance(b, dict) and b.get("type") == "toolCall"):
                    continue
                tc = b.get("toolCall", b)
                args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                code = str(args.get("code", "")) if isinstance(args, dict) else ""
                m = ACTING_RE.search(code)
                if m:
                    # one entry PER matching block, the corpus's unit (one
                    # v7 guarded message carried two intent-comment calls)
                    seq.append((ri, collapse(m.group(0).strip()), last))
    n_acting_total += len(seq)

    main_row = None
    for line in open(MAIN):
        r = json.loads(line)
        if os.path.basename(r["session"]) == sd and r["arm"] == "guarded":
            main_row = r
            break
    assert main_row is not None, f"no guarded main_v7 row for {sd}"
    ts0 = next((t for t in (ts_at[i] for i in sorted(ts_at)) if t), None)

    # fire scan: first request (assistant row) where the rule fires
    fired = None
    for j in sorted(cur_at):
        prior = [(k, sg, sn) for k, sg, sn in seq if k < j]
        if not prior:
            continue
        window = prior[-WINDOW:]
        by_sig = {}
        for i, (_, sg, _sn) in enumerate(window):
            by_sig.setdefault(sg, []).append(i)
        cur = cur_at[j]
        if not cur:
            continue
        for sg, idxs in by_sig.items():
            if len(idxs) < MIN_REPEATS:
                continue
            snaps = {window[i][2] for i in idxs}
            if len(snaps) == 1 and snaps.pop() == cur:
                fired = (j, sg, len(idxs), len(prior))
                break
        if fired:
            break

    tj = ts_at[fired[0]] if fired else None
    rows_out.append({
        "session": sd, "task": main_row["task"],
        "raw": main_row.get("raw"), "wall_s": main_row.get("wall_s"),
        "acting_n": len(seq),
        "braked": fired is not None,
        "fire_row": fired[0] if fired else None,
        "fire_sig": fired[1][:160] if fired else None,
        "fire_occurrences": fired[2] if fired else None,
        "fire_acting_before": fired[3] if fired else None,
        "wall_at_fire_s": (round((tj - ts0) / 1000, 1)
                           if fired and tj and ts0 else None),
    })

assert n_acting_total == GUARDED_ACTING_EXPECTED, (
    f"acting-turn parity FAILED: replay {n_acting_total} != corpus "
    f"{GUARDED_ACTING_EXPECTED} — do not trust the replay, investigate")

n = len(rows_out)
braked = [r for r in rows_out if r["braked"]]
succ = [r for r in rows_out if isinstance(r["raw"], (int, float)) and r["raw"] > 0]
killed = [r for r in braked if isinstance(r["raw"], (int, float)) and r["raw"] > 0]
walls = [r["wall_s"] for r in rows_out if isinstance(r["wall_s"], (int, float))]
fire_walls = [r["wall_at_fire_s"] for r in braked
              if isinstance(r["wall_at_fire_s"], (int, float))]


def mean(x):
    return round(sum(x) / len(x), 1) if x else None


summary = {
    "sessions": n,
    "acting_total": n_acting_total,
    "parity_vs_corpus": GUARDED_ACTING_EXPECTED,
    "braked_sessions": len(braked),
    "brake_rate": round(len(braked) / n, 4),
    "braked_session_success": sum(1 for r in braked
                                  if isinstance(r["raw"], (int, float))
                                  and r["raw"] > 0),
    "would_kill_success": len(killed),
    "wall_mean_all_s": mean(walls),
    "wall_at_fire_mean_s": mean(fire_walls),
    "counterfactual_wall_mean_s": mean(
        [min(r["wall_s"], r["wall_at_fire_s"])
         if r["braked"] and isinstance(r["wall_at_fire_s"], (int, float))
         and isinstance(r["wall_s"], (int, float)) else r["wall_s"]
         for r in rows_out]),
    "fire_positions_acting_before": sorted(r["fire_acting_before"]
                                           for r in braked),
}
states_sha = hashlib.sha256(
    json.dumps(rows_out, sort_keys=True).encode()).hexdigest()[:16]
out = {"summary": summary, "sessions": rows_out, "rows_sha16": states_sha}
with open("brake_replay.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(summary, indent=1))
print("wrote brake_replay.json")
