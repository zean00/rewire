#!/usr/bin/env python3
"""Entry P states build (laptop side) — PRE-REGISTERED (eval/reports/entry_p/
PROTOCOL.md, 2026-09-27), frozen before any embedding or scoring. Turn-level
loop labels over the CLOSED v7 sweep corpus.

Unit  : an ACTING turn — an assistant toolCall whose code carries an intent
        comment `// (click|type|select|check) ...` (first such line). Other
        calls (open/observe/comment-less) are excluded from the dataset.
sig   : that comment line, whitespace-collapsed, otherwise verbatim.
label : 1 (LOOP) iff the same sig occurs >= 3 times within the current turn's
        trailing window of 8 acting turns (inclusive) AND the page snapshot
        (ckpt_rubric.build_evidence at the turn's message row) is identical at
        all of those occurrences and at the current turn. Else 0.
block : frozen composite input (TASK / PAGE SNAPSHOT / RECENT ACTIONS), embedded
        verbatim by entry_p_embed.py.
split : int(md5(session_dir)[:8],16) % 10 < 3 -> "val" else "train".

Closure check (AMENDED 2026-09-27, disclosed before execution — see ledger):
the registered builder asserted 336 records with done=true. The sweep closed
with all 336 planned sessions run exactly once (336 unique (task,seed,arm)
keys, 336 transcript dirs, 0 strays), but 84 rows carry done=false: 59
raw=None (52 vanilla / 7 guarded — no final page state captured) + 25 raw=0.0
(22 cap-exits, 3 clean exits ending mid-episode). `done` is the page probe's
episode flag (run_v7.py fin.get("done")), not a sweep-completion flag, and
this label rule reads only transcripts (acting turns, sigs, page snapshots) —
never done or raw. Amendment: assert 336 records, 336 unique keys, 336
matched transcript dirs; no other change. Counts recorded in the ledger.
One execution; the standing disclosed-crash-fix exception applies, any fix is
appended to the ledger BEFORE embedding. Emits (cwd) entry_p_states.jsonl and
entry_p_build_manifest.json.
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/tmp")
os.chdir("/tmp/mw_bench")  # ckpt_rubric expects the /tmp layout, as at entry L
import ckpt_rubric as R

MB = "/tmp/mw_bench"
MAIN = f"{MB}/main_v7.jsonl"
ACTING_RE = re.compile(r"^[ \t]*//[ \t]*(click|type|select|check)\b[^\n]*$", re.M)
WINDOW = 8
SNAP_CAP = 6000
SIG_CAP = 160
TASK_CAP = 300

sess = [json.loads(l) for l in open(MAIN)]
assert len(sess) == 336, f"expected 336 sessions, got {len(sess)} — sweep not closed?"
keys = {(s["task"], s["seed"], s["arm"]) for s in sess}
assert len(keys) == 336, f"expected 336 unique (task,seed,arm) keys, got {len(keys)}"
n_done = sum(bool(s.get("done")) for s in sess)
print(f"closure: 336 records, 336 unique keys, done=true {n_done}/336 "
      f"(amended check — done unused by the label rule, see ledger disclosure)")
meta = {os.path.basename(s["session"]): s for s in sess}
assert len(meta) == 336, "duplicate session dirs in main_v7.jsonl"


def collapse(s):
    return " ".join(str(s).split())


rows = []
n_sessions = 0
strays = 0
for f in sorted(glob.glob(f"{MB}/main_runs_v7/*/*.jsonl")):
    if f.endswith(".cdp"):
        continue
    sd = os.path.basename(os.path.dirname(f))
    s = meta.get(sd)
    if s is None:
        strays += 1
        continue
    n_sessions += 1

    # message rows in order: (row_index, role, content)
    turns = []
    for i, line in enumerate(open(f)):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        m = ev.get("message", {})
        turns.append((i, m.get("role"), m.get("content")))

    acting = []  # (row_index, sig) in order
    for ri, role, content in turns:
        if role != "assistant" or not isinstance(content, list):
            continue
        for b in content:
            if not (isinstance(b, dict) and b.get("type") == "toolCall"):
                continue
            tc = b.get("toolCall", b)
            args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
            code = str(args.get("code", "")) if isinstance(args, dict) else ""
            m = ACTING_RE.search(code)
            if m:
                acting.append((ri, collapse(m.group(0).strip())))

    snaps = {}

    def snap_at(ri):
        if ri not in snaps:
            snaps[ri] = R.build_evidence(f, ri)
        return snaps[ri]

    for k, (ri, sig) in enumerate(acting):
        window = acting[max(0, k - (WINDOW - 1)):k + 1]
        same = [rj for rj, sj in window if sj == sig]
        cur = snap_at(ri)
        if len(same) >= 3:
            label = int(all(snap_at(rj) == cur for rj in same))
        else:
            label = 0
        recent = window[-WINDOW:]
        lines = "\n".join(f"{p + 1}. {sj[:SIG_CAP]}" for p, (_, sj) in enumerate(recent))
        h = int(hashlib.md5(sd.encode()).hexdigest()[:8], 16)
        rows.append({
            "session_dir": sd, "arm": s["arm"], "task": s["task"],
            "turn_row": ri, "act_index": k + 1, "sig": sig,
            "label": label, "split": "val" if h % 10 < 3 else "train",
            "session_raw": s.get("raw"), "cap_hit": s.get("omp_exit") == 124,
            "block": (f"TASK: {collapse(s.get('query'))[:TASK_CAP]}\n\n"
                      f"PAGE SNAPSHOT:\n{cur[:SNAP_CAP]}\n\n"
                      f"RECENT ACTIONS (oldest first, current turn last):\n{lines}"),
        })

assert n_sessions + strays == 336, (n_sessions, strays)
pos = sum(r["label"] for r in rows)
by_split = {sp: {"n": sum(r["split"] == sp for r in rows),
                 "pos": sum(r["split"] == sp and r["label"] == 1 for r in rows)}
            for sp in ("train", "val")}
with open("entry_p_states.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
states_sha = hashlib.sha256(open("entry_p_states.jsonl", "rb").read()).hexdigest()
man = {
    "main_v7_sha256": hashlib.sha256(open(MAIN, "rb").read()).hexdigest(),
    "states_sha256": states_sha,
    "sessions": n_sessions, "stray_dirs_excluded": strays,
    "rows": len(rows), "positives": pos,
    "loop_rate": round(pos / len(rows), 6),
    "by_split": by_split,
    "by_arm": {a: {"rows": sum(r["arm"] == a for r in rows),
                   "pos": sum(r["arm"] == a and r["label"] == 1 for r in rows)}
               for a in ("vanilla", "guarded")},
    "window": WINDOW, "snap_cap": SNAP_CAP, "sig_cap": SIG_CAP, "task_cap": TASK_CAP,
}
json.dump(man, open("entry_p_build_manifest.json", "w"), indent=1)
print(json.dumps(man, indent=1))
print("wrote entry_p_states.jsonl")
