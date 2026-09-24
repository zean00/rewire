#!/usr/bin/env python3
"""Entry H recipe verification + state-text emission (laptop side).

Registered protocol (IMPLEMENTATION_PLAN.md 2026-09-23): probe input text =
the exact state block the entry-E rubric read consumed, minus the rubric
question/framing sentences. This script reconstructs that block using the
SAME code path as the scorer (ckpt_rubric.build_evidence), the same
CWD-relative transcript paths, and verifies each probe against the
ev_chars recorded in scores_v3_par.jsonl (the scorer wrote len(ev) per row).

Emits entry_h_states.jsonl: {session_dir, session, turn, label, ev}.
One pass, no reads, no GPU.
"""
import json
import os
import sys

sys.path.insert(0, "/tmp")
os.chdir("/tmp/mw_bench")  # relative transcript paths, exactly as at scoring time
import ckpt_rubric as R

probes = [json.loads(l) for l in open("probes_v3_balanced.jsonl")]
rows = [json.loads(l) for l in open("scores_v3_par.jsonl")]
assert len(probes) == len(rows) == 316, (len(probes), len(rows))

labels = {}
for l in open("main_v3.jsonl"):
    r = json.loads(l)
    # frozen pool rule (as _bal_shim/adjudicate): truth_ok = raw > 0; None -> failure
    labels[os.path.basename(r["session"])] = 1 if (r.get("raw") or 0) > 0 else 0

out, mism, nolabel = [], 0, 0
for p, row in zip(probes, rows):
    assert p["session"] == row["session"] and p["turn"] == row["turn"]
    if "error" in row:
        mism += 1
        continue
    ev = R.build_evidence(p["file"], p["evidence_turn"])
    if len(ev) != row.get("ev_chars", -1):
        mism += 1
        continue
    sd = os.path.basename(os.path.dirname(p["file"]))
    if sd not in labels:
        nolabel += 1
        continue
    out.append({"session_dir": sd, "session": p["session"], "turn": p["turn"],
                "label": labels[sd], "ev": ev})

print(f"probes={len(probes)} ev_len_mismatch={mism} no_label={nolabel} emitted={len(out)}")
print("passed-session probes:", sum(o["label"] for o in out), "/", len(out))
print("passed sessions:", len(set(o["session_dir"] for o in out if o["label"])),
      "of", len(set(o["session_dir"] for o in out)))
with open("entry_h_states.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
