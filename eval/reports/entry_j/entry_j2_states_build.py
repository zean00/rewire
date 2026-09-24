#!/usr/bin/env python3
"""Entry J Stage 2 — fresh state-text emission (laptop side).

Registered protocol (IMPLEMENTATION_PLAN.md 2026-09-24): fresh probe input
text = the pinned entry-H recipe — the exact state block the entry-E rubric
read consumed, minus the rubric question/framing sentences, built by the
SAME code path (ckpt_rubric.build_evidence), same CWD-relative transcript
paths, EVIDENCE_CAP=8000. The v3 builder additionally verified against the
entry-G scorer's recorded ev_chars; no prior scorer exists for v4, so the
recipe itself (pinned code, sha-pinned output) is the guarantee.

Emits entry_j2_states.jsonl: {session_dir, session, turn, label, ev}.
One pass, no reads, no GPU.
"""
import json
import os
import sys

sys.path.insert(0, "/tmp")
os.chdir("/tmp/mw_bench")  # relative transcript paths, exactly as at scoring time
import ckpt_rubric as R

probes = [json.loads(l) for l in open("probes_v4_balanced.jsonl")]

labels = {}
for l in open("main_v4.jsonl"):
    r = json.loads(l)
    # frozen pool rule (as _bal_shim/adjudicate): truth_ok = raw > 0; None -> failure
    labels[os.path.basename(r["session"])] = 1 if (r.get("raw") or 0) > 0 else 0

out, nolabel = [], 0
for p in probes:
    ev = R.build_evidence(p["file"], p["evidence_turn"])
    sd = os.path.basename(os.path.dirname(p["file"]))
    if sd not in labels:
        nolabel += 1
        continue
    out.append({"session_dir": sd, "session": p["session"], "turn": p["turn"],
                "label": labels[sd], "ev": ev})

print(f"probes={len(probes)} no_label={nolabel} emitted={len(out)}")
print("passed-session probes:", sum(o["label"] for o in out), "/", len(out))
print("passed sessions:", len(set(o["session_dir"] for o in out if o["label"])),
      "of", len(set(o["session_dir"] for o in out)))
with open("entry_j2_states.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
