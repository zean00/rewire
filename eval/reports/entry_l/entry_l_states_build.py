#!/usr/bin/env python3
"""Entry L confirmation — states build (laptop side). Frozen rule
(IMPLEMENTATION_PLAN.md, entry L launch): pool = sessions with >= 1 review
probe; per session take THE LAST probe by (turn, evidence_turn) — the
completion-claim moment; block = ckpt_rubric.build_evidence(file,
evidence_turn), the same code path the rubric reviewer read; label = the
page's own reward (raw > 0; None -> failure). NO polarity filter — the dial
reads evidence, not claim vocabulary. One pass, no reads, no GPU.

Emits entry_l_states.jsonl: {session_dir, session, turn, evidence_turn,
label, ev}.
"""
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, "/tmp")
os.chdir("/tmp/mw_bench")  # relative transcript paths, exactly as at extract time
import ckpt_rubric as R

MB = "/tmp/mw_bench"
RAW = f"{MB}/probes_v5.jsonl"
MAIN = f"{MB}/main_v5.jsonl"

files = sorted(glob.glob(f"{MB}/main_runs_v5/*/*.jsonl"))
files = [f for f in files if not f.endswith(".cdp")]
print(f"v5 transcripts: {len(files)}")
subprocess.run([sys.executable, "/tmp/ckpt_review.py", *files,
                "--out", RAW], check=True)

probes = [json.loads(l) for l in open(RAW)]
print(f"raw probes: {len(probes)}")

lab = {}
for l in open(MAIN):
    r = json.loads(l)
    lab[os.path.basename(r["session"])] = 1 if (r.get("raw") or 0) > 0 else 0

bys = {}
for p in probes:
    bys.setdefault(p["file"], []).append(p)

out, nolabel = [], 0
for f, rows in sorted(bys.items()):
    last = max(rows, key=lambda r: (r["turn"], r["evidence_turn"]))
    sd = os.path.basename(os.path.dirname(f))
    if sd not in lab:
        nolabel += 1
        continue
    out.append({"session_dir": sd, "session": last["session"],
                "turn": int(last["turn"]),
                "evidence_turn": int(last["evidence_turn"]),
                "label": lab[sd], "ev": R.build_evidence(f, last["evidence_turn"])})

pos = sum(o["label"] for o in out)
print(f"sessions_with_review: {len(out)} / {len(lab)} (no review: "
      f"{len(lab) - len(out)}) no_label: {nolabel}")
print(f"positives: {pos} base_rate: {pos / len(out):.4f} "
      f"constant_brier: {round((pos/len(out))**2*(1-pos/len(out)) + (1-pos/len(out))**2*(pos/len(out)), 4):.4f}")
with open("entry_l_states.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print("wrote entry_l_states.jsonl")
