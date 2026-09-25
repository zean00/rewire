#!/usr/bin/env python3
"""Entry-N port validation — BYTE-PARITY smoke (laptop, no GPU, no server).

The enforce dial must rebuild each claim's evidence EXACTLY as
ckpt_rubric.build_evidence built it for the adjudicated pools: the frozen
readout was validated on that distribution, and any drift in the evidence
text shifts the embedding distribution silently. This smoke replays every
v5 transcript, derives the wire-shape messages the live proxy would see at
each session's completion-claim moment (the LAST probe by (turn,
evidence_turn), the entry-L pool rule), and asserts:

  1. evidence parity — dial_enforce.evidence_from_messages(wire_msgs)
     == ckpt_rubric.build_evidence(file, evidence_turn), byte for byte;
  2. jurisdiction parity — the probe's own assistant prose passes
     dial_enforce.is_completion_claim (the frozen ckpt_review rule), and
     dial_enforce.prose_of reproduces ckpt_review's prose construction.

Any assertion failure exits nonzero — the enforce port is not faithful
and must not serve.

  python3 eval/reports/entry_n/smoke_parity.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp")
import ckpt_review as C          # noqa: E402  (the frozen probe rule)
import ckpt_rubric as R          # noqa: E402  (the frozen evidence rule)

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "dial_enforce", ROOT / "proxy" / "dial_enforce.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

MB = "/tmp/mw_bench"
files = sorted(f for f in __import__("glob").glob(f"{MB}/main_runs_v5/*/*.jsonl")
               if not f.endswith(".cdp"))
print(f"v5 transcripts: {len(files)}")

n_sessions = 0
n_claims = 0
n_ev_mismatch = 0
n_prose_mismatch = 0
n_no_prose = 0

for f in files:
    probes = C.extract(f)
    if not probes:
        continue
    n_sessions += 1
    last = max(probes, key=lambda p: (p["turn"], p["evidence_turn"]))

    # the wire messages the proxy held at the claim turn: every message
    # event up to and including the probe's turn, toolResult -> tool
    wire = []
    blks_at_turn = None
    for i, role, blks in C.turns(f):
        if i > last["turn"]:
            break
        r = "tool" if role == "toolResult" else role
        wire.append({"role": r, "content": blks})
        if i == last["turn"] and role == "assistant":
            blks_at_turn = blks

    if blks_at_turn is None:
        n_no_prose += 1
        continue
    n_claims += 1

    # (1) evidence parity, byte for byte
    live_ev = D.evidence_from_messages(wire)
    ref_ev = R.build_evidence(f, last["evidence_turn"])
    if live_ev != ref_ev:
        n_ev_mismatch += 1
        print(f"EV MISMATCH {f}\n  live={live_ev[:120]!r}\n  ref ={ref_ev[:120]!r}")

    # (2) jurisdiction parity — prose_of on the REAL wire blocks must
    # reproduce ckpt_review's prose construction, and the prose must pass
    # the frozen claim rule (the probe existed, so it must)
    live_prose = D.prose_of(blks_at_turn)
    texts = [str(b.get("text", "")) for b in blks_at_turn
             if isinstance(b, dict) and b.get("type") == "text"]
    ref_prose = "\n".join(t for t in texts if t.strip())
    if live_prose != ref_prose or not D.is_completion_claim(live_prose):
        n_prose_mismatch += 1
        print(f"PROSE/JURISDICTION MISMATCH {f} turn={last['turn']}\n"
              f"  live={live_prose[:120]!r}\n  ref ={ref_prose[:120]!r}")

print(f"sessions with claims: {n_sessions} / {len(files)} "
      f"(claim-free: {len(files) - n_sessions})")
print(f"claim events checked: {n_claims} (no assistant prose: {n_no_prose})")
print(f"evidence mismatches:  {n_ev_mismatch}")
print(f"prose/jurisdiction mismatches: {n_prose_mismatch}")
if n_ev_mismatch or n_prose_mismatch or n_no_prose:
    print("PARITY SMOKE FAILED — the enforce port is not faithful")
    sys.exit(1)
print("PARITY SMOKE PASSED — live evidence builder is byte-faithful")
