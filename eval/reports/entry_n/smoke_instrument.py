#!/usr/bin/env python3
"""Entry-N LAUNCH SMOKE (registered: entry N, infra paragraph) — runs
against the LIVE combined host server (pinned gen line + --embeddings
--pooling last) before any enforce-mode session. Three gates, all must
pass; any failure exits nonzero and the sweep does not start:

  1. instrument identity — dim 3840, L2-normalized vectors (the readout
     dot assumes the server's default --embd-normalize);
  2. A/B/A bit-exact on a REAL state template — the same text embedded
     twice, once with a different text in between, must return
     byte-identical vectors on this stock host build (entry M taught:
     history-dependent embeddings poison everything downstream);
  3. the entry-K batch-composition defect re-test — one text alone vs the
     same text inside a two-text batch POST, cosine reported (the standing
     SERIAL rule stands regardless; this re-measures the defect on the
     ce8caa6 build);
  4. FROZEN-VECTOR PARITY — the same state texts embedded live must
     reproduce the adjudicated entry_l_emb.npz vectors (cos 1.0 exact).
     This gate exists because the first combined-server line silently
     corrupted embeddings: inheriting the gen line's -ctk/-ctv q8_0
     (quantized KV) drifted vectors to cos 0.84-0.89 from the frozen ones
     and dropped the entry-L reproduction to AUROC 0.8367. The registered
     combined line carries NO KV quantization.

Then a plumbing read: the readout scores two real v5 evidence texts
(already adjudicated — no new claims); p must land in (0, 1). Serial
client throughout: one text per POST (the one deliberate batched POST is
gate 3 itself, a defect probe, never a scoring read).

  python3 smoke_instrument.py --url http://<tailscale-ip>:8997
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True,
                help="the combined host server's embeddings base URL")
ap.add_argument("--readout",
                default=str(HERE.parent / "entry_l" / "entry_k_readout.json"))
ap.add_argument("--states",
                default=str(HERE.parent / "entry_l" / "entry_l_states.jsonl"))
ap.add_argument("--limit", type=int, default=2,
                help="real state texts to use (A/B/A + plumbing reads)")
args = ap.parse_args()

ro = json.load(open(args.readout))
W, B, A, C_, TAU = ro["w"], ro["b"], ro["platt_a"], ro["platt_c"], ro["tau"]
DIM = len(W)
_lines = open(args.states).readlines()
states = [json.loads(l)["ev"] for l in _lines][:args.limit]
other = json.loads(_lines[len(_lines) // 2])["ev"][:200]


def embed(payload_input):
    body = json.dumps({"model": "gemma-4-12b-it",
                       "input": payload_input}).encode()
    req = urllib.request.Request(args.url + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]


def one(text):
    (d,) = embed(text)
    return d["embedding"]


fails = []

# --- gate 1: identity --------------------------------------------------------
v = one(states[0])
if len(v) != DIM:
    fails.append(f"dim {len(v)} != readout {DIM}")
nrm = math.sqrt(sum(x * x for x in v))
if abs(nrm - 1.0) > 0.05:
    fails.append(f"vector not L2-normalized (norm {nrm:.4f}) — readout dot "
                 "assumes the server default --embd-normalize")
print(f"gate1 identity: dim={len(v)} norm={nrm:.6f}")

# --- gate 2: A/B/A bit-exact on a real state template -----------------------
a1 = one(states[0])
b1 = one(states[1] if len(states) > 1 else other)
a2 = one(states[0])
if a1 != a2:
    cos = sum(x * y for x, y in zip(a1, a2))
    fails.append(f"A/B/A NOT bit-exact (cos {cos:.6f}) — history-dependent "
                 "embeddings on this build; enforce reads would be poisoned")
print(f"gate2 A/B/A: bit_exact={a1 == a2}")

# --- gate 3: the entry-K batch-composition defect re-test -------------------
solo = one(states[0])
(d1, d2) = embed([states[0], other])
batched = d1["embedding"]
cos = sum(x * y for x, y in zip(solo, batched))
print(f"gate3 batch-defect: alone-vs-batched cos={cos:.6f} "
      f"(defect if << 1; the serial rule stands regardless)")

# --- plumbing: the readout scores real evidence ------------------------------
for i, text in enumerate(states):
    vec = one(text)
    z = B + sum(w * x for w, x in zip(W, vec))
    p = 1.0 / (1.0 + math.exp(-(A * z + C_)))
    if not 0.0 < p < 1.0:
        fails.append(f"plumbing read {i}: p={p} outside (0,1)")
    print(f"plumbing read {i}: chars={len(text)} p={p:.6f} "
          f"(tau={TAU:.6f}, adjudicated v5 row — plumbing only)")

# --- gate 4: frozen-vector parity (the entry-L npz is ground truth) ---------
import numpy as np  # noqa: E402  (laptop env has numpy)

Z = np.load(str(HERE.parent / "entry_l" / "entry_l_emb.npz"), allow_pickle=False)
X = Z["emb"]
worst = 1.0
for i in range(min(3, len(states), X.shape[0])):
    v = np.array(one(states[i]), dtype=np.float64)
    ref = X[i].astype(np.float64)
    c = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
    worst = min(worst, c)
    print(f"gate4 parity row {i}: cos(live, frozen)={c:.6f}")
if worst < 0.999999:
    fails.append(f"frozen-vector parity FAILED (worst cos {worst:.6f}) — "
                 "the server line is not the entry-L embedding instrument "
                 "(first suspect: any KV-cache quantization flag on the "
                 "combined line; entry-L's pinned line ran full-precision KV)")

if fails:
    for f in fails:
        print("FAIL:", f)
    sys.exit(1)
print("LAUNCH SMOKE PASSED — instrument identity, A/B/A bit-exact, "
      "readout plumbing all OK")
