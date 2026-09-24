#!/usr/bin/env python3
"""Entry K diagnosis round 2: WHY does batch composition change vectors?

Tests: (a) L2 norms of batch vs single outputs -> normalization bug?
       (b) cosine between them -> direction change or scale change?
       (c) 8x identical text in one batch vs that text alone
       (d) short+long pair batched vs alone (padding effect)
"""
import json
import urllib.request

import numpy as np

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "gemma-4-12b-it"


def embed(texts):
    body = json.dumps({"input": texts, "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


v3 = [json.loads(l) for l in open("entry_h_states.jsonl")]
t = v3[132]["ev"]  # pick[0] from round 1

A = np.array(embed([t])["data"][0]["embedding"], dtype=np.float64)
B = np.array([it["embedding"] for it in embed([t] * 8)["data"]], dtype=np.float64)
C = np.array([it["embedding"] for it in embed([t, v3[0]["ev"]])["data"]][0],
             dtype=np.float64)

print("norm(single):", np.linalg.norm(A))
print("norm(8x same, each):", [round(float(np.linalg.norm(b)), 6) for b in B])
print("norm(pair-with-other):", np.linalg.norm(C))
print("8x-same all equal to each other:", all((B[i] == B[0]).all() for i in range(8)))
print("8x-same == single:", bool((B[0] == A).all()))
print("pair == single:", bool((C == A).all()))
if not (C == A).all():
    cos = float(A @ C / (np.linalg.norm(A) * np.linalg.norm(C)))
    print("cos(single, pair):", round(cos, 6),
          " max|diff|:", f"{np.abs(A - C).max():.3e}")

# same-text-different-length padding probe: t vs a very short other text
short = "hello"
S1 = np.array(embed([t])["data"][0]["embedding"], dtype=np.float64)
S2 = np.array(embed([short])["data"][0]["embedding"], dtype=np.float64)
P = embed([t, short])["data"]
Pt = np.array(P[0]["embedding"], dtype=np.float64)
print("pair(t,short): t == its single?", bool((Pt == S1).all()))
cos = float(S1 @ Pt / (np.linalg.norm(S1) * np.linalg.norm(Pt)))
print("  cos:", round(cos, 6), " max|diff|:", f"{np.abs(S1 - Pt).max():.3e}")
