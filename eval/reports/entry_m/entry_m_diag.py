#!/usr/bin/env python3
"""Entry M pass-1 recheck-failure diagnosis.

Q: the pass-1 client was already singles/serial — why did 8-row re-embeds
   differ bit-wise? Measure the DIFFERENCES: bit-equality, max|diff|,
   cosine, and whether a second re-embed of the same text is stable NOW.
"""
import json
import random
import urllib.request

import numpy as np

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "qwen3-8b-q4"

v3 = [json.loads(l) for l in open("entry_h_states.jsonl")]
v4 = [json.loads(l) for l in open("entry_j2_states.jsonl")]

z = np.load("entry_m_emb_pass1.npz")
E3 = z["v3_emb"]
E4 = z["v4_emb"]
pool = np.concatenate([E3, E4])


def embed_one(text):
    body = json.dumps({"input": [text], "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["data"][0]["embedding"]


random.seed(13)
picks = random.sample(range(316), 4) + [316 + i for i in sorted(random.sample(range(290), 4))]
for i in picks:
    text = (v3 + v4)[i]["ev"]
    r1 = embed_one(text)
    r2 = embed_one(text)
    v = pool[i]
    r1n, r2n = np.array(r1), np.array(r2)
    bit12 = r1 == r2
    bit_saved1 = r1n.tobytes() == v.tobytes()
    cos = float(np.dot(v, r1n) / (np.linalg.norm(v) * np.linalg.norm(r1n)))
    md = float(np.abs(v - r1n).max())
    print(f"row {i:3d}  re1==re2 {bit12}  saved==re1 {bit_saved1}  "
          f"cos {cos:.9f}  max|d| {md:.3e}")
