#!/usr/bin/env python3
"""Entry K recheck-failure diagnosis (host, no metrics involved).

Pass 1's recheck re-embedded 8 sampled rows as ONE quiet batch of 8 and
compared against vectors saved from 4-way-concurrent chunked posts -> false.
This script isolates the cause:
  1. response "index" ordering (a scramble would corrupt the main pass too)
  2. identical-batch back-to-back determinism
  3. batch-of-8 vs one-at-a-time composition effect (magnitude)
  4. saved pass-1 vectors vs fresh re-embeddings (reproduces the mismatch)
"""
import json
import random
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


def maxdiff(u, v):
    return max(abs(x - y) for x, y in zip(u, v))


v3 = [json.loads(l) for l in open("entry_h_states.jsonl")]
v4 = [json.loads(l) for l in open("entry_j2_states.jsonl")]
allrows = v3 + v4

# exactly the pass-1 sampling procedure
random.seed(13)
picks = random.sample(range(316), 4) + \
    [316 + i for i in sorted(random.sample(range(290), 4))]
texts = [allrows[i]["ev"] for i in picks]
print("picks:", picks)

d = embed(texts)
print("response index fields:", [it["index"] for it in d["data"]])
A = [it["embedding"] for it in d["data"]]

B = [it["embedding"] for it in embed(texts)["data"]]
print("identical-batch rerun bit-exact:",
      all(a == b for a, b in zip(A, B)))

C = [embed([t])["data"][0]["embedding"] for t in texts]
print("batch8-vs-single max|diff|:",
      [f"{maxdiff(a, c):.3e}" for a, c in zip(A, C)])

z = np.load("entry_k_emb.npz")
E = np.concatenate([z["v3_emb"], z["v4_emb"]])
saved = [E[i] for i in picks]
print("saved-vs-batch8 (pass-1 recheck comparison) max|diff|:",
      [f"{maxdiff(s, a):.3e}" for s, a in zip(saved, A)])
print("saved-vs-single max|diff|:",
      [f"{maxdiff(s, c):.3e}" for s, c in zip(saved, C)])
