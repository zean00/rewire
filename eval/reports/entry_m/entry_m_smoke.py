#!/usr/bin/env python3
"""Entry M smoke — throwaway strings ONLY (no state text touched).

Verifies the local build's embeddings endpoint before the registered pass:
  1. single-string embed: dim + unit norm
  2. entry-K batch-defect spot check on THIS build: same text alone vs
     inside a two-text batch (cosine; informative disclosure, the pass
     itself is singles-only regardless of the result)
"""
import json
import urllib.request

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "qwen3-8b-q4"


def emb(texts):
    body = json.dumps({"input": texts, "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return [it["embedding"] for it in d["data"]]


t1 = "The conveyor belt moves the box to the left side of the table."
t2 = "A completely different sentence about red buttons and forms."

a1 = emb([t1])[0]
batch = emb([t1, t2])
n = sum(x * x for x in a1) ** 0.5
cos = sum(x * y for x, y in zip(a1, batch[0])) / (
    sum(x * x for x in a1) ** 0.5 * sum(y * y for y in batch[0]) ** 0.5)
print("dim:", len(a1))
print("single norm:", round(n, 6))
print("alone-vs-batched cos:", round(cos, 6))
assert len(a1) == 4096, "expected Qwen3-8B hidden size 4096"
assert abs(n - 1.0) < 1e-3, "expected L2-normalized output"
print("SMOKE_OK")
