#!/usr/bin/env python3
"""Entry L confirmation — embedding pass (HOST side). Frozen recipe: the
pinned embeddings-only server line, ONE text per POST, strictly serial (the
entry-K standing instrument rule), one pass over the fresh completion-claim
blocks; determinism recheck = 8 sampled rows re-embedded singly, bit-exact.

Inputs : entry_l_states.jsonl (scp'd from the laptop; its sha256 is recorded
         in the manifest — the chain of custody the confirm step verifies)
Outputs: entry_l_emb.npz (emb [N,3840] fp32, labels, groups)
         entry_l_embed_manifest.json
"""
import hashlib
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "gemma-4-12b-it"

STATES = "entry_l_states.jsonl"
states_sha = hashlib.sha256(open(STATES, "rb").read()).hexdigest()
rows = [json.loads(l) for l in open(STATES)]
print(f"states: {len(rows)} sha {states_sha[:16]}…", flush=True)


def embed_one(text):
    body = json.dumps({"input": [text], "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["data"][0]["embedding"]


t0 = time.time()
E = []
for i, r in enumerate(rows):
    E.append(embed_one(r["ev"]))
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
wall = time.time() - t0

random.seed(13)
picks = sorted(random.sample(range(len(rows)), 8))
recheck = all(embed_one(rows[i]["ev"]) == E[i] for i in picks)
print("recheck bit-exact:", recheck, flush=True)

import numpy as np
E = np.array(E, dtype=np.float32)
y = np.array([r["label"] for r in rows], dtype=np.int64)
g = np.array([r["session_dir"] for r in rows])
np.savez("entry_l_emb.npz", emb=E, labels=y, groups=g)
manifest = {
    "server": ("llama-server -m hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf "
               "--alias gemma-4-12b-it --host 127.0.0.1 --port 8998 --embedding "
               "--pooling last -c 32768 -np 4 -ngl 99 --flash-attn on -b 1024 -ub 512"),
    "endpoint": "POST /v1/embeddings, input = raw completion-claim state block verbatim",
    "client": "one text per POST, strictly serial (entry-K standing instrument rule)",
    "dim": int(E.shape[1]),
    "n": int(len(rows)),
    "positives": int(y.sum()),
    "states_sha256": states_sha,
    "embd_normalize": "server default (euclidean/L2)",
    "wall_s": round(wall, 2),
    "determinism_recheck_bitexact": bool(recheck),
}
json.dump(manifest, open("entry_l_embed_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
