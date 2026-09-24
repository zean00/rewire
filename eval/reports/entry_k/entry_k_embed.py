#!/usr/bin/env python3
"""Entry K extraction — frozen recipe (ledger 2026-09-25). ONE pass.

Posts the state texts verbatim (no template, no wrapping) to the stock
llama-server embeddings endpoint (--embedding --pooling last, L2-normalized
by server default). Chunks of <=8 texts, 4 concurrent workers. One pass over
316 v3 + 290 v4 states; determinism recheck = re-embed 8 sampled rows,
require bit-exact equality. Saves entry_k_emb.npz + manifest.
"""
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "gemma-4-12b-it"
V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
V4_STATES_SHA = "4aa4538765304851dcb1c9595acb937b52cc20b21574774c1177ca469f63d233"

for fname, want in (("entry_h_states.jsonl", V3_STATES_SHA),
                    ("entry_j2_states.jsonl", V4_STATES_SHA)):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)


def load(fname):
    rows = [json.loads(l) for l in open(fname)]
    return rows


def embed_batch(texts):
    body = json.dumps({"input": texts, "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return [np_row["embedding"] for np_row in d["data"]]


def embed_all(texts):
    chunks = [texts[i:i + 8] for i in range(0, len(texts), 8)]
    out = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(embed_batch, c): i for i, c in enumerate(chunks)}
        for f, i in futs.items():
            out[i] = f.result()
    return [e for c in out for e in c]


v3 = load("entry_h_states.jsonl")
v4 = load("entry_j2_states.jsonl")
assert len(v3) == 316 and len(v4) == 290

t0 = time.time()
E3 = embed_all([r["ev"] for r in v3])
E4 = embed_all([r["ev"] for r in v4])
wall = time.time() - t0

# determinism recheck: 8 sampled rows re-embedded, bit-exact
import random
random.seed(13)
picks = random.sample(range(316), 4) + [316 + i for i in sorted(random.sample(range(290), 4))]
pool = E3 + E4
re_rows = [[r["ev"] for r in (v3 + v4)][i] for i in picks]
re_emb = embed_batch(re_rows)
recheck = all(a == b for a, b in zip([pool[i] for i in picks], re_emb))

import numpy as np
E3 = np.array(E3, dtype=np.float32)
E4 = np.array(E4, dtype=np.float32)
y3 = np.array([r["label"] for r in v3], dtype=np.int64)
y4 = np.array([r["label"] for r in v4], dtype=np.int64)
g3 = np.array([r["session_dir"] for r in v3])
np.savez("entry_k_emb.npz", v3_emb=E3, v4_emb=E4, labels_v3=y3, labels_v4=y4,
         groups_v3=g3)
manifest = {
    "server": ("llama-server -m hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf "
               "--alias gemma-4-12b-it --host 127.0.0.1 --port 8998 --embedding "
               "--pooling last -c 32768 -np 4 -ngl 99 --flash-attn on -b 1024 -ub 512"),
    "endpoint": "POST /v1/embeddings, input = raw state text verbatim",
    "dim": int(E3.shape[1]),
    "n_v3": int(len(y3)), "n_v4": int(len(y4)),
    "v3_states_sha256": V3_STATES_SHA, "v4_states_sha256": V4_STATES_SHA,
    "embd_normalize": "server default (euclidean/L2)",
    "wall_s": round(wall, 2),
    "determinism_recheck_bitexact": bool(recheck),
}
json.dump(manifest, open("entry_k_embed_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
