#!/usr/bin/env python3
"""Entry K extraction — RE-RUN under the pre-committed crash-fix exception
(ledger 2026-09-25), disclosed BEFORE any metric exists.

Pass 1 (registered client: chunks of <=8, 4 workers) failed its determinism
recheck. Diagnosis (entry_k_diag.py / entry_k_diag2.py, host): multi-input
/v1/embeddings requests on this server build return position/neighbor-
dependent vectors (8 copies of one text in a single request -> 8 different
vectors; same text alone vs batched -> cos 0.87), so the registered client
never measured a well-defined instrument. Singles reproduce bit-exactly.

Fix (client only; server line unchanged): ONE text per POST, serial, one
pass over all 606 texts. Recheck: the 8 sampled rows re-embedded singly,
bit-exact vs the saved vectors. Exactly one re-run; its result stands.
"""
import hashlib
import json
import time
import urllib.request

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "gemma-4-12b-it"
V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
V4_STATES_SHA = "4aa4538765304851dcb1c9595acb937b52cc20b21574774c1177ca469f63d233"

for fname, want in (("entry_h_states.jsonl", V3_STATES_SHA),
                    ("entry_j2_states.jsonl", V4_STATES_SHA)):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)


def embed_one(text):
    body = json.dumps({"input": [text], "model": MODEL}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["data"][0]["embedding"]


def embed_all(texts):
    out = []
    t0 = time.time()
    for i, t in enumerate(texts):
        out.append(embed_one(t))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(texts)} {time.time() - t0:.0f}s",
                  flush=True)
    return out


v3 = [json.loads(l) for l in open("entry_h_states.jsonl")]
v4 = [json.loads(l) for l in open("entry_j2_states.jsonl")]
assert len(v3) == 316 and len(v4) == 290

t0 = time.time()
E3 = embed_all([r["ev"] for r in v3])
E4 = embed_all([r["ev"] for r in v4])
wall = time.time() - t0

# determinism recheck: the 8 sampled rows re-embedded singly, bit-exact
import random
random.seed(13)
picks = random.sample(range(316), 4) + [316 + i for i in sorted(random.sample(range(290), 4))]
pool = E3 + E4
recheck = all(embed_one((v3 + v4)[i]["ev"]) == pool[i] for i in picks)
print("recheck bit-exact:", recheck, flush=True)

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
    "client": ("crash-fix re-run: one text per POST, serial — pass-1 client "
               "(chunks of <=8, 4 workers) measured neighbor-dependent vectors "
               "(see entry_k_diag*.py); server line unchanged"),
    "supersedes_pass1": True,
    "dim": int(E3.shape[1]),
    "n_v3": int(len(y3)), "n_v4": int(len(y4)),
    "v3_states_sha256": V3_STATES_SHA, "v4_states_sha256": V4_STATES_SHA,
    "embd_normalize": "server default (euclidean/L2)",
    "wall_s": round(wall, 2),
    "determinism_recheck_bitexact": bool(recheck),
}
json.dump(manifest, open("entry_k_embed_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
