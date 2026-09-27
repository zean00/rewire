#!/usr/bin/env python3
"""Entry P embedding pass (HOST side) — PRE-REGISTERED (eval/reports/entry_p/
PROTOCOL.md, 2026-09-27). Frozen recipe: the entry-L pinned embeddings-only
server, ONE text per POST, strictly serial (the entry-K standing instrument
rule), one pass over the entry-P blocks; determinism recheck = 8 sampled rows
re-embedded singly, bit-exact, else abort.

Inputs : entry_p_states.jsonl (scp'd from the laptop; the chain of custody is
         states sha -> entry_p_build_manifest.json, verified here)
Outputs: entry_p_emb.npz (emb [N,3840] fp32, labels, split, groups)
         entry_p_embed_manifest.json
"""
import hashlib
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "gemma-4-12b-it"

STATES = "entry_p_states.jsonl"
BUILD = "entry_p_build_manifest.json"
man_b = json.load(open(BUILD))
states_sha = hashlib.sha256(open(STATES, "rb").read()).hexdigest()
assert states_sha == man_b["states_sha256"], (states_sha, man_b["states_sha256"])
rows = [json.loads(l) for l in open(STATES)]
assert len(rows) == man_b["rows"], (len(rows), man_b["rows"])
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
    E.append(embed_one(r["block"]))
    if (i + 1) % 250 == 0:
        print(f"  {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
wall = time.time() - t0

random.seed(13)
picks = sorted(random.sample(range(len(rows)), 8))
recheck = all(embed_one(rows[i]["block"]) == E[i] for i in picks)
print("recheck bit-exact:", recheck, flush=True)

import numpy as np
E = np.array(E, dtype=np.float32)
y = np.array([r["label"] for r in rows], dtype=np.int64)
sp = np.array([r["split"] for r in rows])
g = np.array([r["session_dir"] for r in rows])
assert E.shape == (len(rows), 3840), E.shape
assert set(sp) == {"train", "val"}
np.savez("entry_p_emb.npz", emb=E, labels=y, split=sp, groups=g)
man = {
    "server": "llama-server -m hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf "
              "--alias gemma-4-12b-it --host 127.0.0.1 --port 8998 --embedding "
              "--pooling last -c 32768 -np 4 -ngl 99 --flash-attn on -b 1024 -ub 512",
    "endpoint": "POST /v1/embeddings, input = entry-P composite block verbatim",
    "client": "one text per POST, strictly serial (entry-K standing instrument rule)",
    "dim": 3840, "n": int(len(rows)),
    "positives": int(y.sum()),
    "states_sha256": states_sha,
    "build_manifest_sha256": hashlib.sha256(open(BUILD, "rb").read()).hexdigest(),
    "wall_s": round(wall, 2),
    "determinism_recheck_bitexact": bool(recheck),
}
json.dump(man, open("entry_p_embed_manifest.json", "w"), indent=1)
print(json.dumps({k: man[k] for k in ("dim", "n", "positives", "wall_s",
                                      "determinism_recheck_bitexact")}, indent=1))
