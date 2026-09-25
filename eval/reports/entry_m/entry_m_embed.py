#!/usr/bin/env python3
"""Entry M extraction — Qwen3-8B Q4_K_M on the local RTX 5060 (8 GB).

Pre-registered (ledger 2026-09-25, commit 532ce1a) BEFORE any state text is
embedded. One pass over all 606 texts; client = ONE text per POST, strictly
serial (the entry-K standing instrument rule). Recheck: 8 sampled rows
re-embedded singly, bit-exact required. Smoke (throwaway strings only, run
separately) must have verified dim/norm and the batch-defect status first.
"""
import hashlib
import json
import time
import urllib.request

URL = "http://127.0.0.1:8998/v1/embeddings"
MODEL = "qwen3-8b-q4"
V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
V4_STATES_SHA = "4aa4538765304851dcb1c9595acb937b52cc20b21574774c1177ca469f63d233"
SERVER_LINE = ("llama-server -m models/Qwen_Qwen3-8B-Q4_K_M.gguf "
               "--alias qwen3-8b-q4 --host 127.0.0.1 --port 8998 --embedding "
               "--pooling last -c 8192 -np 1 -ngl 99 --flash-attn on "
               "-b 1024 -ub 512 --cache-ram 0 --no-cache-prompt")
BUILD_COMMIT = "4de092659663c1f74c80f8e8f6ade33e87d50077"
BUILD_PATCHES = [
    "server-context.cpp handle_embeddings_impl: task.params.cache_prompt = false "
    "(embedding tasks never inherited --no-cache-prompt; struct default true made "
    "results depend on the previous request via slot LCP KV keep)",
    "built with -DGGML_CUDA_USE_GRAPHS=OFF (first execution of a shape bucket "
    "(CUDA graph capture) returns different vectors than replays; with graphs off "
    "every request runs the identical eager path)",
]
CARD = "NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB), local machine — not the GPU host; disclosed in entry M registration"

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


def warmup():
    """Saturate the CUDA memory pool with long varied throwaway texts so
    allocator state converges before the pass (diagnosed history leak)."""
    t0 = time.time()
    for i in range(6):
        embed_one(f"Instrument warmup text {i}. " * (80 + 90 * i))
    print(f"warmup done {time.time() - t0:.0f}s", flush=True)


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

warmup()
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
np.savez("entry_m_emb.npz", v3_emb=E3, v4_emb=E4, labels_v3=y3, labels_v4=y4,
         groups_v3=g3)
manifest = {
    "server": SERVER_LINE,
    "server_build": {"repo": "github.com/ggml-org/llama.cpp",
                     "commit": BUILD_COMMIT, "built_local": True,
                     "cuda": "13.3 (V13.3.73)", "arch": "sm_120",
                     "patches": BUILD_PATCHES,
                     "graphs": "disabled (-DGGML_CUDA_USE_GRAPHS=OFF)"},
    "card": CARD,
    "endpoint": "POST /v1/embeddings, input = raw state text verbatim",
    "client": ("one text per POST, serial — the entry-K standing instrument "
               "rule; six long varied warmup texts before the pass (pool saturation) "
               "in steady-state graph replay"),
    "supersedes_pass1": True,
    "supersedes_reason": ("pass-1 and pass-2 rechecks FAILED: repeat-request "
                          "nondeterminism from three stacked sources — slot LCP "
                          "KV keep (fixed by disclosed cache_prompt patch), CUDA "
                          "graph capture-vs-replay (fixed by -DGGML_CUDA_USE_GRAPHS=OFF), "
                          "and memory-pool growth churn (fixed by long-varied warmup). "
                          "All disclosed in the ledger before any metric existed"),
    "dim": int(E3.shape[1]),
    "n_v3": int(len(y3)), "n_v4": int(len(y4)),
    "v3_states_sha256": V3_STATES_SHA, "v4_states_sha256": V4_STATES_SHA,
    "embd_normalize": "server default (euclidean/L2)",
    "quant": "Qwen_Qwen3-8B Q4_K_M GGUF (bartowski), ~4.8 bpw",
    "wall_s": round(wall, 2),
    "determinism_recheck_bitexact": bool(recheck),
}
json.dump(manifest, open("entry_m_embed_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
