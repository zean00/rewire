#!/usr/bin/env python3
"""Entry H embedding pass — HOST side, single batch (registered: embeddings
computed once; no re-runs after seeing metrics).

Inputs : entry_h_states.jsonl (316 rows: session_dir, session, turn, label, ev)
        + the five frozen action-text pairs from the registration.
Outputs: entry_h_emb.npz  (state_emb_raw [316,4096] fp16 L2-normed,
                           state_proj    [316,512] fp16 L2-normed,
                           act_emb_raw   [10,4096] fp16 L2-normed,
                           act_proj      [10,512] fp16 L2-normed,
                           labels [316], groups: session_dir strings)
          entry_h_embed_manifest.json (hashes, snapshot, wall, determinism)
"""
import hashlib
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, "/home/sahal/clm_instrument/CLM/src")
from clm.heads import make_head

CKPT = "/home/sahal/.cache/clm/CLM_v0.1-8B.pt"
CKPT_SHA = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
MODEL = "Qwen/Qwen3-8B"
SNAPSHOT = "b968826d9c46dd6066d109eabc6255188de91218"
MAXLEN = 2048
PAIRS = [  # the five frozen action-text pairs, verbatim from the registration
    ("The agent completed the task successfully.", "The agent failed the task."),
    ("The task's success condition was met.", "The task's success condition was not met."),
    ("The session ended with a correct final result.", "The session ended with an incorrect result or no result."),
    ("The goal was achieved.", "The goal was not achieved."),
    ("Everything the task asked for was done correctly.", "Something the task asked for was done wrong or left undone."),
]


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


rows = [json.loads(l) for l in open("entry_h_states.jsonl")]
assert len(rows) == 316
states_sha = hashlib.sha256(open("entry_h_states.jsonl", "rb").read()).hexdigest()

d = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = d["cfg"]
dev = "cuda"
mk = lambda: make_head(cfg["width"], cfg["depth"], cfg["projection_dim"],
                       cfg["activation"], cfg["layernorm"], cfg["residual"])
state_head, action_head = mk(), mk()
state_head.load_state_dict(d["state_head"])
action_head.load_state_dict(d["action_head"])
state_head.to(dev).eval()
action_head.to(dev).eval()
scale = float(d["logit_scale"].exp().item())

tok = AutoTokenizer.from_pretrained(MODEL)
tok.truncation_side = "left"
model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
model.eval()


@torch.no_grad()
def encode(texts):
    outs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).to(model.device)
        outs.append(model(**ids).last_hidden_state[0, -1].float().cpu().numpy())
    return l2(np.stack(outs))


def project(head, x):
    with torch.no_grad():
        return torch.nn.functional.normalize(head(torch.from_numpy(x).to(dev)), dim=-1).cpu().numpy()


t0 = time.time()
act_texts = [t for pr in PAIRS for t in pr]
A = encode(act_texts)                 # [10, 4096] raw L2
S = encode([r["ev"] for r in rows])   # [316, 4096] raw L2
torch.cuda.synchronize()
wall = time.time() - t0

det = bool(np.array_equal(S[:2], encode([rows[0]["ev"], rows[1]["ev"]])))

ZS = project(state_head, S)           # [316, 512]
ZA = project(action_head, A)          # [10, 512]

labels = np.array([r["label"] for r in rows], dtype=np.int64)
groups = np.array([r["session_dir"] for r in rows])
np.savez("entry_h_emb.npz", state_emb_raw=S.astype(np.float16),
         state_proj=ZS.astype(np.float16), act_emb_raw=A.astype(np.float16),
         act_proj=ZA.astype(np.float16), labels=labels, groups=groups)

manifest = {
    "states_sha256": states_sha,
    "ckpt_sha256": CKPT_SHA,
    "encoder_snapshot": SNAPSHOT,
    "logit_scale": scale,
    "encoder_precision": "bf16 CPU-offload (pre-committed deviation from registered fp16; 15.9GB card)",
    "pooling": "last-token, left-truncated to 2048, L2-normalized",
    "pairs": PAIRS,
    "n_states": int(S.shape[0]),
    "determinism_recheck_bitexact": det,
    "wall_s": round(wall, 2),
}
json.dump(manifest, open("entry_h_embed_manifest.json", "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("n_states", "determinism_recheck_bitexact", "wall_s")}))
