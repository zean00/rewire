"""Entry-H instrument smoke test (mechanics only, NO probe data).

Validates: heads load per the pinned checkpoint cfg; encoder last-token pooling
matches the served recipe (left-truncate to last 2048 tokens, L2-normalize);
projection + scoring reproduce the canonical pipeline (score = exp(logit_scale)
* cosine of head projections); determinism across two passes; VRAM footprint.
Throwaway sentences only — the entry itself is halted, no corpus exists.
"""
import hashlib
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/home/sahal/clm_instrument/CLM/src")
from clm.heads import make_head

CKPT = "/home/sahal/.cache/clm/CLM_v0.1-8B.pt"
CKPT_SHA = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
MODEL = "Qwen/Qwen3-8B"
MAXLEN = 2048  # their Embedder default: truncate_prompt_tokens=2048


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


h = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
assert h == CKPT_SHA, h

d = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = d["cfg"]
dev = "cuda"


def mk():
    return make_head(cfg["width"], cfg["depth"], cfg["projection_dim"],
                     cfg["activation"], cfg["layernorm"], cfg["residual"])


state_head, action_head = mk(), mk()
state_head.load_state_dict(d["state_head"])
action_head.load_state_dict(d["action_head"])
state_head.to(dev).eval()
action_head.to(dev).eval()
scale = float(d["logit_scale"].exp().item())

tok = AutoTokenizer.from_pretrained(MODEL)
tok.truncation_side = "left"  # match vLLM truncate_prompt_tokens semantics (keep the tail)
t0 = time.time()
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
print("load_s", round(time.time() - t0, 1), flush=True)


@torch.no_grad()
def encode(texts):
    outs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).to(model.device)
        hid = model(**ids).last_hidden_state[0, -1]  # last-token pooling (causal LM)
        outs.append(hid.float().cpu().numpy())
    return l2(np.stack(outs))


ctx = "Browser page snapshot:\nbutton Submit\ntextbox username\nheading Checkout complete"
instr = "Did the task succeed?"
s_txt = f"{ctx}\n\n{instr}"  # state_text() layout: context, blank line, question last
a_pos = "The agent completed the task successfully."
a_neg = "The agent failed the task."

t0 = time.time()
S = encode([s_txt])
A = encode([a_pos, a_neg])
torch.cuda.synchronize()
wall = time.time() - t0


def project(head, x):
    with torch.no_grad():
        return torch.nn.functional.normalize(head(torch.from_numpy(x).to(dev)), dim=-1).cpu().numpy()


ZS = project(state_head, S)
ZA = project(action_head, A)
raw_pos = float(S[0] @ A[0])
raw_neg = float(S[0] @ A[1])
sc_pos = scale * float(ZA[0] @ ZS[0])
sc_neg = scale * float(ZA[1] @ ZS[0])
e = np.exp(np.array([sc_pos, sc_neg]) - max(sc_pos, sc_neg))
probs = e / e.sum()

S2 = encode([s_txt])
A2 = encode([a_pos, a_neg])
det = bool(np.array_equal(S, S2) and np.array_equal(A, A2))

free, total = torch.cuda.mem_get_info(0)
print(json.dumps({
    "ckpt_sha256": CKPT_SHA,
    "state_dim": int(S.shape[1]),
    "proj_dim": int(ZS.shape[1]),
    "scale": round(scale, 3),
    "raw_cos_pos": round(raw_pos, 6),
    "raw_cos_neg": round(raw_neg, 6),
    "head_score_pos": round(sc_pos, 4),
    "head_score_neg": round(sc_neg, 4),
    "softmax_pos": round(float(probs[0]), 6),
    "deterministic_bitexact": det,
    "vram_used_gb": round((total - free) / 2**30, 2),
    "encode_wall_s": round(wall, 2),
    "fidelity_note": "encoder bf16 CPU-offload (vllm absent on host); vLLM-pooling parity untested",
}, indent=1))
