#!/usr/bin/env python3
"""Entry L confirmation — THE ONE ADJUDICATION PASS (host side, CPU only).

Applies the exported entry-L readout (entry_k_readout.json, sha-pinned)
verbatim to the fresh completion-claim blocks: p = sigmoid(platt_a *
(w . x + b) + platt_c), computed exactly in the serving shape (plain
arithmetic, no sklearn in the head path; sklearn is used for metrics only).
Gate (verbatim, entry L): AUROC >= 0.75 AND Brier < the fresh pool's own
constant predictor (recomputed from labels). Also recorded: the operational
confusion at tau* = 0.326134 — the accept/reject point the GATE routes on.

Chain of custody: states file sha must equal the embed manifest's pin; the
readout sha must equal the export pin. No refit, no threshold choice, no
arm switch — any mismatch aborts before a metric is computed.
"""
import hashlib
import json

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

READOUT_SHA = "c840ba54f3986b9aa679f126a9b416cc41716e85a00158e322df522664a323e8"
# display-rounded value from the entry-L ledger; the binding quantity is the
# readout artifact's own full-precision tau (checked to rounding precision)
TAU_DISPLAY = 0.326134

got = hashlib.sha256(open("entry_k_readout.json", "rb").read()).hexdigest()
assert got == READOUT_SHA, got
man = json.load(open("entry_l_embed_manifest.json"))
assert man["determinism_recheck_bitexact"] is True
states_sha = hashlib.sha256(open("entry_l_states.jsonl", "rb").read()).hexdigest()
assert states_sha == man["states_sha256"], (states_sha, man["states_sha256"])

R = json.load(open("entry_k_readout.json"))
w = np.array(R["w"], dtype=np.float64)
b = R["b"]
pa, pc = R["platt_a"], R["platt_c"]
tau = R["tau"]
assert abs(round(tau, 6) - TAU_DISPLAY) < 1e-9, tau

z = np.load("entry_l_emb.npz", allow_pickle=False)
X = z["emb"].astype(np.float64)
y = z["labels"].astype(int)
assert X.shape == (man["n"], man["dim"]) and len(y) == man["n"]
assert int(y.sum()) == man["positives"]

scores = X @ w + b
p = 1.0 / (1.0 + np.exp(-(pa * scores + pc)))

base = float(y.mean())
const_brier = float(np.mean((base - y) ** 2))
au = float(roc_auc_score(y, p))
br = float(brier_score_loss(y, p))

bins = np.clip((p * 10).astype(int), 0, 9)
ece = float(sum((bins == k).sum() / len(y) * abs(y[bins == k].mean() - p[bins == k].mean())
                for k in range(10) if (bins == k).sum()))

pred = (p >= tau).astype(int)
tp = int(((pred == 1) & (y == 1)).sum())
fp = int(((pred == 1) & (y == 0)).sum())
tn = int(((pred == 0) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum())

results = {
    "gate": {
        "fresh_auroc": round(au, 4),
        "fresh_brier": round(br, 4),
        "constant_brier": round(const_brier, 4),
        "base_rate": round(base, 4),
        "gate_auroc_ge_075": bool(au >= 0.75),
        "gate_brier_lt_constant": bool(br < const_brier),
        "gate_pass": bool(au >= 0.75 and br < const_brier),
    },
    "operating_point_tau": {
        "tau": tau,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accept_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "reject_tpr": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_reject_rate": round(fp / (fp + tn), 4) if fp + tn else None,
    },
    "ece10": round(ece, 4),
    "pool": {"n": int(len(y)), "positives": int(y.sum()),
             "manifest": man},
}
json.dump(results, open("entry_l_results.json", "w"), indent=1)
print(json.dumps(results["gate"], indent=1))
print("tau* confusion:", results["operating_point_tau"])
