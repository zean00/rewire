#!/usr/bin/env python3
"""Entry L export — mechanical reproduction of the adjudicated entry-K K-a
readout from pinned artifacts, plus the v3-only threshold freeze.

NOT a new fit and NOT an adjudication: the model code below is verbatim
entry_k_fit.py's K-a block, recomputed from the same pinned inputs so the
coefficients can ship as data (serve time = dot product + Platt + threshold,
no sklearn). Two guards:
  1. reproduction check — the recomputed readout must round-match the
     published entry_k_results.json numbers exactly, else abort;
  2. tau is derived ONLY from v3 out-of-fold Platt probabilities; v4 vectors
     are touched by the reproduction check only (no number derived from them
     enters the artifact beyond the check itself).

tau* = the HIGHEST threshold attaining max Youden J (TPR - FPR) on the v3
oof probabilities; sklearn's roc_curve thresholds descend, so the first
argmax is the highest such threshold (the most conservative operating point
on the J plateau). Frozen into entry L with this script.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
V4_STATES_SHA = "4aa4538765304851dcb1c9595acb937b52cc20b21574774c1177ca469f63d233"
FOLDS_SHA = "fe061388aed5fe8a14e4451d1bed329ac0db3acca0246b46971c526ded2eb18a"
EMB_NPZ_SHA = "b413c88a4e89255f8184849f3a313c58e4639597db11b7d796031db8d790c810"
RESULTS_SHA = "deceb7270ca123f0eda7a185bb19584a9be17d115f507f291eafa893769e2cf6"

for fname, want in (("entry_h_states.jsonl", V3_STATES_SHA),
                    ("entry_j2_states.jsonl", V4_STATES_SHA),
                    ("entry_j_folds.json", FOLDS_SHA),
                    ("entry_k_emb.npz", EMB_NPZ_SHA),
                    ("entry_k_results.json", RESULTS_SHA)):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)
man = json.load(open("entry_k_embed_manifest.json"))
assert man["determinism_recheck_bitexact"] is True

z = np.load("entry_k_emb.npz", allow_pickle=False)
X3 = z["v3_emb"].astype(np.float32)
X4 = z["v4_emb"].astype(np.float32)
y3 = z["labels_v3"].astype(int)
y4 = z["labels_v4"].astype(int)
assert X3.shape == (316, man["dim"]) and X4.shape == (290, man["dim"])
assert (y3 == 1).sum() == 28 and (y4 == 1).sum() == 32
folds = json.load(open("entry_j_folds.json"))


def ranker():
    return LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)


# ---- K-a, verbatim entry_k_fit.py --------------------------------------------
oof_sc = np.zeros(len(y3))
for tr, te in folds:
    r = ranker()
    r.fit(X3[tr], y3[tr])
    oof_sc[te] = r.decision_function(X3[te])
platt = LogisticRegression(penalty=None)
platt.fit(oof_sc.reshape(-1, 1), y3)
final = ranker()
final.fit(X3, y3)
p_a = platt.predict_proba(final.decision_function(X4).reshape(-1, 1))[:, 1]
p_oof = platt.predict_proba(oof_sc.reshape(-1, 1))[:, 1]

# ---- reproduction check (abort on any mismatch) -------------------------------
chk = {
    "v3_oof_auroc": round(float(roc_auc_score(y3, p_oof)), 4),
    "fresh_auroc": round(float(roc_auc_score(y4, p_a)), 4),
    "fresh_brier": round(float(brier_score_loss(y4, p_a)), 4),
}
print("reproduction:", chk)
assert chk == {"v3_oof_auroc": 0.8779, "fresh_auroc": 0.8867,
               "fresh_brier": 0.0653}, chk

# ---- tau* from v3 oof ONLY ----------------------------------------------------
fpr, tpr, thr = roc_curve(y3, p_oof)
finite = ~np.isinf(thr)
J = (tpr - fpr)[finite]
tau = float(thr[finite][int(np.argmax(J))])  # thresholds descend -> first max = highest
pred = (p_oof >= tau).astype(int)
tp = int(((pred == 1) & (y3 == 1)).sum())
fp = int(((pred == 1) & (y3 == 0)).sum())
tn = int(((pred == 0) & (y3 == 0)).sum())
fn = int(((pred == 0) & (y3 == 1)).sum())
print(f"tau*={tau:.6f} oof confusion tp={tp} fp={fp} tn={tn} fn={fn} "
      f"tpr={tp / (tp + fn):.4f} fpr={fp / (fp + tn):.4f}")

readout = {
    "recipe": ("K-a (entry K, adjudicated): logreg C=10 class_weight=balanced "
               "max_iter=5000 on 316 v3 Gemma-12B last-token L2-normalized "
               "embeddings; Platt (unpenalized 1-D logistic) on frozen-fold "
               "oof decision scores; applied to the same embedding space"),
    "embed_client_rule": "ONE text per /v1/embeddings request, verbatim state text, no template",
    "apply": ("p = platt.sigmoid(a * (w . x + b) + c); accept iff p >= tau"),
    "w": [float(v) for v in final.coef_[0]],
    "b": float(final.intercept_[0]),
    "platt_a": float(platt.coef_[0][0]),
    "platt_c": float(platt.intercept_[0]),
    "tau": tau,
    "v3_oof": {"auroc": chk["v3_oof_auroc"], "tp": tp, "fp": fp, "tn": tn,
               "fn": fn, "tpr_at_tau": round(tp / (tp + fn), 4),
               "fpr_at_tau": round(fp / (fp + tn), 4)},
    "reproduction_check": chk,
    "provenance": {
        "entry_k_emb.npz": EMB_NPZ_SHA,
        "entry_h_states.jsonl": V3_STATES_SHA,
        "entry_j2_states.jsonl": V4_STATES_SHA,
        "entry_j_folds.json": FOLDS_SHA,
        "entry_k_results.json": RESULTS_SHA,
        "embed_manifest": man,
    },
}
json.dump(readout, open("entry_k_readout.json", "w"))
print("wrote entry_k_readout.json",
      hashlib.sha256(open("entry_k_readout.json", "rb").read()).hexdigest())
