#!/usr/bin/env python3
"""Entry J Stage 2 adjudication — frozen readout applied cold. ONE execution.

Registered recipe (IMPLEMENTATION_PLAN.md 2026-09-24): ranker = logistic
regression (C=10, class_weight="balanced", max_iter 5000) refit on ALL 316
v3 state embeddings; Platt map = unpenalized 1-D logistic fitted on the
out-of-fold decision scores of those 316 under the frozen entry-H fold
split (seed 13, grouped by session; folds dumped at Stage 1 in
entry_j_folds.json). Fresh states -> decision score -> Platt -> probability.
Gate on the fresh pool: AUROC >= 0.75 AND Brier < the fresh pool's own
constant predictor. No fit, threshold, or recalibration touches fresh data.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

V3_EMB_SHA = "416cc76a6c6bc47ad5f832a66bdfd011b5004c3d75ff15285f9b1aae72d0db55"
V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
FOLDS_SHA = "fe061388aed5fe8a14e4451d1bed329ac0db3acca0246b46971c526ded2eb18a"

for fname, want in (("entry_h_emb.npz", V3_EMB_SHA),
                    ("entry_h_states.jsonl", V3_STATES_SHA),
                    ("entry_j_folds.json", FOLDS_SHA)):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)

z3 = np.load("entry_h_emb.npz", allow_pickle=False)
X3 = z3["state_emb_raw"].astype(np.float32)
y3 = z3["labels"].astype(int)
folds = json.load(open("entry_j_folds.json"))

z4 = np.load("entry_j2_emb.npz", allow_pickle=False)
X4 = z4["state_emb_raw"].astype(np.float32)
y4 = z4["labels"].astype(int)
assert len(y3) == 316 and (y3 == 1).sum() == 28
assert len(y4) == 290


def ranker():
    return LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)


# Platt map from frozen-fold out-of-fold scores on the 316 (v3 data only)
oof_sc = np.zeros(len(y3))
for tr, te in folds:
    r = ranker()
    r.fit(X3[tr], y3[tr])
    oof_sc[te] = r.decision_function(X3[te])
platt = LogisticRegression(penalty=None)
platt.fit(oof_sc.reshape(-1, 1), y3)

# final ranker on all 316, applied cold to fresh embeddings
final = ranker()
final.fit(X3, y3)
p4 = platt.predict_proba(final.decision_function(X4).reshape(-1, 1))[:, 1]

base = y4.mean()
const_brier = float(np.mean((base - y4) ** 2))
au = float(roc_auc_score(y4, p4))
br = float(brier_score_loss(y4, p4))

bins = np.clip((p4 * 10).astype(int), 0, 9)
deciles = [{"bin": int(b), "n": int((bins == b).sum()),
            "mean_p": round(float(p4[bins == b].mean()), 4) if (bins == b).sum() else None,
            "freq_y": round(float(y4[bins == b].mean()), 4) if (bins == b).sum() else None}
           for b in range(10)]

results = {
    "inputs": {"v3_emb_sha256": V3_EMB_SHA, "v3_states_sha256": V3_STATES_SHA,
               "folds_sha256": FOLDS_SHA,
               "v4_states_sha256": hashlib.sha256(
                   open("entry_j2_states.jsonl", "rb").read()).hexdigest()},
    "fresh_pool": {"n": int(len(y4)), "positives": int(y4.sum()),
                   "base_rate": round(float(base), 4),
                   "constant_brier": round(const_brier, 4)},
    "frozen_readout": {"ranker": "logreg C=10 balanced on all 316 v3 states",
                       "platt": "unpenalized 1-D logistic on frozen-fold oof scores"},
    "fresh": {"pooled_auroc": round(au, 4), "pooled_brier": round(br, 4),
              "calibration_deciles": deciles},
    "gate": {"auroc_ge_075": bool(au >= 0.75), "brier_lt_constant": bool(br < const_brier),
             "gate_pass": bool(au >= 0.75 and br < const_brier)},
}
json.dump(results, open("entry_j2_results.json", "w"), indent=1)
print(json.dumps({k: results[k] for k in ("fresh_pool", "fresh", "gate")}, indent=1))
