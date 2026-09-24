#!/usr/bin/env python3
"""Entry J Stage-1 fit — frozen protocol (ledger 2026-09-24), ONE execution.

Arms, fixed by the registration before this script ran:
- J-a: L2 logistic regression, C in {0.01, 0.1, 1, 10}, NO class weights;
  inner SGKF(4, seed 13) selects C by mean inner Brier (tie -> smaller C).
- J-b: logreg(C=10, class_weight="balanced") as a pure ranker; per
  outer-train, inner SGKF(4, seed 13) inner-oof decision scores feed a
  2-parameter Platt map (unpenalized 1-D logistic); ranker refit on the
  full outer-train, Platt applied out-of-fold.

Splits byte-identical to entry H: StratifiedGroupKFold(5, shuffle, seed 13),
groups = session_dir. Gate per arm on pooled out-of-fold predictions:
AUROC >= 0.75 AND Brier < 0.081. No other adjustments of any kind.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 13
CONSTANT_BRIER = 0.081
CS = (0.01, 0.1, 1.0, 10.0)

for fname, want in (("entry_h_emb.npz",
                     "416cc76a6c6bc47ad5f832a66bdfd011b5004c3d75ff15285f9b1aae72d0db55"),
                    ("entry_h_states.jsonl",
                     "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01")):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)

z = np.load("entry_h_emb.npz", allow_pickle=False)
X = z["state_emb_raw"].astype(np.float32)
y = z["labels"].astype(int)
groups = z["groups"].astype(str)
N = len(y)
assert N == 316 and (y == 1).sum() == 28

folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                  random_state=SEED).split(np.zeros(N), y, groups=groups))
json.dump([[tr.tolist(), te.tolist()] for tr, te in folds],
          open("entry_j_folds.json", "w"))


def ece10(p, yy):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e, tot = 0.0, len(yy)
    for b in range(10):
        m = bins == b
        if m.sum():
            e += m.sum() / tot * abs(yy[m].mean() - p[m].mean())
    return float(e)


def calib_table(p, yy):
    bins = np.clip((p * 10).astype(int), 0, 9)
    return [{"bin": b, "n": int((bins == b).sum()),
             "mean_p": round(float(p[bins == b].mean()), 4) if (bins == b).sum() else None,
             "freq_y": round(float(yy[bins == b].mean()), 4) if (bins == b).sum() else None}
            for b in range(10)]


def ranker():
    return LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)


def direct(C):
    def f(Xtr, ytr):
        lr = LogisticRegression(C=C, max_iter=5000)
        lr.fit(Xtr, ytr)
        return lr.predict_proba
    return f


results = {"inputs": {"emb_sha256": "416cc76a6c6bc47ad5f832a66bdfd011b5004c3d75ff15285f9b1aae72d0db55",
                      "states_sha256": "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"},
           "arms": {}}

# ---- J-a: unbalanced logreg, inner selection by mean Brier -------------------
oof_a = np.zeros(N)
picks = []
for tr, te in folds:
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    scores = {}
    for C in CS:
        brs = []
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], groups=groups[tr]):
            m = LogisticRegression(C=C, max_iter=5000)
            m.fit(X[tr][itr], y[tr][itr])
            brs.append(brier_score_loss(y[tr][ite], m.predict_proba(X[tr][ite])[:, 1]))
        scores[C] = float(np.mean(brs))
    best = min(CS, key=lambda c: (scores[c], c))   # tie -> smaller C
    picks.append({"outer_fold_te_size": int(len(te)),
                  "inner_brier": {str(c): round(v, 5) for c, v in scores.items()},
                  "picked_C": best})
    m = LogisticRegression(C=best, max_iter=5000)
    m.fit(X[tr], y[tr])
    oof_a[te] = m.predict_proba(X[te])[:, 1]

results["arms"]["j_a_unbalanced"] = {
    "pooled_auroc": round(roc_auc_score(y, oof_a), 4),
    "pooled_brier": round(brier_score_loss(y, oof_a), 4),
    "pooled_ece10": round(ece10(oof_a, y), 4),
    "fold_selections": picks,
    "calibration_deciles": calib_table(oof_a, y),
}

# ---- J-b: entry-H balanced ranker + nested Platt ------------------------------
oof_b = np.zeros(N)
for tr, te in folds:
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    sc = np.zeros(len(tr))
    for itr, ite in inner.split(np.zeros(len(tr)), y[tr], groups=groups[tr]):
        r = ranker()
        r.fit(X[tr][itr], y[tr][itr])
        sc[ite] = r.decision_function(X[tr][ite])
    platt = LogisticRegression(penalty=None)
    platt.fit(sc.reshape(-1, 1), y[tr])
    r = ranker()
    r.fit(X[tr], y[tr])
    oof_b[te] = platt.predict_proba(r.decision_function(X[te]).reshape(-1, 1))[:, 1]

results["arms"]["j_b_platt"] = {
    "pooled_auroc": round(roc_auc_score(y, oof_b), 4),
    "pooled_brier": round(brier_score_loss(y, oof_b), 4),
    "pooled_ece10": round(ece10(oof_b, y), 4),
    "calibration_deciles": calib_table(oof_b, y),
}

for arm, r in results["arms"].items():
    au, br = r["pooled_auroc"], r["pooled_brier"]
    r["gate_auroc_ge_075"] = bool(au >= 0.75)
    r["gate_brier_lt_constant"] = bool(br < CONSTANT_BRIER)
    r["gate_pass"] = bool(au >= 0.75 and br < CONSTANT_BRIER)

json.dump(results, open("entry_j_results.json", "w"), indent=1)
for arm in ("j_a_unbalanced", "j_b_platt"):
    a = results["arms"][arm]
    print(arm, "AUROC", a["pooled_auroc"], "Brier", a["pooled_brier"],
          "ECE", a["pooled_ece10"], "gate_pass", a["gate_pass"])
print("J-a picks:", [p["picked_C"] for p in results["arms"]["j_a_unbalanced"]["fold_selections"]])
