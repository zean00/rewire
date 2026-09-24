#!/usr/bin/env python3
"""Entry H Stage-1 fit — frozen protocol, ONE execution.

Pre-declared conventions (fixed before this script ever ran):
- H1 variants: PRIMARY = head-space cosine margin (their canonical instrument;
  scale 100.811 is a positive monotone factor, absorbed by the fitted 2-param
  logistic); SECONDARY = encoder-space cosine margin — which equals the
  registered literal "s·a+ − s·a−" because every embedding is L2-normalized.
- H1 gate: pooled out-of-fold predictions over all 5 pairs (1580 rows);
  per-pair AUROCs reported for the paraphrase leg (judged on PRIMARY:
  every pair AUROC >= 0.70 AND max-min range <= 0.10).
- H1 per-fold fit: LogisticRegression(penalty=None) on the scalar margin —
  exactly the registered 2-parameter (w, b); no class weights (as registered).
- H2 features: 4096-d state embedding; inner selection by grouped 4-fold CV
  (same seed) on training folds only, mean inner AUROC; outer pooled 316 oof.
- Gate (each arm): AUROC >= 0.75 AND Brier < 0.081 (constant predictor).
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn

SEED = 13
CONSTANT_BRIER = 0.081

man = json.load(open("entry_h_embed_manifest.json"))
assert man["determinism_recheck_bitexact"] is True
z = np.load("entry_h_emb.npz", allow_pickle=False)
S_raw = z["state_emb_raw"].astype(np.float32)      # [316, 4096] L2-normed
ZS = z["state_proj"].astype(np.float32)            # [316, 512]  L2-normed
ZA = z["act_proj"].astype(np.float32)              # [10, 512]   L2-normed
A_raw = z["act_emb_raw"].astype(np.float32)        # [10, 4096]  L2-normed
y = z["labels"].astype(int)
groups = z["groups"].astype(str)
N = len(y)
assert N == 316 and (y == 1).sum() == 28

SK = [np.linalg.norm(S_raw[i]) for i in range(3)]  # sanity: normalized inputs
assert all(abs(s - 1) < 1e-2 for s in SK)


def brier(p, yy):
    return float(np.mean((p - yy) ** 2))


def ece10(p, yy):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e, tot = 0.0, len(yy)
    for b in range(10):
        m = bins == b
        if m.sum():
            e += m.sum() / tot * abs(yy[m].mean() - p[m].mean())
    return float(e)


def oof_folds():
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(N), y, groups=groups))


folds = oof_folds()

results = {"manifest": {k: man[k] for k in
                        ("states_sha256", "ckpt_sha256", "encoder_precision")}, "arms": {}}

# ---------------- H1: two conventions x five pairs ----------------------------
for name, Sm, Am in (("h1_headspace", ZS, ZA), ("h1_encoderspace", S_raw, A_raw)):
    per_pair = []
    pooled_p, pooled_y = [], []
    for j in range(5):
        a_pos, a_neg = Am[2 * j], Am[2 * j + 1]
        m = Sm @ a_pos - Sm @ a_neg                      # [316]
        oof = np.zeros(N)
        for tr, te in folds:
            lr = LogisticRegression(penalty=None, solver="lbfgs")
            lr.fit(m[tr].reshape(-1, 1), y[tr])
            oof[te] = lr.predict_proba(m[te].reshape(-1, 1))[:, 1]
        per_pair.append({"pair": j + 1, "auroc": round(roc_auc_score(y, oof), 4),
                         "brier": round(brier(oof, y), 4)})
        pooled_p.append(oof)
        pooled_y.append(y)
    P = np.concatenate(pooled_p)
    Y = np.concatenate(pooled_y)
    aucs = [pp["auroc"] for pp in per_pair]
    results["arms"][name] = {
        "pooled_auroc": round(roc_auc_score(Y, P), 4),
        "pooled_brier": round(brier(P, Y), 4),
        "pooled_ece10": round(ece10(P, Y), 4),
        "per_pair": per_pair,
        "per_pair_auroc_range": round(max(aucs) - min(aucs), 4),
    }

# ---------------- H2: inner-selected head on 4096-d state embedding ----------
torch.manual_seed(SEED)
X = S_raw
pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4096, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, epochs=50):
    torch.manual_seed(SEED)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward()
        opt.step()
    m.eval()
    def pred(Xnew):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(Xnew, dtype=torch.float32))).numpy()
    return pred


def logreg(C):
    def f(Xtr, ytr):
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        lr.fit(Xtr, ytr)
        return lambda Xnew: lr.predict_proba(Xnew)[:, 1]
    return f


configs = {f"logreg_C{c}": logreg(c) for c in (0.01, 0.1, 1.0, 10.0)}
configs["mlp"] = fit_mlp

# inner selection inside each outer-train fold
inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
oof = np.zeros(N)
picked = []
for tr, te in folds:
    scores = {}
    for cname, mk in configs.items():
        i_p = np.zeros(len(tr))
        tr_g = groups[tr]
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], groups=tr_g):
            f_ = mk(X[tr][itr], y[tr][itr])
            i_p[ite] = f_(X[tr][ite])
        scores[cname] = roc_auc_score(y[tr], i_p)
    best = max(scores, key=scores.get)
    picked.append({"outer_fold_te_size": int(len(te)), "inner_auroc": {k: round(v, 4) for k, v in scores.items()},
                   "picked": best})
    f_ = configs[best](X[tr], y[tr])
    oof[te] = f_(X[te])

results["arms"]["h2"] = {
    "pooled_auroc": round(roc_auc_score(y, oof), 4),
    "pooled_brier": round(brier(oof, y), 4),
    "pooled_ece10": round(ece10(oof, y), 4),
    "fold_selections": picked,
}

for arm, r in results["arms"].items():
    au, br = r["pooled_auroc"], r["pooled_brier"]
    r["gate_auroc_ge_075"] = bool(au >= 0.75)
    r["gate_brier_lt_constant"] = bool(br < CONSTANT_BRIER)
    r["gate_pass"] = bool(au >= 0.75 and br < CONSTANT_BRIER)

json.dump(results, open("entry_h_results.json", "w"), indent=1)
print(json.dumps(results["arms"], indent=1)[:2400])
