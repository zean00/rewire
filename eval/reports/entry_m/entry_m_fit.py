#!/usr/bin/env python3
"""Entry M adjudication — frozen recipe (ledger 2026-09-25, commit 532ce1a).
ONE execution. Verbatim transplant of entry_k_fit.py: same states, same
folds, same arms (K-a -> m_a, K-b -> m_b), same gate; only the vectorizer
differs (Qwen3-8B Q4_K_M via llama-server, local 5060).

DIAGNOSTIC ONLY: the fresh pool was adjudicated in entries J and K; these
numbers attribute the J-vs-K gap and earn no deployment right.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

V3_STATES_SHA = "8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01"
V4_STATES_SHA = "4aa4538765304851dcb1c9595acb937b52cc20b21574774c1177ca469f63d233"
FOLDS_SHA = "fe061388aed5fe8a14e4451d1bed329ac0db3acca0246b46971c526ded2eb18a"
CS = (0.01, 0.1, 1.0, 10.0)

for fname, want in (("entry_h_states.jsonl", V3_STATES_SHA),
                    ("entry_j2_states.jsonl", V4_STATES_SHA),
                    ("entry_j_folds.json", FOLDS_SHA)):
    got = hashlib.sha256(open(fname, "rb").read()).hexdigest()
    assert got == want, (fname, got)
man = json.load(open("entry_m_embed_manifest.json"))
assert man["determinism_recheck_bitexact"] is True

z = np.load("entry_m_emb.npz", allow_pickle=False)
X3 = z["v3_emb"].astype(np.float32)
X4 = z["v4_emb"].astype(np.float32)
y3 = z["labels_v3"].astype(int)
y4 = z["labels_v4"].astype(int)
assert X3.shape == (316, man["dim"]) and X4.shape == (290, man["dim"])
assert (y3 == 1).sum() == 28 and (y4 == 1).sum() == 32
folds = json.load(open("entry_j_folds.json"))


def ranker():
    return LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)


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
    return [{"bin": int(b), "n": int((bins == b).sum()),
             "mean_p": round(float(p[bins == b].mean()), 4) if (bins == b).sum() else None,
             "freq_y": round(float(yy[bins == b].mean()), 4) if (bins == b).sum() else None}
            for b in range(10)]


results = {"manifest": man, "arms": {}}

# ---- m_a (K-a verbatim) ------------------------------------------------------
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
results["arms"]["m_a"] = {
    "recipe": "K-a transplanted: C=10 balanced + Platt on frozen-fold oof",
    "v3_oof_auroc": round(roc_auc_score(y3, platt.predict_proba(oof_sc.reshape(-1, 1))[:, 1]), 4),
}

# ---- m_b (K-b verbatim) ------------------------------------------------------
briers = {}
for C in CS:
    oof_p = np.zeros(len(y3))
    for tr, te in folds:
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(X3[tr], y3[tr])
        oof_p[te] = m.predict_proba(X3[te])[:, 1]
    briers[C] = float(brier_score_loss(y3, oof_p))
best_C = min(CS, key=lambda c: (briers[c], c))
mkb = LogisticRegression(C=best_C, max_iter=5000)
mkb.fit(X3, y3)
p_b = mkb.predict_proba(X4)[:, 1]
results["arms"]["m_b"] = {
    "recipe": "K-b transplanted: unweighted, C by v3 oof Brier, refit on all 316",
    "v3_oof_brier_by_C": {str(c): round(v, 5) for c, v in briers.items()},
    "picked_C": best_C,
}

# ---- gate on the fresh 290 (comparability only) -------------------------------
base = y4.mean()
const_brier = float(np.mean((base - y4) ** 2))
for arm, p in (("m_a", p_a), ("m_b", p_b)):
    au = float(roc_auc_score(y4, p))
    br = float(brier_score_loss(y4, p))
    results["arms"][arm].update({
        "fresh_auroc": round(au, 4),
        "fresh_brier": round(br, 4),
        "fresh_ece10": round(ece10(p, y4), 4),
        "calibration_deciles": calib_table(p, y4),
        "gate_auroc_ge_075": bool(au >= 0.75),
        "gate_brier_lt_constant": bool(br < const_brier),
        "gate_pass": bool(au >= 0.75 and br < const_brier),
    })
results["fresh_pool"] = {"n": int(len(y4)), "positives": int(y4.sum()),
                         "base_rate": round(float(base), 4),
                         "constant_brier": round(const_brier, 4)}

json.dump(results, open("entry_m_results.json", "w"), indent=1)
for arm in ("m_a", "m_b"):
    a = results["arms"][arm]
    print(arm, "AUROC", a["fresh_auroc"], "Brier", a["fresh_brier"],
          "gate_pass", a["gate_pass"])
print("constant_brier", round(const_brier, 4))
