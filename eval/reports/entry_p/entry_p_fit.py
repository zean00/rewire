#!/usr/bin/env python3
"""Entry P fit + gates (HOST side) — PRE-REGISTERED (eval/reports/entry_p/
PROTOCOL.md, 2026-09-27). ONE execution, no refits, no threshold tuning.

Train : LogisticRegression(C=10, class_weight="balanced", max_iter=5000) on
        split=="train" rows.
Platt : (a, b) fit on GroupKFold(5) out-of-fold decision scores over the
        sorted unique train session dirs (deterministic). p = sigmoid(a*z+b).
Gates (val rows only, frozen): AUROC >= 0.75 AND Brier < val constant
        predictor. Everything else is descriptive.

Chain of custody: states sha == build-manifest pin; embed manifest
bit-exact flag; emb rows/splits/groups == states rows. Any mismatch aborts
before a metric is computed.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

STATES = "entry_p_states.jsonl"
BUILD = "entry_p_build_manifest.json"
EMB = "entry_p_emb.npz"
EMB_MAN = "entry_p_embed_manifest.json"

man_b = json.load(open(BUILD))
states_sha = hashlib.sha256(open(STATES, "rb").read()).hexdigest()
assert states_sha == man_b["states_sha256"], (states_sha, man_b["states_sha256"])
man_e = json.load(open(EMB_MAN))
assert man_e["determinism_recheck_bitexact"] is True
assert man_e["states_sha256"] == states_sha

rows = [json.loads(l) for l in open(STATES)]
z = np.load(EMB, allow_pickle=False)
X = z["emb"].astype(np.float64)
y = z["labels"].astype(int)
sp = z["split"].astype(str)
g = z["groups"].astype(str)
assert X.shape == (len(rows), man_e["dim"]), X.shape
assert int(y.sum()) == man_b["positives"]
assert list(sp) == [r["split"] for r in rows]
assert list(g) == [r["session_dir"] for r in rows]

tr, va = sp == "train", sp == "val"
ytr, yva = y[tr], y[va]
assert ytr.min() == 0 and ytr.max() == 1  # both classes present in train

# ranker: frozen C=10 balanced, fit on all train rows; val stays cold
ranker = LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)
ranker.fit(X[tr], ytr)
z_va = ranker.decision_function(X[va])

# Platt on grouped out-of-fold train scores
sess_tr = sorted(set(g[tr]))
sess_to_fold = {s: i for i, s in enumerate(sess_tr)}
fold_of = np.array([sess_to_fold[s] for s in g[tr]])
oof = np.zeros(int(tr.sum()))
for f in range(5):
    m_tr, m_te = fold_of != f, fold_of == f
    if m_te.sum() == 0:
        continue
    r = LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000)
    r.fit(X[tr][m_tr], ytr[m_tr])
    oof[m_te] = r.decision_function(X[tr][m_te])
platt = LogisticRegression(C=1e6, max_iter=5000)
platt.fit(oof.reshape(-1, 1), ytr)
a = float(platt.coef_[0][0])
b = float(platt.intercept_[0])
p_va = 1.0 / (1.0 + np.exp(-(a * z_va + b)))

const_va = float(yva.mean())
const_brier = float(np.mean((const_va - yva) ** 2))
au = float(roc_auc_score(yva, p_va))
br = float(brier_score_loss(yva, p_va))


def ece10(p, yy):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e, tot = 0.0, len(yy)
    for k in range(10):
        m = bins == k
        if m.sum():
            e += m.sum() / tot * abs(yy[m].mean() - p[m].mean())
    return float(e)


def confusion(p, yy, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (yy == 1)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    tn = int(((pred == 0) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    return {"t": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


# F1-max threshold on val: DESCRIPTIVE only, never a gate
ts = np.unique(p_va)
best_t, best_f1 = 0.5, -1.0
for t in ts:
    f1 = f1_score(yva, (p_va >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = float(f1), float(t)

# descriptives from states: loop rate by arm; late-turn concentration
arm = np.array([r["arm"] for r in rows])
by_arm = {a: {"rows": int((arm == a).sum()),
              "pos": int(((arm == a) & (y == 1)).sum())} for a in ("vanilla", "guarded")}
ai = np.array([r["act_index"] for r in rows])
sd_arr = np.array([r["session_dir"] for r in rows])
raw = np.array([r["session_raw"] if r["session_raw"] is not None else -1 for r in rows])
cap = np.array([bool(r["cap_hit"]) for r in rows])
last5 = {s: int(ai[sd_arr == s].max()) - 4 for s in set(sd_arr)}
late = np.array([ai[i] >= last5[sd_arr[i]] for i in range(len(rows))])


def late_share(mask):
    m = mask & late
    return round(float(y[m].mean()), 4) if m.sum() else None


results = {
    "manifest": {"build": man_b, "embed": man_e},
    "platt": {"a": a, "b": b},
    "train_oof": {"auroc": round(float(roc_auc_score(ytr, 1 / (1 + np.exp(-oof)))), 4),
                  "ece10": round(ece10(1 / (1 + np.exp(-oof)), ytr), 4)},
    "gate": {
        "val_auroc": round(au, 4), "val_brier": round(br, 4),
        "val_constant_brier": round(const_brier, 4), "val_base_rate": round(const_va, 4),
        "val_n": int(va.sum()), "val_positives": int(yva.sum()),
        "gate_auroc_ge_075": bool(au >= 0.75),
        "gate_brier_lt_constant": bool(br < const_brier),
        "gate_pass": bool(au >= 0.75 and br < const_brier),
    },
    "descriptives": {
        "val_ece10": round(ece10(p_va, yva), 4),
        "confusion_0.5": confusion(p_va, yva, 0.5),
        "val_f1_max": {"threshold": round(best_t, 6), "f1": round(best_f1, 4),
                       "confusion": confusion(p_va, yva, best_t)},
        "by_arm": by_arm,
        "late5_loop_share_cap_hit": late_share(cap),
        "late5_loop_share_success_raw_gt0": late_share(raw > 0),
        "late5_loop_share_other": late_share(~cap & (raw <= 0)),
    },
}
json.dump(results, open("entry_p_results.json", "w"), indent=1)
print(json.dumps(results["gate"], indent=1))
print("descriptives:", json.dumps(results["descriptives"], indent=1))
