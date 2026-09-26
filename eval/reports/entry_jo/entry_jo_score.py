#!/usr/bin/env python3
"""Entry JO — THE ONE SCORING PASS (laptop side; CPU-only server, loopback).

Chain of custody, all checked before any metric:
  - gguf / head / adapter sha pins (PROTOCOL.md table)
  - states sha == the entry-L embed manifest pin
  - metric code validated by recomputing the dial's frozen entry-L results
    from entry_l_emb.npz + entry_k_readout.json to 4 dp (a metric-code
    control: the same auroc/brier/ece10 functions are then applied to Jev)

All 140 rows scored through the quantizer's own adapter (imported, text-only
route, prompt + head math unmodified), strictly serial, one request in
flight, each row appended to entry_jo_scores.jsonl (crash-resumable, every
row scored exactly once). No exclusions, no refits, no tuning — PROTOCOL.md.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path("/home/sahal/models/jev-omni-q4")
ENTRY_L = HERE.parent / "entry_l"
SERVER = "http://127.0.0.1:9021"

SHA_GGUF = "35bf51cb0dee2504e9f16093278ca853da2888251577236371630124605c40cd"
SHA_HEAD = "47b346e120dc9110ef7c0b610cf5d1f459ddcdb05ffd3cd126b6938d88a8bb77"
SHA_ADAPTER = "11d07fdf14f22890ee92b872481052b7c66c0ac65adabf83a70bbdd5150d9b83"

QUESTION = "Has the agent completed the task?"
OPTIONS = ["Yes", "No"]

FROZEN_DIAL = {"auroc": 0.9011, "brier": 0.0861, "ece10": 0.0852}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def custody():
    got = {
        "gguf": sha256(MODEL_DIR / "Jev-Omni-Unified-Q4_K_M.gguf"),
        "head": sha256(MODEL_DIR / "decision-head-f32.npz"),
        "adapter": sha256(HERE / "jev_omni_gguf_decide.py"),
    }
    assert got["gguf"] == SHA_GGUF, got["gguf"]
    assert got["head"] == SHA_HEAD, got["head"]
    assert got["adapter"] == SHA_ADAPTER, got["adapter"]
    man = json.load(open(ENTRY_L / "entry_l_embed_manifest.json"))
    states_sha = sha256(ENTRY_L / "entry_l_states.jsonl")
    assert states_sha == man["states_sha256"], (states_sha, man["states_sha256"])
    return got


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "jev_omni_gguf_decide", HERE / "jev_omni_gguf_decide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- metrics: identical definitions to entry_l_confirm.py, no sklearn -------

def _ranks(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(y, p):
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    r = _ranks(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece10(y, p):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * 10).astype(int), 0, 9)
    return float(sum((bins == k).sum() / len(y)
                     * abs(y[bins == k].mean() - p[bins == k].mean())
                     for k in range(10) if (bins == k).sum()))


def metric_triple(y, p):
    base = float(np.mean(y))
    return {"auroc": round(auroc(y, p), 4),
            "brier": round(brier(y, p), 4),
            "constant_brier": round(float(np.mean((base - y) ** 2)), 4),
            "ece10": round(ece10(y, p), 4),
            "base_rate": round(base, 4)}


def validate_metric_code():
    """Recompute the dial's frozen entry-L numbers; must match to 4 dp."""
    R = json.load(open(ENTRY_L / "entry_k_readout.json"))
    z = np.load(ENTRY_L / "entry_l_emb.npz", allow_pickle=False)
    X = z["emb"].astype(np.float64)
    y = z["labels"].astype(int)
    scores = X @ np.array(R["w"], dtype=np.float64) + R["b"]
    p = 1.0 / (1.0 + np.exp(-(R["platt_a"] * scores + R["platt_c"])))
    got = metric_triple(y, p)
    for k, want in (("auroc", FROZEN_DIAL["auroc"]), ("brier", FROZEN_DIAL["brier"]),
                    ("ece10", FROZEN_DIAL["ece10"])):
        assert got[k] == want, (k, got[k], want)
    return got, p, R["tau"]


def score_pool(adapter):
    rows = [json.loads(l) for l in open(ENTRY_L / "entry_l_states.jsonl")]
    out_path = HERE / "entry_jo_scores.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            done.add(json.loads(l)["i"])
    head_path = MODEL_DIR / "decision-head-f32.npz"
    with open(out_path, "a") as f:
        for i, r in enumerate(rows):
            if i in done:
                continue
            prompt_chars = len(r["ev"]) + len(QUESTION) + 60
            res = adapter.decide(SERVER, head_path, r["ev"], QUESTION, OPTIONS)
            rec = {"i": i, "session_dir": r["session_dir"], "label": r["label"],
                   "p_yes": res["probabilities"]["Yes"],
                   "prompt_chars": prompt_chars}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(rows)}] label={r['label']} "
                  f"p_yes={rec['p_yes']:.4f} chars={prompt_chars}", flush=True)
    scored = [json.loads(l) for l in open(out_path)]
    assert len(scored) == len(rows) and sorted(s["i"] for s in scored) == list(range(len(rows)))
    scored.sort(key=lambda s: s["i"])
    return rows, scored


def main():
    os.nice(5)
    pins = custody()
    print("custody OK:", {k: v[:12] for k, v in pins.items()})
    dial_metrics, p_dial, tau = validate_metric_code()
    print("metric-code control (dial recompute):", dial_metrics)

    adapter = load_adapter()
    rows, scored = score_pool(adapter)

    y = np.array([r["label"] for r in scored])
    p = np.array([s["p_yes"] for s in scored])
    m = metric_triple(y, p)
    m["clipped_rows"] = int(sum(s["prompt_chars"] > 6000 for s in scored))

    gate = {"auroc_ge_075": bool(m["auroc"] >= 0.75),
            "brier_lt_constant": bool(m["brier"] < m["constant_brier"]),
            "ece10_le_010": bool(m["ece10"] <= 0.10)}
    gate["pass"] = bool(all(gate.values()))

    pd_ = np.clip(p_dial, 1e-9, 1 - 1e-9)
    pj = np.clip(p, 1e-9, 1 - 1e-9)
    diff = np.abs(pj - pd_)
    err_j, err_d = np.abs(pj - y), np.abs(pd_ - y)
    d_acc_j = (pj >= tau).astype(int)
    d_acc_d = (pd_ >= tau).astype(int)
    paired = {
        "tau_dial": tau,
        "mean_abs_p_diff": round(float(diff.mean()), 4),
        "max_abs_p_diff": round(float(diff.max()), 4),
        "both_accept": int(((d_acc_j == 1) & (d_acc_d == 1)).sum()),
        "both_reject": int(((d_acc_j == 0) & (d_acc_d == 0)).sum()),
        "jev_only_accept": int(((d_acc_j == 1) & (d_acc_d == 0)).sum()),
        "dial_only_accept": int(((d_acc_j == 0) & (d_acc_d == 1)).sum()),
        "jev_abs_error_win": int((err_j < err_d - 1e-12).sum()),
        "dial_abs_error_win": int((err_d < err_j - 1e-12).sum()),
    }

    results = {"pool_sha256": sha256(ENTRY_L / "entry_l_states.jsonl"),
               "pins": pins, "server": SERVER,
               "jev_q4km": m, "gate": gate,
               "dial_frozen_entry_l": FROZEN_DIAL,
               "dial_recompute_control": dial_metrics,
               "paired_descriptive": paired,
               "n": len(scored), "positives": int(y.sum())}
    json.dump(results, open(HERE / "entry_jo_results.json", "w"), indent=1)
    print(json.dumps({"jev_q4km": m, "gate": gate, "paired": paired}, indent=1))


if __name__ == "__main__":
    main()
