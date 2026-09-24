#!/usr/bin/env python3
"""Entry E post-sweep pipeline — frozen stages, run in order AFTER the v3
sweep completes (168/168 in main_v4.jsonl).

  python3 adjudicate_v3.py extract   # ckpt_review.py over all v4 transcripts
  python3 adjudicate_v3.py balance   # frozen balance rule -> balanced probes
  python3 adjudicate_v3.py judge \
      --bin scores_v3_bin.jsonl --rub scores_v3_rub.jsonl  # frozen verdict

Reads (between balance and judge) run by hand, one pass each, on the
verified single-slot server through the tunnel:
  ckpt_score.py  --probes probes_v4_balanced.jsonl --out scores_v3_bin.jsonl \
                 --base-url http://127.0.0.1:18443/v1
  ckpt_rubric.py --probes probes_v4_balanced.jsonl --out scores_v3_rub.jsonl \
                 --base-url http://127.0.0.1:18443/v1
"""
import argparse
import glob
import json
import re
import subprocess
import sys
import collections

MB = "/tmp/mw_bench"
V3_DIRS = f"{MB}/main_runs_v4"
PROBES = f"{MB}/probes_v3.jsonl"
BALANCED = f"{MB}/probes_v4_balanced.jsonl"
MAIN = f"{MB}/main_v4.jsonl"
KEY_RE = re.compile(r"main_runs_v4/([^/]+)_s(\d+)_(\w+)/")
# frozen polarity vocabulary (same as every prior entry)
VOCAB = {"success": ["success"], "failure": ["failed"], "neutral": []}


def stage_extract():
    files = sorted(glob.glob(f"{V3_DIRS}/*/*.jsonl"))
    files = [f for f in files if not f.endswith(".cdp")]
    print(f"v4 transcripts: {len(files)}")
    subprocess.run([sys.executable, "/tmp/ckpt_review.py", *files,
                    "--out", PROBES], check=True)


def compat_rows(rows):
    """Frozen compat: derive `said` from the claim text with the frozen
    polarity vocabulary (case-insensitive substring, the same matching
    semantics as ckpt_review's own matcher), and replace vocab_hits with
    the frozen-vocab hits exactly as the v2 compat did. `session` and
    `vocab_hits` are filled only when absent."""
    out = []
    for r in rows:
        r = dict(r)
        low = r["claim"].lower()
        if "success" in low:
            said = "success"
        elif "failed" in low:
            said = "failure"
        else:
            said = "neutral"
        r["said"] = said
        r["vocab_hits"] = VOCAB.get(said, [])
        if "session" not in r:
            r["session"] = "/".join(r["file"].split("/")[:-1])
        r["turn"] = int(r["turn"])
        r["evidence_turn"] = int(r["evidence_turn"])
        out.append(r)
    return out


def stage_balance():
    rows = compat_rows([json.loads(l) for l in open(PROBES)])
    polar = [r for r in rows if r["said"] in ("success", "failure")]
    cells = {s: [r for r in polar if r["said"] == s]
             for s in ("success", "failure")}
    for s in cells:
        cells[s].sort(key=lambda r: (r["file"], r["turn"]))  # frozen order
    ns, nf = len(cells["success"]), len(cells["failure"])
    q = max(20, min(ns, nf))
    print(f"polar probes: success {ns}, failure {nf}; balance Q = {q}")
    bal = cells["success"][:q] + cells["failure"][:q]
    print(f"balanced set: {len(bal)} probes "
          f"(success {min(q, ns)}, failure {min(q, nf)})")
    # composition reported BEFORE any read (no read data touched here)
    with open(BALANCED, "w") as f:
        for r in bal:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {BALANCED}")


def load_truth():
    truth = {}
    for line in open(MAIN):
        r = json.loads(line)
        truth[(r["task"], int(r["seed"]), r["arm"])] = r["raw"]
    return truth


def stage_judge(bin_f, rub_f):
    probes = compat_rows([json.loads(l) for l in open(BALANCED)])
    bs = [json.loads(l) for l in open(bin_f)]
    rs = [json.loads(l) for l in open(rub_f)]
    assert len(probes) == len(bs) == len(rs), \
        (len(probes), len(bs), len(rs))
    truth = load_truth()

    rows, skipped = [], 0
    for p, b, r in zip(probes, bs, rs):
        if "error" in b or "error" in r:
            skipped += 1
            continue
        m = KEY_RE.search(p["file"])
        if not m or p.get("evidence_turn", -1) < 0:
            skipped += 1
            continue
        key = (m.group(1), int(m.group(2)), m.group(3))
        if key not in truth:
            skipped += 1
            continue
        # frozen rule: cdp_err sessions are stored raw=null and count as
        # failures (no page state -> task not completed)
        rows.append({"key": key, "turn": p["turn"],
                     "truth_ok": truth[key] is not None and truth[key] > 0,
                     "said": b["said"],
                     "p_done": b["p_done_yes"],
                     "E": r.get("E") if "error" not in r else None})
    print(f"joined rows: {len(rows)}  skipped: {skipped}")

    def cut(name, subset):
        pol = [x for x in subset if x["said"] in ("success", "failure")]
        # frozen success-claim cut: wrong = said success on a failed session;
        # right = said success on a passed session
        w = [x for x in pol if x["said"] == "success" and not x["truth_ok"]]
        c = [x for x in pol if x["said"] == "success" and x["truth_ok"]]
        print(f"\n-- {name}: polar {len(pol)} "
              f"(success {sum(1 for x in pol if x['said']=='success')}, "
              f"failure {sum(1 for x in pol if x['said']=='failure')}); "
              f"wrong-success {len(w)}, right-success {len(c)}")
        return w, c

    def rates(frame, T, wrong, correct):
        def flagged(x):
            pv = x["p_done"] if frame == "binary" else x["E"]
            if pv is None:
                return None
            if x["said"] == "success":
                return pv < T
            return pv > T
        det = sum(1 for x in wrong if flagged(x))
        fa = sum(1 for x in correct if flagged(x))
        d = det / len(wrong) * 100 if wrong else float("nan")
        f_ = fa / len(correct) * 100 if correct else float("nan")
        return d, f_, det, fa

    w, c = cut("SUCCESS-CLAIM CUT (frozen bar)", rows)
    if not w or not c:
        print("empty cut — cannot judge")
        return
    det_bin, fa_bin, _, _ = rates("binary", 0.5, w, c)
    print(f"\nbinary  T=0.5      det {det_bin:5.1f}%   FA {fa_bin:5.1f}%")
    verdict = "INDETERMINATE (cell < 30)" if len(c) < 30 else "NOT MET"
    for T in (1.0, 2.0):
        d, f_, dr, fr = rates("rubric", T, w, c)
        ok = (len(c) >= 30 and d >= det_bin - 5.0 and f_ <= fa_bin - 15.0)
        print(f"rubric  T={T:<4}     det {d:5.1f}%   FA {f_:5.1f}%   "
              f"[bar det>={det_bin-5:.0f}%, FA<={fa_bin-15:.0f}% -> "
              f"{'PASS' if ok else 'no'}]")
        if ok:
            verdict = "PASS (review-frame spot earned)"
    print(f"\nCELL SIZES: right-success {len(c)}, wrong-success {len(w)} "
          f"(frozen rule: any cell < 30 -> INDETERMINATE)")
    print(f"VERDICT (frozen bar): {verdict}")

    # descriptive: floor disease + inversion on correct successes
    cs = [x for x in rows if x["said"] == "success" and x["truth_ok"]
          and x["E"] is not None]
    if cs:
        low_b = sum(1 for x in cs if x["p_done"] < 0.25)
        low_r = sum(1 for x in cs if x["E"] < 1.0)
        print(f"\ndescriptive on correct successes (n={len(cs)}): "
              f"binary p_done<0.25 {low_b} ({low_b/len(cs)*100:.0f}%), "
              f"rubric E<1.0 {low_r} ({low_r/len(cs)*100:.0f}%)")
    ws_ = [x for x in rows if x["said"] == "success" and not x["truth_ok"]]
    if ws_:
        inv = sum(1 for x in ws_ if isinstance(x["p_done"], (int, float))
                  and x["p_done"] >= 0.5)
        print(f"descriptive on WRONG successes (n={len(ws_)}): "
              f"binary p_done>=0.5 {inv} ({inv/len(ws_)*100:.0f}%)")

    # direction split on failure claims (descriptive)
    fc = [x for x in rows if x["said"] == "failure"]
    if fc:
        fw = [x for x in fc if x["truth_ok"]]  # said failure but passed
        print(f"descriptive failure claims (n={len(fc)}): truly-failed "
              f"{len(fc)-len(fw)}, truly-passed {len(fw)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "balance", "judge"])
    ap.add_argument("--bin", default=f"{MB}/scores_v3_bin.jsonl")
    ap.add_argument("--rub", default=f"{MB}/scores_v3_rub.jsonl")
    a = ap.parse_args()
    if a.stage == "extract":
        stage_extract()
    elif a.stage == "balance":
        stage_balance()
    else:
        stage_judge(a.bin, a.rub)
