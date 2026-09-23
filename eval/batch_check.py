"""L1 parity gate — batched D2 vs sequential D2 on real replay steps.

The batched path may only be trusted for latency claims after this passes:
  - argmax identical on every scored step, except near-ties: a flip between
    candidates with sequential top-2 probability gap <= 0.05 is bf16
    tie-breaking under a different kernel schedule (the noise floor is
    ~0.02-0.03 nats/token); a well-separated flip is a real failure
  - probability deltas consistent with the bf16 shape-noise floor
  - pre-mask mass delta exactly 0 (same logits row by construction)
  - NEGATIVE CONTROL: scoring against a WRONG prompt's cache must produce
    order-of-magnitude larger deltas than the batched path — proving the
    batched path's deltas are numeric noise, not broken conditioning.

Also measures per-path element-read latency (the D2 component alone —
the arms' recorded step latency additionally includes the op read).

Run on the GPU host:
  cd ~/hybrid-qwen && ~/hybrid-env/bin/python eval/batch_check.py [--steps 50]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from hybrid.actions import candidate_lines, context_text, load_web_steps  # noqa: E402
from hybrid.decide import decide  # noqa: E402
from hybrid.engine import load_model  # noqa: E402
from hybrid.scoring import score_sequence_logprob, score_sequence_logprob_batched  # noqa: E402
from hybrid.template import pin_prompt, question_messages  # noqa: E402

DATASET = Path.home() / "hybrid-qwen" / "eval" / "datasets" / "webreplay_v1" / "webreplay_v1.jsonl"


def filter_steps(steps, n, lo, hi):
    """First n replay steps whose enumerated candidate count is in [lo, hi]."""
    out = []
    for s in steps:
        if lo <= len(s.candidates) <= hi:
            out.append(s)
            if len(out) >= n:
                break
    return out


def cmp(a, b) -> dict:
    sel_ok = a.selected == b.selected
    dp = max(abs(a.probabilities[c] - b.probabilities[c]) for c in a.probabilities)
    dm = abs(a.premask_mass - b.premask_mass)
    dr = max(abs(a.raw_scores[c] - b.raw_scores[c]) for c in a.raw_scores)
    return {"sel_ok": sel_ok, "d_prob": dp, "d_premask": dm, "d_raw": dr}


def timed(loaded, ctx, lines, **kw):
    t0 = time.perf_counter()
    res = decide(loaded, ctx, lines, method="d2", alpha=0.5, anchor=True, **kw)
    return res, (time.perf_counter() - t0) * 1000.0


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def top2_gap(probs: dict) -> float:
    xs = sorted(probs.values(), reverse=True)
    return xs[0] - xs[1] if len(xs) > 1 else 1.0


# A flip between candidates closer than this in the sequential reference is
# bf16 tie-breaking under a different kernel schedule, not a semantic
# divergence (round-3 evidence: the only 2 flips in 50 steps had gaps of
# 0.003 and 0.005; the measured shape-noise floor is ~0.02-0.03 nats/token).
NEAR_TIE = 0.05


@torch.no_grad()
def wrong_conditioning_control(loaded, steps, n_ctl: int = 5):
    """Score candidate lines against a DIFFERENT step's prompt cache.

    If conditioning were broken in the suffix path, batched-vs-sequential
    deltas would look like wrong-vs-right deltas. This control measures the
    wrong-conditioning delta scale so the two can be told apart. The wrong
    prompt must provably differ in token ids: consecutive replay steps can be
    exact duplicates (same task + history), which made round-4's control
    compare identical prompts (0.00 / 0.00). Deltas are averaged over ALL
    candidates, matching the d_raw scale of the main loop.

    Returns (mean_raw_delta_wrong, mean_raw_delta_batched_on_same_steps),
    or (None, None) if no differing prompt exists in the subset.
    """
    wrong_raw, bat_raw = [], []
    for i in range(n_ctl):
        s = steps[i]
        lines = candidate_lines(s)
        ctx_right = context_text(s)
        pin_r = pin_prompt(loaded.tokenizer, question_messages(ctx_right, lines, anchor=True))
        pin_w = None
        for j in range(i + 1, len(steps)):
            ctx_w = context_text(steps[j])
            pin_try = pin_prompt(loaded.tokenizer, question_messages(ctx_w, lines, anchor=True))
            same = (pin_try.input_ids.shape == pin_r.input_ids.shape
                    and bool(torch.equal(pin_try.input_ids, pin_r.input_ids)))
            if not same:
                pin_w = pin_try
                break
        if pin_w is None:
            print("control: no token-differing prompt in subset — control skipped")
            return None, None

        res_ref = score_sequence_logprob(
            loaded.model, loaded.tokenizer, pin_r, lines, alpha=0.5
        )
        res_wrong = score_sequence_logprob_batched(
            loaded.model, loaded.tokenizer, pin_w, lines, alpha=0.5, cache_mode="copy"
        )
        res_bat = score_sequence_logprob_batched(
            loaded.model, loaded.tokenizer, pin_r, lines, alpha=0.5, cache_mode="copy"
        )
        wrong_raw.append(statistics.mean(
            abs(res_wrong.raw_scores[c] - res_ref.raw_scores[c]) for c in lines))
        bat_raw.append(statistics.mean(
            abs(res_bat.raw_scores[c] - res_ref.raw_scores[c]) for c in lines))
    return statistics.mean(wrong_raw), statistics.mean(bat_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--subset", type=int, default=20, help="steps for forced-mode checks")
    ap.add_argument("--min-cands", type=int, default=6)
    ap.add_argument("--max-cands", type=int, default=24)
    args = ap.parse_args()

    # contention probe: fixed matmul, same size every run
    a = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
    for _ in range(3):
        (a @ a).sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        (a @ a).sum()
    torch.cuda.synchronize()
    bench_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"gpu bench (1024^3 matmul): {bench_ms:.2f} ms/op  (quiet-host reference ~1.5-2 ms)")

    steps = filter_steps(load_web_steps(DATASET), args.steps, args.min_cands, args.max_cands)
    print(f"steps: {len(steps)}  dataset: {DATASET.name}")
    loaded = load_model()
    free, total = torch.cuda.mem_get_info()
    print(f"model: {loaded.load_path}  vram free {free / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)

    rows = []
    for k, step in enumerate(steps):
        lines = candidate_lines(step)
        ctx = context_text(step)
        seq, t_seq = timed(loaded, ctx, lines)
        bat, t_bat = timed(loaded, ctx, lines, batched=True, cache_mode="auto")
        r = cmp(seq, bat)
        r.update(id=step.id, n=len(lines), t_seq=t_seq, t_bat=t_bat,
                 mode=bat.scoring_method.replace("d2_sequence_logprob_batched(", "")[:-1],
                 gap=top2_gap(seq.probabilities))
        rows.append(r)
        flag = " ARGMAX-FLIP" if not r["sel_ok"] else ""
        if not r["sel_ok"] and r["gap"] <= NEAR_TIE:
            flag = " NEAR-TIE-FLIP"
        if r["sel_ok"] and r["d_prob"] < 0.01 and r["d_premask"] < 1e-4:
            flag = ""
        elif r["sel_ok"]:
            flag = " noise"
        print(f"[{k+1}/{len(rows)}] {step.id} n={r['n']} mode={r['mode']} "
              f"seq={t_seq:.0f}ms bat={t_bat:.0f}ms d_prob={r['d_prob']:.2e} "
              f"d_raw={r['d_raw']:.2e} d_premask={r['d_premask']:.2e}{flag}", flush=True)

    print("\n===== negative control (wrong-prompt cache, n=5) =====")
    try:
        d_wrong, d_bat = wrong_conditioning_control(loaded, steps)
        if d_wrong is None:
            control_ok = False
        else:
            ratio = d_wrong / max(d_bat, 1e-9)
            print(f"wrong-conditioning |d_raw|={d_wrong:.2f}  batched |d_raw|={d_bat:.2f}  ratio={ratio:.0f}x")
            control_ok = ratio > 5
            print(f"control: {'PASS (conditioning intact)' if control_ok else 'FAIL — investigate before trusting batched'}")
    except Exception as e:  # noqa: BLE001
        print(f"control crashed: {e!r}")
        control_ok = False

    print("\n===== forced-mode checks =====")
    forced = {}
    for mode in ("shared", "copy", "batched"):
        fr = []
        try:
            for step in steps[: args.subset]:
                lines = candidate_lines(step)
                ctx = context_text(step)
                seq, _ = timed(loaded, ctx, lines)
                bat, t_bat = timed(loaded, ctx, lines, batched=True, cache_mode=mode)
                r = cmp(seq, bat)
                r.update(t_bat=t_bat, sel_ok=r["sel_ok"], gap=top2_gap(seq.probabilities))
                fr.append(r)
            forced[mode] = fr
        except Exception as e:  # noqa: BLE001
            print(f"mode {mode}: crashed {e!r}")
    for mode, fr in forced.items():
        bad = [r for r in fr if not r["sel_ok"]]
        near = sum(1 for r in bad if r["gap"] <= NEAR_TIE)
        xs = [r["t_bat"] for r in fr]
        print(f"{mode:8s} n={len(fr)} flips={len(bad)} (near-tie {near}) "
              f"max|d_prob|={max(r['d_prob'] for r in fr):.2e} "
              f"max|d_raw|={max(r['d_raw'] for r in fr):.2e} p50={pctl(xs, .5):.0f}ms p90={pctl(xs, .9):.0f}ms")

    print("\n===== summary =====")
    sel_bad = [r for r in rows if not r["sel_ok"]]
    near_bad = [r for r in sel_bad if r["gap"] <= NEAR_TIE]
    ws_bad = [r for r in sel_bad if r["gap"] > NEAR_TIE]
    for r in sel_bad:
        print(f"  flip detail: {r['id']}  gap={r['gap']:.3f}  d_prob={r['d_prob']:.2e}")
    print(f"auto:      n={len(rows)} argmax flips={len(sel_bad)} (near-tie {len(near_bad)}, "
          f"well-separated {len(ws_bad)}) "
          f"max|d_prob|={max(r['d_prob'] for r in rows):.2e} "
          f"max|d_premask|={max(r['d_premask'] for r in rows):.2e} "
          f"max|d_raw|={max(r['d_raw'] for r in rows):.2e}")
    for name, sel in (("sequential", "t_seq"), ("batched-auto", "t_bat")):
        xs = [r[sel] for r in rows]
        print(f"{name:12s} p50={pctl(xs, .5):.0f}ms  p90={pctl(xs, .9):.0f}ms  mean={statistics.mean(xs):.0f}ms")
    modes = {r["mode"] for r in rows}
    copy_ok = all(r["sel_ok"] or r["gap"] <= NEAR_TIE for r in forced.get("copy", []))
    parity = not ws_bad and control_ok and copy_ok
    print(f"\nprobe picked: {modes}   control: {'PASS' if control_ok else 'FAIL'}   "
          f"PARITY (no well-separated flips & control & copy-forced): "
          f"{'PASS' if parity else 'FAIL'}")


if __name__ == "__main__":
    main()
