"""S7 multimodal read test — D2 element selection over screenshot observations.

Every scored result in the PoC so far conditioned on text-DOM observations. The
checkpoint is multimodal; this experiment asks whether the DECIDE read changes
when the observation includes the rendered page:

  arm d2_text   context = task + history + DOM candidate lines (the standard read)
  arm d2_image  same context, but the user message carries the page screenshot
                as an image part; prefill runs with pixel_values, the candidate
                scoring is unchanged text-suffix machinery (batched mode)

Capture is fresh (playwright full-page screenshots of the L2 demo site) with
hand-authored gold — the Mind2Web HF release is text-only, so historical
screenshots are unavailable. Same rendering contract as everywhere else
(eval/livedemo.py's extractor; hybrid.actions renderers).

Usage on the GPU host:
  ~/hybrid-env/bin/python eval/s7_multimodal.py --capture          # no GPU
  ~/hybrid-env/bin/python eval/s7_multimodal.py                    # both arms
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("livedemo", ROOT / "eval" / "livedemo.py")
livedemo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(livedemo)

_spec2 = importlib.util.spec_from_file_location("s7_tasks", ROOT / "eval" / "s7_tasks.py")
s7_tasks = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(s7_tasks)
TASKS = s7_tasks.TASKS

PAGES = ["index.html", "loyalty.html", "profile.html"]
OUT = ROOT / "eval" / "reports" / "s7"


def capture():
    """No-GPU phase: full-page screenshots + live candidate sets per page."""
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    served = livedemo.serve_site(8792)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            obs = {}
            for pg in PAGES:
                page.goto(f"http://127.0.0.1:8792/{pg}", wait_until="networkidle")
                page.wait_for_timeout(200)
                img = OUT / f"{pg.replace('.html', '')}.png"
                page.screenshot(path=str(img), full_page=True)
                obs[pg] = {"image": str(img),
                           "candidates": page.evaluate(livedemo.EXTRACT_JS)}
                print(f"captured {pg}: {len(obs[pg]['candidates'])} candidates")
            browser.close()
    finally:
        served.join(timeout=0.1)
    (OUT / "s7_obs.jsonl").write_text(json.dumps(obs, ensure_ascii=False, indent=1))
    print(f"observations: {OUT / 's7_obs.jsonl'}")


def load_obs() -> dict:
    data = json.loads((OUT / "s7_obs.jsonl").read_text())
    for pg in PAGES:
        if pg not in data:
            raise SystemExit(f"missing observation for {pg} — run --capture first")
    return data


def run_arms(batched: bool):
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from hybrid.actions import WebStep, candidate_lines, context_text
    from hybrid.decide import decide
    from hybrid.engine import MODEL_ID, load_model
    from hybrid.scoring import score_sequence_logprob_batched
    from hybrid.template import PinnedPrompt, question_messages

    if batched:
        import os
        os.environ["HYBRID_BATCHED_D2"] = "1"

    obs = load_obs()
    loaded = load_model()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        print(f"image arm infeasible: no processor for {MODEL_ID}: {e!r}")
        processor = None

    records = []
    for t in TASKS:
        o = obs[t["page"]]
        cands = o["candidates"]
        if not any(c["bid"] == t["gold_bid"] for c in cands):
            print(f"skip {t['task']!r}: gold {t['gold_bid']} not in live extraction")
            continue
        step = WebStep(id=f"s7_{t['page'][:2]}_{t['gold_bid']}", task=t["task"],
                       history=[], op=t["op"], value=None, candidates=cands,
                       gold_bid=t["gold_bid"], n_candidates_total=len(cands))
        lines = candidate_lines(step)
        ctx = context_text(step)
        gold_line = lines[[c["bid"] for c in cands].index(t["gold_bid"])]

        # arm 1: text-DOM (standard read)
        r0 = time.perf_counter()
        res_t = decide(loaded, ctx, lines, method="d2", alpha=0.5, anchor=True)
        ms_t = (time.perf_counter() - r0) * 1000

        rec = {"page": t["page"], "task": t["task"], "op": t["op"],
               "gold_bid": t["gold_bid"], "n_candidates": len(lines),
               "text": {"sel": res_t.selected, "correct": res_t.selected == gold_line,
                        "conf": res_t.confidence, "premask": res_t.premask_mass,
                        "ms": round(ms_t, 1)}}

        # arm 2: screenshot-conditioned read (same candidates, image in context)
        if processor is not None:
            try:
                img = Image.open(o["image"]).convert("RGB")
                msgs = question_messages(ctx, lines, anchor=True)
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": msgs[0]["content"]},
                ]}]
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, enable_thinking=False,
                    tokenize=True, return_dict=True, return_tensors="pt").to(loaded.model.device)
                pinned = PinnedPrompt(input_ids=inputs["input_ids"], text="",
                                      read_mode="pinned_nothink", template_kwargs={})
                # mm_token_type_ids is REQUIRED by qwen3_5 when multimodal
                # payloads are passed (mrope can't be computed without it)
                prefill = {k: v for k, v in inputs.items()
                           if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}
                r0 = time.perf_counter()
                res_i = score_sequence_logprob_batched(
                    loaded.model, loaded.tokenizer, pinned, lines, alpha=0.5,
                    cache_mode="batched", prefill_kwargs=prefill)
                ms_i = (time.perf_counter() - r0) * 1000
                rec["image"] = {"sel": res_i.selected, "correct": res_i.selected == gold_line,
                                "conf": res_i.confidence, "premask": res_i.premask_mass,
                                "ms": round(ms_i, 1)}
            except Exception as e:  # noqa: BLE001
                rec["image"] = {"error": repr(e)[:300]}
        records.append(rec)
        print(f"[{len(records)}/{len(TASKS)}] {t['page']} '{t['task'][:40]}' "
              f"text={'OK' if rec['text']['correct'] else 'MISS'} "
              f"image={rec.get('image', {}).get('correct', 'n/a')}", flush=True)

    with open(OUT / "s7_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(records)


def report(records: list[dict]):
    def arm_stats(key):
        rows = [r[key] for r in records if key in r and "correct" in r[key]]
        if not rows:
            return None
        ms = sorted(r["ms"] for r in rows)
        return {"n": len(rows),
                "elem_acc": sum(r["correct"] for r in rows) / len(rows),
                "mean_conf": sum(r["conf"] for r in rows) / len(rows),
                "premask_ok": sum(r["premask"] >= 0.05 for r in rows) / len(rows),
                "p50_ms": ms[len(ms) // 2]}

    st_t, st_i = arm_stats("text"), arm_stats("image")
    lines = ["# S7 multimodal read test — screenshot-conditioned D2", "",
             "**Setup.** Fresh playwright captures of the L2 demo pages, hand-authored "
             "gold (Mind2Web HF is text-only), live-DOM candidate sets, same renderers "
             "as all web arms. d2_image = same pinned decision read with the page "
             "screenshot as an image part (prefill only; candidate scoring unchanged).", ""]
    if st_t:
        lines.append(f"- d2_text : n={st_t['n']}  elem_acc={st_t['elem_acc']:.3f}  "
                     f"conf={st_t['mean_conf']:.3f}  premask_ok={st_t['premask_ok']:.2f}  "
                     f"p50={st_t['p50_ms']:.0f}ms")
    if st_i:
        lines.append(f"- d2_image: n={st_i['n']}  elem_acc={st_i['elem_acc']:.3f}  "
                     f"conf={st_i['mean_conf']:.3f}  premask_ok={st_i['premask_ok']:.2f}  "
                     f"p50={st_i['p50_ms']:.0f}ms")
    lines += ["", "| page | task | gold | text pick | image pick |", "|---|---|---|---|---|"]
    for r in records:
        t_i = r.get("image", {})
        pick = str(t_i.get("sel", t_i.get("error", "n/a")))[:60]
        lines.append(f"| {r['page']} | {r['task'][:52]} | {r['gold_bid']} "
                     f"({'OK' if r['text']['correct'] else 'MISS'}) | {pick} |")
    (OUT / "s7_multimodal.md").write_text("\n".join(lines) + "\n")
    print(f"report: {OUT / 's7_multimodal.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="screenshots only (no GPU)")
    ap.add_argument("--seq", action="store_true", help="sequential D2 (default: batched)")
    args = ap.parse_args()
    if args.capture:
        capture()
        return
    run_arms(batched=not args.seq)


if __name__ == "__main__":
    main()
