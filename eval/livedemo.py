"""L2 live three-mode demo — router.route() wired to a real browser.

One trajectory over a staged 4-page site (eval/livedemo_site/), exercising all
three runtime modes on LIVE DOM observations:

  s1 navigate   DECIDE      element selection over live candidates -> click
  s2 security   DECIDE      element selection among decoy actions -> click
  s3 dialog     DECIDE      pure decision: the two buttons of an open
                            confirmation dialog (aria-modal filter -> the page
                            behind the overlay is not offered as candidates)
  s4 fill email TOOL_CALL    op decide -> target decide (gated) -> op re-read
                            with the element in view -> THINK value
  s5 pick freq  TOOL_CALL    same chain for a SELECT; value = option label
  s6 summarize  TEXT        free text over the order-confirmation page

The live DOM extractor reproduces eval/datasets/build_webreplay.py's rendering
contract field-for-field (text policy: direct text -> one-level-down child
texts; parent/grandparent context; path; key attrs), so hybrid.actions renders
live candidates exactly like replayed ones.

Honesty rules: the model's choice is executed as-is and verified against gold;
on a miss the driver records the miss, then performs the gold action itself
(`recovered`) so the rest of the trajectory stays on script. Screenshots for
every step land in eval/reports/live_demo/.

Run on the GPU host (model + chromium):
  cd ~/hybrid-qwen && ~/hybrid-env/bin/python eval/livedemo.py --batched
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import functools
import http.server
from pathlib import Path

SITE = Path(__file__).resolve().parent / "livedemo_site"
REPORT_DIR = Path(__file__).resolve().parent / "reports" / "live_demo"

_BATCHED = False


# ---------------------------------------------------------------- extraction

EXTRACT_JS = r"""
() => {
  const SHORT = (s, n) => { s = (s || "").replace(/\s+/g, " ").trim();
                            return s.length > n ? s.slice(0, n - 1) + "…" : s; };
  const directText = (el) => Array.from(el.childNodes || [])
      .filter(n => n.nodeType === 3).map(n => (n.textContent || "").trim())
      .filter(Boolean).join(" ");
  const INTERACTIVE = new Set(["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]);
  const modal = document.querySelector('[aria-modal="true"]:not([hidden])');
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll("*")) {
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute("role") || "";
    if (!INTERACTIVE.has(el.tagName) && role !== "button" && role !== "tab" &&
        !el.hasAttribute("onclick")) continue;
    if ((el.getAttribute("type") || "").toLowerCase() === "hidden") continue;
    if (modal && !modal.contains(el)) continue;   // dialog open: it owns the page
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") continue;
    let own = directText(el);
    if (!own) own = Array.from(el.children)
        .map(ch => directText(ch)).filter(Boolean).join(" ");
    let text = SHORT(own, 80);
    if (!text && el.tagName === "INPUT") text = SHORT(el.getAttribute("value") || "", 80);
    const attrs = {};
    for (const k of ["aria-label", "placeholder", "title", "id", "href", "name", "type", "value"]) {
      const v = el.getAttribute(k);
      if (v) attrs[k] = SHORT(v, 40);
    }
    const path = [];
    let p = el.parentElement;
    while (p && p !== document.documentElement && path.length < 4) {
      path.push(p.tagName.toLowerCase()); p = p.parentElement;
    }
    const parent = el.parentElement, gp = parent ? parent.parentElement : null;
    const parent_text = SHORT(parent ? directText(parent) : "", 40);
    const gp_text = SHORT(gp ? directText(gp) : "", 40);
    const sub = SHORT((el.innerText || el.value || ""), 200);
    const bid = el.getAttribute("data-bid") || ("auto-" + out.length);
    const key = JSON.stringify([tag, text, path, parent_text]);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ bid, tag, text, attrs, path, parent_text, gp_text, sub });
  }
  return out;
}
"""


def serve_site(port: int) -> threading.Thread:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- scenario

class Step:
    def __init__(self, sid, page, task, history, mode, gold_bid=None,
                 expect_op=None, expect_value=None, exec_note=""):
        self.id, self.page, self.task, self.history = sid, page, task, history
        self.mode, self.gold_bid = mode, gold_bid
        self.expect_op, self.expect_value = expect_op, expect_value
        self.exec_note = exec_note


STEPS = [
    Step("s1_navigate", "index.html", "Open the loyalty program page.", [],
         "decide", gold_bid="b103", expect_op="CLICK",
         exec_note="Clicked 'Loyalty Rewards' in the site navigation"),
    Step("s2_security", "loyalty.html",
         "Sign out of all other sessions.",
         ["Clicked 'Loyalty Rewards' in the site navigation"],
         "decide", gold_bid="b223", expect_op="CLICK",
         exec_note="Clicked 'Sign out of all other sessions' in the Security section"),
    Step("s3_dialog", "loyalty.html",
         "Finish signing out of all other sessions using the confirmation dialog.",
         ["Clicked 'Loyalty Rewards' in the site navigation",
          "Clicked 'Sign out of all other sessions'",
          "A confirmation dialog opened"],
         "decide", gold_bid="b302", expect_op="CLICK",
         exec_note="Confirmed the sign-out in the dialog"),
    Step("s4_fill_email", "profile.html",
         "Subscribe to the newsletter with the email hana.mercer@example.com.",
         ["Opened the newsletter preferences page"],
         "tool_call", gold_bid="b411", expect_op="TYPE",
         expect_value="hana.mercer@example.com",
         exec_note="Typed the email into the Email address field"),
    Step("s5_pick_freq", "profile.html",
         "Choose the Weekly digest.",
         ["Opened the newsletter preferences page",
          "Typed the email into the Email address field"],
         "tool_call", gold_bid="b413", expect_op="SELECT",
         expect_value="Weekly digest",
         exec_note="Selected 'Weekly digest' in the Digest frequency dropdown"),
    Step("s6_summarize", "order.html",
         "Summarize this order confirmation in two sentences.",
         ["The order confirmation page is open"],
         "text"),
]


# ---------------------------------------------------------------- webstep glue

def to_webstep(step, cands):
    from hybrid.actions import WebStep
    return WebStep(
        id=step.id, task=step.task, history=list(step.history),
        op=step.expect_op or "CLICK", value=step.expect_value,
        candidates=cands, gold_bid=step.gold_bid or "",
        n_candidates_total=len(cands),
    )


def make_reread(webstep, lines):
    """H2_web escalation: rich top-3 blocks as replacement candidates."""
    from hybrid.actions import rich_blocks

    def reread(top_lines):
        bids = [webstep.candidates[lines.index(l)]["bid"] for l in top_lines if l in lines]
        blocks = rich_blocks(webstep, bids)
        return "\n\nFocus on these options:\n" + "\n".join(blocks), blocks

    return reread


# ---------------------------------------------------------------- execution

def bid_sel(bid):
    return f'[data-bid="{bid}"]'


def modal_open(page):
    return page.evaluate(
        "() => { const m = document.querySelector('[aria-modal=\"true\"]');"
        " return !!m && !m.hidden; }")


def note_visible(page):
    return page.evaluate("() => !document.getElementById('signed-out-note').hidden")


def exec_click(page, bid):
    page.click(bid_sel(bid), timeout=4000)


def exec_type(page, bid, value):
    page.click(bid_sel(bid), timeout=4000)
    page.fill(bid_sel(bid), value)


def exec_select(page, bid, value):
    sel = bid_sel(bid)
    try:
        page.select_option(sel, label=value)
        return value
    except Exception:
        labels = page.evaluate(
            "() => Array.from(document.querySelectorAll('select option'))"
            ".map(o => o.textContent.trim())")
        match = next((l for l in labels if value and value.lower() in l.lower()), None)
        if match is None:
            raise
        page.select_option(sel, label=match)
        return match


def _norm(s):
    return " ".join(str(s or "").split()).lower()


def verify(step, page, rec):
    """Post-action verification. Returns (ok, detail)."""
    if step.mode == "decide":
        chosen_bid = rec.get("chosen_bid")
        ok = chosen_bid == step.gold_bid
        detail = f"clicked bid {chosen_bid} vs gold {step.gold_bid}"
        if step.id == "s1_navigate":
            ok = ok and page.url.endswith("loyalty.html")
            detail += f", url now {page.url.rsplit('/', 1)[-1]}"
        if step.id == "s2_security":
            ok = ok and modal_open(page)
            detail += f", modal_open={modal_open(page)}"
        if step.id == "s3_dialog":
            ok = ok and note_visible(page)
            detail += f", signed-out note visible={note_visible(page)}"
        return ok, detail
    if step.mode == "tool_call":
        op_ok = rec.get("op_pred") == step.expect_op
        tgt_ok = rec.get("target_bid") == step.gold_bid
        detail = f"op {rec.get('op_pred')} (gold {step.expect_op}); " \
                 f"target {rec.get('target_bid')} vs gold {step.gold_bid}"
        if step.expect_value is not None:
            if step.expect_op == "TYPE":
                cur = page.input_value(bid_sel(step.gold_bid))
                val_ok = _norm(rec.get("value")) == _norm(step.expect_value) and \
                    _norm(cur) == _norm(step.expect_value)
                detail += f"; value '{rec.get('value')}' in field '{cur}'"
            else:
                cur = page.eval_on_selector(
                    bid_sel(step.gold_bid), "e => e.options[e.selectedIndex].textContent.trim()")
                val_ok = cur == step.expect_value
                detail += f"; dropdown now '{cur}'"
            return op_ok and tgt_ok and val_ok, detail
        return op_ok and tgt_ok, detail
    return True, "text output recorded"


def recover(step, page):
    """Driver performs the gold action so the trajectory stays on script."""
    if step.id == "s3_dialog":
        if not modal_open(page) and not note_visible(page):
            exec_click(page, "b223")  # reopen the dialog first
        if not note_visible(page):
            exec_click(page, step.gold_bid)
        return
    if step.mode == "decide":
        exec_click(page, step.gold_bid)
    elif step.mode == "tool_call" and step.expect_op == "TYPE":
        exec_type(page, step.gold_bid, step.expect_value)
    elif step.mode == "tool_call" and step.expect_op == "SELECT":
        exec_select(page, step.gold_bid, step.expect_value)


# ---------------------------------------------------------------- main

def main():
    global _BATCHED
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--batched", action="store_true",
                    help="HYBRID_BATCHED_D2=1 (requires the parity gate)")
    ap.add_argument("--out", default=str(REPORT_DIR))
    ap.add_argument("--extract-only", action="store_true",
                    help="no model: start the site, extract candidates, check gold bids")
    args = ap.parse_args()
    _BATCHED = args.batched
    if _BATCHED:
        os.environ["HYBRID_BATCHED_D2"] = "1"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    OPS = ("CLICK", "TYPE", "SELECT")
    if not args.extract_only:
        from hybrid.actions import candidate_lines, context_text
        from hybrid.engine import load_model
        from hybrid.router import DecideRequest, Gate, TextRequest, ToolCallRequest, route

        gate = Gate.from_json(Path(__file__).resolve().parents[1] / "configs" / "webgate.json")
        OP_CANDIDATES = (
            "CLICK — press or activate the target element (buttons, links, icons)",
            "TYPE — enter text into the target element (input fields, search boxes)",
            "SELECT — choose an option from the target element (dropdowns, comboboxes)",
        )
        loaded = load_model()

    serve_site(args.port)
    records = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        base = f"http://127.0.0.1:{args.port}"
        current_page = None

        for step in STEPS:
            # continue on the live page when the trajectory stays put — a
            # reload would reset the dialog state and any form fields the
            # agent filled earlier
            if step.page != current_page:
                page.goto(f"{base}/{step.page}", wait_until="networkidle")
                page.evaluate("() => { document.documentElement.style.scrollBehavior = 'auto'; }")
                current_page = step.page
            if step.id == "s3_dialog" and not modal_open(page):
                exec_click(page, "b223")  # setup: history says the dialog opened
            cands = page.evaluate(EXTRACT_JS)
            if args.extract_only:
                gold_ok = (not step.gold_bid) or any(c["bid"] == step.gold_bid for c in cands)
                print(f"[{step.id}] page={step.page} candidates={len(cands)} "
                      f"gold={step.gold_bid or 'n/a'} gold_present={gold_ok}", flush=True)
                if step.gold_bid and not gold_ok:
                    print("  FAIL: gold bid missing from live extraction", flush=True)
                page.screenshot(path=str(outdir / f"{step.id}.png"))
                continue
            webstep = to_webstep(step, cands)
            lines = candidate_lines(webstep)
            rec = {"id": step.id, "page": step.page, "mode": step.mode,
                   "task": step.task, "n_candidates": len(cands)}
            outcome = None

            if step.mode == "decide":
                req = DecideRequest(
                    question=context_text(webstep), candidates=lines,
                    gate=gate, strategy="reread", reread=make_reread(webstep, lines))
                outcome = route(loaded, req)
                chosen_bid = None
                if outcome.choice in lines:
                    chosen_bid = cands[lines.index(outcome.choice)]["bid"]
                outcome.chosen_bid = chosen_bid
                rec.update(choice=outcome.choice, chosen_bid=chosen_bid,
                           confidence=outcome.confidence, escalated=outcome.escalated,
                           n_decisions=outcome.n_decisions,
                           latency_ms=round(outcome.latency_ms, 1))
                if chosen_bid:
                    try:
                        exec_click(page, chosen_bid)
                    except Exception as e:  # noqa: BLE001 - unclickable choice is a miss
                        rec["exec_error"] = repr(e)

            elif step.mode == "tool_call":
                def value_question(tool, target_line):
                    op = OPS[tool]
                    where = "type into the field" if op == "TYPE" else "choose from the dropdown"
                    return (f"{context_text(webstep, op)}\n\n"
                            f"The chosen target element is: {target_line}\n\n"
                            f"Reply with only the value to {where} — exactly as it appears in "
                            f"the task above. No quotes, no explanation.")

                def tool_reread(target_line):
                    # H3B: final op decided with the chosen element in view —
                    # op-first alone mispredicts TYPE/SELECT without it
                    ctx = (context_text(webstep)
                           + "\n\nOptions:\n" + "\n".join(lines)
                           + f"\n\nThe chosen element is: {target_line}")
                    return ctx, list(OP_CANDIDATES)

                req = ToolCallRequest(
                    tool_question=context_text(webstep),
                    tool_candidates=list(OP_CANDIDATES),
                    target_question=lambda t: context_text(webstep, op=OPS[t]),
                    target_candidates=lambda t: lines,
                    gate=gate, target_reread=make_reread(webstep, lines),
                    tool_reread=tool_reread,
                    value_needed=lambda t: OPS[t] in ("TYPE", "SELECT"),
                    value_question=value_question,
                    target_to_ref=lambda line: cands[lines.index(line)]["bid"]
                        if line in lines else None,
                    beam_ops=True,
                )
                outcome = route(loaded, req)
                tc = outcome.tool_call
                op_pred = OPS[tc["tool"]] if tc["tool"] is not None else None
                rec.update(op_pred=op_pred, op_gold=step.expect_op,
                           target_bid=tc["target"], value=tc["value"],
                           confidence=outcome.confidence, escalated=outcome.escalated,
                           n_decisions=outcome.n_decisions, think_ran=outcome.think_ran,
                           latency_ms=round(outcome.latency_ms, 1))
                if tc["target"]:
                    try:
                        if op_pred == "TYPE":
                            exec_type(page, tc["target"], tc["value"] or "")
                        elif op_pred == "SELECT":
                            tc["value"] = exec_select(page, tc["target"], tc["value"] or "")
                            rec["value"] = tc["value"]
                        else:
                            exec_click(page, tc["target"])
                    except Exception as e:  # noqa: BLE001
                        rec["exec_error"] = repr(e)

            else:  # text
                body = page.evaluate("() => document.body.innerText")
                body = re.sub(r"\s+", " ", body).strip()[:1800]
                req = TextRequest(
                    messages=[{"role": "user", "content": f"{body}\n\nTask: {step.task}"}],
                    think=False, max_new_tokens=160)
                outcome = route(loaded, req)
                rec.update(text=outcome.text, think_ran=outcome.think_ran,
                           latency_ms=round(outcome.latency_ms, 1))

            # verify + recover
            if step.mode != "text":
                ok, detail = verify(step, page, rec)
                rec["verified"] = ok
                rec["verification"] = detail
                rec["recovered"] = not ok
                if not ok:
                    recover(step, page)
                step.history.append(step.exec_note)
            rec["events"] = outcome.events
            page.screenshot(path=str(outdir / f"{step.id}.png"))
            records.append(rec)
            print(f"[{step.id}] mode={step.mode} ok={rec.get('verified')} "
                  f"lat={rec.get('latency_ms')}ms", flush=True)

        browser.close()

    if args.extract_only:
        print("extract-only smoke pass complete")
        return

    with open(outdir / "trajectory.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    write_report(outdir, records)
    print(f"report: {outdir / 'live_demo.md'}")


def write_report(outdir: Path, records: list[dict]):
    def fmt(rec):
        mode = rec["mode"]
        if mode == "decide":
            head = (f"clicked `{rec.get('chosen_bid')}` (“{rec.get('choice')}”), "
                    f"confidence {rec.get('confidence'):.3f}")
        elif mode == "tool_call":
            head = (f"op {rec.get('op_pred')} (gold {rec.get('op_gold')}), target "
                    f"`{rec.get('target_bid')}`, value `{rec.get('value')}`")
        else:
            head = "free text below"
        esc = f", escalated={rec.get('escalated')}" if mode != "text" else ""
        lines = [
            f"### {rec['id']} — {mode.upper()} mode",
            f"- Task: {rec['task']}",
            f"- Candidates offered: {rec.get('n_candidates', '—')}",
            f"- Result: {head}{esc}, {rec.get('n_decisions', 1)} decision(s), "
            f"latency {rec.get('latency_ms')} ms",
        ]
        if mode != "text":
            mark = "✅" if rec.get("verified") else "❌"
            lines.append(f"- Verification: {mark} {rec.get('verification')}"
                         + (" — driver executed the gold action to keep the trajectory on script"
                            " (`recovered`)" if rec.get("recovered") else ""))
        else:
            lines.append(f"- Output: {rec.get('text')}")
        lines.append(f"- Events: `{json.dumps(rec.get('events'), default=str)[:400]}`")
        lines.append(f"- Screenshot: ![{rec['id']}](live_demo/{rec['id']}.png)")
        return "\n".join(lines)

    n_ok = sum(1 for r in records if r["mode"] != "text" and r.get("verified"))
    n_scored = sum(1 for r in records if r["mode"] != "text")
    rows = "\n".join(
        f"| {r['id']} | {r['mode']} | {r.get('n_candidates', '—')} | "
        f"{r.get('n_decisions', 1)} | {r.get('escalated', '—')} | "
        f"{r.get('think_ran', '—')} | {r.get('latency_ms')} ms | "
        f"{'pass' if r.get('verified') else ('text' if r['mode'] == 'text' else 'MISS')} |"
        for r in records)
    md = f"""# Live three-mode demo — one browser trajectory

**Setup.** Qwen3.5-4B frozen (bf16); staged site served from `eval/livedemo_site/`
on 127.0.0.1; candidates extracted from the LIVE DOM by `eval/livedemo.py`'s
extractor (same rendering contract as the Mind2Web replay pipeline); escalation
gate `configs/webgate.json`; batched D2: {"ON" if _BATCHED else "off"}.

**Trajectory.** Aurora Books storefront -> loyalty page -> sign-out confirmation
dialog -> newsletter form (email + digest frequency) -> order-confirmation summary.
Modes exercised: DECIDE (element selection; pure dialog decision), TOOL_CALL
(op -> gated target -> THINK value), TEXT (summary).

| step | mode | candidates | decisions | escalated | think | latency | verify |
|---|---|---|---|---|---|---|---|
{rows}

**Scored steps: {n_ok}/{n_scored} passed** (verification = the executed action had the
gold effect in the real browser: right navigation, dialog state, field value, dropdown).

{chr(10).join(chr(10) + fmt(r) for r in records)}
"""
    (outdir / "live_demo.md").write_text(md)


if __name__ == "__main__":
    main()
