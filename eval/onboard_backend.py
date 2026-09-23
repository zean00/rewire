#!/usr/bin/env python3
"""Onboard a new llama.cpp (OpenAI-compatible) backend for the decide chain.

One offline command against a live endpoint — no in-process model, no GPU
swap beyond whatever is already serving — that answers the PER-MODEL part of
the chain-port checklist and emits a profile the chain-on-llama.cpp port will
consume for its per-backend constants:

  1. THINKING CONTROL (per-model) — does the model think by default, and
     which request knob reliably yields non-empty content? Detected, not
     assumed: the knob list is tried in order until one works.
  2. SCORER FEASIBILITY (per-model) — can the chain's two decision reads run
     over HTTP at all, and what do they cost?
       d1 (op read):     candidate first tokens visible in top-logprobs,
                         margin between the best two, warm round-trip cost.
       d2 (element read): forced-decoding cost per scored token (grammar-
                         forced single tokens over a warm prefix cache),
                         extrapolated to a full 20-candidate decision.
  3. DISCRIMINATION CHECK (--cal-check via --cal-set, optional) — on the CAL
     SPLIT of webreplay_v1 ONLY (even lines; the odd-line test tasks are
     never read — standing no-overfit rule, enforced by construction): does
     d1 pick the gold op, and at what measured price per element decision on
     a small forced-decoding sample?

What this script deliberately does NOT do: fit the chain's gate constants
(accept gate, answer threshold, temperatures). Gate constants calibrated
without the full chain decision context did not transfer before (the H_web
lesson); fitting belongs to the chain port's own calibration step, over the
same cal split, once the port exists. This script measures; it does not tune.

Usage:
  python3 eval/onboard_backend.py --base-url http://127.0.0.1:8998/v1 \
      --model gemma-4-12b-it \
      [--cal-set eval/datasets/webreplay_v1/webreplay_v1.jsonl] \
      [--out eval/backend_profiles/<model>.json]

Exit 0 = onboarding ran to a verdict (any verdict); 1 = endpoint unreachable.
"""
import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Protocol vocabulary only — no site names, no task nouns (same discipline as
# the chain and the anti-overfit lint; the probes must never encode a task).
OP_LINES = ("CLICK — press/activate the selected element\n"
            "TYPE — enter a value into the selected element\n"
            "SELECT — choose an option from the selected element\n"
            "ANSWER — no element action is needed this turn\n")
OPS = ("CLICK", "TYPE", "SELECT")

# Knobs tried in order until one yields non-empty content on this backend.
THINK_KNOBS = [
    ("reasoning_budget_tokens", {"reasoning_budget_tokens": 0}),
    ("enable_thinking_prefill", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("include_reasoning", {"include_reasoning": False}),
]

HARD_PROMPT = ("A user forgot their password. In ONE short sentence, what "
               "should they do first?")


def endpoint(base_url, path):
    """Join base and path without doubling the version prefix (accepts both
    http://host:8998 and http://host:8998/v1 as --base-url)."""
    base = base_url.rstrip("/")
    if path.startswith("/v1/") and base.endswith("/v1"):
        path = path[len("/v1"):]
    return base + path


def post(base_url, path, body, timeout):
    req = urllib.request.Request(
        endpoint(base_url, path), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def chat(base_url, model, messages, timeout, **extra):
    body = {"model": model, "messages": messages, "temperature": 0}
    body.update(extra)
    return post(base_url, "/v1/chat/completions", body, timeout)


def content_of(payload):
    try:
        return payload["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def generated_logprob(payload):
    """Logprob of the actually-generated first token (grammar-forced too)."""
    try:
        return payload["choices"][0]["logprobs"]["content"][0]["logprob"]
    except Exception:
        return None


def top_logprobs_of(payload):
    """First generated token's top-N logprobs as {token: logprob}."""
    try:
        lp = payload["choices"][0].get("logprobs") or {}
        toks = lp.get("content") or []
        if not toks:
            return {}
        return {t.get("token", ""): t.get("logprob")
                for t in (toks[0].get("top_logprobs") or [])}
    except Exception:
        return {}


# --------------------------------------------------------------- probes

def probe_chat(base_url, model, timeout):
    _, dt = chat(base_url, model, [{"role": "user", "content":
                                    "Reply with exactly: ok"}], timeout,
                 max_tokens=8)
    return {"warm_ms": round(dt * 1000)}


def probe_thinking(base_url, model, timeout):
    """Detect thinking-by-default and the first knob that yields content."""
    msgs = [{"role": "user", "content": HARD_PROMPT}]
    payload, _ = chat(base_url, model, msgs, timeout, max_tokens=256)
    thinking_by_default = not content_of(payload).strip()
    knob_name, knob = "none", {}
    if thinking_by_default:
        for name, k in THINK_KNOBS:
            try:
                p, _ = chat(base_url, model, msgs, timeout, max_tokens=256, **k)
            except urllib.error.HTTPError:
                continue  # knob unknown to this backend — try the next
            if content_of(p).strip():
                knob_name, knob = name, k
                break
    # the chosen knob must keep the tool-call path working (generic tool)
    tools = [{"type": "function", "function": {
        "name": "catalog_scan",
        "description": "Scan the widget catalogue.",
        "parameters": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["full", "poll"]}},
            "required": ["mode"]}}}]
    try:
        p, _ = chat(base_url, model,
                    [{"role": "user", "content":
                      "Scan the catalogue with full mode please."}],
                    timeout, max_tokens=128, tools=tools, **knob)
        tcs = p["choices"][0]["message"].get("tool_calls") or []
        tool_ok = bool(tcs)
    except Exception:
        tool_ok = False
    return {"thinking_by_default": thinking_by_default,
            "knob": knob_name, "knob_payload": knob, "tool_call_ok": tool_ok}


def _d1_prompts():
    """Generic element-line probes; distinct-first-word op candidates."""
    out = []
    for task, line in [
        ("choose the first item in the list",
         "- button 'Add' [ref=e1]\n- link 'Item one' [ref=e2]\n"
         "- textbox 'Search' [ref=e3]"),
        ("enter a word into the search field",
         "- textbox 'Search' [ref=e5]\n- button 'Go' [ref=e6]"),
        ("pick one of the options shown in the menu",
         "- combobox 'Menu' [ref=e7]\n- button 'Save' [ref=e8]"),
        ("say the task is finished",
         "- button 'Add' [ref=e9]\n- link 'Item two' [ref=e10]"),
    ]:
        out.append([
            {"role": "system", "content":
                "You are a browser-automation controller. The user gives a "
                "page snapshot and a task. Reply with EXACTLY ONE word: the "
                "name of the operation to perform next."},
            {"role": "user", "content":
                f"Task: {task}.\n\nPage snapshot:\n{line}\n\nOperations:\n"
                f"{OP_LINES}\nNext operation (one word):"}])
    return out


def probe_d1(base_url, model, knob, timeout):
    """Op-read scorer: candidate first tokens visible in top-logprobs?"""
    cands = ["CLICK", "TYPE", "SELECT", "ANSWER"]
    visible, margins, times = [], [], []
    for i, msgs in enumerate(_d1_prompts()):
        payload, dt = chat(base_url, model, msgs, timeout, max_tokens=1,
                           logprobs=True, top_logprobs=20, **knob)
        if i:  # first call warms the prefix cache — timing starts after it
            times.append(dt)
        top = top_logprobs_of(payload)
        found = {c: top[c] for c in cands if c in top}
        visible.append(len(found))
        if len(found) >= 2:
            ordered = sorted(found.values(), reverse=True)
            margins.append(ordered[0] - ordered[1])
    return {
        "n_probes": 4,
        "visible_mean": round(statistics.mean(visible), 2) if visible else 0,
        "visible_rate": round(sum(1 for v in visible if v >= 1) / len(visible), 2)
                        if visible else 0,
        "median_margin": round(statistics.median(margins), 3) if margins else None,
        "warm_ms": round(statistics.median(times) * 1000) if times else None,
    }


def _forced_score(base_url, model, knob, messages, forced_word, timeout):
    """One grammar-forced token; returns its logprob or None on hard failure."""
    body = {"model": model, "messages": messages, "max_tokens": 1,
            "logprobs": True, "top_logprobs": 20, "temperature": 0,
            "grammar": json.dumps(forced_word)}
    body.update(knob)
    try:
        payload, _ = post(base_url, "/v1/chat/completions", body, timeout)
    except urllib.error.HTTPError:
        body.pop("grammar")  # endpoint rejects grammar — unforced fallback
        try:
            payload, _ = post(base_url, "/v1/chat/completions", body, timeout)
        except Exception:
            return None
    except Exception:
        return None
    return generated_logprob(payload)


def probe_d2(base_url, model, knob, timeout, budget_ms=2000):
    """Element-read scorer: forced-decoding cost per scored token.

    Times k sequential grammar-forced single-token requests over a growing
    shared prefix (warm prefix cache). The per-decision extrapolation is
    timing only — the real d2 scoring semantics land with the port.
    """
    words = [w for w in
             "- button 'Confirm the selection change' [ref=e14]".split(" ")
             if w]
    times = []
    for k in range(1, 7):
        prefix = " ".join(words[:k])
        msgs = [
            {"role": "system", "content":
                "You are a browser-automation controller. Score the candidate "
                "line silently; do not explain."},
            {"role": "user", "content":
                f"Task: choose the first item in the list.\n\nLine so far:\n"
                f"{prefix}"}]
        t0 = time.time()
        lp = _forced_score(base_url, model, knob, msgs, words[k], timeout) \
            if k < len(words) else None
        if lp is None and k == 1:
            return {"path": "unavailable"}
        if k > 1:  # first request warms the prefix cache
            times.append(time.time() - t0)
    ms = statistics.median(times) * 1000 if times else None
    est = round(20 * 6 * ms) if ms else None  # 20 cands x ~6 scored tokens
    return {"path": "grammar_forced", "n_forced": 12,
            "ms_per_forced_token": round(ms) if ms else None,
            "est_decision_ms": est,
            "within_budget": bool(est is not None and est <= budget_ms)}


# ------------------------------------------------- cal split discrimination

def _cal_items(path):
    """webreplay_v1 cal split ONLY: even lines (odd lines are the test split
    and are never read — by construction, not by convention)."""
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 2 == 0 and line.strip():
                items.append(json.loads(line))
    return items


def _render_item(item):
    cands = "\n".join(
        f"- [{c.get('tag','')}] '{(c.get('text') or '').strip()}' "
        f"[ref={c.get('bid')}]" for c in item["candidates"])
    hist = "\n".join(item.get("history") or [])
    return (f"Task: {item['task']}\n\nActions so far:\n{hist or '(none)'}\n\n"
            f"Page elements:\n{cands}")


def cal_check(base_url, model, knob, cal_path, n_op, n_el, timeout, seed):
    """Does d1 pick the gold op on cal items; what does one element decision
    cost with word-level forced scoring? Informational only — no thresholds
    are fitted here (see module docstring)."""
    items = _cal_items(cal_path)
    rng = random.Random(seed)
    op_items = rng.sample(items, min(n_op, len(items)))
    op_ok = 0
    for item in op_items:
        msgs = [
            {"role": "system", "content":
                "You are a browser-automation controller. Reply with EXACTLY "
                "ONE word: the operation to perform next."},
            {"role": "user", "content":
                _render_item(item) +
                "\n\nOperations:\nCLICK — press/activate the selected "
                "element\nTYPE — enter a value into the selected element\n"
                "SELECT — choose an option from the selected element\n"
                "\nNext operation (one word):"}]
        try:
            payload, _ = chat(base_url, model, msgs, timeout, max_tokens=1,
                              logprobs=True, top_logprobs=20, **knob)
        except Exception:
            continue
        top = top_logprobs_of(payload)
        visible = {c: top[c] for c in OPS if c in top}
        if visible and max(visible, key=visible.get) == item["op"]:
            op_ok += 1
    op_acc = round(op_ok / len(op_items), 3) if op_items else None

    # element decision: word-level forced scoring over a few CLICK items — a
    # price-and-signal sample, not the port's scorer
    el_pool = [it for it in rng.sample(items, min(4 * n_el, len(items)))
               if it["op"] == "CLICK"][:n_el]
    el_ok = el_n = 0
    el_ms = []
    for item in el_pool:
        t0 = time.time()
        best_bid, best_score = None, None
        for cand in item["candidates"]:
            words = [w for w in (cand.get("text") or "").split(" ") if w][:6]
            if not words:
                words = ["item"]
            score = 0.0
            broken = False
            for j, w in enumerate(words):
                prefix = " ".join(words[:j])
                msgs = [
                    {"role": "system", "content":
                        "Score the element line silently; do not explain."},
                    {"role": "user", "content":
                        f"Task: {item['task']}\n\nLine so far:\n"
                        f"- [{cand.get('tag','')}] {prefix}".rstrip()}]
                lp = _forced_score(base_url, model, knob, msgs, w, timeout)
                if lp is None:
                    broken = True
                    break
                score += lp
            if broken:
                continue
            if best_score is None or score > best_score:
                best_score, best_bid = score, cand.get("bid")
        if best_bid is not None:
            el_n += 1
            if str(best_bid) == str(item["gold_bid"]):
                el_ok += 1
        el_ms.append(time.time() - t0)
    return {
        "split": "cal-only (even lines; test split never read)",
        "op_accuracy": op_acc, "op_n": len(op_items),
        "el_accuracy": round(el_ok / el_n, 3) if el_n else None,
        "el_n": el_n,
        "el_s_per_decision": round(statistics.mean(el_ms), 1) if el_ms else None,
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None,
                    help="profile path (default eval/backend_profiles/"
                         "<model>.json)")
    ap.add_argument("--cal-set", default=None,
                    help="webreplay_v1.jsonl — enables the discrimination "
                         "check (cal split only)")
    ap.add_argument("--cal-n", type=int, default=60)
    ap.add_argument("--el-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget-ms", type=int, default=2000,
                    help="per-decision d2 cost that still counts as "
                         "CHAIN_READY")
    args = ap.parse_args()

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f": {detail}" if detail else ""), flush=True)

    try:
        pc = probe_chat(args.base_url, args.model, args.timeout)
        check("chat reachable", True, f"warm {pc['warm_ms']} ms")
    except Exception as e:
        check("chat reachable", False, repr(e)[:120])
        print("VERDICT: UNREACHABLE")
        return 1

    pt = probe_thinking(args.base_url, args.model, args.timeout)
    check("thinking control", pt["tool_call_ok"],
          f"thinking_by_default={pt['thinking_by_default']} "
          f"knob={pt['knob']} tool_calls={pt['tool_call_ok']}")

    pd1 = probe_d1(args.base_url, args.model, pt["knob_payload"],
                   args.timeout)
    check("d1 op-read scorer", pd1["visible_rate"] >= 0.75
          and (pd1["median_margin"] or 0) > 0.25,
          f"visible={pd1['visible_mean']}/4 rate={pd1['visible_rate']} "
          f"margin={pd1['median_margin']} warm={pd1['warm_ms']} ms")

    pd2 = probe_d2(args.base_url, args.model, pt["knob_payload"],
                   args.timeout, args.budget_ms)
    check("d2 element-read cost", pd2.get("path") not in
          ("unavailable", None) and pd2.get("within_budget"),
          f"path={pd2.get('path')} "
          f"{pd2.get('ms_per_forced_token')} ms/token -> "
          f"~{pd2.get('est_decision_ms')} ms/decision "
          f"(budget {args.budget_ms} ms)")

    cal = None
    if args.cal_set:
        cal = cal_check(args.base_url, args.model, pt["knob_payload"],
                        args.cal_set, args.cal_n, args.el_n, args.timeout,
                        args.seed)
        check("cal discrimination (op read)", (cal["op_accuracy"] or 0) >= 0.5,
              f"op {cal['op_accuracy']} on n={cal['op_n']}; element "
              f"{cal['el_accuracy']} on n={cal['el_n']} at "
              f"{cal['el_s_per_decision']} s/decision")

    if pd1["visible_rate"] >= 0.75 and pd2.get("within_budget"):
        verdict = "CHAIN_READY"
    elif pd1["visible_rate"] >= 0.75:
        verdict = "CHAIN_FEASIBLE_SLOW"
    elif pd1["visible_rate"] >= 0.5:
        verdict = "PARTIAL_D1_ONLY"
    else:
        verdict = "NO_SCORER"
    print(f"VERDICT: {verdict}", flush=True)

    profile = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url, "model": args.model,
        "thinking": pt, "d1_op_read": pd1, "d2_element_read": pd2,
        "cal_check": cal, "verdict": verdict,
        "note": "no gate constants fitted here — fitting belongs to the "
                "chain port's calibration step over the same cal split",
    }
    out = Path(args.out) if args.out else (
        ROOT / f"eval/backend_profiles/{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    print(f"profile -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
