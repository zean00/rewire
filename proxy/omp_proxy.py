"""OpenAI-compatible proxy in front of the hybrid runtime — the omp three-arm run.

One server, three model names; omp picks the arm via --model poc-proxy/<name>:

  qwen35-4b-vanilla          passthrough: chat template (with omp's tool schemas)
                             + greedy generate, native thinking (Baseline-G config)
  qwen35-4b-vanilla-nothink  passthrough, enable_thinking=False (Baseline-A config)
  qwen35-4b-decide           deterministic structural router. Turns whose last
                             message is a browser observation (aria snapshot with
                             [ref=eN]) run the decide chain — op-first read over
                             {CLICK, TYPE, SELECT, ANSWER} -> op-conditioned
                             element read (batched D2, gate logged not applied) ->
                             op re-read with the element in view -> think value —
                             and the answer is returned as a CONSTRUCTED one-action
                             eval cell (tab.click/fill/select on "aria-ref=eN", each
                             cell ending with a fresh ariaSnapshot so the next turn
                             has candidates). ANSWER wins -> no-think generation.
                             Every other turn passes through (no-think) unchanged.

The harness (omp) is untouched: it sees a standard OpenAI-compatible endpoint and
a model that writes one-line browser actions. Same loop, same tools, same prompts
across arms — only the model endpoint differs.

Endpoints: GET /v1/models, GET /health, POST /v1/chat/completions (stream and
non-stream). Decision log: eval/reports/omp_arms/decisions.jsonl.

Usage (on the GPU host):
  python proxy/omp_proxy.py --port 8999
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HYBRID_BATCHED_D2", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from hybrid.actions import OP_CANDIDATES, OPS  # noqa: E402
from hybrid.decide import decide  # noqa: E402
from hybrid.decisions import DecisionResult, with_temperature  # noqa: E402
from hybrid.engine import generate_answer, load_model  # noqa: E402

# (v3.12.0) learned-adapter plumbing — stdlib-only sibling module; its absence
# or breakage must never touch serving, so the import is defensive
try:
    import adapter_learning as learning  # noqa: E402
except Exception as _e:  # pragma: no cover
    print(f"adapter_learning unavailable (continuing without): {_e!r}",
          flush=True)
    learning = None

LOGDIR = ROOT / "eval/reports/omp_arms"
DECISIONS = LOGDIR / "decisions.jsonl"

MODEL_NAMES = {
    "qwen35-4b-vanilla": "vanilla",
    "qwen35-4b-vanilla-nothink": "vanilla-nothink",
    "qwen35-4b-decide": "decide",
}

# (v3.13.0) Remote backend arm — the proxy as a faithful OpenAI shim in front
# of an external llama-server. The element-read d2 scorer is NOT ported (this
# llama.cpp build removed prompt-token logprobs, and teacher-forced sequence
# scoring over HTTP would cost seconds per decision), so there is no decide
# arm here: the shim forwards the harness's request verbatim. A native
# end-to-end run on the bigger model is then directly comparable to the
# vanilla-4B baseline measured the same way.
REMOTE: dict | None = None
# (v3.16.0) optional completion self-review config (None = disabled);
# set in main() from the backend config's completion_review block.
COMPLETION_REVIEW: dict | None = None
# (v3.16.1) the DELTA frame's BEFORE snapshot, per conversation (keyed by
# the first message's repr — a stand-in for a session id, which the proxy
# does not have). Captured on every observation turn while the feature is
# enabled; bounded, oldest dropped.
REVIEW_FIRST_OBS: dict[str, str] = {}
# (v3.16.2) annotate mode's caveat for the model's prose answer, stashed by
# the fallthrough review and appended to the response in compute(); cleared
# every turn like the other one-shot stashes.
REVIEW_NOTE: str | None = None
REMOTE_FWD_KEYS = ("temperature", "top_p", "max_tokens", "tools", "tool_choice",
                   "response_format", "stop", "seed")


def _review_cfg(block) -> dict | None:
    """(v3.16.0) Normalize the completion_review block; None = disabled.
    Modes: "log" records the reads, "gate" may challenge the uncorroborated
    stand-down, "annotate" (v3.16.2) attaches a verification caveat to the
    model's prose answer instead. Unknown mode strings fall back to "log"
    (the measured-safe posture)."""
    if not isinstance(block, dict) or not block.get("enabled"):
        return None
    return {"mode": block.get("mode")
            if block.get("mode") in ("gate", "annotate") else "log",
            "p_done_threshold": float(block.get("p_done_threshold", 0.5))}


def review_gate_fires(cr: dict | None, p_done, has_fallback: bool) -> bool:
    """(v3.16.0) The gate predicate, pure so the suite can exercise it:
    challenge an uncorroborated stand-down only when gate mode is on, the
    independent read confidently says the goal is NOT done, and the chain
    holds a concrete alternative action. Any doubt = log only. (v3.16.1)
    the p_done argument now carries the MINIMUM of the three framing
    reads — all three must clear the threshold for the stand-down to
    stand; that unanimity is the measured winner (78%/17% vs 67%/18%)."""
    if not cr or cr.get("mode") != "gate":
        return False
    try:
        p = float(p_done)
    except (TypeError, ValueError):
        return False
    return p < float(cr.get("p_done_threshold", 0.5)) and has_fallback


def review_first_obs_capture(key: str, obs_text: str) -> None:
    """(v3.16.1) Remember a conversation's FIRST page snapshot — the
    BEFORE frame of the DELTA read. Bounded at 32 conversations; the
    oldest entry is dropped. No-op when the feature is off (the caller
    guards on COMPLETION_REVIEW) or the turn carries no snapshot."""
    if not obs_text or key in REVIEW_FIRST_OBS:
        return
    REVIEW_FIRST_OBS[key] = " ".join(str(obs_text).split())[:4000]
    while len(REVIEW_FIRST_OBS) > 32:
        REVIEW_FIRST_OBS.pop(next(iter(REVIEW_FIRST_OBS)))


def review_last_prose(msgs) -> str:
    """(v3.16.1) The model's most recent assistant prose — the claim the
    COND framing's claim-check reads (the backtest graded exactly these
    assertions). Whitespace-compacted, capped at 500 chars; '' when the
    model has said nothing yet."""
    for m in reversed(msgs or []):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            txt = c
        elif isinstance(c, list):
            txt = " ".join(str(b.get("text", ""))
                           for b in c
                           if isinstance(b, dict) and b.get("type") == "text")
        else:
            txt = ""
        txt = " ".join(txt.split())
        if txt:
            return txt[:500]
    return ""


def review_frames_run(task, msgs, obs_text, log, site):
    """(v3.16.2) The 3-frame completion review (GOAL / DELTA / COND, the
    single-model framing backtest's min-of-three), shared by both trigger
    sites. Pure reads: records log["review_read"] (tagged with the trigger
    site) and returns (p_dict, p_min); any read failure is recorded and
    returns (None, None). Gating stays with the caller — the challenge
    exists only at the fallthrough site, and only in gate mode."""
    try:
        _ev = " ".join(str(obs_text or "").split())[:8000]
        _rkey = repr(msgs[0])[:300] if msgs else "?"
        _before = REVIEW_FIRST_OBS.get(_rkey, "")
        _prose = review_last_prose(msgs)
        _rp: dict = {}
        _rg = DECIDE_FN(
            f"GOAL: {task}\n\nThe page's accessibility "
            f"snapshot after the latest action:\n{_ev}\n\n"
            "Did the attempt achieve the goal?", ["yes", "no"])
        _rp["p_goal"] = float(
            (_rg.probabilities or {}).get("yes", 0.0))
        _rd = DECIDE_FN(
            f"GOAL: {task}\n\nBEFORE any action, the page "
            f"looked like:\n"
            f"{_before or '(no earlier page captured)'}\n\n"
            f"AFTER the latest action, the page looks "
            f"like:\n{_ev}\n\nBased on what CHANGED between "
            "BEFORE and AFTER, did the attempted task "
            "succeed?", ["yes", "no"])
        _rp["p_delta"] = float(
            (_rd.probabilities or {}).get("yes", 0.0))
        _pclaim = 0.5
        if _prose:
            _rc = DECIDE_FN(
                "You are given a claim about a web page and "
                "the page's accessibility snapshot as "
                f"evidence.\n\nClaim: \"{_prose}\"\n\n"
                f"Snapshot:\n{_ev}\n\nDoes the evidence "
                "support the claim?", ["yes", "no"])
            _pclaim = float(
                (_rc.probabilities or {}).get("yes", 0.5))
        _verdict = ("supported" if _pclaim >= 0.5
                    else "not supported")
        _rcc = DECIDE_FN(
            "The following is a web page's accessibility "
            "snapshot after an attempt to complete a task."
            f"\n\nA separate check judged the agent's claim "
            f"as {_verdict} against this evidence.\n\n"
            f"Snapshot:\n{_ev}\n\nDid the attempted task "
            "succeed, based only on this snapshot?",
            ["yes", "no"])
        _rp["p_cond"] = float(
            (_rcc.probabilities or {}).get("yes", 0.0))
        _pmin = min(_rp.values())
        log["review_read"] = {
            **{k: round(v, 3) for k, v in _rp.items()},
            "p_min": round(_pmin, 3), "site": site,
            "mode": (COMPLETION_REVIEW or {}).get("mode"),
            "method": _rcc.scoring_method}
        return _rp, _pmin
    except Exception as _e:
        log["review_read"] = {"error": repr(_e)[:120], "site": site}
        return None, None


def load_remote(path) -> dict | None:
    """Read the optional backend config. None = arm absent (missing file is
    the normal disabled state; an unreadable present file is a warning)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text())
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            return None
        model = str(cfg.get("model", "gemma-4-12b-it"))
        return {
            "wire_model": str(cfg.get("wire_model") or model),
            "base_url": str(cfg.get("base_url", "http://127.0.0.1:8998/v1")).rstrip("/"),
            "model": model,
            "timeout_s": int(cfg.get("timeout_s", 900)),
            "extra_payload": dict(cfg.get("extra_payload") or {}),
        # (v3.14.0) chain keys ride the same dict — dropped keys here
        # would silently skip main()'s chain binding (the launch bug:
        # arm "absent" with no error while the config said chain:true)
        "chain": bool(cfg.get("chain", False)),
        "chain_model": str(cfg.get("chain_model", "gemma-12b-decide")),
        "chain_gates": str(cfg.get("chain_gates",
                                   "configs/webgate_remote.json")),
        "chain_workers": int(cfg.get("chain_workers", 4)),
        # (v3.16.0) optional completion self-review, the learned-adapter
        # pattern: block missing or enabled:false = feature absent,
        # byte-for-byte v3.15.0 behavior. mode "log" adds the independent
        # completion read's verdict to the decision row; mode "gate"
        # additionally challenges an UNCORROBORATED stand-down (the
        # review-flagged turn): a confident NOT-DONE ships the chain's own
        # fallback cell instead of passing a likely-premature completion
        # report. Default disabled — the checkpoint-review experiment
        # measured 77% detection on false success reports but 32% false
        # alarms on correct ones, so the gate never starts on.
        "completion_review": _review_cfg(cfg.get("completion_review")),
        }
    except Exception as e:
        print(f"backend config unreadable ({e!r}); remote arm disabled", flush=True)
        return None

GATE = json.load(open(ROOT / "configs/webgate.json"))  # accept_conf, temperature

# The chain's op set extended with the harness-facing ANSWER op (this harness has
# no per-step gold op; "task done / not an element action" must be expressible,
# otherwise the loop would force an element action onto reporting turns).
ANSWER_LINE = ("ANSWER — no browser element action is needed this turn; report to "
               "the user in text (task complete, or the next step is not an "
               "element action)")
OP_LINES_FIRST = list(OP_CANDIDATES) + [ANSWER_LINE]
FIRST_TO_OP = {line.split(" — ")[0]: line.split(" — ")[0] for line in OP_LINES_FIRST}

OP_LINES_ELEM = [
    f"{op} — {'press/activate' if op == 'CLICK' else 'enter a value into' if op == 'TYPE' else 'choose an option from'} the selected element"
    for op in OPS
]

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "searchbox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "option", "tab",
    "treeitem", "slider", "spinbutton", "switch", "textarea",
}

MAX_CANDIDATES = 20
# element-confidence gate for ANSWER-forced actions: legit targets read
# 0.7+, post-task wander picks 0.1-0.4 (T2-v33 decisions log)
ANSWER_GATE = 0.5

# (v3.14.0) decision-read transport indirection: run_chain's reads, value
# thinks and model fallbacks go through these shims, so the SAME chain logic
# runs on the in-process 4B (defaults, below) or over llama-server HTTP
# (REMOTE chain mode, bound in main() from the backend config's "chain"
# flag — eval/chain_http.py: grammar-forced d2 + closed-channel prefill
# reads, calibrated gates from configs/webgate_remote.json).
DECIDE_FN = None       # (question, candidates, **decide kwargs) -> DecisionResult
VALUE_FN = None        # (task, hist, op, sel_line, n_cand) -> value str
FALLBACK_FN = None     # (req, nudge=False) -> (content, finish, tool_calls)
CHAIN_REMOTE = False


def _decide_local(question, candidates, **kw):
    return decide(LOADED, question, candidates, **kw)


def _value_local(task, hist, op, sel_line, n_cand):
    return think_value(LOADED, task, hist, op, sel_line, n_cand)


def _fallback_local(req, nudge=False):
    return passthrough(req, enable_thinking=False, nudge=nudge)


DECIDE_FN, VALUE_FN, FALLBACK_FN = _decide_local, _value_local, _fallback_local
# Answer turns only need a sentence; at the default 1024-token cap the
# post-task model writes repetitive junk for ~80 s per turn (unoptimized
# GDN decode), and a dying session's straggler generation then blocks the
# single-threaded server long enough for the next session's client retries
# to eat its budget (T3-v33 starved 8.6 of its 10 minutes this way).
PASSTHROUGH_MAX_NEW = None
# consecutive no-candidate chain turns (per task): observe once, then the
# next no-candidate turn may navigate back to the task URL
NO_CAND_STREAK = 0
LAST_TASK = None
# full text of the last non-redrive task this browser session saw — the
# compaction redrive erases the original task message from the rebuilt
# context, and phrase tracking needs more than LAST_TASK's 120 chars
LAST_TASK_FULL: str | None = None
# one-shot action stashed by the answer gate's model handoff: if the model's
# answer comes back malformed or empty (long-context template echo — fatal,
# the harness disposes on unparseable turns), compute() ships this instead
ANSWER_GATE_FALLBACK: dict | None = None
ANSWER_TEXT_FALLBACK: str | None = None
# the session's accumulated action history. The per-request hist is whatever
# the current context retained — compactions drop the newest actions, so
# pending-phrase progress decayed toward zero under the harness's rapid
# compaction cadence (decide3110l). The proxy sees every turn; union-merge
# here, and compose the handoff from this so no progress is ever lost.
LAST_HIST: list[str] = []
# JS-syntax markers in raw content with NO extracted tool call: the model
# echoed the system prompt's eval template instead of answering. Prose
# answers never match; well-formed cells are extracted before this runs.
MALFORMED_ANSWER_RE = re.compile(
    r"\bconst\b|=>|input\[type=|throw new Error|\)\);")
# ids of eval cells the chain itself emitted (stuck-breaker feed test)
CHAIN_CELL_IDS: set[str] = set()
LOCK = threading.Lock()
LOADED = None


def log_decision(rec: dict) -> None:
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- wire parsing

def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(x for x in parts if x) or None
    return content if isinstance(content, str) else None


def norm_messages(messages: list[dict]) -> list[dict]:
    """OpenAI wire format -> the shape Qwen's chat template expects."""
    out = []
    for m in messages:
        role = m.get("role")
        c = _content_text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = tc.get("function", {})
                args = f.get("arguments")
                if isinstance(args, str):
                    # the Qwen3.5 template iterates arguments as a mapping;
                    # the OpenAI wire format carries them as a JSON string.
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                tcs.append({"type": "function",
                            "function": {"name": f.get("name"),
                                         "arguments": args if isinstance(args, dict) else {}}})
            out.append({"role": "assistant", "content": c, "tool_calls": tcs})
        elif role == "tool":
            out.append({"role": "tool", "content": c or ""})
        else:
            out.append({"role": role, "content": c})
    return out


def eval_codes(messages: list[dict]) -> list[str]:
    """Every eval tool-call code string in conversation order."""
    codes = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            if f.get("name") != "eval":
                continue
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            code = (args or {}).get("code") or ""
            codes.append(code)
    return codes


def eval_events(messages: list[dict]) -> list[tuple[str, str]]:
    """(tool_call id, code) per eval call, conversation order. Chain-emitted
    ids are remembered by eval_cell, which is what distinguishes chain cells
    from model-written cells that copy the chain's cell style."""
    events = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            if f.get("name") != "eval":
                continue
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            events.append((tc.get("id") or "",
                           (args or {}).get("code") or ""))
    return events


# the harness's compaction redrive replaces the task message with this
# protocol preamble (v3.11.10 — task recovery below)
REDRIVE_RE = re.compile(r"context replaced|<handoff>", re.I)


def extract_task(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            t = _content_text(m.get("content"))
            if t:
                # drop harness wrappers (system reminders/injections) so the
                # chain's reads see the goal, not the envelope
                t = re.sub(r"<system-reminder>.*?</system-reminder>\s*", "",
                           t, flags=re.S)
                t = re.sub(r"<system-injection>.*?</system-injection>\s*", "",
                           t, flags=re.S)
                t = t.strip()
                if t:
                    # (v3.11.10) after a compaction the rebuilt context opens
                    # with the redrive preamble, not the task — reads on
                    # handoff vocabulary route ANSWER and the model turns
                    # template-echo junk (decide3110i). The handoff document
                    # itself states the goal; recover it. Only fires when a
                    # Goal section genuinely exists in the message.
                    if REDRIVE_RE.search(t[:400]):
                        gm = re.search(r"##\s*Goal\s*\n(.+?)(?:\n\n|\Z)", t,
                                       re.S)
                        if gm and gm.group(1).strip():
                            return gm.group(1).strip()[:600]
                    return t[:600]
    return "(no task found)"


def hist_from_handoff(messages: list[dict]) -> list[str]:
    """Action lines recorded in the redrive's handoff document (v3.11.12).
    compose_handoff writes its Progress section as exact hist-format lines;
    reading them back is what lets phrase tracking survive a compaction —
    without it every redrive restarts from the task's first phrase, and the
    harness compacts often enough that the chain loops on steps 1-3 until
    the deadline (decide3110k: eleven compactions, never left the home
    page)."""
    for m in messages:
        if m.get("role") == "user":
            t = _content_text(m.get("content"))
            if t and REDRIVE_RE.search(t[:400]):
                return [mm.group(1) for mm in re.finditer(
                    r"(?m)^- ((?:OPEN|CLICK|SELECT|TYPE|SUBMIT)[^\n]*)$", t)]
    return []


def tab_name(messages: list[dict]) -> str | None:
    for code in reversed(eval_codes(messages)):
        m = re.search(r'browser\.open\(\s*\{[^}]*?name:\s*"([^"]+)"', code, re.S)
        if m:
            return m.group(1)
        m = re.search(r'browser\.tab\(\s*"([^"]+)"', code)
        if m:
            return m.group(1)
    return None


def browser_in_play(messages: list[dict]) -> bool:
    return any("browser.open(" in c or "browser.tab(" in c for c in eval_codes(messages))


# --- domain generalization (v3.9.0) -----------------------------------------
# The decision core is domain-general (op read, element read, pending-step
# answer-route override, repeat guard); each domain provides surfaces: an
# observation parser -> candidate lines, an action writer -> tool call, and
# a phrase vocabulary. Routing is STRUCTURAL — which tools the conversation
# has actually called — never a model read. Before the first tool call the
# route falls back to the same deterministic task-shape signals the browser
# chain has always used (URL in task).

CODE_TOOLS = {"bash", "read", "write", "edit", "glob", "grep"}
CODE_SHAPE_RE = re.compile(
    r"\.(sh|py|txt|js|json|csv|md)\b|\bchmod\b|\bscript\b|\bexecutable\b", re.I)


def detect_domain(messages: list[dict]) -> str:
    """'browser' | 'coding' | 'universal'. Universal = neither adapter's
    surfaces are in play (conversation, research, unknown tools): the
    Tier-1 floor handles it — safe passthrough plus the domain-free wins
    (audit log, loop brake). Only the FINAL fallback changed in v3.10.0:
    every browser/coding route below is byte-for-byte the v3.9.0 logic."""
    if browser_in_play(messages):
        return "browser"
    names = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            n = (tc.get("function", {}) or {}).get("name")
            if n:
                names.add(n)
    if names & CODE_TOOLS:
        return "coding"
    if names & {"eval"}:  # js/py kernel without browser cells: not coding
        return "browser"
    task = extract_task(messages)
    if re.search(r"https?://", task) or re.search(r"\bbrowser\b", task, re.I):
        return "browser"
    if CODE_SHAPE_RE.search(task):
        return "coding"
    return "universal"


# --- universal tier (v3.10.0) ------------------------------------------------
# Everything outside the adapters: conversation, research, unknown tools.
# Generation IS the right tool here — the chain stands down — but the floor
# adds what needs no domain knowledge: an audit log on every turn (the old
# fallback logged nothing) and a loop brake. The brake is the browser chain's
# stuck-detector generalized: LOOP_STREAK identical tool-call turns in a row
# means the model is thrashing; a corrective system note + a token cap
# interrupt it. It never blocks a call — the model can override the note —
# so a legitimate repeat degrades to a one-turn cost, not a failure.

LOOP_STREAK = 3
UNIVERSAL_NUDGE = (
    "You have now made the same tool call {n} times in a row with no change "
    "in the result. Do not repeat that call again. Either try a materially "
    "different approach, or finish now and report your best answer with "
    "what you already have.")


def _norm_text(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def universal_sig(m: dict) -> str | None:
    """Signature of one assistant turn's tool calls (names + normalized
    arguments). None for plain-text turns — nothing to repeat."""
    tcs = m.get("tool_calls") or []
    if not tcs:
        return None
    parts = []
    for tc in tcs:
        fn = (tc.get("function", {}) or {}).get("name") or ""
        args = (tc.get("function", {}) or {}).get("arguments") or ""
        argn = (_norm_text(args) if isinstance(args, str)
                else _norm_text(json.dumps(args, sort_keys=True)))
        parts.append(f"{fn}({argn})")
    return " + ".join(parts)


def universal_route(messages: list[dict]) -> dict:
    """Pure decision for a universal turn: passthrough, or passthrough with
    the loop-brake nudge when the last LOOP_STREAK tool-call turns share one
    signature. Text-only assistant turns are skipped, so calls repeated
    across interleaved commentary still count as a streak."""
    sigs = [s for s in (universal_sig(m) for m in messages
                        if m.get("role") == "assistant") if s is not None]
    streak = 0
    if sigs:
        streak = 1
        for s in reversed(sigs[:-1]):
            if s != sigs[-1]:
                break
            streak += 1
    brake = streak >= LOOP_STREAK
    return {"route": "loop-brake" if brake else "passthrough",
            "streak": streak,
            "last_sig": (sigs[-1][:120] if sigs else ""),
            "nudge": UNIVERSAL_NUDGE.format(n=streak) if brake else None}


def run_universal(req: dict) -> tuple[str, str | None, list[dict]]:
    """Tier-1 floor: audit-log the turn, brake a detected loop, pass through.
    Same return shape as passthrough. (v3.12.0) before serving, the turn goes
    to the learning plumbing's floor hook — trajectory capture when enabled,
    then a registered learned adapter gets the first shot; None falls through
    to the unchanged floor path."""
    task = extract_task(req["messages"])
    log = {"arm": "decide-universal", "domain": "universal",
           "task": task[:120]}
    r = universal_route(req["messages"])
    log.update({"routed": r["route"], "streak": r["streak"],
                "last_sig": r["last_sig"]})
    served = learning.floor_hook(req, log, task) if learning else None
    if served is not None:  # a learned adapter answered this turn
        return served
    if r["nudge"]:
        global PASSTHROUGH_MAX_NEW
        PASSTHROUGH_MAX_NEW = 512  # one-shot: stop a loop from bleeding budget
        log["nudged"] = True
    log_decision({**log, "ts": time.time()})
    return passthrough(req, enable_thinking=False, nudge=r["nudge"])


def last_tool_result(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "tool":
            return _content_text(m.get("content"))
        if m.get("role") == "user" and m.get("tool"):
            return _content_text(m.get("content"))
    return None


def parse_snapshot(text: str) -> list[dict]:
    """Interactive elements from an aria snapshot: role, name, ref, raw line."""
    cands = []
    for line in text.splitlines():
        if "[ref=e" not in line:
            continue
        m = re.search(r"\[ref=(e\d+)\]", line)
        if not m:
            continue
        # the snapshot wraps a line in single quotes when the accessible
        # name carries YAML-special characters (": " or "#" — common in
        # real product titles): the role token sits AFTER the quote, and
        # (v3.11.15) those elements used to be dropped silently — on
        # catalogue pages every card with such a title vanished from the
        # candidates, "Add to basket" buttons collapsed into same-name
        # runs, and the ordinal-snap resolver snapped "the first book" to
        # the pager's "next" link (decide3110n page-flipped to the
        # deadline)
        mm = re.match(r"\s*-\s+'?\s*([A-Za-z0-9_]+)", line)
        role = mm.group(1).lower() if mm else "generic"
        if role not in INTERACTIVE_ROLES:
            continue
        q = re.search(r'"([^"]*)"', line)
        cands.append({"ref": m.group(1), "role": role,
                      "name": (q.group(1) if q else ""), "raw": line.strip()[:160]})
    return cands


def render_lines(cands: list[dict]) -> list[str]:
    """Mirror hybrid.actions.candidate_lines style: [i] role 'text' ref=eN."""
    out = []
    for i, c in enumerate(cands, 1):
        parts = [f"[{i}] {c['role']}"]
        if c["name"].strip():
            parts.append(f"'{c['name'][:80]}'")
        parts.append(f"ref={c['ref']}")
        out.append(" ".join(parts)[:160])
    return out


def extract_history(messages: list[dict]) -> list[str]:
    """Prior browser actions, from constructed or model-written eval cells."""
    hist = []
    for code in eval_codes(messages):
        # chain cells and model cells that mimic them carry // markers
        # ("switch to"/"choose"/"set" show up when the model writes its own)
        for m in re.finditer(
                r'//\s*(click|type|fill|select|switch to|choose|set)\s+'
                r'"((?:[^"\\]|\\.)*)"(?:\s+into\s+"((?:[^"\\]|\\.)*)")?'
                r'\s+@(e\d+)(?:\s+\[task: ([^\]]*)\])?', code):
            kind, a, b, ref = m.group(1), m.group(2), m.group(3), m.group(4)
            tag = m.group(5)
            if kind == "click":
                hist.append(f"CLICK '{a}' on [{ref}]"
                            + (f" (task: {tag})" if tag else ""))
            elif kind in ("type", "fill"):
                hist.append(f"TYPE '{a}' into '{b or ''}' on [{ref}]")
            else:
                hist.append(f"SELECT '{a}' on [{ref}]")
        for m in re.finditer(
                r'await\s+tab\.(click|fill|select|type)\(\s*'
                r'"(?:aria-ref=|@)?(e\d+)"'
                r'(?:\s*,\s*"((?:[^"\\]|\\.)*)")?', code):
            kind, ref, val = m.group(1), m.group(2), m.group(3)
            if kind == "click":
                hist.append(f"CLICK on [{ref}]")
            elif kind in ("fill", "type"):
                if val:  # fill("") is the clear-before-type helper, not a step
                    hist.append(f"TYPE '{val}' into [{ref}]")
            elif val is not None:
                hist.append(f"SELECT '{val}' on [{ref}]")
    return hist[-12:]


# ---------------------------------------------------------------- chain stages

def pending_field(task: str, hist: list[str]) -> tuple[str | None, list[str]]:
    """(first not-yet-filled field, filled fields) from 'into the X field'
    phrases in the task, checked against TYPE history entries only (values
    are ignored — a password value must not mark the password field done).
    Each TYPE entry covers at most ONE task phrase: a merged aria marker
    ("Username Password") from a single type cell must not mark BOTH fields
    done (v380 T2: the password was never typed, but the merged marker let
    the submit override fire on an empty password)."""
    types = []
    for h in hist:
        m = re.match(r"TYPE '.*?' into '(.*?)' on \[", h)
        if m:
            # fragments shorter than 4 chars are noise
            types.append([w for w in re.split(r"\W+", m.group(1).lower())
                          if len(w) >= 4])
    used: set[int] = set()
    filled = []
    for m in re.finditer(r"into the ([a-z][a-z ]{2,20}?) field", task.lower()):
        fld = m.group(1).strip()
        done = False
        for j, words in enumerate(types):
            if j in used:
                continue
            if any(fld in w or w in fld for w in words):
                used.add(j)
                filled.append(fld)
                done = True
                break
        if not done:
            return fld, filled
    return None, filled


def ctx(task: str, hist: list[str], n_cand: int, op: str | None = None,
        options: list[str] | None = None, chosen: str | None = None) -> str:
    """Mirror of hybrid.actions.context_text, fed from harness conversation."""
    lines = [f"Task: {task}", "", "Actions taken so far:"]
    lines += [f"- {h}" for h in hist] if hist else ["- (none yet)"]
    lines.append("")
    lines.append(f"Current page has {n_cand} candidate elements (listed in the options).")
    lines.append("")
    if op is None:
        offered = ", ".join(list(OPS) + ["ANSWER"])
        lines.append("What should the next step's action be? The operation must be "
                     f"one of {offered} (with a value for TYPE/SELECT if needed).")
    else:
        lines.append(f"The next step's operation is {op}. "
                     f"Which element should the {op} act on?")
    if options:
        lines.append("")
        lines.append("Options:")
        lines += options
    if chosen:
        lines.append("")
        lines.append(f"The chosen element is: {chosen}")
    return "\n".join(lines)


def select_switch_override(task: str, value: str, ref: str,
                           hist: list[str]) -> str:
    """T4-v374: on the second step of "select 'Option 1' ... then switch the
    dropdown to 'Option 2'", the value-thinker echoed the just-set option; the
    repeat guard then saw (SELECT, [e9], 'Option 1') as done and parked the
    chain on passthrough-repeat while the model flailed. If the thinker's
    value is ALREADY set on this element and the task quotes exactly one
    other option (or names one after switch/change/set ... to), take the
    task-named one. Fires only on SELECT turns with a set-history conflict."""
    already = set()
    for h in hist:
        m = re.match(r"SELECT '(.*?)' on \[" + re.escape(ref) + r"\]", h)
        if m:
            already.add(m.group(1))
    if not (already and value in already):
        return value
    quoted = [s for s in re.findall(r"[\"']([^\"']{2,40})[\"']", task)
              if s not in already]
    if len(quoted) == 1:
        return quoted[0]
    m = re.search(r"(?:switch|change|set|toggle)[^.;]{0,40}?\bto\s+([^.,;]+)",
                  task, re.I)
    if m:
        cand = m.group(1).strip().strip("'\"")
        if cand and cand not in already:
            return cand
    return value


def think_value(loaded, task: str, hist: list[str], op: str, sel_line: str,
                n_cand: int) -> str:
    """THINK stage (~24 tokens), mirrors eval.webreplay._think_value."""
    ask = ("What exact text should be typed into it? Reply with the text and "
           "nothing else." if op == "TYPE" else
           "Which option should be selected? Reply with the option's value and "
           "nothing else.")
    content = ctx(task, hist, n_cand, op) + f"\n\nChosen element: {sel_line}\n\n{ask}"
    answer, _ = generate_answer(loaded, [{"role": "user", "content": content}],
                                max_new_tokens=24, enable_thinking=False)
    answer = re.split(r"<\|[^|>]*\|>", answer)[0].strip()
    return answer.strip('"').strip()


# (v3.14.4) payload sanity for TYPE/SELECT fires. The observed refusal —
# "I am sorry, but I cannot provide the text to be typed into the textbox
# because I do not have" — echoes the ask's own wording, i.e. the thinker
# answering the question instead of doing the task. Patterns anchor at the
# start AND require refusal vocabulary, so ordinary values ("I am legend",
# "cannot reproduce" mid-string) never trip it; the cost of a rare false
# positive is one model-authored turn, the cost of firing was a refusal
# typed into the page.
REFUSAL_RE = re.compile(
    r"^\s*(?:"
    r"i\s?(?:am|'m|’m)\s+(?:sorry|afraid|unable|not\s+able)"
    r"|i\s+(?:cannot|can'?t|can\s+not)\s+"
    r"(?:provide|assist|help|do|type|comply|answer|fulfill|complete"
    r"|generate|produce|create|reveal|share|supply)"
    r"|i\s+(?:do\s+not|don'?t)\s+have"
    r"|sorry[,!]|unfortunately|as\s+an\s+ai\b)",
    re.I)


def bad_payload(value: str | None) -> str | None:
    """Why the payload must not be fired, or None when it looks sane."""
    if value is None or not value.strip():
        return "empty-value"
    if REFUSAL_RE.match(value.strip()[:120]):
        return "refusal-payload"
    return None


def action_code(tab: str, op: str, ref: str, value: str | None,
                name: str = "", input_ordinal: int = 0,
                task_phrase: str | None = None) -> str:
    # All actions run page-side via tab.evaluate: the daemon's ref-click RPC
    # hangs (element never reaches Playwright's "stable" state under
    # software rendering), while page-side DOM manipulation works reliably.
    # tab.select() also matches value attributes only, never labels.
    # Each cell carries a // comment marker (for history extraction and
    # adjudication) since the code itself no longer contains tab.* calls.
    # (v3.14.2) task_phrase rides the CLICK marker: a click that executed a
    # task phrase the name-based consumption can never see (the ordinal
    # snap's "open the first book's detail page" -> CLICK 'In Her Wake'
    # shares no vocabulary with its phrase) is tagged, so later turns
    # consume the phrase and stand down instead of re-resolving it on the
    # detail page the click just opened.
    def q(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')

    lines = [f'const tab = await browser.tab("{tab}");']
    if op == "CLICK":
        want = q((name or "").lower())
        tag = f" [task: {q(task_phrase)}]" if task_phrase else ""
        lines.append(f'// click "{name}" @{ref}{tag}')
        # Page-side text click. When several elements share the exact text
        # (the quotes page has a header "Login" link AND a "Login" submit
        # button), prefer real buttons/inputs over anchors — v3.3's
        # first-match rule clicked the header link and re-opened the login
        # form instead of submitting it. (v3.4 tried ref-based
        # tab.click("aria-ref=eN") first; it resolved without error but did
        # not navigate, so it is gone.)
        lines.append(
            "await tab.evaluate(`(() => {"
            # (v3.15.0) every candidate string is lowercased below, so the
            # wanted name must be too: a model-emitted "Tab #1" (page
            # casing, 22+ click-miss errors in the benchmark transcripts)
            # could otherwise never equal the DOM's "tab #1".
            "const want = \"" + want + "\".toLowerCase();"
            "if (!want) throw new Error('empty element name');"
            # (v3.14.4) checkboxes and radios are activatable controls too;
            # their accessible name lives in the LABEL, not in the element
            # (a bare <input type=checkbox> has empty textContent), so the
            # name sources below read the label channels as well. No
            # input[type=reset]: a fuzzy miss must never wipe a form.
            "const els = Array.from(document.querySelectorAll("
            "\"a, button, [role='button'], [role='checkbox'], [role='radio'], "
            "[role='tab'], [role='menuitem'], [role='link'], [role='option'], "
            "[role='treeitem'], [role='switch'], "
            "input[type='submit'], input[type='button'], input[type='checkbox'], "
            "input[type='radio'], input[type='image']\"));"
            "const txt = e => (e.textContent || e.value || '').trim().toLowerCase();"
            # (v3.14.1) catalogue cards hide the accessible name: the title
            # link's DOM text is truncated with literal dots ("slow states of
            # collapse: ...") and the FULL name sits in the title attribute,
            # while the image link's textContent is empty and the name is the
            # img alt. Match against every string the element offers, and
            # compare dot-truncated strings as prefixes ("... poets" style
            # truncation can never contain or be contained).
            "const strs = e => { const out = []; "
            "const add = s => { if (s == null) return; "
            "s = String(s).trim().toLowerCase(); "
            "if (s && out.indexOf(s) < 0) out.push(s); };"
            "add(e.textContent); add(e.value); "
            "add(e.getAttribute ? e.getAttribute('title') : null);"
            "add(e.getAttribute ? e.getAttribute('aria-label') : null);"
            # (v3.14.4) the channels the accessible-name computation itself
            # uses for form controls: implicit <label><input>…</label>,
            # explicit <label for=id>, aria-labelledby targets, and the
            # name attribute some widgets carry in lieu of any text.
            "if (e.labels) for (const l of e.labels) add(l.textContent);"
            "const cl = e.closest ? e.closest('label') : null;"
            "if (cl) add(cl.textContent);"
            "const lb = e.getAttribute ? e.getAttribute('aria-labelledby') : null;"
            "if (lb) for (const id of lb.split(/\\s+/)) { "
            "const d = document.getElementById(id); if (d) add(d.textContent); }"
            "add(e.name);"
            "const im = e.querySelector ? e.querySelector('img') : null; "
            "if (im) add(im.getAttribute('alt'));"
            "return out; };"
            "const noDots = s => s.replace(/\\.{3}\\s*$/, '').trim();"
            "const hitT = (t, w) => { if (!t) return false; "
            "if (t === w) return true; "
            "const s = noDots(t);"
            "if (s.length >= 4 && (w.indexOf(s) === 0 || s.indexOf(w) === 0)) return true;"
            "return t.length >= 4 && (t.indexOf(w) >= 0 || w.indexOf(t) >= 0); };"
            "const notLink = e => e.tagName !== 'A';"
            # (v3.14.4) on a text tie, a state control (checkbox/radio) is
            # picked only when nothing else matches: clicking a same-named
            # checkbox instead of the intended button would toggle state.
            "const pick = list => (list.find(e => notLink(e) && "
            "e.type !== 'checkbox' && e.type !== 'radio') || "
            "list.find(notLink) || list[0]);"
            "const exact = els.filter(e => strs(e).some(t => t === want));"
            # Accessibility names carry glyphs the DOM textContent lacks:
            # FontAwesome ::before content lands in the accname only, so
            # the-internet's login button is "\uf090 Login" in the snapshot
            # but " Login" in textContent — the fallback must match in BOTH
            # directions, and a too-short DOM text (<4 chars) may not ride
            # the reverse direction.
            "const fuzzy = els.filter(e => exact.indexOf(e) < 0 && "
            "strs(e).some(t => hitT(t, want)));"
            "const hit = exact.length ? pick(exact) : pick(fuzzy);"
            "if (!hit) throw new Error('no clickable element matching ' + want);"
            # (v3.15.0) submit precondition, pure native semantics: when the
            # click would submit a form the BROWSER itself would reject
            # (willValidate && !checkValidity — required empty, pattern/email
            # mismatches), stop before the click and say which fields. This
            # is routing, not blocking: the error lands in the tool result
            # and the turn goes back to the model WITH the reason. Honors
            # the page's own novalidate/formNovalidate opt-outs, and ignores
            # non-visible controls (the agent could never fill those).
            # Generic web hygiene — no task or benchmark vocabulary.
            "const form = hit.closest ? hit.closest('form') : null;"
            # (v3.15.0a) some engines never implemented the novalidate/
            # formNovalidate IDL properties (a webview served .novalidate as
            # undefined with the attribute present), so the opt-out is read
            # from the attribute as well.
            "if (form && !(form.novalidate || form.hasAttribute('novalidate'))"
            " && !(hit.formNovalidate || hit.hasAttribute('formnovalidate'))) {"
            "const bad = Array.prototype.filter.call(form.elements, e => "
            "e.willValidate && !e.checkValidity() && "
            "(e.offsetParent !== null || e.type === 'hidden'));"
            "if (bad.length) { const nm = e => (e.name || e.id || e.type);"
            "throw new Error('submit-precondition: ' + bad.length + "
            "' form field(s) invalid/empty: ' + "
            "bad.slice(0, 3).map(nm).join(', ') + "
            "' — fill them, then submit'); } }"
            "hit.scrollIntoView({block: 'center', inline: 'center'});"
            "hit.click();"
            "return txt(hit) || (strs(hit)[0] || '');"
            # The evaluate string is an EXPRESSION: without the trailing ()
            # it merely evaluates to a function object and the body never
            # runs (the silent v3.3/v3.5 submit no-op). TYPE/SELECT always
            # had the invocation; CLICK alone was missing it.
            "})()`);"
            "await new Promise(r => setTimeout(r, 1500));")
    elif op == "TYPE":
        want = q(name or "")
        val = q(value or "")
        lines.append(f'// type "{value}" into "{name}" @{ref}')
        lines.append(
            "await tab.evaluate(`(() => {"
            "const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');"
            "const want = norm(\"" + want + "\");"
            "const val = \"" + val + "\";"
            # (v3.14.4) text-entry controls only: the els[ordinal] fallback
            # must never land on a checkbox/radio/button (typing into one is
            # a no-op that still dispatches events). Password stays —
            # enter-password needs it.
            "const els = Array.from(document.querySelectorAll("
            "'input:not([type=hidden]):not([type=checkbox]):not([type=radio])"
            ":not([type=button]):not([type=submit]):not([type=image])"
            ":not([type=file]):not([type=range]):not([type=color])"
            ":not([type=reset]), textarea'));"
            "const hay = e => norm([e.name, e.id, e.placeholder,"
            "e.getAttribute('aria-label'),"
            "e.labels && e.labels[0] && e.labels[0].textContent]"
            ".filter(Boolean).join(' '));"
            "let hit = want ? els.find(e => hay(e) === want || "
            "hay(e).includes(want)) : null;"
            f"if (!hit) hit = els[{max(input_ordinal - 1, 0)}];"
            "if (!hit) throw new Error('no input matching ' + want);"
            "const proto = hit.tagName === 'TEXTAREA' ? "
            "HTMLTextAreaElement.prototype : HTMLInputElement.prototype;"
            "Object.getOwnPropertyDescriptor(proto, 'value').set.call(hit, val);"
            "hit.dispatchEvent(new Event('input', {bubbles: true}));"
            "hit.dispatchEvent(new Event('change', {bubbles: true}));"
            "return hit.name || hit.id || hit.type;"
            "})()`);"
            "await new Promise(r => setTimeout(r, 300));")
    elif op == "SELECT":
        want = q((value or "").lower())
        lines.append(f'// select "{value}" @{ref}')
        lines.append(
            "await tab.evaluate(`(() => {"
            "const norm = s => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();"
            "const toks = s => norm(s).split(' ').filter(Boolean);"
            "const want = norm(\"" + want + "\");"
            "const sels = Array.from(document.querySelectorAll('select'));"
            "if (!sels.length) throw new Error('no <select> on page');"
            "const sel = sels.length === 1 ? sels[0] : "
            "sels.find(s => s.offsetParent !== null) || sels[0];"
            # the value-thinker rephrases the task ("price from low to high")
            # rather than quoting the option text, so match on normalized
            # equality, containment, then best token overlap (>= 2 tokens)
            "let best = null, bestScore = 0;"
            "for (const o of sel.options) {"
            "const on = norm(o.text), ov = norm(o.value);"
            "let score = 0;"
            "if ((on && on === want) || (ov && ov === want)) score = 100;"
            "else if ((on && (on.includes(want) || want.includes(on))) || "
            "(ov && (ov.includes(want) || want.includes(ov)))) score = 50;"
            "else { const wt = new Set(toks(want)); "
            "for (const t of toks(o.text)) if (wt.has(t)) score += 10; }"
            "if (score > bestScore) { bestScore = score; best = o; }"
            "}"
            "if (!best || bestScore < 20) throw new Error('no option matching ' + want);"
            "sel.value = best.value;"
            "best.selected = true;"
            "sel.dispatchEvent(new Event('input', {bubbles: true}));"
            "sel.dispatchEvent(new Event('change', {bubbles: true}));"
            "return best.text.trim();"
            "})()`);"
            "await new Promise(r => setTimeout(r, 300));")
    lines.append("display(await tab.ariaSnapshot());")
    return "\n".join(lines)


# --- pending-step answer-route override (v3.8.0, phrase windows v3.8.1) ----

_PHRASE_VERBS = (r"open|browse|visit|go to|goto|return to|back to|click|press"
                 r"|submit|add|select|choose|pick|sort|switch( [a-z']+){0,2} to"
                 r"|toggle|log ?out|log ?in")
# the window stops at commas too: a 60-char run-on blob spanning ", then"
# merges two task steps into one phrase, which one click then consumes
# (v380 T1: "browse the Travel category, then return to the site home
# page" was eaten by the Travel click; v380 T5: the sort step vanished
# inside "submit the login form, sort the products …" and the add
# override fired before the sort)
CLICK_PHRASE_RE = re.compile(r"\b(" + _PHRASE_VERBS + r")\b[^.;,]{0,60}", re.I)
# phrases that target a dropdown-ish control rather than a click target
# (the switch-to verb may carry an object: "switch the dropdown to X")
_SELECT_VERBS = {"sort", "select", "choose", "pick"}
_PHRASE_STOP = {"the", "a", "an", "to", "and", "or", "of", "in", "on", "for",
                "with", "then", "your", "our", "site", "page", "it", "its",
                "from", "by", "at", "as", "into", "up", "please", "also",
                "again", "back"}
_SUPERLATIVE_RE = re.compile(r"\b(cheapest|first|last|most expensive|top)\b",
                             re.I)


def _content(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower())
            if w and w not in _PHRASE_STOP}


def _hist_actions(hist: list[str], kind: str) -> list[tuple[str, str | None]]:
    """(name, task-tag) pairs for a hist action kind. The tag (v3.14.2)
    marks a click that executed a task phrase its NAME can never consume —
    the ordinal snap's resolution ("open the first book's detail page" ->
    CLICK 'In Her Wake'): without it the phrase re-resolved on the detail
    page the click had just opened and the chain wandered product pagers."""
    out: list[tuple[str, str | None]] = []
    for h in hist:
        m = re.match(kind + r" '(.*?)' on \[", h)
        if m:
            t = re.search(r"\(task: (.*?)\)$", h)
            out.append((m.group(1), t.group(1) if t else None))
    return out


def _same_phrase(a: str, b: str) -> bool:
    n = lambda s: " ".join(s.split()).strip().rstrip(".").lower()
    return n(a) == n(b)


def _phrase_hit(phrase_toks: set, name: str) -> bool:
    nt = _content(name)
    if not nt:
        return False
    return nt <= phrase_toks or len(phrase_toks & nt) >= 2


def pending_step(task: str, hist: list[str],
                 cands: list[dict]) -> tuple[str, int | None] | None:
    """The task's earliest action-imperative phrase that no history entry has
    consumed, plus the candidate it names (None if the phrase names nothing
    on the current page). Both live answer-route failures have this shape —
    the op read says ANSWER with work left: T1-v375 stood down on the task's
    FIRST navigation (element conf 0.342 < gate) and the model observe-looped
    to the deadline; T5-v374/375 routed the still-pending "add the cheapest
    product to the cart" turns to answer the same way. URLs are stripped
    before extraction ("open https://…" would otherwise swallow the next
    phrase into one blob). Click phrases are consumed by CLICK history and
    may name any role except option/textbox; dropdown phrases (sort/select/
    choose/pick/switch-to) are consumed by SELECT history and typically name
    the combobox ("Sort products"), whose action the combobox->SELECT flip
    already knows how to perform."""
    task = re.sub(r"https?://\S+", " ", task)
    # the trailing "Report …" clause states the deliverable, not an action —
    # its nouns ("the login succeeded") otherwise match the log-in verb
    task = re.split(r"\breport\b", task, flags=re.I)[0]
    done = _hist_actions(hist, "CLICK")
    sels = [(n, None) for n, _ in _hist_actions(hist, "SELECT")]
    used: set[int] = set()
    sused: set[int] = set()
    for m in CLICK_PHRASE_RE.finditer(task):
        ph = m.group(0).strip()
        ph_toks = _content(ph)
        if not ph_toks:
            continue
        # accnames fuse the phrase's words ("log out" vs 'Logout'): the
        # space-free join lets a spaced two-word phrase match a fused name
        if len(ph_toks) == 2:
            ph_toks = ph_toks | {"".join(re.findall(r"[a-z0-9]+", ph.lower()))}
        verb = m.group(1).lower()
        selectish = verb in _SELECT_VERBS or verb.startswith("switch")
        # each recorded action consumes at most ONE phrase, in task order —
        # open-page and submit-form phrases often map to the same element
        # name ("Login"), and one click must not mark both steps done. The
        # other pool is tried second: the URL-stripped "open" preamble can
        # fuse a select phrase into a click-verb blob ("open <url> and
        # select … from the dropdown"), and either kind of recorded action
        # marks that segment done.
        consumed = False
        own, other = ((sels, sused), (done, used)) if selectish \
            else ((done, used), (sels, sused))
        for pool, u in (own, other):
            for j, d in enumerate(pool):
                if j not in u and (_phrase_hit(ph_toks, d[0])
                                   or (d[1] and _same_phrase(d[1], ph))):
                    u.add(j)
                    consumed = True
                    break
            if consumed:
                break
        if consumed:
            continue
        best_i, best_ov, best_btn = None, -1, False
        for i, c in enumerate(cands):
            if c["role"] in ("option", "textbox"):
                continue
            if not _phrase_hit(ph_toks, c["name"]):
                continue
            ov = len(ph_toks & _content(c["name"]))
            if ov > best_ov or (ov == best_ov and c["role"] == "button"
                                and not best_btn):
                best_i, best_ov, best_btn = i, ov, c["role"] == "button"
        if best_i is not None:
            return ph, best_i
        # a phrase that names nothing on this page is skipped, NOT returned:
        # an un-consumable no-hit phrase (the URL-stripped "open …" blob
        # prefix of T1/T5) would otherwise fire the override on every future
        # turn and block the answer route forever. Exception (v3.11.0): an
        # ordinal phrase that names nothing is an UNNAMED POSITIONAL target
        # ("open the first book's detail page") — resolve it by page
        # structure (the Nth listing item) when the page has item groups;
        # pages without them keep the skip (T1's pre-fix stand-down shape).
        om = _ORDINAL_RE.search(ph)
        if om:
            gi = _ordinal_target(cands, om.group(1).lower(), hist)
            if gi is not None:
                return ph, gi
        continue
    return None


def pending_phrases(task: str, hist: list[str]) -> list[str]:
    """Every action phrase of the task no history action has consumed, in
    task order — the handoff document's Next Steps (v3.11.9). Same
    consumption rules as pending_step, without the page matching: a phrase
    that names nothing on the CURRENT page is still pending for a successor
    that will navigate somewhere else first."""
    task = re.sub(r"https?://\S+", " ", task)
    task = re.split(r"\breport\b", task, flags=re.I)[0]
    done = _hist_actions(hist, "CLICK")
    sels = [(n, None) for n, _ in _hist_actions(hist, "SELECT")]
    used: set[int] = set()
    sused: set[int] = set()
    out: list[str] = []
    for m in CLICK_PHRASE_RE.finditer(task):
        # collapse runs of blanks (the URL-strip above can leave them) —
        # display-only: the token sets below never see whitespace
        ph = re.sub(r"\s+", " ", m.group(0)).strip()
        ph_toks = _content(ph)
        if not ph_toks:
            continue
        if len(ph_toks) == 2:
            ph_toks = ph_toks | {"".join(re.findall(r"[a-z0-9]+", ph.lower()))}
        verb = m.group(1).lower()
        selectish = verb in _SELECT_VERBS or verb.startswith("switch")
        consumed = False
        own, other = ((sels, sused), (done, used)) if selectish \
            else ((done, used), (sels, sused))
        for pool, u in (own, other):
            for j, d in enumerate(pool):
                if j not in u and (_phrase_hit(ph_toks, d[0])
                                   or (d[1] and _same_phrase(d[1], ph))):
                    u.add(j)
                    consumed = True
                    break
            if consumed:
                break
        if not consumed:
            out.append(ph)
    return out


def compose_handoff(task: str, obs_text: str, hist: list[str]) -> str:
    """The harness's context-compaction request asks the model to write a
    handoff document. Write it from the chain's own state instead
    (v3.11.9): the 4B model answers real compaction prompts with a bare
    EOS or code-fragment junk (T1 decide3110h — an empty <|im_end|> turn
    disposed the session with 8 minutes of budget left), while the
    post-compaction redrive only needs the task's own phrases and the last
    page it can act on, both already tracked here. Zero model tokens, no
    long-context cliff. The task text goes in verbatim so a fresh
    context's phrase windows rebuild exactly."""
    mu = re.search(r"URL: (\S+)", obs_text or "")
    mt = re.search(r"Title: ([^\n]*)", obs_text or "")
    page = (f"URL: {mu.group(1)}" + (f" | Title: {mt.group(1).strip()}"
                                     if mt else "")) if mu else \
        "unknown — no page observed yet"
    done = [h for h in hist
            if re.match(r"(OPEN|CLICK|SELECT|TYPE|SUBMIT)", h)]
    done_txt = "\n".join("- " + d for d in done[-8:]) or "- nothing yet"
    pend = pending_phrases(task, hist)
    dm = re.split(r"\breport\b", task, flags=re.I)
    deliver = dm[-1].strip().rstrip(".") if len(dm) > 1 and dm[-1].strip() \
        else ""
    lines = ["## Goal", task.strip(), "", "## Progress", done_txt, "",
             f"Current page: {page}", ""]
    if deliver:
        lines += ["## Key Decisions",
                  f"- Deliverable: report {deliver} — answer it once every"
                  " step below is done", ""]
    lines += ["## Next Steps"]
    for i, ph in enumerate(pend[:5], 1):
        lines.append(f"{i}. {ph.strip()}")
    lines.append(f"{len(pend[:5]) + 1}. When no step remains, produce the"
                 " deliverable from the current page.")
    return "\n".join(lines)


def superlative_index(task: str, cands: list[dict], idx: int) -> int | None:
    """A task superlative ("add the cheapest product") resolves positionally:
    among candidates sharing the picked element's exact name, cheapest/first/
    top -> first in DOM order, last/most expensive -> last. DOM order is
    display order, so after a low-to-high sort the first 'Add to cart' IS the
    cheapest product (T5) — identical names give the element read a
    near-uniform distribution and its pick among them is a coin flip."""
    if not _SUPERLATIVE_RE.search(task):
        return None
    name = cands[idx]["name"]
    same = [i for i, c in enumerate(cands) if c["name"] == name]
    if len(same) < 2:
        return None
    if re.search(r"\b(last|most expensive)\b", task, re.I):
        return same[-1]
    return same[0]


# --- unnamed positional targets (v3.11.0) ------------------------------------

_ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b",
    re.I)
_ORDINAL_N = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def _same_item_name(a: str, b: str) -> bool:
    """Two candidates are the same listing item when their names are equal
    or one is the ellipsized ("...") truncation of the other: catalogue
    cards emit the item as an image link (full title) plus a title link,
    and long titles are ellipsized in the aria read (v3.11.15)."""
    if a == b:
        return True
    if a.endswith("...") and b.startswith(a[:-3].rstrip()):
        return True
    if b.endswith("...") and a.startswith(b[:-3].rstrip()):
        return True
    return False


def _item_groups(cands: list[dict]) -> list[int]:
    """Start indices of the page's listing items, read off structure alone:
    a catalogue card emits its item twice — image link + title link share
    the accessible name (or one is the other's ellipsized truncation) — so
    a run of >= 2 consecutive link/button candidates with one name is one
    item. Navigation vocabulary (category links, pager, login) appears once
    per name and never groups."""
    runs: list[int] = []
    i, n = 0, len(cands)
    while i < n:
        c = cands[i]
        if c["role"] in ("link", "button") and c["name"]:
            j = i + 1
            while (j < n and cands[j]["role"] in ("link", "button")
                   and _same_item_name(c["name"], cands[j]["name"])):
                j += 1
            if j - i >= 2:
                runs.append(i)
            i = j
        else:
            i += 1
    return runs


def _ordinal_target(cands: list[dict], word: str,
                    hist: list[str]) -> int | None:
    """Resolve an unnamed positional target ("open the first book's detail
    page") to the Nth listing item in DOM order; last -> the final item.
    None = stand down: pages without item structure, ordinals that outrun
    the item count, and the item whose element was the previous action's
    own click (no instant re-pick) all keep the old skip behavior."""
    groups = _item_groups(cands)
    if not groups:
        return None
    if word == "last":
        idx = groups[-1]
    else:
        n = _ORDINAL_N.get(word)
        if n is None or n > len(groups):
            return None
        idx = groups[n - 1]
    m = re.search(r"on \[(e\d+)\]", hist[-1]) if hist else None
    if m and cands[idx]["ref"] == m.group(1):
        return None
    return idx


def build_read_window(task: str, hist: list[str], cands_all: list[dict],
                      log: dict) -> tuple[list[dict], tuple | None]:
    """The element read's candidate window: the first MAX_CANDIDATES
    candidates with two kinds of displacement into the tail — task-named
    candidates the cap would hide (the-internet's ~40-link home page:
    "Form Authentication" sits past the first 20 and T3 burned its deadline
    on the confusable "Basic Auth"), and (v3.11.14) the pending step's
    resolved target, resolved over ALL candidates: on long catalogue pages
    the cap is all page chrome (banner, search, category sidebar, sort,
    pager) and the listing items sit beyond it, so the ordinal-snap resolver
    saw no item groups and the answer gate stood down into the model-junk
    loop on every turn (decide3110m). Returns (window, pend); pend is
    (phrase, window-index) or None.

    (v3.11.16) the two displacements used to run pend-first and
    extras-second, and the extras rebuild retook the very tail slot the
    pend target had just claimed: on books.toscrape "next" (>= 4 chars, in
    "go to the next page") is itself task-named, so the pager link replaced
    the resolved "first book" at pend's index and the chain re-clicked it —
    decide3110n and decide3110p both log the correct resolution and the
    wrong click on one line. The extras window is built first; the pending
    step is this turn's chosen action and its target splices in ahead of
    the extras.
    """
    cands = cands_all[:MAX_CANDIDATES]
    extra: list[dict] = []
    if len(cands_all) > MAX_CANDIDATES:
        def _named_in_task(c: dict) -> bool:
            nm = c["name"].strip()
            if len(nm) < 4 or c["role"] in ("option", "textbox"):
                return False
            tn = re.sub(r"[^a-z0-9]+", " ", task.lower())
            nn = re.sub(r"[^a-z0-9]+", " ", nm.lower()).strip()
            return bool(nn) and f" {nn} " in f" {tn} "
        extra = [c for c in cands_all[MAX_CANDIDATES:] if _named_in_task(c)
                 and all(c["ref"] != x["ref"] for x in cands)]
        if extra:
            cands = cands[:max(MAX_CANDIDATES - len(extra), 1)] + extra
    pend = None
    if pending_field(task, hist)[0] is None:
        pend = pending_step(task, hist, cands_all)
        if pend is not None:
            tgt = cands_all[pend[1]]
            log["pending_scan"] = {
                "ncand_all": len(cands_all),
                "ngroups": len(_item_groups(cands_all)),
                "resolved": render_lines([tgt])[0][:80],
            }
            if all(c["ref"] != tgt["ref"] for c in cands):
                cands = (cands[:max(MAX_CANDIDATES - len(extra) - 1, 1)]
                         + [tgt] + extra)
            pend = (pend[0], next(i for i, c in enumerate(cands)
                                  if c["ref"] == tgt["ref"]))
    return cands, pend


def eval_cell(code: str, title: str) -> dict:
    # internal format; completion_payload/sse_chunks add the wire wrapping.
    # The id is minted here and remembered so the stuck-breaker can tell
    # chain-generated feed cells from model-written cells that copy the
    # same display(...) style.
    cid = "call_" + uuid.uuid4().hex[:24]
    CHAIN_CELL_IDS.add(cid)
    return {"id": cid, "name": "eval",
            "arguments": {"language": "js", "title": title, "code": code}}


def anchor_index(task: str, cands: list[dict], idx: int,
                 probs: dict[str, float] | None,
                 lines: list[str]) -> tuple[int, str | None]:
    """Exact-phrase anchor: when the task names an element verbatim ("Form
    Authentication") and the element decide picked a confusable neighbour
    ("Basic Auth" — the T3-v36 pick opened an HTTP-auth dialog and burned the
    deadline), redirect to the named candidate. A candidate anchors when its
    space-normalized name (>= 4 chars) appears in the space-normalized task.
    Among the anchored set, keep the element decide's preference."""
    def norm(s: str) -> str:
        return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "
    tn = norm(task)
    named = [i for i, c in enumerate(cands)
             if c["role"] not in ("option", "textbox")
             and len(c["name"].strip()) >= 4 and norm(c["name"]) in tn]
    if not named or idx in named:
        return idx, None
    if probs is None:
        return named[0], lines[named[0]]
    best = max(named, key=lambda i: probs.get(lines[i], 0.0))
    return best, lines[best]


def anchor_blocked(pend: tuple | None, idx: int) -> bool:
    """(v3.14.1) the exact-phrase anchor must not yank the pending step's own
    resolution. The anchor fires when a candidate's name (>= 4 chars) appears
    verbatim in the task — but short catalogue vocabulary does too: 'next'
    sits inside "go to the next page of the catalogue", so on T1 the read had
    picked the ordinal-snap target ('In Her Wake', the first book of page 2,
    conf 0.52) and the anchor yanked the emitted click back onto the pager
    (next@e516) — the same wrong click the 4B showed before v3.11.16, from a
    different override (the anchor, not the window race). When the element
    read's pick IS the pend target — named or positional, both spliced into
    the window by ref — it is already task-derived and stays."""
    return pend is not None and pend[1] == idx


def turn_kind(messages: list[dict]) -> str:
    """'observation' — the newest message is a tool result the chain may act
    on; 'prose' — browser context exists but the newest message is plain
    text (the harness's context-compaction/handoff request, a user
    interjection): the chain must stand down, because acting answers a
    summary request with a browser cell and, via the observe route,
    re-feeds a fresh ~30 KB snapshot into a context that was just
    compacted — the T1-v3110 compaction jam; 'fresh' — no tool results yet
    (the opening task, open-URL flow)."""
    def _tool(m: dict) -> bool:
        return m.get("role") == "tool" or (m.get("role") == "user"
                                           and bool(m.get("tool")))
    msgs = messages or [{}]
    if _tool(msgs[-1]):
        return "observation"
    if any(_tool(m) for m in msgs):
        return "prose"
    return "fresh"


def run_chain(req: dict) -> tuple[dict, str, dict]:
    """Decide chain on a browser-observation turn. Returns (tool_call, finish, log)."""
    global LAST_HIST
    msgs = req["messages"]
    task = extract_task(msgs)
    hist = extract_history(msgs)
    # (v3.11.12) a post-compaction context retains almost no action cells;
    # the handoff document embedded in the redrive message lists them in
    # hist format — merge them back so pending-phrase tracking resumes
    # where the predecessor left off instead of restarting at phrase 1
    hh = hist_from_handoff(msgs)
    if hh:
        have = set(hist)
        hist = [h for h in hh if h not in have] + hist
    # (v3.11.13) accumulate across turns; ref-sensitive checks (repeat
    # detection, TYPE ordinals) must use the per-request lines only —
    # recovered/accumulated lines carry refs from older page snapshots
    hist_retained = list(hist)
    if last_tool_result(msgs) or browser_in_play(msgs):
        have = set(LAST_HIST)
        LAST_HIST = LAST_HIST + [h for h in hist if h not in have]
    else:
        LAST_HIST = list(hist)  # brand-new session
    hist = LAST_HIST
    global PASSTHROUGH_MAX_NEW, NO_CAND_STREAK, LAST_TASK, LAST_TASK_FULL, \
        ANSWER_GATE_FALLBACK, ANSWER_TEXT_FALLBACK, REVIEW_NOTE
    ANSWER_GATE_FALLBACK = None  # stale stash from any earlier turn
    ANSWER_TEXT_FALLBACK = None
    REVIEW_NOTE = None
    PASSTHROUGH_MAX_NEW = None
    if task[:120] != LAST_TASK:
        LAST_TASK = task[:120]
        NO_CAND_STREAK = 0
    # (v3.11.10) extract_task recovers the goal from the handoff's Goal
    # section; if the redrive carried no such section, fall back to the
    # last real task this browser session saw. Non-redrive tasks are
    # remembered for the next redrive. (v3.11.14) this runs BEFORE the
    # candidate window is built — the pending-step resolution and the
    # task-named displacement both read the task.
    if REDRIVE_RE.search(task[:400]):
        if LAST_TASK_FULL:
            task = LAST_TASK_FULL[:600]
    else:
        LAST_TASK_FULL = task
    log = {"arm": "decide", "task": task[:120]}
    t0 = time.perf_counter()
    obs_text = last_tool_result(msgs) or ""
    # (v3.16.1) the completion review's DELTA frame needs the session's
    # FIRST snapshot — captured here on every observation turn while the
    # feature is enabled; zero cost when the config leaves it off
    if COMPLETION_REVIEW is not None and obs_text:
        review_first_obs_capture(repr(msgs[0])[:300] if msgs else "?",
                                 obs_text)
    try:  # (v3.11.14) production observation capture for post-mortems
        with open(LOGDIR / "last_obs.txt", "w") as f:
            f.write(obs_text)
    except Exception:
        pass
    cands_all = parse_snapshot(obs_text)
    try:  # (v3.11.14) production observation capture for post-mortems
        with open(LOGDIR / "last_obs.txt", "w") as f:
            f.write(obs_text)
    except Exception:
        pass
    # (v3.11.16) the read window — task-named extras plus the pending
    # step's resolved target, both displaced past the cap — is built by
    # build_read_window (extracted so the suite can regression-test the
    # slot race between the two displacements).
    cands, pend = build_read_window(task, hist, cands_all, log)
    lines = render_lines(cands)
    log["n_candidates"] = len(lines)
    if lines:
        NO_CAND_STREAK = 0
    if turn_kind(msgs) == "prose":
        # (v3.11.1) a prose turn in a browser conversation — the harness's
        # context-compaction/handoff request above all — is not an
        # observation: act on it and you either answer a summary request
        # with a browser cell or observe-feed a fresh snapshot into a
        # just-compacted context (the compaction jam loop). Pass through,
        # logged (this route used to be silent).
        tail_txt = _content_text(msgs[-1].get("content") if msgs else "").lower()
        if "handoff" in tail_txt or "context replaced" in tail_txt:
            # (v3.11.9) the compaction request: compose the handoff from
            # the chain's own state instead of asking the model — the 4B
            # model answers real compaction prompts with a bare EOS or
            # code-fragment junk, and each wasted summary turn costs the
            # task minutes of budget.
            log["routed"] = "compose-handoff"
            try:  # (v3.11.14) capture the compaction request verbatim
                with open(LOGDIR / "compaction_request.txt", "w") as f:
                    f.write(_content_text(msgs[-1].get("content")))
            except Exception:
                pass
            doc = compose_handoff(task, obs_text, hist)
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return {"compose": doc}, "compose", log
        log["routed"] = "passthrough-prose"
        # the compaction summary is the only output this route produces and
        # the post-compaction redrive reads the task phrases back out of it —
        # but it must not inherit the 4096 thinking-mode default: a
        # long-context decode runs ~150-250 ms/token, and a 4096-token cap
        # turned one compaction into an 11-minute stall that ate T1's whole
        # deadline. 1024 keeps a full handoff, ends the ramble.
        PASSTHROUGH_MAX_NEW = 1024
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return None, "passthrough", log

    if not lines:
        # No interactive elements in view. Cases, in order: (1) the last two
        # cells were already observe/open feeds — a genuinely element-less
        # page, stop feeding and let the model navigate or answer; (2) the
        # transcript shows the browser somewhere that is neither the task URL
        # nor a page under it — navigate back now; (3) blind (bare snapshot
        # results carry no URL: header) or on-site but element-less — give
        # the current tab ONE observe before navigating: a just-executed
        # action is often mid-navigation on a page that has plenty of
        # elements (v3.3 re-opened saucedemo's logged-in inventory page back
        # to the login form here, and the model re-ran the whole login until
        # the deadline); (4) no browser yet — first move, open the task URL.
        m_url = re.search(r"https?://[^\s'\")<>]+", task)
        url = m_url.group(0).rstrip(".,;") if m_url else None
        # A stuck loop means two CHAIN feed cells (open/observe) in a row.
        # Model-written junk cells can end with the same display(...) call —
        # v3.4 counted those too and handed control back to the model right
        # when the chain had candidates to act on.
        events = eval_events(msgs)
        feed = (lambda i_c: i_c[0] in CHAIN_CELL_IDS
                and "//" not in i_c[1]
                and i_c[1].strip().endswith("display(await tab.ariaSnapshot());"))
        open_urls = [u.rstrip("/") for u in re.findall(r"URL: (\S+)", obs_text)]
        task_r = (url or "").rstrip("/")
        elsewhere = bool(open_urls) and not any(
            u == task_r or u.startswith(task_r + "/")
            or task_r.startswith(u + "/") for u in open_urls)
        if elsewhere and url:
            NO_CAND_STREAK = 0
        elif browser_in_play(msgs) and NO_CAND_STREAK >= 1 and url:
            # one recovery observe already happened and still nothing — the
            # page is genuinely dead (401/404/blank): go back to the task URL
            NO_CAND_STREAK = 0
        elif browser_in_play(msgs) and len(events) >= 2 and all(feed(e) for e in events[-2:]):
            log["routed"] = "passthrough-stuck"
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return None, "passthrough", log
        elif browser_in_play(msgs):
            # put an observation on the floor so the next turn has
            # [ref=eN] candidates to decide over (v3.4: BEFORE any re-open)
            NO_CAND_STREAK += 1
            log["routed"] = "observe"
            log["no_cand_streak"] = NO_CAND_STREAK
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return eval_cell(f'const tab = await browser.tab("{tab_name(msgs) or "main"}");\n'
                             'display(await tab.ariaSnapshot());',
                             "browser observe"), "tool_calls", log
        elif not url:
            if re.search(r"ToolError|timed out|TypeError|ReferenceError", obs_text):
                # the last cell failed: hand the turn to the vanilla model,
                # which sees the error and adapts; the chain only acts on
                # clean reads.
                log["routed"] = "passthrough-error"
                log["latency_ms"] = (time.perf_counter() - t0) * 1000
                log_decision({**log, "ts": time.time()})
                return None, "passthrough", log
            log["routed"] = "passthrough-first"
            return None, "passthrough", log
        else:
            NO_CAND_STREAK = 0
        name = tab_name(msgs) or "main"
        log["routed"] = "open"
        log["url"] = url
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return eval_cell(f'const tab = await browser.open({{ name: "{name}", url: "{url}" }});\n'
                         'display(await tab.ariaSnapshot());',
                         "browser open"), "tool_calls", log

    op_res = DECIDE_FN(ctx(task, hist, len(lines)), OP_LINES_FIRST,
                       method="d2", alpha=1.0, anchor=True)
    op = op_res.selected.split(" — ")[0]
    top3 = sorted(op_res.probabilities.items(), key=lambda kv: -kv[1])[:3]
    log["op_first"] = {"op": op, "conf": op_res.confidence,
                       "premask": op_res.premask_mass,
                       "top3": [[k[:40], round(v, 4)] for k, v in top3]}
    # ANSWER routing is new (the PoC had only action steps) and uncalibrated.
    # A bare margin override is not enough: the small model says ANSWER on
    # nearly every post-task turn, and forcing the best action then once
    # clicked its way through the whole page (T2-v33). So when the op read
    # says ANSWER, still run the element read and act only if it finds a
    # solid target (element confidence >= ANSWER_GATE); otherwise hand the
    # turn to the model, which can actually decide the task is finished.
    forced = None
    pending_fired = False
    if op not in OPS:
        forced = max(((k.split(" — ")[0], v) for k, v in op_res.probabilities.items()
                      if k.split(" — ")[0] in OPS), key=lambda kv: kv[1])[0]
        log["answer_forced_op"] = forced
    eff_op = op if op in OPS else forced
    if eff_op is None or not lines:
        PASSTHROUGH_MAX_NEW = 128
        log["routed"] = "passthrough-answer"
        # (v3.16.2) third, log-only review site: an answer turn on a page
        # with no interactive candidates — the model answers unreviewed
        # here too. Same guard as the other answer-ok site.
        if (COMPLETION_REVIEW is not None and forced is not None
                and obs_text):
            review_frames_run(task, msgs, obs_text, log, "no-candidates")
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return None, "passthrough", log
    op = eff_op

    if len(lines) >= 2:
        res = DECIDE_FN(ctx(task, hist, len(lines), op), lines,
                        method="d2", alpha=0.5, anchor=True)
        idx = lines.index(res.selected) if res.selected in lines else 0
        tempered = with_temperature(res, GATE["temperature"])
        element_log = {"selected": lines[idx], "conf": res.confidence,
                       "conf_tempered": tempered.confidence,
                       "premask": res.premask_mass,
                       "escalated_flag": tempered.confidence < GATE["accept_conf"]}
    else:
        # single interactive element: nothing to choose between
        idx, tempered, res = 0, None, None
        element_log = {"selected": lines[idx], "conf": None,
                       "single_candidate": True}
    # (v3.15.0) the task ledger, explicit in the decision log: which action
    # phrases the task still has open after this turn's history is applied.
    # Observability only — enforcement already exists (the v3.8.0 pending
    # override fires the pending step on ANSWER misroutes); this makes the
    # state legible from decisions.jsonl alone and feeds the review flags.
    _ledger = pending_phrases(task, hist)
    if _ledger:
        log["ledger"] = {"pending": [p[:60] for p in _ledger],
                         "n": len(_ledger)}
    if forced is not None:  # ANSWER-forced: the element read is the gate
        # raw confidence, not tempered — tempering resharpens the
        # distribution and reads 0.5+ even on 0.1-level garbage
        gate_conf = res.confidence if res is not None else 1.0
        ok = gate_conf >= ANSWER_GATE
        # v3.8.0 pending-step override: ANSWER with work left is a misroute.
        # If a task step is still undone (no TYPE field pending — typed steps
        # come first) and its phrase maps to a candidate, act on it instead
        # of answering; if the phrase names nothing here, open the gate and
        # let the element read's pick stand (T1's "first book" case).
        # (v3.11.14) pend was resolved over the FULL candidate list at
        # window-build time, with its target displaced into the read —
        # recomputing on the truncated list is what stood down: the listing
        # items live beyond the window and the resolver saw no item groups.
        pending_fired = False
        if pend is not None:
            ph, pi = pend
            if pi is not None:
                idx = pi
            # audit: a phrase-mapped pick name-hits its own phrase; a
            # positional (ordinal-snap) pick shares no vocabulary with it
            positional = not _phrase_hit(_content(ph), cands[idx]["name"])
            # (v3.11.16) the superlative same-name collapse re-picks only
            # for the element read's own selection (pi None — T5's "cheapest
            # Add to cart"); a positional pend pick IS the task's resolution
            # (the item group's start, already first of its name run), so
            # the collapse must not move it.
            si = None if positional else superlative_index(task, cands, idx)
            if si is not None:
                idx = si
            sel_line = lines[idx]
            ref = cands[idx]["ref"]
            element_log["ref"] = ref
            element_log["pending_swap"] = sel_line
            element_log["pending_phrase"] = ph
            log["element"] = element_log
            log["pending_override"] = {"phrase": ph, "idx": idx,
                                       "named": not positional,
                                       "positional": positional}
            pending_fired = True
        log["answer_gate"] = {"el_conf": gate_conf, "ok": ok}
        if not ok and not pending_fired:
            # (v3.11.11) the model handoff below is fatal on long
            # (post-compaction) contexts: the 4B echoes the system prompt's
            # JS template as raw content and the harness disposes the
            # session (decide3110i/j both died HERE, after the composed
            # handoff and task recovery had worked). Stash the chain's own
            # action — the element read's pick, or its best FRESH pick when
            # the top one is already done — and compute() ships it if the
            # model's answer comes back malformed or empty. One-shot.
            def _done(i: int) -> bool:
                r = f"[{cands[i]['ref']}]"
                return any(h.startswith(op) and r in h for h in hist)

            # (v3.11.14) when the gate fell through with no pending action
            # and the task's remaining clause is a report ("Report the …"),
            # the deliverable is the page's own subject: stash the page's
            # main heading as the answer instead of a speculative click —
            # the click fallback kept the session alive but wandered
            # (decide3110m ended up re-clicking the Travel link), and the
            # model at this context size cannot produce the answer itself.
            # Generic page structure — the first heading in the snapshot,
            # no site or task vocabulary.
            if (forced is not None
                    and re.search(r"\breport\b", task, re.I)):
                # headings are not interactive (never candidates): read the
                # first heading line straight off the snapshot, document
                # order = the page's main heading
                hm = re.search(r'^\s*-\s+heading\s+"([^"]+)"', obs_text, re.M)
                if hm:
                    ANSWER_TEXT_FALLBACK = hm.group(1).strip()[:300]
                    log["gate_answer_fallback"] = ANSWER_TEXT_FALLBACK[:60]
            if ANSWER_TEXT_FALLBACK is None and op not in ("TYPE", "SELECT"):
                order = [idx] + sorted(
                    (i for i in range(len(lines)) if i != idx),
                    key=lambda i: -(res.probabilities.get(lines[i], 0.0)
                                    if res is not None else 0.0))
                pick = next((i for i in order if not _done(i)), None)
                if pick is not None:
                    pord = 1 + sum(1 for c in cands[:pick]
                                   if c["role"] in ("textbox", "searchbox",
                                                    "textarea"))
                    ANSWER_GATE_FALLBACK = eval_cell(
                        action_code(tab_name(msgs) or "main", op,
                                    cands[pick]["ref"], None,
                                    cands[pick]["name"], pord),
                        f"browser {op.lower()}")
                    log["gate_fallback"] = {"op": op,
                                            "element": lines[pick][:60]}
            PASSTHROUGH_MAX_NEW = 128
            log["routed"] = "passthrough-answer"
            # (v3.15.0) log-only completion-review flag: the chain declined
            # to act on this turn, so whatever the model reports next is
            # uncorroborated by a chain action. Never routes — the
            # checkpoint-review experiment measured the self-read brake at
            # 77% detection but 32% false alarms (59% on chain sessions),
            # so this exists for post-hoc review and the harness's own
            # objective channel, not as a gate.
            log["review"] = ["completion-unverified:answer-gate-fallthrough"]
            # (v3.16.0-v3.16.2) optional completion self-review
            # (config-gated, default OFF). v3.16.1: the single DONE/NOT-DONE
            # read becomes the validated 3-frame MINIMUM from the
            # single-model framing backtest (2026-09-22, 596 ground-truth
            # probes): GOAL (task + newest snapshot), DELTA (session-first
            # snapshot vs newest — change evidence), COND (the model's own
            # last assertion claim-checked, the verdict fed to the done
            # read). v3.16.2: the reads move into review_frames_run and a
            # second, log-only trigger site covers the gate-PASSED answer
            # path (the stand-down exits after the answer gate passed —
            # exactly where the 2026-09-22 live run's wrong report slipped
            # through unreviewed). The challenge still exists ONLY here at
            # the fallthrough, and only in gate mode; mode "annotate"
            # attaches a verification caveat to the model's prose answer
            # when all three frames read NOT-done (the measured ~89%-precision
            # point), instead of ever blocking.
            if COMPLETION_REVIEW is not None:
                _rp, _pmin = review_frames_run(
                    task, msgs, obs_text, log, "answer-gate-fallthrough")
                if review_gate_fires(COMPLETION_REVIEW, _pmin,
                                     ANSWER_GATE_FALLBACK is not None):
                    _fb, ANSWER_GATE_FALLBACK = \
                        ANSWER_GATE_FALLBACK, None
                    log["review_gate"] = "challenge:continue"
                    log["latency_ms"] = (time.perf_counter() - t0) * 1000
                    log_decision({**log, "ts": time.time()})
                    return _fb, "tool_calls", log
                if (_rp is not None
                        and COMPLETION_REVIEW["mode"] == "annotate"
                        and all(v < COMPLETION_REVIEW["p_done_threshold"]
                                for v in _rp.values())):
                    REVIEW_NOTE = ("[verification note] independent page "
                                   "checks did not confirm the task's "
                                   "completion; treat this report as "
                                   "unverified.")
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return None, "passthrough", log
    # exact-phrase anchor runs AFTER the gate so a gate-failing pick is
    # still handed to the model even if the task names some other element,
    # and ONLY on navigation steps: the task mentions "login" on every T2
    # turn, and the v3.7 anchor yanked the type pick off the textbox onto
    # the Login link mid-task (v371 guard: skip on TYPE ops and on textbox
    # picks — value steps have their own field-needle logic below). The
    # pending override's pick is already task-derived: the anchor must not
    # yank it again (on the T5 add turn the anchor's verbatim match is the
    # "Cart" header link — navigating there instead of adding to cart).
    if (res is not None and op != "TYPE" and cands[idx]["role"] != "textbox"
            and not pending_fired and not anchor_blocked(pend, idx)):
        idx2, swapped = anchor_index(task, cands, idx, res.probabilities, lines)
        if swapped is not None:
            idx = idx2
            sel_line = lines[idx]
            ref = cands[idx]["ref"]
            element_log["ref"] = ref
            element_log["anchor_swap"] = swapped
            log["element"] = element_log
    if (not pending_fired and op != "TYPE"
            and cands[idx]["role"] != "textbox"):
        si = superlative_index(task, cands, idx)
        if si is not None and si != idx:
            idx = si
            sel_line = lines[idx]
            ref = cands[idx]["ref"]
            element_log["ref"] = ref
            element_log["superlative_swap"] = sel_line
            log["element"] = element_log
    sel_line = lines[idx]
    ref = cands[idx]["ref"]
    element_log["ref"] = ref
    log["element"] = element_log

    reread = DECIDE_FN(ctx(task, hist, len(lines), None,
                           options=lines, chosen=sel_line),
                       OP_LINES_ELEM, method="d2", alpha=1.0, anchor=True)
    op_final = (OPS[OP_LINES_ELEM.index(reread.selected)]
                if reread.selected in OP_LINES_ELEM else op)
    # dropdown-ish elements are not clickable targets: their action is
    # selecting an option (T5-v36 burned the deadline clicking a
    # <select> that the click matcher rightly refuses to touch)
    combobox_flipped = False
    if op_final == "CLICK" and cands[idx]["role"] in ("combobox", "listbox"):
        op_final = "SELECT"
        combobox_flipped = True
    log["op_reread"] = {"op": op_final, "conf": reread.confidence,
                        "combobox_flip": combobox_flipped}

    # field-needle swap: the task binds each value to a named field ("into
    # the password field"), and the element decide sometimes grabs the
    # neighbouring input (T5-v32 typed the password over the username).
    # Redirect to the candidate whose name carries the pending field, but
    # never onto an already-filled field, and never onto a name matching a
    # satisfied one (quotes' merged "Username Password" name). Runs after the
    # op re-read so a TYPE that emerges there is still redirected.
    if op_final == "TYPE":
        fld, filled = pending_field(task, hist)
        if fld:
            names = [c["name"].strip().lower() for c in cands]
            ok = fld in names[idx] and not any(s in names[idx] for s in filled)
            if not ok:
                swap = next((i for i, n in enumerate(names)
                             if fld in n and i != idx
                             and not any(s in n for s in filled)), None)
                if swap is not None:
                    idx = swap
                    sel_line = lines[idx]
                    ref = cands[idx]["ref"]
                    element_log["ref"] = ref
                    element_log["field_swap"] = sel_line
                    log["element"] = element_log

    value = None
    if op_final in ("TYPE", "SELECT"):
        value = VALUE_FN(task, hist, op_final, sel_line, len(lines))
        if op_final == "SELECT":
            value = select_switch_override(task, value, ref, hist)
        log["value"] = value
        # (v3.14.4) the thinker answered the ask instead of the task (a
        # refusal sentence or nothing): firing types the refusal INTO the
        # page. Stand down — the model's own hands handled these turns
        # correctly every time it got them.
        bad = bad_payload(value)
        if bad:
            log["stand_down"] = bad
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return None, "passthrough", log

    # this exact action (same op, element, value) is already anywhere in the
    # session history: the task's step was done, so hand the turn to the
    # model instead of redoing it (e.g. after a tab close/reopen cycle).
    # Selecting the same dropdown with a DIFFERENT value is legitimate
    # progress, so the value must match to count as a repeat. The [ref]
    # brackets keep e2 from matching history entries about e20 — and this
    # check is deliberately REF-based, so it reads only the per-request
    # lines (hist_retained): accumulated lines carry refs from older page
    # snapshots where the same eN names a different element.
    def _repeated(i: int) -> bool:
        r = f"[{cands[i]['ref']}]"
        return any(h.startswith(op_final) and r in h
                   and (value is None or f"'{value}'" in h)
                   for h in hist_retained)

    # v3.14.3: the task's steps are ALL consumed and the pick re-clicks a
    # name some earlier cell already acted on — the model is re-running a
    # finished step after a navigation shifted refs, which the ref-based
    # check below cannot see (T3-v3142: after the tagged Logout the op read
    # re-clicked 'Login' on the emptied form at its new ref e23, and the
    # bogus "username invalid" flash then poisoned the model's report; the
    # same stray click sat unnoticed in the passing v3.14.0 run). Pagination
    # re-clicks never reach this branch: they only fire while a next-page
    # phrase is still pending, and pending_phrases is empty here.
    if (op_final == "CLICK" and not pending_phrases(task, hist)
            and any(h.startswith("CLICK") and f"'{cands[idx]['name']}'" in h
                    for h in hist)):
        log["stand_down"] = "all-consumed-name-repeat"
        # (v3.16.2) second, log-only review site: the answer gate PASSED but
        # the turn still hands to the model here — the exact uncovered path
        # of the 2026-09-22 live run's wrong report. Records the three frame
        # verdicts; never challenges, never annotates.
        if COMPLETION_REVIEW is not None and forced is not None:
            review_frames_run(task, msgs, obs_text, log,
                              "answer-ok-standdown")
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return None, "passthrough", log

    if _repeated(idx):
        # the top pick is already done: walk the element decide's ranking for
        # a fresh candidate rather than either redoing the step or parking
        # the chain on repeat-passthrough while the model flails.
        alt = None
        if res is not None:
            ranked = sorted(range(len(lines)),
                            key=lambda i: -res.probabilities.get(lines[i], 0.0))
            # a fresh candidate must also clear the confidence floor — a
            # low-confidence escalation once produced a click on an unnamed
            # element right after the task was already done
            alt = next((i for i in ranked
                        if not _repeated(i)
                        and res.probabilities.get(lines[i], 0.0) >= ANSWER_GATE),
                       None)
        if alt is not None:
            idx = alt
            ref = cands[idx]["ref"]
            element_log["ref"] = ref
            element_log["escalated_to"] = lines[idx]
            log["element"] = element_log
            log["escalated"] = True
        else:
            log["routed"] = "passthrough-repeat"
            # (v3.16.2) second, log-only review site (see the
            # all-consumed-name-repeat stand-down above): gate-passed answer
            # turn handing to the model unreviewed before this change.
            if COMPLETION_REVIEW is not None and forced is not None:
                review_frames_run(task, msgs, obs_text, log,
                                  "answer-ok-standdown")
            log["latency_ms"] = (time.perf_counter() - t0) * 1000
            log_decision({**log, "ts": time.time()})
            return None, "passthrough", log

    # 1-based ordinal of the chosen element among textbox-like candidates
    # (parse order == DOM order); the page-side TYPE cell falls back to it
    # when the aria name is missing or merged with a neighbour's. A merged or
    # unnamed field re-picked for the next value must advance past what
    # earlier TYPE cells already filled through this same ref (T2-v33 typed
    # the password over the username on quotes' merged textbox).
    tbox_ord = 1 + sum(1 for c in cands[:idx]
                       if c["role"] in ("textbox", "searchbox", "textarea"))
    tbox_ord += sum(1 for h in hist_retained
                    if h.startswith("TYPE") and f"[{ref}]" in h)
    # (v3.14.2) the emitted click IS the pending step's positional
    # resolution — via the forced override or the element read's own pick,
    # both land on idx == pend[1]: tag the cell with the phrase it executes
    # so later turns consume it (an ordinal phrase shares no vocabulary with
    # the element it resolved to, and untagged it re-resolved on the detail
    # page the click had just opened — the T1 wander through product pagers).
    exec_phrase = None
    if (op_final == "CLICK" and pend is not None and idx == pend[1]
            and not _phrase_hit(_content(pend[0]), cands[idx]["name"])):
        exec_phrase = pend[0]
    code = action_code(tab_name(msgs) or "main", op_final, ref, value,
                       cands[idx]["name"], tbox_ord, task_phrase=exec_phrase)
    log["code"] = code
    if forced is not None:
        # (v3.15.0) log-only review flag: this turn ACTED while the op read
        # said ANSWER — the completion story is unverified by construction.
        log["review"] = ["acted-under-answer"]
    log["latency_ms"] = (time.perf_counter() - t0) * 1000
    log_decision({**log, "ts": time.time()})
    return eval_cell(code, f"browser {op_final.lower()}"), "tool_calls", log


# --- coding adapter (v3.9.0 generalization) ---------------------------------
# Same decision core as the browser chain, different surfaces. Observation
# side: `ls -la` output -> [fN] file candidates (instead of aria snapshots).
# Action side: omp `write` / `bash` tool calls (instead of eval cells).
# Phrase side: create/run/chmod/fix imperatives over file targets (instead
# of click/select imperatives over page elements). Step content is
# template-class only: a task-quoted echo string, or a two-template script
# library (echo-script, sum-script). Anything beyond the templates is NOT
# actionable for the chain — the phrase is skipped as unactionable and the
# turn passes through to the model, which authors the change; the chain
# resumes executing and gating on the next phrase (the C3/C5 hybrid shape).
# Per-domain gate table: starts at the calibrated browser operating point
# and is calibrated on coding decisions of its own.

OP_LINES_CODE = [
    "CREATE — create or write the next file or script the task asks for",
    "RUN — run the script or command the task asks to run",
    "OBSERVE — look at the workspace or the last command output first",
    "ANSWER — the task's steps are done; report the result to the user in text",
]
OPS_CODE = {ln.split(" — ")[0] for ln in OP_LINES_CODE}
CODE_GATE = {"accept_conf": GATE["accept_conf"],
             "temperature": GATE["temperature"],
             "answer_gate": ANSWER_GATE}
CODE_PHRASE_STOP = {"the", "a", "an", "to", "and", "or", "of", "in", "on",
                    "for", "with", "then", "your", "our", "it", "its",
                    "that", "this", "from", "by", "at", "as", "into", "up",
                    "so", "again", "both", "them", "they", "order", "value",
                    "output", "script", "file", "please"}
FILE_TOK_RE = re.compile(r"[\w./-]*[\w-]+\.(sh|py|txt|js|json|csv|md)\b", re.I)
# verbs whose phrase window may contain the next verb ("run setup.sh and
# then run main.sh" is TWO steps); windows are cut at the next verb and at
# clause punctuation INCLUDING the em-dash (v3.8.0's comma-blob lesson)
CODE_PHRASE_VERB_RE = re.compile(
    r"\b(create|make|write|add|run|execute|start|launch|chmod|fix|edit|"
    r"modify|update|change|delete|remove)\b", re.I)
CODE_CREATE_VERBS = {"create", "make", "write", "add"}
CODE_EDIT_VERBS = {"fix", "edit", "modify", "update", "change"}
CODE_RUN_VERBS = {"run", "execute", "start", "launch"}
CODE_PRONOUN_RE = re.compile(r"\b(it|that|this|them|both|they)\b", re.I)


def code_cwd(messages: list[dict]) -> str:
    """The harness stamps the working directory into the user message."""
    for m in messages:
        if m.get("role") == "user":
            t = _content_text(m.get("content")) or ""
            mm = re.search(r"current working directory: '([^']+)'", t)
            if mm:
                return mm.group(1)
    return "."


def extract_code_hist(messages: list[dict]) -> list[dict]:
    """Prior coding steps, conversation order: {kind, target, raw}.
    kind: WRITE/EDIT/READ (file tools) or RUN/CHMOD (bash, CHMOD when the
    command chmods). target: basename of the file the entry acts on (first
    file token for bash commands). Both chain and model entries count —
    the model's own edits consume task phrases exactly like chain cells."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            name = f.get("name")
            if name not in ("bash", "write", "edit", "read"):
                continue
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            args = args or {}
            if name == "bash":
                cmd = args.get("command") or ""
                tok = FILE_TOK_RE.search(cmd)
                kind = "CHMOD" if re.search(r"\bchmod\b", cmd) else "RUN"
                out.append({"kind": kind,
                            "target": tok.group(0).rsplit("/", 1)[-1] if tok else None,
                            "tpath": tok.group(0) if tok else None,
                            "raw": cmd})
            else:
                p = args.get("path") or ""
                out.append({"kind": name.upper(),
                            "target": p.rsplit("/", 1)[-1], "tpath": p,
                            "raw": p})
    return out


def parse_ls(text: str) -> list[dict]:
    """Files/dirs from an `ls -la` observation -> candidates."""
    out = []
    for ln in (text or "").splitlines():
        parts = ln.split(None, 8)
        if len(parts) < 9 or not re.match(r"^[-dl][rwxsStT.-]{9}$", parts[0]):
            continue
        name = parts[8].split()[0] if parts[8].split() else ""
        if name in (".", "..", ""):
            continue
        out.append({"role": "dir" if parts[0].startswith("d") else "file",
                    "name": name})
    return out


def ctx_code(task: str, hist: list[dict], n_cand: int, op: str | None = None,
             options: list[str] | None = None,
             chosen: str | None = None) -> str:
    """Coding mirror of ctx(): same shape, workspace files as candidates."""
    lines = [f"Task: {task}", "", "Steps taken so far:"]
    lines += [f"- {h['kind']} {h['target'] or h['raw'][:60]}" for h in hist]
    if not hist:
        lines.append("- (none yet)")
    lines.append("")
    lines.append(f"The workspace has {n_cand} candidate files (listed in the options).")
    lines.append("")
    if op is None:
        offered = ", ".join(sorted(OPS_CODE))
        lines.append("What should the next step's action be? The operation must be "
                     f"one of {offered}.")
    else:
        lines.append(f"The next step's operation is {op}. Which file should it act on?")
    if options:
        lines.append("")
        lines.append("Options:")
        lines += options
    if chosen:
        lines.append("")
        lines.append(f"The chosen file is: {chosen}")
    return "\n".join(lines)


def code_cell(kind: str, path: str | None, command: str | None,
              content: str | None, intent: str) -> dict:
    """All chain actions are bash cells: writes go through heredocs so a
    missing parent directory (`src/`) is created by the same cell and the
    bytes land verbatim under a quoted delimiter."""
    cid = "call_" + uuid.uuid4().hex[:24]
    if kind == "WRITE":
        d = path.rsplit("/", 1)[0] if "/" in path else "."
        cmd = (f"mkdir -p '{d}' && cat > '{path}' <<'CHAIN_EOF'\n"
               f"{content}CHAIN_EOF\n")
        return {"id": cid, "name": "bash",
                "arguments": {"command": cmd, "i": intent}}
    return {"id": cid, "name": "bash",
            "arguments": {"command": command, "i": intent}}


def _code_content(target: str, window: str, task: str) -> str | None:
    """Template-class content only. Task-quoted echo string first, then the
    two-script library (echo-script, sum-script); None = not a template —
    the model must author it."""
    m = re.search(r"prints?\s+exactly:?\s*(.+)$", window, re.I)
    if m:
        val = m.group(1).strip().strip("'\"").rstrip(".,;")
        if val:
            return f'#!/bin/bash\necho "{val}"\n'
    if target.endswith(".py") and re.search(r"\bsum\b", task, re.I):
        toks = [t.group(0) for t in FILE_TOK_RE.finditer(task)
                if t.group(0).endswith(".txt")]
        if len(toks) == 1:
            return f'print(sum(int(l) for l in open("{toks[0]}") if l.strip()))\n'
    return None


def _wrote_entry(e: dict) -> bool:
    """Entries that put a file on disk: write/edit tools, or a bash command
    that writes/patches (heredoc >, sed, patch, tee)."""
    return e["kind"] in ("WRITE", "EDIT") or (
        e["kind"] == "RUN" and re.search(r">|sed |patch |tee ", e["raw"] or ""))


def _basename(p: str | None) -> str:
    return (p or "").rsplit("/", 1)[-1]


def pending_code_step(task: str, hist: list[dict], cands: list[dict],
                      cwd: str, forced_target: str | None = None) -> dict | None:
    """The task's earliest unconsumed imperative phrase, resolved to an
    executable step — or None (nothing pending, or the pending step needs
    authoring beyond the chain's templates, which is the model's job).
    Phrases are consumed in task order, one history entry per phrase —
    the browser override's order discipline, file targets in place of
    element names. The first unconsumed phrase decides: if it cannot be
    made actionable, nothing fires (a later actionable phrase must NOT
    jump the queue — that would act out of task order)."""
    task = re.split(r"\breport\b", task, flags=re.I)[0]
    verbs = list(CODE_PHRASE_VERB_RE.finditer(task))
    # sentence boundary = period at a word edge; a period inside a file
    # token (setup.sh) is NOT a boundary — file names always carry one
    CLAUSE_SPLIT = re.compile(r"\.(?=\s|$)|[;,\u2013\u2014]|\breport\b")
    windows = []
    for k, m in enumerate(verbs):
        end = verbs[k + 1].start() if k + 1 < len(verbs) else len(task)
        w = CLAUSE_SPLIT.split(task[m.start():end])[0]
        windows.append((m.group(1).lower(), w.strip()))
    used: set[int] = set()

    def _targets(e: dict) -> set:
        return {_basename(t.group(0)) for t in FILE_TOK_RE.finditer(e["raw"] or "")} \
            | ({_basename(e["target"])} if e["target"] else set())

    for verb, win in windows:
        win_files = [t.group(0) for t in FILE_TOK_RE.finditer(win)]
        task_files = sorted({_basename(t.group(0))
                             for t in FILE_TOK_RE.finditer(task)})
        vcls = ("create" if verb in CODE_CREATE_VERBS
                else "edit" if verb in CODE_EDIT_VERBS
                else "run" if verb in CODE_RUN_VERBS else "chmod")
        if re.search(r"\bexecutable\b|\bchmod\b", win, re.I):
            vcls = "chmod"  # "make both executable" is chmod, not create
        # resolve the phrase's target: the in-window token carries the task's
        # own path ("src/setup.sh"); matching against history is by basename,
        # but the executed path prefers the file actually written before
        # ("run setup.sh" must run the src/setup.sh that was created)
        target = win_files[0] if win_files else None
        pronoun = bool(CODE_PRONOUN_RE.search(win)) or target is None
        lw = None
        if target is None:
            lw = next((e for e in reversed(hist)
                       if _wrote_entry(e) and e.get("tpath")), None)
            if lw:
                target = lw["tpath"]
        if target is None and len(task_files) == 1:
            target = task_files[0]
            pronoun = False
        if target is None and forced_target:
            target = _basename(forced_target)
            pronoun = False
        tbase = _basename(target) if target else None
        consumed = False
        for j, e in enumerate(hist):
            if j in used:
                continue
            tg = _targets(e)
            if vcls == "chmod":
                hit = e["kind"] == "CHMOD" and (
                    not win_files or any(f in tg for f in map(_basename, win_files)))
            elif vcls == "run":
                hit = e["kind"] == "RUN" and tbase is not None and tbase in tg
            else:  # create / edit
                hit = _wrote_entry(e) and tbase is not None and tbase in tg
            if hit:
                used.add(j)
                consumed = True
                break
        if consumed:
            continue
        # first unconsumed phrase: resolve to a step, or stand down
        if vcls == "chmod":
            # chmod targets: files on disk so far without a chmod entry
            chmodded = {t for e in hist if e["kind"] == "CHMOD" for t in _targets(e)}
            made: list[str] = []
            for e in hist:
                if _wrote_entry(e) and e.get("tpath"):
                    nm = _basename(e["tpath"])
                    if nm and nm not in {m.rsplit("/", 1)[-1] for m in made}:
                        made.append(e["tpath"] if e["tpath"].startswith("/")
                                    else f"{cwd.rstrip('/')}/{e['tpath']}")
            made = [m for m in made if _basename(m) not in chmodded]
            if not made:
                return None
            arglist = " ".join(made)
            return {"phrase": win, "op": "CHMOD", "cell": code_cell(
                "RUN", None, f"chmod +x {arglist}", None,
                f"chain: make {', '.join(_basename(m) for m in made)} executable")}
        if target is None:
            return None
        if vcls == "run" and pronoun is not None and lw is None:
            # a bare/pronoun run of a file written earlier: execute the
            # actual path that was written ("run setup.sh" -> src/setup.sh)
            we = next((e for e in reversed(hist) if _wrote_entry(e)
                       and e.get("tpath") and _basename(e["tpath"]) == tbase), None)
            if we:
                target = we["tpath"]
        abspath = target if target.startswith("/") else f"{cwd.rstrip('/')}/{target}"
        if vcls == "run":
            cmd = f"python3 {abspath}" if tbase.endswith(".py") else f"bash {abspath}"
            # no rerun belt here: "run X again" tasks legitimately repeat the
            # exact command, and the phrase consumption already marks each
            # chain run consumed on the next turn
            return {"phrase": win, "op": "RUN", "target": tbase,
                    "cell": code_cell("RUN", None, cmd, None,
                                      f"chain: run {tbase}")}
        if vcls in ("create", "edit"):
            content = _code_content(tbase, win, task)
            if content is None:
                return None  # needs authoring: the model's job
            if any(_wrote_entry(e) and tbase in _targets(e)
                   and content in (e["raw"] or "") for e in hist):
                return None  # identical bytes already on disk: no rewrite loops
            return {"phrase": win, "op": "WRITE", "target": target,
                    "cell": code_cell("WRITE", abspath, None, content,
                                      f"chain: write {target} per task step")}
        return None
    return None


def last_wrote_target(hist: list[dict]) -> str | None:
    """File most recently put on disk (pronoun 'run it' / 'fix it' target)."""
    for e in reversed(hist):
        if _wrote_entry(e) and e["target"]:
            return _basename(e["target"])
    return None


def run_code_chain(req: dict) -> tuple[dict, str, dict]:
    """Decide chain on a coding turn. Same skeleton as run_chain: survey the
    workspace, op read, pending-step answer-route override, repeat guard —
    over file candidates and write/bash cells. Returns (tool_call, finish, log)."""
    msgs = req["messages"]
    task = extract_task(msgs)
    hist = extract_code_hist(msgs)
    cwd = code_cwd(msgs)
    obs = last_tool_result(msgs) or ""
    cands = parse_ls(obs)
    lines = [f"[f{i}] {c['role']} {c['name']}" for i, c in enumerate(cands)]
    t0 = time.perf_counter()
    global PASSTHROUGH_MAX_NEW
    PASSTHROUGH_MAX_NEW = None
    log = {"arm": "decide-code", "domain": "coding", "task": task[:120],
           "n_candidates": len(lines)}

    if not hist:
        # first move: put the workspace on the floor (the browser chain's
        # open-the-URL analogue)
        log["routed"] = "survey"
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return code_cell("RUN", None, "pwd && ls -la", None,
                         "chain: survey the workspace"), "tool_calls", log

    op_res = DECIDE_FN(ctx_code(task, hist, len(lines)), OP_LINES_CODE,
                       method="d2", alpha=1.0, anchor=True)
    op = op_res.selected.split(" — ")[0]
    top3 = sorted(op_res.probabilities.items(), key=lambda kv: -kv[1])[:3]
    log["op_read"] = {"op": op, "conf": op_res.confidence,
                      "premask": op_res.premask_mass,
                      "top3": [[k[:40], round(v, 4)] for k, v in top3]}
    if op not in OPS_CODE:
        op = max(((k.split(" — ")[0], v) for k, v in op_res.probabilities.items()
                  if k.split(" — ")[0] in OPS_CODE), key=lambda kv: kv[1])[0]
        log["op_forced"] = op

    # pending-step override, coding surfaces: the earliest unconsumed task
    # phrase acts regardless of the op read's ANSWER lean (the browser
    # v3.8.0 lesson — with work left, ANSWER is a misroute)
    pend = pending_code_step(task, hist, cands, cwd)
    if pend is None and len(lines) >= 2 and not any(_wrote_entry(e) for e in hist):
        # (len>=2: DECIDE needs a distinct set — a one-file ls has nothing
        # to disambiguate, so the turn passes through for the model)
        # a run-pronoun over pre-existing files ("run the script" with no
        # write history to resolve "the script" from): the element read
        # picks the file, same decide machinery as the browser's element
        # step; a low-confidence pick opens the route instead of forcing
        res = DECIDE_FN(ctx_code(task, hist, len(lines), "RUN",
                                 options=lines),
                        lines, method="d2", alpha=0.5, anchor=True)
        if res.selected in lines and res.confidence >= CODE_GATE["answer_gate"]:
            pick = res.selected.split()[-1]
            pend = pending_code_step(task, hist, cands, cwd, forced_target=pick)
            if pend is not None:
                log["element"] = {"selected": res.selected,
                                  "conf": res.confidence,
                                  "premask": res.premask_mass}
    if pend is not None:
        log["pending_override"] = {"phrase": pend["phrase"], "op": pend["op"],
                                   "target": pend.get("target")}
        log["routed"] = "pending-step"
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return pend["cell"], "tool_calls", log

    # nothing pending: ANSWER-forced turns pass through for the model's
    # report; a "repeat until it passes"-shaped task with a failure still
    # on the floor is NOT done — keep the model working instead
    if op == "ANSWER":
        bad = re.search(r"Traceback|AssertionError|FAIL", obs, re.I)
        wants_pass = re.search(r"until it passes|verify", task, re.I)
        if bad and wants_pass:
            log["routed"] = "passthrough-verify"
            log["verify_gate"] = bad.group(0)
        else:
            PASSTHROUGH_MAX_NEW = 128
            log["routed"] = "passthrough-answer"
        log["latency_ms"] = (time.perf_counter() - t0) * 1000
        log_decision({**log, "ts": time.time()})
        return None, "passthrough", log
    # non-answer route with nothing executable pending: the model authors
    # or observes (full generation budget — its edit IS the next step)
    log["routed"] = "passthrough"
    log["latency_ms"] = (time.perf_counter() - t0) * 1000
    log_decision({**log, "ts": time.time()})
    return None, "passthrough", log


# ---------------------------------------------------------------- passthrough

def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    tcs = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            j = json.loads(m.group(1))
            if isinstance(j, dict) and "name" in j:
                tcs.append(j)
        except ValueError:
            pass
    if not tcs:
        # XML-parameter form the tools template demonstrates:
        # <tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>...
        for m in re.finditer(r"<function=([A-Za-z0-9_]+)>(.*?)(?:</function>|<\|[^|>]*\|>|\Z)",
                             text, re.S):
            args = {}
            for pm in re.finditer(r"<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>",
                                  m.group(2), re.S):
                args[pm.group(1)] = pm.group(2)
            if args:
                tcs.append({"name": m.group(1), "arguments": args})
    content = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S)
    content = re.sub(r"<function=[^>]*>.*?(?:</function>|<\|[^|>]*\|>|\Z)", "",
                     content, flags=re.S)
    content = re.split(r"<\|[^|>]*\|>", content)[0].strip()
    if not tcs:  # bare-JSON fallback (small models drop the tags)
        m = re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}', content)
        if m:
            try:
                tcs.append(json.loads(m.group(0)))
                content = (content[:m.start()] + content[m.end():]).strip()
            except ValueError:
                pass
    # greedy decode can repeat one identical tool-call block until the token
    # cap; keep first occurrence of each (name, arguments) and bound the count
    seen, uniq = set(), []
    for t in tcs:
        key = (t.get("name"), json.dumps(t.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return content, uniq[:4]


HIST_KEEP_LINES = 2
HIST_KEEP_RESULTS = 1  # was 3: three kept snapshots were 15-25k tokens of
# stale page state re-prefilled by every model turn (21-46k-token prompts,
# 12-35s turns, three compactions per browser task); the newest result is
# the only page state anything can act on
USER_MSG_CAP = 100000        # chars; only messages past this get the elision
USER_MSG_KEEP_HEAD = 20000   # chars kept from the head (original task)
USER_MSG_KEEP_TAIL = 60000   # chars kept from the tail (recent state)

def slim_history(messages: list[dict]) -> list[dict]:
    """Passthrough turns re-prefill the whole conversation, and old aria
    snapshots dominate it — multi-minute prefill gaps on long tasks (T1/T5
    burned half their budget re-reading pages they had already left). Keep
    the last HIST_KEEP_RESULTS tool results verbatim (the model's working
    evidence: stubbing the SECOND result once flipped its early turns into
    close-and-redo junk, v3.7/v3.7.1 T4 n=2), stub every older oversized
    tool result down to its URL/Title header, and elide the middle of any
    single message over USER_MSG_CAP chars. The omp client still holds the
    full texts; this only shrinks what the model re-reads each turn."""
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep = set(tool_idxs[-HIST_KEEP_RESULTS:])
    out = []
    for i, m in enumerate(messages):
        if (m.get("role") == "tool" and i not in keep
                and isinstance(m.get("content"), str)
                and len(m["content"]) > 600):
            head = m["content"].splitlines()
            stub = "\n".join(head[:HIST_KEEP_LINES])
            if len(head) > HIST_KEEP_LINES:
                stub += "\n… [earlier page content trimmed]"
            m = {**m, "content": stub}
        elif isinstance(m.get("content"), str) and len(m["content"]) > USER_MSG_CAP:
            # (v3.11.8) a message over ~25k tokens — in practice the
            # harness's context-compaction request, which embeds the whole
            # transcript in one user turn — keeps its head (the original
            # task) and tail (recent pages + the instruction) while the
            # middle is elided. Pure length policy: no content is read, and
            # prompts under the cap are untouched. Beyond ~35k tokens of
            # context this GPU's whole generation path falls off a cliff
            # (prefill 8.5s at 35k vs 78s at 48k, decode 35-70ms/tok at
            # 20-44k standalone vs 4.7-9.4s/tok at 48k in-server), so the
            # elision is what keeps a compaction turn inside the deadline.
            c = m["content"]
            cut = len(c) - USER_MSG_KEEP_HEAD - USER_MSG_KEEP_TAIL
            m = {**m, "content": (c[:USER_MSG_KEEP_HEAD]
                                  + f"\n… [{cut} chars of middle history"
                                    " elided for length] …\n"
                                  + c[-USER_MSG_KEEP_TAIL:])}
        out.append(m)
    return out


def passthrough(req: dict, enable_thinking: bool | None,
                nudge: str | None = None) -> tuple[str, str | None, list[dict]]:
    msgs = slim_history(norm_messages(req["messages"]))
    if nudge:
        # role "user", not "system": templates across backends (HF, llama.cpp
        # conversions) accept a trailing user turn universally; a trailing
        # system turn is template-dependent
        msgs = msgs + [{"role": "user", "content": nudge}]
    tools = req.get("tools") or None
    text = LOADED.tokenizer.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=True,
        enable_thinking=enable_thinking, tokenize=False)
    ids = LOADED.tokenizer(text, return_tensors="pt", add_special_tokens=False,
                           ).input_ids.to(LOADED.model.device)
    global PASSTHROUGH_MAX_NEW
    max_new = PASSTHROUGH_MAX_NEW or (4096 if enable_thinking is not False
                                      else 1024)
    PASSTHROUGH_MAX_NEW = None  # one-shot: only the route that set it uses it
    # OpenAI semantics: a client that asks for a token budget gets it — the
    # harness sizes its compaction/summary calls, and our internal default
    # must not inflate what the client asked for (never inflate: only trim)
    req_max = req.get("max_tokens")
    if req_max:
        max_new = min(max_new, int(req_max))
    def _gen_once() -> torch.Tensor:
        # greedy decode loop, not GenerationMixin.generate: generate() builds
        # a fresh 4D attention mask every step, and at compaction-size
        # contexts (40k+) that per-step allocation thrashes a full caching
        # allocator — 70-126 SECONDS per token (a harness compaction never
        # returned). The plain forward+past loop below is 34-68 ms/token at
        # the same lengths, same weights, same cache.
        eos_ids = {LOADED.tokenizer.eos_token_id}
        gcfg_eos = getattr(LOADED.model.generation_config, "eos_token_id", None)
        if isinstance(gcfg_eos, int):
            eos_ids.add(gcfg_eos)
        elif gcfg_eos:
            eos_ids.update(gcfg_eos)
        eos_ids.discard(None)
        print(f"[gen] start prompt_toks={ids.shape[1]} max_new={max_new} "
              f"think={enable_thinking}", flush=True)
        _g0 = time.perf_counter()
        with torch.no_grad():
            out = LOADED.model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            print(f"[gen] prefill {ids.shape[1]} toks in "
                  f"{time.perf_counter() - _g0:.1f}s", flush=True)
            _d0 = time.perf_counter()
            nxt = out.logits[:, -1:].argmax(-1)
            gen_ids = [int(nxt)]
            # (v3.11.17) EOS floor: the 4B's first greedy token is EOS on
            # awkward prompts (the bare-EOS answers that motivated the
            # composed handoff) — hold EOS off for the first few tokens so a
            # report turn emits words before it may stop. A client max_tokens
            # below the floor still wins (the outer bound is checked first).
            # (v3.11.17) also removes a decode-loop duplication: the second
            # copy re-assigned gen_ids = [last_token] after the first loop
            # finished and exited immediately on that token (EOS), so every
            # passthrough generation returned a single EOS token — an empty
            # answer on every model turn of the decide311016 sweep, all five
            # tasks.
            while len(gen_ids) < max_new and (
                    len(gen_ids) < 4 or gen_ids[-1] not in eos_ids):
                out = LOADED.model(input_ids=nxt, past_key_values=past,
                                   use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                gen_ids.append(int(nxt))
                if len(gen_ids) % 8 == 0:
                    print(f"[gen] decode {len(gen_ids)} toks, "
                          f"{(time.perf_counter() - _d0) / len(gen_ids) * 1000:.0f}"
                          f"ms/tok", flush=True)
            seq = torch.cat([ids, torch.tensor([gen_ids], device=ids.device)],
                            dim=1)
        # phase split: tokenization+prefill vs decode tells a stall apart
        # (a compaction-size request should be prefill ~15s + decode
        # ~100ms/token; anything far off points at the stage to blame)
        print(f"[gen] done prompt_toks={ids.shape[1]} new_toks={len(gen_ids)} "
              f"max_new={max_new} wall={time.perf_counter() - _g0:.1f}s",
              flush=True)
        return seq

    # (v3.11.5) a CUDA failure mid-generation (OOM, or the WSL2 dxg fd leak —
    # torch's expandable-segments allocator pins a dxg file descriptor per
    # mapped segment until it hits the 1024-fd ceiling and everything starts
    # failing) must not waste a whole harness turn: clear the cache and
    # retry once. A re-prefill is seconds; a lost turn was minutes.
    try:
        seq = _gen_once()
    except RuntimeError as e:
        print(f"[gen] retry: {type(e).__name__}: {str(e)[:160]}", flush=True)
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as ce:
            print(f"[gen] cache clear failed: {type(ce).__name__}", flush=True)
        seq = _gen_once()
    gen = LOADED.tokenizer.decode(seq[0][ids.shape[1]:], skip_special_tokens=False)
    with open(LOGDIR / "last_response.txt", "w") as f:
        f.write("=== PROMPT TAIL ===\n" + text[-1500:] +
                "\n=== RAW GEN ===\n" + gen[:3000] + "\n")
    content, tcs = parse_tool_calls(gen)
    finish = "tool_calls" if tcs else "stop"
    return content, finish, tcs


REMOTE_CHAR_BUDGET = 100000  # chars across all forwarded messages; ~25k
# tokens against llama-server's 32768-token slots (-c 131072 -np 4), leaving
# room for the reply inside the slot.


def _msg_chars(m: dict) -> int:
    c = m.get("content")
    if isinstance(c, str):
        return len(c)
    return len(json.dumps(c)) if c is not None else 0


def fit_char_budget(messages: list[dict], budget: int) -> list[dict]:
    """(v3.14.1) Second-stage trim for the remote fallback: slim_history
    stubs old tool results, but a conversation can still exceed the
    backend's per-slot context (T1 died at 43.5k tokens against a 32k slot
    — the backend 400s, the proxy raises, the harness disposes the
    session). While the total is over budget, elide the middle of the
    LARGEST message except the newest (the current observation is the only
    page state the turn can act on); heads/tails keep enough to act on.
    Best effort: if only the newest message is big, send as-is."""
    msgs = [dict(m) for m in messages]
    if len(msgs) <= 1:
        return msgs
    while sum(_msg_chars(m) for m in msgs) > budget:
        i = max(range(len(msgs) - 1), key=lambda j: _msg_chars(msgs[j]))
        c = msgs[i].get("content")
        text = c if isinstance(c, str) else (json.dumps(c) if c is not None
                                             else "")
        if len(text) <= 1600:
            break  # nothing left worth cutting
        keep = max((budget - sum(_msg_chars(m) for m in msgs)) // 4, 1600)
        msgs[i]["content"] = (text[:keep]
                              + "\n… [middle of a long message trimmed for "
                                "the remote model] …\n"
                              + text[-keep:])
    return msgs


def remote_passthrough(req: dict):
    """(v3.13.0) The remote twin of passthrough(): forward the OpenAI-shaped
    request to the configured HTTP backend and map the reply onto the proxy's
    internal (content, finish, tool_calls) triple. Never touches LOADED —
    the backend arm runs with no in-process model on the GPU.

    (v3.14.1) the messages are slimmed exactly like the local passthrough
    (same proven policy) and then char-budgeted to the backend's per-slot
    window — verbatim forwarding 400'd T1's answer turn (16 messages with
    page snapshots ≈ 43k tokens vs the 32k slot) and the raise disposed
    the session with the task one click from done."""
    msgs = fit_char_budget(
        slim_history(list(req.get("messages", []))), REMOTE_CHAR_BUDGET)
    body = {"model": REMOTE["model"], "messages": msgs}
    for k in REMOTE_FWD_KEYS:
        if req.get(k) is not None:
            body[k] = req[k]
    body.update(REMOTE["extra_payload"])  # e.g. reasoning_budget_tokens: 0
    body.setdefault("max_tokens", 4096)  # loops must not run to the context end
    t0 = time.time()
    payload: dict | None = None
    err: Exception | None = None
    for attempt in (1, 2):  # one retry — a backend blip must not waste a turn
        try:
            http = urllib.request.Request(
                REMOTE["base_url"] + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(http, timeout=REMOTE["timeout_s"]) as r:
                payload = json.loads(r.read())
            break
        except TimeoutError as e:
            err = e
            break  # a wedged backend call must not be paid for twice
        except Exception as e:
            err = e
            if attempt == 1:
                time.sleep(2.0)
    if payload is None:
        raise err
    ch = (payload.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tcs = []
    for tc in (msg.get("tool_calls") or []):
        try:
            fn = tc.get("function") or {}
            if not fn.get("name"):
                continue
            argv = fn.get("arguments")
            if isinstance(argv, str):
                argv = json.loads(argv)
            tcs.append({"id": tc.get("id") or ("call_" + uuid.uuid4().hex[:24]),
                        "name": fn["name"],
                        "arguments": argv if isinstance(argv, dict) else {}})
        except Exception:
            continue
    finish = ch.get("finish_reason") or "stop"
    if finish == "length":
        finish = "stop"  # the harness reads finish=length as an unfinished turn
    content = msg.get("content")
    log_decision({"arm": "remote", "model": REMOTE["model"], "finish": finish,
                  "n_tool_calls": len(tcs),
                  "latency_ms": round((time.time() - t0) * 1000),
                  "head": (content or "")[:100] if not tcs else "",
                  "ts": time.time()})
    return (content, finish, tcs)


# ---------------------------------------------------------------- HTTP layer

def completion_payload(model: str, content: str | None, tool_calls: list[dict],
                       finish: str, prompt_toks: int, out_toks: int) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": t.get("id") or ("call_" + uuid.uuid4().hex[:24]),
             "type": "function",
             "function": {"name": t["name"],
                          "arguments": json.dumps(t.get("arguments", {}),
                                                  ensure_ascii=False)}}
            for t in tool_calls]
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_toks, "completion_tokens": out_toks,
                      "total_tokens": prompt_toks + out_toks}}


def chunk_base(model: str):
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}


def sse_chunks(model: str, content: str | None, tool_calls: list[dict],
               finish: str, role_first: bool = True):
    base = chunk_base(model)
    if role_first:
        yield base | {"choices": [{"index": 0, "delta": {"role": "assistant"},
                                   "finish_reason": None}]}
    if content:
        for i in range(0, len(content), 200):
            yield base | {"choices": [{"index": 0, "delta": {"content": content[i:i + 200]},
                                       "finish_reason": None}]}
    for k, t in enumerate(tool_calls):
        yield base | {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": k, "id": t.get("id") or ("call_" + uuid.uuid4().hex[:24]),
             "type": "function",
             "function": {"name": t["name"],
                          "arguments": json.dumps(t.get("arguments", {}),
                                                  ensure_ascii=False)}}]},
            "finish_reason": None}]}
    yield base | {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    yield "data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": name, "object": "model", "owned_by": "poc"}
                for name in MODEL_NAMES]})
        elif self.path == "/health":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model", "qwen35-4b-vanilla")
        if "/" in model and model.rsplit("/", 1)[-1] in MODEL_NAMES:
            # (v3.13.0) tolerate provider-prefixed wire names: poc-proxy/<arm>
            model = model.rsplit("/", 1)[-1]
        mode = MODEL_NAMES.get(model, "vanilla")
        want_stream = bool(req.get("stream"))
        with open(LOGDIR / "last_request.json", "w") as f:
            json.dump({"model": model, "stream": want_stream,
                       "n_messages": len(req.get("messages", [])),
                       "n_tools": len(req.get("tools") or []),
                       "keys": sorted(req.keys()),
                       "messages": req.get("messages", []),
                       "tools": req.get("tools", [])}, f, indent=1)
        def compute():
            if mode == "remote":
                return remote_passthrough(req)
            if mode == "decide":
                dom = detect_domain(req["messages"])
                if dom == "coding":
                    tool_call, finish, log = run_code_chain(req)
                elif dom == "browser":
                    tool_call, finish, log = run_chain(req)
                else:
                    if CHAIN_REMOTE:  # no local model: floor == remote chat
                        return FALLBACK_FN(req)
                    return run_universal(req)  # Tier-1 floor (v3.10.0)
                if isinstance(tool_call, dict) and "compose" in tool_call:
                    # (v3.11.9) the chain wrote the compaction handoff
                    # itself — ship it as content, no model call at all
                    return (tool_call["compose"], "stop", [])
                if tool_call is None:  # ANSWER / non-action turn
                    resp = FALLBACK_FN(req)
                    # (v3.11.11) a malformed or empty answer on the answer
                    # gate's model handoff is fatal — the harness disposes
                    # the session on unparseable turns (decide3110i/j). Ship
                    # the chain's stashed action instead; (v3.11.14) a
                    # stashed page-heading answer when the task's remaining
                    # clause is a report. Prose answers never match the
                    # malformed marker; well-formed tool calls are extracted
                    # before this check.
                    global ANSWER_GATE_FALLBACK, ANSWER_TEXT_FALLBACK, \
                        REVIEW_NOTE
                    fb, ANSWER_GATE_FALLBACK = ANSWER_GATE_FALLBACK, None
                    atext, ANSWER_TEXT_FALLBACK = ANSWER_TEXT_FALLBACK, None
                    c, fin, _t = resp
                    if fin != "tool_calls" and (
                            not (c or "").strip()
                            or MALFORMED_ANSWER_RE.search(c)):
                        if atext is not None:
                            log_decision({"arm": "decide",
                                          "routed": "junk-fallback-answer",
                                          "junk": (c or "")[:120],
                                          "answer": atext[:80],
                                          "ts": time.time()})
                            return (atext, "stop", [])
                        if fb is not None:
                            log_decision({"arm": "decide",
                                          "routed": "junk-fallback",
                                          "junk": (c or "")[:120],
                                          "ts": time.time()})
                            return (None, "tool_calls", [fb])
                    # (v3.16.2) annotate mode: the review's verdict rides
                    # the model's own prose answer as a caveat — the report
                    # is never blocked, edited, or replaced.
                    note, REVIEW_NOTE = REVIEW_NOTE, None
                    if (note and fin != "tool_calls" and (c or "").strip()
                            and not MALFORMED_ANSWER_RE.search(c)):
                        resp = (c.rstrip() + "\n\n" + note, fin, _t)
                    return resp
                return (None, finish, [tool_call])
            if mode == "vanilla-nothink":
                return passthrough(req, enable_thinking=False)
            return passthrough(req, enable_thinking=None)

        if not want_stream:
            try:
                with LOCK:
                    content, finish, tcs = compute()
            except Exception as e:  # surface the error, keep the loop alive
                log_decision({"arm": mode, "error": repr(e)[:500], "ts": time.time()})
                self._json({"error": {"message": str(e)[:400], "type": "proxy_error"}}, 500)
                return
            self._json(completion_payload(model, content, tcs, finish, 0, 0))
            return

        # omp aborts a stream that stays silent >2.5s waiting for its first
        # event, so open the stream before any compute and hold it open with
        # empty-delta keep-alives while the worker generates (the reference
        # conv path can take minutes on a browser-tool prompt).
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        gone: list = []

        def send_data(data: str):
            if gone:
                return
            try:
                payload = (data + "\n\n").encode()
                self.wfile.write(hex(len(payload))[2:].encode() + b"\r\n" + payload + b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                gone.append(True)

        send_data("data: " + json.dumps(chunk_base(model) | {"choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""},
             "finish_reason": None}]}))

        box: dict = {}

        def work():
            try:
                with LOCK:
                    box["resp"] = compute()
            except Exception as e:
                box["err"] = e

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=1.0)
            if worker.is_alive():
                send_data("data: " + json.dumps(chunk_base(model) | {"choices": [
                    {"index": 0, "delta": {}, "finish_reason": None}]}))

        if "err" in box:
            log_decision({"arm": mode, "error": repr(box["err"])[:500], "ts": time.time()})
            send_data("data: " + json.dumps({"error": {"message": str(box["err"])[:400],
                                                       "type": "proxy_error"}}))
            send_data("data: [DONE]\n\n")
            return

        content, finish, tcs = box["resp"]
        for chunk in sse_chunks(model, content, tcs, finish, role_first=False):
            send_data(chunk if isinstance(chunk, str) else "data: " + json.dumps(chunk))
        try:
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--adapter-config", default=str(ROOT / "eval/adapter_learning.json"),
                    help="learned-adapter plumbing config (missing file = "
                         "everything disabled; hot-reloaded on mtime change)")
    ap.add_argument("--backend-config", default=str(ROOT / "eval/proxy_backend.json"),
                    help="remote backend config (v3.13.0; missing file or "
                         "enabled:false = no remote arm; other arms unchanged)")
    ap.add_argument("--no-local-model", action="store_true",
                    help="skip load_model(): serve only the remote backend arm "
                         "(this GPU cannot hold the 4B and llama-server at once)")
    args = ap.parse_args()
    global LOADED, REMOTE, DECIDE_FN, VALUE_FN, FALLBACK_FN, CHAIN_REMOTE, \
        ANSWER_GATE
    if learning is not None:
        lcfg = learning.init(args.adapter_config)
        learning.set_log_file(DECISIONS)
        print(f"learning: enabled={lcfg.get('enabled')} "
              f"capture={lcfg.get('capture')} "
              f"route={lcfg.get('route_to_learned')} "
              f"config={args.adapter_config}", flush=True)
    REMOTE = load_remote(args.backend_config)
    if REMOTE:
        MODEL_NAMES[REMOTE["wire_model"]] = "remote"
        print(f"remote backend: poc-proxy/{REMOTE['wire_model']} -> "
              f"{REMOTE['base_url']} (model {REMOTE['model']}, "
              f"extra={REMOTE['extra_payload']})", flush=True)
        # (v3.16.0) optional completion self-review rides the backend
        # config; absent/disabled = None = byte-for-byte v3.15.0 behavior
        global COMPLETION_REVIEW
        COMPLETION_REVIEW = REMOTE.get("completion_review")
        if COMPLETION_REVIEW:
            print(f"completion-review: enabled mode="
                  f"{COMPLETION_REVIEW['mode']} "
                  f"p_done_threshold={COMPLETION_REVIEW['p_done_threshold']}",
                  flush=True)
    if REMOTE and REMOTE.get("chain"):
        # (v3.14.0) the decide arm travels with the backend: reads over
        # llama-server HTTP (eval/chain_http.py), fallbacks to the remote
        # chat, gates from the HTTP-fitted cal split (webgate_remote.json)
        try:
            import chain_http
            _base, _mdl = REMOTE["base_url"], REMOTE["model"]
            _w = int(REMOTE.get("chain_workers", 4))
            _tmo = int(REMOTE.get("timeout_s", 120))

            def _decide_remote(question, candidates, **kw):
                try:
                    res = chain_http.decide_http(
                        _base, _mdl, question, candidates,
                        alpha=kw.get("alpha", 1.0),
                        anchor=kw.get("anchor", True),
                        timeout=_tmo, workers=_w)
                except Exception as e:
                    log_decision({"arm": "decide-remote",
                                  "read_error": repr(e)[:200],
                                  "ts": time.time()})
                    res = None
                if res is None:  # degrade on the chain's low-confidence path
                    res = DecisionResult(
                        selected=candidates[0], confidence=0.0,
                        probabilities={c: 1.0 / len(candidates)
                                       for c in candidates},
                        scoring_method="http_read_failed")
                return res

            def _value_remote(task, hist, op, sel_line, n_cand):
                ask = ("What exact text should be typed into it? Reply with "
                       "the text and nothing else." if op == "TYPE" else
                       "Which option should be selected? Reply with the "
                       "option's value and nothing else.")
                content = ctx(task, hist, n_cand, op) + \
                    f"\n\nChosen element: {sel_line}\n\n{ask}"
                return chain_http.think_value_http(_base, _mdl, content,
                                                   timeout=_tmo)

            def _fallback_remote(req, nudge=False):
                return remote_passthrough(req)

            (DECIDE_FN, VALUE_FN, FALLBACK_FN,
             CHAIN_REMOTE) = (_decide_remote, _value_remote,
                              _fallback_remote, True)
            gp = ROOT / REMOTE.get("chain_gates",
                                   "configs/webgate_remote.json")
            _gates = json.load(open(gp))
            GATE.update({"accept_conf": _gates["accept_conf"],
                         "temperature": _gates["temperature"]})
            ANSWER_GATE = float(_gates.get("answer_gate", 0.5))
            CODE_GATE.update({"accept_conf": _gates["accept_conf"],
                              "temperature": _gates["temperature"],
                              "answer_gate": ANSWER_GATE})
            MODEL_NAMES[REMOTE.get("chain_model", "gemma-12b-decide")] = "decide"
            print(f"chain-remote: decide arm rides {REMOTE['wire_model']} "
                  f"({REMOTE['base_url']}), gates={_gates['accept_conf']}/"
                  f"{_gates['temperature']}/{ANSWER_GATE} from {gp.name}, "
                  f"wire model poc-proxy/"
                  f"{REMOTE.get('chain_model', 'gemma-12b-decide')}",
                  flush=True)
        except Exception as e:
            print(f"chain-remote unavailable (local decide only): "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
    if args.no_local_model:
        print("no local model loaded (--no-local-model): remote arm only; "
              "in-process arms will fail if invoked", flush=True)
    else:
        LOADED = load_model()
        # warmup: a cold process's first CUDA mappings and the triton kernels'
        # JIT are both fragile-slow; burn them on a tiny forward at startup,
        # where a failure costs nothing and gets retried by whoever starts us
        try:
            _w = LOADED.tokenizer("Hi", return_tensors="pt").input_ids.to(
                LOADED.model.device)
            with torch.no_grad():
                LOADED.model(input_ids=_w, use_cache=True)
            print("warmup ok", flush=True)
        except Exception as e:
            print(f"warmup failed (continuing): {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
    print("remote backend arm; serving on" if args.no_local_model
          else "model loaded; serving on", args.port,
          "(chain v3.14.3: stand down when every task phrase is consumed "
          "and the pick re-clicks a name some earlier cell already clicked "
          "— the v3.14.2 sweep proved the T1 fix end to end (tagged 'In Her "
          "Wake' click, clean stand-down, correct answer) and showed T3's "
          "one chain blemish: with the task done, the model's op read "
          "re-clicked 'Login' after the tagged logout; the logout redirect "
          "had shifted the button's ref (e21->e23) so the ref-based repeat "
          "check passed, the emptied form flashed 'Your username is "
          "invalid!', and the model's report quoted that flash instead of "
          "the two success messages it had already seen (the same stray "
          "click sat unnoticed inside the passing v3.14.0 run — its report "
          "just happened to shrug it off). A ref shift must not resurrect a "
          "finished step: when pending_phrases is empty and the chosen "
          "CLICK's name appears anywhere in click history, the turn passes "
          "through untainted; pagination re-clicks are untouched because "
          "they only run while a next-page phrase is still pending; T2/T5 "
          "remain model report variance (v3.14.0 read the same snapshots "
          "right); below, v3.14.2: consume the ordinal phrase a pend click executed — "
          "the re-test showed the three v3.14.1 fixes hold (no context 400, "
          "the truncated-title click lands, no anchor yank onto the pager) "
          "but T1 then wandered: after the ordinal snap executed 'open the "
          "first book's detail page' as CLICK 'In Her Wake', the phrase "
          "stayed unconsumed (the name shares no vocabulary with it), so on "
          "the detail page the chain re-resolved it against the product "
          "pager's own listing groups and clicked prev/next products until "
          "the model answered with the wrong book. A click that emits the "
          "pend step's positional resolution now tags its cell marker with "
          "the phrase ('// click … [task: open the first book's detail "
          "page]'), extract_history carries the tag, and pending_step/"
          "pending_phrases consume a phrase by exact tag match — the chain "
          "then stands down into the answer gate, whose report-clause "
          "fallback hands the page's own heading (the book title) to the "
          "model; the pending_override log's 'named' flag also now means "
          "what it says (not positional) instead of 'a pend tuple existed'; "
          "below, v3.14.1: the three T1 failure causes, each fixed at its own "
          "layer — (1) remote fallback overflow: remote_passthrough now "
          "slims the forwarded messages exactly like the local passthrough "
          "(same proven slim_history policy) and then fits them to a "
          "100k-char budget (~25k tokens inside llama-server's 32768-token "
          "slots) by middle-eliding the largest non-newest message; T1's "
          "answer turn forwarded 16 messages ≈ 43.5k tokens against a 32k "
          "slot, the backend 400'd, and the raise disposed the session one "
          "click from done; (2) page-side click matcher: the matcher read "
          "only textContent/value, but catalogue cards carry the accessible "
          "name in the title attribute (title link, DOM text truncated with "
          "literal dots) and the img alt (image link, empty textContent) — "
          "matching now tries every string the element offers, dot-truncated "
          "strings compare as prefixes, and the returned name still prefers "
          "DOM text; (3) double-next misroute: the exact-phrase anchor (the "
          "'Form Authentication' vs 'Basic Auth' guard) fires when a "
          "candidate's name appears verbatim in the task, and 'next' sits "
          "inside 'go to the next page of the catalogue' — after the element "
          "read had correctly picked the ordinal-snap target (T1 page 2: "
          "'In Her Wake', conf 0.52) the anchor yanked the emitted click "
          "back onto the pager next@e516 (page 3), the read then picked "
          "page 3's own first book and the session died on fixes 1+2 before "
          "it could recover; anchor_index is now barred from yanking the "
          "read's pick when that pick IS the pending step's resolved target "
          "(named or positional) — the same protection the pend-override "
          "path already had, extended to the non-forced path; v3.16.2: "
          "coverage and posture fixes from the 2026-09-22 live test — "
          "(1) the 3-frame review reads move into review_frames_run and "
          "gain a SECOND, log-only trigger site: the gate-passed answer "
          "stand-downs (all-consumed-name-repeat, passthrough-repeat) and "
          "the no-candidates answer exit, all tagged in review_read.site — "
          "the live run's wrong login report slipped through exactly these "
          "uncovered paths; (2) mode annotate joins log and gate: the "
          "review never blocks — when all three frames read NOT-done (the "
          "measured ~89%-precision point of the vote dial) a generic "
          "verification caveat is appended to the model's own prose answer, "
          "so an agent framework or user can see when a completion report "
          "is uncorroborated; the challenge stays fallthrough-only and "
          "gate-only, still default OFF, and the direction-split audit "
          "warned that topline false-alarm rates flatter the gate — on "
          "correct success claims the min rule would challenge ~93% "
          "(bench-graded), hence annotate's strict operating point; "
          "v3.16.1: "
          "the completion review's single DONE/NOT-DONE read becomes the "
          "validated 3-frame MINIMUM (single-model framing backtest on the "
          "596 ground-truth probes: GOAL — task text + newest snapshot; "
          "DELTA — session-first snapshot vs newest, change evidence; COND "
          "— the model's own last assertion claim-checked and the verdict "
          "fed to the done read; the framings disagree with each other by "
          "0.18-0.28 of probability mass, the same model decorrelating "
          "from itself) — min-of-three caught 78% of wrong success claims "
          "at 17% false alarms vs the single frame's 67%/18%, and on chain "
          "sessions the goal frame alone moves 66%/34% to 78%/21%; the "
          "gate still requires a concrete alternative action and still "
          "defaults OFF behind the same completion_review config block — "
          "same model, no second opinion, ~2s of extra reading per flagged "
          "stand-down; v3.16.0: "
          "the completion self-review becomes a CONFIG OPTION, the "
          "learned-adapter pattern (backend config completion_review "
          "block; missing or enabled:false = byte-for-byte v3.15.0) — "
          "mode log records the independent completion read's verdict "
          "(p_done) on the review-flagged stand-down row; mode gate "
          "additionally challenges an UNCORROBORATED stand-down when the "
          "read confidently says NOT done and the chain holds a concrete "
          "alternative action — the fallback cell ships and the turn "
          "keeps working. Default OFF, deliberately: the objective-label "
          "experiment measured 77% detection on false success reports "
          "against 32% false alarms on correct ones, so the gate is "
          "offered, never assumed; retirement decisions recorded the same "
          "day — MiniWoB demoted to smoke test, the learned-adapter loop "
          "frozen (config-gated, unproven), the universal tier dormant, "
          "the local 4B generation stack queued for retirement when the "
          "standing proxy moves to a served backend; v3.15.0: "
          "completion-hygiene pass, all routing-never-blocking and "
          "standards-only (overfit audit: no benchmark reward, task-name or "
          "site vocabulary anywhere; every change can fail honestly in both "
          "directions) — (1) submit precondition in the CLICK cell, pure "
          "native form semantics: when the click would submit a form the "
          "browser itself would reject (willValidate && !checkValidity, "
          "novalidate/formNovalidate honored, non-visible controls ignored) "
          "the cell throws 'submit-precondition: <fields>' BEFORE clicking, "
          "so the error lands in the tool result and the model fills the "
          "fields instead of eating a terminal wrong submit (mechanism 1 of "
          "the MiniWoB autopsy: 9 sessions submit-first); (2) ARIA "
          "interactive roles added to the click matcher (tab, menuitem, "
          "link, option, treeitem, switch) — same class as the v3.14.4 "
          "checkbox fix, driven by the ARIA contract, not by any benchmark; "
          "(3) the task ledger explicit in decisions.jsonl (pending_phrases "
          "state per decide turn — observability; enforcement was already "
          "the v3.8.0 pending override); (4) log-only completion-review "
          "flags ('completion-unverified:answer-gate-fallthrough' on the "
          "uncorroborated stand-down, 'acted-under-answer' on a fire under "
          "an ANSWER op read) — the objective-ground-truth experiment "
          "measured the self-read brake at 77% detection / 32% false "
          "alarms, so these are review signals for the harness's objective "
          "channel, never a gate; (5) read-batching audit: candidate "
          "scoring was already client-side thread-pooled over "
          "grammar-forced generations (eval/chain_http.py), and the 2.1x "
          "wall overhead is stand-down duration (item 1/3 territory), not "
          "read latency; (6) CLICK name case: the wanted element name is "
          "now lowercased like every candidate string — a model-emitted "
          "page-cased name ('Tab #1', 'Close', 'Submit') could otherwise "
          "never match the DOM's lowercase text (22+ click-miss errors in "
          "the transcripts); below, "
          "v3.14.0: decide arm over llama-server HTTP — with the "
          "backend config's chain:true the decide arm's reads travel to the "
          "llama-server backend (eval/chain_http.py): the d2 element/op reads "
          "run as grammar-forced generations whose per-token logprobs are the "
          "model's RAW distribution (verified: a forced token reads its true "
          "logprob, e.g. '[' at -24.24 while the true top-1 thinking marker "
          "reads -0.00 — masked logprobs would read ~0), summed as the "
          "teacher-forced sequence logprob with the same alpha normalization "
          "and DecisionResult the in-process scorer uses; the read position "
          "is pinned by the closed-channel prefill "
          "chat_template_kwargs enable_thinking:false (verified: op words "
          "land on the content channel, CLICK -0.45 vs SELECT -11.36 — "
          "without it the position is thinking-channel-dominated), while "
          "generation paths keep reasoning_budget_tokens:0 (the two knobs "
          "must not be combined); forced reads run on a thread pool against "
          "llama-server's parallel slots; gate constants come from "
          "configs/webgate_remote.json fitted by eval/calibrate_chain_http.py "
          "on the webreplay_v1 CAL SPLIT ONLY (the documented webgate fitter "
          "— ECE temperature grid + coverage/accuracy accept threshold — "
          "with HTTP reads, so the constants describe the backend that runs "
          "them); a failed HTTP read degrades to the chain's own "
          "low-confidence path (hand the turn to the model), never a 500; "
          "the wire model poc-proxy/<chain_model, default gemma-12b-decide> "
          "dispatches to the full run_chain machinery (read window, pending "
          "step, ordinal snap, compaction handoff, answer gates) with the "
          "model fallback riding the v3.13.0 remote arm; below, v3.13.0: "
          "remote backend arm — the wire model named in "
          "eval/proxy_backend.json (--backend-config) forwards the OpenAI "
          "request to the llama-server backend verbatim (messages, tools, "
          "tool_choice, temperature/top_p/max_tokens) plus "
          "reasoning_budget_tokens: 0 — the thought channel force-closed at "
          "token one, the only reliable no-think knob on this build — and the "
          "reply is mapped into the proxy's internal (content, finish, "
          "tool_calls) triple (finish length->stop, server tool_calls decoded "
          "to internal shape), sharing the streaming keep-alives, the 500 "
          "error path and the decisions.jsonl audit line with the in-process "
          "arms; missing config file or enabled:false = arm absent, "
          "byte-for-byte v3.12.0 dispatch; --no-local-model skips load_model "
          "— this 16GB GPU holds llama-server's ~8.5GB or the 4B's ~10GB, not "
          "both, so the backend-arm process carries no weights at all; the "
          "element-read d2 scorer does not travel over this HTTP surface "
          "(prompt-token logprobs are gone in llama.cpp v0.4.x), so the "
          "decide arm on the remote backend stays the documented follow-up); "
          "below, v3.12.0: learned-adapter plumbing — the adapter-building loop from the design discussion, config-gated end to end (eval/adapter_learning.json, --adapter-config; missing file or enabled:false = byte-for-byte v3.11.17 behavior; hot-reloaded on mtime change, no restart to flip): (1) CAPTURE — every Tier-1 floor turn now passes through learning.floor_hook before the floor serves it; when capture is on, the new messages of each unrecognized-tool-family conversation are appended to a per-session trajectory file (eval/reports/adapter_capture/<family>/sessions/, family = the request's exact declared tool-name set, sha1 fingerprint; text kept, images dropped, long content truncated, per-session byte cap); (2) BUILD — offline (adapter_builder.py, never in a request path): the captured trajectories are offered to a CONFIGURABLE builder model (any OpenAI-compatible endpoint: base_url/model/api_key/api_key_env/temperature/max_tokens in the config's builder block — the served 4B works as the default endpoint but a stronger model is the intended setup) which fills a constrained JSON SPEC: phrase -> tool-action mapping with argument captures, stand-down vocabulary, repeat limits; (3) VALIDATE — the gate is not skippable: the candidate spec is replayed against the real captured turns with one interpreter state threaded through each session (it must reproduce the captured action tool on >=80% of action turns, NEVER fire on a turn where the session reported or stood down, survive malformed input) plus an anti-overfit lint (no URLs, no task nouns in the spec — protocol vocabulary only, config-extensible allowlist); (4) REGISTER — only a passing spec lands in eval/learned_adapters/<family>/adapter_spec.json + manifest (auto_register off = manual --promote); (5) ROUTE — floor_hook consults the registry by fingerprint (mtime-cached hot-load) before the floor: the spec interpreter emits the family's own tool call (OpenAI wire shape, logged arm=decide-learned with routed=adapter-cell), or stands down (routed=adapter-standdown -> the model answers, floor takes over); observation gate (actions fire on tool results or a fresh conversation only), per-phrase max_uses, identical-repeat loop guard, per-session state; THE SAFETY POSTURE: a learned adapter is DATA not code — the builder model writes only a JSON spec, the interpreter is fixed reviewed code, nothing a model writes is ever exec'd on this host, and floor_hook never raises (any plumbing error logs learning-error and the floor serves the turn as before); v3.11.17: decode-loop duplication — the greedy loop ran twice, and the second copy re-assigned gen_ids to just the LAST token the first loop produced (always the EOS token) and exited on it immediately: every passthrough generation returned a single EOS — an empty answer on every model turn of the decide311016 sweep, all five tasks (the chain executed every task action correctly — verified end states on T1/T2/T3/T5 — but every final report was empty, and the v3.11.14 heading fallback shipped page headings on the gate-routed turns while T4/T5 died with no answer at all). One loop; plus a 4-token EOS floor: the 4B's first greedy token is EOS on awkward prompts (the bare-EOS answers that motivated the composed handoff), so EOS is held off for the first few tokens — a client max_tokens below the floor still wins; v3.11.16: window-build race — the pending-step target and the task-named extras both displace into the same read-window tail slot, and the extras rebuild ran second, slicing the resolved target straight back out: on books.toscrape the word 'next' (>=4 chars, in 'go to the next page') is itself task-named, so on every catalogue turn the pager link retook the slot the 'first book' ordinal-snap had just claimed — the decision log showed the correct resolution ('In Her Wake') and the wrong click (next@e516) on the same line, twice (decide3110n looked fixed by v3.11.15 and still page-flipped, decide3110p the same). The extras window is now built first and the pending target splices in ahead of it — the pending step is this turn's chosen action, it wins the slot; the superlative same-name collapse is also barred from re-picking a positionally-resolved pend target (it served the element read's own selection, pi None); v3.11.15: snapshot parse fix — the aria snapshot single-quotes any element whose accessible name carries YAML-special characters (': ' or '#', i.e. most real product titles) and the role regex read past it, silently DROPPING those elements from the candidates: on catalogue pages every card with such a title vanished, 'Add to basket' buttons collapsed into same-name runs, and the v3.11.14 ordinal-snap — now resolving over the full candidate list as designed — snapped 'the first book's detail page' onto whatever junk run came first (decide3110n: the pager's next link, page-flipping to the deadline); the quote is skipped and the element parses; the pending-step resolution also logs its candidate/group counts so any further mis-snap is legible from the decision log alone; v3.11.14: the pending step resolves over the FULL candidate list, before the read window is cut — on long catalogue pages the first 20 elements are page chrome (banner, search, category sidebar, sort, pager) and the listing items sit beyond it, so the ordinal-snap resolver saw no item groups and the answer gate stood down into the model-junk loop on every turn after the first compaction (decide3110m: the composed handoff worked, the post-compaction 'next' click worked, then 20+ answer-gate/junk-fallback cycles burned the deadline clicking the pager, the sort dropdown and the site logo); a pending step's target — named or positional — is now displaced into the read window like a task-named candidate, and when the gate falls through with only a report clause left, the chain stashes the page's own main heading as the answer instead of a speculative click; production captures added: every observation lands in last_obs.txt, the compaction request in compaction_request.txt; v3.11.13: accumulated history — the composed handoff could only record actions the compaction request happened to retain, so 1-2 actions were lost per compaction and progress decayed to zero under the harness's compaction cadence (decide3110l wandered the home page for its whole deadline). The proxy sees every turn: run_chain now union-merges each turn's actions into a session-wide list and composes the handoff from it, so no progress is ever lost; ref-sensitive checks stay on the per-request lines (accumulated lines carry refs from older page snapshots); v3.11.12: handoff progress recovery — a post-compaction context retains almost no action cells, so phrase tracking restarted at the task's first phrase and the harness's rapid compaction cadence looped the chain on steps 1-3 until the deadline (decide3110k: eleven compactions, the browser never left the home page). The redrive's handoff document lists executed actions in exact hist format; run_chain now merges them back into hist, making progress self-sustaining across compactions; v3.11.11: junk-fallback — a malformed or empty model answer on the answer gate's handoff no longer disposes the session: the chain stashes its own computed action at the handoff and compute() ships it when the model echoes JS template fragments or returns nothing — both decide3110i and decide3110j died here, after the composed handoff and task recovery had worked; v3.11.10: task recovery — the harness's compaction redrive replaces the task message with 'Context replaced. The <handoff> …' — reads on handoff vocabulary routed ANSWER and the capped model turn turned template-echo junk (decide3110i disposed 28s in, right after the composed handoff worked). extract_task now recovers the goal from the handoff document's Goal section, and the browser chain falls back to the last task it saw; protocol vocabulary only, no task-specific strings; v3.11.9: composed handoff — the browser chain answers the harness's compaction request with a handoff document written from its own tracked state (task verbatim, actions taken, current page, pending task phrases, deliverable), not with a model generation: the 4B model answered real compaction prompts with a bare EOS or code-fragment junk (an empty turn disposed a T1 session with 8 minutes of budget left); gated on harness-protocol vocabulary in the newest message, all other prose still passes through; v3.11.8: middle elision — any single message over 100k chars (the harness's compaction request embeds the whole transcript as one user turn) keeps its head and tail, ~20k+60k chars, and the middle is elided with a marker: pure length policy, prompts under the cap untouched — this GPU's generation path falls off a cliff past ~35k context tokens (prefill 8.5s at 35k vs 78s at 48k; decode 35-70ms/tok at 20-44k standalone vs 4.7-9.4s/tok at 48k in-server), and the elision is what keeps a compaction turn inside the task deadline; v3.11.7: prefill by measurement — single-call prefill everywhere, no allocator override: 35k prefill measures 8.5s on a clean GPU; the chunked variant was tried and reverted (threading the hybrid cache across chunks drops sdpa off the flash path — a 66k chunked prefill ballooned resident memory and never returned), and torch's expandable_segments allocator was dropped after it destabilised on this WSL dxg driver (per-segment dxg fd leak to the 1024-fd ceiling, then handles_ asserts); the server starts with a 65535 fd ceiling and a tiny startup warmup, and a mid-generation CUDA failure retries once after a cache clear instead of wasting the harness turn; v3.11.4: one live page — HIST_KEEP_RESULTS 3→1, only the newest tool result stays verbatim: the last three kept 15-25k tokens of stale snapshots re-prefilled by every model turn (12-35s turns, three compactions per task); v3.11.3: plain greedy decode — passthrough generation is a forward+past_key_values loop instead of GenerationMixin.generate, whose per-step 4D mask rebuild thrashed the full CUDA allocator at compaction-size contexts: 70-126 SECONDS per token (a harness compaction never returned within the deadline); the loop measures 34-68 ms/token at the same lengths; v3.11.2: generation budget — passthrough honours the client's max_tokens (OpenAI semantics, never inflate) and the compaction-prose route caps its own summary at 1024 new tokens; model runtime loads with sdpa attention + fused causal_conv1d/fla kernels (prefill at 20-34k tokens: minutes -> seconds); below that, v3.11.1: prose guard — the chain acts only on observation turns, newest message a tool result; the harness's context-compaction/handoff prose passes through logged instead of being answered with browser cells (the T1 compaction-jam loop); v3.11.0: ordinal snap — an unnamed positional target ('open the first book's detail page') resolves by page structure, the Nth adjacent same-name listing item, instead of standing down; pages without item structure keep the old skip; below that, v3.10.0: universal tier — routing picks browser chain, coding adapter, or the Tier-1 floor; the floor audit-logs every non-adapter turn and brakes model thrashing (LOOP_STREAK identical tool calls in a row -> corrective nudge + token cap, never a block); below that, v3.9.0: domain-general core + per-domain adapters — structural domain routing (which tools the conversation has called; never a model read) picks the browser chain or the coding adapter; coding adapter: ls candidates + write/bash cells + create/run/chmod/fix phrase vocabulary over file targets, template-class content only (task-quoted echo strings, echo/sum script templates) with everything beyond templates passed through for the model to author; browser chain: v3.8.1 pending-step answer-route override with comma-stopped phrase windows, one-TYPE-one-phrase, SELECT-consumption and superlative snap; earlier: v3.8.0 pending override, v3.7.5 SELECT switch-override, v3.7.4 click matcher, guarded anchor, combobox flip, page-side click, observe recovery, TYPE ordinal fallback, repeat escalation)", flush=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
