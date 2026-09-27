#!/usr/bin/env python3
"""(v3.18.0) Entry-R mechanical brake — the frozen stuck-loop terminator.

Registration: IMPLEMENTATION_PLAN.md, entry R (2026-09-28, before any v8
data). The brake arm serves the guarded chain MINUS the dial; at every
request, BEFORE any chain compute, the acting-turn history on the wire is
tested against the entry-P loop rule, verbatim:

  acting turn : an assistant tool call whose `code` argument carries an
                intent comment matching
                ^[ \t]*//[ \t]*(click|type|select|check)\b  (first such
                line, multiline). sig = that line, whitespace-collapsed,
                otherwise verbatim — element refs included.
  snapshot    : ckpt_rubric.build_evidence on the wire (the entry-N
                byte-parity construction): the newest NON-BLANK tool
                result's text, whitespace-compacted, 8000-char cap.
  FIRE        : within the trailing window of 8 acting turns (inclusive
                of the most recent acting turn) some sig occurs >= 3
                times AND the snapshot is identical (exact string) at all
                of those occurrences AND at the current request (the
                newest non-blank tool result at the end of the messages —
                the state the pending action would land on).

On FIRE the request never reaches the model: the frozen brake note ships
as a prose-only completion (no tool calls), which ends the episode. The
sequence is reconstructed from the request's messages at every request —
no server-side session state; harness compaction can only drop old
occurrences, which the identical-snapshot requirement then fails closed.
Any internal failure logs brake-error and serves on (the dial's disclosed
fail-open posture): what remains impossible is the SILENT fire.

Stdlib only. The proxy imports this defensively: absence or breakage here
must never touch serving (the brake is then simply absent).
"""
import hashlib
import json
import re
import time

ACTING_RE = re.compile(
    r"^[ \t]*//[ \t]*(click|type|select|check)\b[^\n]*$", re.M)
WINDOW = 8          # trailing acting turns, inclusive (entry-P WINDOW)
MIN_REPEATS = 3     # entry-P threshold
SNAP_CAP = 8000     # ckpt_rubric.EVIDENCE_CAP / dial_enforce.EVIDENCE_CAP

BRAKE_NOTE = ("BRAKE (mechanical): the identical action has been attempted "
              "3+ times on an unchanged page. The episode is terminated. "
              "Report your final answer now.")

CFG: dict | None = None      # set by init(); None = feature absent


def init(block) -> str:
    """Normalize the backend config's brake block. Missing/disabled = the
    feature is absent (byte-for-byte prior serving). Returns a status
    string for the startup banner."""
    global CFG
    if not isinstance(block, dict) or not block.get("enabled"):
        CFG = None
        return "disabled (brake block absent or enabled:false)"
    CFG = {"note": str(block.get("note", BRAKE_NOTE))}
    return (f"enabled window={WINDOW} min_repeats={MIN_REPEATS} "
            f"snap_cap={SNAP_CAP}")


def _code_of(tool_call) -> str:
    """The eval cell's code from one wire tool call. arguments is a JSON
    string on the OpenAI wire ({"language":...,"title":...,"code":...})."""
    fn = tool_call.get("function") or {}
    args = fn.get("arguments", "")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args
    if isinstance(args, dict):
        return str(args.get("code", ""))
    return ""


def _acting_sequence(messages):
    """((sig, snapshot) per acting turn on the wire, in order; the current
    request's snapshot). An acting turn's snapshot is the newest non-blank
    tool result BEFORE it — exactly build_evidence at its transcript row.
    The trailing value after the walk is the current request's snapshot."""
    seq = []
    cur = ""
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            t = m.get("content")
            if isinstance(t, list):
                t = "\n".join(str(b.get("text", "")) for b in t
                              if isinstance(b, dict)
                              and b.get("type") == "text")
            t = t if isinstance(t, str) else ""
            if t.strip():
                cur = " ".join(t.split())[:SNAP_CAP]
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                mm = ACTING_RE.search(_code_of(tc))
                if mm:
                    seq.append((" ".join(mm.group(0).split()), cur))
    return seq, cur


def check(messages, log_decision, session_key=None):
    """None = keep serving; (note, "stop", []) = brake fires. Never
    raises: any internal failure is logged as brake-error and the request
    serves on. session_key rides the log only (the rule is stateless)."""
    if CFG is None:
        return None
    try:
        seq, cur = _acting_sequence(messages)
        window = seq[-WINDOW:]
        by_sig = {}
        for i, (sig, _snap) in enumerate(window):
            by_sig.setdefault(sig, []).append(i)
        fired = None
        for sig, idxs in by_sig.items():
            if len(idxs) < MIN_REPEATS:
                continue
            snaps = {window[i][1] for i in idxs}
            if cur and len(snaps) == 1 and snaps.pop() == cur:
                fired = (sig, len(idxs))
                break
        if fired is None:
            return None
        log_decision({"arm": "brake", "routed": "fire",
                      "sig": fired[0][:160], "n_occurrences": fired[1],
                      "window_n": len(window), "acting_n": len(seq),
                      "snap_sha12": hashlib.sha256(
                          cur.encode()).hexdigest()[:12],
                      "session_key": session_key, "ts": time.time()})
        return (CFG["note"], "stop", [])
    except Exception as e:
        try:
            log_decision({"arm": "brake", "routed": "brake-error-serve-on",
                          "error": repr(e)[:200], "ts": time.time()})
        except Exception:
            pass
        return None
