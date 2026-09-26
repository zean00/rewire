#!/usr/bin/env python3
"""(v3.17.0) Entry-N enforce-mode dial — the frozen completion-claim
verifier, promoted from log-only to ROUTING.

Registration: IMPLEMENTATION_PLAN.md, entry N (live, 2026-09-25). At every
completion-claim moment — the proxy's session-ending prose answer on the
decide arm — the claim's evidence text (the frozen rule: the newest
non-blank tool result, whitespace-compacted, 8000-char cap) is embedded by
the host's combined llama-server and scored by the frozen readout
(entry_k_readout.json, sha c840ba54…, unchanged):

    p = sigmoid(platt_a * (w . x + b) + platt_c)     accept iff p >= tau

Jurisdiction is the frozen probe rule (ckpt_review.extract, the entry-L
pool definition): the answer carries >= 40 chars of prose and at least one
sentence with a claim-vocabulary hit. Answers outside that rule ship
unchanged, logged as outside-jurisdiction (they were never probes in the
validated pool).

Registered D2 policy: a sub-tau claim is VETOED — the answer is discarded,
the agent gets a note that its claim failed verification (no score, no
threshold, no hint at what would pass — safeguard (e)) plus one fresh
snapshot cell, and the episode continues. Cap = 2 continuation vetoes per
session (<= 3 claim events); the third sub-tau claim event is the final
veto: the answer SHIPS (episode terminated with state as-is) and is logged
as cap-terminate — page reward alone decides success, the dial never
grants it. First claims are flagged in the log: the dial audit uses FIRST
claims only, because a post-veto state is an intervention outcome, out of
the readout's distribution.

Instrument failure (embeddings endpoint unreachable, malformed vector,
readout error) = disclosed fail-open: the answer ships, logged as
instrument-error-accept. What remains impossible is the SILENT pass —
every session-ending prose answer is either scored, or its failure to
score is in the decision log.

Serial rule (the entry-K standing instrument rule): one text per
/v1/embeddings POST, strictly serial — satisfied structurally, because the
proxy's compute() holds LOCK across the whole turn, so embed calls cannot
overlap. Embedding vectors arrive L2-normalized from the server
(--embd-normalize default); the dot product never re-normalizes.

Stdlib only. The proxy imports this defensively: absence or breakage here
must never touch serving (the dial is then simply absent).
"""
import hashlib
import json
import math
import re
import time
import urllib.request

# Verbatim from ckpt_review.py — the vocabulary that made a sentence a
# probe in every adjudicated pool (entries H through L). Changing it is a
# new pre-registered decision.
CLAIM_VOCAB = [
    "success", "successfully", "logged in", "logged out", "logged into",
    "secure area", "added to", "added the", "cart", "badge", "sorted",
    "sorted by", "submitted", "clicked", "entered", "typed",
    "confirmation", "confirmed", "verified", "welcome", "complete",
    "completed", "task is done", "invalid", "error", "failed", "not found",
]
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

PROSE_MIN_CHARS = 40
EVIDENCE_CAP = 8000

# Registered D2 note — no score, no threshold, no policy detail (the agent
# must not be able to write claims FOR the gate).
VETO_NOTE = ("Your completion claim failed verification against the "
             "current page state. Continue working: fix what is missing, "
             "or re-claim completion if the task is in fact done.")

CFG: dict | None = None      # set by init(); None = feature absent
_STATE: dict = {}            # session key -> {"events": int, "vetoes": int}
_STATE_MAX = 64


def init(block) -> str:
    """Normalize the backend config's dial block. Missing/disabled = the
    feature is absent (byte-for-byte prior serving). Returns a status
    string for the startup banner."""
    global CFG, _STATE
    _STATE = {}
    if not isinstance(block, dict) or not block.get("enabled"):
        CFG = None
        return "disabled (dial block absent or enabled:false)"
    ro = block.get("readout", "")
    try:
        with open(ro) as f:
            r = json.load(f)
        w = [float(x) for x in r["w"]]
        cfg = {"w": w, "b": float(r["b"]),
               "a": float(r["platt_a"]), "c": float(r["platt_c"]),
               "tau": float(r["tau"]),
               "url": str(block.get("embeddings_url", "")).rstrip("/"),
               "model": str(block.get("embed_model", "gemma-4-12b-it")),
               "cap": int(block.get("cap", 2)),
               "timeout_s": float(block.get("timeout_s", 60)),
               "note": str(block.get("note", VETO_NOTE)),
               "readout": ro}
    except Exception as e:
        CFG = None
        return f"FAILED to load ({e}) — feature absent"
    CFG = cfg
    return (f"enabled tau={cfg['tau']:.6f} cap={cfg['cap']} dim={len(w)} "
            f"readout={ro} embeddings={cfg['url']}")


def _session_key(messages, session_key=None) -> str:
    """Per-session identity. The caller's tab name ('w<seed>c0') is the
    truthful key — seed-unique per session and stable across harness
    compactions. The first message's repr head is only the fallback: the
    v6 sweep disclosed that the harness's first wire message is constant
    across sessions, which globally shared the cap state."""
    if session_key:
        return f"tab:{session_key}"
    return repr(messages[0])[:300] if messages else "?"


def prose_of(content) -> str:
    """The claim text: str content as-is; block content as the \\n-join of
    its NON-BLANK text blocks (the transcript rule's construction)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(b.get("text", "")) for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(t for t in parts if t.strip())
    return ""


def is_completion_claim(prose: str) -> bool:
    """The frozen probe rule: >= 40 chars of prose and at least one
    sentence with a claim-vocabulary hit."""
    if len(prose.strip()) < PROSE_MIN_CHARS:
        return False
    for sent in SENT_SPLIT.split(prose):
        slo = sent.lower()
        if any(v in slo for v in CLAIM_VOCAB):
            return True
    return False


def evidence_from_messages(messages) -> str:
    """build_evidence on the wire: the newest NON-BLANK tool result's text
    blocks joined with \\n, whitespace-compacted, capped. Byte-parity with
    ckpt_rubric.build_evidence on the transcript was smoke-verified
    (eval/reports/entry_n/smoke_parity.py)."""
    last = ""
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, list):
            t = "\n".join(str(b.get("text", "")) for b in c
                          if isinstance(b, dict) and b.get("type") == "text")
        else:
            t = c if isinstance(c, str) else ""
        if t.strip():
            last = t
    return " ".join(last.split())[:EVIDENCE_CAP]


def embed(text: str) -> list:
    """One text per POST, strictly serial (the caller holds LOCK)."""
    body = json.dumps({"model": CFG["model"], "input": text}).encode()
    req = urllib.request.Request(
        CFG["url"] + "/v1/embeddings", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=CFG["timeout_s"]) as r:
        payload = json.load(r)
    vec = payload["data"][0]["embedding"]
    if len(vec) != len(CFG["w"]):
        raise ValueError(f"embedding dim {len(vec)} != readout dim {len(CFG['w'])}")
    return vec


def score(text: str) -> tuple:
    """(p, vec_norm) in the exact serving shape; no re-normalization."""
    vec = embed(text)
    nrm = math.sqrt(sum(x * x for x in vec))
    z = CFG["b"] + sum(wi * xi for wi, xi in zip(CFG["w"], vec))
    p = 1.0 / (1.0 + math.exp(-(CFG["a"] * z + CFG["c"])))
    return p, nrm


def route(messages, content, log_decision, session_key=None) -> dict | None:
    """The one funnel for a session-ending prose answer. Returns
    {"note": ...} to VETO (the caller ships the note + a re-observe cell),
    None to ship unchanged. session_key: the caller's per-session identity
    (the agent's tab name); without it, falls back to the first message.
    Never raises: any internal failure is logged
    as instrument-error-accept and the answer ships (disclosed fail-open)."""
    if CFG is None:
        return None
    prose = prose_of(content)
    try:
        key = _session_key(messages, session_key)
        st = _STATE.get(key)
        if st is None:
            st = {"events": 0, "vetoes": 0}
            _STATE[key] = st
            while len(_STATE) > _STATE_MAX:
                _STATE.pop(next(iter(_STATE)))
        if not is_completion_claim(prose):
            log_decision({"arm": "dial-enforce", "routed": "outside-jurisdiction",
                          "chars": len(prose.strip()), "claim_event": st["events"],
                          "ts": time.time()})
            return None
        st["events"] += 1
        ev = evidence_from_messages(messages)
        t0 = time.perf_counter()
        try:
            p, nrm = score(ev)
        except Exception as e:
            log_decision({"arm": "dial-enforce", "routed": "instrument-error-accept",
                          "error": repr(e)[:200], "claim_event": st["events"],
                          "first_claim": st["events"] == 1,
                          "evidence_sha12": hashlib.sha256(
                              ev.encode()).hexdigest()[:12],
                          "ts": time.time()})
            return None
        first = st["events"] == 1
        if p >= CFG["tau"]:
            decision = "accept"
        elif st["vetoes"] >= CFG["cap"]:
            decision = "cap-terminate"
        else:
            decision = "veto"
        log_decision({"arm": "dial-enforce", "routed": decision,
                      "p": round(p, 6), "tau": CFG["tau"],
                      "claim_event": st["events"], "first_claim": first,
                      "vetoes_done": st["vetoes"],
                      "evidence_chars": len(ev),
                      "evidence_sha12": hashlib.sha256(
                          ev.encode()).hexdigest()[:12],
                      "vec_norm": round(nrm, 4),
                      "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
                      "ts": time.time()})
        if decision != "veto":
            return None
        st["vetoes"] += 1
        return {"note": CFG["note"]}
    except Exception as e:  # the dial itself must never take down a turn
        try:
            log_decision({"arm": "dial-enforce", "routed": "dial-error-accept",
                          "error": repr(e)[:200], "ts": time.time()})
        except Exception:
            pass
        return None
