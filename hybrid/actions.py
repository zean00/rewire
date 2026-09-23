"""Web action space (browser-use phase, Gate B).

Maps a replayed web step (task + action history + enumerated page elements) into
the same DECIDE interface the MCQ eval used:

  question   = task + action history (the "what are we doing" context)
  candidates = numbered element lines, e.g.  "[3] button 'Sign in'"

Number prefixes give every candidate a distinct first token, which is what makes
the pre-mask validity check and the format anchor behave exactly as in eval_v1
(phase 1 finding: anchor lifts premask mass 0.016 -> 0.912).

Element selection is the scored decision; the operation (CLICK/TYPE/SELECT) is
predicted by a separate 3-way read over OPS. The null option (NO_ACTION) is part
of the runtime's option set — Mind2Web gold never selects it, so the false-
action rate is measurable. DESTRUCTIVE_* powers the asymmetric escalation gate:
irreversible-looking steps escalate even at high confidence (agent-loop costs
are asymmetric — a wrong fast click can be unrecoverable; a wasted think step
only costs seconds). Keyword heuristic, documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

OPS = ("CLICK", "TYPE", "SELECT")
NULL_ACTION = "NO_ACTION — no element on this page is appropriate for the next step"

OP_CANDIDATES = (
    "CLICK — press or activate the target element (buttons, links, icons)",
    "TYPE — enter text into the target element (input fields, search boxes)",
    "SELECT — choose an option from the target element (dropdowns, comboboxes)",
)

DESTRUCTIVE_KEYWORDS = (
    "delete", "remove", "cancel", "sign out", "log out", "empty", "purge",
    "confirm", "submit", "pay", "payment", "checkout", "purchase", "order",
    "send", "transfer", "permanently",
)


@dataclass
class WebStep:
    id: str
    task: str
    history: list[str]
    op: str
    value: str | None
    candidates: list[dict]          # [{bid, tag, text, attrs}, ...]
    gold_bid: str
    n_candidates_total: int


def load_web_steps(path) -> list[WebStep]:
    import json
    steps = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                steps.append(WebStep(
                    id=d["id"], task=d["task"], history=d["history"],
                    op=d["op"], value=d.get("value"),
                    candidates=d["candidates"], gold_bid=d["gold_bid"],
                    n_candidates_total=d["n_candidates_total"],
                ))
    return steps


def context_text(step: WebStep, op: str | None = None) -> str:
    """Context for the decision reads and the generation baselines.

    op=None: the MCQ-style "what should the next step be" question (v1 arms).
    op set: op-first factorization (v2) — the operation is already decided, so
    the element read asks which element the operation should act on.

    Deliberately does NOT render the candidate lines — question_messages adds
    them as the Options block (same contract as the MCQ eval: the question is
    the context, the candidates are the choices).
    """
    lines = [f"Task: {step.task}", "", "Actions taken so far:"]
    if step.history:
        lines += [f"- {h}" for h in step.history]
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append(f"Current page has {len(step.candidates)} candidate elements (listed in the options).")
    lines.append("")
    if op is None:
        lines.append(f"What should the next step's action be? The operation must be "
                     f"one of {', '.join(OPS)} (with a value for TYPE/SELECT if needed).")
    else:
        lines.append(f"The next step's operation is {op}. "
                     f"Which element should the {op} act on?")
    return "\n".join(lines)


def rich_blocks(step: WebStep, bids: list[str]) -> list[str]:
    """Longer per-candidate renderings for the escalation re-read (v2).

    The Gate B failure analysis showed text-free elements (svg icons) are only
    identifiable by context, so the escalation re-read swaps *reasoning* for
    *information*: full subtree text, grandparent text, all key attributes.
    Blocks start with 'Candidate <n>' — distinct first tokens, so the pre-mask
    check stays well-defined.
    """
    by_bid = {c["bid"]: c for c in step.candidates}
    blocks = []
    for i, bid in enumerate(bids, 1):
        c = by_bid.get(bid)
        if c is None:
            continue
        attrs = c.get("attrs") or {}
        parts = [f"Candidate {i}: {c['tag']}"]
        if (c.get("text") or "").strip():
            parts.append(f"'{_short(c['text'], 80)}'")
        for k in ("aria-label", "placeholder", "title", "id", "href"):
            if attrs.get(k):
                parts.append(f"{k}={_short(attrs[k], 40)}")
        path = c.get("path") or []
        if path:
            parts.append(f"path:{' > '.join(path[::-1])}")
        if (c.get("parent_text") or "").strip():
            parts.append(f"in '{_short(c['parent_text'], 40)}'")
        if (c.get("gp_text") or "").strip():
            parts.append(f"section '{_short(c['gp_text'], 40)}'")
        block = " ".join(parts)
        if (c.get("sub") or "").strip():
            block += f"\n  on-screen text: {_short(c['sub'], 200)}"
        blocks.append(block)
    return blocks


def candidate_lines(step: WebStep) -> list[str]:
    """Numbered element lines: the DECIDE candidate set for element selection.

    Rendering carries the three signals that survived the smoke-test failure
    analysis: own text, DOM path, and parent text (icon/svg elements are only
    identifiable by context — their gold line is otherwise indistinguishable
    from every other empty svg on the page).
    """
    out = []
    for i, c in enumerate(step.candidates, 1):
        attrs = c.get("attrs") or {}
        parts = [f"[{i}] {c['tag']}"]
        if (c.get("text") or "").strip():
            parts.append(f"'{_short(c['text'], 80)}'")
        hint = None
        for k in ("aria-label", "placeholder", "title", "id"):
            if attrs.get(k):
                hint = f"{k}={_short(attrs[k], 30)}"
                break
        if hint is None and attrs.get("href"):
            hint = f"href={_short(attrs['href'], 30)}"
        if hint:
            parts.append(hint)
        path = c.get("path") or []
        if path:
            parts.append(f"path:{' > '.join(path[::-1])}")
        if (c.get("parent_text") or "").strip():
            parts.append(f"in '{_short(c['parent_text'], 40)}'")
        out.append(_short(" ".join(parts), 160))
    return out


def _short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s[: n - 1] + "…" if len(s) > n else s


def is_destructive(step: WebStep, selected_bid: str) -> bool:
    """Keyword heuristic over the selected element's text/attrs (v1, documented)."""
    for c in step.candidates:
        if c["bid"] != selected_bid:
            continue
        hay = " ".join([
            c.get("text") or "",
            str((c.get("attrs") or {}).get("aria-label") or ""),
            str((c.get("attrs") or {}).get("title") or ""),
        ]).lower()
        return any(k in hay for k in DESTRUCTIVE_KEYWORDS)
    return False
