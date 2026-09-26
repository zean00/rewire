"""Entry-N enforce dial — decision-logic tests, no GPU, no network.

The dial's embed() is monkeypatched to a deterministic vector, so score()
exercises the real readout math (sigmoid(a*(w.x+b)+c)) against a toy
readout written to a temp file and loaded through the real init().
"""

import json
import math

import pytest

import importlib.util
from pathlib import Path

_PROXY = Path(__file__).resolve().parents[1] / "proxy" / "dial_enforce.py"
_spec = importlib.util.spec_from_file_location("dial_enforce", _PROXY)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


TAU = 0.5


def toy_readout(tmp_path):
    # p = sigmoid(1*(1.0*x0 + 0) + 0) = sigmoid(x0): x0=4 -> 0.982, x0=-4 -> 0.018
    ro = {"w": [1.0, 0.0], "b": 0.0, "platt_a": 1.0, "platt_c": 0.0,
          "tau": TAU, "recipe": "test"}
    f = tmp_path / "ro.json"
    f.write_text(json.dumps(ro))
    return f


@pytest.fixture()
def dial(tmp_path, monkeypatch):
    status = D.init({"enabled": True, "readout": str(toy_readout(tmp_path)),
                     "embeddings_url": "http://127.0.0.1:1", "cap": 2})
    assert "enabled" in status
    monkeypatch.setattr(D, "embed", lambda text: [4.0, 0.0])  # p ~ 0.982
    yield D
    D.CFG = None
    D._STATE.clear()


LOG = []


def log(rec):
    LOG.append(rec)


CLAIM = ("The task is complete: the form was submitted successfully and "
         "the confirmation message is visible. All steps are done.")
NO_VOCAB = ("I have taken a careful look at the whole page and considered "
             "every element that could possibly matter here today.")


def msgs_with_tool_result(text):
    return [{"role": "user", "content": "do it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "eval", "arguments": {"code": "x"}}}]},
            {"role": "tool", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "content": CLAIM}]


def msgs_same_first_message(text, tab="w111c0"):
    """The live harness shape the v6 sweep disclosed: a constant first wire
    message across sessions, the per-session tab name only inside an eval
    cell, then the tool result and the claim."""
    return [{"role": "user", "content": "You are omp. Same for everyone."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "eval",
                              "arguments": {"code":
                                            f"browser.open({{name:'{tab}'}})"}}}]},
            {"role": "tool", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "content": CLAIM}]


# --- jurisdiction (the frozen probe rule) -----------------------------------

def test_short_prose_is_outside(dial):
    assert not D.is_completion_claim("Done.")


def test_long_prose_without_vocab_is_outside(dial):
    assert not D.is_completion_claim(NO_VOCAB)


def test_vocab_sentence_in_long_prose_is_a_claim(dial):
    assert D.is_completion_claim(CLAIM)


# --- evidence builder (the frozen build_evidence rule) ----------------------

def test_evidence_is_newest_non_blank_tool_result(dial):
    msgs = msgs_with_tool_result("  page   snapshot\nhere  ")
    assert D.evidence_from_messages(msgs) == "page snapshot here"


def test_evidence_skips_blank_results(dial):
    msgs = [{"role": "tool", "content": "earlier page"},
            {"role": "tool", "content": [{"type": "text", "text": "   "}]},
            {"role": "assistant", "content": CLAIM}]
    assert D.evidence_from_messages(msgs) == "earlier page"


def test_evidence_caps_at_8000(dial):
    msgs = msgs_with_tool_result("x" * 9000)
    assert len(D.evidence_from_messages(msgs)) == 8000


# --- routing (registered D2 policy) --------------------------------=========

def test_accept_ships_unchanged(dial):
    dial.embed = lambda text: [4.0, 0.0]
    LOG.clear()
    out = D.route(msgs_with_tool_result("snap"), CLAIM, log)
    assert out is None
    assert LOG[-1]["routed"] == "accept"
    assert LOG[-1]["first_claim"] is True
    assert LOG[-1]["claim_event"] == 1


def test_sub_tau_vetoes_then_ships_at_cap(dial, monkeypatch):
    monkeypatch.setattr(dial, "embed", lambda text: [-4.0, 0.0])  # p ~ 0.018
    LOG.clear()
    msgs = msgs_with_tool_result("snap")
    # claim event 1: veto -> {"note": ...}; note leaks no numbers
    out = D.route(msgs, CLAIM, log)
    assert out is not None and out["note"] == D.VETO_NOTE
    assert not any(ch.isdigit() for ch in out["note"])
    assert LOG[-1]["routed"] == "veto" and LOG[-1]["first_claim"] is True
    # claim event 2 (re-claim 1): second veto
    out = D.route(msgs, CLAIM, log)
    assert out is not None
    assert LOG[-1]["routed"] == "veto" and LOG[-1]["first_claim"] is False
    # claim event 3 (re-claim 2): final veto ships = cap-terminate
    out = D.route(msgs, CLAIM, log)
    assert out is None
    assert LOG[-1]["routed"] == "cap-terminate"


def test_outside_jurisdiction_ships_without_scoring(dial, monkeypatch):
    LOG.clear()
    called = []
    monkeypatch.setattr(dial, "embed",
                        lambda text: called.append(text) or [4.0, 0.0])
    out = D.route(msgs_with_tool_result("snap"), "Done.", log)
    out2 = D.route(msgs_with_tool_result("snap"), NO_VOCAB, log)
    assert out is None and out2 is None
    assert not called  # never touched the instrument
    assert all(r["routed"] == "outside-jurisdiction" for r in LOG)


def test_instrument_failure_is_disclosed_fail_open(dial, monkeypatch):
    def boom(text):
        raise ConnectionError("embeddings down")
    monkeypatch.setattr(dial, "embed", boom)
    LOG.clear()
    out = D.route(msgs_with_tool_result("snap"), CLAIM, log)
    assert out is None  # ships...
    assert LOG[-1]["routed"] == "instrument-error-accept"  # ...disclosed


def test_sessions_are_independent(dial, monkeypatch):
    monkeypatch.setattr(dial, "embed", lambda text: [-4.0, 0.0])
    LOG.clear()
    a = [{"role": "user", "content": "task A"}] + msgs_with_tool_result("s")[1:]
    b = [{"role": "user", "content": "task B"}] + msgs_with_tool_result("s")[1:]
    assert D.route(a, CLAIM, log) is not None  # session A veto 1
    assert D.route(b, CLAIM, log) is not None  # session B veto 1 (own cap)
    rec = [r for r in LOG if r.get("claim_event") == 1]
    assert len(rec) == 2


def test_disabled_init_never_routes(tmp_path):
    assert D.init({"enabled": False}) != ""
    assert D.CFG is None
    assert D.route(msgs_with_tool_result("s"), CLAIM, log) is None


def test_session_key_tabs_are_independent_even_with_same_first_message(dial):
    # Regression (v6 sweep, disclosed before any metric): sessions keyed on
    # the first wire message shared ONE global cap state, so after the first
    # session's two vetoes every later session's sub-tau claims shipped as
    # cap-terminate. The tab name must separate them.
    LOG.clear()
    dial.embed = lambda text: [-4.0, 0.0]  # p ~ 0.018: every claim is sub-tau
    a = msgs_same_first_message("state A", tab="w111c0")
    b = msgs_same_first_message("state B", tab="w222c0")
    va = D.route(a, CLAIM, log, session_key="w111c0")
    assert va is not None and va["note"]  # session A veto 1
    vb = D.route(b, CLAIM, log, session_key="w222c0")
    assert vb is not None and vb["note"]  # session B has its OWN cap: veto 1
    recs = [r for r in LOG if r.get("routed") == "veto"]
    assert len(recs) == 2


def test_session_key_fallback_is_first_message(dial):
    # Without a tab name the old fallback applies — documented, and now
    # understood to be a cross-session SHARE when the harness's first
    # message is constant. Two tab-less calls share one state.
    LOG.clear()
    dial.embed = lambda text: [-4.0, 0.0]  # p ~ 0.018: every claim is sub-tau
    m = msgs_same_first_message("state", tab="w111c0")
    assert D.route(m, CLAIM, log) is not None          # veto 1
    assert D.route(m, CLAIM, log) is not None          # veto 2 (same state)
    assert D.route(m, CLAIM, log) is None              # cap-terminate ships
    recs = [r for r in LOG if r.get("routed") == "cap-terminate"]
    assert len(recs) == 1


def test_score_math_matches_readout_shape(dial):
    # real score() math vs the hand formula, monkeypatched embed
    dial.embed = lambda text: [4.0, 0.0]
    p, nrm = D.score("whatever")
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-4.0)))
    assert nrm == pytest.approx(4.0)


def test_load_remote_passes_dial_block(tmp_path):
    # Regression (v3.17.0 launch): load_remote built an allowlisted dict and
    # dropped the dial block, so the startup banner read "enforce dial:
    # disabled" with the config saying enabled — the v3.14.0 dropped-key
    # launch bug, one layer over. The banner gate caught it before any
    # session; this test pins the pass-through. omp_proxy imports torch at
    # module level, so it skips on the torch-less laptop and runs on the host.
    if importlib.util.find_spec("torch") is None:
        pytest.skip("omp_proxy imports torch at module level")
    spec = importlib.util.spec_from_file_location(
        "omp_proxy_load_remote",
        Path(__file__).resolve().parents[1] / "proxy" / "omp_proxy.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    cfg = {"enabled": True, "wire_model": "gemma-4-12b-it",
           "base_url": "http://127.0.0.1:8997/v1", "model": "gemma-4-12b-it",
           "chain": True, "completion_review": {"enabled": False},
           "dial": {"enabled": True, "readout": "ro.json", "cap": 2}}
    f = tmp_path / "backend.json"
    f.write_text(json.dumps(cfg))
    r = M.load_remote(str(f))
    assert r["dial"] == cfg["dial"]
    assert r["completion_review"] is None  # disabled block stays absent
    assert r["chain"] is True
