# S-experiment: d2_over_modes — can the decision-read mechanism itself pick the output mode?

Question (user): "with the same decision mechanism, we can determine which mode should be used
with this input — why not?" Measured here: point the exact D2 sequence-scoring read (the same
machinery that picks op / element / value) at three output-contract candidates and route on the
argmax. Test split n=900 (Mind2Web replay); batched D2, alpha=1.0, anchor=True; two context
variants. Runner: `eval/mode_probe.py`; per-step records: `eval/reports/mode_probe/mode_probe_records.jsonl` (1,800).

**Preregistered gold mode (deterministic from the gold action, not fitted):** DECIDE iff gold op
= CLICK (a pure choice suffices); TOOL_CALL iff gold op ∈ {TYPE, SELECT} (a value must be
produced). TEXT has no replay gold — distractor candidate only, its wins scored as automatic
failures. **Preregistered cost asymmetry:** DECIDE-routed on a gold-TYPE/SELECT step is a
guaranteed action loss; TOOL_CALL-routed on a gold-CLICK step costs only latency; TEXT anywhere
is a guaranteed loss.

## Setup

| variant | prompt context | candidates | p50/read |
|---|---|---|---|
| full | task + history + all element lines + "Which output contract should this turn use?" (H3B op-re-read rendering) | DECIDE / TOOL_CALL / TEXT (OP_DESC style) | 147 ms |
| bare | task + history only + the same mode question | same | 118 ms |

Gold modes on test: DECIDE 760 (0.844 — the CLICK base rate), TOOL_CALL 140 (0.156), TEXT 0.

## Result 1 — the mode read cannot pick the mode

| router | mode acc | confusion (gold TOOL_CALL → DECIDE) | premask_ok | TEXT wins |
|---|---|---|---|---|
| always-DECIDE (constant, for reference) | 0.844 | — | — | — |
| d2_over_modes · full | 0.834 | **133/140 (95%)** | 0.321 | 0 |
| d2_over_modes · bare | 0.840 | 136/140 (97%) | 0.257 | 0 |

- The dedicated mode read is **below the trivial constant router** at its own task (0.834 vs 0.844).
  It collapses to DECIDE: full-context errors by gold op = TYPE 104, SELECT 29, CLICK 16. The
  likelihood is dominated by the input's surface form — the prompt IS a selection task (element
  options rendered), so "pick an element" wins the argmax regardless of the required contract.
- **The runtime's own pre-mask validity guard rejects ~70% of these reads** (0.321 / 0.257 pass):
  with elements in view, the model's first-token mass goes to element numbers, not mode names —
  the mode question is not a salient one for the model.
- TEXT never wins on any step (P4 confirmed).
- Mode derived from the chain's existing op-first read (zero extra cost): 0.321 — the
  element-blind op read cannot even serve as a mode proxy (its known SELECT/CLICK bias).
- Wrong routes are near-ties, not confident errors: error margin (top1−top2) p50 = 0.058 — the
  likelihoods barely separate the contracts at all. The contract is not a linguistic property of
  the input; there is almost nothing for the read to read.

## Result 2 — executed end-to-end, every model-read router loses to affordance routing

Execution splice (no new inference): TOOL_CALL-routed steps take the measured per-item H3B chain
action correctness (the chain IS the runtime's TOOL_CALL mode); DECIDE-routed steps take the D2E
per-item unconditioned element decide, executed as a click (success iff gold CLICK and element
right); TEXT-routed = failure.

| router (all deployable except the last) | branch sizes TC/DC/TX | action acc | Δ vs status quo |
|---|---|---|---|
| always TOOL_CALL (affordance routing, status quo) | 900/0/0 | **0.320** | — |
| d2_over_modes · full | 23/877/0 | 0.273 | **−4.7 pts** |
| d2_over_modes · bare | 12/888/0 | 0.273 | −4.7 pts |
| mode from op-first read (derived) | 641/259/0 | 0.297 | −2.3 pts |
| form_present (DOM: any input/textarea/select candidate) | 358/542/0 | 0.298 | −2.2 pts |
| **perfect mode router (oracle upper bound)** | 140/760/0 | 0.292 | **−2.8 pts** |

**The decisive row is the oracle.** Even with a perfect mode router, action accuracy DROPS 2.8
points below always-TOOL_CALL. The reason: DECIDE mode's element read is unconditioned (D2E elem
acc 0.320 overall; 246/760 = 0.324 on these steps) while the chain's element read is
op-conditioned (0.356) — the op conditioning is accuracy-relevant context, not overhead (the
context-parity lesson, again). DECIDE mode is not "TOOL_CALL minus latency"; it is a strictly
weaker reader, and its only justification is when the harness knows from structure that no value
is needed and the cheap read suffices (e.g., a 2-option confirmation dialog).

## Verdict

The proposal is now measured, not argued: the same mechanism **can** be pointed at the mode
question, and it routes at/below the trivial constant, fails the runtime's own validity guard
~70% of the time, and costs −4.7 pts action accuracy executed end-to-end — while the
harness-side affordance decision is free and correct by construction. Even a perfect mode router
cannot win, because the contract declaration is what enables the strong (conditioned) reads.
Mode selection stays in the harness; the reads stay inside the contract they are declared for.

**Caveats kept visible:** replay only — TEXT mode is untestable here (no free-form gold) and was
scored as a distractor; the two-way DECIDE↔TOOL_CALL boundary is what was measured. The
execution simulation splices measured per-item records (H3B chain, D2E element decide) rather
than re-running a coupled runtime — legitimate for accuracy because the arms are
per-item-independent, but it does not measure added latency interaction (the mode read itself
costs +147 ms p50 on top of whatever mode it picks). Mode-candidate rendering mirrors the
measured OP_DESC style; other renderings were not swept.
