# Rewire

> **A decision/read-and-route layer over one frozen LLM.**
> No weight changes. No inference-engine changes. One layer between the engine and its OpenAI-compatible API — and the model gains a fast, grounded *pointing* reflex alongside its ability to write.

**Rewire** is a proof-of-concept response to TypeSafe's [System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). When Jev shipped — a new model class that answers with typed decisions instead of text, trading guaranteed accuracy for speed and hallucination-free-by-construction answers — the open-source wave that followed mostly raced to *replicate or compete* with it. Rewire makes a different bet: **the reflex and the writer are complements, not competitors.** So instead of training a new model, Rewire wires Jev-style reflexes *into* a general LLM everyone already has:

> 🌐 **[Read the interactive story page](https://zean00.github.io/rewire/)** — the 5-minute visual version of this project (served by GitHub Pages; the full audit report lives at [`poc_report.html`](poc_report.html)).

| | |
|---|---|
| **DECIDE** | Scores candidate answers by reading the model's **logits directly** — zero generated tokens. One glance (~0.26 s) instead of writing an essay (~17 s). Because it only ever picks from options you list, it **cannot invent** an answer — it can be *wrong*, never *fabricated*. |
| **GATE** | Knows when it doesn't know: calibrated confidence thresholds decide when to accept, think harder, or **ask a human** (10% ask rate at precision 1.0 in live web tests). From entry L on, GATE also runs the trained **completion-claim verifier dial** (see below). |
| **THINK** | Full LLM generation — reasoning, multimodal input, free-form text — spent only where it's earned (~18% escalation on the offline suite). |

The layer speaks the **OpenAI-compatible protocol**, so any existing agent harness can adopt it by changing one base URL:

```
your agent  →  OpenAI-compatible API  →  ⚡ REWIRE (decide · gate · think)  →  inference engine  →  ❄ frozen weights
```

---

## Results (7-day PoC, all numbers pre-registered)

One frozen open-weights **4B** model throughout (`Qwen/Qwen3.5-4B`), with a **12B** spot-check (`gemma-4-12b-it` via llama.cpp) to confirm the effect isn't a small-model fluke. Same model, same prompts, same harness on every row — only the wiring differs.

| Benchmark | no-think (baseline) | think-always (default LLM) | Rewire chain |
|---|---|---|---|
| Multiple choice, n=500 | 0.764 @ 0.29 s | **0.490** @ 17.29 s | **0.856** @ 0.26 s |
| Web actions replay, n=900 | 0.258 | 0.115 @ 27.4 s/step | **0.356** @ 1.77 s (p ≈ 8×10⁻⁹) |
| Tool calls, n=900 | 0.274 | 0.100 (79% unparseable) | **0.320** @ 2.09 s (p = 0.0033) |
| Live agent harness, 3-arm | 0 / 5 tasks | — | **5 / 5** (4B) · 0/5 → **4/5** (12B) |

Highlights:

- **Thinking made multiple choice *worse*, not better** (49% vs 76% just answering) — deliberation is a tax, and on selection-shaped work it buys nothing.
- **Batched logit reads**: scoring 41 candidates went from 1.70 s → **0.34 s** p50 (5×), 98.3% agreement with sequential reads.
- **The gate knows what it doesn't know**: on live staged sites, the reflex's two errors were exactly its two lowest-confidence calls; ask-a-human fired on 10% of decisions with precision 1.0.

### The honest ledger (and why we lead with it)

Every claim above shipped with a pass/fail bar written *before* the data ran. Five designs died at their own gates; the negatives are documented as fully as the positives:

- **Model self-verification doesn't work — in words**: logit reads over 316 probes carry no outcome signal (AUROC 0.51–0.57 — worse than a constant predictor; reads with ≥90% confidence corresponded to 0% actual passes). Multi-read and paraphrase-armor variants failed their pre-registered bars too (3/28 confirm, 32/288 false-approvals). *(Two days later the same bar was cleared by reading state vectors instead of words — see the next section.)*
- **A one-line regex beat every learned verifier**: a `Last reward: -\d` check on the harness floor catches 55/55 reward-blind agent failures with **zero** false vetoes. Sometimes the right wire is copper.
- **The hardest benchmark is an honest tie**: MiniWoB++ 11/48 vs 10/48 — reported as a tie, mechanism-analyzed in the report, and demoted to a smoke test rather than tuned (that would be overfitting by definition).
- **Infrastructure bugs were caught by adversarial checks**: silent KV-cache read corruption (bit-parity check → root cause → byte-exact re-verification), temperature-invariance under grammar constraints, primacy/order artifacts. All documented in the ledger.

**Verdict**: the routing mechanism works wherever the task hands the read a real signal — choices, candidates, rankings — at ~1/60th the latency and zero hallucination surface. Asked to divine things it was never shown, it fails — and the project measured exactly where that line sits. (Then it found the way across it — next section.)

---

## The verifier dial — the CLM-inspired reversal (Sep 23–25, entries H–M)

The honest ledger above ends with every LLM read suspended behind a written bar: **AUROC ≥ 0.75 AND Brier better than the cell's constant predictor, pre-registered, one pass, on fresh sessions.** The program then found a way back in — from an unexpected direction.

[Contrastive-LM's CLM](https://github.com/Contrastive-LM/CLM) (Apache-2.0) is a verifier class built on a frozen LLM encoder whose raw **state vectors** — not its words — are matched against tiny trained heads. We imported the *idea*, not the checkpoint, and ran it through the same gauntlet:

| Entry | Question | Fresh-pool result | Gate |
|---|---|---|---|
| **H** | CLM's off-the-shelf contrastive heads, zero-shot | ranking reaches 0.99, but calibration fails (ECE 0.24; Brier over constant) | ✗ fail |
| **J** | the same frozen Qwen3-8B encoder + our tiny head (logistic ranker + 2-parameter Platt, trained on 316 probes) | **AUROC 0.9889 · Brier 0.0374** (constant 0.0982), on 290 probes from 168 fresh sessions | ✓ **earned** |
| **K** | the same recipe on the chain's own Gemma-4-12B backbone via its stock `/v1/embeddings` endpoint — no second model | **0.8867 · 0.0653** | ✓ earned |
| **L** | live confirmation: 168 brand-new sessions, the dial frozen *before* the sweep, one adjudication pass | **0.9011 · 0.0861** (constant 0.1512) | ✓ **ships** |
| **M** | is 4-bit quantization the gap? Qwen3-8B Q4_K_M through the same serving stack | **0.9868 · 0.0344 ≈ bf16 (0.9889)** | attribution: quantization exonerated |

What ships (entry L): when the agent claims a task is done, the GATE embeds the evidence block it was already reading — verbatim, no question attached — through the engine's stock embeddings API (~95 ms), and scores it with a frozen dial exported as pure data: 3,840 weights, two Platt scalars, one threshold, `p = σ(a·(w·x+b)+c)`. At the frozen operating point on 140 fresh completion claims: **accept precision 93.3%, and 113 of 114 false "done" claims caught (99.1%)** — at the deliberate cost of rejecting 46% of genuine completions. That threshold is a policy dial, not a defect; rebalancing it is a new pre-registered decision.

The reversal worth underlining: **asking the model in words carries no signal; reading where its state lands carries nearly all of it.** Same frozen model, same evidence, same bar — a different instrument. And the dial is backbone-agnostic: entry M shows the recipe ports to any open-weights encoder in hours, quantized weights included.

---

## Quickstart

```bash
pip install -e .
pytest            # pure-CPU unit tests (scoring math, runtime, metrics)
```

### A DECIDE read (GPU required)

```bash
python examples/text_decision.py
```

Pins one toy question, then scores the candidates two ways — **D1** (restricted first-token logits, one forward pass) and **D2** (sequence log-likelihood over candidates) — printing selections, calibrated confidence, and the premask validity check.

### The Rewire proxy

[`proxy/omp_proxy.py`](proxy/omp_proxy.py) is the shipped layer — the OpenAI-compatible server the live harness experiments ran through. Two modes:

```bash
# 1) chain mode — the proxy carries no weights; it forwards to any OpenAI-compatible
#    engine and intercepts decision-shaped turns for decide/gate reads:
python proxy/omp_proxy.py --port 8999 \
    --backend-config eval/proxy_backend.json --no-local-model

# 2) local-model mode — serves a model in-process (GPU):
python proxy/omp_proxy.py --port 8999
```

Then point any agent harness at `http://127.0.0.1:8999/v1` instead of the engine directly. In the three-arm experiment the harness (omp) was changed by **zero lines**: it saw a standard endpoint and three model names (`vanilla`, `vanilla-nothink`, `decide`); every decision is logged to `eval/reports/omp_arms/decisions.jsonl`.

### Core library

```python
from hybrid.engine import load_model
from hybrid.template import pin_prompt, question_messages
from hybrid.scoring import score_first_token, score_sequence_logprob
```

`hybrid.scoring` implements the read primitives (D1 restricted first-token logits, D2 sequence log-prob, premask validity), `hybrid.runtime` composes them into the decide → gate → think chain, and `hybrid.calibration.metrics` holds the ECE/Brier/AUROC harness used for every calibration claim in the report.

---

## Repository layout

```
rewire/
├── hybrid/                 # core library: scoring, decide, gate, runtime, calibration
│   ├── scoring/            #   D1 first-token logits · D2 sequence logprob · premask
│   ├── calibration/        #   ECE / Brier / AUROC harness
│   └── runtime.py          #   the decide → gate → think chain
├── proxy/                  # omp_proxy.py — the OpenAI-compatible Rewire layer (v3.15.0)
├── eval/                   # benchmark harnesses, datasets, frozen configs, reports
│   ├── datasets/           #   toy_mcqa.jsonl · webreplay_v1/ (with MANIFEST)
│   └── reports/            #   measurement records: PoC eras + entry_h…entry_m (the verifier dial)
├── configs/                # model / thresholds / benchmark configs
├── examples/               # minimal DECIDE example
├── tests/                  # pytest suite
├── scripts/                # env check · optional host-sync helper
├── index.html              # the interactive story page (the Pages site root)
├── poc_report.html         # the full audit report (self-contained)
├── POC_CONCLUSION.md       # structured conclusion: 7 PoC eras + the verifier-dial arc
└── IMPLEMENTATION_PLAN.md  # the 1,600-line pre-registration ledger — the complete audit trail
```

## Method (why the numbers are believable)

- **Pre-registration**: framing, evidence rules, and pass/fail thresholds were written *before* each run; adversarial sanity gates were allowed to kill designs (and killed five).
- **One pass per instrument**: no reruns-until-significant; every deviation is disclosed in the ledger with dates and PIDs.
- **Bit-exact reproduction**: scorers reproduce across passes to 0.0000 (316/316 probes, sha256-pinned score files).
- **Blind adjudication**: outcomes graded from transcripts and environment records, never exit codes.
- **Constants as adversaries**: every learned component is compared against the constant predictor on its own data — the sidecar that merely matched the constant (82.9%) was killed.

Full methodology, per-experiment logs, and the decision record: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) · distilled findings: [POC_CONCLUSION.md](POC_CONCLUSION.md).

## Where this goes

The PoC's last finding is its most provocative: the reflex works when it's *told* exactly what to score, and the model's untrained self-judgment carries no signal — in its words. Its state vectors are another matter: the verifier dial (above) cleared the written acceptance bar on fresh sessions and now ships as the GATE's completion-claim verifier. Next, in pre-registration order: **enforce mode** (the dial steering behavior — a rejected claim re-opens the episode, with the reject policy and retry cap frozen before it goes live), a **τ rebalance** (the frozen threshold sits at a conservative corner — 46% of genuine completions are sent back), and **mid-thought checkpoint reads** (score a thought *while* it happens, not only at its end). The endgame direction stands: **native decide** — decision heads trained into the backbone (typed outputs, calibrated confidence as a first-class objective), with the contract discipline, per-model calibration refits, and the written bar carried over as the spec any native implementation must beat — a bar that now has a measured proof it can be met. Mechanical gates first, model reads only where deterministic signals can't see.

## Credits

- [System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe AI, for the framing and the provocation.
- [Contrastive-LM / CLM](https://github.com/Contrastive-LM/CLM) (Apache-2.0) — the frozen-encoder-plus-tiny-head verifier class that inspired the dial arc (entries H–M).
- [djev-run](https://github.com/taeold/djev-run) and vLLM's System One diffusion server (PR #57250) — the open-source System One wave this project responds to differently.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [Qwen](https://huggingface.co/Qwen), [MiniWoB++](https://miniwob.pluslab.org/) — infrastructure and benchmarks.

## License

[MIT](LICENSE)
