# Rewire

> **A decision/read-and-route layer over one frozen LLM.**
> No weight changes. No inference-engine changes. One layer between the engine and its OpenAI-compatible API — and the model gains a fast, grounded *pointing* reflex alongside its ability to write.

**Rewire** is a proof-of-concept response to TypeSafe's [System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). When Jev shipped — a new model class that answers with typed decisions instead of text, trading guaranteed accuracy for speed and hallucination-free-by-construction answers — the open-source wave that followed mostly raced to *replicate or compete* with it. Rewire makes a different bet: **the reflex and the writer are complements, not competitors.** So instead of training a new model, Rewire wires Jev-style reflexes *into* a general LLM everyone already has:

> 🌐 **[Read the interactive story page](https://zean00.github.io/rewire/)** — the 5-minute visual version of this project (served by GitHub Pages; the full audit report lives at [`poc_report.html`](poc_report.html)).

| | |
|---|---|
| **DECIDE** | Scores candidate answers by reading the model's **logits directly** — zero generated tokens. One glance (~0.26 s) instead of writing an essay (~17 s). Because it only ever picks from options you list, it **cannot invent** an answer — it can be *wrong*, never *fabricated*. |
| **GATE** | Knows when it doesn't know: calibrated confidence thresholds decide when to accept, think harder, or **ask a human** (10% ask rate at precision 1.0 in live web tests). From entry L on, GATE also runs the trained **completion-claim verifier dial** — and since entry N it *vetoes* claims it can't verify (see below). |
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

## The dial goes live, the open replica flunks, the A/B answers, the brake pulls (entries N–R, Sep 25–28)

**Entry N — the dial starts steering (Sep 25–26).** Enforce mode promotes the frozen readout from logging to routing: at every completion claim, `p < τ*` → the claim is **vetoed**, the agent is told only that verification failed (no score, no threshold, nothing to game), and the episode re-opens — cap 2 vetoes, then the claim ships and the page reward still decides truth. Validated *before any enforce session existed*: 13 unit tests, a byte-parity replay of all 168 prior transcripts, a 4-gate live instrument smoke, and the live path reproducing entry L's adjudicated numbers **exactly** (0.9011 / 0.0861). The 168-session sweep ran under one disclosed mid-run bug (the per-session veto cap keyed globally — only session 1 saw vetoes; page rewards and the audit pool unaffected, disposition recorded before any metric). Integrity held everywhere: **149/149 recovered first claims logged with matching evidence hashes and recomputable scores, zero silent sub-τ accepts.** On the fresh first-claims pool — deliberately harder than entry L's, 92 of 149 being early or mid-session claim-shaped prose — the frozen operating point held: **AUROC 0.8487, Brier 0.0963 vs constant 0.1451, accept precision 92.3%, false-alarm rate 0.85%.**

**Entry JO — the open-weights Jev replica, tested honestly (Sep 26).** The community [Jev-Omni Q4_K_M](https://huggingface.co/Reza2kn/Jev-Omni-Q4_K_M-GGUF) quant (gemma-4-12B backbone + the original FP32 decision head) was run through its own unmodified adapter on the *same* 140-claim pool where the dial scores 0.9011 / 0.0861: **AUROC 0.4028 — below chance — Brier 0.3715, ECE-10 0.3624. It fails all three pre-registered criteria.** The failure is structured, not noisy: of its 20 very-confident (p ≥ 0.9) accepts, exactly 1 was a true completion, and its accept set carries no information over the base rate. This is the project's core bet with a control group: a decision head with excellent calibration *on its own benchmark* (quantizer-reported ECE 0.04) does not transfer across distributions. **Calibration is paid for in labeled data on the serving distribution** — exactly what the dial's label factory (page-reward-labeled claim moments) builds, and what nothing else provides.

**Entry O — the decisive A/B, adjudicated: NO VERIFIED IMPROVEMENT (Sep 26–27).** The question entries L–N deliberately deferred: does the full stack — chain reads + mechanical gates + dial-enforce — complete *more* tasks than the bare model given the same prompt and tools? 168 paired instances × 2 arms = 336 sessions on identical paired pages, arm order balanced; the verdict bar was frozen before session 1 (guarded wins iff the success-rate gap is ≥ +0.10; anything less records as NO VERIFIED IMPROVEMENT). **Verdict: the bar is not met — vanilla finishes ahead.** Success rate (raw > 0 per paired page, no-page-state counted as failure, no exclusions): **guarded 23/168 (13.69%) vs vanilla 29/168 (17.26%)** — gap −0.036, paired wins 12 vs 18 (138 ties). Routing integrity held under audit (27 accepts, zero silent sub-τ passes, 139/139 first claims matched to decision records by evidence hash, live-vs-recomputed p agree 126/126), and the dial's third fresh-pool audit was its strongest yet (**AUROC 0.891, Brier 0.0616 vs constant 0.1232**, accept precision 0.867, false-alarm 1.8%). The failure is deployment, not detection: verification at a verified operating point converted into *fewer* completions at **2.04× the wall clock** (339.8 s vs 166.8 s; 76 vs 20 cap-hits); the veto loop re-opened failing sessions (240 vetoes across 125 sessions) and gave them more rope, not more successes — vetoed sessions succeeded 7/125 vs 16/43 non-vetoed. Reading: on short-horizon tasks where the page reward is visible in-context, a frozen 12B actor does not convert verification into completions — the dial's value claims (cheap, calibrated, safe) are confirmed and the deployment thesis is falsified, which is exactly what the +0.10 bar was built to force. Successor work targets the measured waste instead (11 guarded sessions earned raw > 0 and *then* looped to the cap): entry P attacks wall time, not completions, and needs no superiority claim to pay.

**Entry P — the stuck/loop head, executed: GATE PASS (Sep 27).** From the entry-O watch item (guarded sessions hitting the time cap at 3.8× the vanilla rate — 76 vs 20 — including 11 sessions that had already *earned* success and then looped to the cap): a second dial head on the same frozen embedding — one more linear layer, zero extra model calls — trained to flag *repeating an action on an unchanged page*. Mechanical label rule (same action signature ≥3× in a trailing 8-acting-turn window with a bit-identical page snapshot), composite input block (task + page snapshot + last ≤8 action signatures), session-level validation split, entry-L instrument pins, entry-L gate shape. Registered, committed, and synced *before* the sweep closed (the build guard was verified live refusing to run at 215/336); executed after the entry-O adjudication and a corpus backup, under two disclosed pre-execution fixes (a done-flag closure amendment — all 336 planned sessions ran once, 84 rows carry the page probe's episode flag false, which the label rule never reads; and an endpoint fix — the registered combined server via `--url` after the stale standalone-embeddings pin refused connections, first launch aborted at row 0, zero vectors taken). **Result: both frozen gates pass on the first execution** — val AUROC **0.9621** (bar ≥ 0.75) and val Brier **0.0855 vs constant 0.2022** (bar < constant), n=327 val rows / 92 positives, ECE-10 0.0875, precision 0.934 / recall 0.620 at τ = 0.5, embeddings bit-exact on the 8-row recheck. The loop labels concentrate exactly where entry O measured the waste (last-5-acting-turn share: cap-hit sessions 23.0%, success sessions 24.6%, other 7.9%). Acting on its alerts is deliberately a future registered entry (controller graduation), not this one.

**Entry Q — the adversarial-page probe: ROBUST (Sep 27).** The concern the two practitioner write-ups raised — a page's own text talking a verifier into a false "done" — tested directly on the frozen dial: 80 paired mid-task states from the closed v7 pool, each scored clean and again with one of three forgery templates appended to its evidence (a success banner, a status line, and the literal forged reward line the mechanical gates regex-match). **Adversarial false-alarm 0.0000 — 80/80 states held below τ\*, all three templates** — and the direction is the finding: forged text pushed p DOWN ~30× (mean 0.0182 → 0.0004; max 0.2131 → 0.0063). The dial does not read the words "SUCCESS" as evidence of completion; it reads the state's structure, and foreign text breaks the pattern it associates with genuinely finished pages. Since the forged reward line is *page content*, the same closure covers the mechanical gates' forgery surface. Scope caveat recorded: the pool is mid-task states (clean p far below τ\*); the measured claim is that forgery cannot move the dial UP from incomplete — it moved it down, uniformly.

**Entry R — the mechanical brake arm: BRAKE DELIVERS TIME (Sep 28).** The last experiment, registered before any v8 data: the guarded serving (chain kept, **dial removed by config**) plus a 100% mechanical stuck-loop brake — entry P's label rule evaluated on the wire *before every model call* (same action signature ≥3× in the trailing 8 acting turns with a bit-identical page snapshot ⇒ the episode terminates with a frozen note and zero tokens spent; a braked failure gets exactly ONE retry on a fresh seed). Same 168 pages, instance success = any session raw > 0, instance wall = sum of its sessions. **PRIMARY (beat vanilla completions by ≥ +0.10) failed**: brake **27/168 = 16.07%** vs vanilla 17.26% (gap −0.0119). The retry is the honest headline: **1 success in 20 retries** — when a frozen worker is stuck enough to trip a loop breaker, a fresh page almost never rescues it; the stuckness lives in the model, not the instance. **SECONDARY passed**: instance wall mean **266.1 s vs guarded 339.8 s (ratio 0.78, bar ≤ 0.8)** at 16.07% ≥ the 10.69% success floor. Descriptives: live fire rate 16.1% (the pre-sweep zero-cost replay over v7 transcripts predicted 16.7%, parity 837/837 acting turns with the entry-P corpus); 8 of the 27 fires ended sessions whose page reward was **already banked** (post-success "victory laps"); cap-exits fell 45% → 26%. Verdict: the brake is a validated **clock-saver, not a solver** — paired +4 successes vs the guarded stack it replaces, at 0.78× the wall, and no completion edge over the bare model. The program closes with two deployed-grade instruments (honest-verifier dial, waste-cutter brake), one validated data asset (the label factory), and two honest negatives (verification and retries don't buy completions on a frozen 12B).

### Outside the lab: three independent sources converge (Sep 27)

- **"Just Ask Jev" (ICLR 2027 submission, [arXiv:2609.29429](https://arxiv.org/abs/2609.29429))** benchmarks the *commercial* Jev across 44 alignment benchmarks: median AUROC 0.886 zero-shot and pooled ECE 0.047 — but the **per-distribution median ECE is 0.168** (a base-rate mismatch, not a ranking problem), label-free correction fails, ~10 own-distribution labels are the cheap fix, and per-generator score-shape drift means every model swap needs a calibration refit. Each of those independently reproduces a measured negative in this repo: thresholds never transfer; calibration is per-distribution or it lies; the dial is refit per backbone by design. The paper also validates entry JO from the other side — excellent on-benchmark calibration that does not survive contact with a new distribution.
- **Two practitioner write-ups on Jev-in-agent-loops (Sep 2026)** converge on the same architecture from the field: the decision layer must sit *beside* the loop (routing decisions back through the generating model re-reads the full context — the "cache tax" makes pure routing ≈ 2/3 the cost of never routing); the decider cannot be the prover ("a confident answer cannot prove that a file was saved" — evidence in, judgment out); deploy shadow-first and switch over only when confidence bands match on your own traffic; and stuck-detection is the highest-value monitor to build next. The entry discipline *is* the shadow-first pattern; entry P *is* the stuck detector; and the write-ups' adversarial caveat (a page can talk a classifier into a false "done") is a registered concern for any future controller-mode entry.

---

## The conclusion, in plain language (Sep 27, finalized Sep 28)

**We built a cheap "supervisor" for a worker (one frozen AI model), and the honest conclusion is: the supervisor can reliably tell when the worker is lying about being finished — but catching the lie and sending him back to work doesn't get more work done. It actually got slightly *less* done, at twice the cost.**

1. **The worker's words tell you nothing; his "body language" tells you almost everything.** When the model says "I'm done," its text is just noise for judging whether that's true. But a tiny math probe reading the model's internal state — its hidden fingerprints — catches a false "done" about 9 times in 10, and held that up on three fresh test batches. Nobody trained the model to expose this; the information was just sitting there.
2. **A good lie detector still doesn't buy you success.** In the final head-to-head (336 paired tasks): supervised **23** vs unsupervised **29** — worse, at 2× the time. Most of the tasks the worker fails, he fails because he *can't do the task*, not because he's overconfident; sending him back just gives him more time to fail. (Silver lining: the supervised worker almost never walks away claiming a false victory.)
3. **The supervisor earns its keep on a different job: catching wasted time.** A second tiny head spots "the worker keeps clicking the same button on an unchanged page" — spinning in circles. It passed its quality bar on the first try, and it fires exactly when sessions get stuck, including ones that already succeeded and then kept pointlessly going. A clock-saver, not a success-saver.
4. **The expensive part is the training examples, not the cleverness.** A ready-made "decision AI" from the open-source world (Jev-Omni) scored *worse than a coin flip* on our tasks, despite great scores on its own home turf. The lesson: you can't borrow judgment. You have to label real examples from your own job — and our pipeline of automatically labeled examples is the actual asset.
5. **The last two experiments closed the remaining questions.** A cheater's test: we let a webpage literally forge the "reward: 1.0" line the checker reads — the lie detector was *not* fooled; forged text actually pushed its suspicion the other way (it reads the room's structure, not the posters on the wall). And the brake itself was built and run as a full experiment (168 tasks): it kept completions level with the bare model (16.1% vs 17.3%) while cutting the guarded stack's time by 22% (266s vs 340s average) — a validated clock-saver. Its one sober discovery: giving a stuck worker a *fresh copy* of the task rescued almost nothing (1 in 20) — the stuckness is in the worker, not the page.

**Bottom line:** the reflex layer works for cheap pointing decisions and for detecting lies and loops, but it can't raise a worker who lacks the skill. The brake-pull we promised actually ran — and delivered exactly the time savings, and exactly not the completions, that the honest ledger predicted. The long-term bet stands: build this judgment *into* the model itself rather than bolting it on.

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
├── proxy/                  # omp_proxy.py — the OpenAI-compatible Rewire layer (v3.17.2) + dial_enforce.py
├── eval/                   # benchmark harnesses, datasets, frozen configs, reports
│   ├── datasets/           #   toy_mcqa.jsonl · webreplay_v1/ (with MANIFEST)
│   └── reports/            #   measurement records: PoC eras + entry_h…entry_p (dial, enforce, A/B, stuck head)
├── configs/                # model / thresholds / benchmark configs
├── examples/               # minimal DECIDE example
├── tests/                  # pytest suite
├── scripts/                # env check · optional host-sync helper
├── index.html              # the interactive story page (the Pages site root)
├── poc_report.html         # the full audit report (self-contained)
├── POC_CONCLUSION.md       # structured conclusion: 7 PoC eras + the verifier-dial arc + the dial goes live
└── IMPLEMENTATION_PLAN.md  # the 2,100-line pre-registration ledger — the complete audit trail
```

## Method (why the numbers are believable)

- **Pre-registration**: framing, evidence rules, and pass/fail thresholds were written *before* each run; adversarial sanity gates were allowed to kill designs (and killed five).
- **One pass per instrument**: no reruns-until-significant; every deviation is disclosed in the ledger with dates and PIDs.
- **Bit-exact reproduction**: scorers reproduce across passes to 0.0000 (316/316 probes, sha256-pinned score files).
- **Blind adjudication**: outcomes graded from transcripts and environment records, never exit codes.
- **Constants as adversaries**: every learned component is compared against the constant predictor on its own data — the sidecar that merely matched the constant (82.9%) was killed.

Full methodology, per-experiment logs, and the decision record: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) · distilled findings: [POC_CONCLUSION.md](POC_CONCLUSION.md).

## Where this goes

The PoC's last finding is its most provocative: the reflex works when it's *told* exactly what to score, and the model's untrained self-judgment carries no signal — in its words. Its state vectors are another matter: the verifier dial (above) cleared the written acceptance bar on fresh sessions and now ships as the GATE's completion-claim verifier. Enforce mode shipped (entry N — the dial now vetoes unverified claims, cap 2, nothing leaked to the agent). **The decisive A/B answered, honestly negative** (entry O — the dial verified truth at 0.89 AUROC yet the guarded stack completed *fewer* tasks than vanilla, 13.7% vs 17.3%, at 2× the wall time); the **stuck/loop head passed its gates on the first pass** (entry P — 0.9621 AUROC on the loop label); **the adversarial concern closed** (entry Q — forged page text, including the literal reward line, cannot move the dial up; it moves p down ~30×); and **the brake ran as the final experiment** (entry R — the mechanical stuck-loop terminator + one fresh retry on the same 168 pages: completions level with vanilla at 16.1% vs 17.3%, wall time 0.78× the guarded stack — a validated clock-saver, with the sober finding that fresh-page retries rescue 1/20 of stuck cases; the stuckness is in the frozen model). What remains open: a **τ rebalance** (the frozen threshold sits at a conservative corner — 46% of genuine completions are sent back), **mid-thought checkpoint reads** (score a thought *while* it happens, not only at its end), and — the lesson entry R priced — competence work on the actor itself, since verification, retries, and loop-breaking all bought time and honesty but no completions. The endgame direction stands: **native decide** — decision heads trained into the backbone (typed outputs, calibrated confidence as a first-class objective), with the contract discipline, per-model calibration refits, and the written bar carried over as the spec any native implementation must beat — a bar that now has a measured proof it can be met, *and* a measured proof that skipping the per-distribution label cost fails (entry JO). Mechanical gates first, model reads only where deterministic signals can't see.

## Credits

- [System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — TypeSafe AI, for the framing and the provocation.
- [Contrastive-LM / CLM](https://github.com/Contrastive-LM/CLM) (Apache-2.0) — the frozen-encoder-plus-tiny-head verifier class that inspired the dial arc (entries H–M).
- [djev-run](https://github.com/taeold/djev-run) and vLLM's System One diffusion server (PR #57250) — the open-source System One wave this project responds to differently.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [Qwen](https://huggingface.co/Qwen), [MiniWoB++](https://miniwob.pluslab.org/) — infrastructure and benchmarks.

## License

[MIT](LICENSE)
