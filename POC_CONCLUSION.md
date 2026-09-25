# REFLEX/DECIDE PoC — Structured Conclusion

**Window:** 2026-09-17 → 2026-09-25 (the seven-day PoC, then the verifier-dial arc) · **Ledger:** `IMPLEMENTATION_PLAN.md` (1,600+ lines, every entry pre-registered or adjudicated) · **Report:** `poc_report.html` (covers the PoC eras; entries H–M live in the ledger)

---

## 0. TL;DR

We asked whether a calibrated **read-and-route layer over one frozen open-weights model** — no fine-tuning, inference-time only — beats that same model's vanilla configs at browser/computer use. **Answer: yes on offline benchmarks (decisively, against the model's own default and its best vanilla config), no measured gain end-to-end on a real agent harness over a real benchmark (a tie), and the reason is now precisely known:** the decision layer's reads are reproducible but carry no outcome signal (coin-flip discrimination, worse-than-constant calibration), and the failure modes that actually decide episodes are caught perfectly by a mechanical check the environment hands out for free. The program ends with a recorded strategic verdict: keep the skeleton, wire the mechanical gate, suspend the LLM reads behind an explicit bar, and re-run one decisive A/B.

**Addendum (Sep 23–25, entries H–M, inspired by [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)):** the suspension bar was then met — by a different instrument. A tiny calibrated head over the frozen model's **state vectors** (not its words) cleared the same written bar on fresh sessions (AUROC 0.9889 on a Qwen3-8B encoder; 0.8867 on the chain's own Gemma-12B via the stock embeddings endpoint; **0.9011 / Brier 0.0861 vs constant 0.1512 on 168 brand-new live sessions**) and shipped as the GATE's completion-claim verifier. A follow-up attribution entry exonerated quantization (Q4 ≈ bf16). See Era 8.

---

## 1. The question, and the claim style

> *Can this PoC be better — speed and accuracy — at computer or browser use than vanilla Qwen (later: vanilla Gemma 12B), without competing with frontier models?*

Three pre-registered gates (MCQ, web replay, tool calls), fairness rules fixed in advance (identical prompts/gates across arms, calibration on disjoint splits, per-item McNemar tests), and a self-vs-self claim discipline: **every comparison is the same checkpoint against itself under different runtime policies.** Nothing was ever attributed to weight changes — the weights were never touched.

## 2. What was built

- **The runtime (local, 4B era):** DECIDE (logit reads over candidate answers — zero decode steps), THINK (native thinking, spent only where it pays), GATE (temperature-scaled confidence with fitted acceptance thresholds), deterministic escalation policy. Batched read scoring with a bit-parity gate (5× latency).
- **The chain proxy (harness era):** an OpenAI-compatible shim that rides an off-the-shelf agent harness (omp) with **zero harness changes** — only the base URL changes. Evolved v3.2 → v3.16.2: decide reads over HTTP, gates refit per backend, completion self-review frames (config-gated), coverage fix, annotate posture.
- **The verification program:** binary done-read, 4-level rubric read, multi-read agreement — each pre-registered with frozen bars, one pass each, order controls, and a bit-exact reproduction check culture.
- **The measurement discipline itself:** pre-registration before every read, sanity gates with the power to kill designs, disclosure of every deviation. This survived contact with reality better than any single mechanism.

## 3. The seven eras, with their headline numbers

**Era 1 — Offline gates (Sep 18–19).** All three gates passed. MCQ: hybrid-with-re-decide **0.856 vs 0.764 no-think vs 0.490 think-always** at 0.26 s vs 17.3 s. Web replay: gated decide **0.356 vs 0.258** element accuracy (p≈8×10⁻⁹). Tool calls: chain **0.320 vs 0.274** one-shot (p=0.0033), 0 parse failures by construction. Lessons baked in: context parity (op accuracy 0.507→0.843 when the factor chain sees what the end-to-end prompt sees), the degenerate-baseline trap (one-shot JSON is 99.6% CLICK), and "the default is the worst config."

**Era 2 — Latency and routing (Sep 19).** Batched reads: 1.70 → 0.34 s p50 at statistically identical accuracy, gated on bit-parity. Mode routing measured and **settled against the reads**: even an oracle mode-router loses action accuracy (−2.8) because declaring the output contract is what enables the conditioned reads — the harness routes, the reads decide *inside* the contract.

**Era 3 — The live harness (Sep 20–21).** Same omp, same 4B, only the base URL changes: vanilla **0/5** vs chain proxy **3/5 → 4/5 → 5/5** (seven root-caused chain defects fixed structurally, none by prompting — including a click cell that was an uninvoked function expression). Universal tier: one proxy serves browser *and* coding. Backend swap to Gemma-4-12B on llama.cpp: 12B native **0/5** — the chain, not the scale, was the difference-maker; chain-12B back to 4/5 after four more defects, each proven at the log level.

**Era 4 — The honest benchmark (Sep 21).** MiniWoB++, 96 episodes, environment-scored: **chain 11/48 vs vanilla 10/48 — a statistical tie** at 2.1× wall time. Transcript forensics found the mechanisms: ANSWER stand-down (79% of reads), wrong-is-terminal asymmetry (29 confident terminal mistakes vs vanilla's 18 inert stumbles), matcher vocabulary gap (140 errors), refusal-as-payload. Two bug-grade fixes shipped (v3.14.4); the tempting MiniWoB-specific recalibrations were **rejected as overfitting in spirit** — the discipline that defines the rest of the program.

**Era 5 — The review program (Sep 21–22).** Completion self-review frames, HISTORY frame, voting rules, single-model framing backtest, held-out real-site sweep (the coverage fix works in the wild), sidecar kill-test (AgentJev-0.6B's 82.9% = constant predictor, killed by audit). Most of it deployed nothing; all of it recorded why.

**Era 6 — The measuring stick and entries A–G (Sep 22–23).** First the stick was fixed: **server read-corruption root-caused** (prompt/slot cache epochs + concurrent batching distort forced logprobs) → verified config `-np 1 --no-cache-prompt`, bit-exact after. Sequential dispatch: exact parity, +13.7% wall — not deployed. Evidence-first reorder: FAIL. Then the properly-sized judgment (entry E, 168 fresh sessions, 316 probes): binary det 88.8%/FA 83.3% (floor disease at scale), rubric FA **0/24** but det 11.9% — frozen bar failed → INDETERMINATE by cell size, redirected to "low-FA confirmer." Entry F: **order-robust** (6.3% flip; the 0-FA property replicates under reversed scale). Entry G: sampling-noise design killed by its own sanity gate (**temperature is not applied under grammar** in this llama-server build — zero probes wasted), amended to 5-paraphrase prompt-noise; P1 reproduced entry E **bit-exactly 316/316**; the gate **NOT EARNED** — framing moves the read ~1.5 points while outcome moves it 0.15–0.4; the read measures the sentence, not the page. **The confirmer role is closed.**

**Era 7 — The closing arc (Sep 23).** The omp context-shaking storm that destroyed 14 transcripts root-caused to the read-only base config's `compaction.thresholdPercent: 30` (maintenance fires at exactly 39,321 tokens), with a launch-time `--config` overlay as the fix. Autopsy of the 20 zero-pass tasks: one root disease — **the model cannot read the reward line** (115 failing sessions show `Last reward: −X` in the snapshot the model just read; 55 end with success claims anyway); a five-line mechanical brake catches all of them at zero false vetoes on the passes. JevBench public-suite calibration: documented dead end (not public, wrong backend class), substituted with the same measurement on our own stack: **every read worse than a constant predictor** (Brier 0.131–0.373 vs constant 0.081; AUROC 0.51–0.57; confident tail inverted). Strategic verdict recorded (below).

**Era 8 — The verifier dial (Sep 23–25, entries H–M; inspired by [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)).** The strategic entry suspended all LLM reads behind the explicit bar — then a public project showed a verifier class built on frozen encoder **state vectors** instead of words. Each rung pre-registered against the same frozen bar: **entry H** — CLM's own off-the-shelf contrastive heads fail (ranking is real at 0.99 out-of-session, calibration is indefensible: ECE 0.24, Brier over constant; the conjunction gate fired as designed); **entry J** — the same frozen Qwen3-8B encoder plus our 2-parameter Platt recalibration head clears Stage 1 (0.9937/0.0169) and then Stage 2 on 168 fresh sessions: **AUROC 0.9889 / Brier 0.0374 vs constant 0.0982 — the verifier role is earned**, the first read in the program's history to clear the bar out of sample; **entry K** — the recipe transplanted to the chain's own Gemma-4-12B through the stock `/v1/embeddings` endpoint passes with zero second-model residency (0.8867/0.0653), and leaves a standing instrument rule (one text per embeddings request — batched multi-input vectors on this server build are ill-defined, up to cos 0.87 apart); **entry L** — the integration registration: the head exports as pure data (3,840 weights + Platt + τ* = 0.3261) and clears its live confirmation on 168 brand-new sessions: **AUROC 0.9011 / Brier 0.0861 vs constant 0.1512 — GATE PASSED, the dial ships** (accept precision 93.3%, 113/114 false completion claims caught, 46% of genuine completions conservatively rejected at the frozen τ); **entry M** — the attribution diagnostic: Qwen3-8B Q4_K_M through the same serving stack reproduces bf16 (0.9868/0.0344 vs 0.9889/0.0374), so quantization is exonerated and the encoder gap is Gemma's own representation. The reversal, in one sentence: **in words the model's self-judgment carries no signal; in state vectors it carries nearly all of it.**

## 4. What stands as confirmed wins

1. **Reads beat generation for selection** — twice, independent designs, p≈10⁻⁷–10⁻¹¹ (offline replay).
2. **The default (think-always) is the worst agentic config** — for this model class, decisively.
3. **Context parity** is the single most transferable engineering lesson of the PoC.
4. **Confidence ranking transfers even when thresholds don't** — errors are the lowest-confidence events (live: both errors caught at a 10% ask rate, precision 1.0).
5. **The chain proxy was the difference-maker on the live harness** (0/5 → 5/5 at 4B; 0/5 native 12B → 4/5 chained).
6. **The measurement discipline works**: it caught the server corrupting its own reads, killed a degenerate sidecar at audit, killed the sampling-noise design before it wasted a pass, killed an uncritical CLM transplant at its calibration leg, and turned "the chain is worse" into "it's a tie, here's why."
7. **Reproducibility at the bit level is achievable** — verified server config + order controls + P1 316/316 bit-exact, extended by entries H–M's bit-exact determinism rechecks on every embeddings pass.
8. **The verifier dial (Era 8)** — a trained readout finally cleared the frozen bar on fresh sessions, ships as pure data over the engine's own embeddings endpoint, and its threshold story (46% conservative rejections) is recorded as a policy dial, not a defect.

## 5. What stands as measured negatives

1. **No end-to-end benchmark gain**: MiniWoB tie (11/48 vs 10/48), with the asymmetry that the chain's errors are confident and terminal.
2. **No verifier in the model's words**: six entries of candidate LLM verifiers (binary, rubric, multi-read) — none both detects and stays safe; the one real property (low-FA at strict-LOW challenge) never transfers to approval. *(Superseded Sep 25, Era 8: a trained state-vector dial — a different instrument, same frozen bar — cleared it and ships.)*
3. **No signal in the words-channel reads at all, as it turns out**: framing moves them 10× more than outcome; discrimination ≈ coin flip; calibration worse than a constant; the confident tail inverted. *(The state-vector channel proved highly readable — Era 8.)*
4. **The thresholds never transfer across domains** — every gate must be refit per domain (measured again and again, by design).
5. **Mode selection cannot live in the reads** — the harness must declare the contract (oracle-router result).
6. **The external calibration benchmark is unreachable** (JevBench suite not public; API needs a diffusion vLLM branch) — documented, not deferred.
7. **The environment's own mechanical signals beat every LLM read** at the job that decides episodes.

## 6. Durable assets

- `poc_report.html` (full report with every number and its provenance), `IMPLEMENTATION_PLAN.md` (the complete pre-registration/adjudication ledger, synced to the host).
- The chain proxy codebase (v3.16.2: gates, frames, universal tier, config-gated review, onboarding probe `eval/onboard_backend.py`).
- The verified-numbers server recipe (`-np 1 --no-cache-prompt` + tunnel discipline) and the read-stability checks.
- The MiniWoB corpus: 168 v3 sessions with page-reward ground truth, 316 balanced probes, five frozen score files (binary, reversed, rubric, entry-E rubric, paraphrase) — sha256-recorded.
- The verifier-dial artifacts (`eval/reports/entry_h…entry_m/`): frozen states/folds, embeddings manifests with bit-exact rechecks, the exported readout (`entry_k_readout.json`: 3,840 weights + Platt + τ*), and the entry-L fresh-sweep confirmation pool — all sha256-recorded.
- The calibration harness (`calibration_v3.py`) — now the **standing acceptance test** for any future read.
- The failure forensics (`autopsy_features.json`, `autopsy_signatures.json`) and the shake root-cause + overlay fix.

## 7. The decision record (strategic entry, 2026-09-23)

**Does the chain earn its complexity? Not in its current form.** The LLM-read layers carry no measurable benefit and real cost (+13.7% wall from sequential dispatch alone; two defect classes were chain-borne); the working parts are the non-LLM parts. Four-point plan, recorded so the next sweep starts from it:

1. **Now, harness only:** teach the reward line in one sentence; mechanical finish-gate (veto success while the snapshot shows a negative reward, re-open the episode); assert-zero-shakes overlay on every launch.
2. **The one A/B that matters:** chain + mechanical gate vs vanilla + the same prompt fix. If the tie survives, the LLM decision layers come out of the browser-task path.
3. **The bar for any future read (explicit, reversible):** AUROC ≥ 0.75 AND Brier better than the cell's constant on held-out session truth, pre-registered, order-controlled. Nothing measured comes close (0.51–0.57).
4. **Keep unconditionally:** decision log, gate skeleton + ask-user ranking, verified server config, pre-registration methodology, the calibration harness.

**Addendum (Sep 25).** Point 3's bar was met — by an instrument class the plan didn't name: a calibrated linear dial over frozen-encoder state vectors (entries H–J earned it, K made it native to the chain's own engine, L confirmed it live on 168 fresh sessions and shipped it, M attributed the remaining encoder gap to representation, not quantization). Points 1, 2, and 4 stand unchanged; the dial's enforce-mode routing is the next pre-registration.

## 8. Honest limits

Offline-replay evidence was always the accuracy backbone; live-browser results are feasibility-scale (n=20, 2 errors), not benchmarks. Everything is one-model-vs-itself — better than its default, never frontier-class. Value accuracy was measured on copy-from-task strings only. The machinery ports to any open-weights decoder LM; **none of the numbers port** (every constant was fit, not assumed). And the endgame analyses (autopsy, calibration, strategy) are post-hoc descriptives — recorded as such, never converted into deployment decisions after the fact.

---

*One line, if only one line may be kept: the PoC proved the routing mechanism works where it can read real signal, measured precisely where the signal ends, and ended by pointing the next unit of effort at the harness floor instead of the model ceiling — and the dial arc then found the missing signal in the one place nobody had thought to read: the model's own state.*
