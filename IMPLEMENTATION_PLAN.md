# Implementation Plan — Qwen3.5-4B Hybrid Inference PoC (rev. 3)

Companion to [qwen35_hybrid_reflex_decide_think_poc_v2.md](./qwen35_hybrid_reflex_decide_think_poc_v2.md) (rev. 3).
Goal: execute the **four-week core** (Phases 0–3) that answers the headline question — *does an explicit calibrated DECIDE primitive beat Qwen3.5's native adaptive-thinking mode on accuracy-per-latency?* — then decide on conditional phases based on the result and the pre-registered kill criteria (doc §18.1).

---

## 0. Environment (verified 2026-09-17)

```text
Host:      <gpu-host>  (SSH, key auth)
GPU:       RTX 5070 Ti, 16,303 MiB (Blackwell, sm_120)
Driver:    610.43.02, CUDA UMD 13.3
OS:        WSL2 (Ubuntu-family, kernel 6.18)
RAM:       30 GB        Disk: ~690 GB free
Docker:    available; containers with GPU reservations:
           <redacted-1>, <redacted-2>, <redacted-3>
Baseline:  ~4.1 GB VRAM in use ⇒ ~12 GB free (enough for BF16 inference)
```

Notes:

- `nvidia-smi` is not on the default WSL PATH — use `export PATH=/usr/lib/wsl/lib:$PATH` or the full path.
- **Containers are left running by default; only restart (stop/start) them when a specific running phase actually needs the VRAM** — never preemptively. Baseline ~12 GB free covers the four-week core (BF16 inference of the 4B is ~9.3 GB); expect to touch containers only for conditional-phase work (D4/vLLM serving with larger batches). When a phase does need headroom, stop the minimum set, and restart those containers as soon as the phase's run completes:
  ```bash
  ssh <gpu-host> 'docker stop <redacted-1> <redacted-2> <redacted-3>'
  # restore immediately after the run:
  ssh <gpu-host> 'docker start <redacted-2> <redacted-3> <redacted-1>'
  ```
  These are other people's workloads — never remove containers or images, and never leave them stopped overnight.
- Layout: development on the local workstation; training/inference runs on the GPU host. Keep the repo synced via git (no NFS assumptions).

---

## Step 0 — Toolchain pinning *(day 1, blocking)*

Everything downstream fails if sm_120 support is missing. Verify before writing any PoC code.

| # | Task | Acceptance |
|---|------|-----------|
| 0.1 | Create venv on GPU host: `python3.11 -m venv ~/hybrid-env` | `python --version` ≥ 3.11 |
| 0.2 | Install Blackwell-enabled PyTorch (cu128+ wheel or newer): `pip install torch --index-url https://download.pytorch.org/whl/cu128` | `python -c "import torch; print(torch.cuda.get_device_capability())"` → `(12, 0)` and a successful small CUDA matmul |
| 0.3 | Install stack: `transformers accelerate peft bitsandbytes numpy pydantic pytest` | `import bitsandbytes` succeeds on GPU (run its CUDA check) |
| 0.4 | Verify vLLM with Qwen3.5 hybrid support: `pip install vllm` (or official container), serve a hello-world completion | one completion returns; `--enable-prefix-caching` accepted |
| 0.5 | Download model: `huggingface-cli download Qwen/Qwen3.5-4B` | weights present; ~9–10 GB BF16 |
| 0.6 | Load model in HF Transformers; run one text generation and one image+text generation | generation works, VRAM ≤ ~11 GB |
| 0.7 | Record versions in `configs/versions.txt` (torch, transformers, vllm, driver) | file committed |

**Gate:** 0.2 or 0.4 failing ⇒ fix toolchain before anything else; do not proceed on CPU.

---

## Step 1 — Phase 0: validity harness *(week 1)*

Build the measurement skeleton the whole PoC stands on.

| # | Task | Acceptance |
|---|------|-----------|
| 1.1 | Scaffold repo per doc §20 (`hybrid/`, `eval/`, `configs/`, `tests/`, `examples/`) | `pytest` runs green on empty suite |
| 1.2 | Latency telemetry wrapper (prefill_ms / scoring_ms / total_ms, CUDA-event based, P50/P95 over ≥50 reps) | telemetry example produces a report |
| 1.3 | **Scoring-position pinning**: one chat template constant, `enable_thinking: false`, documented logits read-point (assistant-turn final position). Module: `hybrid/scoring/readpoint.py` | unit test asserts identical logits for identical pinned input across two runs |
| 1.4 | **Scoring-position sensitivity report**: repeat one MCQA cell with (a) thinking off, (b) thinking on reading after `</think>`; report distribution shift | `eval/reports/scoring_position.md` committed |
| 1.5 | **Pre-mask mass instrumentation** (doc §9.1): fraction of raw next-token probability on the candidate set, recorded per decision | metric appears in every decision record (`DecisionResult`) |
| 1.6 | Baseline A (generation) + Baseline B (forced THINK) harness cells on a 20-item toy set | both produce answers + telemetry |
| 1.7 | **Native auto-mode cell (Baseline G)**: stock template, model-default mode, same telemetry | G runs on the toy set; latency/tokens recorded |

**Gate (kill-criteria 5 preview):** if pre-mask mass on toy tasks is routinely < 5%, stop and reformulate candidate presentation before Phase 1.

---

## Step 2 — Phase 1: DECIDE v0 + baseline table *(week 2)*

| # | Task | Acceptance |
|---|------|-----------|
| 2.1 | D1 restricted first-token logits (gather candidate rows — not full-vocab softmax) | 2-choice toy decisions correct; single forward per decision |
| 2.2 | D2 sequence log-likelihood with `α ∈ {0, 0.5, 1}` normalization + **PMI/DC variant** (content-free context) | scores match a hand-computed example; PMI variant flagged in `scoring_method` |
| 2.3 | `decide(question, candidates)` API + `DecisionResult` (doc §22) incl. `premask_mass` | example script runs end-to-end |
| 2.4 | Eval set v1: 300–1,000 items across 2–3 doc §15 domains (one text intent, one ambiguity-escalation, one visual MC); JSONL with gold labels; ~15% held out for calibration | dataset card committed; balance stats reported |
| 2.5 | Baseline C (generated MC) + Baseline D (DECIDE only) cells | runner produces the full table |
| 2.6 | **Baseline table v1**: {A, B, C, D(D1), D(D2), **G**} × {accuracy, p50/p95 latency, tokens, ECE, CAA} | `eval/reports/baseline_table_v1.md` committed |
| 2.7 | Order-shuffle control on one cell (doc §16.2) | selection-stability number recorded |

**Decision point (end week 2):** read the table against kill criteria 1 (G parity) and 4 (D2+correction dominance). If D2-corrected already matches plausible head results, pre-cancel Phase 5B.

---

## Step 3 — Phase 2: state integration + filler control *(week 3)*

| # | Task | Acceptance |
|---|------|-----------|
| 3.1 | Session state: `<DECIDE_RESULT …/>` injection back into context; event log (doc §23) | session trace replays from event log |
| 3.2 | `THINK → DECIDE` and `DECIDE → GENERATE` loops with budgets (SHORT/MEDIUM) | one continuous session runs on the eval set |
| 3.3 | **Neutral-filler control** (doc §16.2): same loop with THINK replaced by content-matched filler paragraph | both arms run on the ambiguity subset |
| 3.4 | Analysis: Δ accuracy / Δ calibration, real THINK vs filler | `eval/reports/filler_control.md`; differential is the S5 verdict |

**Gate (kill criterion 2):** filler ≈ real THINK ⇒ H4 falsified; record and continue to calibration with the claim withdrawn.

---

## Step 4 — Phase 3: calibration + escalation *(week 4)*

| # | Task | Acceptance |
|---|------|-----------|
| 4.1 | Temperature scaling fit on calibration split, **two+ templates**; report transfer ECE | `calibration/temperature.py` + transfer report |
| 4.2 | Content-free prior correction at the distribution level (§11) | ECE before/after on held-out template |
| 4.3 | Escalation policy: confidence/margin/entropy thresholds tuned on calibration split for target Useful Escalation Rate | thresholds in `configs/thresholds.yaml`, tuned by sweep, not hand-picked |
| 4.4 | Full Experiment E run vs **Baseline G** on the eval set: accuracy, RAR, UER, CAA | `eval/reports/headline.md` — the thesis verdict |
| 4.5 | Kill-criteria review: write the §18.1 verdict for criteria 1–5 | one page, committed, before any conditional phase starts |

**Gate:** this is the PoC's answer. Conditional phases (below) start only after 4.5.

---

## Conditional phases (post-week 4, only on positive/interesting verdicts)

| Phase | Trigger | First task |
|-------|---------|-----------|
| 4 — REFLEX | S8 positive or marginal | fixed policy set; REFLEX-vs-DECIDE latency; Experiment F vs G |
| 5 — D4 parallel scoring | D2 is on the headline path and candidate scaling matters | fork KV+GDN state in vLLM/SGLang; batched-suffix oracle for correctness; **then** the 3→100 candidate scaling study |
| 5B — D5 option-attention head | Q11 alive (heads not dominated in week 2) | head training on frozen features; in-domain + target-disjoint eval (jevlike OOD protocol) |
| 6 — Multimodal validation | S7 attempted | image → DECIDE/REFLEX without captioning; visual MC + UI-screenshot domains |
| 7 — QLoRA fine-tuning | confidence separation is the bottleneck | 10–50K targeted examples; never full-weight |

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| sm_120 quant-lib gaps (bitsandbytes/torchao) | medium | Step 0.3 gate; BF16 fallback for inference-first phases |
| WSL2 VRAM fragmentation from co-tenant containers | medium | stop/restart procedure (§0); run big cells when host is quiet |
| Hybrid GDN state fork harder than expected in vLLM | medium | D4 is conditional; batched-suffix scoring is the correct fallback |
| Thinking-template contamination of logits | high | Phase 0 pins read-point; sensitivity report is a committed deliverable |
| Native auto mode unbeatable | unknown | kill criterion 1 pre-registers the pivot (calibration/audit tooling) |
| Eval set too small for ECE stability | medium | 500–1,000 calibration items; report bootstrap CIs on ECE |

---

## Deliverable cadence

- Every step lands a committed report under `eval/reports/` — the PoC's product is the reports, not the runtime.
- Week 2 table, week 3 filler-control differential, week 4 headline verdict are the three checkpoints where the project direction is explicitly re-decided against doc §18.1.

---

## Progress log

### 2026-09-17/18 — Step 0 complete, Phase 0 running

Toolchain (all gates PASS):

- No system pip and no sudo on the GPU host → installed **uv** in user space. First venv used system python3.12, which **lacks Python.h** — this silently breaks all triton JIT (fla kernels, torch.compile). Fix: `uv python install 3.12` (standalone CPython with headers) and rebuild the venv on it. **Standing rule: always build venvs from uv-managed Pythons on this host.**
- torch 2.11.0+cu128 on sm_120, bf16 matmul OK; transformers 5.17.0 (has `Qwen3_5ForConditionalGeneration`; loads via `AutoModelForImageTextToText`).
- `flash-linear-attention` works (GDN optimized kernels); `causal-conv1d` still builds from source and is skipped (conv part runs reference torch — acceptable).
- Model downloaded via snapshot_download (8.8 GB; hf_transfer is deprecated → Xet transfers; unauthenticated ≈ 5–40 MB/s).
- transformers 5.x from_pretrained takes `dtype=`; loader has torch_dtype fallback.

Operational lessons:

- **Long remote runs must be launched with `nohup … &` on the host** — a dropped SSH session kills a foreground remote process (this killed the first Phase 0 run).
- `pkill -f <script>` from inside the ssh command string matches the command itself and kills the session — never do that.
- Device discipline: pinned prompts may be CPU (`pin_prompt`) or CUDA (`pin_after_think`) — scorers must normalize explicitly (fixed in `_candidate_logprob`).
- VRAM: model BF16 loads at ~9.6 GB peak — Phase 0–3 fits in the ~12 GB left by the co-tenant containers; no container restarts needed so far.

First findings (run 1, `~/phase0_run1.md` on host):

- Qwen3.5's template inserts an **empty `<think></think>` block even with `enable_thinking=False`** — the pinned read point is "after an empty think block". Documented; pinned and deterministic.
- **Pre-mask mass is bimodal** on bare MCQA: p50 0.023 (below the 0.05 validity gate, 6/20 pass) but max 0.994. The model either commits to the option format or wants to converse. Format anchoring goes on the Phase 1 template-sensitivity agenda.
- Uncorrected toy accuracy: D2 0.90, D1 0.80.
- Scoring-position sensitivity: 9/10 selection agreement over 10 items; the one flip was post-think turning a correct answer incorrect (toy-002) — the neutral-filler control is going to matter.
- Baselines A/B/G accuracies in run 1 are invalid (letter-extraction bug — models answer with option text). Fixed to choice-text matching with unparsed tracking; B budget raised to 768 (B/G hit their caps in run 1). Re-run in flight.

### 2026-09-18 — Phase 0 complete (run 2, `eval/reports/phase0_validity.md`)

**Baseline table (20 trivial-factual items, greedy):**

| cell | accuracy | mean latency |
|---|---|---|
| A no-think | **19/20** | ~0.25 s |
| B forced-think (768) | 8/20 | ~25.7 s |
| G native default | 7/20 | ~17.5 s |

- **Headline finding: the open-weights template has NO mode routing.** The chat template's only branch is `enable_thinking` (false ⇒ empty `<think></think>`; anything else ⇒ always think). The Qwen Chat `auto` mode is server-side. Baseline G redefined: native default ≡ think-always (measured ≡ B). Doc §2/§16/§18.1/§32 updated accordingly — **explicit routing fills a measured gap, it does not re-implement a shipped feature.**
- **Overthinking measured**: on easy items, forced/native thinking is ~100× slower and 2.4× less accurate than no-think. First empirical leg of the escalation thesis (hard-item counterpart comes with the Phase 1 ambiguity subset).
- Pre-mask mass bimodal (p50 0.023, only 6/20 over the 0.05 gate, max 0.994) — format anchoring on the Phase 1 agenda.
- Pinned read point deterministic (maxdiff 0.0); sensitivity 9/10 (the one flip: post-think flipped a correct answer to incorrect).
- D2 uncorrected 0.90 / D1 0.80 toy accuracy.
- Peak VRAM 9.62 GB — containers untouched throughout.

**Phase 0 verdict: harness valid, Phase 1 unblocked.** Next: eval set v1 (add ambiguity-escalation subset), D2 variants + PMI correction, baseline table v1 (per Step 2).

### 2026-09-18 — Phase 1 in flight; decision-cell results (from checkpointed records)

Infrastructure note: first full-table run crashed on a `statistics.fmedian` (py3.14-only) after the 2.5 h of generative cells — records were lost. baseline_table.py now checkpoints per cell (`baseline_table_v1_records.jsonl`, resume-safe) and runs decision cells first. Also: `causal-conv1d` still unbuilt (conv reference path only — generation ~23 s/item think-always; acceptable for now).

**Decision-cell results (n=500, 5 domains, pinned no-think read point):**

| cell | overall acc | p50 | ECE | notes |
|---|---|---|---|---|
| D1 | 0.534 | 155 ms | 0.077 | calibrated but crude; 0.31/0.29 on arithmetic (shared first tokens — documented D1 limitation, now measured) |
| D2 | **0.752** | 190 ms | 0.223 | ties C no-think generation (0.764 @ 278 ms) while faster, no parsing failures; per-domain: arc_easy 0.920, arc_challenge 0.750, arith_easy 0.740, arith_hard 0.570, intent 0.780 |
| D2P (pmi=1.0) | 0.648 | 371 ms | 0.125 | PMI *hurts* accuracy overall — and collapses arith_hard to 0.28 (below chance): the candidate prior carries real signal on numeric domains. Domain-dependent correction; never global. Improves ECE though (0.125 vs 0.223) |

Early conclusions:

1. **Bounded scoring (D2) is viable as the DECIDE primitive** — generation-level accuracy at lower latency, with probabilities for free. The thesis's core mechanism works on this backbone.
2. **D2 is overconfident (ECE 0.223)** → Phase 3 temperature fitting is load-bearing, as planned.
3. **Pre-mask gate does not transfer from D1 to D2 as a validity instrument**: ARC items pass the 0.05 gate only ~25% of the time yet D2 scores highest there (0.92/0.75). First-token mass is the wrong diagnostic for sequence likelihood. Revised role: premask gate applies to D1/R0; for D2 it is diagnostic only. Doc §9.1/§16.2 need this amendment (pending).
4. **Premask double-count bug**: for candidates sharing a first token (numeric choices), summed first-token masses exceed 1 (arith p50 up to 1.98). Fix: de-duplicate first tokens in the mass sum, or per-candidate mass. Trivial fix, queue for next sync.
5. G (re-run) 0.490 @ 17.2 s p50 vs C 0.764 @ 0.28 s — think-always loses across the mixed set, hard domains included; per-domain table pending.

Pending: generative cells re-run (A/C/G/B) → extras (alpha sensitivity, anchor, reshuffle) → chained filler control + calibration → headline runner with fitted thresholds.

### 2026-09-18 — Phase 1 COMPLETE (`eval/reports/baseline_table_v1.md`)

**Main table (n=500, 5 domains):**

| cell | acc | p50 | ECE | note |
|---|---|---|---|---|
| A no-options | 0.308 | 0.26 s | — | blind answering |
| C options,no-think | **0.764** | 0.29 s | — | best generation cell |
| G think-always | 0.490 | **17.3 s** | — | 60× slower, 27 pts worse than C |
| B forced-think (n=30) | 0.500 | 25.6 s | — | ≈ G, confirms Phase 0 |
| D1 | 0.534 | 0.16 s | 0.077 | calibrated but crude |
| **D2** | **0.752** | 0.19 s | 0.223 | ties C, faster, probabilities attached |
| D2P (pmi) | 0.648 | 0.37 s | 0.125 | PMI hurts globally |

**Per-domain (the regime story):**

| | arc_easy | arc_hard | arith_easy | arith_hard | intent |
|---|---|---|---|---|---|
| C | 0.980 | 0.920 | 0.670 | 0.590 | 0.660 |
| G | 0.440 | 0.380 | 0.440 | **0.790** | 0.400 |
| D2 | 0.920 | 0.750 | 0.740 | 0.570 | **0.780** |

- **Thinking earns its cost exactly where computation is needed**: G is the ONLY cell to crack arith_hard (0.790 vs C 0.590) while collapsing everywhere else. The escalation policy's job — route computation-shaped items to THINK — has a real target.
- **D2 is the best intent/action selector (0.780 vs C 0.660, G 0.400)**: bounded scoring beats generation precisely in the DECIDE use case it was designed for.
- ARC-hard (2021-hard) is easy for a 2026 4B: C 0.920. Difficulty axis came from arithmetic instead.

**Extras (ARC-Easy):**
- **Format anchor: premask p50 0.016 → 0.912, accuracy 0.920 → 0.970.** One instruction line fixes the Phase 0 pre-mask validity problem almost entirely → D2-anchored becomes the default DECIDE configuration going forward.
- Order-shuffle stability 94/100 — order robustness fine.
- Length-norm sensitivity: α=0 (raw sum) 0.980 / α=0.5 0.970 / α=1.0 0.920 — mean-logprob normalization costs 6 pts on ARC-Easy; α is a real hyperparameter (doc's `n^α` vindicated).

**Bugs fixed en route:** `statistics.fmedian` (py3.14-only) crash → per-cell checkpointing; arc_challenge/arc_hard label mismatch (records normalized, dataset relabeled); order-shuffle denominator; premask double-count for shared first tokens.

**Next:** filler control (running) → calibration → thresholds → headline S8/Q12.

### 2026-09-18 — Phase 2 COMPLETE (`eval/reports/filler_control.md`) — kill criterion 2 TRIGGERED

Neutral-filler THINK control (S5/H4): base / filler-think / real-think arms end in identical pinned D2 decisions.

- Real-think vs no-think differential: **−6.5 pts** (threshold: ±3). Thinking's *content* does not causally help re-scoring — on arithmetic it fixes 46.5% of previously-wrong items but breaks more previously-right ones.
- **§18.1 criterion 2 triggered.** Pre-registered pivot executed: escalation target becomes think-**generation** (confidence-gated mode selection); re-DECIDE kept as the preregistered comparison arm.

### 2026-09-18 — Phase 3 COMPLETE (`eval/reports/calibration_v1.md`)

- Deterministic cal/test split; fitted temperature **T = 0.5** on D2 raw scores; transfer ECE 0.223 → 0.210.
- Raw-score thresholds do NOT transfer (monotone transform preserves ranking, not thresholds — the mis-gated first headline run escalated 90%; archived as `headline_records_gateraw_ablation.jsonl`, the FrugalGPT cascade failure mode in miniature).
- Final `configs/thresholds.yaml`: accept if T-scaled confidence ≥ 0.70 AND margin ≥ 0.10; premask < 0.05 forces escalation; probe: arc_easy 2/8 escalate, arith_hard 6/8.
- Gate fires ~28% on cal; **18% (92/500) measured** on the headline run.

### 2026-09-18 — Headline S8/Q12 COMPLETE (`eval/reports/headline.md`) — final verdict

**Run-integrity note:** the first "final" run was confounded — checkpoint resume silently reused H_generate records produced by the *unanchored* pre-Phase-3 gate while H_redecide ran with the anchored gate (log line `checkpointed hybrid arms: ['H_generate']`). Detected post-hoc by comparing the arms' first-decision confidences (0.9998 vs 0.38 on identical items; mismatches exactly on ARC where the premask gate bites). Confounded records archived as `headline_records_unanchoredgen_ablation.jsonl`; H_generate re-run under the matched anchored gate (arms now escalate the *identical* 92 items).

**Final matched-gate table (n=500, greedy, RTX 5070 Ti):**

| arm | acc | p50 | CAA | escalation |
|---|---|---|---|---|
| A never-think | 0.308 | 0.26 s | 1.20 | 0% |
| C no-think generate | 0.764 | 0.29 s | 2.67 | 0% |
| G think-always | 0.490 | 17.29 s | 0.03 | 0% |
| **H-redecide** | **0.856** | 0.26 s | **3.31** | 18% |
| H-generate | 0.786 | 0.25 s | 3.13 | 18% |

**Decomposition (identical for both arms up to the escalation branch):** fast path = 408 items at **0.882 accuracy, p50 0.26 s** (anchored D2 + T-scaled gate). Escalated tail = the same 92 hard items (60 of them arith_hard): re-DECIDE scores **0.739**, think-generation only **0.359** (p95 9.5 s vs 19.9 s).

**Verdict against §18.1:**
- **S8 SURVIVED.** H-redecide dominates G on both axes (0.856 vs 0.490, 66× lower p50) and dominates C (McNemar 62/16 discordant, χ²=26.0, p≈3·10⁻⁷; +9.2 pts at lower latency). H-generate's +2.2 pts over C is *not* significant (48/37) — the headline is carried by re-DECIDE escalation.
- Criterion 1 not triggered (G matches nothing on accuracy+CAA). Criterion 2 triggered → pivot measured; nuance: the re-DECIDE win is a *policy* win — per the filler control its mechanism is the second read at a changed position, not thought continuity. Criterion 3 not triggered (ECE 0.210 transfers). Criterion 4 partially: PMI hurt globally, no trained heads needed at PoC scale. Criterion 5 measured-then-fixed via format anchor.
- **Honest caveats:** arith_hard pure accuracy still belongs to G (0.790 vs H-redecide 0.660) — the computation regime is real and the gate routes 60% of it but re-scoring doesn't reach G-level; escalated-latency p95 19.9 s (think-budget saturation); n=500, one backbone, greedy, single seed.
- **Open problem for any follow-on:** the routing signal separates knowledge-uncertainty from computation-needs poorly — arith_easy fast-path is near-ceiling while escalated arith_hard only recovers to 0.66. A learned router or per-domain gating (features beyond D2 confidence: candidate numeric structure, premask entropy shape) is the obvious next experiment; escalate-to-ensemble over candidates is the second.

### 2026-09-18 — Browser-use phase started (Gate B: offline web replay)

New phase: same-model comparison (hybrid RDT vs vanilla Qwen 4B) on computer/browser use. Not competing with frontier agents — the claim is S8-shaped: ≥ no-think accuracy at competitive latency, ≫ think-always on both axes.

**Infrastructure:** `eval/datasets/build_webreplay.py` streams Mind2Web (`osunlp/Mind2Web`, train) → `webreplay_v1`: **1801 steps / 250 tasks** (test 900, cal 901), per-step: task + action history + ≤20 enumerated elements (tag/text/attrs/DOM path/parent text, gold kept, order shuffled per item) + op + gold bid. `hybrid/actions.py` renders the DECIDE interface (numbered element lines); `eval/webreplay.py` runs arms D2E (anchored D2 + 3-way op read) / H_web (MCQ gate → think → re-DECIDE) / C_web (vanilla no-think, replies "number OP") / G_web (vanilla think-always), checkpoint-resume per item.

**First smoke failure (instructive):** naive candidate rendering (tag + own text only) collapsed BOTH arms to ~0.20 — gold elements are often information-free in isolation (`svg` icons with no text), and containers dumped whole page sections (near-duplicate lines). D2E's own confidence collapsed in parallel (p50 0.13). Fixed with DOM path + parent text + smarter text extraction + dedup → D2E 0.375 / C_web 0.275 on smoke. Candidate representation is the hard subsystem of web-DECIDE, not the read.

**Gate B interim results (test split, n=900, D2E & C_web complete):**

| arm | n | elem_acc | op_acc | p50 |
|---|---|---|---|---|
| D2E | 900 | **0.320** | 0.490 | 1.60 s |
| C_web | 900 | 0.258 | **0.816** | 0.26 s |

- **D2E > vanilla no-think on element selection: McNemar 118/62, χ²=16.8, p≈4×10⁻⁵.** Latency is 6× worse (1.6 s vs 0.26 s) because D2 re-prefills the prompt per candidate sequentially — batching is the known fix (20 prefills → 1–2 batched forwards).
- **C_web op_acc 0.816 vs D2E's separate 3-way op read 0.490** — op prediction conditioned on the already-chosen element is the wrong factorization to score; needs reordering (op first) or joint rendering.
- **H_web (MCQ gate + think→re-DECIDE) FAILS on web: aborted at n=68.** Escalation 71% (MCQ T=0.5/0.70 does not transfer — raw confidences compress with 20 similar-length options: conf AUC only 0.648, premask AUC 0.493 = useless here), p50 22.5 s, overall acc 0.279 < D2E 0.320. Decomposition: gate ROUTING works (accepted items 0.500 vs D2E's 0.450 on same items) but the escalation MECHANISM is harmful — think→re-DECIDE scores **0.188 on the tail vs 0.271 for plain D2E on the same items** (think hurts web element selection more than the MCQ filler control suggested). Records kept as `webreplay_records` H_web partial = uncalibrated-gate ablation.
- perf: prebuilt `causal-conv1d` 1.7.0 wheel is ABI-incompatible with torch 2.11 (undefined symbol at import) — uninstalled; reference conv path stays.

**Next:** G_web (think-always vanilla default, 200 steps) → full Gate B table + verdict; then cal-split recalibration (fit web temperature/thresholds, test margin-style gate signals) and batched D2 scoring to close the latency gap.

### 2026-09-19 — Gate B COMPLETE (`eval/reports/webreplay.md`) — verdict

| arm | n | elem_acc | op_acc | parse_fail | p50 | escalation |
|---|---|---|---|---|---|---|
| **D2E** (decide over elements) | 900 | **0.320** | 0.490 | 0 | 1.60 s | — |
| C_web (vanilla no-think) | 900 | 0.258 | **0.816** | 0 | 0.26 s | — |
| H_web (MCQ gate, aborted) | 78 | 0.244 | 0.000 | 0 | 22.6 s | 69% |
| G_web (vanilla think-always = native default) | 200 | 0.115 | 0.840 | 0 | 27.4 s | — |

**Verdict against the phase question ("can the PoC beat vanilla Qwen 4B at browser use?"):**

- **vs the native default (think-always): dominated on BOTH axes, decisively.** D2E 0.320 vs 0.115 element accuracy (2.8×; McNemar 55/4, χ²=42.4, p ≈ 6×10⁻¹¹ on 200 shared steps) at 1.60 s vs 27.4 s p50 (17× faster). Think-always *collapses* on web element selection — the model's own default mode is its worst possible configuration for GUI steps. The MCQ regime story (thinking pays only on computation-shaped items) transfers: a web click is a perception-selection item.
- **vs the best vanilla config (no-think): +6.2 pts element accuracy, p ≈ 4×10⁻⁵** (McNemar 118/62, n=900), at 6× the latency — the gap is sequential per-candidate prefill, batched scoring is the known fix. With batching, the expected landing zone is ~0.3–0.5 s, which would make D2E dominant on both axes here too.
- **H_web: the MCQ hybrid does not transfer as-is** (aborted at n=78): gate routing still has signal (accepted 0.500 vs 0.450 same-items D2E) but think→re-DECIDE is harmful on web (0.188 tail vs 0.271 plain D2E) and the calibrated thresholds don't transfer (conf AUC 0.648, premask AUC 0.493). Web-specific redesign needed: cal-split threshold fit + a different escalation target (or none: plain D2E already beats both vanillas on accuracy).
- Op accuracy remains vanilla's win (joint "number OP" reply 0.82–0.84 vs our post-hoc 3-way read 0.49) — factorization bug on our side, fix is op-before-element or joint rendering.
- Zero parse failures on every arm's element output.

**Answer to the phase question: yes on the default comparison (both axes, decisively), accuracy-only on the no-think comparison pending batching.** Next: batched D2 scoring → cal-split gate recalibration → web-specific escalation redesign (op-first read; escalate-to-generation candidate) → optional live demo.

### 2026-09-19 — Gate B v2 (web-specific redesign) — verdict

Redesign per the Gate B diagnosis: op-first read conditioning the element read, escalation swapped from think→re-DECIDE to a **rich re-read** (top-3 candidates with subtree/section context, zero think tokens), gate fitted on the cal split (`eval/webgate.py` → `configs/webgate.json`: temperature 2.0, accept_conf 0.121 tempered = 40% coverage, cal accepted-acc 0.456 vs overall 0.340; margin AUC 0.699 vs 0.648 confidence).

| arm | n | elem_acc | op_acc | p50 | escalation |
|---|---|---|---|---|---|
| D2E2 (op-first → op-conditioned element read) | 900 | 0.344 | 0.256 | 1.70 s | — |
| H2_web (gated rich re-read) | 900 | **0.356** | 0.256 | 1.77 s | 27% |
| (reference: D2E / C_web as above) | 900 | 0.320 / 0.258 | 0.490 / 0.816 | — | — |

- **Decide beats vanilla no-think again, decisively:** D2E2 0.344 vs C_web 0.258 (McNemar 144/66, χ²=28.2, p≈1.1×10⁻⁷); H2_web 0.356 vs C_web (χ²=33.2, p≈8×10⁻⁹).
- **Escalation sign flipped from harmful to positive:** on the 245 gated (low-confidence) items, rich re-read recovers 0.200 → 0.241 (32 wins / 22 losses, p=0.22, n.s.) — vs v1's think→re-DECIDE which *dropped* the tail 0.271 → 0.188. Information escalation > reasoning escalation on web steps, as the Phase 2 filler-control lesson predicted.
- **Gate routing signal transfers after recalibration:** accepted 75% of items score 0.398 vs escalated tail 0.244 (cal fit predicted exactly this separation). The gate is usable as an "ask-the-user / flag-uncertainty" signal even where escalation doesn't convert to accuracy.
- **Op-first read costs op accuracy:** 0.256 vs v1's element-conditioned read 0.490 — op-first sees no element, and predicts SELECT for 61% of gold CLICK items. Element identity is the strongest op signal. Fix (measured next round): keep op-first as a *prior* to condition the element read, then re-read the 3-way op **with the chosen element in view** (`OPFIX_web` arm + H3 chain stage).
- op-first element gain is +2.4 pts (p=0.11, n.s.) on its own; the net v2 decide config carries it because it costs nothing.

**v2 verdict:** the decide-mode advantage over vanilla no-think is robust across two independent read designs; the web-calibrated gate routes correctly; escalation is now safe (non-harmful) but converts only ~+1.1 pts overall. The remaining open items for the phase are latency (batched D2 scoring) and the v3 tool-call chain (op → element → value) measured as action accuracy.

### 2026-09-19 — Gate B v3 (tool-call chain) COMPLETE — verdict

Phase question: the user's target architecture — DECIDE (which tool) → THINK (what value) → structured tool-call output — measured end-to-end on Mind2Web replay (`op` = tool, `element` = target, `value` = argument; action_acc = op ∧ element ∧ value-exact).

| arm | n | elem_acc | op_acc | value_acc (exact) | action_acc | parse_fail | p50 |
|---|---|---|---|---|---|---|---|
| **H3B_web** (chain: op-first → gated element → op re-read w/ options → think value) | 900 | **0.356** | 0.843 | **0.600** (n=30) | **0.320** | **0** | 2.09 s |
| H3_web (chain, op re-read without options) | 900 | 0.356 | 0.507 | 0.346 (n=26) | 0.204 | 0 | 1.92 s |
| C3_web (one-shot no-think JSON tool call) | 900 | 0.303 | 0.844 | — (never engages, n=2) | 0.274 | 0 | 0.75 s |
| G3_web (one-shot think-always JSON = native default) | 200 | 0.100 | 0.180 | — | 0.100 | **158/200** | 31.3 s |

**Findings:**

- **The chain beats one-shot no-think JSON on the full action (McNemar 113/72, χ²=8.6, p=0.0033)** and dominates think-always JSON (χ²=34.6, p≈4×10⁻⁹), at zero parse failures (schema-constrained assembly) — the user's DECIDE→THINK→GENERATE design is the measured winner for structured browser actions.
- **The decisive fix was context parity on the op stage** (H3 → H3B): giving the op re-read the same element-line options the vanilla prompt sees lifted op 0.507 → 0.843 (≈ C3's 0.844, p=1) and action 0.204 → 0.320. The chain's stages must see what the end-to-end prompt sees; a narrow-context stage read is how factorization loses to joint generation.
- **One-shot no-think JSON is degenerate as an agent:** C3 predicts CLICK on 896/900 items (never TYPE/SELECT), so its value stage engages on 2/900 items; its 0.844 op_acc is the CLICK base rate. It looks competitive on CLICK-heavy joint metrics while being unable to type or select. Honest finding to keep visible.
- **Think-always (native default) is unusable for tool calls:** 79% parse failures (thinking crowds the budget), worst accuracy, 15× slower.
- Chain cost: p50 2.09 s vs C3 0.75 s — 3-4 sequential decide prefills; batched D2 scoring is the known fix (expected to mostly close this).
- Value stage: exact-match 0.600 (loose 0.633) on op∧element-correct TYPE/SELECT items with ~24-token generations — Mind2Web values are mostly copy-from-task strings; harder value computations are untested here.

**Browser-use phase verdict (v1+v2+v3 combined):** decide-mode runtime beats the native default on both axes decisively (accuracy 2.8-3.6×, latency 14-17×), beats the best vanilla config on element selection (p≈1e-8..1e-7) and now on the full structured action (p=0.0033); the calibrated gate routes correctly (accepted 0.398 vs tail 0.244) with safe information escalation (+4.1 pts on the tail). Open items: batched D2 scoring for latency parity, live-browser demo, multimodal read test (S7).

### 2026-09-19 — L1: batched D2 scoring COMPLETE — parity gate PASS, ~5× latency cut

**Machinery** (`hybrid/scoring/sequence_logprob.py`): one prefix prefill (`use_cache=True`) + per-candidate continuation scoring against the shared cache. Cache strategy picked by a three-condition probe (memoized per process): (1) conditioning — suffix logprob on the pristine cache vs the full-forward reference within the measured bf16 shape-noise floor (0.35 nats); (2) crop-rewind bit-equality → "shared"; (3) when rewind is lossy (GDN: it is), tile validation on a fresh pristine prefix — 4 identical rows must score bit-identically AND within the noise floor of both the untiled suffix score and the copy-cache score → "batched" (production default); else "copy". `_tile_cache` uses `CacheLayerMixin.reorder_cache` with an all-zeros beam index — an exact batch tile that covers the full-attention KV and the linear-attention state dicts (which are dicts `{state_idx: tensor}`, invisible to a tensor walk). Continuations are right-padded per chunk; pads sit AFTER each row's real tokens and the recurrent state is causal, so pads cannot corrupt row scores.

**Parity gate** (`eval/batch_check.py`, round 6 — PASS):
- auto (probe picks batched), n=50: argmax flips **0**, max |Δprob| 7.8e-3, max |Δpremask| **0.00**, max |Δraw| 0.37 nats (inside the noise floor).
- Near-tie criterion: an argmax flip only disqualifies when the sequential top-2 probability gap > 0.05 (bf16 tie-breaking under different kernel schedules; measured noise ~0.02-0.03 nats/token motivated this — round 4 saw a flip at gap 0.000).
- Negative control (score against a token-differing wrong prompt's cache): |Δraw| 0.57 vs 0.05 for the right prompt — **11× separation**.
- `shared` is dead on GDN (16 near-tie flips; crop-rewind not bit-exact: lp1 -4.2945 vs rewound lp2 -3.8995); `copy` is correct but sequential (1 near-tie flip at gap 0.000).

**Batched arms** (D2E2 + H2_web re-run on the same 900 test steps; records `eval/reports/webreplay_records_batched.jsonl`, sequential preserved in `webreplay_records.jsonl`):

| arm | elem_acc (seq → batched) | per-step agreement | p50 (seq → batched) | p90 |
|---|---|---|---|---|
| D2E2 | 0.344 → **0.348** | 98.33% (43 elem / 11 op flips) | 1.70 s → **0.34 s** | 0.41 s |
| H2_web | 0.356 → **0.353** | 97.56% (77 / 11 flips) | 1.77 s → **0.37 s** | 0.51 s |

**Verdict:** the latency claim is now honest — ~5× per-step p50 cut at statistically identical accuracy (every disagreement inside the near-tie band; premask deltas exactly 0). Decide mode sits within ~1.3× of vanilla no-think latency (0.34 s vs 0.26 s) while keeping the element-accuracy win. Sequential p50s in the Gate B tables above remain the provenance record.

### 2026-09-19 — L2: live three-mode demo COMPLETE (`eval/reports/live_demo/live_demo.md`)

`eval/livedemo.py` drives playwright chromium against a staged 4-page site (`eval/livedemo_site/`), extracting candidates from the LIVE DOM with a JS extractor that reproduces the replay rendering contract field-for-field (text policy: direct text → one-level-down; parent/grandparent context; path; key attrs; `[aria-modal]` dialog filter; dedupe key (tag, text, path, parent_text)). `router.route()` picks the mode per affordance; the model's action is executed as-is and verified against the gold effect in the real browser; on a miss the driver records it and performs the gold action (`recovered`) to keep the trajectory on script.

| step | mode | result |
|---|---|---|
| s1 navigate | DECIDE | clicked b103 'Loyalty Rewards' → page navigated (conf 0.516; 2.45 s incl. the one-time cache-strategy probe) |
| s2 security | DECIDE | clicked b223 'Sign out of all other sessions' among decoys (0.22 s) |
| s3 dialog | DECIDE | pure 2-candidate decision inside the open confirmation dialog (0.11 s) |
| s4 fill email | TOOL_CALL | **PASS** (beam chain, below): TYPE branch → b411 email input wins the log-linear argmax → think value 'hana.mercer@example.com' typed and verified in-field |
| s5 pick freq | TOOL_CALL | **PASS**: all three op branches converge on b413; SELECT scores best → op re-read SELECT → think value 'Weekly digest' → dropdown set in-page |
| s6 summarize | TEXT | accurate 2-sentence order summary (total $69.55, dates, 139 points — all correct) |

**Scored 5/5** after a chain upgrade that the first miss (4/5 in the two H3B-chain runs) motivated. Diagnosis (offline probe on the live context): the op-first read for s4 is a genuine three-way near-tie — SELECT 0.354 / CLICK 0.326 / TYPE 0.319, gold TYPE — and the H3B chain commits to the op before any element is seen, so the target read was conditioned on the wrong op and drifted to the 'Subscribe' button (then the op re-read, seeing a button, coherently said CLICK — a well-formed but wrong action). Fix (`router.ToolCallRequest.beam_ops`, opt-in): score the target under EVERY offered op, then let the element-conditioned op re-read arbitrate — pairs (branch op, element) scored by the log-linear product log P(op) + log P(element|op) + log P(op'|element); the winner's re-read op is the final tool. On s4 the TYPE branch (b411, target conf 0.306) wins at −3.18 vs CLICK/b415 at −3.40 (re-read confidences alone would have tied: 0.434 vs 0.427 — the prior and target terms break the tie correctly); on s5 all branches converge on b413. Chain stages are fully auditable events (decide_tool / beam_target ×3 / beam_reread / beam_argmax / think_value).

**Honest caveat:** the replay-measured chain (action_acc 0.320, op 0.843) is the H3B default (`beam_ops=False`, unchanged); the beam variant is validated on the live demo + probe only. Replay-scale measurement (900 test steps, vs H3B) is the follow-up before claiming a general action-accuracy gain. Cost: tool-call steps now run 3 target decides + up-to-2 re-reads + the value generation — s4 1.51 s, s5 1.04 s end-to-end.

**Bugs found & fixed (root-caused):**
- **Post-generate RoPE poisoning:** HF `generate()` leaves the multimodal `rope_deltas` set even for text-only prompts; the next cached (past_key_values) forward carrying an attention_mask then takes the stale-delta branch in qwen3_5's `compute_3d_position_ids`, which builds positions over the FULL mask length (P+S=186) instead of the S=17 new tokens → RuntimeError in `apply_rotary_pos_emb`. Fixed by resetting `rope_deltas` after generate (`engine.generate_answer`) plus a guard at the scoring entry points; verified tiled scores are BIT-EQUAL before/after a generate. Image-conditioned prefills pass explicit continuation positions instead (text arange + stored rope delta, mirroring the wrapper's own incremental-generation hook).
- **Special-token leak in outputs:** THINK-value and TEXT outputs carried trailing `<|im_end|>\n<|endoftext|>` (engine decodes with skip_special_tokens=False); both router paths now cut at the first special token (mirrors webreplay's `_think_value`).
- Env: transformers 5.17's processor `apply_chat_template` defaults to `tokenize=False` (older releases tokenized) — S7's image path needs `tokenize=True` explicitly.

### 2026-09-19 — L3: S7 multimodal read test COMPLETE (`eval/reports/s7/s7_multimodal.md`)

Feasibility (settled earlier): the Mind2Web HF release is text-only, so S7 uses fresh playwright full-page captures of the demo pages + hand-authored gold (26 tasks; 25 scored — one gold collapses under the shared dedupe contract: four identical 'Add to cart' buttons in different cards). Same live extractor, same renderers.

| arm | n | elem_acc | mean conf | premask_ok | p50 |
|---|---|---|---|---|---|
| d2_text (standard text-DOM read) | 25 | 0.960 | 0.328 | 1.00 | 195 ms |
| d2_image (screenshot-conditioned) | 25 | **0.960** | 0.336 | 0.96 | 395 ms |

**Verdict:** the multimodal read works end-to-end — image prefill (pixel_values + image_grid_thw + mm_token_type_ids, real mrope via get_rope_index) → batched text-suffix scoring with explicit continuation positions — and changes nothing on these pages: identical picks, slightly higher confidence, +200 ms p50 for the vision prefill. Caveat kept visible: 3-page self-authored site with clean DOM, 0.96 is ceiling-ish; this test establishes **feasibility and parity**, not superiority.

**Browser-use phase fully closed:** all three open items measured — batched scoring (parity-gated 5× latency cut), live three-mode demo (4/5 scored steps, one honest miss), multimodal read test (image parity). Remaining candidates for a next phase: batched H3B chain arm, scaled live evaluation on real sites.

### 2026-09-19 — Report: architecture-stack diagram added (doc only)

`poc_report.html` (Approach section) gained a visual layer stack answering "what was modified vs stock, layer by layer": agent harness (≈ the Claude Code / opencode loop; NEW, thin) → mode router + decision reads `hybrid/` (NEW — the PoC's core) → inference API (transformers 5.17 / torch 2.11) → tokenizer / chat template → model weights (frozen bf16) → environment — the lower four marked UNTOUCHED, with the logits-access boundary ("D2 is inexpressible behind closed chat APIs") annotated on the inference row and the mode-selection-lives-in-the-harness point on the harness row. No code, numbers, or claims changed.

### 2026-09-19 — S-experiment: d2_over_modes — can the decision mechanism pick the output mode? COMPLETE (`eval/reports/mode_probe/mode_probe.md`)

Directly tests the user's proposal: point the same D2 sequence-scoring read at three output-contract candidates (DECIDE / TOOL_CALL / TEXT) and route on the argmax, instead of harness-side affordance declaration. Preregistered: gold mode = DECIDE iff gold op CLICK, TOOL_CALL iff TYPE/SELECT (deterministic); misroute cost asymmetry (DECIDE-on-value-step = guaranteed action loss; TOOL_CALL-on-CLICK = latency only; TEXT = guaranteed loss). Two context variants: full (elements in view, H3B op-re-read rendering + mode question) and bare (task+history only). Batched D2, n=900 test. Runner: `eval/mode_probe.py`; records: `eval/reports/mode_probe/mode_probe_records.jsonl` (1,800).

| mode router | mode acc | gold TOOL_CALL→DECIDE | action acc (simulated) | Δ vs status quo |
|---|---|---|---|---|
| always TOOL_CALL (affordance routing, status quo) | — | 0/140 | **0.320** | — |
| d2_over_modes · full (147 ms p50/read) | 0.834 | 133/140 (95%) | 0.273 | −4.7 pts |
| d2_over_modes · bare (118 ms p50/read) | 0.840 | 136/140 (97%) | 0.273 | −4.7 pts |
| mode derived from the chain's op-first read | 0.321 | partial | 0.297 | −2.3 pts |
| form_present DOM signal | — | — | 0.298 | −2.2 pts |
| **perfect mode router (oracle)** | 1.000 | 0/140 | 0.292 | **−2.8 pts** |

**Verdict: mode selection stays in the harness — now measured, not argued.** The dedicated mode read routes below the trivial always-DECIDE constant (0.834 vs 0.844 = the CLICK base rate): it collapses to DECIDE on 95% of value-needing steps because the input's surface form (a rendered selection task) dominates the likelihood; ~70% of the reads fail the runtime's own pre-mask validity guard (first-token mass goes to element numbers, not mode names); wrong routes are near-ties (error margin p50 0.058 — the contract is not a linguistic property of the input); TEXT never wins a step. Executed end-to-end it costs −4.7 pts action accuracy. The decisive row is the ORACLE: even a perfect mode router loses 2.8 pts, because DECIDE mode's element read is unconditioned (D2E elem 0.320; 0.324 on gold-CLICK steps) while the chain's is op-conditioned (0.356) — the contract declaration is accuracy-relevant context, not overhead (context-parity lesson again). DECIDE mode is not "TOOL_CALL minus latency"; it is a strictly weaker reader whose only justification is harness-known structure (e.g., 2-option dialogs).

**Honest caveats:** replay only — TEXT untestable (no free-form gold; distractor only), so the measured boundary is DECIDE↔TOOL_CALL. The action-accuracy simulation splices measured per-item records (H3B chain; D2E unconditioned element decides) rather than re-running a coupled runtime — legitimate because arms are per-item-independent, but added-latency interaction is not modeled (the mode read itself costs +147 ms p50). Mode-candidate rendering mirrors the measured OP_DESC style; no rendering sweep. Gold-mode definition and cost asymmetry were preregistered in the runner docstring before the run.

(doc-only follow-up, same day) `poc_report.html` Approach section gained a turn-sequence diagram ("One turn through the layers — who initiates what"): three-lane flow (agent harness steps 1/2/6 → router steps 3/4/5 → model as called-by-chips), boundary-crossing legend, and a two-card contrast of "if DECIDE were a model-invoked tool" (generation + parse + model-picked mechanism) vs the runtime's harness-initiated reads. No numbers changed.
(doc-only follow-up 2, same day) `poc_report.html` Approach section gained a vanilla-vs-PoC comparison: two-lane visual (one-mechanism generate loop vs mechanism-per-affordance routed loop) + a per-turn diff table with the measured numbers (answer mechanism, mode selection, option choice, tool calls, uncertainty, auditability, untouched layers). No numbers changed.

### 2026-09-20 — Three-arm harness-level comparison (omp unchanged): vanilla 0/5 vs decide-proxy 3/5 — plus the failure catalog behind chain v3.2→v3.7.5

**The user's question, operationalized.** Take an off-the-shelf agent harness (omp / oh-my-pi) and a local Qwen3.5-4B, change NOTHING in the harness — only the OpenAI-compatible base URL — and compare (A) vanilla endpoint vs (B) the PoC's decide-hybrid proxy, on five real-browser tasks. (C) the PoC's own harness (live-scale: element 0.850 / 0.90 adjudicated, effects 0.950, value 9/9, p50 0.75 s/step) is the reference ceiling. Tasks (each `omp -p --max-time 600`, fresh tab per run, adjudicated from session transcripts + the proxy decision log, end states verified from page snapshots — never from model claims): books.toscrape 4-hop navigation (T1); quotes.toscrape login with task-given credentials (T2); the-internet Form Authentication login+logout (T3); the-internet dropdown select-then-switch (T4); saucedemo login + price-sort + add cheapest to cart (T5).

**Arms.** A: **0/5** — no task reached a verified end state; the vanilla 4B thrashes (invents a phantom "CSRF token requirement" on T2, re-plans in prose, burns every deadline). B: **3/5** after the chain-hardening ladder below (T2 44 s, T3 40 s, T4 61 s — all near-pure chain execution; T1/T5 still deadline). C: 0.850/0.90 on the replay-scale metric — not directly comparable per-task, shown as the reference the proxy arm climbs toward.

**Final v3.7.5 sweep (one version, all five tasks):**

| task | result | verified end state |
|---|---|---|
| T1 books | ✗ 600 s | chain picked "Travel" at conf 0.342 then stood down; model observe-loop burned the budget (prefill of ~1.6k-token snapshots × ~30–60 s/turn) |
| T2 quotes | **✓ 44 s** | 5 chain cells (Login link → merged-textbox TYPE with ordinal advance → submit); logout link in final snapshot; "login succeeded" report matches the page |
| T3 internet | **✓ 40 s** | 6 chain cells, zero detours; "You logged into a secure area!" then "You logged out of the secure area!" both in snapshots |
| T4 dropdown | **✓ 61 s** | `select "Option 1"` then `select "Option 2"` both chain cells; report "Option 2" |
| T5 saucedemo | ✗ 600 s | chain did login + sort (price low→high verified: Fleece Jacket $49.99 at list tail); model never added to cart — observe-loop after the ANSWER gate kept routing its turns to answer |

**The failure catalog — seven root-caused live failures, each fixed by a structural (non-prompt) change:**

1. **The uninvoked click (v3.3–v3.5; the big one).** `tab.evaluate()` evaluates an *expression*: the CLICK cell was `(() => { … })` with **no trailing `()`** — it evaluated to the function object (serialized `{}`) and the body never ran. Every click for three versions was a silent no-op reported as success ("the cell executed without error"); TYPE/SELECT always ended `})()` — which is exactly why typing worked and clicking never did. Found by attaching a probe page to omp's own shared headless Chrome (DevToolsActivePort) and testing evaluation semantics side-by-side; fixed as v3.6 (page-side DOM clicks, never ref-RPC; button-over-link tie-break).
2. **Anchor yank (v3.7).** The exact-phrase anchor (task names an element → prefer it) fired on every turn: "login" appears in T2's task text, so the chain yanked the TYPE pick onto the Login link mid-task. v3.7.1: guard to navigation steps (skip on TYPE ops and textbox picks).
3. **Context-slimming sensitivity (v3.7/3.7.1).** Stubbing all but the LAST tool result flipped the model's own turns into close-and-redo junk (n=2: `browser.close()` compulsion loops, `<tool_call>`-leak finals). v3.7.2: keep the last 3 tool results verbatim, stub only deep history (early sessions byte-identical to unslimmed).
4. **Candidate-cap alphabetical truncation (v3.7.3).** MAX_CANDIDATES=20 hid "Form Authentication" past #20 of the-internet's ~40 homepage links; the decide never saw it, the chain opened Basic Auth, and an HTTP-auth dialog burned the deadline. v3.7.3: task-named candidates beyond the cap displace the tail. Next T3 run clicked Form Authentication directly.
5. **Accessibility-name vs DOM-text mismatch (v3.7.4).** FontAwesome `::before` glyphs exist in the ACCESSIBILITY NAME only: the-internet's login button is `"\uf090 Login"` in the aria snapshot but `" Login"` in textContent, so the click matcher's one-directional `text.includes(want)` can never match a glyph-prefixed want ("no clickable element matching"). v3.7.4: bidirectional contains with a ≥4-char guard; T3 then passed 25 s with zero detours. (Sibling of the quotes case: the accname and the DOM text are different surfaces, and the model only ever sees the first.)
6. **Value-thinker echo (v3.7.4→3.7.5).** On T4's second step ("switch the dropdown to 'Option 2'") the value-thinker echoed the just-set "Option 1"; the repeat guard (correctly) saw that action as done and parked the chain on passthrough-repeat while the model flailed with imitation cells that drop helper declarations (`toks is not defined`) and `tab.select("#e9")` loops (a snapshot ref id is not a CSS selector). v3.7.5: when the thinker's value is already set on that element and the task names exactly one other option, take the task-named one; the chain then executed both selects itself (61 s sweep). T4 went 8.6 min fail → 61 s pass.
7. **The remaining gaps (documented, not fixed):** the ANSWER gate (0.5) is uncalibrated on live traffic and kept routing T5's add-to-cart turns to answer; T1 needs the chain to act on low-confidence first picks (0.34) or the prefill/decode budget to shrink; and the model's free-form half (add-to-cart, reports, post-task churn) remains the weak link even when the chain is perfect.

**Testing discipline that stuck** (bought with two shipped bugs): semantic tests eval the EXACT generated evaluate-string — AST-extracted from the proxy — against a DOM stub, asserting the side effect and the return value. `node --check` cannot catch an uninvoked function expression or a missing declaration; both shipped before this discipline existed. The switch-override shipped with 7 passing cases (echo+override, no-history no-op, correct-thinker no-op, cross-element no-op, unquoted switch-phrase, T5 sort untouched).

**Honest caveats.** n=5 tasks, n=1 per cell — model-side variance is real (T5: same chain quality, the model added the product in the v3.7.2-era run and observe-looped in both v3.7.4/v3.7.5 runs). The proxy deliberately stands down after its own task steps, so the model's free-form half decides several outcomes — that is the honest cost of "harness unchanged": the proxy can only act when its chain recognizes a browser-observation step, and every final report still comes from the 4B's plain generation. omp itself was untouched (config-level base-URL/model change only), verified.

**Verdict.** The proxy converts a 0/5 harness-model pairing into 3/5 with verified end states, and the three passes are qualitatively different from anything the vanilla arm produced: 40–60 s, five-to-six auditable chain cells, every value correct. The remaining two failures are now sharply characterized (uncalibrated answer gate; low-confidence stand-down + latency budget) rather than mysterious — the gap to arm C is a work list, not a fog. The decision-level mechanism survives its trip through the standard API: all of the above runs through one OpenAI-compatible `chat/completions` endpoint whose only nonstandard behavior is *when it takes longer to answer because it decided more*.

### 2026-09-20 (later) — Fix + re-test: chain v3.8.0→v3.8.1, the pairing goes 3/5 → 4/5

**What was targeted.** The two failures of the v3.7.5 sweep above, both chain-side: (T5) the ANSWER gate kept routing the still-pending add-to-cart turns to "answer"; (T1) the chain stood down at element-conf 0.342 on the FIRST navigation and the model observe-looped to the deadline. The fix is one mechanism — the **pending-step answer-route override**: when the op read says ANSWER, extract the task's action-imperative phrases (URL stripped, the trailing "Report …" clause stripped — its nouns match the log-in verb), let recorded history consume them **one action per phrase, in task order**, and if the earliest unconsumed phrase names a candidate, act on that candidate instead of answering (bypassing the 0.5 gate). Cheapest/first/last among same-named candidates resolve positionally in DOM order — which is only correct AFTER the price sort, so phrase-ordering discipline is load-bearing, not cosmetic.

**The v3.8.0 sweep adversarially caught three more defects** (the fix's own first live run): the 60-char phrase window crossed commas, so `"open … browse the Travel category, then return to the site home page"` became ONE blob that the single Travel click consumed — T1's override never fired on "return home"; the same blob swallowed T5's sort step inside `"submit the login form, sort the products …"`, and the add override fired BEFORE the sort, clicking the first "Add to cart" in A→Z DOM order = the Backpack ($29.99), not the Onesie ($7.99) — the final snapshot proved it (Remove buttons after Bike Light and Backpack, none after Onesie). A merged aria marker (`"Username Password"` from one TYPE cell) marked BOTH fields filled, so T2's override submitted with an empty password — the quotes sandbox logged it in on the username alone, so the END STATE passed while the mechanism was wrong (caught by reading the cells, not the report). And sort phrases could neither match their combobox (role excluded) nor be consumed by SELECT history.

**v3.8.1** (26-case semantic suite, ALL PASS, including replays against the REAL candidate lists scraped from the failing sessions): phrase windows stop at commas; one TYPE entry covers one task phrase (a merged marker fills nothing twice); `sort`/`select`/`choose`/`pick`/`switch … to` are dropdown-flavored — they consume SELECT history and may name the combobox (the existing op-reread combobox→SELECT flip performs the action); two-token phrases join space-free (`"log out"` ↔ `'Logout'`); the other consumption pool is tried second (the URL-stripped `"open …and select …"` blob is click-verb-shaped but select-consumed).

**Re-test (one version, all five tasks, fresh tabs):**

| task | v3.7.5 | v3.8.1 |
|---|---|---|
| T4 dropdown | ✓ 61 s | **✓ 14 s** — both selects, repeat-parked, model answers |
| T2 quotes | ✓ 44 s | **✓ 44 s** — password typed BEFORE submit now (the merged-marker fix); logout in final snapshot |
| T3 internet | ✓ 40 s | **✓ 40 s** — submit AND log-out steps via the override ("log out" → 'Logout') |
| T5 saucedemo | ✗ 600 s | **✓ 49 s** — sort (`OVR "sort the products…" → combobox → SELECT price low→high`) then add (`OVR "add the cheapest product…" → sorted-first Add to cart`); Onesie verifiably in cart (Remove after Onesie in the final snapshot) |
| T1 books | ✗ 600 s (1/4 steps) | ✗ 600 s — **3/4 steps**: Travel ✓, Home via override ✓, next-page ✓; died on the documented gap: "open the first book's detail page" names no candidate, the element decide re-picked the Travel attractor, and the model's free-form half could not recover in the remaining budget |

**Score: vanilla-harness pairing 3/5 → 4/5** (vanilla endpoint unchanged at 0/5). Both targeted failures were addressed: T5 fixed and verified end-to-end; T1 materially improved with its residual cause re-characterized — resolving an UNNAMED target ("the first book") is a semantic boundary of the 13M-parameter chain head, not a routing bug; the vanilla 4B could not do it either, in ten minutes of free-form attempts.

**New caveat found by the re-test: runs are not isolated.** The daemon's shared browser context persists localStorage across runs and arms — T5's badge truthfully read 3 (Onesie added this run + Bike Light and Backpack left over from the v3.8.0 run's mis-clicks). End-state adjudication must therefore read WHICH items are in the cart, not just counts; and cross-arm comparisons carry this leakage. (The override also bypasses the gate on matched phrases by design — a task whose phrases match but whose steps were done elsewhere is guarded by one-action-one-phrase consumption, not by page-state verification.)

**Verdict.** The fix cycle itself became the demo of the method: the decision log caught the override's own mis-routes at the same grain that caught the original gate mis-routes (which phrase consumed, which element named, which cell written), each defect became a semantic test case against the exact failing page's candidates, and the re-test shows the two characterized failures converted into one verified pass and one measured capability boundary. All of it still behind an unchanged omp pointed at one OpenAI-compatible URL.

### 2026-09-20 (later still) — Generalization: chain v3.9.0, one proxy serves browser AND coding, zero browser drift

**What was built.** The browser-only decide chain became a domain-general decision core with per-domain adapters. The split is structural, not a model guess: `detect_domain()` looks at which tools the conversation has actually CALLED (browser tools → browser; bash/read/write/edit/glob/grep → coding), falling back on turn 1 (no calls yet) to the same deterministic task-shape regexes the browser chain already used (URL/browser word → browser; file/chmod/script shape → coding). The coding adapter (`run_code_chain`) is the browser skeleton wearing file surfaces: the survey cell is `pwd && ls -la` (the open-the-URL analogue) whose output parses into `[fN] role name` file candidates; the pending-step override port walks the task's create/write/run/chmod/fix imperatives in task order (windows cut at clause boundaries — the comma lesson, plus em-dash from the start), consumes one history entry per phrase (task order is load-bearing — chmod-after-both-writes), and resolves targets from in-window tokens (kept raw, e.g. `src/setup.sh`), pronouns (`run it` → most recent write), or single-task files. Content discipline is template-class only (task-quoted echo strings, the echo/sum script templates): everything beyond templates returns None and the turn passes through for the MODEL to author — the C3/C5 hybrid shape where the chain does the mechanical steps and the model does the thinking. One live defect surfaced in the first sweep and was fixed in place: the element read over ls candidates crashed on a ONE-file directory (DECIDE requires ≥2 candidates — a one-file ls has nothing to disambiguate), now guarded to pass through.

**Zero-drift proof, two levels.** (a) Source: v3.9.0 vs deployed v3.8.1 is exactly 4 hunks (detect_domain insert, coding adapter section, compute() branch, banner); the browser path is byte-identical. (b) Semantics: the 16-case browser suite runs against BOTH sources from one file and passes on both; the 34-case coding suite pins the new path (AST-extracted exact functions, torch-free, both suites permanent in tmp_edit/). Live re-sweep on the real sites with the unchanged omp → `poc-proxy/qwen35-4b-decide`:

| task | v3.8.1 | v3.9.0 (generalized) |
|---|---|---|
| T4 dropdown | ✓ 14 s | ✓ 12 s — both selects, repeat-parked, answer |
| T2 quotes | ✓ 44 s | ✓ 44 s — username AND password on the merged marker before submit; login verified |
| T3 internet | ✓ 40 s | ✓ 40 s — submit + log out via override; "You logged out of the secure area!" |
| T5 saucedemo | ✓ 49 s | ✓ 48 s — sort→combobox 'Sort products', add→post-sort cheapest; Onesie verifiably in cart (badge 5 = leftovers caveat, read WHICH items) |
| T1 books | ✗ 600 s (3/4) | ✗ 600 s (3/4) — same documented shape: Travel, Home (override), next-page; deadline on the unnamed "first book" target (element re-picks the Travel attractor) |

**Browser verdict: 4/5 replication, no drift — and the first sweep's inflated wall-times were a measurement artifact, now root-caused.** The clean-GPU rerun matches v3.8.1 timings to within 2 s on every task. The first v3.9.0 sweep (27/97/42/142 s) had IDENTICAL routing turn for turn but ran while an unknown client was sending long requests to the same proxy — an agent-handoff document generation (a context-compaction prompt addressed to "the successor", about this very project) occupied the proxy's global LOCK for ~15 min, queueing every sweep turn behind it. Two operational findings from that incident: (1) the GPU's full 15.9 GB was the proxy's own model — the idle host services hold nothing; (2) something (another agent session) is pointed at port 8999 and WILL interleave with sweeps — clean timings require a quiet proxy. (Resolution, 2026-09-20: the client was the opencode agent running inside the local Zed editor since Sep 7 — its model config points at 8999; killed by PID with the user's authorization, and the port has been clear since.) Also verified while the proxy was idle: a bare one-message conversation through qwen35-4b-decide routes passthrough-first (chain stands down, zero chain actions in the log) and answers in **3.3 s** — simple conversation works through the decide id, though the vanilla-nothink id remains the zero-overhead route for pure chat. Domain detection: 10/10 clean across both suites — no coding-turn ever woke the browser chain, no browser-turn ever woke the coding adapter.

**Coding suite (C1–C5, same tasks as the vanilla baseline), decide arm:** 5/5, matching the vanilla 5/5 — with the chain faster (10/6/7/25/27 s vs vanilla 27/23/33/34/38 s) because chain cells skip generation. End states verified on disk, not from model claims: C1 `src/setup.sh`+`src/main.sh` executable, `ready`/`running` in order; C2 `sum.py` prints 4321; C3 greet.sh prints `hello fixed`; C4 config flipped to `'production'`, report shows debug→production in order; C5 calc.py fixed (`a + b`), `ALL TESTS PASS`. Routing: 10 surveys + 21 pending-step cells chain-driven; 15 passthrough turns where the model authored beyond templates (C4's show.py, C4's config edit, C5's bug fix) — and the run-again steps after a model edit resolve correctly. **The honest headline stands: short local coding loops never exposed the 4B's thrashing (vanilla already 5/5), so the coding chain's win is auditability — every mechanical step it drove is in the decision log with its phrase, target, and cell — not a score gain. Browser remains the discriminating domain (0/5 vs 4/5).**

**Deployment state:** v3.9.0 on the host (md5 e0fec30fd722aee6adaf873a10009afa local+host), proxy restarted clean at 16:23 (PIDs 801944/801945 — the restart cleared a mystery client's queued 15-min generation and dropped GPU memory 15.9 GB → 9.0 GB, confirming the idle host services hold nothing), banner confirms v3.9.0, /v1/models serves all three ids (vanilla, vanilla-nothink, decide). Runners: run_omp_decide_v390_all.sh (browser), run_omp_code_decide390.sh (coding).

### 2026-09-20 (latest) — v3.10.0: the universal tier (Tier-1 floor) + llama.cpp port feasibility

**What was built.** The proxy now routes three ways: browser chain, coding adapter, or — new — the **universal tier**: everything that is neither (conversation, research, unknown tools). Only detect_domain's FINAL fallback changed (`return "browser"` → `return "universal"`); every browser/coding route is byte-for-byte v3.9.0 logic and both regression suites pass unchanged (browser 16×2 sources; coding 35 after pinning the new semantics of two formerly-default-routed cases). The floor adds the domain-free wins without an adapter: **(1) an audit log on every turn** — the old fallback logged nothing (the passthrough-first silence that cost us an hour of debugging), every universal turn now lands in decisions.jsonl with its task, route, streak, and signature; **(2) the loop brake** — the browser chain's stuck-detector generalized: when the last 3 assistant tool-call turns carry one identical signature (tool + normalized arguments; text-only turns skipped so commentary-wrapped repeats still count), the passthrough carries a corrective user-turn nudge ("you have made the same call 3 times… try a different approach or finish now") plus a one-shot 512-token cap. It never blocks — a legitimate repeat costs one nudge, not a failure. The nudge rides as a trailing `user` message, not `system`: trailing system turns are template-dependent across backends, trailing user turns are universal.

**Live verification.** Conversation: 2.5 s, correct, logged `decide-universal / passthrough / streak 0`. Simulated thrash: a research conversation with 3 identical `web_search` calls (empty results each time) → `routed: loop-brake, streak: 3, nudged: true`, and the model answered in text instead of repeating the call (finish_reason: stop, no tool_call) in 8.7 s. Routing smoke: T4 (browser, Option 2 reported) and C2 (coding, 4321) both pass. Deployed as md5 b4748e11939e774d69890d817435de4b, PID 807779, banner v3.10.0.

**llama.cpp port feasibility (recon, not built).** The expected obstacle does not exist: there is **no separate decision head to port**. `decide()` scores candidates by first-token logit comparison over the backbone itself (`score_first_token` in hybrid/scoring.py — pin the prompt, compare candidate first-token probabilities). Porting the stack to llama.cpp/quantized backends therefore reduces to: (a) passthrough already speaks OpenAI chat — swap the generation call to llama-server; (b) re-implement the scorer against llama-server's completion logprobs (`n_probs` / prompt-logprobs), one prefill per candidate with the shared prefix cache absorbing the cost; (c) validate the GGUF's chat template reproduces the no-think pin. No custom GGUF, no head surgery. The host currently has no llama.cpp installed, so this remains the well-scoped next build.

**Updated map.** Tier 1 (universal floor: audit + loop-brake + no-think + caps) — built and verified. Tier 2 (structured adapters: browser, coding) — built, verified, no drift. Remaining roadmap: research adapter (third structured domain), llama.cpp scorer port (design above).

### 2026-09-21 — Fix T1 and re-test: chain v3.11.0→v3.11.17, the browser sweep goes 4/5 → 5/5 — and a self-inflicted decode bug is caught by its own instrumentation

**What was targeted.** The one non-passing task of the 4/5 pairing: T1 (books.toscrape 4-hop navigation + unnamed "first book" target), with the standing constraint *generalized, no overfit* — every fix must be page-structure/protocol vocabulary, never site- or task-specific strings. T1 took seventeen chain sub-versions, and the arc exposed TWO stacked root causes plus one self-inflicted regression — each caught by the same discipline: the decision log at the grain of phrase/resolution/click, plus new production captures (every observation dumped to `last_obs.txt`, the compaction request to `compaction_request.txt`, the pending-step resolution logged with candidate and item-group counts).

**The mechanism ladder (each step driven by a named live failure):**

- **v3.11.0 ordinal snap** — the documented capability boundary ("open the FIRST book's detail page" names no accessible name) fell: a listing card emits its item twice (image link + title link share the accessible name), so runs of ≥2 consecutive same-name link/button candidates ARE the page's items, and first/second/…/last indexes them in DOM order. Pages without item structure keep the old skip. Structure only — no task nouns.
- **v3.11.1 prose guard** — the harness's context-compaction request is prose, not an observation; acting on it (browser cells for a summary ask) jammed T1 in a compaction loop. The chain now acts only on observation turns.
- **v3.11.2–v3.11.7 generation runtime** (the latency half of the failure): passthrough honours client max_tokens (never inflate); GenerationMixin.generate's per-step 4D mask rebuild replaced with a plain forward+past_key_values loop — 70–126 **seconds** per token at compaction-size contexts became 34–68 ms/token; HIST_KEEP_RESULTS 3→1 (stale snapshots were 15–25k re-prefilled tokens per turn); sdpa + fused kernels (prefill minutes→seconds at 20–34k). Allocator archaeology on WSL2: torch expandable_segments pins a dxg fd per segment to the 1024 ceiling then asserts (`!handles_.at(i)`), chunked prefill drops sdpa off the flash path — both tried and reverted; single-call prefill (8.5 s at 35k measured) + 65535 fd ceiling + one retry after cache clear.
- **v3.11.8 middle elision** — the compaction request embeds the whole transcript as ONE user turn; any single message over 100k chars keeps head+tail with the middle elided. This GPU falls off a cliff past ~35k context tokens (prefill 8.5 s at 35k vs 78 s at 48k; decode 35–70 ms/tok vs 4.7–9.4 **s**/tok), and the elision is what keeps compaction turns inside the deadline. Pure length policy.
- **v3.11.9 composed handoff** — the 4B answers real compaction prompts with bare EOS or template junk (an empty turn disposed a session with 8 minutes of budget left). The chain now writes the handoff itself from its tracked state (task verbatim, actions, current page, pending phrases, deliverable), gated on harness-protocol vocabulary.
- **v3.11.10–v3.11.13 compaction survival**: the redrive replaces the task with "Context replaced…" — extract_task recovers the goal from the handoff's Goal section with last-task fallback (v3.11.10); the post-compaction context retains almost no action cells, so the handoff lists executed actions in exact hist format and run_chain merges them back — progress becomes self-sustaining (v3.11.12: decide3110k looped on steps 1–3 through eleven compactions); and since the proxy sees every turn, run_chain union-merges each turn's actions into a session-wide list the handoff composes from (v3.11.13: decide3110l lost 1–2 actions per compaction and wandered the home page for its whole deadline).
- **v3.11.14 full-list pending resolution + answer fallback** — the pending step now resolves over ALL candidates BEFORE the read window is cut (on catalogue pages the first 20 are all page chrome; the ordinal-snap saw no item groups and the answer gate stood down into the model-junk loop — decide3110m), the resolved target is displaced into the window, and when the gate falls through with only a report clause left, the chain stashes the page's main heading (generic structure, never interactive) as a one-shot answer instead of a speculative click.
- **v3.11.15 snapshot parse fix** — the aria snapshot SINGLE-QUOTES any element whose accessible name carries YAML-special characters (`: ` or `#` — most real book titles), and the role regex read past the quote, silently DROPPING those elements: on the real page-3 snapshot 44 of 115 candidates vanished, "Add to basket" buttons collapsed into false same-name runs, and the v3.11.14 ordinal-snap — now working as designed — snapped onto the junk that remained (decide3110n: the pager's "next"). Verified against the real production snapshot: candidates 93→115, item groups 11→20 = exactly one per book card. Sibling fix: long titles are ellipsized in the aria read, so image link (full) + title link ("…") of one card didn't group — equality-or-ellipsis-prefix now groups them.

**The window slot race (v3.11.16) — the fix that made the resolution visible.** decide3110p logged the CORRECT resolution and the WRONG click on one line: `pending_scan.resolved = "In Her Wake" ref=e135`, click = next@e516. Root cause: the pending target and the task-named extras both displace into the same read-window tail slot, and the extras rebuild ran SECOND — the word "next" (≥4 chars, in "go to the next page") is itself task-named, so the pager link retook the slot the ordinal-snap had just claimed. The window build is now one extracted function (`build_read_window`, regression-tested): extras first, pending target spliced in ahead of them — the pending step is this turn's chosen action and wins the slot. The superlative same-name collapse is also barred from re-picking a positionally-resolved target.

**The decode-loop duplication (v3.11.17) — caught by the sweep it broke.** The v3.11.16 sweep (decide311016) returned a paradox: every chain action correct on all five tasks (verified end states on four), yet every model REPORT empty — the heading fallback shipped page subjects on the gate-routed turns ("Quotes to Scrape", "Login Page") and T4/T5 died with no answer. Direct probe of the generation path returned `finish_reason=stop, content=''`. Root cause: the greedy decode loop existed TWICE, and the second copy re-assigned `gen_ids = [last_token]` — always the EOS token — then exited on it immediately: **every passthrough generation returned a single EOS**, an empty answer on every model turn. An edit artifact of the v3.11.x series, invisible to the torch-free suites (they extract chain functions, not the generation loop) and masked for a run by the fallback machinery quietly carrying every report. Fixed to one loop, plus a 4-token EOS floor (the 4B's first greedy token is EOS on awkward prompts — the bare-EOS mode that motivated the composed handoff; a client max_tokens below the floor still wins). Suites: 46+16+35+25 = 122 ALL PASS.

**Final sweep (v3.11.17, one version, all five tasks, fresh tabs, end states verified from snapshots — never model claims):**

| task | v3.9.0/v3.10 | v3.11.17 |
|---|---|---|
| T4 dropdown | ✓ 12 s | **✓ 11 s** — both selects; report "**Option 2** is now selected." (real model generation) |
| T2 quotes | ✓ 44 s | **✓ 24 s** — logged in (Logout link, no Login in final snapshot); report "**Login succeeded.**" |
| T3 internet | ✓ 40 s | **✓ 23 s** — "You logged out of the secure area!" in final snapshot; report quotes it |
| T5 saucedemo | ✓ 48 s | **✓ 68 s** — sort verified (7.99→49.99); Onesie $7.99 (the cheapest) shows Remove = in cart; badge 3 = 1 added + 2 documented shared-context leftovers; report says 3 — matches the page |
| T1 books | ✗ 600 s (3/4) | **✓ 44 s** — Travel → Home → composed handoff → next page → the ordinal-snap's "In Her Wake" @e135 clicked (the pend target visible at its spliced slot in the log); final page IS the In Her Wake detail page; report names it |

**5/5 — the first perfect browser sweep**, roughly 2:50 total for all five tasks. The two report-side mechanisms built for T1 (heading fallback, junk-fallback-answer) ended up UNNEEDED in the final sweep — every report is real 4B generation — but they remain as one-shot safety nets.

**Honest ownership.** The decode duplication was self-inflicted during the v3.11.x edit series; it shipped because the regression suites are torch-free by design (the GPU host runs no pytest) and nothing exercised the generation loop until real reports were scored. The sweep that caught it is the same instrumentation pattern that caught the slot race: judge outcomes from page end states and decision logs, never from mechanism confidence. T1's 44 s pass stands on eleven stacked structural fixes, each verified against the live failure that demanded it — no site names, no task vocabulary anywhere in the mechanisms.

### 2026-09-21 — v3.12.0: the learned-adapter plumbing — capture, builder, gate, registry, hot-load routing, all config-gated

**The request.** The design answer ("adapters can be learned along the way: floor first, capture, offline builder, validation gate, registry, hot-load") made real. Everything — capture, learned-adapter routing, and the builder step with its model setup — is enabled/disabled by config.

**Shape: a learned adapter is DATA, not code.** The one design decision everything else follows. The builder model does not write Python — it fills a constrained JSON **spec**: an ordered list of phrase rules (`when` regex over the task's instruction verbs → `tool` + `args` with `$1..$9` captures), stand-down vocabulary, and repeat limits. The interpreter that executes specs (`run_spec` in `adapter_learning.py`) is fixed, reviewed, stdlib-only code. Nothing a model writes is ever exec'd, imported, or eval'd on the host. The worst a malformed spec can do is stand down.

**The five pieces** (new `eval/adapter_learning.py`, stdlib-only; `adapter_builder.py`, offline CLI; `eval/adapter_learning.json`, config):

1. **Capture** — `run_universal` now hands every floor turn to `learning.floor_hook` before serving. With capture on, each unrecognized-tool-family conversation appends only its NEW messages (images dropped, long content truncated, per-session byte cap) to `eval/reports/adapter_capture/<family>/sessions/<sid>.jsonl`, with a meta line carrying the tool schemas. Family identity = the request's exact declared tool-name set, sha1'd to 12 hex (order-insensitive, deduped). Session identity = family + system-prompt head + task head (the wire format has no session id; same documented approximation as the browser chain's task tracking).
2. **Builder** — offline (`python3 eval/adapter_builder.py --family <key>`), never in a request path. Loads the captured trajectories, builds a compact dossier, and asks the **configured builder model** — any OpenAI-compatible endpoint: `builder.base_url / model / api_key / api_key_env / temperature / top_p / max_tokens / timeout_s` — to fill the spec. Default endpoint is the local proxy itself (works out of the box; a stronger model is the intended setup). Identity fields (fingerprint/slug/tools) come from the capture and are overwritten over whatever the model claims.
3. **Validation gate — not skippable.** Shape check; anti-overfit lint (no URLs in spec literals; task nouns flagged by token against the captured task texts, regex escapes stripped first so `\bazure` can't hide "azure"; config-extensible protocol-vocabulary allowlist); then **replay**: the spec runs over every captured conversation prefix with ONE interpreter state threaded through each session (replay must live the same life the deployed session did — this threading was a real bug the first gate smoke caught: fresh-state-per-fixture re-fired step 1 on every mid-session turn). Pass requires ≥ `min_action_match` (default 80%) of captured action turns reproduced by tool name, **zero** false fires on turns where the session stood down or reported (rejected outright — the dangerous direction), and zero crashes on malformed-input probes. A failing candidate stays a candidate; even manual `--promote` re-validates.
4. **Registry** — a passing spec lands at `eval/learned_adapters/<family>/adapter_spec.json` + `manifest.json` (version, builder model, validation stats). `auto_register: false` holds everything for manual promotion. `--forget` unregisters; `--validate-only` re-runs the gate without a model call; `--list` shows families, sessions, brakes, adapter status.
5. **Hot-load routing** — `floor_hook` consults the registry by fingerprint (mtime-cached; no restart to pick up a new adapter, same as config hot-reload) and, on a match, runs the interpreter: an action becomes an OpenAI tool call for the FAMILY's own tool (logged `arm=decide-learned, routed=adapter-cell`); stand-down (no matching unconsumed phrase, observation gate, or loop guard) returns the turn to the floor and the model (`adapter-standdown`). The interpreter inherits the browser chain's discipline: actions fire on tool-result turns or a fresh conversation only; per-phrase `max_uses`; an identical-signature repeat cap (`max_identical_repeats`) that stands down instead of thrashing.

**Config** (`--adapter-config`, default `eval/adapter_learning.json`; missing file or `"enabled": false` = byte-for-byte v3.11.17 behavior; mtime hot-reload per floor turn): master `enabled`, `capture`, `route_to_learned`, dirs, `min_sessions_to_build`, `auto_register`, `min_action_match`, `lint_allowlist`, and the whole `builder` block. The read-only `~/.omp/agent/config.yml` is untouched — this is proxy-side only.

**Failure posture.** `floor_hook` never raises: config errors fall back to disabled, capture errors are swallowed, an adapter crash logs `learning-error` and the floor serves the turn. The plumbing cannot take the safety net down.

**Tests.** New torch-free suite `test_adapter_learning.py` (46 cases: config gating incl. hot-reload, fingerprinting, capture increment/caps/never-raises, spec shape, interpreter semantics incl. observation gate/loop guard/hostile spec, registry hot-load incl. fingerprint mismatch and mtime invalidation, floor-hook routing/fallback/crash survival, gate incl. false-fire rejection/overfit lint/no-action refusal, builder parse/assemble, capture→fixtures round trip, and an AST check that `run_universal` dispatches to `floor_hook`). One real bug found and fixed during the suite's own shakedown: the replay state-threading above. All suites green on v3.12.0: 46 learning + 46 browser-ordinal + 16 browser + 35 coding + 25 universal.

**Live verification (host).** Proxy v3.12.0 deployed (PID 909136, `learning: enabled=True capture=True route=True` at startup, /health ok). Synthetic `florb_scan`/`florb_open` family, driven live: **capture** — two 3-turn sessions rode the floor (6 `decide-universal passthrough` audit lines) and landed as 2 trajectory files (meta + 3 turns each) under `adapter_capture/florb_open_florb_scan-6fb39a0300c9/`; **builder, real run** — the configured builder model (the served 4B itself, the default endpoint) read the captured trajectories and proposed a spec with a literal task noun (`azure`) in a phrase: the gate **held** — `overfit: task token 'azure'`, replay action match 0/4, candidate NOT registered, exactly the "confidently wrong adapter is worse than no adapter" outcome the gate exists for (a stronger builder model is the documented fix); **routing** — a well-formed spec promoted via `--promote` (which re-validates against the real captured sessions) answered a fresh session's first turn in **0.0 s** with the family's own `florb_scan({"mode": "full"})` tool call — no model generation — logged `arm=decide-learned, routed=adapter-cell` with the matched phrase, while the session still captured (capture keeps flowing under a learned adapter); **kill switch** — `"enabled": false` in the config file (mtime hot-reload, no restart) sent the same request back through the floor as a normal 1.8 s model generation. Left on the host as evidence: the held candidate + its validation report, the registered spec, and the capture sessions.

### 2026-09-21 — backend swap experiment: llama.cpp + Gemma 4 12B NVFP4 (packages A+B of the bigger-model plan)

**The request.** "Please start the experimentation" — the agreed order: (B) stronger-builder test via config-only change first, (A) the llama.cpp backend swap (scorer port + recalibration comes after), (C) benchmarks later. This session did the acquisition, the capability smoke, the scorer feasibility probe, and the builder experiment end to end.

**Acquisition (host, no sudo).** llama.cpp release v0.4.1 ships no Linux binaries (releases are tags-only), and the host has no cmake — `uv tool install cmake` (lands in `~/.local/bin`, no root needed) unblocked a source build: `cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`, `cmake --build … -j16 --target llama-server llama-cli` (~5 min on the 16 cores). Flag drift found by the server refusing to start: `--flash-attn` now takes `on|off|auto` (used `on`). Model: FreedomAISVR/Gemma-4-12B-it-NVFP4-GGUF, 6.97 GB, at `hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf`.

**Serving.** `llama-server -m …gguf --alias gemma-4-12b-it --port 8998 --jinja -c 65536 -ctk q8_0 -ctv q8_0 -ngl 99 --flash-attn on -b 1024 -ub 512` — model loads in ~3 s (mmap), 8.5 GB VRAM with the full 65k context, health OK. **Decode measured 85.5 tok/s** on the 5070 Ti (in-response `predicted_per_second`, 64-token gen) vs the HF 4B path's ~15–30 tok/s — the 12B NVFP4 is *faster* than the current 4B serving stack, and prompt processing measured ~100 tok/s at these lengths.

**Capability smoke** (`eval/smoke_gemma_v1.py`, v1.1):
- **Tool calling: PASS** — with `--jinja`, the server emits proper OpenAI `tool_calls` (`florb_scan({"mode": "full"})`, 1.4 s). The Gemma 4 tool template works out of the box.
- **Plain chat "empty content" mystery** — the model ships a thinking template; by default `content` comes back EMPTY with the thinking in `message.reasoning_content`. Not a bug — a control problem (below).
- **Thinking control — the session's hardest-won finding.** `chat_template_kwargs {"enable_thinking": false}` only PREFILLS a closed thought channel (`<|turn>model\n<|channel>thought\n<channel|>`); on a hard prompt the model simply re-opens the channel and thinks — caught red-handed burning the entire 16384-token budget (50k chars of reasoning ending in a literal repetition loop, `content` empty). `response_format json_object` does NOT help (llama.cpp lets the thought channel finish before the grammar applies). The real switch is **request-level `reasoning_budget_tokens: 0`**: the server force-closes the thought channel the moment it opens → finish=stop in ~200 tokens, clean JSON. GOTCHA: combining `enable_thinking: false` with the budget BREAKS enforcement — the start tag moves into the prompt prefill, the generation-side watcher never arms. Budget alone; no template flag.
- **`/tokenize`**: uses the `content` field for raw text; the `messages`+`jinja` path returns empty tokens in this build.

**Scorer feasibility (the decide-arm port question).** Both paths work:
- Method A — `/v1/chat/completions`, `max_tokens=1, logprobs=true, top_logprobs=20`, `enable_thinking=false`: candidate action tokens visible with clean margins (CLICK −0.285 vs TYPE −1.394 vs SELECT −17.0), warm round-trips **44–83 ms**.
- Method B — `/completion` with a hand-rendered raw prompt (special tokens pass through the tokenizer) + `n_probs=20` → `completion_probabilities` gives exact logprobs for the generated token + top-20: first token CLICK at −0.0 (p≈1.0), no chat-template dependency at all. (v0.4.x REMOVED `prompt_logprobs`; `n_probs` is the replacement and suffices — the scorer only needs argmax + margins.)
- **Verdict: the scorer ports cleanly.** Remaining work for package A is the actual port + recalibration of chain constants (ANSWER_GATE et al.) against Gemma's token distributions — a later session.

**Builder-strength experiment (package B, config-only).** New config `eval/adapter_learning_gemma.json`: same capture dir (the existing 3 florb sessions), ISOLATED registry dir `eval/learned_adapters_gemma/` (nothing touches the live 4B registry), `builder.base_url http://127.0.0.1:8998/v1`, `builder.model gemma-4-12b-it`. One plumbing knob had to exist first: **`builder.extra_payload`** (merged verbatim into the builder request) — and with it a real gotcha: `load_config` whitelist-filters the builder block against `CONFIG_DEFAULTS["builder"]`, so the unknown key was silently DROPPED and two "fixed" runs still failed before the dump-the-request-body instrumentation exposed it. Declared in the defaults; 46/46 tests still green.

**Result: GATE PASS where the 4B was held.** The Gemma 12B read the same 3 captured sessions and produced a generalizing spec — `scan the (.*)` → `florb_scan(mode=full)`, `open the (.*)` → `florb_open(target=$1)` via capture `the (.*)`, stand_down `report/summarize/done`, `max_identical_repeats: 2` — that passed everything: **4/4 action fixtures reproduced, 0 false fires on stand-down turns, 0 crashes, lint clean** (no URLs, no task nouns; `azure` never appears). The 4B builder's candidate for the identical capture was HELD at this same gate (`overfit: task token 'azure'`, 0/4 replay). Promoted via `--promote` (which re-validates; registered into the isolated gemma registry with manifest + validation report). The "stronger builder model" fix the v3.12.0 verification predicted is confirmed with the first non-4B builder.

**Restore (end state verified).** llama-server stopped by ps-verified PID (GPU 87 MiB), proxy v3.12.0 relaunched on 8999 (PID 929034, banner + `learning: enabled=True capture=True route=True`, /health ok), and a fresh florb session routed `adapter-cell` with `florb_scan(mode=full)` in 0.0 s — the learned-adapter loop survived the backend round-trip untouched. Left on the host: the built binaries, the GGUF, `eval/smoke_gemma_v1.py` (v1.1), `eval/adapter_learning_gemma.json`, and the gemma registry.

**Swap procedure (one-liners).** To llama.cpp: stop the proxy by ps-verified PID → launch the llama-server line above → wait for `/health`. Back: kill llama-server by PID → `cd ~/hybrid-qwen && ulimit -n 65535 && nohup ~/hybrid-env/bin/python eval/omp_proxy.py --port 8999 >> eval/reports/omp_arms/proxy.log 2>&1 < /dev/null &` → banner + `/health`.

### 2026-09-21 — v3.13.0: the end-to-end measurement — Gemma 4 12B NVFP4 on the real harness, native (no chain): 0/5, same as the 4B native; the chain remains the difference-maker

**The request.** "Please measure end to end task" — after the builder experiment the open question was exactly this: does the bigger model actually finish the five browser tasks end to end? Scoping decision, stated up front: measure **Gemma 12B natively** this session — the proxy as a faithful OpenAI shim in front of llama-server, no chain, no scorer — because the element-read d2 scorer (teacher-forced sequence logprob over shared-prefix candidate lines) cannot travel over llama-server v0.4.x in one call (`prompt_logprobs` removed) and a forced-decoding port would cost 5–8 s per decision. A native run is directly comparable to the vanilla-4B baseline (same harness, same base-URL-only change) and to the chain's 5/5 sweep. The chain-on-Gemma port stays the documented follow-up.

**v3.13.0 — the remote backend arm** (config-gated, torch-free of weights):
- `--backend-config` (default `eval/proxy_backend.json`; missing file or `enabled:false` = arm absent, byte-for-byte v3.12.0 dispatch). Config: `wire_model gemma-4-12b-it → base_url http://127.0.0.1:8998/v1`, `extra_payload {"reasoning_budget_tokens": 0, "repeat_penalty": 1.1}`, `timeout_s 300`.
- `remote_passthrough(req)` forwards the harness's OpenAI request verbatim (messages, tools, tool_choice, temperature/top_p/max_tokens) + the extra payload, and maps the reply onto the proxy's internal `(content, finish, tool_calls)` triple (server tool_calls JSON-args decoded; `finish:length→stop` — the harness reads length as an unfinished turn). Streaming keep-alives, the 500 error path and the `decisions.jsonl` audit (`arm=remote` with latency and tool-call count) are all shared with the in-process arms. One retry on transient errors; **no retry on read-timeout** (a wedged backend call must not be paid for twice).
- `--no-local-model` skips `load_model()` — the 16 GB GPU holds llama-server's ~8.5 GB or the 4B's ~10 GB, not both; the shim process carries no weights.
- Dispatch tolerates provider-prefixed wire names (`poc-proxy/<arm>` normalized); `max_tokens` floors at 4096 when the client sends none so a loop cannot run to context end.
- Tests: torch-free `test_remote_backend.py` — **25/25** against a stub HTTP server (forwarding shape, extra-payload merge, tool-call decode, malformed tool_calls tolerance, length→stop, retry vs timeout, audit line, AST structural checks) plus all existing suites (46 learning + 46 ordinal + 16 browser + 35 coding + 25 universal) green. One model entry appended to the LOCAL omp `models.yml` (writable, unlike config.yml): `poc-proxy/gemma-4-12b-it`.

**Attempt 1 (killed, kept as evidence `gemma120b_attempt1`).** T4 ground to a halt at turn 4: facing a third browser-tool traceback, Gemma went into an **endless generation loop** — 44k tokens and climbing at ~70 t/s while the proxy's global LOCK waited on it, starving the harness until its 600 s dispose. Fixed structurally, not by hand-waving: `repeat_penalty 1.1` (a standard sampler knob, in config, not task vocabulary), the `max_tokens` floor, `timeout_s 300`, and timeout-no-retry. Attempt 2 was infra-clean: **122 requests, 113 tool calls delivered, 0 errors, p50 10.7 s, max 62 s — no runaway survived the knobs.**

**The result (attempt 2, judged from transcripts and page end states — never exit codes; exit=0 also covers graceful surrender with a report):**

| task | vanilla 4B native | **Gemma 12B native** | chain 4B (v3.11.17) |
|---|---|---|---|
| T4 dropdown | ✗ 600 s | **✗ 600 s** — ~40 turns fighting the harness's private eval-cell browser API (`tab.click`/`tab.id` semantics), never selected | ✓ 11 s |
| T2 quotes | ✗ 600 s | **✗ 132 s** — never found the Log in link, then reported, confidently wrong, that "the site has no login page" | ✓ 24 s |
| T3 internet | ✗ 600 s | **✗ 308 s** — typed the credentials but the Login click never navigated; never logged out | ✓ 23 s |
| T5 saucedemo | ✗ 600 s | **✗ 601 s** — 44 turns; logged in, then stuck keyboard-navigating the sort dropdown | ✓ 68 s |
| T1 books | ✗ 600 s | **✗ 99 s** — asyncio `eval`-cell import fight, surrendered with an import error report | ✓ 44 s |
| **score** | **0/5** | **0/5** | **5/5 (2:50)** |

**What the number means — and what it does not.** A 3× bigger model does not survive the raw harness interface either: native Gemma reasons coherently (its per-turn prose is genuinely adaptive — it re-plans after every traceback, 25+ distinct strategies on T4) but it has never seen this harness's private browser API, so it spends its turns on API archaeology instead of the task, and under error pressure it can produce confident false reports. The native-4B baseline scored 0/5 the same way, for the same class of reason (thrash). The chain's 5/5 is therefore not a model-scale story: it is an **interface** story — the decision chain replaces the raw API with the snapshot/element/cell vocabulary the model CAN drive, and it does so for the small model. End-to-end task success tracks the mechanism layer, not parameters. (Also consistent with the builder experiment: the same Gemma is a *strong* builder — its spec passed the gate 4/4 where the 4B was held — authoring JSON specs is not the same skill as driving an unseen tool API.)

**Honest notes.** Per-turn latency is worse than the 4B's (p50 10.7 s vs the 4B's short fast turns): bigger prefill contexts and 3× params on a 16 GB card. T4's ~40-turn thrash consumed the window even with coherent turns — turn count, not turn quality, is the binding constraint on an unfamiliar API. And the fair "Gemma + chain" arm remains unbuilt: the element-read scorer needs the documented forced-decoding port (est. 5–8 s per decision over HTTP, plus chain-constant recalibration on Gemma's token distributions) before the 5/5 claim can be tested on the bigger model.

**Restore (end state verified).** llama-server and the shim stopped by ps-verified PIDs (GPU 87 MiB), proxy **v3.13.0** relaunched on 8999 with the local 4B (weights 16.0 GB, `warmup ok`, v3.13.0 banner, remote arm present but idle without llama-server), and a `poc-proxy/qwen35-4b-vanilla-nothink` conversation answered `standing-ok` through the full path. Artifacts kept: attempt-1 + attempt-2 session transcripts and .out files under `/tmp/omp3arm/gemma120b_attempt1*` and `/tmp/omp3arm/gemma120b*`, the `arm=remote` audit lines in `decisions.jsonl`, `eval/proxy_backend.json` + `eval/omp_proxy_v3120.py.bak` on the host.

### 2026-09-21 — `eval/onboard_backend.py`: the per-model onboarding probe (chain-port checklist, automated)

The "apply the decide-think chain to any llama.cpp model" checklist splits by how often each step actually runs:

- **One-time, per backend** — the d2 element scorer re-implemented over forced decoding (or a batching scheme). Written once against the llama.cpp HTTP surface; every model on that surface reuses it. `reasoning_budget_tokens: 0` and the rest of the plumbing land here too.
- **Every model, always** — (a) detecting *this* model's no-think knob (thinks-by-default is a model/template trait, not a server trait), (b) checking that the two decision reads are *feasible on this model's token distributions* (are CLICK/TYPE/SELECT/ANSWER visible in top-logprobs with margin; what does a forced token cost), (c) recalibrating the chain's gate constants against those distributions — calibrated on the webreplay **cal split only**, never tuned to the test tasks (standing rule; and the H_web lesson says constants fitted without full chain context do not transfer, so fitting stays with the port's own calibration step).

The script automates (a) + (b) plus a cal-split discrimination sample, and emits a machine-readable profile — the input (c) needs:

```
python3 eval/onboard_backend.py --base-url http://127.0.0.1:8998/v1 \
    --model gemma-4-12b-it \
    [--cal-set eval/datasets/webreplay_v1/webreplay_v1.jsonl]
# -> profile at eval/backend_profiles/<model>.json, verdict:
#    CHAIN_READY | CHAIN_FEASIBLE_SLOW | PARTIAL_D1_ONLY | NO_SCORER
```

Probes (request shapes verified live on Gemma 12B via `tmp_edit/smoke_gemma_v1.py`): thinking-knob detection (tries `reasoning_budget_tokens: 0`, then chat-template prefill, then `include_reasoning`; verifies the chosen knob keeps tool calls working); d1 op-read feasibility (top-20 logprobs visibility + margin + warm ms over 4 generic protocol-vocabulary prompts — no task nouns, same discipline as the chain); d2 element-read cost (grammar-forced single tokens over a warm prefix, extrapolated to a 20-candidate decision vs a 2 s budget); optional `--cal-set` discrimination check that reads **even lines only** (`i % 2 == 0` in `_cal_items` — the odd-line test split is never opened, enforced by construction, covered by a test that plants "TEST ITEM DO NOT READ" rows). Exit 0 = ran to a verdict; 1 = endpoint unreachable. Torch-free, stdlib-only, safe to run from the laptop against the host.

Tests: `tmp_edit/test_onboard_backend.py` (12/12, stub llama-server, no torch). The stub initially answered 200 on *every* path, which masked a real bug the suite then caught: `post()` joined `base_url` + `/v1/chat/completions` → `/v1/v1/...` against any `--base-url` ending in `/v1` — every probe would 404 on a real server. Fixed with an `endpoint()` normalizer (accepts both `http://host:8998` and `http://host:8998/v1`), and the stub now returns a real 404 on wrong paths so this class of bug fails loudly.

Not done here, on purpose: fitting accept-gate/temperature/answer-threshold constants (step c). The profile records the measurements; the chain-on-llama.cpp port's calibration step consumes them over the same cal split.

### 2026-09-21 — v3.14.0: the chain rides llama-server — decide reads over HTTP, gates refitted on the cal split, Gemma 4 12B + chain on the real harness

The port the v3.13.0 measurement called "the documented follow-up": run the SAME decision chain with Gemma 4 12B NVFP4 as the decide arm's model, reads served by llama-server over HTTP, and measure it on the same five browser tasks where native Gemma went 0/5.

**Reads over HTTP, without prompt logprobs.** llama.cpp v0.4.x removed prompt-token logprobs, so the d2 scorer needed a different transport. `eval/chain_http.py` gets the teacher-forced sequence logprob by **grammar-forced generation**: a one-line GBNF grammar `root ::= "<candidate>"` on `/v1/chat/completions` with `logprobs: true`. Verified live on this build: a forced token reports its RAW per-token logprob (a forced `[` read −24.24 while the true top-1 thinking marker read −0.00 — masked logprobs would read ~0), so the summed token logprobs ARE the sequence logprob, reproducible to the exact string. Two gotchas: grammar piece boundaries can differ from `/tokenize` (17 vs 18), so alpha normalization uses the GENERATED token count; and forced reads need the read position pinned by `chat_template_kwargs: {"enable_thinking": false}` (closed-channel prefill → op words land on the content channel: CLICK −0.45 / ANSWER −1.01 / TYPE −9.04 / SELECT −11.36; without it the position is `<|channel>`-dominated with no ops in top-20), while generation paths keep `reasoning_budget_tokens: 0` — the two knobs must not be combined (prior finding: combined enforcement silently breaks). Forced reads run on a thread pool (4 workers against llama-server `--parallel 4`): warm forced repeat 0.33 s, 4 concurrent decodes 1.40 s wall.

**Transport indirection, one way in.** The chain's only model touches are the decide reads, the ~24-token value generation, and the answer-turn fallback — everything else in run_chain is string logic. `eval/omp_proxy.py` v3.14.0 routes those three through `DECIDE_FN` / `VALUE_FN` / `FALLBACK_FN` shims (local in-process by default, byte-identical); with a backend config that says `"chain": true`, `main()` binds the HTTP implementations, loads the refitted gates, and registers the wire model `poc-proxy/gemma-12b-decide`. A failed HTTP read degrades to a 0-confidence result on the chain's own low-confidence path (hand the turn to the model), never a 500. Tests: `test_chain_http.py` 10/10 (stub llama-server), `test_chain_remote.py` 9/9 (AST + degrade path), `test_remote_backend.py` 25/25.

**Gates refitted, calibrated not tuned.** `eval/calibrate_chain_http.py` mirrors the documented webgate fitter (ECE temperature grid + coverage/accuracy accept threshold) with HTTP reads, on the webreplay CAL SPLIT ONLY (even lines; odd-line test split never opened — enforced by construction). 120 records: element acc 0.333, op acc 0.492, AUC(conf) 0.650, AUC(margin) 0.658, AUC(premask) 0.606 → fitted **temperature 1.0, accept_conf 0.337, answer_gate 0.30** (accepted reads 0.474 accurate vs 0.333 overall). The constants are much looser than the 4B's 0.70/0.50 because Gemma's read scores separate far less cleanly — exactly why per-model recalibration exists. Launch bug found by the live check: `load_remote()` rebuilt the backend dict with only the v3.13.0 keys, silently dropping `chain` (arm "absent" with no error, wire model fell through to a local-model path). Fixed + regression-tested; the chain-remote banner now gates every launch (`gates=0.337/1.0/0.3 from webgate_remote.json`).

**The measurement — same five tasks, three arms:**

| task | native Gemma 12B (v3.13.0) | chain-Gemma 12B HTTP (v3.14.0) | chain-4B in-process |
|---|---|---|---|
| T4 dropdown | 600 s timeout, fail | **22 s, PASS** (select Option 1, switch to Option 2, both real `<select>` mutations) | 11 s ✓ |
| T2 quotes | 132 s, fail | **64 s, PASS** (login → page shows Logout link; report backed by page evidence) | 24 s ✓ |
| T3 internet | 308 s, fail | **155 s, PASS** (login → "You logged into a secure area!", logout → "You logged out of the secure area!", reported verbatim; caveat: one post-completion Login misroute, self-recovered) | 23 s ✓ |
| T5 saucedemo | 601 s timeout, fail | **81 s, PASS** (sort low→high, clicked Add-to-cart @e35 — verified the Onesie $7.99's own button — badge 1) | 68 s ✓ |
| T1 books | 99 s, fail | **103 s, FAIL** (Travel→Home→next all executed; then a double-"next" misroute, a stale-ref click failed loudly, and the final turn's ~43.5k-token history exceeded the 32k slot window → backend 400 → session disposed) | 44 s ✓ |
| **score** | **0/5** | **4/5 (~7 min total)** | **5/5 (2:50)** |

The headline: **the chain mechanism transfers across model families AND across inference stacks.** The same decision chain that took the 4B from 0/5→5/5 takes Gemma 12B from 0/5→4/5 — with its reads served as grammar-forced generations over llama-server HTTP, gates refitted on the cal split (0.337/1.0/0.30 — the fitter's honest constants for Gemma's flatter score distributions), and the full run_chain logic (read window, pending step, ordinal snap, answer gates) untouched. Every executed action was adjudicated from page evidence in the session transcripts, not from exit codes.

T1's fail is half infrastructure: the proxy's remote fallback forwards the harness history verbatim, books.toscrape snapshots are ~8k tokens each, and 43.5k tokens > the 32k per-slot window (131072 ÷ `--parallel 4`) — llama-server 400s, the harness disposes. The 4B never faced this (in-process reads, faster turns, shorter histories). Documented follow-ups: (a) the remote fallback should elide oversized histories the way v3.11.8 does for single messages; (b) Gemma's reads mis-route more often than the 4B's (double-next = the element read re-picking a consumed step; post-completion ANSWER misses at the loose 0.30 gate) — the cost of a 0.65-AUC scorer vs the 4B's crisper one; the chain's guards (repeat brake, loud matcher failure) contained both without false actions.

Host restored end-state: chain proxy + llama-server stopped by ps-verified PIDs; standing proxy relaunched with the LOCAL 4B (v3.14.0, warmup ok, default config — no chain flag, local decide arm unchanged) and `poc-proxy/qwen35-4b-vanilla-nothink` answers `standing-ok`. Artifacts kept: sweep transcripts + .out under `/tmp/omp3arm/gemma_chain/`, cal records at `eval/reports/webreplay_records_http.jsonl`, gates at `configs/webgate_remote.json`, chain-remote decisions in `eval/reports/omp_arms/decisions.jsonl`; the 400 request preserved at `~/.omp/logs/http-400-requests/1789962813436-*.json` (local).

### 2026-09-21 — v3.14.1 → v3.14.3: "fix it and re-test" — T1 was four mechanism defects, not model capability; the fix chain ends with chain-Gemma 12B back at 4/5, every defect proven at the log level

**The question:** is T1's failure fixable plumbing or a capability wall? **Answer, with receipts:** plumbing. Four defects, each diagnosed from transcripts + the decision log, each fixed at its own layer, each verified in a live re-run.

**v3.14.1 — three defects, fixed and verified in the 11:52 re-test:**
1. *Context overflow* — remote_passthrough forwarded the harness history verbatim; T1's morning 400 was 16 messages ≈ 43.5k tokens against llama-server's 32,768-token slots (131072 ÷ `--parallel 4`). Fix: `slim_history` (the proven local policy) then `fit_char_budget` — 100k chars, largest-middle trim, newest tool result kept verbatim. Verified: zero 400s in every later run.
2. *Click matcher blindness* — the JS matcher read only `textContent`/`value`; the books site renders titles in `title` attrs and image `alt` texts. Fix: collect title attr + first img alt per element; exact tier → fuzzy tier (trimmed-ellipsis prefix match, bidirectional containment at ≥4 chars). Verified: `In Her Wake` clicked and the result snapshot was the real detail page.
3. *Exact-phrase anchor yank* — "next" (4 chars) appears verbatim inside "go to the next page of the catalogue", so the exact-phrase anchor overrode the correct pick with the pager link. Fix: `anchor_blocked(pend, idx)` — the pend target's index can never be yanked by the anchor. Verified: no anchor swap on the In Her Wake turn.

**v3.14.2 — the fourth defect the re-test exposed.** With the anchor fixed, the ordinal task phrase executed by the *positional* pend snap ("open the first book's detail page" → CLICK 'In Her Wake') was never consumed — the name shares no vocabulary with the phrase — so on the detail page the chain re-resolved the phrase against the product pager's own listing-shaped groups and wandered detail pages (wrong book reported). Fix: a tag round-trip. `action_code` emits `// click "In Her Wake" @e135 [task: open the first book's detail page]`; `extract_history` carries the tag; `_hist_actions` returns (name, tag) pairs; `pending_step`/`pending_phrases` consume a phrase by `_phrase_hit` OR exact tag match. The freed turn routes into the answer gate, whose report-clause fallback hands the page's own heading to the model. Verified: the tagged click in the transcript, no post-detail wander, answer "In Her Wake" — twice (v3.14.2 sweep and the v3.14.3 re-run).

**v3.14.3 — the fifth defect, found by adjudicating the other tasks.** T3's chain steps were all correct (including the tagged logout), but with the task done the model's op read re-clicked 'Login'; the logout redirect had shifted the button's ref (e21→e23), so the ref-based repeat check passed, the emptied form flashed "Your username is invalid!", and the model's report quoted that flash instead of the two success messages it had already seen. The same stray click sat unnoticed inside the *passing* v3.14.0 run. Fix: stand down when `pending_phrases` is empty AND the chosen CLICK's name appears anywhere in click history — a ref shift must not resurrect a finished step. Pagination re-clicks are untouched (they only fire while a next-page phrase is pending). Verified causally: in the v3.14.3 re-run the decision log shows the model re-picking Login @e23 and `stand_down=all-consumed-name-repeat` blocking it; T3 then answered "You logged out of the secure area!" — correct for the first time since v3.14.0.

**Version-over-version (chain-Gemma 12B, adjudicated from transcripts, not .out optimism):**

| task | v3.14.0 | v3.14.1 | v3.14.2 | v3.14.3 |
|---|---|---|---|---|
| T1 books | FAIL (ordinal re-resolution → wrong book) | FAIL (4th defect: untagged ordinal) | **PASS** (tagged click, "In Her Wake") | **PASS** (reproduced) |
| T2 quotes | PASS (read the Logout link) | FAIL (misread same snapshot) | FAIL (same misread) | FAIL (same misread, third time — action layer flawless, submit result showed Logout) |
| T3 internet | PASS (shrugged off the stray re-click) | FAIL (quoted the stray flash) | FAIL (same) | **PASS** (stray click blocked at the log level) |
| T4 dropdown | PASS | PASS | PASS | PASS (untouched path) |
| T5 saucedemo | PASS (badge 1, clean cart) | FAIL (ended on a plan, no answer) | FAIL (claimed badge 3; observed 2 — cart cookie left over from v3.14.1's run) | **PASS** (profile wiped, badge 1 observed and reported) |
| **score** | **4/5** | **1/5** | **2/5** | **4/5** |

The v3.14.1/v3.14.2 T2/T3/T5 flips are **model report variance, not chain regression**: the action layer executed identically in every version (verified turn-by-turn), and v3.14.0 proves the correct reports are reachable — the remote model's final write-up is bimodal about reading its own snapshots (quotes.toscrape accepts ANY credentials, so the login genuinely succeeded and the Logout link was the success signal; the model saw it on the good day and missed it twice after). T5's badge drift is environmental: saucedemo persists the cart in a cookie inside the shared headless browser's profile, so later runs start with prior runs' items; the profile was wiped before the v3.14.3 T5 re-run. Suites: `test_remote_backend.py` 41/41 (char-budget regression, tag round-trip, anchor guard, stand-down guard), `test_browser_v381.py` 19, `test_browser_v3110_ordinal.py` 46, `test_chain_remote.py` 9/9, `test_chain_http.py` 10/10, universal 25, coding 35, adapter 46, onboard 12/12.

Artifacts: v3.14.1/v3.14.2/v3.14.3 transcripts under `/tmp/omp3arm/gemma_chain/` (sessions 05-11…05-14Z = v3.14.2 sweep) and `/tmp/omp3arm/gemma_chain_v3143/` (T3 05-37Z, T1 05-38Z, T5 05-42Z, T2 05-43Z); stand-down row at ts=1789969084 in `eval/reports/omp_arms/decisions.jsonl`. T2's residual fail is the documented model-variance boundary of the chain: quotes.toscrape accepts any credentials, the submit's own snapshot shows the Login link replaced by Logout (curl-verified any-creds behavior), the chain typed and submitted flawlessly — and the remote model still wrote "unsuccessful" in 3 of 4 runs. Everything mechanical is proven; the last point is the model's reading comprehension, which v3.14.0 demonstrated is reachable on a good day.

### 2026-09-21 (evening) — MiniWoB++ 96-episode benchmark: chain 23% vs vanilla 21% — a tie; the interim "chain is worse" was small-sample noise; transcript-level root cause, two safe fixes (v3.14.4), and the checkpoint-review experiment measured

**The run.** 24 MiniWoB++ tasks × 2 deterministic seeds × 2 arms (`poc-proxy/gemma-12b-decide` vs `poc-proxy/gemma-4-12b-it`, same llama-server Gemma 4 12B NVFP4 behind both), interleaved per instance, objective CDP rewards, 420 s omp budget, 120 s episode cap. 96/96 episodes completed. Runner `/tmp/mw_run.py`, rows `main.jsonl`, transcripts `main_runs/`, tally `final_tally_v3143.txt`.

| arm | pass | avg wall | fail by "wrong" | fail by timeout/no-answer |
|---|---|---|---|---|
| vanilla Gemma 12B | **10/48 (21%)** | 107 s | 18 | 30 |
| chain Gemma 12B (v3.14.3) | **11/48 (23%)** | 223 s | 29 | 19 |

Paired per (task, seed): chain 5, vanilla 4, 39 ties — a coin flip, not a difference. Two findings matter more than the headline:
- **The benchmark's scoring asymmetry hides the chain's failure mode.** MiniWoB scores a *wrong* action −1 and ends the episode; a *failed* action (selector misses, no-op) costs time only. The chain's mistakes are confident and terminal (29 wrong-action fails vs vanilla's 18); vanilla's stumbles are inert. On short-horizon tasks where one click ends the game, the decision layer has no room to pay for itself.
- **The chain is 2.1× slower** (223 s vs 107 s): ANSWER was the top-1 op read on 78.83% of decide reads (283/359), standing the chain down while the clock burned.

**Root cause (transcript-level, four mechanisms).** `/tmp/mw_bench/investigation.md` has the full autopsy (`/tmp/mw_autopsy.py` over `main_runs/`):
1. **Submit-first ordering** (9 sessions): the first chain action is the terminal button before its prerequisite — submit before the form is filled, Login before credentials.
2. **Refusal-as-payload** (4 sessions): the TYPE/SELECT value composer emits *"I am sorry, but I cannot provide the text…"* and the chain types the apology into the page.
3. **ANSWER-stand-down** (above): 79% of reads say "no browser action needed".
4. **Matcher vocabulary gap** (140 errors): the click cell's selector covers `a, button, [role=button], input[type=submit|button]` — checkboxes and radios are unclickable (`no clickable element matching`).

Context: the 12B decide reads are near coin-flips on MiniWoB's tiny synthetic pages — mean top-1 confidence 0.340, median margin 0.048, 51% of reads inside a 5-point margin. The gates were calibrated on real-site replay snapshots, and MiniWoB sits below their operating resolution.

**Safe fixes vs overfit traps.** The working rule: a fix is safe only if it helps *any* agent on *any* site, not just this benchmark's score. MiniWoB scores abstention 0 above a wrong action −1, so ANY timidity tweak — recalibrating ANSWER on MiniWoB outcomes, blacklisting Submit/Login labels, parsing the reward display — raises the number while making the agent worse. All rejected. Two fixes qualified as bug-grade and domain-general → **v3.14.4**:
1. **Refusal/empty-payload guard** at the chain fire boundary: `bad_payload()` (refusal regex + empty check) → `stand_down` + passthrough to the model instead of typing an apology into a page. 15/15 probe sentences classified correctly, including the trap "I cannot say Vancouver" (a *value*, not a refusal).
2. **State-control coverage in the click matcher**: selector adds `[role=checkbox], [role=radio], input[type=checkbox|radio|image]`; name sources add `aria-label`, `<label for>`/wrapped labels, `aria-labelledby`, `name`; the pick prefers non-state controls so a "Submit" hit never lands on a same-named checkbox. TYPE restricted to text-entry controls (hidden/checkbox/radio/button/submit/image/file/range/color/reset excluded; password kept).

Validation end-to-end in a real browser: the old JS reproduces the exact benchmark error on seed 3887806147; the new JS clicks the checkbox and Submit → raw=1. TYPE finds 0 targets on checkbox pages, 2 on enter-password, fills "hd9mv". Suites unchanged and green (41 + 19 + 46 + 9 + 10 + 25 + 35 + 46 + 12). Deployed on the host with ps-verified kills, backup at `eval/omp_proxy.py.v3143`, live PID verified via `/v1/models`.

**The checkpoint-review experiment** (the "review the model's thinking at each checkpoint" idea, run to a verdict on both halves). Design: extract every assistant claim that asserts world state; against the newest page snapshot run two grammar-forced reads — CLAIM ("does the evidence support this claim?") and INDEPENDENT ("did the attempted task succeed?"); a **brake** = the asserted polarity contradicted by the independent read.
- *Labeled half* (6 adjudicated sweep sessions, 24 probes): brake fires 8/24; detection **6/9 (67%)**, false alarms **2/4 (50%)**. The claim and independent reads turn out CORRELATED — same model, same evidence, same blind spots — so the second read inherits the first's mistakes.
- *Objective half* (all 96 benchmark sessions → 815 probes, 596 with CDP ground truth; extractor regenerated to carry `evidence_turn` after the first 206-row extraction KeyError'd on the missing field): 473 polar claims, 175 wrong-polarity. Brake catches **117/175 (67%)** with **53/298 (18%)** false alarms. On final-report claims only: detection **77%** (67/87), FA **32%** (27/85). The claim read alone is near-uninformative (50% det / 48% FA) — the signal is entirely the independent read. Threshold curve is shallow: 0.5 → 67%/18%, 0.05 → 47%/6%.
- *Verdict:* real review signal — 77% of false success reports are flagged by a second read of the page — but not deployable as an automatic gate at this model size: a third of correct final reports would be challenged (59% on chain sessions). Deployable form: a log-only review flag for a second look, or ground the brake in objective evidence channels (the CDP reward the harness already sees) rather than another self-read. Full scores: `scores_labeled.jsonl`, `scores_miniwob.jsonl`, analysis `analysis_miniwob.txt`.

**V2 results (chain-only, same 48 episodes, v3.14.4):** **10/48 (21%)** — v1 chain scored 11/48, vanilla 10/48; paired vs vanilla 5/5/38, vs v1 chain 47/48 identical outcomes (the one flip, `enter-text-2` s3487846274 pass→fail, is the model's stochastic layer, not the diff: the changed code paths never fire on that task). Avg wall 219 s (v1: 223 s). The fixes are confirmed real at the mechanism level — refusal payloads **7 sessions → 0**; click-checkboxes matcher errors **55 → 17** on the same episodes, and those episodes changed character from inert timeouts to actual checkbox attempts (one now earns fractional credit −0.33 where v1 scored None) — but the aggregate is unmoved: MiniWoB's terminal-wrong, short-horizon structure cannot see bug-grade fixes either. Three arms, one conclusion: vanilla 10/48 ≈ chain v1 11/48 ≈ chain v2 10/48, with the chain at 2.1× wall time and a different (more terminal) failure mix. The decision layer's value case remains the stateful multi-step regime where it was already proven (sweeps, real sites), not this benchmark.

### 2026-09-21 (evening) — v3.15.0: the six-lever completion-hygiene pass, under an explicit overfit audit

Commissioned as "do all of the improvements, make sure it doesn't overfit." The audit discipline, stated up front and applied to every change: MiniWoB scores *abstention* 0 above *wrong action* −1, so any timidity tweak raises the score while making the agent worse. A change is accepted only if (a) its logic reads page structure and task text only — never reward, benchmark vocabulary, or site names; (b) acceptance is held-out — the offline suites plus the five-real-site sweep, with MiniWoB numbers reported as sanity only; (c) it can fail honestly in both directions — a change that could only raise the benchmark score would be timidity in disguise.

What shipped (all six levers, as routed-not-retreating designs):
1. **Submit-prerequisite gate (routing, never blocking)** — inside the CLICK cell, before clicking: if the click would submit a form that the *browser's own* constraint validation would reject (`willValidate && !checkValidity()`, `novalidate`/`formNovalidate` honored, non-visible controls ignored), the cell throws `submit-precondition: N form field(s) invalid/empty: <names> — fill them, then submit`. The throw lands in the tool result = routing with the reason, the model fills the named fields and resubmits. Two engine facts learned live: the IAB webview returns `undefined` for the `HTMLFormElement.novalidate` IDL property with the attribute present (the opt-out is read from the attribute as well), and the gate must ignore `type=hidden` (validation-relevant but never visible). Browser-validated end-to-end on a purpose-built 5-scenario page (empty required form throws with both field names and nothing submits; novalidate/plain-button/prefilled forms click through).
2. **Task ledger** — `pending_phrases(task, hist)` state is now written into every decide row of `decisions.jsonl` (`ledger: {pending: [...], n}`): what the chain believes is still owed, per turn, in the audit log. Enforcement was already the v3.8.0 pending override; this is its observability.
3. **Completion-review flags (log-only)** — the two deployable signals from the checkpoint-review experiment: `completion-unverified:answer-gate-fallthrough` on an uncorroborated stand-down, `acted-under-answer` on a fire under an ANSWER op read. Never a gate (the objective-ground-truth ROC said 32% false alarms on correct final reports); they exist so the harness's objective channel — or a human reviewer — knows which final reports to double-check.
4. **ARIA role coverage** — click matcher now also addresses `[role=tab], [role=menuitem], [role=link], [role=option], [role=treeitem], [role=switch]`, same class as the v3.14.4 checkbox fix: driven by the ARIA contract, not by any benchmark task. Widget-page validated: all six roles fire the cell. (The MiniWoB `click-tab` residual was NOT this: its jQuery UI tabs are runtime `role=presentation` anchors already covered by `a` — the real-page test clicks "Tab #1" cleanly; see item 6 for the actual bug those transcripts hid.)
5. **Read-batching audit** — confirmed candidate scoring was already client-side thread-pooled over grammar-forced generations (`eval/chain_http.py`); the 2.1× wall overhead is stand-down duration, not read latency. Nothing to change; documented rather than duplicated.
6. **The case bug (found by validation, one line)** — the CLICK matcher lowercases every candidate string but never lowercased the wanted name, so a model-emitted page-cased name (`Tab #1`, `Close`, `Submit` — 22+ click-miss errors in the benchmark transcripts, part of the 140-error matcher-gap mechanism) could never match the DOM's lowercase text. `want` is now lowercased like every candidate. Strictly converts throws into clicks for case-mismatched names only; cannot increase timidity. The click-tab mystery is closed by it: real-page test reproduced the pass, the transcripts' capitalized misses were this bug.

**Validation.** Offline: all nine suites green against the new file (41 remote-backend + 10 chain-http + 9 chain-remote + 19 browser-v3.8.1 + 46 ordinal + 25 universal + 35 coding + 46 adapter + 12 onboard = 243 cases), plus the node CLICK-matcher suite. Held-out: the five-real-site sweep on the 12B llama-server stack, adjudicated from transcripts (never exit codes): **4/5 correct** — T1 books (correct title "In Her Wake" after Travel → home → page 2 → detail), T3 internet (logout confirmation quoted), T4 dropdown ("Option 2"), T5 saucedemo (sorted low-to-high, cheapest $7.99 Onesie added, badge 1); T2 quotes failed on the deadline by a model-layer loop — the chain's login submit *succeeded* (logout link visible), the model misread that state, navigated back to /login, and burned the remaining budget fighting the site's CSRF-reloaded form. Same score as the v3.14.3 baseline (4/5, different failing task — T1 then, T2 now — stochastic layer). Mechanism evidence from the same sweep: 21 ledger rows in `decisions.jsonl`, 4 review flags (3 gate-fallthrough, 1 acted-under-answer), the submit gate present in every click cell and correctly never firing (fields were always filled before submit — it is insurance for the submit-first ordering, browser-validated, and its zero fires here are the healthy outcome, not a skip). MiniWoB sanity (reported only, chain arm, 4 episodes on the two form tasks where submit-ordering matters): **login-user 1 pass / 1 deadline-wrong; form-sequence 0/2** — both form-sequence deaths are model-layer (the slider is not a chain primitive and the model claimed success anyway; 62-106 s wrong-value claims, no submits involved). The telling number is in the audit log: 11 review-flagged decide-turns in the sanity window, 8 of them `acted-under-answer` on the failing login-user session whose ledger still showed 3 pending phrases — the uncorroborated-completion signature is loud in the log and invisible to the score, which is precisely the deployable form the checkpoint-review experiment predicted.

### 2026-09-21 (night) — v3.16.0: the completion self-review becomes a config option; retirement decisions recorded

Two decisions from the keep/remove review, both now made real:

**1. Self-read gating is an option, not a removal — the learned-adapter pattern.** The user's call: keep the completion self-review but make it enabled/disabled by config, exactly like the learned-adapter loop. Shipped as a `completion_review` block in the backend config (`proxy_chain_remote_review.json`), same contract as `adapter_learning.json`:
- **missing, or `enabled: false`, or junk = byte-for-byte v3.15.0.** The proxy binds `COMPLETION_REVIEW = None` and never reads the model an extra time; the fallback path is untouched code.
- `mode: "log"` — on the review-flagged stand-down (the `completion-unverified:answer-gate-fallthrough` case), the chain runs one independent completion read ("has the task's goal been fully completed?", candidates DONE/NOT DONE, same grammar-forced channel as every decide read) and records `review_read: {p_done, mode, method}` on the decision row. Pure observability: the 77%-detection signal lands in `decisions.jsonl` where the objective channel can use it.
- `mode: "gate"` — everything log does, plus: when the independent read *confidently* says NOT done (`p_done < p_done_threshold`, default 0.5) **and** the chain still holds a concrete alternative action (the `ANSWER_GATE_FALLBACK` cell stashed at the handoff), the challenge fires — the stashed cell ships as the turn's action (`review_gate: "challenge:continue"`) instead of the stand-down. A challenge requires a real alternative; when there is nothing better to do than the model's answer, the chain stands down regardless. A failed read degrades to passthrough with `review_read.error` logged — the option can only ever route, never block the turn.
- Default OFF, deliberately, and the default is the honest posture: the objective-label experiment measured **77% detection of false success reports against 32% false alarms on correct ones** — at this model size the gate is offered, never assumed. The predicate (`review_gate_fires`) is a pure function with a full truth-table test; unknown `mode` strings fall back to `log` (the measured-safe posture); `p_done_threshold` is configurable.

**Validation.** New suite `test_review_gate.py` (18 cases): config normalization (absent/disabled/junk → off, unknown mode → log, threshold parsing), the predicate truth table, the hook's placement after the fallthrough flag, its guard on `COMPLETION_REVIEW`, degrade-on-error, the challenge return shape, and the main() binding. All ten suites green, 261 cases. Live validation runs against the 12B llama-server stack with `mode: "gate"` on the T2 quotes task — results recorded below.

**2. Retirement decisions (from the keep/remove review, accepted).** Recorded here so the codebase's intent is explicit — none of these is a code deletion today; they are declared postures:
- **Kept (the load-bearing core):** the grammar-forced micro-read decision core + chain state tracking (pending phrases, ordinal snap, compaction handoff) — the proven value case on real multi-step tasks; the routing-never-blocking trio (submit-precondition, pending override, junk fallback); the measurement instrument (decisions.jsonl, review flags, ledger); the serving bridge + calibration fitter (llama-server HTTP + `calibrate_chain_http.py`).
- **MiniWoB demoted to a smoke test.** Three arms, one conclusion (vanilla 10/48 ≈ chain v1 11/48 ≈ chain v2 10/48, chain 2.1× wall): the benchmark's short-horizon, terminal-wrong scoring cannot see the layer's value case, and tuning for it would be overfitting by definition. It stays in CI as a smoke/regression harness only; all acceptance claims are held-out real-site sweeps. MiniWoB numbers are ever reported as sanity.
- **The learned-adapter loop is frozen (dormant, config-gated).** Fully built (capture → build → validate → register → route) but unproven on any real workload; the config contract already guarantees absent-config = inert. No further investment until a concrete family shows up where a phrase→action spec beats the model.
- **The universal tier is dormant.** The routing skeleton (browser/coding/floor) works and costs nothing at rest, but no new surface work until a real workload demands it.
- **The local 4B generation stack is queued for retirement** when the standing proxy migrates to a served backend (the 12B llama-server is the working pattern): the in-process weights loader, the plain-greedy decode loop and the WSL memory fight (v3.11.x sagas) all exist only to serve a model the remote arm now replaces. Retirement means deleting the *activation path*, not the history — the changelog stays.

### 2026-09-22 — the single-model framing backtest: the checker's weakness is the question, not the model

The user's constraint — **one model, no second opinion** — turned the decorrelation question into a measurable one: if the completion check's weakness is that both reads see the same snapshot through the same frame, can the SAME model, reading DIFFERENT evidence, beat its own baseline? The saved objective probe set made this free: 815 probes (596 with ground truth after the same skips as the original analysis), each already carrying the claim, the stored claim-read verdict, and the episode's objective outcome (`main.jsonl` CDP raw). Machinery: `/tmp/ckpt_reframe.py` (same grammar-forced recipe as `ckpt_score.py`, rows in probe order, resumable) re-read every probe through three framings, one variable each; `/tmp/ckpt_reframe_analysis.py` joins and cuts. Run on the 12B llama-server over an ssh tunnel, ~35 min, 815/815 rows, 0 errors.

The framings (all the same model, same decode, ~1 s per read):
- **GOAL** — the evidence snapshot **plus the task's goal line** ("Log in with the given username and password"). The baseline read never knew what the agent was trying to do; a real agent always does.
- **DELTA** — **BEFORE (session's first snapshot) + AFTER (newest)**: change evidence. A successful login is invisible in a static snapshot but obvious in a diff (a Logout link appeared).
- **COND** — the baseline snapshot **plus the stored claim-read verdict** ("a separate check judged the agent's claim as supported/not supported"): the completion read conditions on the claim check instead of re-deriving from scratch.

**Results at the baseline operating point (T=0.5, flag = read contradicts the asserted polarity):**

| framing (all-polar det / FA) | det (of 175 wrong) | FA (of 298 correct) |
|---|---|---|
| baseline p_done (reference) | 117 (67%) | 53 (18%) |
| GOAL | 121 (69%) | 55 (18%) |
| DELTA | 100 (57%) | 38 (13%) |
| COND | 129 (74%) | 47 (16%) |
| **min(GOAL, DELTA, COND)** | **137 (78%)** | **52 (17%)** |
| mean(GOAL, DELTA, COND) | 115 (66%) | 34 (11%) |

Final-report cut: baseline 77%/32%; COND 79%/32%; min-agg **85%** det at 39% FA; mean(done,goal,delta) 67% det at **22%** FA (the conservative recipe). Per-arm at T=0.5, all-polar — the deployment-relevant cut: **on chain sessions the GOAL read alone moves 66% det / 34% FA → 78% det / 21% FA** (78/100 caught, 20/96 false), and min-agg reaches 84/30; on vanilla sessions DELTA has by far the best FA (7/202). Framing disagreement confirms real decorrelation: mean |p_a − p_b| is 0.18–0.28 across framing pairs — the same model, asked differently, disagrees with itself by a quarter of the probability mass, and the min-aggregate converts exactly that disagreement into catches.

Reading it honestly:
1. **The single-model thesis has measured headroom.** The strongest single-model recipe (three cheap forced reads, require all to say done) catches **+11 more wrong claims at the same false-alarm rate** as the baseline — with no second model, no training, ~2 s of extra reading per flagged stand-down. The "checker shares the actor's blind spots" conclusion stands corrected: it shares the snapshot, not the fate — vary the frame and the blind spots vary with it.
2. **DELTA alone is the most conservative read** (best FA everywhere: 13% at 0.5, 9% at 0.7) but the weakest catcher standalone — its value is inside the aggregate, where it vetoes premature "done" verdicts. Consistent with the ground-truth definition (final episode raw): intermediate states genuinely aren't done.
3. **Threshold matters as much as framing**: the baseline read at T=0.7 already reads 77%/17% all-polar (vs 67/18 at 0.5). The original experiment froze T=0.5; the curve says the operating point was conservative.
4. Deployment shape, ready but not shipped: the `completion_review` read becomes the 3-read min (goal + delta + cond), same config gate, ~2 s extra on the flagged stand-down path. The backtest says that change is worth roughly +11 points of detection for free; a held-out real-site sweep with the review proxy would be its acceptance run.

Caveat, unchanged: this is still the same model family self-grading against MiniWoB-shaped pages with final-episode ground truth; the numbers grade the *checker*, not the agent, and acceptance for any shipped form remains a held-out real-site sweep.

### 2026-09-22 — v3.16.1: the 3-frame completion review deployed; live test finds the trigger-site coverage fact

The backtest's deployment shape shipped the same day it was measured. In `run_chain`'s answer-gate fallthrough (the uncorroborated stand-down site: element gate failed, nothing pending, the model is about to answer), the single DONE/NOT-DONE read is replaced by the backtest's three frames, each a grammar-forced yes/no read through the same `DECIDE_FN`:

- **GOAL** — task goal line + newest snapshot ("Did the attempt achieve the goal?").
- **DELTA** — the session's FIRST snapshot (new per-conversation `REVIEW_FIRST_OBS` capture, bounded at 32 entries) vs the newest ("Based on what CHANGED…").
- **COND** — the snapshot alone, after a claim-check read of the model's own last prose is folded in as a verdict line.

The gate predicate is unchanged and now consumes **p_min = min(goal, delta, cond)**; `review_read` logs all four numbers. Config, default-OFF posture, log/gate modes, and degrade-on-error are byte-identical to v3.16.0. `test_review_gate.py` extended to 31 checks (capture semantics: dedupe-first/compaction/bounded-oldest-dropped; `review_last_prose` extraction; the four prompt phrasings; the min and predicate wiring); `test_chain_remote.py` DECIDE_FN count 4 → 7. All ten suites green, **274 cases**. Deployed to the GPU host with the v3.16.0 copy backed aside (md5 `1ed932b3…` both sides), proxy relaunched on `mode: "gate", p_done_threshold: 0.5`, binding line verified in the live banner.

**Live test on the 12B (honest adjudication — the feature never fired, and that is the finding):**
1. **T2 quotes, full task** — the chain drove all four actions and the login objectively succeeded (Logout links in results #5/#6); the model then navigated *back* to /login and reported "Login failed" — the same model-layer report failure as v3.16.0's live run. But every late answer turn had the element gate **pass** (`answer_gate ok=True`, routed passthrough-repeat/passthrough), so the fallthrough site — the review's only trigger — was never entered. No `review_read` row.
2. **Truncated task (open site → login page → type username; no password/submit/report clause)** — the model did exactly what was asked and reported it truthfully (a clean completion, and it did not invent the missing steps). Its final text turn produced **no decide row at all** — the request bypassed the decide chain entirely. Again no review row.
3. **Live replay of the hook's exact four reads** against the served 12B on run 1's real artifacts (real task, real BEFORE/AFTER snapshots, real wrong claim): **p_goal 0.067, p_delta 0.116, p_cond 0.254 → p_min 0.067** (challenge would have fired, had the site been reached), and **p_claim 0.963** — the claim check *vindicates* the model's final-page description. That is the day's sharpest datum: the model's error was about **history** (the login succeeded at click time; its own back-navigation poisoned the final snapshot), not about the final page. The review grades the final snapshot, so on this transcript it would have flagged a truly-done session — one live instance of the ~17–39% false-alarm rate already measured at T=0.5, and a concrete example of *which* session shape pays it.

**The coverage fact, recorded as a finding rather than a fix:** the review covers exactly one uncorroborated-report site — the gate-failed fallthrough. Today's failure happened on a gate-passed answer path, and run 2's honest answer bypassed the chain entirely. Extending the hook to the gate-passed ANSWER path (where a fallback cell also exists) is the obvious follow-up, but run 1's replay is the caution: there the challenge would have shipped a re-click against a page the model itself had drifted away from. The right acceptance for that extension is a dedicated real-site sweep, not this transcript. Standing 4B proxy restored and `standing-ok` verified after the test.

### 2026-09-22 — the HISTORY frame: a pre-registered negative result (deployed nothing, learned two things)

The natural follow-up to the coverage finding: the model's live failure was *forgetting its own success* (success evidence appeared mid-session, the model navigated away, denied it), and all three deployed frames read only the NEWEST snapshot — so a fourth frame, **HISTORY** ("did the goal appear to be achieved at **any point** in the session's observed pages?", reading the ordered page list), was the obvious candidate. The user asked explicitly for no overfit fix, so the experiment ran **pre-registered**: framing, evidence rule (all distinct page states in order, consecutive dups collapsed, 1200-char page cap, 8000-char budget newest-last), thresholds {0.3, 0.5, 0.7}, cuts, and the acceptance bar were frozen in the scorer header **before any result was seen**; a mechanical guard asserts the template adds no site/task vocabulary beyond the GOAL frame's existing measurement-only goal lines; one pass over all probes, no iteration on outcomes. Machinery: `/tmp/ckpt_histframe.py`, `/tmp/ckpt_hist_analysis.py`, results in `/tmp/mw_bench/scores_hist.jsonl` (815/815 rows, 0 errors, ~20 min on the 12B).

**Result: DO NOT DEPLOY, by the frozen rule.** At T=0.5 all-polar, min(goal, delta, cond, hist) measures **138/175 (79%) det / 54/298 (18%) FA** vs min(goal, delta, cond) **137 (78%) / 52 (17%)** — **+0.6 det for +0.7 FA**, nowhere near the required +5 det at ≤ +2 FA. HISTORY standalone is 69% det / 17% FA — statistically indistinguishable from GOAL, no special value anywhere in the grid (it is also the frame the *motivating case* failed live: on the real quotes transcript, p_hist_yes = 0.344).

Reading it honestly, two mechanisms, both recorded:
1. **The target class is small and no frame catches it.** Wrong *failure* claims (success happened, model denied it) are 30 of the 175 wrongs; baseline catches 2, the best single frame 9, HISTORY 3. The evidence for forgotten success is subtle, transient, and buried — one line of Logout in ~7,700 characters of accessibility tree. A cheap single read over a page dump cannot isolate it; per-page reads or an extractive pre-pass might, but those are hypotheses for a *future* pre-registered experiment, not a same-day tweak.
2. **The min aggregation can only consume pessimism.** Adding a read to a min can never raise it, so a 4-frame min can only gain det by paying FA; HISTORY's one real virtue — it is excellently calibrated against phantom success (3/244 false alarms on correct failure claims, 1.2%) — is exactly the direction a min cannot use. An aggregation that could listen to a *conservative optimist* (e.g., require pessimism from ≥2 frames rather than all) is the interesting open question this leaves behind.

Where this leaves the coverage finding: the review's fallthrough-only trigger site and the gate-passed answer path remain the real open items; the history-frame idea was the cheap candidate for them and is now measured and closed rather than shipped on intuition. The 12B stack ran only for this measurement and was restored immediately after — standing 4B proxy back up, standing-ok verified.

### 2026-09-22 — the voting-rule experiment: a pre-registration error, caught by the data, and what it taught

The history experiment left an open question ("can the aggregation listen to a conservative optimist?") that is answerable offline — the five read probabilities already exist for all probes — so a second pre-registered analysis ran the same day, no GPU, no new reads: symmetric k-of-N pessimism voting over the three deployed frames (`/tmp/ckpt_kofn_analysis.py`, frozen variants k∈{1,2,3}, frozen deployment bar: k=2 must land within −1 pt det at ≤ −3 pts FA, near-misses count as misses).

**The pre-registration contained a factual error, and the data caught it.** It labeled the deployed reference rule "3-of-3 pessimism" (all three frames must flag). The deployed gate actually fires when `min(g,d,c) < threshold` — and since a minimum dips when *any one* input dips, the deployed rule is **"any single dissenter vetoes" on success claims** (and, symmetrically, *all three* must dispute a failure claim for the flag to fire there). The decomposition proves it: the established min result (137 det / 52 FA) equals the measured 1-of-3 behavior on wrong success claims (135/145) plus the 3-of-3 behavior on wrong failure claims (2/30). A min is a scalar statistic, not a vote; its count-equivalent flips with the claim's polarity. The frozen comparison was therefore void — neither side of it is the deployed rule — and per the discipline nothing was re-run with a new bar after seeing numbers: the printed verdict (KEEP MIN) is kept as the outcome (nothing deploys), its premise retracted, and the correction is appended to the analysis script itself.

**Two real findings survive, both products of the error:**

1. **The vote-count dial, measured on the direction that matters.** On wrong *success* claims (145 of 175 wrongs) and correct *success* claims (54), the family traces: any-one-vetoes (the deployed rule) **93% det / 93% FA**; 2-of-3 **77% / 48%**; all-three **57% / 7%**. No point dominates another — the vote count is an operating-point dial, not a free improvement. The deployed rule sits at its most aggressive end.
2. **The topline false-alarm numbers flatter the gate.** The headline "17–18% FA" mixes claim directions: 244 of 298 correct claims are *correct failure* claims, where a done-leaning read rarely fires. On correct *success* claims — the subs a stand-down gate actually operates on — the deployed rule challenges ~93% (bench-graded) at T=0.5, worse than the baseline read's 74% on the same subs. The 78%/17% toplines remain valid as *claim-checking* scores; they must not be read as the gate's expected challenge rate on correct completions. This strengthens, with data, the posture the feature was shipped under: default OFF, offered never assumed, and no gate mode anywhere until a held-out real-site sweep measures the challenge rate on real correct completions.

Also resolved: the "conservative optimist" question is half-answered — the frontier exists and is now quantified, but nothing dominates; a rule that beats the deployed one in both directions does not appear in the frozen variant set, and choosing one after peeking is exactly the overfit this program forbids. Fresh data (the held-out sweep) is the honest next instrument. No code changed; the standing 4B proxy was never touched in this step.

### 2026-09-22 — v3.16.2: the coverage fix ships, plus the annotate posture; held-out real-site sweep PRE-REGISTERED (before running)

Three experiments burned the dev set (596 probes, now exhausted); the only honest instrument left is fresh real-site data. v3.16.2 builds the instrument and pre-registers its acceptance rules before the first session runs.

**Code shipped (all ten suites green; review-gate suite now 41 checks, chain-remote 11):**
1. The four review reads move out of `run_chain` into a shared module-level `review_frames_run(task, msgs, obs_text, log, site)` — pure reads, records `review_read` (now tagged with the trigger `site`), returns `(p_dict, p_min)`, read failure degrades to log-only. Gating stays with the caller.
2. **Second trigger site, log-only**: the gate-PASSED answer stand-downs — `all-consumed-name-repeat` and the `passthrough-repeat` exit, both guarded `forced is not None`, tagged `site="answer-ok-standdown"`. This is exactly the path the 2026-09-22 live run's wrong login report slipped through (gate passed everywhere; the fallthrough-only review never saw it).
3. **Third trigger site, log-only**: the no-candidates answer exit (`site="no-candidates"`, guarded `forced is not None and obs_text`).
4. **`mode: "annotate"`** joins log and gate: never gates, never challenges. At the fallthrough site, when ALL three frames read NOT-done (`p_goal, p_delta, p_cond < p_done_threshold` — the vote-dial's measured ~89%-precision operating point), the turn stashes `REVIEW_NOTE`; `compute()` appends it to the model's own prose answer as `[verification note] independent page checks did not confirm the task's completion; treat this report as unverified.` Tool-call turns and malformed answers pass untouched. The direction-split audit (93% would-challenge on correct success claims) is the data behind making the strictest read a *caveat*, not a block.
5. Challenge stays fallthrough-only, gate-only; default OFF unchanged; `mode` normalizes unknowns to `log`; `annotate` never fires the gate predicate (tested).

Deployed: v3.16.1 backed aside (`omp_proxy.py.v3161`, md5 `1ed932b3…`), new file md5 `c37a3387…` both sides, host syntax-checked, banner verified. For the sweep the config runs `mode: "log"` (gate config backed aside as `proxy_chain_remote_review.json.gate.v3161`).

**The sweep, frozen before the first session runs.** 8 sequential fresh sessions on the 12B + review proxy, one task per fresh session dir, via `timeout 480 omp -p --model poc-proxy/gemma-12b-decide --auto-approve --max-time 360 --session-dir /tmp/<dir> --no-title "<task>"`:
1. quotes.toscrape.com — log in as admin/supersecret123 (full task, report clause).
2. quotes.toscrape.com — open the site, go to the login page, type the username (truncated: no password/submit).
3. quotes.toscrape.com — go to the login page and confirm it loads (login page only, no typing).
4. books.toscrape.com — open the Travel category and open the first book's detail page.
5. news.ycombinator.com — report the title of the top story.
6. en.wikipedia.org — search for Ada Lovelace and open the article.
7. books.toscrape.com — from the Travel category, report the first book's title (report clause exercises the text-fallback path).
8. news.ycombinator.com — open the item at position 3.

**Adjudication, from transcripts + decisions.jsonl only (never exit codes):** each session's final report is labeled by objective evidence (correct success / wrong success / correct failure / no claim). From the logged rows: `review_read` count by site (coverage), p_min at final answer turns, and the exact offline counterfactual of gate mode — `would_fire = site=="answer-gate-fallthrough" AND p_min < 0.5 AND "gate_fallback" in row` (the stash and its log line are set in the same fallthrough path, so this is the predicate, not a proxy). Annotate counterfactual: all three < 0.5 at any review row in the session.

**Acceptance bar, frozen:** gate mode may be *offered* (still never default) only if the sweep shows ≥ 1 real catch (would_fire on a wrong success claim) with ≤ 1 would_fire on correct success claims. Annotate may be *offered* only if its would-flag rate on sessions whose reports are correct is ≤ 50%. Anything else: the posture stays log-only/OFF and the number is recorded as the measured answer. No iteration: tasks run once, no threshold or prompt changes after results; an ambiguous sweep resolves to OFF.

### 2026-09-22 — the held-out sweep adjudicated: the coverage fix works in the wild, and the sweep's one real wrong claim is exactly what the gate would have caught

The 8-task sweep ran end to end on the 12B + review proxy in **log mode** (nothing gated, nothing annotated — pure measurement, as frozen). Decisions delta: 52 rows after the sweep marker; transcripts adjudicated per task from session JSONL.

**Per-session labels and review reads:**

| task | final report (adjudicated) | review site | p_goal/p_delta/p_cond → p_min | gate would-fire | annotate would-flag |
|---|---|---|---|---|---|
| t1 quotes login (full) | **CORRECT success** — Logout link in the final snapshot, model claimed success | answer-ok-standdown | 0.002 | no (log-only site) | no |
| t2 quotes truncated | **CORRECT** — `textbox: admin` in snapshot, model reported exactly that | answer-ok-standdown | 0.336 | no | no |
| t3 login page loads | **CORRECT** — form displayed, model said so | — (turn never hit a review site) | — | no | no |
| t4 books → Travel → first detail | **no report** — HTTP 400 death; page trail shows it opened the *home* page's first book, never Travel's | answer-gate-fallthrough | 0.003 | **yes** | yes |
| t5 HN top story | **CORRECT** — "I said no and Apple said yes" is row 1 in the snapshot | answer-gate-fallthrough | **0.0** | no — *no fallback cell stashed* | no |
| t6 Wikipedia Ada Lovelace | **no report** — HTTP 400 death; reached the right article, then drifted to "Ada Lovelace (microarchitecture)" | answer-ok-standdown | 0.004 | no | no |
| t7 Travel first book title | **CORRECT** — "It's Only the Himalayas" matches the heading in the snapshot | answer-ok-standdown | 0.276 | no | no |
| t8 HN item at position 3 | **WRONG success claim** — reported "Can gzip be a language model?" as position 3; gzip is row **2**, row 3 is "MiMo v2.6" | answer-gate-fallthrough | **0.0** | **yes** (fresh-pick cell stashed: CLICK "Language Modeling is Compression") | yes |

**Findings, against the frozen bars:**

1. **The coverage fix demonstrated itself.** 4 of 8 review reads fired at the NEW `answer-ok-standdown` site — the gate-passed answer path where the 2026-09-22 live run's wrong report slipped through unreviewed. v3.16.1 would have logged zero review reads on those four sessions. The 3-frame rows carry `site` tags end to end; read failures: none.
2. **Gate bar (≥1 real catch, ≤1 interruption on correct completions): MET.** The sweep's single wrong success claim (t8) is exactly the would-fire: p_min 0.0 at the fallthrough with a concrete alternative action stashed. Zero would-fires on the four correct success reports. One hair-raiser recorded honestly: **t5, a correct claim, read p_min 0.0 at the fallthrough** — all three frames said NOT-done on a true report; the only thing that would have prevented a false challenge is the absent fallback cell (a pure report task stashes no click). The cell requirement did real work as a false-alarm brake on this sample, which is a mechanism, not a guarantee.
3. **Annotate bar (≤50% flags on correct reports): MET** — 0 of 5 correct-report sessions would have been flagged; both would-flags landed on sessions whose end state was genuinely wrong or dead (t4, t8). The strict all-three-below-threshold point behaved as the vote-dial measurement predicted.
4. **Design split validated on n=8:** correct final reports arrived at the gate-PASSED site (log-only, by design) — t1 at p_min 0.002 proves a correct completion can read near-zero, so gating the answer-ok site would have been a false-alarm machine; the wrong claim arrived at the gate-FAILED fallthrough, where the challenge is allowed. The v3.16.2 choice to keep site 2 log-only looks right on this sample.

**Posture (unchanged by decision, changed only by evidence):** both frozen offer-bars are met, so gate and annotate are now *measured* postures, not just shipped code — but the sample is 8 sessions and 1 wrong claim, the default remains OFF, and the standing 4B path runs without the feature at all. Nothing in production changed; the review config stays `mode: "log"` on the experiment stack.

**Infra note (not a chain finding):** t4 and t6 died mid-session with `HTTP 400` from the model server — request payloads of ~130–160 KB (~40k tokens) against the 12B's 32k-per-slot context (`n_ctx_slot = 32768`, `-np 4` on a 131k window). The known big-page overflow failure mode; the response body isn't captured in the 400 log, so overflow is inferred from request size + slot config, not read off an error string. Any future live run on the 12B should either raise the window per slot or expect long web-page sessions to die this way.

Standing 4B proxy restored immediately after the sweep (`warmup ok`, 4B model family serving, banner healthy). 12B llama-server and review proxy stopped — they ran only for the sweep.

### 2026-09-22 — rubric read PRE-REGISTERED (4-level ordinal read from the same model; frozen before any GPU run)

Motivation: the survey of the reflex-model genre (AgentJev, Laya) plus the sweep's floor disease (t5: a true report reading p_min 0.0; t1: 0.002) motivate one borrowed idea testable without training and without a second model — replacing the binary done/not-done read with a 4-level ordinal read scored as an expected level. Also recorded here, a DOWNGRADE of a second borrowed idea: per-read temperature fitting cannot address the measured failure — the gate consumes raw scores against thresholds, a global monotone rescale changes no threshold's det/FA, and the observed errors are saturated ties (0.0) in BOTH directions, which no single temperature separates. Kept only as a future add-on if reported probabilities are ever needed.

Design (frozen before the first read):
- **Question:** can a 4-level ordinal read (far from done / partly done / nearly done / fully done) of the SAME evidence from the SAME 12B give a better claim-checking dial than the binary INDEPENDENT read, and a less saturated floor on correct reports — zero training, no second model?
- **Frame:** stem identical to the deployed-style INDEPENDENT read (`ckpt_score.py done_question`): "The following is a web page's accessibility snapshot after an attempt to complete a task on the page." + Evidence block; the final question becomes "How far along is the attempted task, based only on this snapshot?"; Options 1–4 in the fixed order above; same answer instruction sentence as the baseline reads.
- **Evidence rule:** identical to the baseline (`build_evidence`: newest tool-result snapshot up to the probe's evidence turn, whitespace-compacted, 8000-char cap). All 815 probes, same order, resumable.
- **Recipe:** identical grammar-forced choice — temperature 0, GBNF lock per candidate, summed raw logprobs, alpha=1 length norm, softmax over the 4 candidates; model `gemma-4-12b-it`, thinking disabled.
- **Score:** E = Σ i·p_i with i = 0..3 in option order (0 = far from done).
- **Flag rule:** mirrors the binary dial exactly — a success claim is flagged iff E < T; a failure claim is flagged iff E > T. Frozen grid: T ∈ {1.0, 2.0} on the ALL-polar cut. Descriptive, non-gating: per-arm split, FINAL-report cut, direction split, floor disease (share of correct-success claims with E < 1.0 vs the baseline's share with p_done < 0.25), and an E histogram on correct vs wrong claims.
- **Acceptance, frozen:** PROMISING iff ∃ T ∈ {1.0, 2.0} with det ≥ 67% AND FA ≤ 13% on ALL-polar — i.e. Pareto-beats the baseline INDEPENDENT read's measured 67/18 by ≥ 5 FA points at ≥ equal det. Comparison against the three-read min(g,d,c) point (78/17) is recorded as context only; a single read matching it would be notable but is NOT the gate. Anything less: record the negative, deploy nothing.
- **No iteration:** prompt, options, and grid are frozen above; one pass; no rewording, no added thresholds, no subsetting after results. The bench set is dev-grade evidence; no deployment decision rides on this alone.

### 2026-09-22 — reflex-sidecar kill-test PRE-REGISTERED (off-the-shelf checkpoints; frozen before any download)

- **Question:** do the off-the-shelf reflex checkpoints (AgentJev-0.6B; Laya English, ModernBERT-large 421M) carry ANY browser-domain signal, zero-shot, on our claim-checking question? Expectation stated up front: likely near chance — Laya's own card puts its base below the majority class zero-shot, and both projects' published skill comes from fine-tuning on four non-browser workflows. This is a kill-test, not a bake-off.
- **Data:** the same 815 claim probes (same newest-snapshot evidence rule), the same polarity flag cut, the same `main.jsonl` truth join used by every bench experiment.
- **Protocol:** each checkpoint answers the boolean "Did the attempted task succeed, based only on this snapshot?" per probe through its native serving path, with the snapshot passed as the state; per-model truncation by its own context ceiling (512 / 2048 tokens) is part of the test and is recorded, not avoided. det/FA at each model's native 0.5 point plus its probability dial; no per-model threshold tuning.
- **Acceptance, frozen:** the kill-test PASSES to the fine-tune stage iff ≥ 1 checkpoint reaches accuracy ≥ (majority share + 5 points) on the joined polar set; the majority share is computed and printed BEFORE any model output is seen. Otherwise: "no zero-shot signal" is recorded, and the fine-tune-on-our-probes path remains a separate future decision, NOT triggered by this test.
- **One pass; nothing deployed; the 12B and the standing 4B proxy are untouched** (sidecar inference runs locally on CPU).

### 2026-09-22 — the rubric read adjudicated: frozen bar says NOT PROMISING; the pre-registered descriptive cuts say the floor disease is nearly cured

Ran exactly as frozen: one pass over all 815 probes, same evidence rule and grammar-forced recipe as the baseline INDEPENDENT read, E = Σ i·p_i over (far from done / partly done / nearly done / fully done), flag grid T ∈ {1.0, 2.0}, adjudicated on the same 596-row join (473 polar: 175 wrong / 298 correct).

**Frozen primary verdict: NOT PROMISING.**
- rubric T=1.0: det 30% / FA 30%; T=2.0: det 56% / FA 20%.
- baseline binary at T=0.5: 67/18 (three-read min(g,d,c) context: 78/17).
The 4-level read does not produce a better general claim-checking dial. Nothing deploys; no prompt, option, or threshold iteration, per the no-iteration clause.

**Pre-registered descriptive cuts (recorded, not gating):**
1. **Floor disease nearly cured.** On CORRECT success claims, the binary read falls below 0.25 for 65% (35/54) — the sweep's t5/t1 disease — while the rubric reads E < 1.0 for only 9% (5/54). Histogram: 37 of 54 correct successes read 2–2.5 ("nearly done"), 2 read < 0.5. The ordinal scale is where the "confident no on a true report" failure goes to die.
2. **Direction split inverts the topline.** On success claims — what a stand-down review actually judges — the rubric at T=2.0 reads 62% det / 24% FA where the binary reads 79% det / 74% FA (the measured ~93% gate-challenge family); at T=1.0, 28/9. On failure claims the rubric detects 40% (T=1.0) vs the binary's 7%, at 35% FA. The ALL-polar topline that failed the bar is carried by the 244 correct-failure claims where a passive read scores well — the same direction-mixing artifact the voting analysis flagged in the gate's disfavor, now working against the rubric.
3. **The confidently-wrong tail persists.** 26 of 145 wrong success claims read E > 2.5 ("fully done"). A finer answer scale does not fix evidence latching; t8-style mistakes stay invisible to any read of the wrong row.

Posture: the rubric replaces nothing. The measured trade — far better separation on success claims and a live floor, worse aggregate dial — is dev-set evidence only. If the stand-down use case ever justifies a rubric-mode read in `review_frames_run`, the honest instrument is a fresh real-site sweep, not more bench passes.

### 2026-09-22 — where the review reads' ~2 s goes: prefill, and the fix is client-side, not a server flag

The sweep-era server log (773 timed requests in the sweep launch, 150 read-shaped) plus two 6-request micro-benches on the live 12B:

1. **Identical prompts, sequential: caching already works.** The build's slot selector is prefix-aware (`selected slot by LCP similarity`); reads 2–6 of the same prompt re-evaluated **1 token each (~160–220 ms vs 765 ms / 2,504 tokens for the first)** — ~4.5× cheaper, zero flags.
2. **Identical prompts, parallel (what forced_choice does at workers=2): defeated.** The pair splits across two slots and both prefill in full — the sweep's observed pattern (394-token candidate pairs on slots 2 and 3, each paying full prefill).
3. **Divergent prefixes (today's four review frames diverge at the first token): defeated.** LRU rotates across all four slots and every read pays a full ~2.5k-token prefill. `--cache-reuse 256` was tried and measured: **no improvement** (same rotation, same full prefills) — the flag does not reach this case.
4. **Actionable levers, in order, all client-side:** (a) dispatch the candidate reads sequentially so the second candidate hits the warm slot; (b) restructure the four frame prompts so the shared evidence leads and the per-frame question trails, giving the prefix-matching selector something to hold. Neither requires a server change; recorded as an infra lead, chain code unchanged.

Also recorded here: the temperature-fitting idea stays demoted (see the rubric pre-registration entry) — the gate consumes raw scores against thresholds, so a monotone rescale changes no operating point, and the observed errors are saturated ties in both directions.

### 2026-09-22 — the reflex-sidecar kill-test adjudicated: a formal PASS that the post-hoc audit kills

Ran exactly as frozen: both checkpoints via their native serving paths (agent-jev `AgentJevModel` full-state release → Qwen3-0.6B-Base backbone, load_state_dict missing=0 unexpected=0, shipped boolean temperature 1.0717 applied; Laya `agent.system_one` noul path, its own out-of-range temperature clamp warning recorded: choice 11+ ships 0.1006), all 473 polar probes, zero truncations (median evidence 567 chars — the native ceilings never bit), P(true)@0.5, majority printed before any model output (63.0% → bar 68.0%).

**Formal verdict by the letter of the frozen bar: PASS.**
- agentjev: accuracy 392/473 = **82.9%** (det 145/175 = 83%, FA 51/298 = 17%), bar 68.0%.
- laya: accuracy 320/473 = 67.7% (det 112/175 = 64%, FA 90/298 = 30%) — below bar.

**Post-hoc audit (pure arithmetic on the same frozen join — no model re-runs, no threshold changes): the pass is degenerate.** Join composition: 199 success claims (145 wrong / 54 right), 274 failure claims (30 wrong / 244 right); truth_ok=False for **389/473 = 82.2%**. A constant "the task failed" predictor — P(true)=0 for every probe — scores accuracy 82.2%, det 83% (145/175), FA 18% (54/298) on this identical join. AgentJev, which answered "failed" on 458/473 probes (histogram: 409 below 0.1), beats that constant by **exactly 3 probes out of 473, all in the FA column, zero additional detections** — its entire operating point is the degenerate corner. Laya spreads its probabilities but lands **14.6 points below** the constant. **There is no zero-shot signal here.**

**The flaw is mine and is recorded: the pre-registered bar referenced the wrong majority.** "Majority share" was computed on the claim-consistency classes (63% right) while `accuracy@0.5` is measured against the outcome classes (82.2% failed) — so the bar sat ~14 points below the degenerate predictor and a constant passes it. The kill-test design (off-the-shelf checkpoints, native paths, one pass) was sound; the acceptance arithmetic was not.

**Consequences, in order of record:**
- The kill-test did its job: it killed. The fine-tune stage does **not** earn a green light from this evidence — fine-tuning a checkpoint that is a constant on our probes sharpens nothing. Nothing deploys; no checkpoint is registered anywhere in the chain.
- No re-run with a "fixed" bar now — that would be iteration after results. If the sidecar door is ever pursued, the honest instrument is a **fresh pre-registration on a composition-balanced sweep** (the join's 82%-failed skew is itself a finding: this dev set cannot distinguish "reads success claims low" from "is a good detector" — the same skew that let a constant score 82%).
- What survives from agent-jev is the **recipe, not the checkpoint**: a 0.6B backbone + tiny set-encoder head, boolean/choice/score contracts, calibrated temperatures, prefix-shared candidate encoding — the right shape for a future sidecar fine-tuned on *our* recycled probes, tested on balanced fresh data. Laya's shipped-temperature pathology (0.1006, clamped) and below-constant accuracy close its door.

### 2026-09-22 — PRE-REGISTERED: the cache-lever change, a fresh balanced evaluation set, and the rubric's one fresh-set bar

Motivated by the kill-test audit (the 82%-failed join let a constant score 82%): the measuring stick is fixed first, the cache change is treated as infrastructure with exact-parity expectations, and the rubric gets exactly one fresh-set judgment. Everything below is frozen before any extraction or model read; one pass each; no prompt, threshold, or wording iteration after results.

**A. Sequential candidate dispatch (`eval/chain_http.py decide_http`).**
- The change: the per-candidate grammar-forced reads and the trailing premask read share one byte-identical prompt (the candidate is GBNF-forced, not embedded), so they are dispatched **sequentially** instead of via the 4-worker pool — the server's LCP slot cache then serves reads 2..N+1 from the warm slot. Transport only: no prompt, recipe, or d2 math changes; scores are expected identical.
- Acceptance (one pass, on 120 fixed decisions = the first 120 rows of the balanced fresh set in sorted order, done-question wording): old-parallel vs new-sequential must show the **same selected candidate on ≥ 99% of decisions and max |Δp_yes| ≤ 0.01** (llama.cpp batching may wobble the 4th decimal; a real flip is a bug), and **median decision wall time must not regress**. Deploy iff parity holds and timing does not regress; the measured improvement is reported against a ≥ 25% target but the target is not the gate.

**B. Evidence-first review frames — bench A/B first; port only if passed.**
- The change (frozen design, stated now): every review frame **leads with the shared snapshot evidence** and trails its frame-specific instruction and question; DELTA keeps BEFORE/AFTER as two labeled blocks after the shared AFTER snapshot; frames then run sequentially. Content identical, order differs.
- Acceptance (one pass, on a frozen 150-probe slice of the OLD probes in sorted order — deliberately NOT the fresh set, so the fresh set is never spent on infrastructure): per-frame yes/no agreement old-order vs evidence-first ≥ 97%, and median per-frame decision wall time (sequential) ≤ old parallel. PASS → port the reorder into the host's log-only `review_frames_run` (backup first); FAIL → do not port, report. The standing 4B proxy is not touched either way (in-process reads have no llama.cpp slot cache; the file exists only on the host and stays as-is).

**C. Fresh balanced evaluation set (the v2 sweep — never probed, never read, never analyzed at claim level).**
- Source: `main_runs_v2` — 48 chain-arm sessions with objective CDP labels in `main_v2.jsonl`. No claim-level design decision ever touched these transcripts. Page instances may repeat v1 task/seed cells (noted as a caveat); claims, trajectories, and snapshots are new. `cal_runs` excluded (calibration-contaminated); `sanity_runs_v3150` excluded (4 sessions, different purpose).
- Extraction: `ckpt_review.py` unchanged; claim polarity by the frozen vocabulary rule (model-independent); labels `raw > 0`; join by (task, seed, arm) against `main_v2.jsonl`.
- Balance rule (model-blind, frozen before any model read): polar cells only (said ∈ {success, failure}); quota Q = max(20, min over the four cells of cell size); keep the first Q rows per cell in (file, turn) sorted order. Composition reported before any read. If the wrong-success or right-success cell lands under 30, the rubric verdict below is **INDETERMINATE — cells too small**, and is reported that way, not salvaged.
- No new sessions this round: the v1 arms-runner did not survive as a script and reconstructing it is its own project; if the fresh cells come out too small, running more sessions is the documented next step, decided after this set is read.

**D. The rubric's one fresh-set judgment (frozen bar).**
- Reads: the binary done-question (`ckpt_score` recipe, unchanged) and the rubric read (`ckpt_rubric`, unchanged, flag grid T ∈ {1.0, 2.0}) — one pass each over the balanced fresh set, 12B, sequential transport.
- Cut: success-claim probes only (said = success by vocabulary). det = flagged rate on wrong-success claims; FA = flagged rate on right-success claims.
- Bar: the rubric earns the review-frame spot iff ∃ T ∈ {1.0, 2.0} with **det_rubric(T) ≥ det_binary − 5 points AND FA_rubric(T) ≤ FA_binary − 15 points** — the deployment-shaped trade (bounded detection give-up, clearly fewer false challenges). The floor-disease replication check is reported descriptively (E < 1.0 rate vs binary p < 0.25 rate on correct successes). Verdict recorded either way; a pass does not auto-deploy — the port is a separate documented change judged on this entry.

### 2026-09-22 — ENTRY A ADJUDICATED: FAIL — and the root cause is the server's prompt cache, not the dispatch change (amendment A2 pre-registered before any re-run)

**Entry A verdict, as frozen: FAIL.** The 120-decision parity run on the live server: 21/120 selected-candidate mismatches, max |Δp_yes| = 0.61, median wall 542 ms (sequential) vs 459 ms (parallel). Per the frozen rule — deploy iff parity holds — the sequential dispatch is **not** deployed; `eval/chain_http.py` reverts.

**Root cause found while investigating (evidence below): this llama.cpp build's default prompt/slot cache returns numerically wrong, time-varying read results.** Three probes (rows 3/56/94 of the 120-slice), same done-question, same recipe:
- Sequential single reads, repeated back-to-back: stable within a moment (0.4518/0.4518; 0.1911/0.1911; 0.6732/0.6732) — but the **same sequential read shifted between moments** after other traffic ran (probe 56: 0.1911 → 0.2730, stable both times; probe 94: 0.6732 → 0.8287). Cache contents poison later reads that partially match.
- Concurrent yes/no pairs (the deployed recipe, workers ≥ 2): jitter on every repeat (probe 3: 0.4323/0.4193/0.4472; probe 94: 0.8087/0.8310/0.7059) — the two same-prompt requests race on the slot cache.
- Parity-old (parallel) and yesterday's sweep values differ from today's clean-sequence values by up to 0.61 and 0.58 respectively — the same prompt, three "deterministic" answers, one per cache epoch.
- Control: sequential reads interleaved with three other prompts did **not** shift (0.4518 × 5) — small clean interleaves are safe; the corruption arrives with concurrent traffic and persists in the cache.

**Consequences, recorded before any re-run:**
1. **Historical score caveat (voids margins, not conclusions re-derived):** every sweep-era score file (scores_miniwob, scores_reframe, scores_rubric) was produced on this caching server with workers ≥ 2, so its p values carry nondeterminism of the observed magnitude (up to ~0.2 typical, 0.6 worst). Analyses whose margins exceed that noise stand as recorded; borderline margins are void. The fresh-set reads (entries C/D) — run clean — are the superseding instrument for the rubric judgment.
2. **No further read (entry B, C, D) may run on the caching server.** The server must first pass a determinism test with prompt caching disabled.
3. Entry A's deploy question is re-adjudicated **only** via the amendment below — same 120 decisions, same bars, on a server that passes integrity. (Disclosure: the 3-probe determinism peeks and the interleave control above happened before this registration and motivated it; they are recorded as evidence, not as thresholds.)

**A2 (frozen now): server-integrity gate + parity re-run.**
- Config: relaunch 12B with the same flags **plus `--no-cache-prompt`**. Flag-effect sanity: an identical repeated read must get no cache speedup (wall ~ first-read wall).
- Integrity bar (one pass): probes 3/56/94 of the 120-slice — sequential singles × 3, interleaved with other prompts and repeated after a concurrent burst, across two separated passes: all reads within 0.002 of each other; AND concurrent yes/no pairs × 3 must equal the sequential value within 0.002 (parallel is only rehabilitated if it too is exact here).
- Then the original entry A parity run (same 120 decisions, same wording, old-parallel vs new-sequential) with the original bars: same selected ≥ 99%, max |Δp_yes| ≤ 0.01, median decision wall not regress. Deploy sequential iff all hold; report the cache-off wall cost separately (correctness is the gate; the price is reported, not traded away).

### 2026-09-22 — A2 result: sequential reads are now perfectly deterministic; concurrent batching is a SECOND, independent corruption channel. A3 pre-registered before the -np 1 test.

**A2 integrity, as run:** `--no-cache-prompt` relaunch (PID 1252079, flags otherwise identical). Sequential singles on probes 3/56/94: **12/12 reads identical** (0.1420 / 0.9082 / 0.7828) across interleaves, after concurrent bursts, and across a 20 s separated second pass. Sequential determinism: **PASS** — the cache channel is closed. Flag-effect sanity: sequential values differ from the caching server's (probe 3: 0.4518 → 0.1420), confirming the config took effect (the 0.4518 "stable" values earlier today were themselves cache-tainted — stable within an epoch, wrong across epochs).

**Concurrent yes/no pairs: FAIL, and the failure is structural.** Values (0.6344 / 0.8031 / 0.4959 for probes 3/56/94) reproduce **exactly across the two server configs** (cache on: parity-old pass; cache off: today), with occasional composition-dependent flips (0.7913, 0.2645). Two same-prompt requests batched together on n_slots = 4 change the grammar-forced logprobs systematically — not floating-point wobble (Δp up to 0.77), not the cache. Parallel dispatch is **not rehabilitated**; the "old" arm of any parity run on this build is a corrupt reference.

**Why chain_http.py stays reverted:** entry A's frozen letter says deploy iff parity holds; parity cannot hold against a reference the same entry declares invalid, and the warm-cache latency prize that motivated the change is exactly what correctness now forbids. Revert stands; the real fix is server-side.

**A3 (frozen now): single-slot server config as the numerics-integrity fix.**
- Config: `-np 1` (one slot — concurrent requests serialize server-side; total KV tokens unchanged at 131072) **plus `--no-cache-prompt`**, all other flags identical.
- Integrity bar (one pass, same probes 3/56/94): sequential singles × 3 across burst/separation within 0.002 (expected, already shown on this cache setting), **and** concurrent yes/no pairs × 3 equal to the sequential value within 0.002 — equality here is the whole point: client-side workers stop mattering because the server serializes.
- On pass: the entry A parity re-run (same 120 decisions, old-parallel vs new-sequential client code) with the original bars — same selected ≥ 99 %, max |Δp_yes| ≤ 0.01, median wall not regress. On pass, deployment of the sequential edit follows the letter, with the recorded caveat that it is value-neutral (the win it sought is dead); the **production recommendation** from this whole thread is the server config: `-np 1 --no-cache-prompt` (or equivalent single-stream reads) for any llama.cpp server feeding forced-read decisions — including the standing 4B proxy's server, flagged as a recommendation only, not touched.
- Entries B, C, D run on this verified config only.

### 2026-09-22 — A3 result and ENTRY A FINAL ADJUDICATION: exact parity, timing regresses → sequential dispatch not deployed; the deployable fix is the server config

**A3 integrity: PASS.** Single-slot relaunch (`-np 1 --no-cache-prompt`, KV total unchanged at 131072): 36/36 integrity reads identical (0.1420 / 0.9082 / 0.7828), concurrent pairs now exactly equal sequential — client-side workers stop mattering because the server serializes. These values also match the `-np 4 --no-cache-prompt` sequential values bit-for-bit: two independent configs agree on ground truth.

**Entry A parity re-run on the verified server (same 120 decisions, original bars):**
- same selected: **120/120** (bar ≥ 99 %) — PASS
- max |Δp_yes|: **0.00e+00** on every decision (bar ≤ 0.01) — PASS, exact
- median decision wall: old-parallel **456 ms** vs new-sequential **519 ms** (+13.7 %) — **FAIL** ("must not regress")

**Final: entry A not deployed.** `eval/chain_http.py` reverted to the original parallel dispatch (md5 2b239439cc99f957cac0873a0bad9f01, byte-identical to pre-experiment). The sequential arm pays one network round-trip per read on the tunneled link (~60 ms/decision) that the parallel arm overlaps away; on a serialized server the numerics are identical, so the parallel code is kept.

**The deployable outcome of the whole cache thread is the server configuration:** any llama.cpp server feeding forced-read decisions on this build must run **`-np 1 --no-cache-prompt`** (or read strictly one request at a time with cache off). Evidence chain: multi-slot + prompt cache corrupts reads two independent ways (slot-cache partial-prefix reuse shifts sequential reads across cache epochs — measured 0.19 → 0.27 and 0.67 → 0.83 for identical prompts; concurrent same-prompt batching changes forced logprobs reproducibly across cache on/off — measured 0.63 vs 0.14 for the same prompt). The standing 4B proxy's server is flagged for the same recommendation — recommendation only, untouched.

**All sweep-era read-score files (scores_miniwob, scores_reframe, scores_rubric) carry the corruption noise** (produced on multi-slot + cache, workers ≥ 2). Margins larger than the observed noise stand descriptively; borderline margins are void. Entries B, C, D below run on the verified single-slot config, which supersedes for the rubric judgment.

### 2026-09-22 — ENTRY B ADJUDICATED: FAIL — evidence-first reorder changes outcomes; do not port (and one bonus integrity proof)

On the verified single-slot server, 150 frozen old-slice probes, three review frames re-ordered to lead with the shared snapshot (content identical, order only), old-order vs evidence-first:
- GOAL frame agreement 60.7 % (max |Δp| 0.77), DELTA 70.7 % (max 0.93), COND 86.7 % (max 0.80) — **frame agreement overall 72.7 % vs the 97 % bar: FAIL.**
- Median per-decision wall 416 ms (old, parallel) vs 422 ms (new, sequential) — no timing case either.
- Control read (byte-identical prompt in both arms): **150/150 agreement, max |Δp| exactly 0.0000** — on the verified server the pipeline reproduces bit-exactly, so the frame flips are the reorder itself, not noise.

Finding of record: **prompt order is not transport.** Moving the evidence ahead of the instruction changes what the model answers on a quarter to a third of frames. The evidence-first port is rejected on its frozen bar; the host's log-only `review_frames_run` stays as-is.

### 2026-09-22 — ENTRIES C+D ADJUDICATED on the verified server: rubric verdict INDETERMINATE (frozen rule), with descriptive numbers that redraw the picture

**C (set):** 69-probe balanced fresh set from `main_runs_v2` (cells wrong-success 20 / right-success 9 / wrong-failure 20 / right-failure 20; Q = 20). Both cells of the success-claim cut are < 30 → by the pre-registered rule the rubric verdict is **INDETERMINATE — cells too small**, reported descriptively, not salvaged. One harness note: the first read pass died per-row on a v1/v2 probe-schema mismatch (`vocab_hits`/`session` fields); re-run on a compat probes file that re-derives those two fields from already-frozen ones — no prompt, recipe, or threshold changed; the first pass wrote only error rows (nothing analytical was seen twice).

**D (reads, 12B, verified server, one pass):** success-claim cut (said = success by vocabulary):
| instrument | det (wrong-success, n=20) | FA (right-success, n=9) |
|---|---|---|
| binary done-read (brake rule) | 5/20 = 25 % | 8/9 = 89 % |
| rubric E < 1.0, T = 1.0 | 3/20 = 15 % | 0/9 = 0 % |
| rubric E < 1.0, T = 2.0 | 0/20 = 0 % | 0/9 = 0 % |
Frozen bar (∃ T: det ≥ 20 % AND FA ≤ 74 %): not met at either T — T=1.0 misses det by one claim (needs 4, has 3). Verdict stands INDETERMINATE by cell size; even taken at face value the bar is not passed.

**Descriptive — and the reason this set matters:**
- **Floor-disease replication, inverted emphasis:** on correct successes the sweep-era pattern replicates in direction and stronger — rubric E < 1.0: **0/9**; binary p < 0.25: **8/9**. The rubric never low-balls a genuinely passed page; the binary read low-balls nearly all of them.
- **The binary done-read is outcome-inverted on success claims in fresh data:** wrong-success p_done median **0.58** (reads failed pages as succeeded) vs right-success p_done median **0.06** (reads passed pages as not succeeded). As an outcome classifier on the balanced set, p_done ≥ 0.5 scores **49 %** against a 58 % failed-share constant — below the degenerate predictor, on a set where that comparison is now fair (balanced cells).
- Both instruments detect little (det 25 % / 15 % / 0 %): the 12B's done-read, read clean for the first time, is not a usable success verifier on this domain as-is.

**Standing caveat applied:** all sweep-era read scores were produced on the corrupting server config and are superseded by this pass for any borderline judgment. **What this buys:** the deferred decision (reconstruct the arms runner and run ~3× more passed sessions to size the right-success cell at ≥ 30) now has a clean instrument waiting for it; the rubric's 0-FA property is the interesting signal, and it is the one thing the small cell cannot yet license.

### 2026-09-22 — PRE-REGISTERED (entry E): the v3 sweep — runner rebuilt, 168 fresh chain sessions, the rubric's properly-sized judgment

The deferred decision from entries C/D is now executed: size the right-success cell at ≥ 30 by running more sessions, then give the frozen rubric bar its real judgment. Frozen before any v3 session runs.

**Runner reconstruction (infra, recovered from artifacts):** local MiniWoB page server on 127.0.0.1:8077 (`/tmp/miniwob-html/miniwob/html`, already running); omp 18.1.17 non-interactive (`omp -p <prompt> --model poc-proxy/gemma-12b-decide --session-dir <dir>`, cwd = the workspace) against the standing host proxy (8999, `poc-proxy` provider in `~/.omp/agent/models.yml`, auth none); the agent drives the page via omp's browser primitives inside its eval tool (runner prompt template reused **verbatim** from the v2 transcripts, including the `w<seed>c0` tab name and the 120-second episode line); rewards collected per session by attaching CDP to the agent's Chromium (launched with `--remote-debugging-port=0`; resolved port read from the browser's `DevToolsActivePort` file), locating the tab by `wseed`, and evaluating the page globals (`WOB_DONE_GLOBAL`, `WOB_RAW_REWARD_GLOBAL`, `WOB_REWARD_GLOBAL`, `WOB_REWARD_REASON`, `WOB_EPISODE_ID`, `WOB_TASK_READY`, query text from `#query`) until the episode ends, plus a settle wait after omp exits — matching the v2 `.cdp` row schema. Sessions run **serially** (as v2 did; concurrency would stretch turns against the 120 s page cap); omp hard-capped at 480 s (v1's observed cap); labels `raw > 0` from the page's own reward, never exit codes.

**Declared differences from v2 (stated, not thresholded):** the 12B server now runs the **verified numerics config** (`-np 1 --no-cache-prompt`) instead of v2's corrupting one — v3 is the first sweep whose agent turns and chain reads are numerically sound; the proxy is now v3.16.2 (review-frames log-only) vs v3.14.3 at v2 time. The judgment below does not depend on v2 pass rates; pass-rate comparability is descriptive only.

**Sweep spec (frozen):** the 24 tasks of the frozen GOALS list × **7 fresh seeds** = **168 chain-arm sessions**; seeds drawn once from `random.Random(20260922)` as `randint(10**9, 10**10 − 1)`, regenerated on any collision with each other or with any v1/v2 seed; task-major order, seed-inner; session dirs `/tmp/mw_bench/main_runs_v3/<task>_s<seed>_chain/`; labels appended to `main_v3.jsonl` as sessions complete. **The sweep runs to 168 regardless of interim results** — no stopping rule, no mid-sweep peeks at claim content (outcome labels may be counted; transcripts are not read or analyzed until the sweep completes).

**Judgment (frozen — identical bar to entry D, on v3 only):** extract probes with `ckpt_review.py` unchanged; polarity by the frozen vocabulary rule; balance model-blind (polar cells only, Q = max(20, min cell size), first Q rows per cell in (file, turn) order); composition reported before any read; reads = binary done-question + rubric 4-level, T ∈ {1.0, 2.0}, one pass each, on the verified single-slot server; cut = success-claim probes; **bar: ∃ T with det_rubric(T) ≥ det_binary − 5 points AND FA_rubric(T) ≤ FA_binary − 15 points**; any cell < 30 → INDETERMINATE again, reported not salvaged. A pass does not auto-deploy — porting remains a separate documented change. v2's read set is not pooled with v3 for the bar (v2 was already read); pooled numbers may be reported descriptively alongside.

**Launch-time infra note (disclosed deviation from the runner text above, before session 1 of the sweep):** the pre-registered runner text says the agent talks to "the standing host proxy (8999)". At launch that port is owned by the standing **4B** proxy (its own log shows in-process 4B generation, ~7 s/token — and a probe request for `gemma-12b-decide` was answered by that local model without ever reaching the 12B server, whose log stayed silent). The 12B chain proxy that served v2 on 8999 no longer exists. Repair, no recipe change: a second `omp_proxy.py` instance was launched on **8990** with the same frozen chain config (`eval/proxy_chain_remote_review.json` — backend `http://127.0.0.1:8998/v1`, gates `configs/webgate_remote.json`, `--no-local-model` so it carries no weights and cannot answer locally); `~/.omp/agent/models.yml` gained a dedicated provider `poc-proxy-12b` → `http://<gpu-host>:8990/v1`, and the sweep's model string is `poc-proxy-12b/gemma-12b-decide` (the shared `poc-proxy` provider is untouched). The standing 4B proxy on 8999 was not touched. Also, for robustness on this omp version the CDP collector discovers the browser **tab-first** (scans every candidate DevTools port for the session's unique `wseed` URL — the daemon browser is standing across sessions, so pid-delta discovery is unreliable), attaches with `suppress_origin` (Chrome 403s origin-bearing CDP handshakes), lets the reward settle 6 s after `done`, and after attach keeps polling until the page's own 120 s cap can expire even if omp exits earlier. One sanity session (throwaway seed 9999817251, outside the frozen plan) verified the full path: chain-proxy → verified-config 12B, page-side automation, CDP reward `raw=1`, transcript captured. Sweep session 1 (click-button/s2551038599) then succeeded (`raw=1`, 84 s, clean omp exit).

### 2026-09-22 — ENTRY E ADJUDICATED: 168/168 fresh sessions, both cells properly sized by design — and the verdict is INDETERMINATE (cell rule) with the bar failing detection at both thresholds on point estimates

**Sweep completion:** 168/168 sessions in `main_v3.jsonl` — **21 passed (12.5 %)**: click-button 5/7, click-button-sequence 7/7, click-checkboxes 2/7, click-test 7/7; every pass-capable task is click-family and the other 20 tasks produced zero passes. 135 sessions failed with the page's own reward at 0 (52 of them hit the 480 s omp cap, exit 124, per the runner log); 12 ended `cdp_err` (reward collector found no page state; frozen rule: failures). Truth = `raw > 0` from the page, never exit codes.

**Pipeline (one pass per stage):** extract over all 169 transcript files → 994 probe rows; the extra file was the aborted pre-resume attempt (disclosure 1) → removed by timestamp filter → **988 rows over 168 sessions**. Frozen balance: polar success 207 / failure 158 → Q = 158 → **316 balanced probes across 22 tasks**, composition printed before any read. Reads: binary done-question + 4-level rubric, one pass each, 316 + 316 rows, **0 errors**, on the verified single-slot server (`-np 1 --no-cache-prompt`) through the read tunnel.

**Judgment (frozen bar, success-claim cut):**

| instrument | det (wrong-success, n=134) | FA (right-success, n=24) |
|---|---|---|
| binary done-read, T=0.5 | 119/134 = 88.8 % | 20/24 = 83.3 % |
| rubric E < T, T=1.0 | 16/134 = 11.9 % | 0/24 = 0.0 % |
| rubric E < T, T=2.0 | 72/134 = 53.7 % | 20/24 = 83.3 % |

Right-success 24 < 30 → **INDETERMINATE** by the pre-registered cell rule. And the bar (∃ T: det ≥ det_binary − 5 = 83.8 %, FA ≤ FA_binary − 15 = 68.3 %) fails on the point estimates at both T anyway: T=1.0 misses detection by ~72 points (though its FA is 68 points better than needed); T=2.0 misses both conditions. **The review-frame spot is not earned; `review_frames_run` stays log-only; nothing ports.**

**Descriptive (reported, not salvaged into a pass):**
- **The rubric's 0-FA property replicates at scale** (0/24 at T=1.0): the graded read never sinks a genuinely passed page below "partly done". The binary read low-balls 20/24 of them (p_done < 0.25 on 83 %) — floor disease replicated at 2.7× entry D's cell size.
- **Detection is the rubric's failure mode:** wrong-success E median 1.87 — a hallucinated-success page usually LOOKS partly-to-mostly done in its final snapshot, and the graded read cannot tell "page looks advanced" from "the task's specific goal was met" on this domain. The bar asked the rubric to match a detector's detection at a confirmer's false-alarm rate; the 12B does neither pair.
- Failure claims are near-perfect truth-trackers: 154/158 sit on genuinely failed sessions (4 on passes — pessimistic claims after a pass, the mirror image of the floor disease).
- Binary det jumped from entry D's 25 % to 88.8 %. Composition explains it: 121/134 wrong-success claims sit on plain failed sessions (incl. heavy cap-outs) and 13 on no-page-state cdp_err sessions, where any honest read says "not done"; **0 sit on the two shake tasks** (shake sessions emit no claims at all — see disclosure 2). Sensitivity: excluding the 13 cdp_err rows moves every number ≤ 4 points (bin 89.3/83.3, rub T=1.0 12.4/0.0, rub T=2.0 49.6/83.3) — nothing hinges on the cdp_err inclusion.

**Why the cell still missed 30:** the balance Q capped at the failure cell (158), and success claims on passed sessions are structurally scarce — the 21 passes are short click-family episodes yielding ~1 polar success claim each (24 total). More sessions of this task mix cannot reach 30; only a higher pass rate or richer pass-session prose can.

**Disclosures (all deviations, in order):**
1. **False-outage episode mid-sweep (operator error, no data impact):** during a cap-out-heavy stretch the 12B server log went silent ~56 min; I suspected a wedge and killed the sweep runner and one in-flight omp before doing the forensics. The signature — ~33 turns/s, 15k-line transcripts, zero model calls, every tool result shaken to a `[shaken ~122 tokens — recover: artifact://…]` stub — is **v2-identical** on click-link/navigate-tree (omp context-shaking starves the chain of candidates → no-candidate fast path spins without model calls; omp 18.1.17 unchanged since v2). No outage existed. The one aborted session (read-table s6374130157, transcript T00-41-25) re-ran to completion on resume (T01-04) and only the completed run's rows were used. Nothing voided; no frozen instrument, model config, or proxy was touched mid-experiment.
2. **Shake-task sessions are valid failures and stay in the tally:** v2 transcripts of the same tasks show the same stub signature (~4,500 stubs/task), so this is the known degenerate loop, not an outage; those sessions fail because the agent never receives page content. They contribute 0 probes.
3. **cdp_err handling:** the 12 cdp_err sessions are stored `raw=null` and count as failures per the frozen rule; the judge implements exactly that (`raw=null → truth_ok=False`, 18 probes joined) instead of skipping them. A descriptive sensitivity check (above) shows the verdict is identical without them.
4. **Plumbing repairs during adjudication, frozen semantics untouched:** (a) the aborted transcript's rows were removed pre-balance by timestamp filter (the file list, not the data, was wrong); (b) v3 `ckpt_review` output lacks v2's `said` field — the compat layer re-derives it from the claim text with the frozen vocabulary (case-insensitive substring, the extractor's own matching semantics) and substitutes frozen-vocab `vocab_hits`, exactly as the v2 compat did; (c) the judge's key regex required a leading `/` while v3 probe paths are relative — the fixed pattern is a strict superset with identical capture groups (absolute-path behavior provably unchanged); (d) the judge's skip-on-null-raw was replaced by the frozen cdp_err rule (disclosure 3). No threshold, cut, vocabulary, or read was changed.
5. **Reads invocation error (first read pass):** `--base-url` was passed WITH a trailing `/v1`, but the frozen scorers append `/v1/chat/completions` themselves → HTTP 404 on every read. The real error was masked by a **pre-existing `ckpt_score` error-path bug** (failed candidates store `("ERR", str)` and the normalization step then compares str > int → TypeError). Both reads were re-run with the bare host URL and produced the clean results above; the errored pass wrote only error rows, so nothing analytical was ever read twice. The instrument bug is noted, not fixed (the instrument stayed frozen).
6. The rubric scorer's `--skip` resume argument went unused — no read was interrupted.

**What this buys:** entry D's picture survives at 2.7× the cell size with fresh, numerically-clean data — and it redirects the next decision. The 0 %-FA-at-T=1.0 property is real and repeatable, but the graded read is blind as a detector; its honest role on this domain is a **low-false-alarm confirmer** (e.g., gating an auto-retry or auto-approve path), not the review-frame flag. Porting nothing. Any future attempt needs a new pre-registration — either a different target role for the rubric, or a pass-rate intervention first (the 30-cell is unreachable at a 12.5 % pass rate dominated by four click tasks).

**Teardown (post-adjudication, entry E infra returned to standing state):** stopped by ps-verified PID, each in its own ssh call — the 12B llama-server (PID 1258921, `127.0.0.1:8998`, up 13 h) and the 8990 chain proxy (PID 1265742, up 12 h); the 18443 read tunnel closed locally. Verified untouched after teardown: the standing 4B proxy (PID 1234409, port 8999) and the film-worker (PID 577653, owner of the `<gpu-host>:8998` listener — the 12B server shared the port only on loopback). The local MiniWoB page server (127.0.0.1:8077) stays up. `models.yml` keeps the `poc-proxy-12b` provider entry, now dormant (its 8990 endpoint is down until relaunched: `eval/omp_proxy.py --port 8990 --backend-config eval/proxy_chain_remote_review.json --no-local-model` on the host, plus a verified-config llama-server on host port 8998). Nothing else on the host was changed.

### 2026-09-23 — PRE-REGISTERED (entry F): option-order permutation control on the entry-E reads — is the binary read reading evidence, or reading position?

**Motivation.** JevBench's protocol report (21 Sep 2026) documents a small decision model dropping 72 % → 21 % when the yes/no answer options were reversed. Entry D/E found our binary done-read outcome-inverted on success claims (wrong-success p_done median 0.58, right-success 0.06) with det 88.8 % / FA 83.3 % — the signature of position latching at least as much as evidence reading. This entry separates the two. It is an **instrument diagnostic**: no frozen bar is judged, nothing deploys, no pooling with entry E for any verdict.

**Design (frozen before any permuted read).** The entry-E balanced probe set is reused byte-identical (`probes_v3_balanced.jsonl`, 316 rows, sha256 `e1c33b6e2da5d2737e8262c0e023301004baa51eb56cc24a06c21b875199fdfe`). Same evidence reconstruction, same question texts, same grammar-forced recipe, same verified server config (`-np 1 --no-cache-prompt`), same one-pass discipline. **The only change is the order candidates are presented in the Options block:** binary `["yes","no"] → ["no","yes"]`; rubric `[far, partly, nearly, fully] → [fully, nearly, partly, far]`. Rubric E is recomputed with the canonical index mapping (far=0 … fully=3) so semantics are unchanged — presentation permutes, meaning does not. New scorer copies (`ckpt_score_ord.py`, `ckpt_rubric_ord.py`) differ from the frozen instruments in the candidate list only; the frozen instruments themselves are not edited. Analysis joins orders by probe row order (the entry-E join discipline).

**Numerics gate (frozen, before the permuted pass).** Because the permuted comparison is only meaningful against a server in entry-E state, the ORIGINAL scorers are first re-run on a fixed 20-probe validation slice (first 10 + last 10 rows in file order) and must reproduce entry E's stored scores at the printed 4 decimals — 20 p_done_yes values and 20 E values, all 40 must match exactly. Any mismatch → stop, investigate, do not compare.

**Metrics (frozen).**
- Primary, binary done-read: per-probe decision flip rate `1[p_yes^A ≥ .5] ≠ 1[p_yes^B ≥ .5]` (A = entry-E canonical order, B = permuted), overall and within the wrong-success (n=134), right-success (n=24), and failure-claim (n=158) cuts; mean |p_yes^A − p_yes^B|; Pearson r; primacy index `mean(p_yes^B) − mean(p_yes^A)` (positive = "yes" gains mass when presented first). Also the full entry-E table recomputed from B alone (det/FA at T=0.5): does reversal alone fix, worsen, or reproduce the inversion?
- Secondary, binary claim-read: flip rate and mean |Δp| (diagnostic only, not gated).
- Tertiary, rubric: decision flips at T=1.0 and T=2.0 under the reversed scale; the entry-E rubric table recomputed from B alone; and the 0-FA robustness check at T=1.0 on right-success (n=24): FA_B ≤ 1/24 → the rubric's no-false-challenge property is order-robust; ≥ 6/24 → order-fragile; between → partial.

**Interpretation thresholds (frozen, primary flip rate = done-read, overall).** ≤ 10 %: order-robust — the entry-E table stands as evidence-driven. 10–25 %: moderately order-sensitive — order-averaged reads (mean both orders) become the recommended instrument fix, in the spirit of djev's multi-draw reads. > 25 %: order-dominated — the entry-E binary det/FA numbers are substantially position artifact; no future use of the binary read without order-averaging.

**Disclosed setup facts.** The 12B llama-server (stopped at entry-E teardown) is relaunched for this pass at `~/llama.cpp/build/bin/llama-server`, host `127.0.0.1:8998`, verified flags (`--jinja -c 65536 -ctk q8_0 -ctv q8_0 -ngl 99 --flash-attn on -b 1024 -ub 512 -np 1 --no-cache-prompt`), read again through the 18443 tunnel; it is torn down again afterwards. The standing 4B proxy (8999) and film-worker are not touched. Server start/stop PIDs and all score-file sha256s are recorded in the results entry.

### 2026-09-23 — ENTRY F ADJUDICATED: both reads are order-robust at their operating points — the entry-E table stands as evidence-driven, not position artifact

**Procedure as frozen:** server relaunched (PID 1470099, `n_slots = 1`, determinism smoke bit-exact twice-over); numerics gate — original scorers on the 20-probe validation slice — **PASS, 40/40 values bit-exact** against entry E (see disclosure 1 for the gate's false alarm); permuted passes then ran clean in one pass each: 316 + 316 rows, **0 errors**, ~10 minutes wall. Score files: `scores_v3_bin_ord.jsonl` (sha256 `0a6a5184b5274472276d27a66306295e977173a0697e91ea5ed13415e35cee8b`), `scores_v3_rub_ord.jsonl` (sha256 `f91870d15557f7cf6ca46af8169304219154b13fcb7fdc7c0193e9b832cf63a9`).

**Primary — binary done-read (the instrument of record):**
- **Decision flip rate 20/316 = 6.3 % → ORDER-ROBUST band (≤ 10 %).** By cut: wrong-success 14/134 (10.4 %), right-success 1/24 (4.2 %), failure-claims 5/158 (3.2 %). Mean |Δp| 0.091; Pearson r 0.769; primacy index −0.068 (presenting "yes" first raises yes-mass slightly — a mild option-1 preference, far from catastrophic).
- The entry-E table recomputed from the reversed order alone: **det 88.8 % / FA 87.5 %** (A was 88.8/83.3) — reversal neither fixes nor meaningfully worsens the inversion; the 4.2-point FA tick is exactly the single right-success flip.
- Direction check: on right-success, p_yes rises under reversal for only 4/24 (17 %) — **the floor disease is evidential, not positional**: the read low-balls genuinely passed pages in either order.

**Secondary — claim-read:** 29/316 = 9.2 % flips, mean |Δp| 0.085 — under the band edge but close to it; noted for any future use of the claim read.

**Tertiary — rubric:**
- T=1.0: 9.5 % success-claim flips; from-B table det 11.2 % / **FA 0.0 %** — **the rubric's no-false-challenge property replicates under the reversed scale (FA_B 0/24 → ORDER-ROBUST band)**.
- T=2.0: 19.0 % flips (moderate order wobble at the loose threshold, as expected for a sum over presentation-ordered probabilities); from-B table det 48.5 % / FA 62.5 %. Mean |ΔE| 0.320, r 0.819.

**Interpretation (nothing deploys; instrument diagnostic).** The feared failure mode — position latching à la JevBench's 72 %→21 % collapse — is **ruled out** for both instruments at their operating points. Two consequences. (1) Entry E's numbers are what the model actually thinks about the evidence, in either presentation order: the binary read's inversion is a genuine evidence-reading failure (it truly "believes" passed pages look undone), and the rubric's 0-FA is a real, order-stable property — not order luck. (2) No order-averaging fix is *required*; djev-style multi-draw averaging remains optional hardening rather than a necessity. The entry-E conclusion sharpens accordingly: the 12B genuinely cannot distinguish "page looks advanced" from "the task's goal was met" on this domain — an evidence limitation, not a prompting artifact. Standing protocol adopted for all future reads: an option-order control pass is cheap (~10 min) and now precedes any instrument judged load-bearing.

**Disclosures:**
1. **The numerics gate first reported FAIL (19/20 binary exact)** — traced to MY gate's join, not the server: the gate keyed scores by `(session, claim)`, and the balanced set contains **6 duplicate-claim keys** (same sentence repeated at different turns of one session; one claim appears 3× with different per-turn evidence and therefore different — each perfectly reproducible — scores: two twins at p_done 0.0057, one at 0.0114). Under the pre-registered join-by-index (the entry-E discipline): **20/20 binary + 20/20 rubric bit-exact → PASS**. No server drift; no data impact. Note for future joins: binary score rows carry no turn field, so any key-based join must include the probe index.
2. **Post-analysis reporting bug (metrics unaffected):** the primacy direction label was coded inverted; the frozen metric `mean(p_B) − mean(p_A)` and every number were correct — only the human-readable label was fixed and the script re-run on unchanged data.
3. The frozen instruments (`ckpt_score.py`, `ckpt_rubric.py`) were not edited; the permuted copies differ by exactly the candidate-order substitution (diffs verified before any read).

### 2026-09-23 — PRE-REGISTERED (entry G): the confirmer gate — can a multi-draw rubric read safely say "yes"? (the djev-style error-bar experiment)

**Motivation.** Entry E ruled the rubric out as a detector (det 11.9 % at T=1.0) and identified its viable role as a **low-false-alarm confirmer** (FA 0/24 at T=1.0); entry F proved that property order-robust. The open role question: can a rubric read, sampled multiple times (the autoregressive analogue of djev's noise-draw reads), serve as a **claim-blind auto-approve gate** — confirming enough genuinely passed sessions to be useful while falsely approving almost none? This entry measures exactly that, on the entry-E probe set, with nothing else changed. Instrument diagnostic again: nothing deploys from this entry regardless of outcome.

**Design (frozen before any read).** Same 316-probe balanced set, byte-identical (sha256 `e1c33b6e2da5d2737e8262c0e023301004baa51eb56cc24a06c21b875199fdfe`). **Rubric-only** — the binary read is deliberately excluded (its 83 % floor disease on passed pages makes it a non-candidate confirmer; entry E data settles that arm without new reads). New script `ckpt_rubric_mc.py` (frozen instruments untouched): for each probe, the **identical rubric question** (canonical option presentation, per entry F's robustness result) is asked **K = 9 independent times** with the answer grammar-locked to the 4-level set `far from done | partly done | nearly done | fully done` at **temperature 0.7** — the autoregressive analogue of djev's noise draws; each draw returns one level sampled from the model's temperature-shaped distribution. Server: the verified 12B config, default seeding (unset), one pass, draws as they come — a single stochastic measurement, never re-rolled. Claim-blind: the gate sees only evidence, never `said`.

**The gate (frozen):** a probe is **CONFIRMED iff all 9 draws land in {nearly done, fully done}** ("unanimous-high" — count-based, no estimator tuning). Descriptive curve alongside: thresholds "≥ j of 9 draws high" for j ∈ {6, 7, 8}.

**Pools and bar (frozen — claim-blind, so the cells are session-truth cells, not claim cuts):** confirmation pool = probes on passed sessions, **n = 28** (24 success-claims + 4 failure-claims that sat on passes); false-approval pool = probes on failed sessions, **n = 288** (134 wrong-success + 154 truly-failed failure-claims). **The confirmer spot is provisionally earned iff the gate confirms ≥ 23/28 passed-session probes (≥ 80 %) AND ≤ 14/288 failed-session probes (≤ 5 %).** Otherwise: not earned, curves reported for diagnosis. "Provisionally" because probe-level counts overstate independence (multiple probes per session); a deployment decision would require session-level validation and a separate documented change — which this entry does not make.

**Comparison points (frozen, computed from the existing entry-E single temp-0 rubric file — no new reads):** the same pools under three single-read gates: E ≥ 2.5, E ≥ 2.0, p_fully ≥ 0.5. Reported as a table next to the multi-draw gate so the comparison is point-vs-curve, not curve-vs-cherry-pick.

**Interpretation notes (pre-committed):** (a) if unanimity confirms far fewer than 80 % of passed sessions, the interesting question becomes where the draws disagree — the per-cell draw-count distributions are reported to distinguish "the model is genuinely unsure on passed pages" from "one stray low draw per page"; (b) sampling at temperature spreads mass by design, so a *worse* raw FA at loose thresholds (j ≤ 7) is expected and uninformative — only the frozen unanimity gate carries the verdict; (c) the entry-F primacy finding (~0.07 option-1 preference) applies identically to all draws and cancels in the comparison.

**Infra (as entry F):** 12B llama-server relaunched with the verified flags on `127.0.0.1:8998` (PID recorded in results), read through the 18443 tunnel, torn down after; standing 4B proxy and film-worker untouched. Sanity before the pass: temp-0 bit-exactness smoke, and a temp-0.7 variation check (identical requests must produce *different* draws). Score-file sha256 recorded in the results entry.

### 2026-09-23 — ENTRY G, pre-registered sanity gate FAILED: the multi-draw design is dead on this server. Amended (before any probe was read) to paraphrase draws at temperature 0

**What the sanity clause was for, and what it caught.** Smoke 1 (temp-0 bit-exactness) passed: two identical grammar reads returned byte-identical output. Smoke 2 (variation) failed: on a deliberately ambiguous page, all 9 temperature-0.7 grammar draws returned `fully done`. The diagnostic pinned it: the same prompt, grammar-locked, with `logprobs` on, returns **byte-identical first-token logprobs at temperature 0.0, 0.7, and 1.5** (`fully −0.037, nearly −3.729, partly −4.676, far −6.069`), and 27/27 draws across T=0.7 and T=1.5 are identical. Finding of record: **this llama-server build applies no temperature on the grammar-constrained path**, and even the free-sampling first-token distribution is degenerately peaked (p(fully) ≈ 0.976). There is no sampling noise to average over: K = 9 unanimity would have been a deterministic re-read of the modal answer, an error bar with zero width by construction. Per the pre-registration's own clause, **the MC pass did not run** — zero probes were read under the dead design, and `ckpt_rubric_mc.py` / `analyze_confirm.py` are retired unused.

**Amended design (frozen before any read): replace sampling noise with prompt noise.** Same 316-probe set (sha256 re-verified unchanged), same recipe as the frozen rubric read (temperature 0, GBNF per candidate, summed raw logprobs, α=1 norm) — but each probe is read under **K = 5 pre-written paraphrases of the framing text** instead of one. The evidence block, the 4-level option list, the canonical order, and the instruction line are byte-identical in all five; only the two framing sentences vary. The five framings, frozen verbatim (P1 is the original canonical text from `ckpt_rubric.py`, so its read should bit-match the entry-E file — a free replication check):

- **P1:** `The following is a web page's accessibility snapshot after an attempt to complete a task on the page.\n\nEvidence (page snapshot):\n{ev}`
- **P2:** `Below is the current state of a web page, captured while an agent tried to carry out a task on it.\n\nPage state:\n{ev}`
- **P3:** `You are looking at a snapshot of a web page, taken after someone attempted a task on it.\n\nSnapshot:\n{ev}`
- **P4:** `Here is the accessibility tree of a web page at the end of a task attempt.\n\nEvidence:\n{ev}`
- **P5:** `This is how the web page looked when the task attempt finished.\n\nPage snapshot:\n{ev}`

New script `ckpt_rubric_par.py` (frozen instruments untouched), one pass, 316 × 5 = 1,580 requests.

**The gate (frozen):** per probe and paraphrase j, E_j as in the frozen rubric read; per-probe mean Ē and min. A probe is **CONFIRMED iff min_j E_j ≥ 2.0 AND Ē ≥ 2.5** — no paraphrase reads the page low, and the average is high. Descriptive curve alongside: min_j ≥ 2.0 alone; Ē ≥ 2.5 alone; all-5 modal level ∈ {nearly, fully}.

**Pools and bar: unchanged** — confirmation pool n = 28 (passed sessions), false-approval pool n = 288 (failed sessions), **earned iff ≥ 23/28 confirmed AND ≤ 14/288 false-approved**, with the same provisionality caveat (probe-level counts overstate independence; nothing deploys from this entry).

**Comparison points (frozen):** unchanged — the three single-read gates (E ≥ 2.5, E ≥ 2.0, p_fully ≥ 0.5) from the existing entry-E file, the no-error-bar baseline. Additionally, computed from the same existing file with no new reads: an **analytic spread** per probe, SD(E) = √(Σᵢ pᵢ(i − E)²), reported per cell — the teacher-forced distribution's own width, as a descriptive answer to "how concentrated was the read, really", and a replacement for the error-bar information the draw design was meant to carry.

**Pre-committed interpretation notes:** (a) if P1's reads do not bit-match the entry-E rubric file, that is itself a reported instability finding (same prompt, same server, two passes) and the replication check fails loudly; (b) if the five paraphrase reads barely disagree (min ≈ mean on most probes), the honest conclusion is "the read is robust to phrasing and the single-read gates are already the whole story" — the multi-read gate then degenerates to the E ≥ 2.5/2.0 comparison and is judged as such, not sold as an improvement; (c) paraphrase spread, where it exists, measures prompt sensitivity, not model uncertainty — labeled as such everywhere.

**Deviations from the original registration (all disclosed here, none retroactive):** the noise source changed (sampling → phrasing) because the sanity gate killed the original; K changed 9 → 5; the gate changed (unanimity-of-draws → min/mean of E); the frozen question text is now one of five frozen texts rather than the only one. Everything else — pools, bar, comparison points, provisionality, claim-blindness, one-pass discipline — is unchanged.

### 2026-09-23 — OMP CONTEXT-SHAKING ROOT-CAUSED (the promised knob hunt): the read-only base config's `compaction.thresholdPercent: 30` fires maintenance at 39,321 tokens; a dead-end rescue then stubs every tool result; a launch-time `--config` overlay disables it without touching config.yml

**Mechanism (omp 18.1.17 source).** The context-maintenance class has a `shake` ("elide") operation that rewrites matched tool-result regions of the session history into `[shaken ~N tokens — recover: artifact://ID (region K)]` stubs, saving the originals as `NNN.shake.log` artifact files inside the session directory. Three frozen configurations ship with it: AGGRESSIVE (`protectTokens 4000, minSavings 0, fenceMinTokens 400`), DEFAULT (`protectTokens 16000, minSavings 4000` — the auto-compaction action), and RESCUE (`protectTokens 0` on top of AGGRESSIVE — the "dead-end shake rescue" that fires when a request path dead-ends; minSavings 0 is why observed stubs include 167-token regions nothing else would touch).

**Root cause (transcripts + decoded trigger).** In the storming sessions the first compaction entry lands at `tokensBefore = 39,392`, method `handoff`. The trigger decoded from the bundle is `floor(window × thresholdPercent)` when `thresholdPercent > 0`, and the read-only `~/.omp/agent/config.yml` sets **`compaction.thresholdPercent: 30`** → floor(131072 × 0.30) = **39,321** — the observed 39,392 crosses it, and the storm plateau (39.3–39.5k before/after every subsequent compaction) sits exactly on this line. Storm anatomy: handoff frees ~40 tokens per cycle (the summary plus the live turn refill the context), 295 handoffs per session; between them the rescue path shakes every eval tool result including the aria snapshots (295 shake artifacts, 5,272 stubs in the worst transcript) → no page state → the no-candidate loop accelerates. **Task-specificity:** 14/14 `click-link` and `navigate-tree` sessions storm (all failures), and 0 of the other 154 sessions contain a single stub — the underlying loop is the model's own failure mode on those two tasks; the 30 % compaction threshold is the amplifier that also destroyed the evidence. The 14 sessions remain valid failures (page-reward labels, no probes taken), consistent with the entry-E adjudication.

**The knob (config.yml untouched).** omp 18.1.17 accepts repeatable launch-time config overlays — `--config <file>` on the command line or `PI_CONFIG_FILES` in the environment — parsed with the same YAML schema as the base config, deep-merged over it, and flagged loudly when invalid (not silently ignored). For any future sweep run: write `/tmp/sweep-overlay.yml` containing `compaction:\n  enabled: false` and add `--config /tmp/sweep-overlay.yml` to the omp launch (alternatives: `strategy: "off"`, or the conservative `thresholdPercent: 90`). A 120-second episode gains nothing from compaction; v2-era sessions ran with none firing. Verification belongs in the next sweep's throwaway smoke session: assert zero `shaken ~` and zero `"type":"compaction"` entries in the transcript.

### 2026-09-23 — ENTRY G ADJUDICATED: confirmer gate NOT EARNED — and the paraphrase experiment closes the rubric's confirmer role entirely

**Run facts.** One pass, 316 probes × 5 framings (1,580 rubric reads, 6,320 candidate scorings), temperature 0, frozen recipe. Score file `scores_v3_par.jsonl`, sha256 `8068bc9233618303311b281498cb3ef72983c0b652c3e1a01df09f72a3a7caf7`, 316 rows, 0 errors. **Pre-committed check (a) passed: P1 (the canonical text) bit-matches the entry-E rubric file on 316/316 probes, max |ΔE| = 0.0000** — the instrument reproduces exactly across passes; everything below is real framing sensitivity, not instability.

**Pools (unchanged, claim-blind):** passed-session probes n = 28, failed-session probes n = 288.

| gate | confirm (of 28) | false-approve (of 288) |
|---|---|---|
| **FROZEN: min_j E_j ≥ 2.0 AND mean ≥ 2.5** | **3/28 = 10.7 %** | **32/288 = 11.1 %** |
| descriptive: min_j ≥ 2.0 | 3/28 = 10.7 % | 59/288 = 20.5 % |
| descriptive: mean ≥ 2.5 | 3/28 = 10.7 % | 32/288 = 11.1 % |
| descriptive: all-5 modal high | 5/28 = 17.9 % | 97/288 = 33.7 % |
| comparison: single-read E ≥ 2.5 (entry-E file) | 5/28 = 17.9 % | 54/288 = 18.8 % |
| comparison: single-read E ≥ 2.0 | 5/28 = 17.9 % | 111/288 = 38.5 % |
| comparison: single-read p_fully ≥ 0.5 | 10/28 = 35.7 % | 114/288 = 39.6 % |

**Verdict (frozen bar: ≥ 23/28 AND ≤ 14/288): NOT EARNED — both legs fail at once.** The frozen gate confirms 10.7 % of passed-session probes (needs ≥ 82 %) while still false-approving 11.1 % of failed-session probes (allows ≤ 4.9 %); every looser descriptive gate false-approves 5–8× the allowance. There is no threshold in the family that both confirms and stays safe.

**Why (pre-committed note (b) resolved — the wild-disagreement branch):** paraphrase spread (max_j − min_j E) averages **1.61** on passed-session probes (max 2.15; 25/28 probes above 0.5) and 1.36 on failed-session probes. Per-framing means on passed sessions: P1 1.78, P2 1.09, P3 1.43, P4 2.10, P5 2.63 — the framing sentences alone move the read by up to 1.5 points, while the passed-vs-failed separation *within* each framing is only ~0.15–0.4. **The read measures the sentence, not the page.** The analytic SD of the entry-E teacher-forced distributions corroborates: mean SD(E) ≈ 1.04–1.14 per pool (a uniform 4-level distribution is 1.118), SD ≤ 0.5 on essentially no probes — the "confident" single reads were near-flat distributions all along.

**Interpretation (as pre-committed in (c)):** the spread is prompt-texture sensitivity, not model uncertainty about page state. Combined with entry F (option order: robust, ±0.09), the picture is complete and internally consistent: **the 12B rubric read is order-stable but framing-fragile, with a near-flat underlying distribution — it is not reading page state.** Entry E's celebrated 0-FA property lives at strict-LOW thresholds (challenge iff E < 1.0) and never transferred to the approval direction; measured directly as an approver, at every threshold the instrument either confirms almost nothing or false-approves massively. **The confirmer role is closed: multi-read agreement cannot certify a signal that is not there.** Nothing ports; nothing deploys.

**Disclosures.** (1) The first launch of the pass died before any request: probe transcript paths are relative and the launch CWD was wrong, producing 316 FileNotFoundError rows and zero GPU reads; the error file was deleted and the pass relaunched from `/tmp/mw_bench` (the entry-C precedent — nothing analytical was read twice). (2) The llama-server PID recorded at entry-G infra setup (1501842) was the ssh-side wrapper, not the server; the process that served the pass on 127.0.0.1:8998 is PID 1501778 (start time consistent with the entry-G infra window), and teardown kills by fresh ps-verified PID. (3) P2–P5 are newly written framings, used here for the first time by design — they are the instrument, not a confound.

### 2026-09-23 — AUTOPSY of the 20 zero-pass tasks (147 failing sessions, transcript forensics): one root disease, and the environment already ships a free verifier for it

**Method.** Feature extraction over all 168 v3 transcripts (`autopsy_features.json`, `autopsy_signatures.json`): per-session cell counts, repeated-cell signatures, duration, last page snapshot, final assistant text, plus manual reading of representative transcripts. Diagnostic only — no pre-registered bar, nothing deployed.

**The 20 zero-pass tasks decompose into three signatures:**

| signature | sessions | share | what it looks like |
|---|---|---|---|
| quick stop (≤ ~16 cells, ends itself) | 99 | 67 % | one failed action → episode over → agent declares done or gives up within ~1–2 min |
| marathon loop (hits the 480 s cap, loop_max 10–37) | 34 | 23 % | after the first −1, alternates click/look on a *frozen* page; the single most-repeated cell is always the pure-look `browser.tab("main"); display(await tab.ariaSnapshot())` |
| shaking storm (omp compaction loop) | 14 | 10 % | click-link / navigate-tree only; root-caused separately to `compaction.thresholdPercent: 30` |

**The root disease under all three: the model cannot read MiniWoB's own reward line.** 115/147 failing sessions have `Last reward: -X.XX` printed in the very last snapshot the model read. **55 sessions end with an explicit success claim while that line is on screen** — e.g. click-collapsible: *"The task is complete. I expanded Section #22 and clicked Submit. Evidence: The final ariaSnapshot shows: **Last reward: −1.00**"* — the model quotes the failure signal as its success evidence. 51 of the 55 are quick stops; 4 are marathons that finally declared victory over a corpse. When the reward is positive, the same read-the-line habit is correct — passes and blind-fails are the same behavior, only the sign differs. Episode restarts (`browser.open` again) happen in only 21/169 sessions, so after a first-try −1 the model almost never re-tries; it either declares done or stares.

**The counterfactual that reframes the whole verifier program:** a five-line finish-brake — *if the latest snapshot contains a negative `Last reward:`, veto the finish and re-open the episode* — would have intercepted every one of the 55 blind finishes (and the 30 non-blind marathons, which sit on the same visible signal) with **zero false vetoes on the 21 passes** (their reward line is +1). No LLM read required; the environment prints the ground truth and the 12B walks past it. Entries C–G's LLM-based verifiers (binary done-read, rubric, multi-read agreement) were all competing against a regex the domain hands out for free — and losing. (Caveat kept honest: the 12 `raw=null` sessions are collector failures counted as failures by the frozen rule; they do not change any rate by more than the previously disclosed ≤4-point sensitivity.)

**Implications carried forward to the strategic entry:** the decide layer's missing piece is not a better LLM read — it is wiring the *existing, mechanical* signals (reward line, episode counters) into the finish gate. That is chain architecture work, not model work.

### 2026-09-23 — JEVBENCH PUBLIC-SUITE CALIBRATION: not runnable (documented dead end); substituted with the same measurement on our own stack — every read is worse than a constant predictor

**Feasibility check (all sources examined 2026-09-23).** The "public suite" is not public. `taeold/djev-run` is a Cloud-Run deployment README: no benchmark commands, no task data, no harness; its leaderboard link (`benchmarksheaven.com/jev-models`, "JevBench v1.3.0, N = 231 public suite") does not resolve in DNS, and no search index has any trace of JevBench or its leaderboard. The upstream `mmastrac/djev` is a single `structured_server.py` plus README — no suite, no scoring code, no CLI. The `/v1/systemone` decision API it serves requires a custom vLLM build (PR #57250: DiffusionGemma seeded canvas, pinned template positions, per-slot token distributions); llama-server cannot serve it and our model is not a diffusion-Jev class. **Verdict: there is nothing to run and nothing to request — the calibration import is dead, not deferred.**

**Substitution (the measurement, not the suite).** What we wanted from JevBench was its calibration axis — the one that "zeroes label-only systems." That measurement is computable on our own data with zero new reads: the three existing v3 read confidences vs session truth (event = session passed), 316 probes, base rate 8.9%. Disclosed as **post-hoc descriptive** — no pre-registered bar, chosen after seeing entries E–G, exactly the kind of analysis the overfit discipline says not to turn into a deployment decision.

| confidence | ECE-10 | Brier | AUROC |
|---|---|---|---|
| constant predictor (always p = 0.09) | — | **0.081** | — |
| binary done-read p_done | 0.150 | 0.131 | 0.568 |
| rubric p_fully | 0.363 | 0.273 | 0.507 |
| rubric E/3 | 0.494 | 0.373 | 0.511 |

Every read is **worse than a constant** on Brier; AUROC 0.51–0.57 is coin-flip territory. The tail is inverted: on probes where the done-read said p_done ≥ 0.8 (18 probes, mean confidence 0.88), the observed pass rate was **16.7 %** — and 0/6 at p_done ≥ 0.9. The label-only baseline (claim polarity alone) scores 56.3 % accuracy against the 91.1 % always-say-failed constant. On the JevBench-style axes our entire read stack scores zero: confidence carries no outcome signal. This is the external yardstick's verdict on entries E–G, reached without their suite.

### 2026-09-23 — STRATEGIC ENTRY: does the chain earn its complexity? — Not in its current form. Verdict, evidence table, and the redirected plan

**Verdict.** On browser tasks, the decision layer's LLM reads do not pay for themselves today. The parts of the program that demonstrably work are the parts that are *not* LLM reads: the audit trail, the gate skeleton, the mechanical signals the environment hands out, and the methodology. Recommendation: fold the layer down to those parts, wire the mechanical finish-gate now, and hold all LLM reads to an explicit bar before they come back.

**The evidence table (whole program, one glance):**
| claim about the chain | verdict | source |
|---|---|---|
| improves pass rate on browser tasks | **no** — 11/48 vs vanilla 10/48, statistical tie | MiniWoB A/B |
| its reads detect wrong success | **no** — det ≤ 25 % everywhere, AUROC 0.51–0.57 | entries C–E + calibration |
| its reads are safe approvers | **no** — 3/28 confirm at 11 % FA; no threshold is both | entry G |
| its reads are order-robust / reproducible | **yes** — 6.3 % flip; P1 bit-exact 316/316 | entry F, entry G |
| its reads are calibrated | **no** — worse than a constant on Brier at every source | calibration entry |
| its low-FA property is real | **yes, but only for strict-LOW challenge** — never transfers to approval | entries D–G |
| the ask-user gate ranks errors | **yes** — errors = lowest confidences, ≈0.4 catches both at 10 % ask, precision 1.0 | live scale eval |
| the decision log earns its keep | **yes** — every v3.14.x defect was proven from logs/transcripts | fix-and-retest entries |
| its complexity carries risk | **yes** — +13.7 % wall (sequential dispatch, not deployed); context-overflow and read-corruption defect classes were chain-borne | entries A, M |
| a better verifier is missing | **wrong frame** — the environment ships one: the reward line. A regex finish-brake catches 55 blind finishes + would redirect the 30 marathons, at 0 false vetoes on the 21 passes | autopsy |

**What to do (in order):**
1. **Harness, now (no model work):** in the runner, teach the reward line in one sentence ("Last reward: +1 means success; any negative reward means the episode ended in failure — say so and re-open"); add a mechanical finish-gate that vetoes success claims while the latest snapshot shows a negative reward and forces an episode re-open; assert-zero-shakes via the `--config` overlay on every launch.
2. **Re-run the only A/B that matters:** chain + mechanical gate vs vanilla + the same one-sentence prompt fix. If the tie survives *that*, the LLM decision layers are cut from the browser-task path.
3. **Suspension bar for LLM reads (explicit, so the decision is reversible):** a read returns to the chain only if, on held-out session truth, it shows AUROC ≥ 0.75 AND Brier better than the cell's constant predictor — measured with the pre-registration discipline (order control included), on fresh sessions. Nothing current comes close (0.51–0.57).
4. **Keep unconditionally:** decision log, gate skeleton + ask-user ranking, verified server config, pre-registration/order-control methodology, the calibration harness built today (it is now the standing acceptance test for any future read).

**What this entry is not:** a claim that the reflex/decide *idea* is wrong — it is a claim that on this domain, with this 12B, the read layer has no signal to gate on, and the measurable wins live one layer down, in mechanics. The strategic decision is recorded here so the next sweep starts from it rather than re-litigating it.

### 2026-09-23 — STRUCTURED CONCLUSION written: `POC_CONCLUSION.md` (workspace-local, companion to poc_report.html)

The whole program — 2026-09-17 → 2026-09-23, seven eras — condensed into one document: the question and claim style, what was built, each era with its headline numbers, the seven confirmed wins, the seven measured negatives, the durable assets, the strategic decision record, and the honest limits. Terminal one-liner: the PoC proved the routing mechanism works where it can read real signal, measured precisely where the signal ends, and ended by pointing the next unit of effort at the harness floor instead of the model ceiling.

---

## NOTE — "Rewire" named; story page shipped (2026-09-23)

Project named **Rewire** (the layer between inference engine and OpenAI-compatible API; no weights, no engine changes). New artifact: `rewire.html` (workspace-local, sibling of `poc_report.html`, never synced to host).

- **Positioning written into the page**: response to TypeSafe's System One Jev; the post-Jev open-source wave replicates/competes, Rewire instead *combines* a generalized classifier (fast, hallucination-free-by-construction, not accuracy-guaranteed) with an LLM's thinking/multimodality/generation — via API-layer wiring only.
- **Structure**: 5-act scroll story — hero with animated pipeline SVG + count-up stats; Act I Jev context (crowd-vs-bet cards); for-everyone "two brains" primer + 3-scenario interactive quiz; interactive layer explorer (agent/REWIRE/engine/frozen-weights, click-to-explain) + vanilla↔rewired base-URL toggle demo (0/5→5/5, 12B 0/5→4/5); evidence tabs (MCQ/web/tools/routing, exact numbers + p-values); 7-era timeline; in-the-wild section (live 3-arm + MiniWoB honest tie + reward-line regex win); honest-ledger section (5 falsifications + 1 caught-and-fixed + discipline chips); verdict quote + 4-point plan; future section (native decide in the architecture with today/tomorrow architecture sketch, multimodal decide 0.960, mechanical-first gating, /v1/decide spec). All numbers sourced from POC_CONCLUSION.md / poc_report.html / this ledger.
- **QA**: rendered in in-app browser at desktop 1440px + mobile 420px; interactivity verified (layer explorer, URL toggle, quiz, tabs); inline JS node --check clean. Two factual fixes made during QA: (1) wrong-terminal attribution corrected to match source — the CHAIN's errors are the confident/terminal ones (29 wrong-action fails vs vanilla's 18 inert stumbles); (2) hero 60× label aligned to report convention (17.29 s vs 0.29 s no-think baseline).
- No new infra (page is static, self-contained, no external requests). Local `python3 -m http.server 8899` serving the workspace for preview.

---

## NOTE — publication prep for public GitHub repo (2026-09-23)

Repo prepared for first public push (`github.com/zean00/rewire`, main has no commits yet — first commit is the public one):

- **README.md** written (positioning vs Jev, DECIDE/GATE/THINK, honest results table incl. negatives, quickstart for example + proxy, repo layout, method/roadmap/credits). **LICENSE** added (MIT). **pyproject.toml** rebranded `hybrid-qwen` → `rewire`; `hybrid/__init__.py` docstring notes the historical import name. **.gitignore** gained `.venv/` + `.zcodeignore`.
- **Proxy promoted into the repo**: final working copy (v3.15.0 lineage, the file that served the live runs) moved from the scratch dir to `proxy/omp_proxy.py`; its backend/adapter configs shipped as `eval/proxy_backend.json` + `eval/adapter_learning.json` (both loopback-only, no identifiers).
- **Scrubs for publication**: `<gpu-host>` placeholders replaced the internal host/user in this ledger (5+3 hits, all in the infra-provenance blocks) and three standing-container names redacted; `scripts/sync.sh` rewritten to require an explicit `REMOTE=user@host` env var.
- **Scratch dir archived out of the repo** to `~/workspace/_refleqwen_archive/tmp_edit_20260923/` (60+ versioned proxies/run scripts/debug files; includes `models.yml.bak.v313`, which contains omp provider entries — kept local, never staged). `.pytest_cache/` removed.
- **Verification**: identifier re-scan over the whole tree = 0 hits; all Python compiles (`compileall`); `git add -n .` previews 77 files, all intended. `pytest` requires the deps env (`pip install -e .` / GPU host venv) — local laptop lacks torch; code paths untouched by cleanup except the `__init__` docstring and proxy usage-path line.
- Note: the host-side copy of this ledger was intentionally **not** overwritten with the scrubbed version (operational host/ports remain useful privately); publishing happens from the local repo.

### 2026-09-23 — ENTRY H PRE-REGISTERED (before any instrument download, embedding, or head fit): the suspension bar's first trained read — a CLM-class contrastive readout (frozen 8B encoder + tiny trained heads) tested as outcome verifier; two-stage: frozen screen on the existing 316 probes, binding test on fresh sessions

**Why this entry exists.** The strategic entry suspended all LLM reads behind an explicit bar: AUROC ≥ 0.75 AND Brier better than the cell's constant predictor on held-out session truth, pre-registered, order-controlled, on fresh sessions. Entries C–G killed the *untrained* reads (AUROC 0.51–0.57; Brier 0.131–0.373 vs constant 0.081; framing moves the read ~1.5 rubric points vs 0.15–0.4 of outcome separation). A trained readout class now exists publicly: **Contrastive-LM's CLM** (github.com/Contrastive-LM/CLM, Apache-2.0) — a frozen Qwen3-8B encoder (last-token pooling) + two 20M-param projection heads (state/action) trained with bidirectional InfoNCE over ~60M QA pairs → ~30M synthetic hard negatives → ~1M agentic trajectories; inference by embedding dot product with candidate caching; embeddings-only fine-tuning supported (`train/finetune.py --task prm`). It attacks, by construction, exactly our three measured deficits: outcome-paired training (our reads had none), no first-token logit texture for framing fragility to act through, and trainable calibration. Their card reports **no base-rate/constant-predictor baselines and no Brier/ECE** — their verifier SOTA numbers (87.6% / 81.6%) are best-of-N *relative selection* on 30–38 tasks, a different regime from our absolute read, and precisely the gap our bar tests. **This entry is registered before any CLM checkpoint is downloaded.** Instrument versions (checkpoint HF revision hash, code commit) are recorded in the run record at download time.

**Claim under test.** A trained contrastive readout over frozen session state carries enough outcome signal to clear the suspension bar — the first instrument class tested against it that was trained toward outcome prediction. Secondary claim (earned or refused separately): framing robustness — the entry-G failure mode — is structurally absent, not merely reduced.

**Frozen corpus and state assembly.** The existing v3 corpus: 168 sessions, 316 probes, pools unchanged and claim-blind (28 passed-session probes / 288 failed-session probes, base rate 8.9%, constant-predictor Brier 0.081). Probe input text = the exact state block the entry-E rubric read consumed for that probe (goal lines + snapshot, same caps), **minus the rubric question/framing sentences** — the instrument must read the page, not the ask. The assembly recipe is pinned to the entry-E scorer on the GPU host; if it cannot be reconstructed and verified bit-exactly against the recorded prompt strings *before any embedding is computed*, that is disclosed and the entry halts (recipe verification precedes data). Contamination: the corpus is local-only (collected 2026-09-22/23, never published); CLM's training sources (Nemotron DQA, synthetic negatives, DeepSWE trajectories) cannot contain it. The encoder is Qwen3-8B, not our frozen 4B — the claim is about the readout class, not our backbone; a pass would leave deployment cost (serving an 8B embedding encoder alongside the chain) recorded but unresolved.

**Stage 1 — screen on the existing 316 probes (unlocks or kills the fresh spend).**
- **H1 (zero-shot geometry):** per probe, margin m = s·a⁺ − s·a⁻ over the five frozen action-text pairs below; probability = logistic(m) with (w, b) fit on training folds of the scalar margin only (2 parameters), applied out-of-fold.
- **H2 (trained head):** features = the 4096-d state embedding (state head only; action head unused). Exactly two architectures: L2 logistic regression (C ∈ {0.01, 0.1, 1, 10}) and one MLP 4096→64→1 (ReLU, Adam lr 1e-3, weight decay 1e-4, ≤ 50 epochs, class-balanced BCE). Selection between and within them by inner grouped-CV on training folds only. Class weights balanced (28 vs 288).
- **Splits:** grouped 5-fold CV by session, folds frozen from seed 13, stratified on session outcome; no session on both sides of any split; no threshold tuning or model selection on out-of-fold predictions; one run per configuration; embeddings computed once, single batch.
- **Frozen action-text pairs (five, task-generic; A⁺/A⁻):**
  1. "The agent completed the task successfully." / "The agent failed the task."
  2. "The task's success condition was met." / "The task's success condition was not met."
  3. "The session ended with a correct final result." / "The session ended with an incorrect result or no result."
  4. "The goal was achieved." / "The goal was not achieved."
  5. "Everything the task asked for was done correctly." / "Something the task asked for was done wrong or left undone."
- **Paraphrase leg (H1, judged separately):** per-pair out-of-fold AUROC; earned only if every pair ≥ 0.70 AND max−min range ≤ 0.10. (Entry G's instrument failed the analog: rewording moved reads ~1.5 points against 0.15–0.4 of real separation.)
- **Stage-1 gate (the bar itself, per arm, on pooled out-of-fold predictions): AUROC ≥ 0.75 AND Brier < 0.081.** The entry's gate passes if **either arm** clears both legs; H1's paraphrase leg earns or refuses only the framing-robustness claim. **FAIL closes entry H**: the role stays suspended and the ledger records the third failed readout class (untrained logits, untrained rubric reads, trained contrastive readout). No fresh sweep is spent on a Stage-1 failure.

**Stage 2 — binding confirmation on fresh sessions (only if Stage 1 passes).** A fresh sweep (new seeds drawn at launch, same harness, page-reward labels), probes drawn by the entry-E recipe, bar applied verbatim on the fresh pool: AUROC ≥ 0.75 AND Brier better than the fresh cell's constant. Sweep size fixed before any fresh session is analyzed. **A Stage-1 pass with a Stage-2 fail is recorded as entry-H failure (screen optimism), never as a win.**

**Cost.** Stage 1: one embedding pass over 316 states on an fp16 8B encoder (~16 GB, fits the GPU host when free) plus minutes of head fitting. Stage 2: one sweep at the standing harness cost. Nothing here touches the frozen 4B, the engine, or the shipped proxy.

**What would make this entry meaningless (pre-committed as violations):** fitting anything on out-of-fold predictions; recomputing embeddings after seeing metrics; un-blinding pools before splits are frozen; substituting CLM's public DeepSWE embeddings for our states (different domain — that would test their data, not our claim).

**Candidate entry I (stub — NOT registered):** decide-replacement arm — CLM `rank()` over our live-site candidate sets vs the D2 ground truth (element 0.850), near plug-compatible via their `/v1/rank` + candidate-embedding caching. Registered separately only if pursued; over-registering untested arms would dilute the discipline this ledger runs on.

## NOTE — story page renamed for GitHub Pages (2026-09-23)

`rewire.html` → `index.html` so GitHub Pages serves it as the site root (`https://zean00.github.io/rewire/` once Pages is enabled: Settings → Pages → Deploy from a branch → main /(root)). README gained the live link + updated layout entry. File content unchanged; still fully self-contained (no external assets), so Pages needs no build step.

### 2026-09-24 (host clock) — ENTRY H HALTED BY ITS OWN PRE-REGISTERED CLAUSE, before any embedding: the frozen corpus is gone — /tmp/mw_bench (168 v3 transcripts, the 316 probes, the entry-E/G scorer recipe, the sha256-pinned score files) was deleted from the GPU host sometime after the last mw-era file (Sep 23, 17:14); full-disk search finds no copy; recipe verification is impossible, and with it all of Stage 1

**Recon findings (read-only, at execution start).** Entry H execution began with its own first pre-registered checkpoint — corpus and recipe verification. Results: (1) `/tmp/mw_bench` does not exist. (2) Other files of the same period survive untouched in /tmp (`llama12b_entryG.log`, Sep 23 17:14; `rev_frames_input.json`, Sep 22; `proxy8990.log`, Sep 23 07:31) — so this was not an age-based tmp wipe but a removal of that directory specifically; the host is shared with other tenants (one tenant container restarted ~49 min before the check), and no attribution is claimed or known. (3) Full-disk search (root filesystem, pruning /proc, /sys, /dev, /var/lib/docker, package caches) for `main_v3.jsonl`, `scores_v3*`, `main_runs_v3`, `probes_v3*`: zero hits. Trash empty. The local workspace and the Sep-23 scratch archive hold proxy and run-script lineage but no MiniWoB corpus. (4) The sha256 pins in this ledger for `scores_v3_bin_ord.jsonl`, `scores_v3_rub_ord.jsonl`, `scores_v3_par.jsonl` now point at files that exist nowhere we can reach.

**The clause, applied as written.** Registered 2026-09-23: "if it cannot be reconstructed and verified bit-exactly against the recorded prompt strings before any embedding is computed, that is disclosed and the entry halts (recipe verification precedes data)." Neither the recipe, nor the recorded prompt strings, nor the 316 probe state texts survive. Stage 1 is impossible as registered; Stage 2 was gated on a Stage-1 pass. **Entry H halts.** No embeddings were computed, no CLM checkpoint was applied to any data, and no outcome metric was approached — the halt fired at the first pre-registered checkpoint, which is the discipline working as designed (and the reason score-file hashes were pinned in the first place).

**Second deviation discovered at recon (recorded for any successor entry).** The GPU is an RTX 5070 Ti with 15.9 GB VRAM (14.7 free at recon) — the registration's "fp16 8B encoder (~16 GB)" cannot fit on this card. A successor entry must pre-register the fidelity deviation explicitly (int8 bitsandbytes or bf16 with CPU offload), before any embedding.

**What survives, verified at recon:** the host repo copy with its eval/reports records (MCQ-era, live-scale, calibration), the 12B model file, the standing 4B proxy process (untouched), and this ledger — every number in the honest ledger was computed from recorded outputs (pinned hashes, in-log tables) and survives independently of the deleted corpus. The MiniWoB sweep runner scripts, however, lived in the deleted directory and are gone with it.

**Successor options (a new registration would be required; nothing executed).** (a) Entry H-prime on a fresh corpus: re-run a v4 sweep per the standing sweep spec (24 tasks x 7 fresh seeds, page-reward labels) to regenerate sessions, then apply the H protocol directly on that fresh data — this collapses the old two-stage design into the binding test on fresh sessions, which was always the only leg that could un-suspend the role; cost is one sweep plus one embedding pass, and the probe recipe must be re-specified from the ledger descriptions since the entry-E recipe artifacts are gone. (b) Keep the bar frozen and the mechanical-first verdict as the terminal record of the verifier program. The GPU-hours are the user's to commit; the decision is recorded open so the next session starts here rather than re-deriving it.

### 2026-09-24 (host clock) — ENTRY H HALT VACATED — the halt's factual premise was wrong: nothing was deleted; the corpus lives on the LOCAL workstation at /tmp/mw_bench, intact, all sha256 pins verified. Recon error disclosed; the registered protocol (2026-09-23) is back in force and resumes at recipe verification

**What the follow-up investigation (prompted by the user) found.** The prior halt entry concluded /tmp/mw_bench had been deleted from this GPU host. That was a machine-location error in the recon: the entire entry-era MiniWoB program ran on the LOCAL workstation (the laptop), not on this GPU host — consistent with the standing sync rule ("only IMPLEMENTATION_PLAN.md and code sync to host; reports stay workspace-local"), which I failed to apply to /tmp paths. Verified locally minutes ago: `/tmp/mw_bench` exists with `main_runs_v3/` = 168 session dirs, `main_v3.jsonl` = 168 labels, the original entry-E files (`scores_v3_bin.jsonl`, `scores_v3_rub.jsonl`), and all three sha256-pinned files **matching the ledger byte-for-byte**: `scores_v3_bin_ord.jsonl` → 0a6a5184…cee8b, `scores_v3_rub_ord.jsonl` → f91870d1…63a9, `scores_v3_par.jsonl` → 8068bc92…7caf7. The sweep machinery (`mw_run.py`, `mw_cdp.py`, `mw_autopsy.py`, `mw_tally.py`, scorer and analysis scripts) is also intact locally. Nothing was ever deleted; no tenant, cleaner, or process removed anything. The GPU-host exonerations recorded in the halt entry (tmpfiles 30-day rule, no cron, clean shell history, no container mounts of /tmp) remain true but were answers to a question about the wrong machine.

**Status of entry H.** The halt is vacated. The pre-registered protocol of 2026-09-23 is back in force, and execution resumes at its actual first checkpoint — recipe verification — which is now genuinely possible: the scorer scripts and transcripts needed to reconstruct the 316 probe state texts exist locally. No embeddings have been computed yet; the Stage-1 gate (AUROC ≥ 0.75 AND Brier < 0.081 on pooled out-of-fold predictions, grouped 5-fold CV by session, seed 13; H1/H2 arms as frozen; paraphrase leg for H1) stands exactly as registered.

**What survives of the halt-era work (unchanged and still useful).** The instrument pins and smoke test (CLM repo @ cca045f, heads checkpoint sha256 b2b4a8c9…eda5, encoder snapshot b968826d, canonical recipe, bit-exact deterministic pipeline) remain valid — they were data-independent. The VRAM finding stands and now matters for THIS card only in that embeddings will be computed on the GPU host (RTX 5070 Ti, 15.9 GB): the registered "fp16 8B encoder" cannot fit; per the registration's own disclosure discipline this deviation is pre-committed **before any probe embedding**: encoder = bf16 with CPU offload (the smoke-tested configuration), disclosed as a fidelity deviation from the registered fp16 description.

**Lesson recorded.** A recon negative ("not found") must state WHICH machine it searched. The halt entry's cost — one misdirected investigation and two ledger entries — is the price of that omission; the discipline of pins and verification-first is what made the error recoverable in minutes rather than fatal.

### 2026-09-24 (host clock) — ENTRY H STAGE 1 ADJUDICATED: gate NOT EARNED by any arm — the CLM contrastive readout fails the frozen bar from every direction, the paraphrase leg is not earned, Stage 2 is not triggered, and the verifier role stays suspended

**Execution record (the registered protocol, run as registered).** Recipe verification came first, as ordered: the 316 probe state texts were rebuilt with the entry-G scorer's own code (`ckpt_rubric.build_evidence`, EVIDENCE_CAP=8000, executed from the corpus root on the LOCAL workstation where the entry-era artifacts live) and checked against the per-row `ev_chars` that scorer recorded — 316/316 exact; the label join reproduced the ledger pools exactly (28 passed-session / 288 failed-session probes; rule `raw > 0`, which admits the 0.5 and 0.2 partial-credit sessions). Then one embedding pass on the GPU host: 316 state texts + 10 action texts (the 5 registered positive/negative pairs, verbatim), frozen heads checkpoint sha256 b2b4a8c9…eda5, canonical recipe, bf16 CPU-offload encoder (the pre-committed fidelity deviation), 107.95 s wall clock, determinism recheck bit-exact. Then the fit, per the frozen script: seed 13, StratifiedGroupKFold 5-fold grouped by session; H1 = zero-shot margins fed to a per-fold 2-parameter logistic; H2 = inner-selected logreg-C/MLP with grouped inner CV and AUROC-only selection. The script self-asserted manifest determinism, 28 positives, and unit-norm inputs before fitting.

**Deviation disclosure (executions).** The fit script ran twice: the first execution crashed inside the H2 inner loop (a torch autograd error — the predict function ran outside `no_grad`) BEFORE any metric was computed, printed, or written; the fix moved `no_grad` inside the predict function, changing no fitted quantity. The second execution is the reported run; nothing was observed from the first, so this was crash-repair, not result-based refitting. Also disclosed (host env): scikit-learn 1.9.1 was installed via uv into the existing hybrid env (which ships no pip) before the fit. The pre-committed encoder-precision deviation (bf16 CPU-offload instead of the registered fp16 description) was disclosed in the vacatur entry before any probe embedding.

**Results (pooled out-of-fold predictions, 1580 H1 rows / 316 H2 rows; gate per arm: AUROC ≥ 0.75 AND Brier < 0.081, the cell's constant predictor).**

| arm | AUROC | Brier | ECE-10 | AUROC leg | Brier leg | gate |
|---|---|---|---|---|---|---|
| h1_headspace (PRIMARY) | 0.7015 | 0.0807 | 0.0113 | FAIL | pass | **FAIL** |
| h1_encoderspace (SECONDARY) | 0.5910 | 0.0805 | 0.0086 | FAIL | pass | **FAIL** |
| h2 (trained head) | 0.9926 | 0.0957 | 0.2431 | pass | FAIL | **FAIL** |

Paraphrase leg (judged on h1_headspace as registered; bar: every pair AUROC ≥ 0.70 AND max−min ≤ 0.10): per-pair AUROCs 0.7159 / 0.4929 / 0.7750 / 0.6796 / 0.6476, range 0.2821. Pair 2 sits at chance and the spread is nearly 3× the allowance — **NOT EARNED**. The head-space margin's direction depends on which phrasing pair you use, which is precisely the fragility this leg was designed to expose.

**Adjudication.** No arm passes both legs; per the registration, Stage 2 (the binding fresh-session confirmation) is not triggered. Verdict: the CLM-class readout does not earn the verifier role at the frozen bar. The verifier role stays suspended; the chain stays mechanical-first; the strategic entry's verdict stands. This is the third independent read family to fail the same bar (prompt reads, rubric reads, now a trained contrastive readout on a frozen 8B encoder) — the bar itself remains the point.

**What the numbers actually say (record, not a reopened gate).** (1) The H1 arms pass the Brier leg only by predicting almost nothing — near-constant probabilities, ECE ~0.01: well-calibrated ignorance. Their ranking is below bar, and not stably so across rewordings. (2) H2 is the mirror image: near-perfect out-of-session RANKING (0.9926 AUROC, with whole sessions held out at both split levels — leakage-controlled), but the worst calibration measured in this program (ECE 0.243, Brier worse than always-say-8.9%). Inner selection chose by AUROC alone, exactly as registered — 4/5 folds picked logreg_C10, one picked the MLP — and the pre-registered conjunction gate was the protection against exactly this failure shape; it fired. (3) A read that can rank but cannot beat a constant is not a probability, and the gate's consumer is a chain decision, not a leaderboard. Both legs were frozen for this reason.

**Artifacts (committed under `eval/reports/entry_h/`).** `entry_h_results.json` sha256 7af52963d5b7d59d97670964276173342dd673ced4535d2d47b9240e61cb1169 · `entry_h_emb.npz` sha256 416cc76a6c6bc47ad5f832a66bdfd011b5004c3d75ff15285f9b1aae72d0db55 · `entry_h_states.jsonl` sha256 8a684a05b05691d171660ed9907f591626307589298336e898016d457e48dd01 (equals the manifest's states pin; recipe-verified 316/316) · `entry_h_fit.py` sha256 fd94c83de5fb501d2f601518ba42d4649a07ef107b5a230f538e3fb9571164ed · plus `entry_h_embed.py`, `entry_h_embed_manifest.json` (determinism_recheck_bitexact: true), `entry_h_states_build.py` (the recipe verifier), and the earlier instrument pins (`instrument_notes.md`, `clm_smoke.py`, `smoke_result.json`).

**Open decision (recorded; nothing executed).** Two paths, the same fork as before this entry ran: (a) a NEW registration for a calibration-directed variant — the honest note is that H2's AUROC leg is already met, so the open question is narrow and mechanical (selection or post-hoc scaling that also clears Brier < 0.081 out-of-fold), but it would still be a new entry with its own frozen bar, not a relaxation of this one; (b) keep the frozen bar and the mechanical-first verdict as the terminal record of the verifier program. GPU-hours and the choice are the user's.

### 2026-09-24 (host clock) — ENTRY J PRE-REGISTERED (before any fit): the one calibration-directed readout — entry H's ranking leg already cleared at 0.9926 out-of-session, so this entry re-aims ONLY the calibration machinery at the SAME frozen bar; pass or fail, it is the last calibration attempt on this corpus

(The label "entry I" stays reserved for the decide-replacement stub registered above; this entry takes the next letter.)

**Diagnosis under test (declared, because it motivates the design).** Entry H's H2 selected rankers by inner AUROC alone, under class-balanced training weights; the winners (logreg C=10 balanced in 4/5 folds, pos-weighted MLP in 1/5) rank held-out sessions near-perfectly (pooled AUROC 0.9926) while their probabilities are dragged toward 0.5 — poison at an 8.9% base rate (Brier 0.0957 vs constant 0.081, ECE 0.243). Entry J changes only the calibration machinery. The bar does not move.

**Data (no new embeddings — recomputing after seeing metrics remains a violation, inherited).** The entry-H artifacts as pinned: `entry_h_emb.npz` sha256 416cc76a…db55, `entry_h_states.jsonl` sha256 8a684a05…48dd01, labels/groups/folds byte-identical to entry H (StratifiedGroupKFold 5-fold, shuffle, seed 13, grouped by session; inner SGKF(4), seed 13). The encoder-precision deviation (bf16 CPU-offload) is inherited as already disclosed; no model runs at all in this entry — it is a CPU fit on frozen vectors.

**Arms (exactly two, fixed before the run):**
- **J-a (unbalanced direct):** L2 logistic regression on the 4096-d L2-normed state embedding, C ∈ {0.01, 0.1, 1, 10}, **no class weights**, max_iter 5000. Within each outer-train, inner SGKF(4, seed 13) selects C by **mean inner Brier** (tie → smaller C). Refit on the full outer-train; pooled out-of-fold over the same frozen 5 outer folds.
- **J-b (entry-H ranker + Platt recalibration):** the entry-H modal winner, logistic regression C=10 class_weight="balanced" max_iter 5000, used as a pure ranker. Inside each outer-train, inner SGKF(4, seed 13) yields inner-out-of-fold decision scores; a 2-parameter Platt map (1-D logistic regression) is fitted on those inner-oof scores only; the ranker is refit on the full outer-train and the frozen Platt map applied to outer-test scores. Pooled out-of-fold over the same 5 outer folds.

No thresholds, no clipping, no probability edits beyond the declared machinery. Both arms report pooled AUROC, Brier, ECE-10, a decile calibration table, and (J-a) the per-fold chosen C.

**Stage-1 gate (the bar itself, per arm, on pooled out-of-fold predictions): AUROC ≥ 0.75 AND Brier < 0.081. Entry J passes if either arm clears both legs.** No paraphrase leg: that leg was the zero-shot-margin robustness check; neither J arm consumes action-text pairs. A pass earns exactly what entry H's Stage-1 pass would have earned — the right to Stage 2, inherited verbatim from the entry-H registration (fresh sweep, seeds drawn at launch, probes by the entry-E recipe, sweep size fixed before any fresh session is analyzed, bar applied on the fresh cell's constant; **a Stage-1 pass with a Stage-2 fail is recorded as entry-J failure — screen optimism — never as a win**).

**Honest status of this entry (why it is a screen, not a discovery).** Entry J is designed AFTER entry H's Stage-1 numbers were seen, on the same 316 probes. That is precisely the risk the two-stage structure exists to contain; this entry is allowed because the structure — frozen bar, frozen folds, Stage-2 fresh confirmation — is intact, and because the cost is one CPU run on already-computed vectors. Its evidence can never be more than "earned the right to be tested."

**Terminal clause (pre-committed).** Entry J is the LAST calibration attempt on this corpus. FAIL → the verifier program closes and the mechanical-first verdict becomes the terminal record (option (b) of the entry-H adjudication, final). PASS → the program's next state is Stage 2, where it either confirms on fresh sessions or is recorded as screen optimism. No successor entry re-aims this role at the same 316 probes.

**One-execution rule and violations (inherited + specific).** As entry H: if the fit script crashes before any metric is observed, one disclosed crash-fix and re-run is permitted; results-based refitting is not. Pre-committed violations: fitting anything on out-of-fold predictions; recomputing embeddings; un-blinding pools before splits are frozen; changing folds, seed, C grid, or Platt construction after seeing any metric; adding arms after the first metric is printed.

**Cost.** Zero GPU, zero new data: one script, one run, minutes of CPU on the already-pinned embeddings.

### 2026-09-24 (host clock) — ENTRY J STAGE 1 ADJUDICATED: PASS — both arms clear the frozen bar that killed every prior read; the entry-H diagnosis was right (ranking was real, only the probability mapping was broken); the verifier readout has earned Stage 2 on fresh sessions

**Execution record.** The registered script ran exactly once, no crash, no re-run. In-script pins asserted before fitting: `entry_h_emb.npz` sha256 416cc76a…db55 and `entry_h_states.jsonl` sha256 8a684a05…48dd01 (the entry-H artifacts, unchanged); folds re-derived byte-identically to entry H (StratifiedGroupKFold 5-fold, shuffle, seed 13, groups=session_dir) and additionally dumped to `entry_j_folds.json` for any successor to verify. The outer folds and inner split rule are identical to entry H; the only new machinery is the two registered arms.

**Results (pooled out-of-fold, 316 rows; gate per arm: AUROC ≥ 0.75 AND Brier < 0.081).**

| arm | AUROC | Brier | ECE-10 | AUROC leg | Brier leg | gate |
|---|---|---|---|---|---|---|
| J-a unbalanced logreg (inner selection by Brier) | 0.9820 | 0.0548 | 0.0965 | pass | pass | **PASS** |
| J-b entry-H ranker + nested Platt | 0.9937 | 0.0169 | 0.0278 | pass | pass | **PASS** |

Both arms pass both legs; **J-b is the stronger arm and is the candidate readout going forward.** Its calibration deciles show a referee that behaves like one: 275 of 316 probes sit in the bottom decile with mean predicted probability 0.009 against an actual success rate of 0.000, and all 21 positives land in the top three deciles — the top bin alone holds 21 probes at mean p = 0.98 against 90.5% actual (19/21 correct, and the misses are near the middle of the scale, where a gate should be uncertain anyway). The constant predictor scores 0.081 on this pool; J-b scores 0.0169 — 4.8× better.

**What was isolated.** J-b is entry H's exact modal ranker (logreg C=10, class-balanced) plus nothing but a 2-parameter Platt map fitted on inner-fold out-of-fold scores — nested, never touching outer predictions. Its pooled AUROC (0.9937) matches entry H's mixed-model 0.9926: the ranking signal was real all along. J-a shows the other registered lever also works: the same feature set with no class balancing and Brier-based selection passes on its own. Entry H's failure is thereby attributed precisely — the class-balanced training weights and the AUROC-only selection criterion, both named in advance in this entry's registration as the suspects.

**Adjudication and status.** Per the registration: this is a **screen on already-looked-at data** (designed after entry H's numbers were seen, same 316 probes), so the pass earns exactly one thing — the right to Stage 2, inherited verbatim from entry H: a fresh sweep (new seeds drawn at launch, same harness, page-reward labels), probes by the entry-E recipe, sweep size fixed before any fresh session is analyzed, bar applied on the fresh pool's own constant predictor; **a Stage-1 pass with a Stage-2 fail is recorded as entry-J failure — screen optimism — never as a win.** The terminal clause flips accordingly: the verifier program does NOT close; its next state is Stage 2. The fresh sweep is one standing-harness run (GPU-hours on the host plus the local MiniWoB page server) — the spend is the user's to authorize, and if authorized, sweep size gets fixed in the ledger before the first fresh session is analyzed. The three prior read families remain dead; this is the first readout class to clear the bar at Stage 1.

**Artifacts (committed under `eval/reports/entry_j/`).** `entry_j_fit.py` sha256 c4c50405f9283db2147b740e937b938d75c4e9e7e71e120ab332830816a857cb (the executed script, pins included) · `entry_j_results.json` sha256 f2e04c4b5bb1aece57793480a310b1822c37bd8a94533d64f99d7d4081f586c3 · `entry_j_folds.json` sha256 fe061388aed5fe8a14e4451d1bed329ac0db3acca0246b46971c526ded2eb18a. Inputs as pinned in entry H; no new embeddings, no new model runs.

### 2026-09-24 — ENTRY J STAGE 2 LAUNCHED: sweep size, seeds, readout, and gate frozen here — before any fresh session runs; the binding test of the first readout class ever to clear the bar

**Authorization.** The user authorized the Stage-2 spend after the Stage-1 pass. Everything below was frozen and committed BEFORE the first fresh session started. One throwaway sanity session (seed 3306495589, `/dev/urandom`-drawn, outside the plan) verifies the full path first — infra check only, its data is never analyzed.

**Sweep (fixed).** 24 tasks × 7 fresh seeds = **168 sessions**, the standing harness cost, serial and resumable. Runner: `run_v4.py` = the entry-E `run_v3.py` verbatim except four frozen deltas — (1) seed RNG **3002643678**, drawn from `/dev/urandom` immediately before this registration; (2) exclusion of every v1/v2/v3 seed so all seeds are fresh; (3) v4 output paths (`main_v4.jsonl`, `main_runs_v4/`, CDP journal); (4) sanity seed. Task list, runner prompt, arm (`chain`), model string (`poc-proxy-12b/gemma-12b-decide`), caps (480 s omp / 120 s page), and CDP reward collection are unchanged; labels remain the page's own reward (`raw > 0` passes; `raw=None`/no page state counts as failure — the frozen rule). Infra relaunched this session exactly per the pinned entry-E recipe: llama-server on `127.0.0.1:8998` (`gemma-4-12b-it-nvfp4.gguf`, `--jinja -c 65536 -ctk q8_0 -ctv q8_0 -ngl 99 --flash-attn on -b 1024 -ub 512`, loopback only — the film-worker keeps the Tailscale listener) and the chain proxy on `:8990` (`eval/omp_proxy.py --port 8990 --backend-config eval/proxy_chain_remote_review.json --no-local-model`, frozen gates `configs/webgate_remote.json`). Standing 4B proxy (8999) and film-worker untouched. Sanity verified before the sweep starts; sweep duration expected ~12 h from the v3 record (mean 241 s/session).

**Probes (frozen recipe, content-blind).** After the sweep completes: entry-E extract (`ckpt_review.py` over all v4 transcripts) → frozen balance rule (polarity vocabulary substring on claim text; keep polar claims; sort by `(file, turn)`; Q = max(20, min(n_success_claims, n_failure_claims)); first-Q of each class) → `probes_v4_balanced.jsonl`. State texts by the pinned entry-H recipe (`build_evidence`, EVIDENCE_CAP=8000, goal lines + snapshot); the states file is sha256-pinned at build. Embeddings: the SAME pinned instrument (heads checkpoint b2b4a8c9…eda5, canonical recipe, bf16 CPU-offload deviation as disclosed), one pass, determinism recheck, fresh manifest.

**Readout (frozen before any fresh embedding exists).** Entry J's J-b exactly: ranker = logistic regression (C=10, class_weight="balanced", max_iter 5000) refit on ALL 316 pinned v3 state embeddings; Platt map = unpenalized 1-D logistic regression fitted on the out-of-fold decision scores of those 316 under the frozen entry-H fold split (seed 13, grouped by session) — nested on v3 data only, the fresh pool is never touched by any fit. Fresh states → ranker decision score → Platt → probability.

**Gate (verbatim, on the fresh pool, one adjudication pass).** AUROC ≥ 0.75 **AND** Brier < the fresh pool's own constant predictor (always predict the fresh base rate). **PASS → the verifier role is earned on fresh sessions; FAIL → recorded as entry-J failure (screen optimism) and the verifier program closes per the terminal clause.** No threshold tuning on fresh data; no refit, recalibration, or arm switch after any fresh metric is seen; probes, states, embeddings, and predictions are all computed by pinned code in one pass each; session-level rewards visible during collection are collection monitoring, not analysis — all reading of probe content happens after the last session lands.

**Teardown commitment.** After adjudication the host returns to standing state: the 8990 chain proxy and the 8998 llama-server stopped by ps-verified PID (each in its own ssh call); the standing 4B proxy and film-worker untouched; `models.yml` provider entry left as-is (dormant). Artifacts: seed plan, sanity record, sweep label file, probe/state/embedding files, prediction file, and results — all sha-pinned and committed.

### 2026-09-25 — ENTRY J STAGE 2 ADJUDICATED: GATE PASSED ON FRESH SESSIONS — the verifier role is EARNED; the first read in the program's history to clear the suspension bar out of sample; the strategic entry's condition is met by the J-b readout

**Sweep record (verified before analysis).** 168/168 sessions, exactly 24 tasks × 7 fresh seeds, seeds from RNG 3002643678 (all v1/v2/v3 seeds excluded), resumable runner run_v4.py (sha e5514aaf…2133, = run_v3.py verbatim except the four registered deltas). Raw-reward distribution: 21×1.0, 1×0.5, 1×0.6, 23×0, 107×−1, 2×−0.33, 1×−0.2, 12×None — **23 passed sessions (13.7%)**; the 12 no-page-state sessions count as failures per the frozen rule. Sweep wall ~11.6 h, one throwaway sanity session outside the plan.

**Probe pipeline (frozen recipe, one pass each).** Extract: 897 claim probes over 168 v4 transcripts, 111 dropped as ungrounded (quoted evidence absent) — the standard entry-E extraction. Balance: polar claims 215 success / 145 failure → Q = max(20, min) = 145 → **290 probes (145/145)**. States: 290 emitted by the pinned entry-H recipe (`ckpt_rubric.build_evidence`, EVIDENCE_CAP=8000), 32 in passed sessions (**11.0% base rate**, vs 8.9% in v3); `entry_j2_states.jsonl` sha256 4aa45387…d233. Embedding: the SAME pinned instrument (heads checkpoint b2b4a8c9…eda5, canonical recipe, bf16 CPU-offload deviation as pre-disclosed), one pass, 101.31 s, **determinism recheck bit-exact**; `entry_j2_emb.npz` sha256 08ccf9ca…b7f9.

**Deviation disclosure.** The v4 embed script crashed once at LOAD time — a stale `assert len(rows) == 316` carried over from the pinned v3 script — before a single embedding was computed; the row count was corrected to 290 and the reported pass is the genuine single embedding pass. `adjudicate_v4.py` is `adjudicate_v3.py` with path substitutions only (mechanical adaptation of the frozen extract/balance recipe; no rule changed).

**Results — the frozen readout applied cold (trained ONLY on the 316 v3 probes; fresh data touched by no fit).** Ranker: logreg C=10 balanced refit on all 316; Platt map fitted on the 316's frozen-fold out-of-fold scores; fresh 290 → decision score → Platt → probability.

| leg | fresh-pool value | bar | verdict |
|---|---|---|---|
| AUROC | **0.9889** | ≥ 0.75 | **PASS** |
| Brier | **0.0374** | < fresh constant 0.0982 | **PASS (2.6× better)** |

Calibration on fresh probes: 255 of 290 sit in the bottom decile at mean p = 0.004 against 0.8% actual success; the top decile (14 probes) reads mean p = 0.985 against 78.6% actual; the middle deciles are where its uncertainty lands, correctly. The screen-optimism hypothesis is **rejected**: Stage-1 numbers replicated out of sample almost unchanged (AUROC 0.9937 → 0.9889; Brier 0.0169 → 0.0374, both legs comfortable on a pool whose base rate nearly doubled).

**Adjudication.** Per the Stage-2 registration: **the gate is passed and the verifier role is earned by the J-b readout** — the first instrument class in the program's history to clear the strategic entry's suspension bar (AUROC ≥ 0.75 AND Brier better than the cell's constant, pre-registered, order-controlled, on fresh sessions). Four read families failed it (untrained logits, untrained rubric reads, trained contrastive readout uncalibrated, calibration-directed screen); the fifth earned it. The terminal clause's FAIL branch does not fire; the program's end state is the one it was built to reach.

**What this verdict is, and is not (recorded carefully).** It IS: the earned right of the J-b readout — frozen CLM instrument + tiny v3-trained calibrated head — to return to the chain as the verifier read, per the strategic entry's own condition. It is NOT yet: an integration. Where the read attaches (the GATE's completion-claim moments are the registered role), what threshold policy routes on p(success), and how failures are presented remain design decisions for a new registration. Two recorded follow-ups: (a) **deployment cost** — the read was computed offline; live serving of the 8B encoder beside the chain remains the open VRAM question (quantized variant = new instrument = re-validation; CPU-side = unmeasured latency; both recorded at Stage-1); (b) **the Gemma-native head candidate** — a head fitted on the chain's own frozen 12B vectors would collapse the deployment question entirely; it is a natural successor entry, deliberately NOT registered here (one question at a time; this entry's scope is the fresh-session verdict).

**Teardown (executed as committed).** The 8990 chain proxy and the 8998 llama-server stopped by ps-verified PID, each in its own ssh call; standing 4B proxy (PID 1234409, :8999) and film-worker verified untouched; the local MiniWoB page server stays up; `models.yml` dormant provider entry left in place.

**Artifacts (committed under `eval/reports/entry_j/`).** seed plan + runner: `run_v4.py` sha256 e5514aaf…2133, `seed_plan_v4.txt` · labels: `main_v4.jsonl` (laptop /tmp/mw_bench, 168 rows) · probes/states/embeddings/results: `entry_j2_states.jsonl` sha256 4aa45387…d233, `entry_j2_emb.npz` sha256 08ccf9ca…b7f9, `entry_j2_results.json` sha256 09c63884…0814, plus `entry_j2_embed.py` 16dab46f…d37a, `entry_j2_embed_manifest.json` (determinism_recheck_bitexact: true), `entry_j2_states_build.py` 89c2258a…4290, `entry_j2_apply.py` fc69f100…41e35, `adjudicate_v4.py`.

### 2026-09-25 — ENTRY K PRE-REGISTERED (before any Gemma embedding exists): the chain's own frozen 12B as the verifier's encoder — same dial recipe, same data, same bar; if it clears, the second-model deployment question dissolves. (Label "entry I" stays reserved for the decide-replacement stub; this entry takes the next letter.)

**Question.** Entry H's H2 showed that ~300 labeled examples suffice to fit a linear, calibratable dial on a strong encoder's raw vectors — on Qwen3-8B it reached AUROC 0.9926 out-of-fold. Gemma-4-12B is a comparable-class encoder whose space has never been tested for outcome readability. If the SAME dial recipe fitted on Gemma's own vectors clears the SAME bar cold on the SAME fresh pool, the verifier needs no second model: the dial rides the engine that already serves the chain, read through the stock embeddings API. This is a backbone-swap experiment, not a new instrument: no CLM component anywhere in it.

**Data (nothing new collected; states byte-identical to prior entries).** Train: the 316 v3 state texts (`entry_h_states.jsonl`, sha256 8a684a05…48dd01) with their labels and the frozen folds (`entry_j_folds.json`, sha256 fe061388…818a, seed 13, grouped by session). Test: the 290 v4 state texts (`entry_j2_states.jsonl`, sha256 4aa45387…d233) with their labels. No re-extraction; only the vectorizer changes.

**Extraction recipe (frozen; flag names verified against the installed build before this registration).** llama-server (stock, unmodified) launched on the GPU host as: `llama.cpp/build/bin/llama-server -m hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf --alias gemma-4-12b-it --host 127.0.0.1 --port 8998 --embedding --pooling last -c 32768 -np 4 -ngl 99 --flash-attn on -b 1024 -ub 512` — embeddings-only mode (no generation), last-token pooling, L2-normalized output (server default `--embd-normalize 2`, the same normalized-space convention the CLM recipe used), total context 32768 split over 4 parallel slots. Vectors obtained via `POST /v1/embeddings` with the state text **verbatim as the input — no chat template, no instruction wrapping** (the embedding endpoint embeds the raw string). Client: chunks of ≤8 texts, 4 concurrent workers, one pass over all 606 texts (316 + 290), no retries that recompute; determinism recheck = re-embed 8 sampled rows and require bit-exact equality; manifest records server launch line, vector dimension, per-pool counts, wall time, and the recheck result. Expected wall ~1 h (prompt-processing bound); runs in the background.

**Arms (the two entry-J survivors, transplanted unchanged; no new architectures).**
- **K-a (J-b transplanted):** logistic regression (C=10, class_weight="balanced", max_iter 5000) trained on all 316 v3 Gemma vectors; Platt map (unpenalized 1-D logistic) fitted on the 316's out-of-fold decision scores under the frozen folds; applied cold to the 290.
- **K-b (J-a transplanted):** unweighted logistic regression, C ∈ {0.01, 0.1, 1, 10}, selected by out-of-fold Brier on v3 under the frozen folds (tie → smaller C), refit on all 316; applied cold to the 290.

**Gate (verbatim, per arm, on the fresh 290): AUROC ≥ 0.75 AND Brier < 0.0982** — the same fresh pool's constant predictor computed at entry-J Stage 2 (base rate 0.1103; the fit script recomputes it from labels rather than trusting the constant). Either arm clearing both legs passes entry K. No paraphrase leg (no action-text pairs exist in this design).

**What would make this entry meaningless (pre-committed violations).** Any choice — normalization, C, Platt construction, client batching — informed by v4 metrics; a second embedding pass; re-extracting or editing state texts; changing the server line after any metric is seen; fitting anything on v4. One execution each for embed and adjudicate; the standing crash-fix exception (disclosed fix before any metric observed) applies.

**Honest status.** v4 is untouched by any K fit, but its summary statistics were seen during entry J's adjudication — the same relative status entry J had to entry H's numbers. A pass therefore earns exactly one thing: the integration registration, which will specify its own live confirmation before anything ships. A fail is terminal for the Gemma-native route (the CLM readout remains the earned fallback, with the deployment question reopened).

**Cost and teardown.** Zero new sessions, zero CLM artifacts, one embeddings-only server on the loopback port for ~1 h, one fit run. Teardown identical to entry J's executed commitments: stop by ps-verified PID in separate ssh calls; standing 4B proxy and film-worker untouched; VRAM verified back to baseline. Artifacts: `entry_k_emb.npz` + manifest, `entry_k_fit.py`, `entry_k_results.json` — all sha-pinned and committed.
### 2026-09-25 — ENTRY K ADJUDICATED: GATE PASSED on both arms — the chain's own frozen Gemma-12B is a readable verifier encoder; the dial rides the serving engine itself, no second model.

**Results (fresh 290, cold; gate AUROC ≥ 0.75 AND Brier < 0.0982, recomputed from labels: base rate 0.1103, constant Brier 0.0982).**
- **K-a** (J-b transplanted: logreg C=10 balanced + Platt on frozen-fold oof): v3 oof AUROC 0.8779 → **fresh AUROC 0.8867, fresh Brier 0.0653, ECE10 0.0501 — PASS** (top decile: 9/9 positive at mean p 0.95; bottom decile 4.8% positive at mean p 0.017).
- **K-b** (J-a transplanted: unweighted, C by v3 oof Brier → picked C=10 at 0.04784): **fresh AUROC 0.8706, fresh Brier 0.0714, ECE10 0.0322 — PASS**.
- Both arms clear both legs. K-a is the better ranker; K-b the better calibrated in ECE terms. Same ordering as entry J (calibration arm wins AUROC). The Gemma-native dial is weaker than the CLM Qwen3-8B readout (fresh AUROC 0.9889 / Brier 0.0374) but is the first backbone whose vectors were never calibrated by anything CLM-shaped, and it needs zero second-model VRAM.

**Disclosed crash-fix under the pre-committed exception — no metric existed at any point before the fix.** Pass 1 (the registered client verbatim: chunks ≤8, 4 concurrent workers) embedded all 606 texts in 65.0 s but **failed its own determinism recheck** (bit-exact = false). Diagnosis ran before any fit (`entry_k_diag.py`, `entry_k_diag2.py`): the stock llama-server build's **multi-input `/v1/embeddings` requests return position/neighbor-dependent vectors** — 8 copies of one text in a single request → 8 distinct vectors; the same text alone vs inside a batch differs by up to cos 0.87; all outputs are unit-norm (not a normalization bug); back-to-back identical requests are bit-reproducible. Conclusion: the registered client never measured a well-defined instrument, and the pass-1 npz was one arbitrary draw of neighbor noise. Fix (client-only; **server line unchanged**): one text per POST, strictly serial; the full pass re-ran exactly once (57.4 s) and its recheck — the 8 sampled rows re-embedded singly — is **bit-exact true**. This is the one re-run the exception allows; its artifacts stand regardless of the gate outcome, and the pass-1 manifest is preserved for the record (`entry_k_embed_manifest_pass1.json`), superseded by the re-run manifest.

**Standing instrument finding (beyond entry K):** on this llama-server build with this Gemma nvfp4 model, embeddings from multi-input requests must never be mixed or compared — any future consumer of this server embeds with one text per request. A batched request is not wrong "with jitter"; it is a different, ill-defined vector.

**Honest status (as registered).** v4's summary statistics were seen during entry J's adjudication — the relative status pre-registered for this entry. The pass earns exactly one thing: the integration registration, which will specify its own live confirmation before anything ships. Both earned readouts now exist: CLM/Qwen3-8B (stronger, 0.9889, needs a second model resident) and Gemma-native (0.8867, rides the engine's stock embeddings endpoint). Choosing between them — or registering the Gemma-native dial with the CLM path as documented fallback — is the integration registration's decision, made there, not here.

**Cost and teardown (executed as committed).** Zero new sessions, zero CLM artifacts, one embeddings-only server (~1.3 h total including diagnosis), one fit run, one disclosed re-run. Teardown: embeddings-only llama-server stopped by ps-verified PID 1833300 (command line matched the pinned launch line); verified gone; standing 4B proxy PID 1234409 (0.0.0.0:8999) and film-worker listener (Tailscale interface, port 8998) untouched; VRAM back to baseline (14.66 free / 15.92 GiB).

**Artifacts (eval/reports/entry_k/, sha256).** `entry_k_emb.npz` b413c88a…90c810 (v3 [316,3840], v4 [290,3840], labels, groups); `entry_k_results.json` deceb727…e92cf6; `entry_k_fit.py` 5476fc71…f2af80f2; re-run `entry_k_embed2.py` f5d96048…071373 + log 9a5c869f…34cdd; pass-1 `entry_k_embed.py` 2ff080a8…91414a; manifests 0be28c92…ce410 (re-run) / 6d75349b…85b1 (pass-1, preserved); diagnosis `entry_k_diag.py` ccc76374…30de1c, `entry_k_diag2.py` 90c5c544…8e22d; state texts byte-identical to entries H/J (8a684a05…48dd01, 4aa45387…d233).
### 2026-09-25 — ENTRY L PRE-REGISTERED: the integration registration — the verifier read ships as the Gemma-native dial (decision recorded), the head exports as data (3840 weights + 4 scalars + a threshold), the threshold is frozen from v3-only, and the last gate before anything routes on it is a fresh-sweep live confirmation with the same gate shape.

**Decision (recorded; the fork the entry-K adjudication left open).** The integrated verifier read is the **Gemma-native dial (entry K)**, not the CLM/Qwen3-8B readout. Recorded basis: equal gate outcome on the same fresh pool with zero second-model residency — the dial rides the serving engine's stock embeddings endpoint. The CLM readout (fresh AUROC 0.9889) remains the documented fallback and the stronger measured instrument should the confirmation fail. Arm: **K-a** (calibrated ranker) over K-b — better fresh AUROC (0.8867 vs 0.8706) and Brier (0.0653 vs 0.0714), and recipe-continuous with the earned J-b shape; K-b's better ECE10 (0.0322 vs 0.0501) is recorded as the known trade-off. This arm choice IS informed by v4 metrics — allowed here by design, because it is exactly the deployment decision entry K's adjudication reserved — and it is frozen from this line forward: no arm switch after any fresh metric, ever.

**The shipped artifact (head-as-data).** `entry_k_readout.json` (sha256 c840ba54…323e8; export script `entry_k_export.py` sha256 b515ca6d…4bf657, both under `eval/reports/entry_l/`): 3840 ranker coefficients + intercept, 2 Platt scalars, tau*, provenance shas, and the standing embed rule. Produced by a **mechanical reproduction** of entry_k_fit.py's K-a block on the pinned artifacts — the reproduction check (v3 oof AUROC 0.8779, fresh AUROC 0.8867, fresh Brier 0.0653, round-4 exact) had to pass before the dump, and passed. Serve time is a dot product, a sigmoid, and a threshold: no sklearn and no code beyond the embeddings endpoint in the serving path. **Decision threshold (tuned on v3 out-of-fold ONLY): tau\* = 0.326134** — the highest threshold attaining max Youden J on the v3 oof Platt probabilities; v3 oof confusion at tau\*: tp 21, fp 5, tn 283, fn 7 (TPR 0.750, FPR 0.0174).

**Attach point (frozen wording).** The GATE's completion-claim moments — the role reserved at entry-J Stage 2. When the chain claims completion, the pipeline extracts the same evidence block the entry-E rubric reviewer already read (ckpt_rubric.build_evidence semantics); the integrated read embeds that block **verbatim, ONE text per /v1/embeddings request** (the standing instrument rule from entry K's defect finding), applies the exported head, and routes: **p ≥ tau\* → the completion claim stands; p < tau\* → the claim is rejected as unverified and routed back**, with p and the block as the presented evidence. The old vocabulary read ("success"/"failed" substring) is demoted to a logged signal; it no longer routes.

**Honest domain gap (recorded, not assumed away).** Training and fresh-validation states were turn-level snapshots extracted by the frozen first-Q balance rule; the live GATE reads the completion-claim block — inside the validated text family, but not identical to its sampling rule. That gap is precisely what the live confirmation exists to measure.

**Live confirmation (pre-committed; the last gate before anything routes on the dial).** Design: **offline-on-fresh.** A fresh sweep in the standing harness — new SEED_RNG drawn at launch excluding every v2/v3/v4 seed, **168 serial single-arm sessions (24 tasks × 7 seeds**, page-reward labels), chain behavior UNCHANGED — the dial does not steer the confirmation sweep; closed-loop routing (the dial changing behavior) is a separate later registration, one question at a time. After the sweep: extract the completion-claim state block per session by the frozen recipe, embed singles, score with `entry_k_readout.json` verbatim, and gate on the fresh pool: **AUROC ≥ 0.75 AND Brier < the fresh pool's own constant predictor, one adjudication pass.** PASS → the integration is earned and the dial may route live completion claims. FAIL → recorded as entry-L failure; the Gemma-native integration does not ship; the CLM/J-b fallback reopens with its own deployment question; no re-aim at the same pool.

**Pre-committed violations.** Any coefficient, Platt, tau, or arm change after any fresh metric is seen; tuning anything on the confirmation pool; steering the confirmation sweep with the dial; batched embeddings requests; refitting or recalibrating on live data; sklearn in the serving path; skipping the export reproduction check.

**Cost.** Zero GPU-hours spent by this registration; the export was CPU-only on pinned artifacts. Measured head cost on the host card: ~95 ms per single-text embedding (606 singles in 57.4 s), head application microseconds — the expected added GATE cost per completion claim. The confirmation sweep is ~2.5 h in the standing harness by v3/v4 precedent; it runs only on the user's authorization, size fixed here (168) before any fresh session is analyzed.
