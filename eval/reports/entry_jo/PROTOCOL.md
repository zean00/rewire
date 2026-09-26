# ENTRY JO — Jev-Omni Q4_K_M GGUF calibration probe (PRE-REGISTERED)

Written and committed BEFORE any pool row was scored. One scoring pass over
the pool, no exclusions, no refits, no threshold tuning, no post-hoc changes.

## Question

Does the community Q4_K_M quantization of Jev-Omni (Reza2kn/Jev-Omni-Q4_K_M-GGUF,
source akhilaaa3/Jev-Omni rev c050d51, backbone gemma-4-12B-it) produce a
calibrated P(task complete) on our held-out completion-claim probes — comparable
to the shipped dial (entry-K readout)? This is the validation gate the 2026-09-26
discussion set: "run the labeled probe pools through the head and read the ECE."

## Pool (frozen)

`eval/reports/entry_l/entry_l_states.jsonl` — 140 rows, 26 positive. The same
held-out pool entry L adjudicated the dial on (held out from the dial's own
entry-K training). Label = page reward > 0 at the completion-claim moment.
Chain of custody: the file's sha256 must equal the entry-L embed manifest pin
(`states_sha256`); mismatch aborts before any scoring.

The dial's frozen entry-L results on this same pool (entry_l_results.json):
AUROC 0.9011, Brier 0.0861, ECE-10 0.0852, base rate 0.1857, constant-Brier
0.1512, tau* 0.326134.

## Instrument (frozen)

- llama-server local build, version string b8821 (1cd67807f), CUDA build but
  run CPU-ONLY (`-ngl 0`): the quantizer measured ~9.0 GiB VRAM for full GPU
  offload (RTX 5080 Laptop) and the local 5060 has 7.7 GiB — GPU does not fit,
  and entry O is live on the host, so the host GPU is out of scope entirely.
- Server line: `llama-server -m Jev-Omni-Unified-Q4_K_M.gguf --embedding
  --pooling none --host 127.0.0.1 --port 9021 --ctx-size 2048 --parallel 1
  --batch-size 2048 --ubatch-size 2048 --threads 10 --no-warmup -ngl 0`
  (quantizer's documented CPU config with ctx 2048 instead of their 1024 text
  config — the pool's largest evidence is ~1,381 chars ≈ ~350 tokens; threads
  10 instead of 8 because this laptop has 16 cores and the sweep client is
  network-bound. Both deviations disclosed here, pre-registered.)
- Loopback only: the embedding endpoint exposes internal model states.
- No KV quantization (standing lesson: it corrupts embeddings).
- Prompt construction and head math come from the quantizer's own adapter
  `jev_omni_gguf_decide.py`, imported and called unmodified (text-only route).

## Chain of custody (all checked before the first score)

| artifact | sha256 |
| --- | --- |
| Jev-Omni-Unified-Q4_K_M.gguf | 35bf51cb0dee2504e9f16093278ca853da2888251577236371630124605c40cd |
| decision-head-f32.npz | 47b346e120dc9110ef7c0b610cf5d1f459ddcdb05ffd3cd126b6938d88a8bb77 |
| jev_omni_gguf_decide.py | 11d07fdf14f22890ee92b872481052b7c66c0ac65adabf83a70bbdd5150d9b83 |
| entry_l_states.jsonl | must equal entry_l_embed_manifest.json `states_sha256` |

## Input mapping (frozen)

- `state` = the pool row's `ev` field, verbatim.
- `question` = "Has the agent completed the task?"
- `options` = ["Yes", "No"] (Yes first; P(complete) = probabilities["Yes"]).
- Adapter's trained text template and chat wrapper exactly as shipped,
  including the thought-channel suffix.

## Smoke gate (before the pool pass, non-pool inputs only)

1. README example: state "The meeting starts at 10 AM. It is now 9 AM.",
   question "Has the meeting started?", options Yes No → expect P(Yes) < 0.5.
2. Inverted synthetic: "The meeting started at 10 AM. It is now 11 AM." →
   expect P(Yes) > 0.5.
A smoke failure stops the pass; any instrument fix is disclosed before the
pool pass restarts (standing one-execution + disclosed-crash-fix rule).

## Metrics (frozen — identical definitions to entry_l_confirm.py)

p_yes per row (n=140, no exclusions; rows whose prompt hits the 2048 ctx cap
are reported as clipped but NOT excluded):

- AUROC (tie-correct rank, numpy; the scorer's metric code is first validated
  by recomputing the dial's p from the frozen `entry_l_emb.npz` +
  `entry_k_readout.json` and must reproduce the frozen 4-dp AUROC/Brier/ECE —
  a metric-code chain of custody. If it does not, the comparison aborts.)
- Brier score vs constant-Brier predictor (recomputed from labels).
- ECE-10, equal-width bins on p (bins = clip((p*10).astype(int), 0, 9)).
- **Pre-committed pass bar**: AUROC ≥ 0.75 AND Brier < constant AND
  ECE-10 ≤ 0.10. Pass ⇒ the quant earns a served-sidecar candidate and its
  own registered entry later. Fail ⇒ recorded honestly, like any outcome.
- Descriptive only (n=140, no significance claims either way): paired
  dial-vs-Jev stats — mean/max |p_jev − p_dial|, agreement at the dial's tau*,
  and per-row absolute-error win/loss vs label.

## Sequencing

Entry O (live sweep) is untouched: no host process touched, no serving config
changed, this runs on the laptop CPU at nice priority. The dial's frozen
numbers are never recomputed on the host.
