# ENTRY P — PRE-REGISTRATION (stuck/loop head on the sweep corpus)

Registered 2026-09-27, BEFORE the v7 sweep closed (row ~210/336 at registration)
and before any state was built, embedded, or scored. This file, the three
scripts below, and the ledger section are the registration. Nothing here may
change after the corpus exists except via the standing disclosed-crash-fix
exception, recorded in the ledger BEFORE embedding.

## Question

Can a linear head on the frozen dial embedding (gemma-4-12b-it, 3840-d,
pooling-last) detect, at turn level and from the same kind of evidence the dial
already reads plus a short action history, that the agent is stuck repeating an
action on an unchanged page — with AUROC >= 0.75 and Brier better than the
constant predictor on held-out sessions?

Motivation: the entry-O watch item. Guarded sessions hit the 480 s cap at ~2.5x
the vanilla rate (48/98 vs 19/98 at row 196), including sessions that had
ALREADY earned raw>0 and then looped to the cap. A working stuck head is the
cheapest candidate second head for the dial (same embedding, one more linear
layer) and attacks a measured wall-time cost.

## Corpus (frozen)

- Sessions: ALL 336 rows of the v7 sweep (`/tmp/mw_bench/main_v7.jsonl`),
  both arms pooled, arm recorded per row. The builder REFUSES to run unless
  main_v7.jsonl holds exactly 336 records with done=true. If the sweep closes
  short of 336, a disclosed ledger amendment is required before any smaller
  corpus is used — never a silent one.
- Transcripts: the one transcript jsonl per session dir under
  `/tmp/mw_bench/main_runs_v7/<task>_s<seed>_<arm>/` (verified: exactly one
  jsonl per dir, .cdp files excluded).
- Session outcome fields (raw, omp_exit, query, arm) come verbatim from
  main_v7.jsonl and are carried as DESCRIPTIVE fields only; they are not
  inputs to the label.

## Unit, signature, label (frozen, mechanical)

- ACTING turn: an assistant toolCall whose `arguments.code` contains a line
  matching `^[ \t]*//\s*(click|type|select|check)\b[^\n]*$` (first such line).
  Comment-less calls (browser open/observe) are excluded from the dataset
  entirely. The four verbs are the frozen acting set (v5 inventory: click
  1796, type 392, select 6, check 5).
- `sig` = that comment line, whitespace-collapsed, otherwise verbatim
  (case preserved; type payloads matter).
- Page snapshot at an acting turn = `ckpt_rubric.build_evidence(path,
  message_row_index)` — the newest toolResult text at or before that row,
  whitespace-collapsed by the builder (the entry-L code path, reused
  verbatim).
- label = 1 (LOOP) iff, within the current turn's trailing window of 8 ACTING
  turns (inclusive), the same sig occurs at >= 3 turns AND the page snapshot
  is IDENTICAL at all of those occurrences and at the current turn. Otherwise
  0. (Identical action on an unchanged page = no progress = stuck; a changing
  page under repeated clicks, e.g. a carousel, is progress and stays 0.)
- Early turns are 0 by construction (window too short for 3 occurrences).

## Input block (frozen verbatim template, embedded as-is)

```
TASK: <query, whitespace-collapsed, cap 300 chars>

PAGE SNAPSHOT:
<build_evidence text, cap 6000 chars>

RECENT ACTIONS (oldest first, current turn last):
<i. sig, cap 160 chars per line, for the last <=8 acting turns>
```

The evidence-only block of entries K/L is deliberately extended with the
action history: a loop is defined by repetition, which a single snapshot
cannot show. This is a new input spec for a new head; the completion dial's
input is untouched.

## Split (frozen, deterministic, session-level)

`h = int(md5(session_dir).hexdigest()[:8], 16)`; split = "val" if `h % 10 < 3`
else "train" (~30% of sessions to val). Computed once by the builder, stored
per row. No seed search, no re-split, no peeking.

## Instrument (frozen — identical pins to the entry-L embed manifest)

- Server: `llama-server -m hybrid-qwen/models/gemma-4-12b-it-nvfp4.gguf
  --alias gemma-4-12b-it --host 127.0.0.1 --port 8998 --embedding --pooling
  last -c 32768 -np 4 -ngl 99 --flash-attn on -b 1024 -ub 512` (HOST side;
  client usage only — the server is never started, stopped, or restarted by
  this entry).
- Client: ONE text per POST to `/v1/embeddings`, strictly serial (the entry-K
  standing instrument rule), model name "gemma-4-12b-it", server-default
  normalization. Determinism recheck: 8 sampled rows re-embedded singly,
  must be bit-exact, else abort before metrics.
- Estimated scale ~7-10k blocks; serial wall ~20-40 min. No parallelism.

## Training (frozen — entry-K recipe)

- Ranker: sklearn `LogisticRegression(C=10.0, class_weight="balanced",
  max_iter=5000)` fit on split=="train" rows.
- Platt: (a, b) fit on out-of-fold decision scores from `GroupKFold(5)` over
  the sorted unique train session_dirs (deterministic, grouped by session).
  p = sigmoid(a * z + b).
- ONE execution of the fit script. No C search, no class-weight search, no
  feature selection, no threshold tuning on val.

## Frozen gates (val rows only — mirrors entry-L)

- AUROC >= 0.75, AND
- Brier < the val pool's constant predictor (recomputed from val labels).

Pass both => the head earns an operational trial in a LATER registered entry
(intervening on its alerts inside the harness). Fail either => recorded like
any outcome; no refit, no salvage on this corpus.

## Descriptives (no gates)

ECE-10 on val; confusion at 0.5 and at the val-F1-max threshold; LOOP rate by
arm; among cap-hit (omp_exit==124) vs success (raw>0) sessions, the share of
final-5 acting turns labeled LOOP (concentration sanity); per-row wall-time
descriptive. Also recorded: full build manifest (shas, counts by split/arm),
embed manifest, states sha chain.

## Sequencing (frozen)

1. v7 sweep closes; entry-O adjudication executes and is recorded FIRST.
2. Corpus backup (transcripts + main_v7.jsonl) reaches the host.
3. `entry_p_states_build.py` runs ONCE (laptop), manifest + sha appended to
   the ledger BEFORE any embedding.
4. `entry_p_embed.py` runs ONCE (host), then `entry_p_fit.py` runs ONCE (host).
5. Results + verdict appended to the ledger, synced, committed.

Entry-L and its pool remain untouched held-out data for the completion head;
entry P never reads, trains on, or evaluates against them. The completion
dial's readout (entry-K export) is not modified by this entry in any way.
