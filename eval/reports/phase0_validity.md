# Phase 0 — Validity Report

## Environment

| key | value |
| --- | --- |
| torch | 2.11.0+cu128 |
| cuda available | True |
| device capability | (12, 0) |
| transformers | 5.17.0 |
| model | Qwen/Qwen3.5-4B |
| load path (HF class) | AutoModelForImageTextToText |
| dtype | torch.bfloat16 |

## 1. Pinned read-point determinism

- two identical forwards, max |logit diff| = **0.000e+00** (deterministic ✔)
- read_mode: `pinned_nothink`, template_kwargs: `{'add_generation_prompt': True, 'tokenize': False, 'enable_thinking': False}`

Pinned prompt (item 0) verbatim:

```text
<|im_start|>user
What is the capital of Australia?

Options:
1. Sydney
2. Canberra
3. Melbourne
4. Perth

Answer with exactly one of the listed options.<|im_end|>
<|im_start|>assistant
<think>

</think>
```
## 2. Pre-mask candidate mass

### D2 (alpha=1.0)

| metric | value |
| --- | --- |
| n | 20 |
| premask min / p50 / max | 0.0003 / 0.0234 / 0.9943 |
| n with mass ≥ 0.05 | 6/20 |
| top-1 accuracy (uncorrected) | 0.900 |

### D1 (first-token)

| metric | value |
| --- | --- |
| n | 20 |
| premask min / p50 / max | 0.0003 / 0.0234 / 0.9943 |
| n with mass ≥ 0.05 | 6/20 |
| top-1 accuracy (uncorrected) | 0.800 |

## 3. Scoring-position sensitivity (pinned vs post-think)

| item | pinned selected | post-think selected | pinned p_top | post-think p_top | agree |
| --- | --- | --- | --- | --- | --- |
| toy-001 | Canberra | Canberra | 0.964 | 0.919 | ✔ |
| toy-002 | Carbon dioxide | Oxygen | 0.902 | 0.732 | ✘ |
| toy-003 | Six | Six | 0.992 | 0.993 | ✔ |
| toy-004 | William Shakespeare | William Shakespeare | 0.953 | 0.889 | ✔ |
| toy-005 | Au | Au | 0.971 | 0.909 | ✔ |
| toy-006 | Mercury | Mercury | 0.925 | 0.757 | ✔ |
| toy-007 | 1945 | 1945 | 0.603 | 0.619 | ✔ |
| toy-008 | Pacific Ocean | Pacific Ocean | 0.819 | 0.900 | ✔ |
| toy-009 | Mandarin Chinese | Mandarin Chinese | 0.976 | 0.998 | ✔ |
| toy-010 | 100 degrees | 100 degrees | 0.749 | 0.775 | ✔ |

agreement: **9/10**

## 4. Baselines A / B / G (smoke cells)

| baseline | accuracy | unparsed | mean gen tokens | mean latency ms |
| --- | --- | --- | --- | --- |
| A (no think, max 256) | 19/20 | unparsed 0 | 7 | 248 |
| B (forced think, max 768) | 8/20 | unparsed 0 | 752 | 25713 |
| G (native default) | 7/20 | unparsed 0 | 512 | 17481 |

peak VRAM allocated: **9.62 GB**

