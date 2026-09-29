# Mirai S probe (2026-09-29) — Qwen3.8-27B-S 2.4-bit GGUF as program worker

**Mandate** (user, 2026-09-29): probe `trymirai/Qwen3.8-27B-S-experimental` via the
GGUF `alesha-pro/Qwen3.8-27B-S-mirai-GGUF` on the host 16 GB GPU; if the initial
probe does not look promising, the 168-session test may be dropped. This is a
PROBE, not a registered entry: no pools, no bars, no adjudication — one report.

## Setup

- Fork `alesha-pro/llama.cpp-mirai-s` (base `ggml-org/llama.cpp` `d834d44e6`,
  built head `b59ae80`, CUDA 13.3, `CMAKE_CUDA_ARCHITECTURES=native`), cloned and
  built on the host (`/home/sahal/llama.cpp-mirai-s`). ~6 min build.
- GGUF `Qwen3.8-27B-S-mirai.gguf`, 11,173,346,688 bytes, at
  `/home/sahal/models-mirai/`. No mmproj (text-only probe).
- Serve line (as run, MTP removed after the logprob checks — see below):

```
llama-server -m models-mirai/Qwen3.8-27B-S-mirai.gguf --host 100.120.167.9 \
  --port 8995 -ngl 99 -fa on -np 1 --jinja -c 65536 -ctk q4_0 -ctv q4_0 \
  -b 1024 -ub 1024
```

Load < 5 s. VRAM 11,916 MiB of 16,303 (4.4 GB headroom) alongside the standing
4 B proxy process (which holds no VRAM). The combined gemma server (:8997) was
stopped for the probe, per the recorded relaunch line; restored at the end
(bit-exact embeddings recheck below).

## Probe battery (laptop → `100.120.167.9:8995`, OpenAI wire)

| Probe | Result |
|---|---|
| P1 coherence 17×23 | "391", thinking in `reasoning_content` (separate channel), 74 tok/s |
| P2 harness turn + native tools | proper `eval` tool call, well-formed JSON args, intent comment `// click "next" @e6` inside `code` — exactly the brake rule's wire format; 101 tok/s decode, 768 tok/s prompt |
| P2b same turn, no tools | native `<tool_call>` XML in content (parseable, but the tools-array path is the one the proxy would use) |
| P3 completion-claim turn | correct claim with snapshot evidence, right vocabulary |
| Long-context (~4.2 k-token, 8-turn) | stable, 1,208 tok/s prompt, 85 tok/s decode, coherent |
| Grammar-forced logprobs | REAL teacher-forced logprobs: `Paris` −1.60 / `Banana` −35.70 / `Tokyo` −24.32 — the decide chain's d2 reads work on this fork |
| `enable_thinking:false` + `reasoning_budget_tokens:0` | both honored (chain's two knobs verified live) |

A first hypothesis "fork logprobs broken" was wrong: the giveaway numbers were
an echo task (model copies the prompt; p≈1 per token is the TRUE value). Forced
reads differentiate cleanly. MTP speculative decoding was dropped from the serve
line during this investigation and not re-enabled (probe ran plain greedy; decode
speed stayed ≥ 85 tok/s at probe sizes).

## Disclosed incident — first sanity battery invalid

The first live sanity battery (4 sessions, `sanity_runs_mirai/…` seeds
4755240311, 1916988017-class, pre-18:29) ran against a WRONG proxy config: I
wrote `base_url http://127.0.0.1:8995/v1`, but the mirai server binds only the
Tailscale IP (`--host 100.120.167.9`), so every chain model call was
connection-refused and the fail-open paths served deterministic chain cells
with no model involvement (empty-name clicks, empty answer bursts). Those
sessions measured nothing about the model. Fix was one config line
(`base_url → http://100.120.167.9:8995/v1`) + proxy restart. The invalid
sessions are retained on disk, distinguishable by seed/timestamp; the second
battery below is the record.

## Second battery — real harness, full chain + brake, no dial

`sanity_mirai.py sanity <task>` (derived from run_v8.py by constant patch):
laptop omp agent → host proxy :8991 (decide chain + mechanical brake, dial
absent) → mirai :8995. Four tasks, fresh seeds:

| task | outcome | wall | notes |
|---|---|---|---|
| click-button | **raw = 1 WIN** | 34.8 s | clean, exit 0 — fastest clean win of any worker this program has served |
| login-user | raw = −1, cap 480 s | 480.0 s | read the task, self-corrected from "admin"/"password" to the real "dolores"/"GG3K", filled the form — then 30+ turns unable to make Login register |
| enter-text | raw = −1, cap 480 s | 480.0 s | typed "1" and submitted WITHOUT reading the instruction, then wandered 33 turns |
| use-autocomplete | raw = −1 | 144.3 s | clean wrong answer |

Behavior profile (consistent across the three failures):

1. **Impulsive first move** — acts before reading the page's instruction area
   (typed a guess; clicked Login before filling anything).
2. **Off-contract flailing when blocked** — reaches for APIs the harness never
   offered (`tab.run`, `tab.id(10)`, puppeteer handles) instead of retrying the
   prescribed `tab.evaluate` recipe; those calls error/timeout and burn wall.
3. **No loop discipline** — 30+ turn click-variant loops; the frozen brake rule
   correctly did NOT fire (sigs varied in wording; many turns carried no intent
   comment at all, so they are not acting turns under the rule).

The decide chain itself worked on mirai: named intent-comment cells, correct
element reads (a replayed login-page read: Username −0.185 vs Password −5.96 vs
Log in −4.08), and the model demonstrated it CAN read the instruction (the
credential self-correction).

## Verdict

**Not promising as a program worker; the 168-session test is dropped** per the
probe mandate. Serving, quantization, speed, wire format, and the chain's read
mechanics all pass — the failure is worker judgment: on the harness contract,
the 2.4-bit 27B is impulsive, contract-forgetting under blockage, and
wall-hungry (two 480 s cap-burners in four sessions), with no evidence it would
approach the incumbent gemma-12b arms (16.1–17.3 % on the frozen 168) at sweep
scale, let alone justify ~a day of GPU. This is also the scaffolding thesis
from the other side: a bigger, differently-raised model does not inherit the
contract discipline the chain was tuned around — the contract is a frozen
per-worker artifact, and this worker was not raised on it.

What would change the verdict: a mirai-raised contract prompt (whitelisted
primitives, read-the-instruction-first step), thinking disabled for acting
turns, or the full (non-experimental) Mirai release. None of these is a probe
fix; each is a new program phase.

## Restoration

- Mirai server (:8995) and mirai proxy (:8991) stopped after the probe.
- Combined gemma server relaunched on :8997 with the exact registered line
  (entry-N correction); embeddings rechecked against the entry-P embed
  manifest — result recorded below.
- Relaunch recipe if mirai is ever revisited: the serve line above plus
  `cd /home/sahal/hybrid-qwen && nohup /home/sahal/hybrid-env/bin/python
  eval/omp_proxy.py --port 8991 --backend-config eval/proxy_chain_remote_mirai.json
  --no-local-model >> eval/reports/omp_arms/proxy_mirai.log 2>&1 < /dev/null &`
  and the laptop `~/.omp/agent/models.yml` `poc-mirai` block (retained).
