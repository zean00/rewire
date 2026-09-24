# Entry H instrument notes — CLM-class contrastive readout

Entry H itself is **halted** (see ledger: the frozen 316-probe corpus was deleted from the
GPU host; recipe verification impossible). This directory pins the instrument so any
successor entry starts from verified facts instead of recon. No probe data was ever
embedded — the smoke test below used throwaway sentences only.

## Pins (recorded 2026-09-24, host clock)

| artifact | pin |
|---|---|
| CLM code repo | github.com/Contrastive-LM/CLM @ `cca045ffdb07b3ebcfe6938537cdeac5e14899c9` (host: `/home/sahal/clm_instrument/CLM`) |
| heads checkpoint | `Contrastive-LM/CLM-v0.1-8B` / `CLM_v0.1-8B.pt`, 73 MB, sha256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |
| checkpoint HF revision | `87655cb835bd76fd66c2da78e1e3709f7fa11a94` (lastModified 2026-09-24T02:03:58Z) |
| encoder | `Qwen/Qwen3-8B`, HF snapshot `b968826d9c46dd6066d109eabc6255188de91218` (host HF cache, ~16 GB) |

**Important:** the 73 MB checkpoint is the two projection heads ONLY; the encoder is the
stock frozen Qwen3-8B the operator serves (`base_model: Qwen/Qwen3-8B`, Apache-2.0).

## Canonical recipe (from their code, pinned)

- Heads: two independent MLPs, each 4096 → 1536 → [LayerNorm → 1536] → 512, GELU,
  LayerNorm inside the hidden block, no residual (cfg in checkpoint matches their
  `make_head`). `logit_scale = exp(4.6132) = 100.811`.
- Score of a (state, candidate) pair = `100.811 · cos(state_head(s), action_head(c))`;
  answer probabilities = softmax over scores (temperature default 1.0).
- `clm-raw` mode = cosine in the encoder's own space, no heads (their ablation model).
- State text layout (the layout the heads are trained on): `context + "\n\n" + instructions`
  — question last, never repeated inside the context; candidates embedded verbatim.
- Serving (their `serve_qwen3_8b.sh` + `Embedder`): vLLM `--runner pooling`, LAST-token
  pooling, `truncate_prompt_tokens = 2048` (left-truncate, keep the tail), client-side
  L2 normalization, exact-string LRU cache.

## Smoke test (mechanics only — `clm_smoke.py`, `smoke_result.json`)

Passes end-to-end on the host: heads load per cfg, encoder last-token pooling (HF
forward, bf16, `device_map="auto"`, left-truncation), projections and scoring run,
**bit-exact deterministic across passes**, ~0.5 s/text. VRAM: **15.9 GB used — at the
card's absolute ceiling** (RTX 5070 Ti 15.9 GB; bf16 weights 16.4 GB do not fit, a few
tensors offloaded to CPU). The printed toy cosines are plumbing checks, not signal
claims.

## Constraints and open questions any successor (H′) registration must pin

1. **Encoder precision is a deviation from the halted registration**: fp16 impossible on
   this card. Pre-commit either int8 (bitsandbytes, needs install) or bf16-with-offload
   (works today, at the ceiling) before any embedding. vLLM is NOT installed in
   hybrid-env; if their exact vLLM pooling path is wanted, that is an install decision to
   make at registration — HF-forward vs vLLM pooling parity is otherwise untested.
2. **Probability sharpness**: softmax over `100.8·cos` saturates — a cosine gap of 0.01 is
   a logit gap of ~1 (odds ≈ e² ≈ 7.4). Their "probabilities" are ranking-sharp, likely
   far from calibrated; the frozen bar's Brier leg is exactly the test for this.
3. **Train/serve layout**: the public PRM dataset's states were chat-templated upstream
   (not in their repo); inference embeds raw `context\n\ninstructions`. Whether training
   states carried the question suffix is unverifiable from the public artifacts — pin the
   chosen convention at registration.
4. **Corpus prerequisites for a fresh-data successor**: MiniWoB page server (local :8077)
   still up; `llama-server` binary and Playwright chromium present on the host; BUT the
   omp harness is absent from the host PATH (reinstall needed) and the 24-task GOALS list
   died with `/tmp/mw_bench` — both must be re-created as part of the registration.
