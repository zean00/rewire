"""Model loading + text baselines A / B / G (doc §16).

Class resolution is defensive on purpose: Qwen3.5-4B is multimodal, and the
correct transformers class can differ across versions. Whatever path loads is
recorded in `LoadedModel.load_path` and must be quoted in reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    load_path: str  # which Auto* class loaded the checkpoint
    dtype: str
    device: str
    attn: str = "?"  # resolved attention implementation ("sdpa", "eager", …)


def load_model(model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    archs = getattr(config, "architectures", []) or []
    load_path = "?"

    model = None
    # Multimodal checkpoints: image-text-to-text classes first, causal-LM fallback.
    for cls_name, auto_cls in (
        ("AutoModelForImageTextToText", _try_auto("AutoModelForImageTextToText")),
        ("AutoModelForVision2Seq", _try_auto("AutoModelForVision2Seq")),
        ("AutoModelForCausalLM", _try_auto("AutoModelForCausalLM")),
    ):
        if auto_cls is None:
            continue
        try:
            try:
                # sdpa: fused-attention kernels. Long-context requests are
                # the dominant cost in agentic use (the harness compacts
                # ~40k-token contexts mid-task); eager attention makes both
                # prefill and decode quadratic in context length, sdpa does
                # not. Fall back to the default impl if the class rejects it.
                model = auto_cls.from_pretrained(
                    model_id, dtype=dtype, device_map="cuda:0",
                    attn_implementation="sdpa")
            except TypeError:  # older transformers: torch_dtype kwarg
                model = auto_cls.from_pretrained(
                    model_id, torch_dtype=dtype, device_map="cuda:0",
                    attn_implementation="sdpa")
            load_path = cls_name
            break
        except Exception as e:  # noqa: BLE001 - record and try next loader
            last_err = e
            try:
                model = auto_cls.from_pretrained(model_id, dtype=dtype, device_map="cuda:0")
                load_path = cls_name
                break
            except Exception:
                pass
    if model is None:
        raise RuntimeError(f"No loader worked for {model_id} (archs={archs}): {last_err}")
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        load_path=load_path,
        dtype=str(dtype),
        device="cuda:0",
        attn=str(getattr(model.config, "_attn_implementation", "?")),
    )


def _try_auto(name: str):
    import transformers

    cls = getattr(transformers, name, None)
    return cls


# --- Generation baselines ---------------------------------------------------


@torch.no_grad()
def generate_answer(
    loaded: LoadedModel,
    messages: list[dict],
    max_new_tokens: int = 256,
    enable_thinking: bool | None = None,
) -> tuple[str, dict]:
    """Greedy generation. enable_thinking=None ⇒ native default (Baseline G).
    False ⇒ Baseline A. Baseline B (forced THINK) passes True."""
    from .template import apply_template

    text, kwargs = apply_template(loaded.tokenizer, messages, enable_thinking=enable_thinking)
    ids = loaded.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(loaded.model.device)
    out = loaded.model.generate(
        input_ids=ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=loaded.tokenizer.eos_token_id,
    )
    # generate() leaves the multimodal rope_deltas set even for text-only
    # prompts; a later cached (past_key_values) forward carrying an
    # attention_mask then takes the stale-delta branch in qwen3_5's position
    # derivation, which builds positions over the FULL mask length instead of
    # the new tokens and crashes apply_rotary_pos_emb (L2 repro: P=169, S=17,
    # mask 186). Clear it so subsequent cached scoring uses the model's own
    # position inference, as in every pre-generate forward.
    inner = getattr(loaded.model, "model", None)
    if inner is not None and getattr(inner, "rope_deltas", None) is not None:
        inner.rope_deltas = None
    produced = loaded.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    # Split think block off when present; record that it happened.
    meta = {"template_kwargs": kwargs, "has_think": "<think>" in produced}
    if "</think>" in produced:
        think, _, answer = produced.partition("</think>")
        meta["think_tokens_approx"] = len(loaded.tokenizer.encode(think, add_special_tokens=False))
        answer = answer
    else:
        answer = produced
    return answer.strip(), meta
