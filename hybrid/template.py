"""Scoring-position pinning (doc §9.1 rev. 3, Phase 0).

Qwen3.5 thinks by default; every logits-based measurement depends on WHERE in
the chat template logits are read. This module is the single source of truth
for the read point. Nothing in the codebase may build decision prompts any
other way.

Pinned read point (default): chat template with `enable_thinking=False`,
generation prompt included, logits read at the LAST prompt position.

Think-on variant (sensitivity checks only): generation with the native
thinking template until `</think>`, then the produced think block is frozen
into the prompt and candidates are scored at the position after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PinnedPrompt:
    input_ids: torch.Tensor  # [1, seq_len]
    text: str
    read_mode: str  # "pinned_nothink" | "postthink"
    template_kwargs: dict  # exactly what was passed to apply_chat_template

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


def apply_template(tokenizer, messages: list[dict], enable_thinking: bool | None = None) -> tuple[str, dict]:
    """Apply the chat template, recording exactly which kwargs were used.

    Tries `enable_thinking=...`; if the tokenizer's template does not accept
    it, falls back to the bare template and records that — the fallback is a
    finding (the model routes thinking differently) not a silent default.
    """
    kwargs: dict = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        try:
            text = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=enable_thinking)
            kwargs["enable_thinking"] = enable_thinking
            return text, kwargs
        except TypeError:
            text = tokenizer.apply_chat_template(messages, **kwargs)
            kwargs["enable_thinking"] = f"UNSUPPORTED (requested {enable_thinking})"
            return text, kwargs
    text = tokenizer.apply_chat_template(messages, **kwargs)
    return text, kwargs


def pin_prompt(tokenizer, messages: list[dict], enable_thinking: bool = False) -> PinnedPrompt:
    """The pinned no-think read point used by all decision scoring."""
    text, kwargs = apply_template(tokenizer, messages, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return PinnedPrompt(
        input_ids=ids.to(tokenizer.device) if hasattr(tokenizer, "device") else ids,
        text=text,
        read_mode="pinned_nothink",
        template_kwargs=kwargs,
    )


@torch.no_grad()
def pin_after_think(
    tokenizer,
    model,
    messages: list[dict],
    max_think_tokens: int = 512,
) -> PinnedPrompt:
    """Generate the native think block, freeze it into the prompt, return the
    post-</think> read point. Sensitivity-check path only — never the default.
    """
    text, kwargs = apply_template(tokenizer, messages, enable_thinking=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    # Generate until the think block closes (plus the newline after it).
    stop_id = tokenizer.encode("</think>", add_special_tokens=False)
    gen = model.generate(
        input_ids=ids,
        max_new_tokens=max_think_tokens,
        do_sample=False,
        eos_token_id=stop_id if stop_id else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    produced = gen[0][ids.shape[1]:]
    think_text = tokenizer.decode(produced, skip_special_tokens=True)
    if "</think>" not in think_text:
        think_text = think_text + "</think>"  # budget exhausted; close it explicitly
    think_text = think_text.split("</think>")[0] + "</think>\n\n"

    combined = text + think_text
    combined_ids = tokenizer(combined, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    return PinnedPrompt(
        input_ids=combined_ids,
        text=combined,
        read_mode="postthink",
        template_kwargs={**kwargs, "max_think_tokens": max_think_tokens},
    )


DECISION_INSTRUCTION = "Answer with exactly one of the listed options."
DECISION_ANCHOR = "Reply with the option text only, nothing else."


def question_messages(question: str, candidates: list[str], anchor: bool = False) -> list[dict]:
    """Standard decision prompt. Fixed wording — template sensitivity is a
    measured quantity (§16.2), not an accident. `anchor=True` appends the
    format-anchor line (Phase 1 experiment: pulls pre-mask mass up by
    constraining the implied answer format)."""
    choice_lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    content = f"{question}\n\nOptions:\n{choice_lines}\n\n{DECISION_INSTRUCTION}"
    if anchor:
        content += f" {DECISION_ANCHOR}"
    return [{"role": "user", "content": content}]


def plain_question(question: str) -> list[dict]:
    """Baseline A prompt: no options shown, brief answer requested."""
    return [{"role": "user", "content": f"{question}\n\nAnswer briefly."}]
