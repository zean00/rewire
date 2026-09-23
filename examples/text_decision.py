"""Example: one pinned DECIDE call (D1 + D2) on the toy set item 0.

Run on the GPU host: python examples/text_decision.py
"""

import json
from pathlib import Path

from hybrid.engine import load_model
from hybrid.scoring import score_first_token, score_sequence_logprob
from hybrid.template import pin_prompt, question_messages

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"


def main() -> None:
    item = json.loads((ROOT / "eval/datasets/toy_mcqa.jsonl").read_text().splitlines()[0])
    candidates = item["choices"]

    loaded = load_model(MODEL_ID)
    pinned = pin_prompt(loaded.tokenizer, question_messages(item["question"], candidates))

    r1, tok_map = score_first_token(loaded.model, loaded.tokenizer, pinned, candidates)
    print(f"Q: {item['question']}")
    print(f"D1: selected={r1.selected!r} confidence={r1.confidence:.3f} premask_mass={r1.premask_mass:.4f}")
    print(f"    distribution: { {k: round(v, 3) for k, v in r1.probabilities.items()} }")
    print(f"    tokenization: {tok_map}")

    r2 = score_sequence_logprob(loaded.model, loaded.tokenizer, pinned, candidates, alpha=1.0)
    print(f"D2: selected={r2.selected!r} confidence={r2.confidence:.3f} premask_mass={r2.premask_mass:.4f}")
    print(f"    raw log-likelihoods: { {k: round(v, 2) for k, v in r2.raw_scores.items()} }")

    gold = item["choices"][item["answer_idx"]]
    print(f"gold: {gold!r} | D1 correct={r1.selected == gold} D2 correct={r2.selected == gold}")


if __name__ == "__main__":
    main()
