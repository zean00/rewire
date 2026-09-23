"""Pre-mask candidate probability mass (doc §9.1 / §16.2 rev. 3).

For every scored decision we record how much raw (unmasked) probability the
candidate set captures at the read position. Below a validity threshold, the
masked softmax is confidence over options the model considers implausible —
flag or discard, never average in.

For single-token candidates (D1) this is exact. For multi-token candidates
(D2) we use the sum of first-token masses — a lower-bound-ish approximation,
documented here rather than hidden.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def premask_mass(
    logits_row: torch.Tensor,  # [vocab] raw logits at the read position
    candidate_first_token_ids: list[int],
) -> float:
    """Sum of first-token masses, de-duplicated: candidates that share a
    first token (e.g. numeric choices "340"/"345") would otherwise add the
    same token's mass twice and exceed 1 (Phase 1 finding, arith domains)."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    idx = torch.tensor(sorted(set(candidate_first_token_ids)), device=probs.device, dtype=torch.long)
    return float(probs[idx].sum().item())


PREMASK_VALID_THRESHOLD = 0.05  # kill-criterion 5 preview value; tune in Phase 0


def is_valid(mass: float, threshold: float = PREMASK_VALID_THRESHOLD) -> bool:
    return mass >= threshold
