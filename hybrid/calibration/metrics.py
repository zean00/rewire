"""Calibration metrics (doc §11/§17)."""

from __future__ import annotations

import torch


def expected_calibration_error(
    confidences: list[float], corrects: list[bool], n_bins: int = 10
) -> dict:
    """Standard 10-bin ECE plus per-bin table.

    confidences: model's top-1 probability per decision.
    corrects:    whether that decision was right.
    """
    assert len(confidences) == len(corrects)
    if not confidences:
        return {"ece": 0.0, "n": 0, "bins": []}

    conf = torch.tensor(confidences, dtype=torch.float64)
    corr = torch.tensor([float(c) for c in corrects], dtype=torch.float64)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    idx = torch.clamp((conf * n_bins).long(), max=n_bins - 1)  # right-edge items into last bin

    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0})
            continue
        acc = float(corr[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (n / len(conf)) * gap
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "accuracy": acc, "avg_confidence": avg_conf, "gap": gap})
    return {"ece": float(ece), "n": len(conf), "bins": bins}
