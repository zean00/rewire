"""CUDA-event latency telemetry (doc §17).

All timings are wall-clock around explicit synchronization points so numbers
are comparable across HF and vLLM paths. Reports use P50/P95 over N reps —
single-shot timings on WSL2 are noisy.
"""

from __future__ import annotations

import json
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Timer:
    """Collects named section timings (ms)."""

    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def section(self, name: str):
        if not torch.cuda.is_available():
            import time

            t0 = time.perf_counter()
            yield
            self.samples.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.samples.setdefault(name, []).append(start.elapsed_time(end))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, xs in self.samples.items():
            xs_sorted = sorted(xs)
            out[name] = {
                "n": len(xs),
                "p50_ms": xs_sorted[len(xs_sorted) // 2],
                "p95_ms": xs_sorted[max(0, int(len(xs_sorted) * 0.95) - 1)],
                "mean_ms": statistics.fmean(xs),
            }
        return out

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


def total_ms(timer: Timer) -> float:
    return sum(v["p50_ms"] for v in timer.summary().values())
