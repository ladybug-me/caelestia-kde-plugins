"""Seeded statistics for the eval arena (stdlib only, deterministic).

Every interval reported by the arena comes through here so the numbers
share one definition set:

- ``bootstrap_ci`` — nonparametric percentile bootstrap, seeded, >=1000
  resamples by default. Deterministic for a fixed seed.
- ``proportion_ci`` — convenience wrapper for 0/1 outcome vectors.
- ``ece`` / ``brier`` — calibration of the router's top-1 confidence
  against binary correctness outcomes.
- ``benjamini_hochberg`` — FDR-adjusted p-values for variant comparisons
  (used by the arena's paired comparisons, not by single-suite runs).

No numpy, no scipy: lists of floats in, floats out.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
    n_boot: int = 1000,
    seed: int = 120,
    level: float = 0.95,
) -> Tuple[float, float, float]:
    """Percentile bootstrap (statistic, low, high) around ``statistic(data)``.

    Deterministic: ``random.Random(seed)`` drives every resample. For a
    mean statistic this is the arena's default interval; for anything
    else pass the statistic explicitly.
    """
    if not data:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(data)
    stats: List[float] = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    alpha = (1.0 - level) / 2.0
    lo = stats[min(n_boot - 1, max(0, int(math.floor(alpha * n_boot))))]
    hi = stats[min(n_boot - 1, max(0, int(math.ceil((1.0 - alpha) * n_boot)) - 1))]
    return (statistic(list(data)), lo, hi)


def proportion_ci(outcomes: Sequence[int], **kw) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of a 0/1 vector (a proportion)."""
    if not outcomes:
        return (float("nan"), float("nan"), float("nan"))
    return bootstrap_ci([float(x) for x in outcomes], **kw)


def paired_bootstrap_delta(
    a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 121
) -> Tuple[float, float, float]:
    """Bootstrap CI of mean(a) - mean(b) over PAIRED items (same length).

    "Improved" requires the interval to exclude zero. Items must be
    aligned; callers guarantee that.
    """
    assert len(a) == len(b), "paired bootstrap requires aligned vectors"
    if not a:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(a)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(
            sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n
        )
    deltas.sort()
    point = sum(a) / len(a) - sum(b) / len(b)
    alpha = 0.025
    lo = deltas[int(math.floor(alpha * n_boot))]
    hi = deltas[min(n_boot - 1, int(math.ceil((1 - alpha) * n_boot)) - 1)]
    return (point, lo, hi)


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilistic predictions (0..1; lower=better)."""
    if not probs:
        return float("nan")
    return sum((p - float(o)) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def ece(probs: Sequence[float], outcomes: Sequence[int], bins: int = 5) -> float:
    """Expected calibration error: |accuracy - confidence| mass-weighted.

    Equal-width bins over [0,1]; empty bins contribute nothing.
    """
    if not probs:
        return float("nan")
    total = len(probs)
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(p, o) for p, o in zip(probs, outcomes)
                   if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not members:
            continue
        n_b = len(members)
        conf = sum(p for p, _ in members) / n_b
        frac = sum(float(o) for _, o in members) / n_b
        acc += (n_b / total) * abs(frac - conf)
    return acc


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> List[bool]:
    """Return, per hypothesis, whether it survives BH at FDR level ``q``."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    # standard step-up: reject all p <= largest p_i meeting p_i <= q*i/n
    cutoff = 0.0
    for rank_pos, idx in enumerate(order):
        i = rank_pos + 1
        if p_values[idx] <= q * i / n:
            cutoff = p_values[idx]
    return [p <= cutoff for p in p_values]
