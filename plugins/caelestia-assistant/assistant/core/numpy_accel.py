"""core.numpy_accel — optional numpy accelerator with a pure-Python
fallback. THE CONTRACT: both paths return IDENTICAL results on the
same inputs (pinned by tests/test_numpy_accel.py). numpy is never
imported at module scope by callers unless they opt in via
have_numpy(); the pure path is always available.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = ["have_numpy", "cosine_topk", "mean_pool"]

_np = None
_checked = False


def have_numpy() -> bool:
    """Feature-detect numpy once; safe to call repeatedly."""
    global _np, _checked
    if not _checked:
        try:
            import numpy as _candidate  # optional extra, never required
            _np = _candidate
        except ImportError:
            _np = None
        _checked = True
    return _np is not None


def cosine_topk(
    query: Sequence[float],
    matrix: Sequence[Sequence[float]],
    k: int,
) -> List[Tuple[int, float]]:
    """Top-k (index, cosine) pairs, descending. Identical results on
    both paths: same tie-breaking (stable by index), same float values
    for the pure path; the numpy path matches to 1e-9 and the test
    pins exact equality of the returned ORDER and (rounded) scores."""
    n = len(matrix)
    if n == 0:
        return []
    qnorm = math.sqrt(sum(x * x for x in query)) or 1.0
    if have_numpy() and k < n:
        q = _np.asarray(query, dtype=_np.float64)
        m = _np.asarray(matrix, dtype=_np.float64)
        qn = _np.linalg.norm(q) or 1.0
        mn = _np.linalg.norm(m, axis=1)
        mn[mn == 0] = 1.0
        sims = (m @ q) / (mn * qn)
        k_eff = min(k, n)
        idx = _np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[_np.argsort(-sims[idx], kind="stable")]
        return [(int(i), float(sims[i])) for i in idx]
    scored = []
    for i, row in enumerate(matrix):
        rnorm = math.sqrt(sum(x * x for x in row)) or 1.0
        num = sum(a * b for a, b in zip(query, row))
        scored.append((i, num / (rnorm * qnorm)))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def mean_pool(rows: Sequence[Sequence[float]],
              weights: Optional[Sequence[float]] = None) -> List[float]:
    """Weighted mean pooling. Identical semantics on both paths."""
    if not rows:
        return []
    dim = len(rows[0])
    if weights is None:
        weights = [1.0] * len(rows)
    total = sum(weights) or 1.0
    if have_numpy():
        m = _np.asarray(rows, dtype=_np.float64)
        w = _np.asarray(weights, dtype=_np.float64)
        return [float(x) for x in (m * w[:, None]).sum(axis=0) / total]
    out = [0.0] * dim
    for row, w in zip(rows, weights):
        for j, x in enumerate(row):
            out[j] += x * w
    return [x / total for x in out]
