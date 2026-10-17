"""cortex.abers — honest probability calibration (Group A3).

The router's softmax p is a RANKING quantity, not a probability: the
arena's calibration suite has said so in cold-start ECE terms since it
existed. This module adds the two classical fixes, stdlib-only:

- **Isotonic regression** (PAVA — pool-adjacent-violators, the standard
  binomial-calibration staircase): a monotone step function fitted on
  (score, outcome) pairs, optionally WEIGHTED. Deterministic; ties
  break by insertion order.

- **Venn-Abers predictive sets** (Vovk & Petej 2014; the IVAP form):
  an exact distribution-free calibration INTERVAL. For a test score p,
  run PAVA twice on the calibration pairs — once with (p, 1) appended,
  once with (p, 0) appended — and read the two predictions at p: the
  interval [p_lo, p_hi] is guaranteed to cover the true outcome
  probability under exchangeability. The midpoint is the point
  prediction; the WIDTH is honest abstention information (a wide
  interval means the calibrator does not know).

- **Weighted conformal**: the split-conformal quantile, re-derived with
  importance WEIGHTS on the calibration scores — the honest tool when
  the calibration stream (dev-shaped) may not match deployment
  (real-shaped): weight the calibration rows by the dev-to-real
  importance ratio and the quantile tracks the reweighted
  distribution. Deterministic weighted quantile, bounded data.

Reporting: ``eval calibration`` now reports ECE for the RAW softmax and
for isotonic / Venn-Abers after them, using deterministic every-4th-row
held-out folds (the corpus's own supervised-split convention) so the
numbers are honest out-of-sample, not in-sample self-grading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["isotonic_fit", "isotonic_predict", "venn_abers",
           "weighted_conformal_quantile", "ece_report"]

Infinity = float("inf")


# ---------------------------------------------------------------------------
# PAVA (weighted).
# ---------------------------------------------------------------------------


def isotonic_fit(pairs: Sequence[Tuple[float, float]],
                 weights: Optional[Sequence[float]] = None
                 ) -> List[Tuple[float, float]]:
    """Weighted PAVA over (score, outcome) pairs sorted by score.

    Returns the fitted step function as [(x, y), ...] with x ascending
    and y the pooled (isotonic) means — a monotone non-decreasing map.
    Deterministic: stable sort by score then insertion order.
    """
    if weights is not None and len(weights) != len(pairs):
        raise ValueError("weights must match pairs")
    data = sorted(
        ((float(p), float(o), float(weights[i]) if weights is not None
          else 1.0) for i, (p, o) in enumerate(pairs)),
        key=lambda t: (t[0],))
    # blocks: (sum_w, sum_wy, mean, [xs...])
    blocks: List[List[Any]] = []
    for x, y, w in data:
        if w <= 0:
            continue
        blocks.append([w, w * y, y / w, [x]])
        # pool while the previous mean exceeds the current (violation)
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2] + 1e-15:
            w2, wy2, _m2, xs2 = blocks.pop()
            w1, wy1, _m1, xs1 = blocks.pop()
            blocks.append([w1 + w2, wy1 + wy2, (wy1 + wy2) / (w1 + w2),
                           xs1 + xs2])
    steps: List[Tuple[float, float]] = []
    for _w, _wy, mean, xs in blocks:
        for x in xs:  # one (x, y) per calibration point: a step map
            steps.append((x, mean))
    return steps


def isotonic_predict(fit_steps: Sequence[Tuple[float, float]],
                     x: float) -> float:
    """Evaluate the PAVA step map at x (right-continuous; below the
    first calibration point it returns the first pooled mean — the
    honest flat extrapolation)."""
    if not fit_steps:
        return 0.5
    lo, hi = 0, len(fit_steps) - 1
    best = fit_steps[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if fit_steps[mid][0] <= x:
            best = fit_steps[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------------
# Venn-Abers (IVAP form).
# ---------------------------------------------------------------------------


def venn_abers(calib: Sequence[Tuple[float, float]],
               test_p: float) -> Tuple[float, float, float]:
    """(p_lo, p_hi, midpoint) for one test score.

    p_hi: PAVA over calib + [(test_p, 1)], evaluated at test_p.
    p_lo: PAVA over calib + [(test_p, 0)], evaluated at test_p.
    The interval covers the true conditional probability under
    exchangeability (Vovk & Petej 2014); the midpoint is the point
    prediction used for scoring.
    """
    hi = isotonic_predict(isotonic_fit(list(calib) + [(test_p, 1.0)]),
                          test_p)
    lo = isotonic_predict(isotonic_fit(list(calib) + [(test_p, 0.0)]),
                          test_p)
    lo = min(lo, hi)  # numerical ordering guard
    return lo, hi, (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Weighted conformal quantile.
# ---------------------------------------------------------------------------


def weighted_conformal_quantile(scores: Sequence[float],
                                weights: Optional[Sequence[float]] = None,
                                alpha: float = 0.1) -> Optional[float]:
    """The split-conformal quantile under importance WEIGHTS.

    Unweighted (weights=None) this is exactly the existing
    ConformalCalibrator.threshold rule: ceil((n+1)(1-alpha))/n order
    statistic. Weighted, the (n+1)(1-alpha) mass quantile of the
    reweighted distribution — the honest shift tool: weight dev-shaped
    calibration rows by their dev-to-real importance ratio and the
    threshold tracks the real distribution, not the dev one. Returns
    None when there is too little data (honesty: an empty calibration
    set guarantees nothing).
    """
    n = len(scores)
    if n < 5:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha in (0, 1)")
    if weights is None:
        s = sorted(scores)
        import math
        rank = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha))))
        return s[rank - 1]
    if len(weights) != n:
        raise ValueError("weights must match scores")
    pairs = sorted(zip((float(s) for s in scores),
                       (float(w) for w in weights)),
                   key=lambda t: t[0])
    total = sum(w for _s, w in pairs)
    if total <= 0:
        return None
    # (n+1)(1-alpha)/n of the TOTAL mass, matching the unweighted rule's
    # finite-sample correction in the equal-weights limit
    import math
    # (n+1)/n finite-sample correction matches the unweighted rule in
    # the equal-weights limit
    target = min(1.0, ((n + 1) / n) * (1.0 - alpha))
    cum = 0.0
    for s, w in pairs:
        cum += w / total
        if cum >= target - 1e-12:
            return s
    return pairs[-1][0]


# ---------------------------------------------------------------------------
# ECE reporting (out-of-sample, deterministic folds).
# ---------------------------------------------------------------------------


def ece_report(rows: Sequence[Tuple[float, int]],
               folds: int = 4, bins: int = 5) -> Dict[str, Any]:
    """ECE before/after calibration over deterministic held-out folds.

    Fold i (every folds-th row, the corpus's own every-4th supervised
    convention) is scored by calibrators fitted on the OTHER rows, so
    the after numbers are out-of-sample. Raw ECE is fold-free (the
    uncalibrated baseline has nothing to fit).
    """
    from assistant.core.eval.stats import ece

    raw = ece([p for p, _o in rows], [o for _p, o in rows], bins=bins)
    iso_vals: List[Tuple[float, int]] = []
    va_vals: List[Tuple[float, int]] = []
    wide = 0
    for i, (p, o) in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j % folds != i % folds]
        if not train:
            continue
        steps = isotonic_fit(train)
        iso_vals.append((isotonic_predict(steps, p), o))
        lo, hi, mid = venn_abers(train, p)
        va_vals.append((mid, o))
        if hi - lo > 0.34:
            wide += 1
    return {
        "n": len(rows),
        "ece_raw": round(raw, 4),
        "ece_isotonic": round(ece([p for p, _ in iso_vals],
                                  [o for _, o in iso_vals], bins=bins), 4),
        "ece_venn_abers": round(ece([p for p, _ in va_vals],
                                    [o for _, o in va_vals], bins=bins), 4),
        "wide_va_intervals": wide,
        "folds": folds,
        "note": ("out-of-sample: fold i scored by calibrators fitted on "
                 "the other folds (deterministic, no RNG)"),
    }
