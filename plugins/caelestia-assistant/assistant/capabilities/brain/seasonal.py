"""brain.seasonal — recurring patterns in the shell's own numbers (C11).

The telemetry the assistant already reads (battery samples, log volume,
notification counts, settings applies) is a TIME SERIES, and the
questions users ask about it are the classical four:

  "does this happen every night?"       -> Holt-Winters seasonality
  "when did it change?"                 -> PELT changepoints
  "what does this remind me of?"        -> SAX motifs (matrix profile)
  "when was it most unusual?"           -> SAX discords (matrix profile)

Everything here is deterministic, bounded (series past the caps get an
honest ABSTAIN on that one metric, not a wrong number), and pure:
no RNG, no clock, no network. The normal inverse CDF comes from
statistics.NormalDist — stdlib, not a formula from a forum.

This module READS numbers the caller passes and returns plain dicts.
It never sees the ledger, never writes, and proposes nothing on its
own — the brief and the rules engine decide what (if anything) to do
with a pattern.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["matrix_profile", "sax_encode", "motifs_and_discords",
           "pelt_changepoints", "holt_winters", "seasonality_report"]

_MP_CAP = 2000        # matrix profile is O(n^2): past this, abstain
_PELT_CAP = 5000      # the DP is O(n) with pruning, cap for safety


def _validate(series: List[float], minimum: int = 4) -> List[float]:
    out = [float(x) for x in series]
    if len(out) < minimum:
        raise ValueError(f"need at least {minimum} points, got {len(out)}")
    return out


# ---------------------------------------------------------------------------
# matrix profile (brute force, window-normalized euclidean distance)
# ---------------------------------------------------------------------------

def _znorm(seg: List[float], eps: float = 1e-9) -> List[float]:
    mean = sum(seg) / len(seg)
    var = sum((x - mean) ** 2 for x in seg) / len(seg)
    std = math.sqrt(var)
    if std < eps:
        return [0.0] * len(seg)
    return [(x - mean) / std for x in seg]


def matrix_profile(series: List[float], window: int) \
        -> Optional[List[float]]:
    """The STUMP-style matrix profile, brute force: for every
    subsequence, the z-normalized euclidean distance to its NEAREST
    OTHER subsequence (self-match excluded, exclusion zone = window/2).
    None (ABSTAIN) when n > _MP_CAP."""
    series = _validate(series, window + 2)
    n = len(series)
    if n > _MP_CAP:
        return None
    if window < 3 or window > n // 2:
        raise ValueError(f"window must be in [3, n//2], got {window}")
    subs = [_znorm(series[i:i + window]) for i in range(n - window + 1)]
    count = len(subs)
    exclude = window // 2
    profile: List[float] = []
    for i in range(count):
        best = math.inf
        si = subs[i]
        for j in range(count):
            if abs(i - j) <= exclude:
                continue
            sj = subs[j]
            d = sum((a - b) ** 2 for a, b in zip(si, sj))
            if d < best:
                best = d
        profile.append(math.sqrt(best) if math.isfinite(best) else
                       math.sqrt(sum(x * x for x in si)))
    return profile


def sax_encode(series: List[float], word_size: int = 8,
               alphabet: int = 4) -> str:
    """SAX: z-normalize, PAA down to word_size frames, quantize on the
    equal-probability breakpoints of N(0,1) (alphabet 3..10)."""
    series = _validate(series, word_size)
    z = _znorm(series)
    n = len(z)
    frame = n / word_size
    from statistics import NormalDist
    nd = NormalDist()
    cuts = [nd.inv_cdf(i / alphabet) for i in range(1, alphabet)]
    word = []
    for f in range(word_size):
        lo, hi = int(f * frame), max(int((f + 1) * frame), int(f * frame) + 1)
        seg = z[lo:hi]
        mean = sum(seg) / len(seg)
        level = sum(1 for c in cuts if mean > c)
        word.append(chr(ord('a') + level))
    return "".join(word)


def motifs_and_discords(series: List[float], window: int,
                        top: int = 3) -> Dict[str, Any]:
    """Top-`top` motif pairs (smallest profile values = most similar
    non-overlapping neighbors) and discords (largest = most unusual)."""
    profile = matrix_profile(series, window)
    if profile is None:
        return {"verdict": "ABSTAIN",
                "note": f"series longer than {_MP_CAP}: the O(n^2) "
                        f"matrix profile abstains"}
    order = sorted(range(len(profile)),
                   key=lambda i: (profile[i], i))
    motifs = [{"index": i, "distance": round(profile[i], 4)}
              for i in order[:top]]
    discords = [{"index": i, "distance": round(profile[i], 4)}
                for i in order[-top:][::-1]]
    return {"verdict": "OK", "window": window,
            "motifs": motifs, "discords": discords}


# ---------------------------------------------------------------------------
# PELT changepoints (Gaussian L2 cost, exponential-marginal pruning)
# ---------------------------------------------------------------------------

def pelt_changepoints(series: List[float], penalty: float = 8.0,
                      min_segment: int = 5) -> List[int]:
    """PELT (Killick, Fearnhead & Eckley 2012, Algorithm 1): exact
    changepoint detection with pruning, O(n) when changes are sparse.
    Returns the indices of the first points of new segments (0
    excluded — the start is not a change). Past _PELT_CAP: abstain
    ([])."""
    series = _validate(series, 2 * min_segment)
    n = len(series)
    if n > _PELT_CAP:
        return []
    cs = [0.0]
    cs2 = [0.0]
    for x in series:
        cs.append(cs[-1] + x)
        cs2.append(cs2[-1] + x * x)

    def cost(a: int, b: int) -> float:
        """Gaussian L2 cost of segment [a, b)"""
        k = b - a
        if k <= 0:
            return 0.0
        s = cs[b] - cs[a]
        s2 = cs2[b] - cs2[a]
        return s2 - s * s / k

    beta = penalty
    f = [math.inf] * (n + 1)
    f[0] = -beta
    parents = [0] * (n + 1)
    candidates = {0}
    for t in range(min_segment, n + 1):
        best = math.inf
        best_s = 0
        for s in sorted(candidates):
            value = f[s] + cost(s, t) + beta
            if value < best:
                best = value
                best_s = s
        f[t] = best
        parents[t] = best_s
        # R(t) = {t} U {argmin} U {s : F(s) + C(s,t) < F(t)} — t itself
        # is a potential segment start for every future step
        survivors = {s for s in candidates
                     if f[s] + cost(s, t) < f[t]}
        candidates = {t, best_s} | survivors
    cps: List[int] = []
    t = n
    while t > 0:
        prev = parents[t]
        if prev > 0:
            cps.append(prev)
        t = prev
    cps.reverse()
    return cps


# ---------------------------------------------------------------------------
# Holt-Winters (additive seasonality, fixed smoothing constants)
# ---------------------------------------------------------------------------

def holt_winters(series: List[float], period: int,
                 alpha: float = 0.3, beta: float = 0.05,
                 gamma: float = 0.3) -> Dict[str, Any]:
    """Additive Holt-Winters over one full season + 2 extra points.
    Returns the level/trend/seasonal components, the forecast for one
    period ahead, and a seasonality-strength measure (Wang et al.:
    1 - Var(residual)/Var(seasonal-detrended)); >= 0.6 is 'has a
    rhythm', which is exactly the honest reading of 'every night'."""
    series = _validate(series, 2 * period + 2)
    n = len(series)
    # init: two-pass — level = overall mean, seasonal = per-phase mean
    # deviation across the WHOLE series (deterministic, and unlike a
    # first-season-only init it does not miss the rhythm when the first
    # season happens to be quiet)
    overall = sum(series) / n
    phase_sums = [0.0] * period
    phase_counts = [0] * period
    for t, x in enumerate(series):
        phase_sums[t % period] += x - overall
        phase_counts[t % period] += 1
    level = overall
    trend = 0.0
    season = [phase_sums[i] / phase_counts[i] for i in range(period)]
    fitted: List[float] = []
    for t in range(n):
        value = series[t]
        phase = t % period
        forecast = level + trend + season[phase]
        fitted.append(forecast)
        old_level = level
        level = alpha * (value - season[phase]) + (1 - alpha) * \
            (level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        season[phase] = gamma * (value - level) + (1 - gamma) * \
            season[phase]
    forecast_next = [level + trend + (season[(n + h) % period])
                     for h in range(period)]
    # seasonality strength (Wang, Smith & Hyndman 2006): 1 -
    # Var(remainder) / Var(deseasonalized). The deseasonalized series
    # removes ONLY the seasonal component (fit minus its seasonal term),
    # so a strong rhythm leaves a far more variable deseasonalized
    # series than remainder — 1 - small/big lands near 1.
    remainder = [series[t] - fitted[t] for t in range(period, n)]
    no_season = [fitted[t] - season[t % period]
                 for t in range(period, n)]
    var_r = statistics.pvariance(remainder) if len(remainder) > 1 else 0.0
    var_d = statistics.pvariance([series[t] - no_season[t - period]
                                  for t in range(period, n)]) \
        if len(no_season) > 1 else 0.0
    strength = max(0.0, min(1.0, 1.0 - var_r / var_d)) if var_d > 0 else 0.0
    return {"period": period, "level": round(level, 4),
            "trend": round(trend, 6),
            "seasonal": [round(s, 4) for s in season],
            "forecast_next": [round(v, 4) for v in forecast_next],
            "seasonality_strength": round(strength, 4),
            "has_rhythm": strength >= 0.6}


# ---------------------------------------------------------------------------
# one report: the four classical questions over one series
# ---------------------------------------------------------------------------

def seasonality_report(series: List[float], period: int = 24,
                       window: int = 6, penalty: float = 8.0) \
        -> Dict[str, Any]:
    """Motifs, discords, changepoints, seasonality — one card."""
    data = [float(x) for x in series]
    out: Dict[str, Any] = {"n": len(data), "period": period}
    if len(data) >= 2 * period + 2:
        hw = holt_winters(data, period, )
        out["seasonality"] = {
            "strength": hw["seasonality_strength"],
            "has_rhythm": hw["has_rhythm"],
            "reading": ("a rhythm exists at this period — 'every night' "
                        "is a fair reading"
                        if hw["has_rhythm"] else
                        "no reliable rhythm at this period")}
    else:
        out["seasonality"] = {
            "verdict": "ABSTAIN",
            "note": f"need >= {2 * period + 2} points for one honest "
                    f"season, got {len(data)}"}
    cps = pelt_changepoints(data, penalty=penalty) if len(data) >= 10 \
        and len(data) <= _PELT_CAP else []
    out["changepoints"] = {"indices": cps,
                           "verdict": "ABSTAIN" if len(data) > _PELT_CAP
                           or len(data) < 10 else "OK"}
    if len(data) >= 8:
        out["profile"] = motifs_and_discords(data, window=min(
            window, max(3, len(data) // 3)))
    else:
        out["profile"] = {"verdict": "ABSTAIN",
                          "note": f"need >= 8 points for a windowed "
                                  f"profile, got {len(data)}"}
    return out
