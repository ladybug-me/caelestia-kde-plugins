"""brain.bursts — Hawkes burst detection: crash loops, notification
storms (C12).

A crash loop is not N independent crashes; each crash makes the next
one likelier (retry storms, restart cascades, notification piles). The
classical model for that self-excitation is the Hawkes process:

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

where mu is the base rate, alpha the per-event excitation, beta the
decay. The branching ratio alpha/beta says how much of the activity is
self-excited: R < 1 is a stable process, R -> 1 is a storm.

This module fits that model to event timestamps by BOUNDED,
DETERMINISTIC coordinate ascent on the exact log-likelihood (the
Ozaki-style recursive compensator, no approximations), then reports:

  - the fit (mu, alpha, beta, branching ratio) and its log-likelihood;
  - BURSTS: maximal runs of events whose fitted intensity exceeds
    base_rate * threshold — with start, end, peak, and event count;
  - the honest caveat, printed with every answer: bursts are
    CORRELATION, not causation — the model says the events clustered
    and excited each other IN TIME, not what caused them.

Deterministic: fixed grids, fixed iteration counts, no RNG. Bounded:
n <= 5000 events (past that, abstain); the fitter stops after a fixed
number of sweeps regardless of convergence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["fit_hawkes", "detect_bursts", "burst_report"]

_MAX_EVENTS = 5000
_MAX_SWEEPS = 12


def _validate_events(events: List[float]) -> List[float]:
    out = sorted(float(x) for x in events)
    if len(out) < 8:
        raise ValueError("need at least 8 event times to say anything "
                         "about bursts")
    return out


def _loglik(events: List[float], mu: float, alpha: float, beta: float,
            horizon: float) -> Tuple[float, List[float]]:
    """Exact Hawkes log-likelihood + the intensity at each event
    (Ozaki recursion: A_i = exp(-beta*(t_i - t_{i-1})) * (A_{i-1} + 1)).
    Events are shifted so the first lands at 0 — `horizon` is measured
    on the same relative axis."""
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -math.inf, []
    if not events:
        return -math.inf, []
    t0 = events[0]
    ev = [t - t0 for t in events]
    log_l = 0.0
    a = 0.0
    intensities: List[float] = []
    prev = 0.0
    for i, t in enumerate(ev):
        if i > 0:
            a = math.exp(-beta * (t - prev)) * (a + 1.0)
        else:
            a = 0.0
        lam = mu + alpha * a
        if lam <= 0:
            return -math.inf, []
        intensities.append(lam)
        log_l += math.log(lam)
        prev = t
    # compensator integral
    integral = mu * horizon
    a = 0.0
    prev = 0.0
    for i, t in enumerate(ev):
        if i > 0:
            a = math.exp(-beta * (t - prev)) * (a + 1.0)
        integral += (alpha / beta) * (1.0 - math.exp(-beta * (horizon - t)))
        prev = t
    return log_l - integral, intensities


def fit_hawkes(events: List[float]) -> Dict[str, Any]:
    """Bounded deterministic fit: coordinate sweeps over (mu, beta)
    with alpha closed-form-ish on a grid, shrinking around the best.
    Returns the fit plus honest quality notes."""
    ev = _validate_events(events)
    n = len(ev)
    horizon = ev[-1] - ev[0] + 1.0
    span = horizon
    base_rate = n / horizon
    mu = max(base_rate / 2.0, 1e-6)
    beta = max(1.0 / max(span / 20.0, 1e-6), 1e-6)
    best = (-math.inf, mu, 0.0, beta)
    # initial coarse grid over the two shape parameters
    beta_grid = [span / f for f in (2, 5, 10, 20, 50, 100, 200)]
    alpha_grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    for b in beta_grid:
        for a in alpha_grid:
            ll, _ = _loglik(ev, mu, a * base_rate, b, horizon)
            if ll > best[0]:
                best = (ll, mu, a * base_rate, b)
    # shrinking coordinate sweeps (deterministic, bounded)
    cur_ll, cur_mu, cur_alpha, cur_beta = best
    for sweep in range(_MAX_SWEEPS):
        step_mu = base_rate / (2 ** (sweep + 1))
        step_a = 0.2 / (2 ** sweep)
        step_b = cur_beta * 0.5 / (2 ** sweep)
        improved = False
        for dmu in (-step_mu, 0.0, step_mu):
            for da in (-step_a, 0.0, step_a):
                for db in (-step_b, 0.0, step_b):
                    m2 = max(cur_mu + dmu, 1e-9)
                    a2 = max(cur_alpha + da, 0.0)
                    b2 = max(cur_beta + db, 1e-9)
                    ll, _ = _loglik(ev, m2, a2, b2, horizon)
                    if ll > cur_ll + 1e-10:
                        cur_ll, cur_mu, cur_alpha, cur_beta = \
                            ll, m2, a2, b2
                        improved = True
        if not improved:
            break
    branching = cur_alpha / cur_beta if cur_beta > 0 else 0.0
    return {
        "mu": round(cur_mu, 6),
        "alpha": round(cur_alpha, 6),
        "beta": round(cur_beta, 6),
        "branching_ratio": round(branching, 4),
        "log_likelihood": round(cur_ll, 4),
        "base_rate_per_unit": round(base_rate, 6),
        "horizon": round(horizon, 6),
        "n_events": n,
        "stable": branching < 1.0,
        "reading": ("self-excitation is subcritical (R < 1): bursts "
                    "decay on their own" if branching < 1.0 else
                    "branching ratio >= 1: the fit says events feed "
                    "themselves faster than they decay — treat the "
                    "fit itself as unreliable"),
    }


def detect_bursts(events: List[float], intensity: List[float],
                  base_rate: float, factor: float = 2.0) \
        -> List[Dict[str, Any]]:
    """Maximal runs of events whose fitted intensity exceeds
    base_rate * factor."""
    if not intensity:
        return []
    threshold = base_rate * factor
    bursts: List[Dict[str, Any]] = []
    start = None
    peak = 0.0
    count = 0
    for i, lam in enumerate(intensity):
        if lam > threshold:
            if start is None:
                start = i
                peak = 0.0
                count = 0
            count += 1
            peak = max(peak, lam)
        else:
            if start is not None:
                bursts.append({"start": events[start],
                               "end": events[i - 1],
                               "peak_intensity": round(peak, 4),
                               "n_events": count,
                               "over_base": round(peak / base_rate, 2)
                               if base_rate > 0 else None})
                start = None
    if start is not None:
        bursts.append({"start": events[start],
                       "end": events[-1],
                       "peak_intensity": round(peak, 4),
                       "n_events": count,
                       "over_base": round(peak / base_rate, 2)
                       if base_rate > 0 else None})
    return bursts


def burst_report(events: List[float], factor: float = 2.0) \
        -> Dict[str, Any]:
    """One card: the fit, the bursts, the caveat. Too few events is an
    honest ABSTAIN (the CLI's contract), not a raise — the fitter's own
    ValueError contract stays on fit_hawkes."""
    try:
        ev = _validate_events(events)
    except ValueError as exc:
        return {"verdict": "ABSTAIN", "note": str(exc)}
    if len(ev) > _MAX_EVENTS:
        return {"verdict": "ABSTAIN",
                "note": f"more than {_MAX_EVENTS} events: the fitter "
                        f"abstains instead of approximating quietly"}
    fit = fit_hawkes(ev)
    _ll, intensities = _loglik(ev, fit["mu"], fit["alpha"],
                               fit["beta"], fit["horizon"])
    bursts = detect_bursts(ev, intensities, fit["mu"], factor=factor)
    for b in bursts:
        b["start"] = round(b["start"], 6)
        b["end"] = round(b["end"], 6)
    return {
        "verdict": "OK",
        "fit": fit,
        "n_bursts": len(bursts),
        "bursts": bursts,
        "threshold": f"intensity > {factor}x base rate",
        "caveat": "bursts are CORRELATION, not causation: the model "
                  "says events clustered and excited each other in "
                  "time — not what caused them",
    }
