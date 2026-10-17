"""diagnostics.forecast — predictive hardware maintenance from the
telemetry time series (exponential-build-4 E).

The gap: telemetry.py reads a snapshot; the battery/thermal advisor
trends a rate. Neither answers "trending toward failure in ~N days".
This module fits a LOCAL LINEAR TREND through a scalar series (SMART
reallocated-sector counts, thermal readings) with a STEADY-STATE
KALMAN filter (Kalman 1960, "A New Approach to Linear Filtering and
Prediction Problems", ASME J. Basic Engineering 82D — the standard
[constant level, constant slope] local-linear model, equations exactly
the textbook's, fixed gain F from the steady-state solution rather
than a per-step update), and projects the filtered level+slope
forward to the series' own alarm threshold.

THE HONESTY THIS MODULE CARRIES (matching the battery/thermal
advisor's standard):

  * the interval around the projection is the IID-RESIDUAL interval:
    sigma of the one-step INNOVATIONS (filter residuals) times the
    horizon, student-t-free because no distribution is claimed — it
    is labeled an INDICATION, not a probability of failure;
  * a linear trend is a LOCAL model: the report says so, and a series
    with fewer than MIN_POINTS points is refused outright rather than
    fitted;
  * SMART reallocated-sector counts can jump overnight (a failing
    disk is nonlinear, sometimes abruptly) — the report carries that
    caveat verbatim rather than smoothing over it;
  * the projection never acts: it is a suggestion string for a human,
    read-only like every telemetry consumer.

Pure stdlib, no RNG, deterministic; state is plain dicts for the
caller's learned-state JSON path. Nothing here reads /proc or /sys —
the CALLER feeds series (telemetry.snapshot() is the live reader);
this keeps the module pure and the I/O enumerated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

__all__ = ["MIN_POINTS", "steady_state_kalman", "trend_projection",
           "failure_horizon", "forecast_report"]

MIN_POINTS = 6


def steady_state_kalman(series: Sequence[float], q: float = 0.01,
                        r: float = 1.0) -> Dict[str, Any]:
    """The [level, slope] local-linear Kalman filter over one series.

    State x = [level, slope]; transition x' = [level + slope, slope]
    with process noise q on the slope; observation = level with noise
    r. The fixed steady-state gain F solves the Riccati iteration to
    convergence (|F_k+1 − F_k| < 1e-9) — after which every step is two
    multiply-adds, which is the point of the steady-state form.
    Returns the filtered levels, the one-step innovations (filter
    residuals), and their sigma — the honesty input for every interval
    this module reports."""
    if len(series) < MIN_POINTS:
        raise ValueError(
            f"a trend needs >= {MIN_POINTS} points; {len(series)} is "
            "an honest refusal, not a confident straight line")
    if q <= 0 or r <= 0:
        raise ValueError("q and r must be positive")

    # Riccati iteration for the 2x2 steady-state gain (the [level,
    # slope] system's P recursions, iterated to a fixed point)
    p11 = p22 = 1.0
    p12 = 0.0
    gain = [0.0, 0.0]
    for _ in range(200):
        # predict: F = [[1, 1], [0, 1]] with slope process noise q
        p11p = p11 + 2 * p12 + p22 + q
        p12p = p12 + p22
        p22p = p22 + q
        # innovation variance for a level observation
        s = p11p + r
        k1, k2 = p11p / s, p12p / s
        if abs(k1 - gain[0]) < 1e-9 and abs(k2 - gain[1]) < 1e-9:
            gain = [k1, k2]
            break
        gain = [k1, k2]
        # update
        p11 = (1 - k1) * p11p
        p12 = (1 - k1) * p12p
        p22 = p22p - k2 * p12p

    level = float(series[0])
    slope = 0.0
    levels: List[float] = []
    innovations: List[float] = []
    for value in series:
        pred = level + slope
        innovation = float(value) - pred
        level = pred + gain[0] * innovation
        slope = slope + gain[1] * innovation
        levels.append(level)
        innovations.append(innovation)
    # sigma from the CONVERGED half of the innovations: the first few
    # steps are the filter locking onto the slope, and folding their
    # ramp into the noise would overstate the band (the honest band is
    # the one the filter achieves once it has converged)
    tail = innovations[len(innovations) // 2:]
    sigma = (math.sqrt(sum(i * i for i in tail) / max(1, len(tail)))
             if tail else 0.0)
    return {"levels": levels, "slope": slope, "level": level,
            "innovations": innovations, "sigma_innovation": sigma,
            "gain": gain, "n": len(series)}


def trend_projection(fit: Dict[str, Any], horizon: int
                     ) -> Dict[str, Any]:
    """Project the filtered [level, slope] forward ``horizon`` steps
    with the iid-residual interval at 1 sigma (the interval GROWS
    linearly with the horizon because the slope term accumulates the
    one-step sigma — the linear-trend model's own honesty)."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    sigma = fit["sigma_innovation"]
    slope = fit["slope"]
    projections = []
    for h in range(1, horizon + 1):
        value = fit["level"] + slope * h
        projections.append({"step": h, "value": value,
                            "sigma": sigma * h})
    return {"projections": projections, "horizon": horizon,
            "slope_per_step": slope}


def failure_horizon(fit: Dict[str, Any], threshold: float
                    ) -> Dict[str, Any]:
    """When does the filtered trend cross ``threshold``? Returns the
    step count and the honest verdicts: already past it, slope too
    flat to say (within the noise), or crossing at step N with the
    iid-residual caveat."""
    slope = fit["slope"]
    level = fit["level"]
    sigma = fit["sigma_innovation"]
    if level >= threshold:
        return {"crossed": True, "steps": 0,
                "note": "already at or past the threshold — the trend "
                        "model is not needed for that"}
    if slope <= 0:
        return {"crossed": False, "steps": None,
                "note": "non-increasing trend: no crossing is projected "
                        "(refusing to invent one)"}
    # noise floor: a slope smaller than sigma/step is indistinguishable
    # from flat over short horizons — say so instead of projecting
    if abs(slope) < sigma / 3.0:
        return {"crossed": False, "steps": None,
                "note": "the slope is within the innovation noise — "
                        "no crossing is claimed (a flatter honest "
                        "answer than a confident number)"}
    steps = (threshold - level) / slope
    return {"crossed": False, "steps": round(steps, 1),
            "sigma_per_step": sigma,
            "note": "iid-residual indication: steps = (threshold − "
                    "level)/slope with sigma from the filter's own "
                    "innovations; a linear-trend LOCAL model, not a "
                    "failure probability"}


def forecast_report(series: Sequence[float], threshold: float,
                    unit: str = "count", q: float = 0.01, r: float = 1.0,
                    max_horizon: int = 90
                    ) -> Dict[str, Any]:
    """The full report for one series against one threshold: the Kalman
    fit, the horizon, the 1-sigma band at the crossing, and the
    suggestion string. Read-only by construction — the caller decides
    what (if anything) to do with a suggestion; nothing acts here."""
    fit = steady_state_kalman(series, q=q, r=r)
    horizon = failure_horizon(fit, threshold)
    projection = (trend_projection(fit, min(int(horizon["steps"]) + 2,
                                            max_horizon))
                  if horizon.get("steps") else None)
    if horizon.get("crossed"):
        suggestion = (f"the series is already at/past the {unit} "
                      f"threshold {threshold} — inspect now; do not "
                      "wait for a trend")
    elif horizon.get("steps"):
        steps = horizon["steps"]
        sigma = horizon.get("sigma_per_step", 0.0)
        suggestion = (f"trending toward the {unit} threshold "
                      f"{threshold} in ~{steps:g} steps at the current "
                      f"rate (iid-residual band ±{sigma:.3g}/step) — "
                      "an indication from a LOCAL linear trend, not a "
                      "failure probability; SMART counters can jump "
                      "nonlinearly overnight")
    else:
        suggestion = (f"no crossing of the {unit} threshold "
                      f"{threshold} is projected: "
                      f"{horizon.get('note', '')}")
    return {
        "n_points": len(series),
        "filtered_level": round(fit["level"], 4),
        "slope_per_step": round(fit["slope"], 6),
        "sigma_innovation": round(fit["sigma_innovation"], 6),
        "horizon": horizon,
        "projection": projection,
        "suggestion": suggestion,
        "caveats": [
            "LOCAL linear trend: honest near the end of the series, "
            "not a physical failure model",
            "the interval is the filter's own iid-residual band; "
            "correlated noise is not modeled",
            "SMART reallocated-sector counts can jump nonlinearly "
            "overnight — re-check against the raw series",
        ],
    }
