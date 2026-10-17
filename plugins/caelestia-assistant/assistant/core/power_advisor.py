"""cortex.power_advisor — predictive battery and thermal advice, inert.

Holt 1957, "Forecasting Seasonals and Trends by Exponentially Weighted
Moving Averages", ONR Memorandum 52 — the two-parameter linear-trend
exponentially-weighted recursion (level + trend, the same recursion
brain/forecast.py::holt implements; this module re-derives it because
an ADVISOR needs the internals holt() discards — the one-step
residuals for the uncertainty and the final level/trend pair — and a
cross-check test pins the two implementations to identical point
forecasts so the duplication cannot drift).

Adams & MacKay 2007, "Bayesian Online Changepoint Detection",
arXiv:0710.3742v2 — the changepoint half: BOCPD over the battery
series' first DIFFERENCES (per-sample drain), reusing
genius/data.py::bocpd verbatim, because a drain-regime change is a
mean shift in the differences, not in the level. When the last step's
changepoint probability is high, the trend extrapolation is built on
stale history and the report says so instead of hiding it.

UNCERTAINTY (stated plainly): the interval is the naive iid-residual
normal interval — half-width z_0.975 * sigma_hat * sqrt(1 + h) with
sigma_hat the residual standard deviation — which UNDERSTATES the
uncertainty of a trending process (residuals of a misspecified trend
are autocorrelated). It is reported as an indication, never as a
guarantee, and the report says "iid-residual interval" so nobody can
mistake it for a calibrated prediction interval.

CONSENT: this module SUGGESTS, it never acts. Every suggestion is an
inert SUGGESTED_NOT_EXECUTED string with a risk tier; the presets it
names are applied by the user through the ordinary settings gates
(preview -> consent -> apply), never by this code. No writes, no
execution, no network; series are caller-supplied (the repo's fixture-
injectable pattern — the same contract telemetry/dreamtime use).

Honesty: charging series (non-negative trend) get "charging: no drain
estimate" instead of an extrapolated time-to-low; thin series (< 4
points) ABSTAIN with the reason; the changepoint check needs >= 5
points (BOCPD needs 4 differences) and is honestly skipped below that;
out-of-range values (battery outside 0..100, thermal outside
25000..120000 millidegrees) are refused, not clamped.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["holt_state", "advise_battery", "advise_thermal", "advise"]

Z_975 = 1.959963984540054  # the normal 97.5% quantile
MIN_POINTS = 4
CHANGEPOINT_MIN = 5  # BOCPD needs >= 4 differences
RECENT_WINDOW = 3  # steps at the end where a change still invalidates the trend


def holt_state(series: Sequence[float], alpha: float = 0.5,
               beta: float = 0.3
               ) -> Dict[str, Any]:
    """The Holt recursion WITH internals: final level, final trend,
    one-step residuals, residual std, and the point forecasts a caller
    can extend. Matches brain.forecast.holt's point forecasts exactly
    (same init level=s[0], trend=s[1]-s[0], same update order — pinned
    by cross-check test). Refuses < 2 points, alpha/beta outside
    (0, 1]."""
    s = [float(x) for x in series]
    if len(s) < 2:
        raise ValueError(
            f"Holt needs >= 2 points (got {len(s)}) — fewer cannot "
            "separate level from trend; abstaining")
    for name, v in (("alpha", alpha), ("beta", beta)):
        if not 0.0 < v <= 1.0:
            raise ValueError(
                f"{name} must be inside (0, 1] (got {v!r})")
    level, trend = s[0], s[1] - s[0]
    residuals: List[float] = []
    for x in s[1:]:
        predicted = level + trend  # the one-step-ahead prediction
        residuals.append(x - predicted)
        prev_level = level
        level = alpha * x + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    dof = max(1, len(residuals) - 1)
    sigma = math.sqrt(sum(r * r for r in residuals) / dof)
    return {"level": level, "trend": trend,
            "residuals": residuals, "residual_std": sigma, "n": len(s)}


def _forecast_path(state: Dict[str, Any], steps: int, z: float
                   ) -> List[Dict[str, Any]]:
    out = []
    for h in range(steps):
        point = state["level"] + (h + 1) * state["trend"]
        half = z * state["residual_std"] * math.sqrt(1.0 + h)
        out.append({"step": h + 1,
                    "point": round(point, 4),
                    "low": round(point - half, 4),
                    "high": round(point + half, 4)})
    return out


def _changepoint_note(series: Sequence[float], variance_floor: float
                     ) -> Dict[str, Any]:
    """BOCPD over first differences: a drain-regime change is a mean
    shift in the per-step deltas. TWO calibration choices, both stated
    because the default would be wrong here: (1) the observation
    variance is estimated from the FIRST HALF of the differences (the
    pre-shift baseline), floored at the probe's granularity —
    bocpd's own default estimates noise over the WHOLE series, so a
    shift contaminates the very scale that should reveal it (a
    blatant -2 -> -20 drain step scores p=0.17 under the default and
    p=1.0 under the baseline estimate — the contamination is not
    hypothetical); (2) a change is RECENT — and therefore invalidates
    the trend extrapolation — when BOCPD places it within the last
    RECENT_WINDOW steps; an older change has been partially absorbed
    by Holt's exponential re-fit (alpha=0.5 halves a shift's weight
    per step), so only recent ones are flagged. Gradual accelerations
    have no single boundary and are honestly NOT flagged — Holt's
    trend already re-weights them. Honestly skipped below
    CHANGEPOINT_MIN points (BOCPD needs 4 differences)."""
    if len(series) < CHANGEPOINT_MIN:
        return {"checked": False,
                "note": f"changepoint check skipped: BOCPD needs >= "
                        f"{CHANGEPOINT_MIN} points ({len(series)} given)"}
    from assistant.capabilities.genius.data import bocpd
    diffs = [float(b) - float(a) for a, b in zip(series, series[1:])]
    baseline = diffs[:max(2, len(diffs) // 2)]
    mean = sum(baseline) / len(baseline)
    var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    variance = max(var, float(variance_floor))
    run = bocpd(diffs, variance=variance)
    probs = run.get("changepoint_prob") or []
    n_diffs = len(diffs)
    detected = [int(i) for i in (run.get("changepoints") or [])]
    recent_probs = probs[-RECENT_WINDOW:] if probs else []
    p_recent_max = max(recent_probs) if recent_probs else 0.0
    recent = [i for i in detected if i > n_diffs - RECENT_WINDOW]
    return {"checked": True,
            "detected_changepoints": detected,
            "recent": bool(recent),
            "recent_window": RECENT_WINDOW,
            "recent_changepoint_prob_max": round(float(p_recent_max), 4),
            "note": ("a recent drain-regime change means the trend "
                      "extrapolation is built on mixed history — treat "
                      "the forecast and the interval as indications"
                      if recent else
                      "no regime change within the recent window")}


def _series(series: Sequence[float], lo: float, hi: float, what: str
            ) -> List[float]:
    out = [float(x) for x in series]
    for x in out:
        if not lo <= x <= hi:
            raise ValueError(
                f"{what} value {x} outside the plausible range "
                f"[{lo}, {hi}] — refusing rather than clamping")
    return out


def advise_battery(series: Sequence[float], sample_minutes: float = 30.0,
                   low_pct: float = 20.0, horizon_steps: int = 8,
                   alpha: float = 0.5, beta: float = 0.3
                   ) -> Dict[str, Any]:
    """Battery drain advice from a caller-supplied capacity series
    (percent, uniformly spaced samples ``sample_minutes`` apart).

    Returns the Holt state, the forecast path with the iid-residual
    interval, the honest time-to-low estimate (None while charging),
    the changepoint note, and an INERT suggestion naming the
    battery-saver preset (applied only through the ordinary gates).
    Thin series (< MIN_POINTS) abstain with the reason."""
    if sample_minutes <= 0:
        raise ValueError("sample_minutes must be positive")
    if not 0.0 <= low_pct <= 100.0:
        raise ValueError("low_pct must be inside 0..100")
    s = _series(series, 0.0, 100.0, "battery")
    if len(s) < MIN_POINTS:
        return {"available": False,
                "reason": f"battery advice needs >= {MIN_POINTS} "
                          f"samples (got {len(s)}); abstaining, not "
                          "guessing a trend"}
    state = holt_state(s, alpha=alpha, beta=beta)
    charging = state["trend"] >= 0.0
    already_low = state["level"] <= low_pct
    minutes_to_low: Optional[float] = None
    if not charging and not already_low and low_pct < state["level"]:
        samples = (state["level"] - low_pct) / -state["trend"]
        minutes_to_low = round(samples * sample_minutes, 1)
    # variance floor 1.0 = one squared integer-percent: the probe's
    # granularity (capacity is reported in whole percent)
    cp = _changepoint_note(s, 1.0)
    suggestion = None
    if already_low:
        suggestion = (
            f"battery is already at or below the {low_pct:g}% threshold "
            f"(level estimate {state['level']:.1f}%) — the battery-saver "
            "preset exists and can be previewed with `python3 -m "
            "assistant.capabilities.settings --preset battery-saver` (this advice "
            "applies nothing)")
    elif minutes_to_low is not None:
        tier = "STATE_CHANGING"
        suggestion = (
            f"SUGGESTED_NOT_EXECUTED [{tier}]: battery is projected to "
            f"reach {low_pct:g}% in ~{minutes_to_low / 60:.1f} h at the "
            "current drain; the battery-saver preset exists and can be "
            "previewed with `python3 -m assistant.capabilities.settings --preset "
            "battery-saver` (this advice applies nothing)")
    elif charging:
        suggestion = (
            "no drain estimate: the series is charging or flat "
            f"(trend {state['trend']:+.3f}%/sample) — nothing suggested")
    return {
        "available": True,
        "n": len(s),
        "sample_minutes": sample_minutes,
        "level": round(state["level"], 4),
        "trend_per_sample": round(state["trend"], 4),
        "trend_pct_per_hour": round(
            state["trend"] * 60.0 / sample_minutes, 4),
        "charging": charging,
        "minutes_to_low": minutes_to_low,
        "low_threshold_pct": low_pct,
        "forecast": _forecast_path(state, horizon_steps, Z_975),
        "interval_note": "iid-residual normal interval — an "
                         "indication, not a calibrated prediction "
                         "interval (residuals of a trending process "
                         "are autocorrelated)",
        "changepoint": cp,
        "suggestion": suggestion,
    }


def advise_thermal(series: Sequence[float], sample_minutes: float = 5.0,
                   high_mc: float = 85_000.0, horizon_steps: int = 8,
                   alpha: float = 0.5, beta: float = 0.3
                   ) -> Dict[str, Any]:
    """Thermal advice from a caller-supplied zone-temperature series
    (millidegrees C, uniformly spaced). Same shape as
    advise_battery; the inert suggestion names load reduction and the
    minimal preset as the existing knobs."""
    if sample_minutes <= 0:
        raise ValueError("sample_minutes must be positive")
    s = _series(series, 25_000.0, 120_000.0, "thermal")
    if len(s) < MIN_POINTS:
        return {"available": False,
                "reason": f"thermal advice needs >= {MIN_POINTS} "
                          f"samples (got {len(s)}); abstaining, not "
                          "guessing a trend"}
    state = holt_state(s, alpha=alpha, beta=beta)
    rising = state["trend"] > 0.0
    already_high = state["level"] >= high_mc
    minutes_to_high: Optional[float] = None
    if rising and not already_high and high_mc > state["level"]:
        samples = (high_mc - state["level"]) / state["trend"]
        minutes_to_high = round(samples * sample_minutes, 1)
    # variance floor 1_000_000 = (1000 mC)^2 = one squared degree C:
    # thermal zones breathe by about a degree between samples — a
    # sharper floor makes the Gaussian model underflow on gross
    # shifts and MISS them (a 25 degC step at a 0.1 degC floor scores
    # p=0.0 everywhere; at this floor it scores p~1.0, verified)
    cp = _changepoint_note(s, 1_000_000.0)
    suggestion = None
    if already_high:
        suggestion = (
            f"the zone is already at or above {high_mc / 1000:.0f} C "
            f"(level estimate {state['level'] / 1000:.1f} C) — reducing "
            "load (closing heavy apps) or previewing the minimal preset "
            "(`python3 -m assistant.capabilities.settings --preset minimal`) are the "
            "existing knobs (this advice applies nothing)")
    elif minutes_to_high is not None:
        suggestion = (
            "SUGGESTED_NOT_EXECUTED [STATE_CHANGING]: the zone is "
            f"projected to reach {high_mc / 1000:.0f} C in "
            f"~{minutes_to_high:.0f} min at the current trend; reducing "
            "load (closing heavy apps) or previewing the minimal preset "
            "(`python3 -m assistant.capabilities.settings --preset minimal`) are the "
            "existing knobs (this advice applies nothing)")
    return {
        "available": True,
        "n": len(s),
        "sample_minutes": sample_minutes,
        "level_mc": round(state["level"], 1),
        "trend_mc_per_sample": round(state["trend"], 2),
        "rising": rising,
        "minutes_to_high": minutes_to_high,
        "high_threshold_mc": high_mc,
        "forecast": _forecast_path(state, horizon_steps, Z_975),
        "interval_note": "iid-residual normal interval — an "
                         "indication, not a calibrated prediction "
                         "interval",
        "changepoint": cp,
        "suggestion": suggestion,
    }


def advise(battery: Optional[Sequence[float]] = None,
           thermal: Optional[Sequence[float]] = None,
           battery_sample_minutes: float = 30.0,
           thermal_sample_minutes: float = 5.0) -> Dict[str, Any]:
    """The combined advisory report (each half honestly absent when
    its series is not supplied). Pure read-only; suggestions are inert
    strings."""
    report: Dict[str, Any] = {}
    if battery is not None:
        report["battery"] = advise_battery(
            battery, sample_minutes=battery_sample_minutes)
    if thermal is not None:
        report["thermal"] = advise_thermal(
            thermal, sample_minutes=thermal_sample_minutes)
    if not report:
        report["note"] = ("no series supplied — nothing to advise; "
                          "telemetry series are caller-supplied "
                          "(sample battery/thermal with the same "
                          "on-demand probes, then pass the series)")
    return report
