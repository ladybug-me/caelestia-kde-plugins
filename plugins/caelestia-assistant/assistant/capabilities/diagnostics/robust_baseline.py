"""diagnostics.robust_baseline — a ROBUST Mahalanobis-distance baseline
over the telemetry layer's numeric metrics, so "is this machine behaving
like itself lately?" has a distribution-free-ish, outlier-resistant
answer instead of a mean/variance guess that the outliers it hunts
would poison.

The method (each piece cited):

- center = coordinate-wise MEDIAN, scale = 1.4826 x MAD (the median
  absolute deviation with its Gaussian-consistency constant) — the
  robust substitution for mean/stdev, per Leys, Klein, Bernard &
  Laurent 2013, "Detecting outliers: Do not use standard deviation
  around the mean, use absolute deviation around the median";
- distance = the Mahalanobis distance with that diagonal robust
  scatter (Mahalanobis 1936, "On the generalised distance in
  statistics"): MD(x) = sqrt(sum_c ((x_c - median_c) / scale_c)^2);
- the p-value tail uses the EXISTING stats.chi2_sf primitive (MD^2 ~
  chi-square(k) under the stated independence/normality approximation
  — the caveat travels with the report).

Honesty rules (the reason this module exists in THIS repo):

- a coordinate whose history never moved (MAD = 0) is EXCLUDED and
  NAMED, never given a fake scale so the arithmetic keeps working;
- a sample coordinate that is absent (no battery, no thermal zone on
  this machine) is excluded and counted, never imputed;
- if fewer than half of the baseline's kept coordinates are present in
  the sample, the distance is REFUSED (thin evidence), not computed
  from a rump;
- fewer than ``min_samples`` history rows is a refusal up front.

Pure functions over caller-supplied numbers: this module never reads
/proc or /sys itself (the telemetry layer owns that read-only surface
and the caller hands the flattened vectors in). Deterministic, no RNG,
no clock, no I/O.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

__all__ = ["flatten_snapshot", "robust_baseline", "mahalanobis",
           "MIN_SAMPLES"]

MIN_SAMPLES = 8  # a median over fewer rows is thin evidence by definition

# The Gaussian-consistency constant for the MAD (Leys et al. 2013).
_MAD_TO_SIGMA = 1.4826


def flatten_snapshot(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """The numeric metrics one telemetry.snapshot() contributes to the
    baseline: load averages, memory used-ratio, mean thermal zone
    temperature. Battery capacity is DELIBERATELY excluded — it cycles
    by design, so it is a poor drift signal (documented, not an
    oversight). Only available probes contribute; a desktop without a
    battery simply has fewer coordinates."""
    out: Dict[str, float] = {}
    load = snapshot.get("loadavg") or {}
    if load.get("available"):
        out["load1"] = float(load["load1"])
        out["load5"] = float(load["load5"])
        out["load15"] = float(load["load15"])
    mem = snapshot.get("meminfo") or {}
    if mem.get("available") and mem.get("used_ratio") is not None:
        out["mem_used_ratio"] = float(mem["used_ratio"])
    thermal = snapshot.get("thermal") or {}
    temps = [z["temp_c"] for z in (thermal.get("zones") or [])
             if isinstance(z, dict) and z.get("temp_c") is not None]
    if temps:
        out["thermal_mean_c"] = round(sum(temps) / len(temps), 3)
    return out


def robust_baseline(history: Sequence[Dict[str, float]],
                    min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """The robust center/scale per coordinate from the history rows
    (caller-supplied flattened snapshots). Degenerate coordinates (MAD
    = 0: the metric never moved) are excluded and NAMED."""
    if len(history) < min_samples:
        raise ValueError(
            f"need >= {min_samples} history rows for a robust baseline "
            f"(got {len(history)}) — refusing to baseline on thin data")
    coords: Dict[str, List[float]] = {}
    for row in history:
        for key, value in (row or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                coords.setdefault(key, []).append(float(value))
    if not coords:
        raise ValueError("history rows contain no numeric coordinates")
    center: Dict[str, float] = {}
    scale: Dict[str, float] = {}
    n: Dict[str, int] = {}
    degenerate: List[str] = []
    for key in sorted(coords):
        vals = sorted(coords[key])
        m = len(vals)
        median = vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2.0
        mad = sorted(abs(v - median) for v in vals)[m // 2]
        n[key] = m
        if mad <= 0:
            degenerate.append(key)  # never moved: no honest scale exists
            continue
        center[key] = median
        scale[key] = round(_MAD_TO_SIGMA * mad, 6)
    if not center:
        raise ValueError(
            "every coordinate is degenerate (MAD = 0) — no honest "
            "distance exists; the history shows no variation at all")
    return {"center": center, "scale": scale, "n": n,
            "kept": sorted(center), "degenerate": degenerate,
            "n_history_rows": len(history),
            "method": ("robust Mahalanobis: median center + 1.4826 x MAD "
                       "scale (Leys et al. 2013), diagonal scatter "
                       "(Mahalanobis 1936)")}


def mahalanobis(sample: Dict[str, float],
                baseline: Dict[str, Any],
                p_threshold: float = 0.01) -> Dict[str, Any]:
    """The robust Mahalanobis distance of one sample from the baseline,
    with per-coordinate contributions and the chi-square tail p-value
    from the EXISTING stats primitive. Missing coordinates are excluded
    and counted; fewer than half present is a refusal (thin evidence);
    the verdict threshold is stated, the p-value is reported, and the
    approximation caveat travels with the answer."""
    center = baseline.get("center") or {}
    scale = baseline.get("scale") or {}
    if not center:
        raise ValueError("baseline has no kept coordinates")
    used: Dict[str, float] = {}
    missing: List[str] = []
    contributions: Dict[str, float] = {}
    for key in sorted(center):
        if key not in sample or sample[key] is None:
            missing.append(key)
            continue
        value = float(sample[key])
        z = (value - center[key]) / scale[key]
        used[key] = value
        contributions[key] = round(z * z, 4)
    if len(used) * 2 < len(center):
        raise ValueError(
            f"sample covers {len(used)} of {len(center)} baseline "
            "coordinates — too thin to place honestly (refusing, not "
            "extrapolating)")
    md = math.sqrt(sum(contributions.values()))
    k = len(used)
    from assistant.capabilities.genius import stats as stats_mod  # lazy: the layer boundary stays explicit
    p = stats_mod.chi2_sf(md * md, k)
    top = sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "distance": round(md, 4),
        "coordinates_used": k,
        "missing_in_sample": missing,
        "top_contributions": dict(top),
        "p_value": p,
        "p_threshold": p_threshold,
        "verdict": ("anomalous" if p is not None and p < p_threshold
                    else "within-baseline"),
        "note": ("MD^2 ~ chi-square(k) assumes independent coordinates "
                 "and roughly Gaussian behavior per coordinate — the "
                 "p-value is an approximation, reported not worshipped"),
    }
