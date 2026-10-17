"""brain.timing — attention-aware suggestion timing (exponential-build
3.4): bias WHEN a proposal surfaces, never WHAT.

The pieces it composes (grep-first: every one already existed):

- the Thompson-sampling reminder bandit — brain's bandit family
  (HourBandit's algorithm, NamedBandit's named arms) — now over TIME
  BUCKETS rather than prompt content: which part of the day does this
  user actually act on proposals?
- the rhythm engine (brain/rhythm.py::hour_of_day_pattern) — the
  user's own activity z-scores per hour, so an active hour is
  preferred over a dormant one;
- the forecast engine (brain/forecast.py::holt) — the trend of recent
  acceptance rates decides whether surfacing soon makes sense at all
  (a falling trend advises deferring, reported not enforced);
- historical apply latency — the median hours-from-surface-to-decision
  per bucket, an attention signal: fast decisions mean the bucket
  catches the user at the keyboard;
- the POOLED PRIOR (brain/pooling.py, exponential-build 3.1) — the
  timing bandit's arms are seeded from the pooled statistics of the
  user's existing proposal posteriors instead of a flat Beta(1,1):
  this is the partial-pooling consumer the shared utility was built
  for.

The output RANKS candidate buckets for the next ``horizon_hours`` and
may advise deferring; it never surfaces anything itself, never changes
what is proposed, and every proposal still goes through the ledger's
own approve/reject flow. Deterministic given a seeded rng (Thompson
sampling is the one stochastic step, exactly like every other bandit
in this repo)."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .forecast import holt
from .pooling import hierarchical_prior, pool_arms
from .rhythm import hour_of_day_pattern

__all__ = ["BUCKET_NAMES", "bucket_of", "suggest_surface_bucket",
           "N_BUCKETS"]

N_BUCKETS = 6  # 4-hour buckets
BUCKET_NAMES = ("night", "early", "morning", "midday", "evening", "late")
_DEFAULT_HORIZON = 12


def bucket_of(hour: int) -> int:
    return int(hour) % 24 // 4


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


def _bucket_evidence(history_events: Sequence[Dict[str, Any]]
                     ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """Per-bucket applied/rejected counts and decision latencies."""
    applied: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    rejected: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    latencies: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    for e in history_events:
        if not isinstance(e, dict) or e.get("hour") is None:
            continue
        b = bucket_of(int(e["hour"]))
        if e.get("applied"):
            applied[b].append(1.0)
        elif e.get("applied") is not None:
            rejected[b].append(1.0)
        if e.get("latency_hours") is not None:
            try:
                latencies[b].append(max(0.0, float(e["latency_hours"])))
            except (TypeError, ValueError):
                continue
    return ({"applied": applied, "rejected": rejected}, latencies)


def _rhythm_multiplier(hour: int, activities: Optional[Sequence[int]]) -> float:
    """1.0 when the user has no recorded activity pattern; up to +50%
    for a strongly active hour (z >= 1.5, the rhythm engine's own
    `notable` threshold), computed from the user's own z-scores."""
    if not activities:
        return 1.0
    z = {row["hour"]: row["z"]
         for row in hour_of_day_pattern(list(activities))}
    score = z.get(hour % 24, 0.0)
    return 1.0 + max(0.0, min(score, 3.0)) / 6.0


def _latency_multiplier(bucket: int,
                        latencies: Dict[int, List[float]]) -> float:
    """1.0 with no latency history; otherwise the bucket's median
    decision latency relative to the overall median (fast bucket =
    above 1.0, capped at 1.5)."""
    all_lat = [v for b in latencies for v in latencies[b]]
    if not all_lat:
        return 1.0
    overall = _median(all_lat)
    if not overall:
        return 1.0
    own = _median(latencies[bucket])
    if own is None:
        return 1.0
    return 1.0 + max(0.0, min((overall - own) / overall, 0.5))


def suggest_surface_bucket(history_events: Sequence[Dict[str, Any]],
                           now_hour: int,
                           seed_arms: Optional[Dict[str, Sequence[float]]] = None,
                           activities: Optional[Sequence[int]] = None,
                           recent_accept_series: Optional[Sequence[float]] = None,
                           horizon_hours: int = _DEFAULT_HORIZON,
                           rng=None) -> Dict[str, Any]:
    """Rank the next ``horizon_hours`` worth of buckets for surfacing.

    ``seed_arms``: the user's existing proposal posteriors
    ({name: [alpha, beta]}) — pooled through brain.pooling so the
    timing arms inherit the population's acceptance profile instead of
    a flat prior. ``history_events``: this timing bandit's own
    {"hour", "applied", "latency_hours"} rows. Returns the ranked
    buckets, the deferral advice from the acceptance trend, and the
    evidence — the caller still decides, and still proposes through
    the ledger."""
    rng = rng or random
    if not 0 <= int(now_hour) < 24:
        raise ValueError("now_hour must be within [0, 24)")
    horizon_hours = max(1, min(int(horizon_hours), 24))

    # 1. the pooled hierarchical prior (3.1) — or the flat prior when
    #    there is nothing to borrow from (pooling says so itself)
    pooled = pool_arms(seed_arms) if seed_arms else \
        {"mu0": 0.5, "k0": 0.0, "pooled": False}
    prior = hierarchical_prior(pooled)

    # 2. the timing bandit's own evidence per bucket
    evidence, latencies = _bucket_evidence(history_events)
    arms: Dict[str, List[float]] = {}
    for b in range(N_BUCKETS):
        alpha = prior["alpha"] + sum(evidence["applied"][b])
        beta = prior["beta"] + sum(evidence["rejected"][b])
        arms[BUCKET_NAMES[b]] = [alpha, beta]

    # 3. Thompson draw per candidate bucket over the next horizon
    candidate_buckets = []
    for offset in range(horizon_hours):
        b = bucket_of(int(now_hour) + offset)
        if b not in candidate_buckets:
            candidate_buckets.append(b)

    ranked: List[Dict[str, Any]] = []
    for b in candidate_buckets:
        alpha, beta = arms[BUCKET_NAMES[b]]
        draw = rng.betavariate(alpha, beta)
        r_mult = _rhythm_multiplier(b * 4 + 2, activities)  # mid-hour
        l_mult = _latency_multiplier(b, latencies)
        ranked.append({"bucket": b, "name": BUCKET_NAMES[b],
                       "thompson_draw": round(draw, 4),
                       "score": round(draw * r_mult * l_mult, 4),
                       "rhythm_multiplier": round(r_mult, 3),
                       "latency_multiplier": round(l_mult, 3),
                       "prior_mean": round(alpha / (alpha + beta), 4)})
    ranked.sort(key=lambda r: (-r["score"], r["bucket"]))

    # 4. the acceptance trend (Holt): falling -> advise deferring,
    #    reported only, never enforced
    defer = False
    trend_note = "no acceptance series supplied — no trend advice"
    if recent_accept_series and len(recent_accept_series) >= 3:
        forecast = holt(list(recent_accept_series), horizon=1)
        if forecast and forecast[0] < 0.5:
            defer = True
            trend_note = (f"Holt forecast of the acceptance rate is "
                          f"{round(forecast[0], 3)} — consider deferring")
        else:
            trend_note = (f"Holt forecast of the acceptance rate is "
                          f"{round(forecast[0], 3)}")

    return {
        "recommended_bucket": ranked[0]["bucket"],
        "recommended_name": ranked[0]["name"],
        "ranked": ranked,
        "defer_advised": defer,
        "trend_note": trend_note,
        "prior_used": prior,
        "note": ("bias WHEN, not WHAT — the caller still proposes "
                 "through the ledger; nothing surfaces automatically"),
    }
