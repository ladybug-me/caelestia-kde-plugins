"""Episodic interaction memory — the cortex's recall layer.

Every routed request, clarification, and outcome is an EPISODE. Episodes
power three capabilities none of the one-shot layers have:

1. RECALL — "what did i change last week?", "what do i keep asking
   about?" — answered from real interaction history, ranked by an
   exponential forgetting curve (Ebbinghaus-style decay; recall weight
   ``w = 0.5 ** (age_days / halflife)``) so recent behavior dominates
   and old habits fade instead of poisoning the statistics forever.
2. ASSOCIATION — co-change counts: which settings get changed together.
   This is market-basket analysis (lift = P(a and b) / (P(a) P(b)))
   over the user's own applied episodes. It powers the proactive
   follow-up suggestion: after applying ``setBarScale``, the most-lifted
   co-occurring setting becomes a LEDGER PROPOSAL (never auto-applied),
   e.g. "you usually also tighten spacing when you shrink the bar —
   propose setSpacingScale 0.9?"
3. HABITS — surface frequency by hour-of-day and by day-of-week
   (plain histograms; ``notable`` deviations reuse the rhythm idea).

Persistence: the caller passes a plain dict (loaded/saved by the brain
state machinery — same atomic-write path as every other learned model);
this module does no file I/O itself. Bounded: ``MAX_EPISODES`` newest
episodes retained (recall beyond that horizon is honestly refused).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

MAX_EPISODES = 500
HALFLIFE_DAYS = 14.0

# The user-editable override lives in the brain state (key
# MEMORY_HALFLIFE_KEY) — per-user, because usage cadences differ; a
# multi-user/community install each carries its own decay. The shipped
# default is 14 days; `caelestia-assist cortex halflife [DAYS]` reads or
# sets it. This is a correctness fix, not a knob: the default was a
# fixed constant while different users demonstrably need different
# forgetting rates.
MEMORY_HALFLIFE_KEY = "cortex_memory_halflife_days"
HALFLIFE_BOUNDS = (0.5, 365.0)

OUTCOMES = ("routed", "ambiguous", "abstain", "applied", "approved", "rejected", "clarified", "undone")


# ---------------------------------------------------------------------------
# Episode shape (plain dicts in the persistent state, kept in order).
# ---------------------------------------------------------------------------


def new_episode(text: str, resolved: str, surfaces: Sequence[str], verdict: str,
                outcome: str, at: Optional[datetime] = None) -> Dict[str, object]:
    """One recorded interaction. ``at`` defaults to a fixed epoch for
    determinism in tests; the service layer stamps the real clock."""
    return {
        "text": text,
        "resolved": resolved,
        "surfaces": list(surfaces),
        "verdict": verdict,
        "outcome": outcome,
        "at": (at or datetime(2026, 1, 1, 12, 0, 0)).isoformat(),
    }


def _episode_time(episode: Dict[str, object]) -> datetime:
    raw = episode.get("at")
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime(2026, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# Store operations (pure functions over the episode list).
# ---------------------------------------------------------------------------


def record(episodes: List[Dict[str, object]], episode: Dict[str, object],
           cap: int = MAX_EPISODES) -> List[Dict[str, object]]:
    """Append and bound. Returns the new list (caller persists it)."""
    out = list(episodes)
    out.append(episode)
    if len(out) > cap:
        out = out[-cap:]
    return out


def _decay_weight(episode: Dict[str, object], now: datetime,
                  halflife_days: float = HALFLIFE_DAYS) -> float:
    age_days = max(0.0, (now - _episode_time(episode)).total_seconds() / 86400.0)
    return 0.5 ** (age_days / max(1e-9, halflife_days))


def resolve_halflife(state: Optional[Dict[str, object]] = None) -> float:
    """The effective half-life: the state's user setting when present and
    in bounds, else the shipped default. Garbage values degrade to the
    default (never a crash, never a negative decay)."""
    raw = (state or {}).get(MEMORY_HALFLIFE_KEY)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return HALFLIFE_DAYS
    lo, hi = HALFLIFE_BOUNDS
    if not (lo <= value <= hi):
        return HALFLIFE_DAYS
    return value


def recall(episodes: Sequence[Dict[str, object]], query: str = "",
           now: Optional[datetime] = None, k: int = 10,
           halflife_days: float = HALFLIFE_DAYS) -> List[Dict[str, object]]:
    """Decay-weighted recall of episodes matching ``query`` (surface-name
    or text-substring match; empty query = recent-everything)."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    lowered = query.lower().strip()
    scored: List[Tuple[float, Dict[str, object]]] = []
    for episode in episodes:
        if lowered:
            surfaces = " ".join(str(s) for s in episode.get("surfaces", []))
            text = str(episode.get("text", "")) + " " + str(episode.get("resolved", ""))
            if lowered not in surfaces.lower() and lowered not in text.lower():
                continue
        scored.append((_decay_weight(episode, now, halflife_days), episode))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("at", ""))))
    return [episode for _w, episode in scored[:k]]


def cooccurrence(episodes: Sequence[Dict[str, object]],
                 outcome_filter: Sequence[str] = ("applied", "approved"),
                 min_count: int = 2) -> Dict[Tuple[str, str], int]:
    """Co-change counts between surfaces within single applied episodes."""
    counts: Dict[Tuple[str, str], int] = {}
    for episode in episodes:
        if episode.get("outcome") not in outcome_filter:
            continue
        surfaces = sorted(set(str(s) for s in episode.get("surfaces", [])))
        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                pair = (surfaces[i], surfaces[j])
                counts[pair] = counts.get(pair, 0) + 1
    return {pair: n for pair, n in counts.items() if n >= min_count}


def lift(coocc: Dict[Tuple[str, str], int],
         surface_counts: Dict[str, int], total: int) -> List[Dict[str, object]]:
    """Market-basket lift over co-change counts: for each pair,
    ``lift = P(a,b) / (P(a) * P(b))`` — pairs lifted above 1 co-occur
    more than independence predicts. Sorted descending, top pairs only."""
    out: List[Dict[str, object]] = []
    if total <= 0:
        return out
    for (a, b), count in coocc.items():
        pa = surface_counts.get(a, 0) / total
        pb = surface_counts.get(b, 0) / total
        if pa <= 0 or pb <= 0:
            continue
        pab = count / total
        value = pab / (pa * pb)
        out.append({"a": a, "b": b, "count": count, "lift": round(value, 3)})
    out.sort(key=lambda row: (-row["lift"], row["a"], row["b"]))
    return out


def followup_suggestion(episodes: Sequence[Dict[str, object]], just_applied: str,
                        now: Optional[datetime] = None, min_count: int = 2,
                        top: int = 3,
                        halflife_days: float = HALFLIFE_DAYS
                        ) -> List[Dict[str, object]]:
    """The proactive second-brain suggestion: after ``just_applied``,
    which settings historically co-changed with it? Decay-weighted,
    lift-ranked, and returned as SUGGESTION DATA ONLY — the caller turns
    them into ledger proposals (never auto-applies)."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    weighted: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for episode in episodes:
        if episode.get("outcome") not in ("applied", "approved"):
            continue
        surfaces = set(str(s) for s in episode.get("surfaces", []))
        if just_applied not in surfaces:
            continue
        weight = _decay_weight(episode, now, halflife_days)
        for surface in surfaces:
            if surface == just_applied:
                continue
            weighted[surface] = weighted.get(surface, 0.0) + weight
            counts[surface] = counts.get(surface, 0) + 1
    ranked = [
        {"surface": surface, "weight": round(weighted[surface], 3), "count": counts[surface]}
        for surface in weighted if counts[surface] >= min_count
    ]
    ranked.sort(key=lambda row: (-row["weight"], row["surface"]))
    return ranked[:top]


def habits(episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Hour-of-day and day-of-week histograms of applied episodes —
    when this user actually customizes their shell."""
    hours = [0] * 24
    weekdays = [0] * 7
    for episode in episodes:
        if episode.get("outcome") not in ("applied", "approved"):
            continue
        when = _episode_time(episode)
        hours[when.hour] += 1
        weekdays[when.weekday()] += 1
    return {"hours": hours, "weekdays": weekdays, "episodes": len(episodes)}


def surface_counts(episodes: Sequence[Dict[str, object]],
                   outcome_filter: Sequence[str] = ("applied", "approved")) -> Dict[str, int]:
    """Frequency count of each surface across applied episodes."""
    counts: Dict[str, int] = {}
    for episode in episodes:
        if episode.get("outcome") not in outcome_filter:
            continue
        for surface in set(str(s) for s in episode.get("surfaces", [])):
            counts[surface] = counts.get(surface, 0) + 1
    return counts


def summarize(episodes: Sequence[Dict[str, object]], now: Optional[datetime] = None) -> Dict[str, object]:
    """One-line memory report for the CLI/bridge: totals, top surfaces,
    top co-changes, habit peaks."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    counts = surface_counts(episodes)
    top_surfaces = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    total_applied = sum(counts.values())
    co = cooccurrence(episodes)
    lifted = lift(co, counts, max(1, len([e for e in episodes if e.get("outcome") in ("applied", "approved")])))
    habit = habits(episodes)
    peak_hour = max(range(24), key=lambda h: habit["hours"][h]) if any(habit["hours"]) else None
    return {
        "episodes": len(episodes),
        "applied_episodes": total_applied,
        "top_surfaces": [{"surface": s, "count": n} for s, n in top_surfaces],
        "top_cochanges": lifted[:5],
        "peak_hour": peak_hour,
    }
