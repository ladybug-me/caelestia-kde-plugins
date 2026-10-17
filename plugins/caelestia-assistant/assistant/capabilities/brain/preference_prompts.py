"""brain.preference_prompts — idle-time paired-comparison preference
elicitation (exponential-build-3 G2), riding the dreamtime idle
cadence with a hard frequency cap.

The design in one paragraph: the assistant already learns preset
preferences, but ONLY from comparisons the user thinks to run
(`settings --prefer A B`). The Elo/Bradley-Terry ladder is only as
good as the pairs the user bothers to compare — and the user does not
know WHICH comparison would be most informative. During idle windows
(the dreamtime cadence: eligible() decides whether a window may open
at all), the assistant may invite ONE pairwise comparison, chosen
where the answer buys the most ranking information: the pair with the
FEWEST recorded comparisons, tie-broken by the CLOSEST Elo ratings
(the most uncertain ordering — the classic active-learning choice for
a Bradley-Terry ladder). The invitation is a prompt RECORD, never an
interruption: it surfaces in `settings --rank` and is answered by the
ordinary `settings --prefer A B`, which consumes it when the pair
matches.

FREQUENCY CAP (the master constraint, enforced structurally):
- at most ONE unanswered prompt exists at any time;
- at least MIN_DAYS_BETWEEN (7) days between prompts, tracked in the
  state's prompt history — a prompt is generated only when the cap
  allows, and the reason is returned when it does not;
- prompts EXPIRE (MAX_AGE_DAYS, 14): an unanswered invitation older
  than that is withdrawn (moved to history as 'expired') rather than
  nagging forever.

Pair selection is deterministic: (fewest comparisons, smallest Elo
rating difference, sorted pair) — same state, same prompt (pinned by
test). Pure module: no I/O, no RNG, no clock reads (``now`` is
caller-supplied, the fixture-injection pattern); the only writer is
the caller persisting the returned state fragment through the
existing brain-state path.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["MIN_DAYS_BETWEEN", "MAX_AGE_DAYS", "pending", "maybe_prompt",
           "answer"]

MIN_DAYS_BETWEEN = 7.0   # days between invitations, hard cap
MAX_AGE_DAYS = 14.0      # an unanswered invitation expires after this

PROMPT_KEY = "compare_prompt"
HISTORY_KEY = "compare_prompt_history"


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def pending(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The unanswered prompt record, or None. Does not apply expiry
    (expiry happens in maybe_prompt, which can move records)."""
    prompt = state.get(PROMPT_KEY)
    return prompt if isinstance(prompt, dict) else None


def _ladder_ratings(pairs: List[Tuple[str, str]]
                    ) -> Dict[str, float]:
    """Elo ratings for pair-selection closeness, reusing the shared
    ranking primitive (the same one --rank renders)."""
    try:
        from .ranking import ladder_report
        report = ladder_report(pairs)
        return {row["item"]: float(row["rating"])
                for row in report.get("ladder", [])}
    except Exception:
        return {}  # selection falls back to comparison counts alone


def _choose_pair(preset_names: List[str],
                 comparisons: List[Dict[str, Any]],
                 ) -> Optional[Tuple[str, str, int, float]]:
    """The most informative pair: fewest recorded comparisons, then
    closest Elo ratings, then sorted order. Returns (a, b, count,
    rating_diff) or None when fewer than two presets exist."""
    names = sorted(set(str(n) for n in preset_names))
    if len(names) < 2:
        return None
    counts: Dict[Tuple[str, str], int] = {}
    for row in comparisons:
        a, b = str(row.get("winner")), str(row.get("loser"))
        key = tuple(sorted((a, b)))
        counts[key] = counts.get(key, 0) + 1
    ratings = _ladder_ratings(
        [(str(r.get("winner")), str(r.get("loser")))
         for r in comparisons])
    best: Optional[Tuple[str, str, int, float]] = None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = (a, b)
            count = counts.get(key, 0)
            diff = abs(ratings.get(a, 1000.0) - ratings.get(b, 1000.0))
            if best is None:
                best = (a, b, count, diff)
                continue
            if (count, diff, (a, b)) < (best[2], best[3], (best[0], best[1])):
                best = (a, b, count, diff)
    return best


def maybe_prompt(state: Dict[str, Any], now: datetime,
                 preset_names: Optional[List[str]] = None
                 ) -> Dict[str, Any]:
    """The idle-cadence entry point: MAYBE record one comparison
    invitation in the state fragment (the caller persists it through
    the existing brain-state path). Pure — takes the whole state,
    returns {"prompt": record|None, "reason": str, "state": state}
    with the state mutated in place (prompt recorded, expired prompts
    moved to history, history bounded to the last 50).

    ``preset_names`` defaults to the settings layer's shipped preset
    list (the same names --prefer validates)."""
    # expiry first: an old invitation is withdrawn, not nagged
    current = pending(state)
    if current is not None:
        created = _parse_iso(current.get("at") or "")
        if created is not None and \
                now - created > timedelta(days=MAX_AGE_DAYS):
            history = state.setdefault(HISTORY_KEY, [])
            history.append({**current, "outcome": "expired"})
            state[HISTORY_KEY] = history[-50:]
            state.pop(PROMPT_KEY, None)
            current = None
    if current is not None:
        return {"prompt": None,
                "reason": "an unanswered comparison prompt is already "
                          f"pending ({current.get('a')} vs "
                          f"{current.get('b')}) — answer it with "
                          "settings --prefer, or let it expire",
                "state": state}
    history = state.get(HISTORY_KEY) or []
    if history:
        last = _parse_iso(history[-1].get("at") or "")
        if last is not None and \
                now - last < timedelta(days=MIN_DAYS_BETWEEN):
            days = (now - last).total_seconds() / 86400.0
            return {"prompt": None,
                    "reason": f"frequency cap: only "
                              f"{days:.1f} of the required "
                              f"{MIN_DAYS_BETWEEN:g} days since the "
                              "last prompt",
                    "state": state}
    if preset_names is None:
        from assistant.capabilities.settings.presets import presets as shipped_presets
        preset_names = [str(p["name"]) for p in shipped_presets()]
    chosen = _choose_pair(preset_names,
                          state.get("preset_comparisons") or [])
    if chosen is None:
        return {"prompt": None,
                "reason": "fewer than two presets exist — nothing to "
                          "compare",
                "state": state}
    a, b, count, diff = chosen
    record = {
        "a": a, "b": b,
        "at": now.isoformat(timespec="seconds"),
        "reason": f"fewest recorded comparisons ({count}) and closest "
                  f"Elo ratings (difference {diff:.1f}) — this answer "
                  "buys the most ranking information",
    }
    state[PROMPT_KEY] = record
    return {"prompt": record, "reason": "invited", "state": state}


def answer(state: Dict[str, Any], a: str, b: str) -> bool:
    """Consume the pending prompt when the recorded comparison answers
    it (either order). Moves it to history with outcome 'answered'.
    Returns whether a prompt was consumed. Pure — the caller persists
    the state through the existing path."""
    current = pending(state)
    if current is None:
        return False
    pair = {str(current.get("a")), str(current.get("b"))}
    if {str(a), str(b)} != pair:
        return False
    history = state.setdefault(HISTORY_KEY, [])
    history.append({**current, "outcome": "answered"})
    state[HISTORY_KEY] = history[-50:]
    state.pop(PROMPT_KEY, None)
    return True
