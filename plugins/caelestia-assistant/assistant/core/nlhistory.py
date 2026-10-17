"""Natural-language undo/history queries (issue #120's "Undo Support").

The existing ``settings/history.py`` answers ``undo(steps)`` and
``undo_by_id(id)`` — numeric addressing only. This module is the
language half: it turns phrases like these into concrete, honest
resolutions against the real history ring:

- "undo the last change"                     -> undo(steps=1)
- "undo my last two changes"                 -> undo(steps=2)
- "undo the bar changes"                     -> undo_by_id(<best entry>)
- "restore yesterday's theme"                -> undo_by_id(<best entry>)
- "revert my last customization"             -> undo(steps=1)
- "what did i change" / "show my history"    -> read-only listing

Grammar (deterministic, pure): a steps-count parser (number words 1-12
plus digits), a time-window parser (today / yesterday / this morning /
this week / "N hours|days ago" — evaluated against a caller-supplied
``now`` so the module stays clock-free and testable), and a scope filter
(the router's own tool vocabulary matched against each history entry's
op paths — "the bar changes" matches entries touching ``bar.*`` paths).

Selection rule (honest, never a guess): candidates are scored by
recency + scope match + label match; a query scoped to a domain that has
NO matching entries returns ``NOT_FOUND`` with the actual history listed,
not a silent fallback to undo-the-latest. When several entries tie, the
NEWEST wins (the reading of "revert my last customization" that undo
semantics implies) and the runner-up margin is reported.

PURE module: no file reads — the caller passes ``entries`` (as produced
by ``settings/history.entries(target)``); this module only plans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Grammar.
# ---------------------------------------------------------------------------

_NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "couple": 2, "few": 3,
}

_STEPS_RE = re.compile(r"\b(?:last|previous|past|recent)\s+(?:my\s+)?(\d+|"
                       + "|".join(_NUMBER_WORDS) + r")\s+(?:changes?|edits?|applies?|customizations?)\b")
_STEPS_LAST_RE = re.compile(r"\b(?:last|latest|most recent|previous)\b")
_UNDO_VERB_RE = re.compile(r"\b(?:undo|revert|roll\s?back|restore|take back)\b")

_HOURS_AGO_RE = re.compile(r"\b(\d+|a|an)\s+hours?\s+ago\b")
_MINUTES_AGO_RE = re.compile(r"\b(\d+|a|an)\s+minutes?\s+ago\b")
_DAYS_AGO_RE = re.compile(r"\b(\d+|a|an)\s+days?\s+ago\b")
_TODAY_RE = re.compile(r"\b(?:today|this morning|this afternoon|this evening|tonight)\b")
_YESTERDAY_RE = re.compile(r"\b(?:yesterday|last night)\b")
_THIS_WEEK_RE = re.compile(r"\b(?:this week|past week|last week)\b")

# F10: day-part sub-windows ("yesterday morning" is NOT all of yesterday).
# night deliberately spans midnight (21:00 -> next-day 05:00).
_DAYPART_RE = re.compile(
    r"\b(?:(yesterday|this|last)\s+)?(morning|afternoon|evening|night)\b")
_DAYPART_HOURS = {
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 29),  # 29-24 == next-day 05:00
}
_WEEKDAY_RE = re.compile(
    r"\b(?:on\s+|last\s+)?(monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b")
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}

_THEME_RE = re.compile(r"\b(?:theme|look|preset|customization|setup)\b")
_LISTING_RE = re.compile(r"\b(?:what did i change|show (?:my )?(?:change |edit )?history|"
                         r"(?:list|show) (?:my )?(?:recent )?changes|change history|my changes)\b")


# ---------------------------------------------------------------------------
# Result shapes.
# ---------------------------------------------------------------------------


@dataclass
class HistoryQuery:
    """The parsed shape of one NL history request (pure data)."""

    kind: str                      # "undo_steps" | "undo_entry" | "list"
    steps: int = 1                 # undo_steps only
    window: Optional[Tuple[datetime, datetime]] = None  # undo_entry/list filter
    scope_words: List[str] = field(default_factory=list)  # domain words ("bar")
    theme: bool = False            # "restore yesterday's theme"


@dataclass
class HistoryPlan:
    """The concrete resolution of a query against real history entries."""

    verdict: str                   # UNDO | LIST | NOT_FOUND | EMPTY
    action: Optional[str] = None   # "undo" | "undo_by_id" | None
    steps: int = 0
    entry_id: Optional[int] = None
    entries: List[Dict[str, object]] = field(default_factory=list)   # relevant entries
    total_entries: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Parsing (pure).
# ---------------------------------------------------------------------------


def parse_query(text: str, now: Optional[datetime] = None) -> HistoryQuery:
    """Parse one NL history request. ``now`` anchors relative windows
    (default: a fixed epoch so parsing is deterministic and testable —
    callers pass the real clock; the module never reads one itself)."""
    lowered = (text or "").lower()
    now = now or datetime(2026, 1, 1, 12, 0, 0)

    window: Optional[Tuple[datetime, datetime]] = None
    minutes = _MINUTES_AGO_RE.search(lowered)
    hours = _HOURS_AGO_RE.search(lowered)
    days = _DAYS_AGO_RE.search(lowered)
    daypart = _DAYPART_RE.search(lowered)
    weekday = _WEEKDAY_RE.search(lowered)
    if minutes:
        n = _word_num(minutes.group(1), 1)
        window = (now - timedelta(minutes=n), now)
    elif hours:
        n = _word_num(hours.group(1), 1)
        window = (now - timedelta(hours=n), now)
    elif days:
        n = _word_num(days.group(1), 1)
        window = (now - timedelta(days=n), now)
    elif daypart:
        base_word, part = daypart.group(1), daypart.group(2)
        start_h, end_h = _DAYPART_HOURS[part]
        if base_word == "yesterday":
            base = now - timedelta(days=1)
        else:
            base = now
        if part == "night" and base_word == "last":
            # "last night" names YESTERDAY's night: 21:00 -> today 05:00
            lo = base.replace(hour=0, minute=0, second=0, microsecond=0) \
                + timedelta(hours=start_h) - timedelta(days=1)
            hi = base.replace(hour=0, minute=0, second=0, microsecond=0) \
                + timedelta(hours=end_h - 24)
        else:
            lo = base.replace(hour=0, minute=0, second=0, microsecond=0) \
                + timedelta(hours=start_h)
            hi = base.replace(hour=0, minute=0, second=0, microsecond=0) \
                + timedelta(hours=end_h)
            hi = min(hi, now)
        window = (lo, hi)
    elif _YESTERDAY_RE.search(lowered):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        window = (start, start + timedelta(days=1))
    elif _TODAY_RE.search(lowered):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        window = (start, now)
    elif weekday:
        target = _WEEKDAYS[weekday.group(1)]
        back = (now.weekday() - target) % 7  # 0 == today
        day = (now - timedelta(days=back)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        window = (day, min(day + timedelta(days=1), now))
    elif _THIS_WEEK_RE.search(lowered):
        window = (now - timedelta(days=7), now)

    steps_match = _STEPS_RE.search(lowered)
    steps = 1
    if steps_match:
        steps = max(1, min(12, _word_num(steps_match.group(1), 1)))
    elif _STEPS_LAST_RE.search(lowered):
        steps = 1

    # Domain scope: straight from the phrase's own words, mapped through
    # the registry's real domains (EVERY tool-path segment, so "spacing"
    # and "greeter" work, not just top-level domains). "undo the bar
    # changes" scopes to bar.*; a generic undo phrase scopes to nothing.
    scope_words: List[str] = []
    from assistant.adapters.caelestia.registry import TOOL_SPECS
    from .lexicon import SYNONYMS, stem

    domains: set = set()
    for spec in TOOL_SPECS:
        if spec.path:
            domains.update(piece.lower() for piece in spec.path.split("."))
    for word in re.findall(r"[a-z]+", lowered):
        candidates = {word, stem(word)}
        for key in (word, stem(word)):
            for mapped in SYNONYMS.get(key, ()):
                candidates.add(mapped)
        for mapped in candidates:
            if mapped in domains and mapped not in scope_words:
                scope_words.append(mapped)

    return HistoryQuery(
        kind="list" if _LISTING_RE.search(lowered) else "undo",
        steps=steps,
        window=window,
        scope_words=scope_words,
        theme=bool(_THEME_RE.search(lowered)),
    )


def _word_num(token: str, default: int) -> int:
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token, default)


# ---------------------------------------------------------------------------
# Resolution against real entries (pure planning, no I/O).
# ---------------------------------------------------------------------------


def _entry_time(entry: Dict[str, object]) -> Optional[datetime]:
    raw = entry.get("at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _entry_scope_hit(entry: Dict[str, object], scope_words: Sequence[str]) -> bool:
    ops = entry.get("ops") or []
    if not scope_words:
        return True
    for op in ops:
        path = str(op.get("path", ""))
        if any(path == word or path.startswith(word + ".") or word in path for word in scope_words):
            return True
    return False


def _entry_theme_hit(entry: Dict[str, object]) -> bool:
    label = str(entry.get("label", "")).lower()
    ops = entry.get("ops") or []
    if "preset" in label or "theme" in label:
        return True
    return len(ops) >= 3  # a multi-op apply is "a customization" in #120's sense


def plan(query: HistoryQuery, entries: Sequence[Dict[str, object]],
         now: Optional[datetime] = None) -> HistoryPlan:
    """Resolve a parsed query against real history entries.

    Selection: window-filtered, scope-filtered entries scored by recency
    (newest first — entries are stored newest-first by history.py, and
    ties break toward the newer). A scoped/theme query with no matching
    entry is NOT_FOUND (with the full history offered for listing), never
    a silent fallback to undo-the-latest."""
    now = now or datetime(2026, 1, 1, 12, 0, 0)
    total = len(entries)
    if query.kind == "list":
        return HistoryPlan(
            verdict="LIST" if total else "EMPTY",
            action=None, entries=list(entries), total_entries=total,
            reason="read-only history listing",
        )

    if not total:
        return HistoryPlan(
            verdict="EMPTY", action=None, total_entries=0,
            reason="the undo history is empty (nothing has been applied yet)",
        )

    if not query.window and not query.scope_words and not query.theme:
        steps = min(query.steps, total)
        return HistoryPlan(
            verdict="UNDO", action="undo", steps=steps,
            entries=list(entries[:steps]), total_entries=total,
            reason=f"undo the last {steps} change{'s' if steps != 1 else ''} (history holds {total})",
        )

    # F10 restore-since reading: a WINDOW (no domain scope) names a point
    # in time to restore TO, not a single entry to revert. "restore
    # yesterday's theme" = put the config back the way it was at the end
    # of yesterday = revert every change applied SINCE then. Those
    # entries are the newest-first history's contiguous prefix (apply
    # order is storage order), so the EXISTING undo(steps=K) engine does
    # the restore exactly; a clock-skewed ring (non-monotonic times)
    # falls back to the newest matching entry, named honestly.
    if query.window and not query.scope_words:
        hi = query.window[1]
        since_prefix: List[Dict[str, object]] = []
        since_all: List[Dict[str, object]] = []
        contiguous = True
        for entry in entries:
            when = _entry_time(entry)
            if when is None:
                contiguous = False  # unparseable timestamp: not provably in order
                continue
            if when > hi:
                since_all.append(entry)
                if contiguous:
                    since_prefix.append(entry)
            else:
                contiguous = False
        if since_all and len(since_all) == len(since_prefix):
            return HistoryPlan(
                verdict="UNDO", action="undo", steps=len(since_prefix),
                entries=since_prefix, total_entries=total,
                reason=f"restores the state as of {hi.isoformat(timespec='minutes')}: "
                       f"reverts the {len(since_prefix)} change(s) applied since "
                       f"then, oldest reverted first",
            )
        if since_all:
            best = since_all[0]
            return HistoryPlan(
                verdict="UNDO", action="undo_by_id",
                entry_id=int(best.get("id", 0)),
                entries=since_all, total_entries=total,
                reason="history timestamps are not in apply order (clock "
                       "skew); reverting the newest change since the named "
                       "time — check the full history before more",
            )
        return HistoryPlan(
            verdict="NOT_FOUND", action=None, entries=[], total_entries=total,
            reason=f"nothing was applied since "
                   f"{hi.isoformat(timespec='minutes')} — there is "
                   f"nothing to restore to that point",
        )

    candidates: List[Dict[str, object]] = []
    for entry in entries:
        when = _entry_time(entry)
        if query.window and when is not None:
            lo, hi = query.window
            if not (lo <= when <= hi):
                continue
        if query.scope_words and not _entry_scope_hit(entry, query.scope_words):
            continue
        if query.theme and not _entry_theme_hit(entry):
            continue
        candidates.append(entry)

    if not candidates:
        return HistoryPlan(
            verdict="NOT_FOUND", action=None, entries=[], total_entries=total,
            reason="no history entry matches that scope/window; showing nothing "
                   "rather than undoing something you did not name",
        )

    best = candidates[0]  # newest-first storage order == recency order
    return HistoryPlan(
        verdict="UNDO", action="undo_by_id", entry_id=int(best.get("id", 0)),
        entries=[best], total_entries=total,
        reason="best-matching entry (newest of the scope/window matches)",
    )
