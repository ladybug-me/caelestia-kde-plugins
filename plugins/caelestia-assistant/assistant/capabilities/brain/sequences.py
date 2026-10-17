"""brain.sequences — PrefixSpan sequential-pattern mining over session
records, for habitual launch sequences.

Pei, Han, Mortazavi-Asl, Pinto, Chen, Dayal, Hsu 2001, "PrefixSpan:
Mining Sequential Patterns by Prefix-Projected Growth", ICDE 2001 —
frequent sequential patterns by recursively extending a prefix and
re-reading only its PROJECTED DATABASE (the suffixes following the
prefix's last item), which avoids the candidate generation of Apriori-
style approaches entirely. The projection here is the paper's for
SINGLE-ITEM events (each event is one item — one app launch); itemset
events (multiple simultaneous items per event) are deliberately out of
scope and stated: launch sequences are the only sequence data this
assistant has, and every event in them is a single app.

DATA SOURCE (the honest precondition): the shell persists no session
log (verified upstream, see brain/workspace.py's data-source note), so
this module consumes EXPLICITLY SUPPLIED session records — the same
caller-supplied-series pattern as workspace.py, rhythm.py and
forecast.py. Schema: {"date": "YYYY-MM-DD", "hour": 0-23, "app": str}.
Sessions group into one sequence per DATE, ordered by (hour, app), and
PrefixSpan mines the subsequences that appear in at least min_support
days' sequences — "you usually launch a terminal after an editor" is
the shape of answer this returns.

Consent discipline: mining is read-only. The only action surface is
propose_patterns, which files the top patterns as ledger proposals of
kind "launch_pattern" (approve/reject only — the existing vocabulary,
no fourth mechanism); nothing here writes, executes, or networks.

Deterministic: items are extended in sorted order, projection keeps
first-occurrence suffixes, and the ranked output sorts by
(-support, pattern) — same input, same output bytes (pinned by test).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["prefixspan", "day_sequences", "mine_launch_patterns",
           "propose_patterns"]

DEFAULT_MAX_LEN = 4  # a habitual launch sequence longer than 4 is noise


def prefixspan(sequences: Sequence[Sequence[Any]], min_support: int = 2,
               max_len: int = DEFAULT_MAX_LEN
               ) -> Dict[Tuple[Any, ...], int]:
    """Frequent sequential patterns over single-item-event sequences,
    by prefix-projected growth (Pei et al. 2001).

    Returns {pattern_tuple: support} for every pattern with
    support >= min_support, where support counts SEQUENCES containing
    the pattern as an ordered subsequence (the standard sequence-
    support definition — one occurrence per sequence, repeats included
    via projection). Refuses min_support < 1 and max_len < 1 (a budget
    of 0 patterns is a caller error, not an honest empty result)."""
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1 (got {min_support})")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1 (got {max_len})")
    db = [tuple(seq) for seq in sequences]
    if not db:
        return {}
    results: Dict[Tuple[Any, ...], int] = {}

    def mine(projected: List[Tuple[Any, ...]],
             prefix: Tuple[Any, ...]) -> None:
        if len(prefix) >= max_len:
            return
        # items frequent in the projected database (each distinct item
        # counted once per sequence — the sequence-support definition)
        counts: Dict[Any, int] = {}
        for seq in projected:
            for item in set(seq):
                counts[item] = counts.get(item, 0) + 1
        for item in sorted(counts, key=lambda x: (str(type(x)), str(x))):
            if counts[item] < min_support:
                continue
            pattern = prefix + (item,)
            results[pattern] = counts[item]
            # project: keep the suffix AFTER the item's first
            # occurrence (the paper's projection — first-occurrence
            # projection preserves completeness for support counting)
            suffixes: List[Tuple[Any, ...]] = []
            for seq in projected:
                try:
                    pos = seq.index(item)
                except ValueError:
                    continue
                if pos + 1 < len(seq):
                    suffixes.append(seq[pos + 1:])
            if suffixes:
                mine(suffixes, pattern)

    mine(db, ())
    return results


def day_sequences(sessions: Sequence[Dict[str, Any]]
                  ) -> List[Tuple[str, ...]]:
    """Session records -> one app sequence per date, ordered by
    (hour, app) — deterministic for any input order. Malformed records
    (missing app/date, non-integer hour, empty app) are refused with
    the reason, never silently dropped: a partial sequence would
   understate support and every pattern's support is a claim."""
    by_day: Dict[str, List[Tuple[int, str]]] = {}
    for i, s in enumerate(sessions):
        app = s.get("app")
        date = s.get("date")
        if not isinstance(app, str) or not app:
            raise ValueError(
                f"session #{i} has no app (got {app!r}) — refusing "
                "rather than mining a partial sequence")
        if not isinstance(date, str) or not date:
            raise ValueError(
                f"session #{i} has no date (got {date!r}) — refusing "
                "rather than mining a partial sequence")
        try:
            hour = int(s.get("hour", 0))
        except (TypeError, ValueError):
            raise ValueError(
                f"session #{i} has a non-integer hour "
                f"({s.get('hour')!r})") from None
        if not 0 <= hour <= 23:
            raise ValueError(
                f"session #{i} hour {hour} outside 0..23 — refusing "
                "rather than clamping")
        by_day.setdefault(date, []).append((hour, app))
    return [tuple(app for _hour, app in sorted(rows))
            for _date, rows in sorted(by_day.items())]


def mine_launch_patterns(sessions: Sequence[Dict[str, Any]],
                         min_support: int = 2,
                         max_len: int = DEFAULT_MAX_LEN,
                         top: int = 8) -> Dict[str, Any]:
    """The launch-pattern report over caller-supplied session records:
    the top frequent sequential patterns with their support (days) and
    share (support / days). Read-only."""
    seqs = day_sequences(sessions)
    if not seqs:
        return {"days": 0, "patterns": [], "min_support": min_support}
    patterns = prefixspan(seqs, min_support=min_support, max_len=max_len)
    ranked = sorted(patterns.items(),
                    key=lambda kv: (-kv[1], kv[0]))[:top]
    return {
        "days": len(seqs),
        "min_support": min_support,
        "patterns": [{"pattern": list(p), "support": s,
                      "share": round(s / len(seqs), 4)}
                     for p, s in ranked],
    }


def propose_patterns(report: Dict[str, Any], ledger: Any,
                     top: int = 3) -> int:
    """File the report's top patterns as launch_pattern ledger
    proposals (the existing approve/reject consent surface; nothing is
    ever auto-applied). Returns the number filed."""
    filed = 0
    for row in report.get("patterns", [])[:top]:
        ledger.propose(
            "launch_pattern",
            " -> ".join(row["pattern"]),
            {"sequence": list(row["pattern"])},
            f"frequent launch sequence ({row['support']}/"
            f"{report.get('days', '?')} days)",
            float(row.get("share", 0.0)))
        filed += 1
    return filed
