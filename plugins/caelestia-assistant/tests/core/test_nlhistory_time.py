"""F10 tests — NL time-expression restore (issue #120 b4b).

Extends cortex/nlhistory's verified grammar:

- richer time windows: N minutes ago, day-part sub-windows ("yesterday
  morning" is NOT all of yesterday; "last night" spans midnight),
  weekday names ("restore what I changed on Friday");
- the RESTORE-SINCE reading: a window with no domain scope names a
  point to restore TO — "restore yesterday's theme" reverts every
  change applied since then via the EXISTING undo(steps=K) engine,
  exactly, because the since-set is the newest-first ring's contiguous
  prefix (apply order == storage order);
- honesty: clock-skewed rings (non-monotonic timestamps) fall back to
  undo_by_id on the newest since-entry, named; nothing-since is
  NOT_FOUND, never a silent no-op; domain-scoped queries keep the
  best-entry undo_by_id reading.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from assistant.core.nlhistory import parse_query, plan as plan_history


def _entry(eid, when, path="bar.scale", label="chat: x"):
    return {"id": eid, "at": when.isoformat(timespec="seconds"),
            "label": label,
            "ops": [{"path": path, "old": 0.9, "new": 1.0}]}


class TimeGrammarTests(unittest.TestCase):
    def test_minutes_ago(self) -> None:
        now = datetime(2026, 9, 28, 15, 0)
        q = parse_query("undo the change 20 minutes ago", now=now)
        lo, hi = q.window
        self.assertEqual((hi - lo).total_seconds(), 20 * 60)

    def test_yesterday_morning_is_subwindow(self) -> None:
        now = datetime(2026, 9, 28, 15, 0)
        q = parse_query("restore the changes from yesterday morning", now=now)
        lo, hi = q.window
        y = (now - timedelta(days=1)).date()
        self.assertEqual(lo, datetime(y.year, y.month, y.day, 5, 0))
        self.assertEqual(hi, datetime(y.year, y.month, y.day, 12, 0))

    def test_this_afternoon_capped_at_now(self) -> None:
        now = datetime(2026, 9, 28, 15, 0)
        q = parse_query("what changed this afternoon", now=now)
        lo, hi = q.window
        self.assertEqual(lo.hour, 12)
        self.assertEqual(hi, now)

    def test_last_night_spans_midnight(self) -> None:
        now = datetime(2026, 9, 28, 9, 0)
        q = parse_query("undo what I changed last night", now=now)
        lo, hi = q.window
        self.assertEqual(lo.date(), datetime(2026, 9, 27).date())
        self.assertEqual(lo.hour, 21)
        self.assertEqual(hi.date(), datetime(2026, 9, 28).date())
        self.assertEqual(hi.hour, 5)

    def test_weekday_window(self) -> None:
        now = datetime(2026, 9, 28, 15, 0)  # a Monday
        self.assertEqual(now.weekday(), 0)
        q = parse_query("restore what I changed on friday", now=now)
        lo, hi = q.window
        self.assertEqual(lo.date(), datetime(2026, 9, 25).date())
        self.assertEqual(hi.date(), datetime(2026, 9, 26).date())

    def test_yesterday_still_full_day(self) -> None:
        now = datetime(2026, 9, 28, 15, 0)
        q = parse_query("restore yesterday's theme", now=now)
        lo, hi = q.window
        self.assertEqual(hi - lo, timedelta(days=1))
        self.assertEqual(hi, datetime(2026, 9, 28, 0, 0))


class RestoreSinceTests(unittest.TestCase):
    NOW = datetime(2026, 9, 28, 15, 0)

    def test_restore_since_yesterday_uses_undo_steps(self) -> None:
        entries = [
            _entry(4, self.NOW - timedelta(hours=1)),
            _entry(3, self.NOW - timedelta(hours=2)),
            _entry(2, self.NOW - timedelta(days=2)),   # before the window
            _entry(1, self.NOW - timedelta(days=3)),
        ]
        q = parse_query("restore yesterday's theme", now=self.NOW)
        result = plan_history(q, entries, now=self.NOW)
        self.assertEqual(result.verdict, "UNDO")
        self.assertEqual(result.action, "undo")
        self.assertEqual(result.steps, 2)
        self.assertIn("restores the state as of", result.reason)
        # oldest reverted first: the prefix preserves ring order
        self.assertEqual([e["id"] for e in result.entries], [4, 3])

    def test_nothing_since_is_not_found(self) -> None:
        entries = [_entry(1, self.NOW - timedelta(days=5))]
        q = parse_query("restore yesterday's theme", now=self.NOW)
        result = plan_history(q, entries, now=self.NOW)
        self.assertEqual(result.verdict, "NOT_FOUND")
        self.assertIsNone(result.action)
        self.assertIn("nothing was applied since", result.reason)

    def test_clock_skew_falls_back_to_newest_by_id(self) -> None:
        # ring order (newest-first) with NON-monotonic timestamps
        entries = [
            _entry(3, self.NOW - timedelta(days=2)),   # older, stored first
            _entry(2, self.NOW - timedelta(hours=1)),  # newer, stored second
        ]
        q = parse_query("restore yesterday's theme", now=self.NOW)
        result = plan_history(q, entries, now=self.NOW)
        self.assertEqual(result.verdict, "UNDO")
        self.assertEqual(result.action, "undo_by_id")
        self.assertEqual(result.entry_id, 2)
        self.assertIn("clock skew", result.reason)

    def test_scoped_window_keeps_best_entry_reading(self) -> None:
        now = datetime(2026, 9, 28, 11, 0)  # mid-morning
        entries = [
            _entry(2, now - timedelta(hours=1), path="bar.scale"),
            _entry(1, now - timedelta(hours=2), path="notifs.maxPopups"),
        ]
        q = parse_query("undo the bar changes from this morning", now=now)
        result = plan_history(q, entries, now=now)
        self.assertEqual(result.action, "undo_by_id")
        self.assertEqual(result.entry_id, 2)

    def test_empty_history_is_empty(self) -> None:
        q = parse_query("restore yesterday's theme", now=self.NOW)
        result = plan_history(q, [], now=self.NOW)
        self.assertEqual(result.verdict, "EMPTY")


if __name__ == "__main__":
    unittest.main()
