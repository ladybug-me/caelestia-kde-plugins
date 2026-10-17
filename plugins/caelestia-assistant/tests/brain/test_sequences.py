"""Tests for exponential-build-3 E1 — PrefixSpan sequential-pattern
mining over session records (Pei et al. 2001, prefix-projected growth).

Under test:

- the MINING is hand-verified: a 4-sequence fixture whose complete
  frequent-pattern set is derived on paper and pinned exactly (support
  counts included), plus the repeated-item case where projection must
  count (a, a) — the subtlety naive implementations miss;
- projection correctness implies the completeness property the paper
  claims: every pattern above support is found (the pinned set IS the
  full set, so a missing pattern would fail the equality);
- session records group into per-date sequences deterministically
  (any input order), and malformed records are refused with reasons —
  never silently dropped, because a partial sequence understates
  every pattern's support;
- the launch-pattern report ranks by (-support, pattern) and carries
  share = support/days;
- propose_patterns files ledger proposals of kind launch_pattern on
  the EXISTING approve/reject surface (nothing auto-applies).
"""
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.sequences import (day_sequences, mine_launch_patterns,
                                       prefixspan, propose_patterns)


class PrefixSpanTests(unittest.TestCase):
    def test_hand_derived_full_pattern_set(self):
        # Derived on paper. S1=[a,b,c] S2=[a,b] S3=[b,c] S4=[a,c,b],
        # min_support=2:
        #   1-patterns: a:3 (S1,S2,S4)  b:4  c:3 (S1,S3,S4)
        #   2-patterns: (a,b):3 (S1,S2,S4)  (a,c):2 (S1,S4)
        #               (b,c):2 (S1,S3; S4 has no c after its b)
        #   3-patterns: (a,b,c):1 only S1  (a,c,b):1 only S4 -> below
        # The pinned dict IS the complete answer: a missing or extra
        # pattern is a correctness failure, not a tolerance question.
        got = prefixspan([["a", "b", "c"], ["a", "b"], ["b", "c"],
                          ["a", "c", "b"]], min_support=2)
        self.assertEqual(got, {("a",): 3, ("b",): 4, ("c",): 3,
                               ("a", "b"): 3, ("a", "c"): 2,
                               ("b", "c"): 2})

    def test_repeated_items_count_through_projection(self):
        # [[a,a,b],[a,b]] with min_support=1: (a,a) must be found
        # (only the first sequence holds two a's), and the projection
        # after the first a keeps (a,b) so (a,a,b) is found too.
        got = prefixspan([["a", "a", "b"], ["a", "b"]], min_support=1)
        self.assertEqual(got, {("a",): 2, ("b",): 2, ("a", "a"): 1,
                               ("a", "b"): 2, ("a", "a", "b"): 1})

    def test_deterministic_output_order(self):
        seqs = [["b", "a"], ["a", "b"], ["c", "a"]]
        self.assertEqual(list(prefixspan(seqs, 2)),
                         list(prefixspan(seqs, 2)))

    def test_budget_and_refusals(self):
        with self.assertRaises(ValueError) as caught:
            prefixspan([["a"]], min_support=0)
        self.assertIn("min_support must be >= 1", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            prefixspan([["a"]], min_support=1, max_len=0)
        self.assertIn("max_len must be >= 1", str(caught.exception))
        self.assertEqual(prefixspan([], min_support=1), {})
        # max_len=1 stops at 1-patterns (the bound is honored)
        got = prefixspan([["a", "b"]], min_support=1, max_len=1)
        self.assertEqual(got, {("a",): 1, ("b",): 1})


class DaySequenceTests(unittest.TestCase):
    def test_groups_by_date_orders_within_day(self):
        records = [
            {"date": "2026-09-26", "hour": 20, "app": "night-app"},
            {"date": "2026-09-25", "hour": 9, "app": "editor"},
            {"date": "2026-09-25", "hour": 8, "app": "browser"},
            {"date": "2026-09-25", "hour": 9, "app": "terminal"},
        ]
        self.assertEqual(day_sequences(records),
                         [("browser", "editor", "terminal"),
                          ("night-app",)])

    def test_malformed_records_refuse_with_reasons(self):
        for bad, why in (
            ({"date": "d", "hour": 1}, "no app"),
            ({"app": "x", "hour": 1}, "no date"),
            ({"app": "x", "date": "d", "hour": 24}, "hour 24"),
            ({"app": "x", "date": "d", "hour": "nine"}, "non-integer"),
        ):
            with self.assertRaises(ValueError, msg=why):
                day_sequences([bad])


class LaunchPatternTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger_path = str(Path(self._tmp.name) / "ledger.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _sessions(self):
        # 3 days; "editor -> terminal" runs on days 1 and 2 (support 2)
        return [
            {"date": "2026-09-25", "hour": 8, "app": "browser"},
            {"date": "2026-09-25", "hour": 9, "app": "editor"},
            {"date": "2026-09-25", "hour": 10, "app": "terminal"},
            {"date": "2026-09-26", "hour": 9, "app": "editor"},
            {"date": "2026-09-26", "hour": 10, "app": "terminal"},
            {"date": "2026-09-26", "hour": 20, "app": "game"},
            {"date": "2026-09-27", "hour": 9, "app": "browser"},
        ]

    def test_report_ranks_and_carries_share(self):
        report = mine_launch_patterns(self._sessions(), min_support=2)
        self.assertEqual(report["days"], 3)
        patterns = report["patterns"]
        self.assertTrue(patterns)
        # supports are non-increasing, patterns tie-broken lexically
        supports = [p["support"] for p in patterns]
        self.assertEqual(supports, sorted(supports, reverse=True))
        editor_terminal = next(p for p in patterns
                               if p["pattern"] == ["editor", "terminal"])
        self.assertEqual(editor_terminal["support"], 2)
        self.assertEqual(editor_terminal["share"], round(2 / 3, 4))
        # every reported pattern is verifiable by hand: browser and
        # terminal each appear on 2 days, editor on 2; game appears on
        # ONE day only, so it is honestly below min_support and absent
        by_pattern = {tuple(p["pattern"]): p["support"]
                      for p in patterns}
        self.assertEqual(by_pattern[("browser",)], 2)
        self.assertEqual(by_pattern[("terminal",)], 2)
        self.assertNotIn(("game",), by_pattern)

    def test_determinism(self):
        self.assertEqual(mine_launch_patterns(self._sessions()),
                         mine_launch_patterns(self._sessions()))

    def test_propose_patterns_files_on_the_ledger(self):
        report = mine_launch_patterns(self._sessions(), min_support=2)
        ledger = Ledger(self.ledger_path)
        filed = propose_patterns(report, ledger, top=2)
        self.assertEqual(filed, 2)
        pending = ledger.pending()
        self.assertEqual(len(pending), 2)
        self.assertTrue(all(p["kind"] == "launch_pattern"
                            for p in pending))
        self.assertTrue(all("frequent launch sequence" in p["reason"]
                            for p in pending))
        # the consent surface is untouched: nothing leaves pending
        # without an explicit decide
        self.assertEqual(len(ledger.pending()), 2)


if __name__ == "__main__":
    unittest.main()
