"""Tests for the session-scoped PENDING plan cache (phase 2.5).

The contract under test (cortex/plans.py's own docstring):
- ops compose by TOOL NAME, LATER WINS (the compound layer's rule);
- the merged list is what the standard planner re-validates (the cache
  never bypasses validation — tested through the chat loop's composed
  plan in test_cli_wiring below);
- COMMIT happens only when an apply went through; a refused apply
  leaves the ops pending;
- DISCARD is explicit and total;
- serialization round-trips through the session payload untouched.

Pure-module tests: no file I/O, no brain state.
"""

from __future__ import annotations

import unittest

from assistant.core.plans import (
    DISCARD_RE,
    MAX_PENDING_OPS,
    PlanCache,
)


class PlanCacheComposeTests(unittest.TestCase):
    def test_later_op_wins_per_tool(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "setBarScale", "value": 0.8}], "thinner")
        merged = cache.compose([{"tool": "setBarScale", "value": 1.2}],
                               "actually bigger")
        self.assertEqual(merged, [{"tool": "setBarScale", "value": 1.2}])
        self.assertEqual(cache.pending_count(), 1)

    def test_new_tools_append_and_survive(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "setBarScale", "value": 0.8}], "thinner")
        merged = cache.compose([{"tool": "setDockIconSize", "value": 28}],
                               "dock smaller")
        tools = [op["tool"] for op in merged]
        self.assertEqual(tools, ["setBarScale", "setDockIconSize"])
        self.assertEqual(cache.pending_count(), 2)

    def test_compose_is_bounded_at_max_pending_ops(self) -> None:
        cache = PlanCache()
        for i in range(MAX_PENDING_OPS + 5):
            cache.compose([{"tool": f"tool{i}", "value": i}], f"turn {i}")
        self.assertEqual(cache.pending_count(), MAX_PENDING_OPS)
        # the OLDEST tools were evicted, the newest kept
        kept = {op["tool"] for op in cache.pending()}
        self.assertIn(f"tool{MAX_PENDING_OPS + 4}", kept)
        self.assertNotIn("tool0", kept)

    def test_pending_returns_copies_not_references(self) -> None:
        cache = PlanCache(ops=[{"tool": "setBarScale", "value": 0.8}])
        snapshot = cache.pending()
        snapshot[0]["value"] = 999
        self.assertEqual(cache.pending()[0]["value"], 0.8)

    def test_compose_records_turns_deduped_and_bounded(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "t", "value": 1}], "make it smaller")
        cache.compose([{"tool": "t", "value": 2}], "make it smaller")  # dup
        cache.compose([{"tool": "t", "value": 3}], "and the dock too")
        self.assertEqual(cache.turns, ["make it smaller", "and the dock too"])
        self.assertLessEqual(len(cache.turns), MAX_PENDING_OPS)

    def test_empty_compose_is_a_noop(self) -> None:
        cache = PlanCache(ops=[{"tool": "t", "value": 1}])
        merged = cache.compose([], "")
        self.assertEqual(merged, [{"tool": "t", "value": 1}])


class PlanCacheLifecycleTests(unittest.TestCase):
    def test_commit_hands_over_and_clears(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "t", "value": 1}], "one")
        committed = cache.commit()
        self.assertEqual(committed, [{"tool": "t", "value": 1}])
        self.assertEqual(cache.pending_count(), 0)
        self.assertEqual(cache.turns, [])

    def test_discard_drops_everything(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "t", "value": 1}], "one")
        cache.discard()
        self.assertEqual(cache.pending_count(), 0)
        self.assertEqual(cache.pending(), [])

    def test_summary_names_the_pending_tools(self) -> None:
        cache = PlanCache()
        self.assertEqual(cache.summary(), "")
        cache.compose([{"tool": "setBarScale", "value": 0.8}], "thinner")
        line = cache.summary()
        self.assertIn("1 pending", line)
        self.assertIn("setBarScale", line)


class PlanCacheSerializationTests(unittest.TestCase):
    def test_roundtrip_preserves_ops_and_turns(self) -> None:
        cache = PlanCache()
        cache.compose([{"tool": "setBarScale", "value": 0.8}], "thinner")
        cache.compose([{"tool": "setDockIconSize", "value": 28}], "dock")
        data = cache.to_dict()
        revived = PlanCache.from_dict(data)
        self.assertEqual(revived.pending(), cache.pending())
        self.assertEqual(revived.turns, cache.turns)

    def test_from_dict_of_none_is_empty(self) -> None:
        revived = PlanCache.from_dict(None)
        self.assertEqual(revived.pending_count(), 0)

    def test_from_dict_skips_malformed_rows(self) -> None:
        revived = PlanCache.from_dict(
            {"ops": [{"tool": "t", "value": 1}, "not-a-dict", 42],
             "turns": ["ok", 7]})
        self.assertEqual(revived.pending(), [{"tool": "t", "value": 1}])
        self.assertEqual(revived.turns, ["ok", "7"])


class DiscardPhraseTests(unittest.TestCase):
    def test_explicit_discard_phrases_match(self) -> None:
        for phrase in ("never mind", "nevermind", "discard",
                       "drop it", "drop that", "drop the plan",
                       "start over", "forget it", "cancel that"):
            self.assertIsNotNone(
                DISCARD_RE.match(phrase), f"{phrase!r} should discard")

    def test_non_discard_phrases_do_not_match(self) -> None:
        for phrase in ("apply these changes", "make the bar smaller",
                       "what if the bar was thinner", "undo the last change"):
            self.assertIsNone(
                DISCARD_RE.match(phrase), f"{phrase!r} must not discard")


if __name__ == "__main__":
    unittest.main()
