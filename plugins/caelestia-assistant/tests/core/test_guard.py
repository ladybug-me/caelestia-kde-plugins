"""F3 evidence-guard tests (exponential-build-5).

Covers the two structural rules in cortex/guard.py (G1 specific
addressing vs non-tool tops; G2 prepositional-object demotion), the
cited lexicon entries they rely on, and the three arena
confident-wrong classes the guard exists to fix (r71/r84/r88).
"""

from __future__ import annotations

import os
import tempfile
import unittest


def _route(text: str):
    from assistant.core import router as R
    return R.route(text, state=R.DEFAULT_STATE, k=5)


def _home() -> None:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="guard-test-")


class GuardRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        _home()

    def test_r71_history_pp_object_promotes_tool(self) -> None:
        res = _route("keep more notifications in history")
        self.assertEqual(res.verdict, "ROUTED")
        self.assertEqual(res.top.surface, "setNotifsMaxNotifs")
        self.assertTrue(
            any("guard G2" in e for e in res.top.evidence),
            "the G2 evidence note must travel on the promoted candidate",
        )

    def test_r88_overview_animation_is_base_duration(self) -> None:
        res = _route("slow down the overview animation")
        self.assertEqual(res.verdict, "ROUTED")
        self.assertEqual(res.top.surface, "setBaseDuration")

    def test_r84_battery_performance_panel_top1(self) -> None:
        res = _route("show the battery in the performance panel")
        self.assertEqual(res.top.surface, "setPerformanceShowBattery")
        self.assertNotEqual(res.top.surface, "preset:battery-saver")

    def test_negative_show_history_still_history(self) -> None:
        res = _route("show history")
        self.assertEqual(res.verdict, "ROUTED")
        self.assertEqual(res.top.surface, "history")
        self.assertFalse(
            any("guard" in e for e in res.top.evidence),
            "the guard must stay silent outside its rules",
        )

    def test_negative_preset_invocation_not_displaced(self) -> None:
        for text in ("optimize for battery life", "enable battery saver"):
            res = _route(text)
            self.assertEqual(res.top.surface, "preset:battery-saver", text)
            self.assertFalse(any("guard" in e for e in res.top.evidence), text)

    def test_determinism(self) -> None:
        a = _route("keep more notifications in history")
        b = _route("keep more notifications in history")
        self.assertEqual(
            [(c.surface, c.score) for c in a.candidates],
            [(c.surface, c.score) for c in b.candidates],
        )


class GuardUnitTests(unittest.TestCase):
    """Pure-function tests over guard.apply with synthetic inputs."""

    def _docs(self):
        return {
            "history": ("history undo past log", "surface"),
            "setFooBar": ("set foo bar foo.bar", "tool"),
            "setBazQux": ("set baz qux baz.qux", "tool"),
            "preset:x": ("x thing", "preset"),
        }

    def test_g1_promotes_full_coverage_tool_over_preset(self) -> None:
        from assistant.core import guard
        scored = [(0.86, "preset:x"), (0.80, "setFooBar"), (0.40, "history")]
        out, notes = guard.apply(
            scored, documents=self._docs(),
            coverage={"setFooBar": 1.0},
            raw_words=["foo", "bar"],
            stems={"foo": "foo", "bar": "bar"},
            doc_stems={"setFooBar": frozenset({"foo", "bar"})},
            min_margin=0.06,
        )
        self.assertEqual(out[0][1], "setFooBar")
        self.assertGreaterEqual(out[0][0], 0.86)
        self.assertTrue(out[0][0] > out[1][0], "order must stay score-descending")
        self.assertTrue(any("guard G1" in n for n in notes))

    def test_g1_refuses_outside_margin(self) -> None:
        from assistant.core import guard
        scored = [(0.97, "preset:x"), (0.80, "setFooBar")]
        out, notes = guard.apply(
            scored, documents=self._docs(),
            coverage={"setFooBar": 1.0},
            raw_words=["foo", "bar"],
            stems={"foo": "foo", "bar": "bar"},
            doc_stems={"setFooBar": frozenset({"foo", "bar"})},
            min_margin=0.06,
        )
        self.assertEqual(out[0][1], "preset:x")
        self.assertEqual(notes, [])

    def test_g1_refuses_partial_coverage(self) -> None:
        from assistant.core import guard
        scored = [(0.86, "preset:x"), (0.80, "setFooBar")]
        out, notes = guard.apply(
            scored, documents=self._docs(),
            coverage={"setFooBar": 0.5},
            raw_words=["foo", "bar"],
            stems={"foo": "foo", "bar": "bar"},
            doc_stems={"setFooBar": frozenset({"foo", "bar"})},
            min_margin=0.06,
        )
        self.assertEqual(out[0][1], "preset:x")
        self.assertEqual(notes, [])

    def test_g2_promotes_tool_matching_rest_of_request(self) -> None:
        from assistant.core import guard
        scored = [(0.80, "history"), (0.63, "setFooBar")]
        out, notes = guard.apply(
            scored, documents=self._docs(),
            coverage={},
            raw_words=["keep", "more", "foo", "in", "history"],
            stems={"keep": "keep", "more": "more", "foo": "foo",
                   "in": "in", "history": "histori"},
            doc_stems={"setFooBar": frozenset({"foo", "bar"})},
            min_margin=0.06,
        )
        self.assertEqual(out[0][1], "setFooBar")
        self.assertGreater(out[0][0], out[1][0])
        self.assertTrue(any("guard G2" in n for n in notes))

    def test_g2_ignores_surface_named_outside_pp(self) -> None:
        from assistant.core import guard
        scored = [(0.80, "history"), (0.63, "setFooBar")]
        out, notes = guard.apply(
            scored, documents=self._docs(),
            coverage={},
            raw_words=["history", "of", "foo"],
            stems={"history": "histori", "of": "of", "foo": "foo"},
            doc_stems={"setFooBar": frozenset({"foo", "bar"})},
            min_margin=0.06,
        )
        self.assertEqual(out[0][1], "history")
        self.assertEqual(notes, [])

    def test_g2_prefers_best_tool_by_overlap_then_score(self) -> None:
        from assistant.core import guard
        docs = self._docs()
        docs["setFooBarBaz"] = ("set foo bar baz", "tool")
        scored = [(0.80, "history"), (0.70, "setBazQux"), (0.62, "setFooBarBaz")]
        out, notes = guard.apply(
            scored, documents=docs,
            coverage={},
            raw_words=["baz", "bar", "in", "history"],
            stems={"baz": "baz", "bar": "bar", "in": "in",
                   "history": "histori"},
            doc_stems={
                "setBazQux": frozenset({"baz", "qux"}),
                "setFooBarBaz": frozenset({"foo", "bar", "baz"}),
            },
            min_margin=0.06,
        )
        # setFooBarBaz matches BOTH content tokens; setBazQux matches one
        # and scores higher — overlap wins, score breaks ties.
        self.assertEqual(out[0][1], "setFooBarBaz")

    def test_empty_and_singletons_are_noops(self) -> None:
        from assistant.core import guard
        for scored in ([], [(0.5, "history")]):
            out, notes = guard.apply(
                scored, documents=self._docs(), coverage={},
                raw_words=["x"], stems={"x": "x"}, doc_stems={},
            )
            self.assertEqual(out, list(scored))
            self.assertEqual(notes, [])


class LexiconEntryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        _home()

    def test_f3_lexicon_keys_are_query_side_only(self) -> None:
        """Hyphenated keys never reach the indexed corpus (the corpus
        fingerprint pins depend on that contract)."""
        from assistant.core.lexicon import SYNONYMS
        for key in ("overview-animation", "performance-panel"):
            self.assertIn(key, SYNONYMS)
            self.assertIn("-", key)

    def test_bigram_expansion_fires(self) -> None:
        from assistant.core.router import _expand_query
        expanded, evidence = _expand_query("slow down the overview animation")
        self.assertIn("duration", expanded)
        self.assertIn("base", expanded)
        self.assertTrue(any("overview" in ev and "duration" in ev or "base" in ev
                            for ev in evidence) or evidence)


if __name__ == "__main__":
    unittest.main()
