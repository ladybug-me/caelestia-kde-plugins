"""A4 confusable clarifier tests.

The clarifier's contract is narrow and honest:

- the committed artifact loads, is sorted deterministically, and every
  hand-pinned pair's tools EXIST in the registry (a pruned tool kills
  its pin rather than shipping a dead question);
- clarify() upgrades the generic ambiguity ask ONLY when both pair
  members are actually among the ranked candidates — never otherwise;
- the runtime never mines (load_pairs reads the committed file only);
- wired through the pipeline: a mined-pair AMBIGUOUS route carries the
  pair's question, and the capability switch restores the generic ask.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.core import confusables
from assistant.adapters.caelestia.registry import TOOL_SPECS

_NAMES = {s.name for s in TOOL_SPECS}


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(
            confusables.PAIRS_PATH.read_text(encoding="utf-8"))

    def test_artifact_loads_and_is_deterministic(self) -> None:
        again = confusables.mine_pairs()
        self.assertEqual(self.data["n_pairs"], again["n_pairs"])
        self.assertEqual(self.data["pairs"], again["pairs"])

    def test_every_pinned_tool_exists(self) -> None:
        for pair in self.data["pairs"]:
            self.assertIn(pair["a"], _NAMES, pair)
            self.assertIn(pair["b"], _NAMES, pair)

    def test_pinned_pairs_carry_answers_and_gain(self) -> None:
        for pair in self.data["pairs"]:
            self.assertTrue(pair["question"].strip(), pair)
            self.assertIn("gain", pair)
            self.assertLessEqual(pair["gain"], 1.0)
            # with two hypotheses the question cannot carry more than
            # one bit of expected information gain
            for tool in pair.get("answers", {}).values():
                self.assertIn(tool, _NAMES)


class ClarifyTests(unittest.TestCase):
    def setUp(self):
        self.pairs = confusables.load_pairs()
        self.assertTrue(self.pairs, "the committed artifact must ship pairs")

    def test_upgrades_only_when_both_members_ranked(self) -> None:
        generic = "several settings could match: 'setTrayRecolour', " \
                  "'setDockRecolourIcons' — which one?"
        q, upgraded = confusables.clarify(
            generic, ["setTrayRecolour", "setDockRecolourIcons",
                      "setBarScale"])
        self.assertTrue(upgraded)
        self.assertIn("TRAY", q)
        # one member missing -> no upgrade, no guess
        q2, up2 = confusables.clarify(
            generic, ["setTrayRecolour", "setBarScale"])
        self.assertFalse(up2)
        self.assertEqual(q2, generic)
        # empty ranking -> generic
        q3, up3 = confusables.clarify(generic, [])
        self.assertFalse(up3)

    def test_runtime_never_mines(self) -> None:
        # a missing artifact degrades to the generic ask, never a live
        # mining run
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "confusable_pairs.json"
            fake.write_text("{}", encoding="utf-8")
            self.assertEqual(confusables.load_pairs.__doc__ is not None,
                             True)  # documented degradation path
            original = confusables.PAIRS_PATH
            try:
                confusables.PAIRS_PATH = fake
                self.assertEqual(confusables.load_pairs(), [])
                q, upgraded = confusables.clarify(
                    "generic", ["setTrayRecolour", "setDockRecolourIcons"])
                self.assertFalse(upgraded)
                self.assertEqual(q, "generic")
            finally:
                confusables.PAIRS_PATH = original


class PipelineWiringTests(unittest.TestCase):
    def test_ambiguous_route_asks_the_pair_question(self) -> None:
        from assistant.core.pipeline import process
        with tempfile.TemporaryDirectory() as tmp:
            res = process("recolour the tray icons to match my theme",
                          file_path=Path(tmp) / "shell.json")
        self.assertEqual(res.verdict, "QUESTION")
        self.assertTrue(res.questions)
        self.assertIn("TRAY", res.questions[0])
        self.assertNotIn("several settings could match",
                         res.questions[0])

    def test_capability_off_restores_generic_ask(self) -> None:
        from unittest import mock
        from assistant.core.pipeline import process
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                    "assistant.core.features.enabled", return_value=False):
                res = process("recolour the tray icons to match my theme",
                              file_path=Path(tmp) / "shell.json")
        self.assertEqual(res.verdict, "QUESTION")
        self.assertIn("several settings could match", res.questions[0])


if __name__ == "__main__":
    unittest.main()
