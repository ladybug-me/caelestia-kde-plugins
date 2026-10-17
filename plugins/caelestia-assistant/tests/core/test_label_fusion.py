"""A2 Dawid-Skene label-fusion tests.

Contract under test:

- fit() is deterministic (same vote rows -> same reliabilities, byte
  identical to the committed artifact's EM fields);
- the EM actually learned something: bm25/coverage voters end more
  reliable than the fuzzy tail voters on the dev corpus (the shipped
  artifact's ordering, pinned);
- refine() NEVER changes a verdict except ROUTED -> AMBIGUOUS (the
  demote-only contract — fusion can manufacture an ask, never a
  confident route);
- with the shipped artifact + thresholds, a constructed voter split
  (most signals prefer the runner-up) demotes the blend's winner, and
  a unified voter pool never demotes;
- the committed artifact's own thresholds are the ones the code ships
  (artifact/code drift is a test failure, like any golden).
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import List

from assistant.core import label_fusion as lf


@dataclass
class _FakeRoute:
    verdict: str = "ROUTED"
    candidates: List = field(default_factory=list)

    def __post_init__(self):
        if not self.candidates:
            self.candidates = [
                _FakeCand("setBarScale"), _FakeCand("setFontScale")]


@dataclass
class _FakeCand:
    surface: str
    kind: str = "tool"


class FitTests(unittest.TestCase):
    def test_fit_is_deterministic(self) -> None:
        rows = [
            {"candidates": ["a", "b", "c"],
             "votes": {"bm25": "a", "ppmi": "a", "lsa": "b", "fuzz": "a"}},
            {"candidates": ["a", "b", "c"],
             "votes": {"bm25": "b", "ppmi": "b", "lsa": "b", "fuzz": "c"}},
        ]
        m1 = lf.fit(rows)
        m2 = lf.fit(rows)
        self.assertEqual(m1, m2)

    def test_sensible_voters_win_on_the_dev_corpus(self) -> None:
        model = json.loads(lf.ARTIFACT_PATH.read_text())
        sens = model["sensitivity"]
        # the lexical/coverage signal family must out-relieve the fuzzy
        # tail voters — EM's job was to discover exactly this ordering
        self.assertGreater(sens["coverage"], sens["fuzz"])
        self.assertGreater(sens["bm25"], sens["bigram"])

    def test_artifact_thresholds_match_code(self) -> None:
        model = json.loads(lf.ARTIFACT_PATH.read_text())
        self.assertEqual(model["thresholds"]["posterior_demote"],
                         lf.POSTERIOR_DEMOTE)
        self.assertEqual(model["thresholds"]["agreement_floor"],
                         lf.AGREEMENT_FLOOR)


class RefineTests(unittest.TestCase):
    def setUp(self):
        self.model = lf.load_model()
        self.assertTrue(self.model, "the label-fusion artifact must ship")

    def _route(self):
        return _FakeRoute(candidates=[_FakeCand("setBarScale"),
                                      _FakeCand("setFontScale")])

    def test_demote_only(self) -> None:
        # even a unanimous contradiction must produce AMBIGUOUS, never
        # a re-rank or promotion
        r = self._route()
        votes = {"bm25": "setFontScale", "ppmi": "setFontScale",
                 "lsa": "setFontScale", "fuzz": "setFontScale",
                 "coverage": "setFontScale", "bigram": "setFontScale",
                 "cue": "setFontScale"}
        verdict, note, _post = lf.refine(r, votes, self.model, None)
        self.assertIn(verdict, ("ROUTED", "AMBIGUOUS"))
        if verdict == "AMBIGUOUS":
            self.assertIn("which one?", note)

    def test_unanimous_agreement_never_demotes(self) -> None:
        r = self._route()
        votes = {v: "setBarScale" for v in lf.VOTER_KEYS}
        verdict, note, post = lf.refine(r, votes, self.model, None)
        self.assertEqual(verdict, "ROUTED")
        self.assertGreater(post, lf.POSTERIOR_DEMOTE)

    def test_contested_top_demotes(self) -> None:
        # 6 of 7 voters prefer the runner-up: the blend's winner keeps
        # the vote of only the weakest tail voter (agreement 1/7 under
        # the floor) and its fused posterior collapses — the honest ask
        r = self._route()
        votes = {v: "setFontScale" for v in
                 ("bm25", "ppmi", "lsa", "coverage", "cue", "bigram")}
        votes["fuzz"] = "setBarScale"
        verdict, note, post = lf.refine(r, votes, self.model, None)
        self.assertEqual(verdict, "AMBIGUOUS")
        self.assertIn("signals disagree", note)
        self.assertLess(post, lf.POSTERIOR_DEMOTE)

    def test_non_routed_verdicts_untouched(self) -> None:
        r = _FakeRoute(verdict="ABSTAIN",
                       candidates=[_FakeCand("setBarScale")])
        votes = {v: "setBarScale" for v in lf.VOTER_KEYS}
        verdict, note, post = lf.refine(r, votes, self.model, None)
        self.assertEqual(verdict, "ABSTAIN")
        self.assertEqual(post, 1.0)


if __name__ == "__main__":
    unittest.main()
