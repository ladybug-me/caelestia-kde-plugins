"""Tests for exponential-build-3 G3 — the ABSTAIN top-2 score gap
(cortex/pipeline.py records it; cortex/explain_unified.py renders it).

Under test:

- the router abstains on a nothing-matches request while still
  producing a ranked list — the pipeline carries the top-2 gap
  (clause, top, top_score, runner_up, runner_up_score, gap) and
  exposes it in to_dict (the --json surface);
- explain_unified's cortex path appends the gap line on ABSTAIN:
  "'top' scored X vs 'runner-up' at Y (gap Z) — neither cleared the
  routing bar", using the router's own numbers (nothing recomputed);
- a ROUTED request carries no gap line (the pipeline went ahead —
  no abstention to explain);
- the no-candidates ABSTAIN (empty text) has no gap and no gap line
  (there is nothing to gap — honest);
- determinism: the same request yields the same gap bytes.
"""
import unittest

from assistant.core.explain_unified import explain_cortex
from assistant.core.pipeline import process
from assistant.core.session import SessionState

ABSTAIN_TEXT = "tune the warp core to eleven"


class AbstainGapTests(unittest.TestCase):
    def test_pipeline_records_the_gap_on_abstain(self):
        result = process(ABSTAIN_TEXT, session=SessionState())
        # F6: warp-core requests name nothing in the registry, so the
        # router now answers with the precise OUT_OF_ONTOLOGY verdict;
        # the top-2 gap evidence is preserved exactly as the old ABSTAIN
        # path kept it.
        self.assertEqual(result.verdict, "OUT_OF_ONTOLOGY")
        gap = result.abstain_gap
        self.assertIsNotNone(gap)
        self.assertEqual(gap["clause"], ABSTAIN_TEXT)
        self.assertEqual(gap["top"], "setArpcIdleTimeout")
        self.assertEqual(gap["runner_up"],
                         "setGameModeDisableToastTransparency")
        self.assertAlmostEqual(gap["gap"], 0.0026, places=4)
        # the gap rides the --json surface too
        self.assertEqual(process(ABSTAIN_TEXT,
                                 session=SessionState()).to_dict()
                         ["abstain_gap"], gap)

    def test_explain_unified_renders_the_gap_line(self):
        report = explain_cortex(ABSTAIN_TEXT)
        self.assertIn(report["headline"].split()[0],
                      ("ABSTAIN", "OUT_OF_ONTOLOGY"))
        self.assertTrue(any("top-2 gap at abstain" in line
                            for line in report["lines"]))
        line = next(line for line in report["lines"]
                    if "top-2 gap" in line)
        self.assertIn("'setArpcIdleTimeout' scored 0.055", line)
        self.assertIn("'setGameModeDisableToastTransparency' at 0.052",
                      line)
        self.assertIn("neither cleared the routing bar", line)

    def test_routed_request_has_no_gap_line(self):
        report = explain_cortex("make the bar bigger")
        self.assertFalse(any("top-2 gap" in line
                             for line in report["lines"]))

    def test_no_candidates_abstain_has_no_gap(self):
        result = process("", session=SessionState())
        self.assertEqual(result.verdict, "ABSTAIN")
        self.assertIsNone(result.abstain_gap)

    def test_determinism(self):
        first = process(ABSTAIN_TEXT, session=SessionState())
        second = process(ABSTAIN_TEXT, session=SessionState())
        self.assertEqual(first.abstain_gap, second.abstain_gap)


if __name__ == "__main__":
    unittest.main()
