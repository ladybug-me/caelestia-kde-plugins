"""tests for cortex.crf_tagger + cortex.coref (exponential-build-4 F).

Pinned: the CRF trains from the same supervision contract and decodes
the same tags as the perceptron, marginals are a normalized
distribution (each position's probabilities sum to 1), the gated
wrapper mirrors tag_gated's contract exactly (untrained -> grammar
fallback, conformal verdict, identical fallback shape), the
calibration comparison measures Brier/LogLoss on held-out rows and
names its winner, and Hobbs-style pronoun resolution binds the
just-mentioned target through the pending-plan composer with the
honest dead end (unresolved pronouns left as-is, never guessed).
"""
import unittest

from assistant.core.conformal import ConformalCalibrator
from assistant.core.coref import resolve_pronoun, resolve_turn
from assistant.core.crf_tagger import CRFTagger, calibration_comparison, tag_gated_crf
from assistant.core.plans import PlanCache

ROWS = [
    ("make the bar a bit smaller",
     ["O", "O", "B-TARGET", "B-CUE", "I-CUE", "B-GENERIC"]),
    ("bar slightly thinner", ["B-TARGET", "B-CUE", "B-GENERIC"]),
    ("dock smaller please", ["B-TARGET", "B-GENERIC", "O"]),
    ("make the dock icons smaller",
     ["O", "O", "B-TARGET", "I-TARGET", "B-GENERIC"]),
    ("bar a smidge shorter", ["B-TARGET", "B-CUE", "B-GENERIC", "O"]),
    ("panel slightly bigger", ["B-TARGET", "B-CUE", "B-GENERIC"]),
    ("make the panel wider", ["O", "O", "B-TARGET", "B-GENERIC"]),
    ("dock icons a bit larger",
     ["B-TARGET", "I-TARGET", "B-CUE", "I-CUE", "B-GENERIC"]),
    ("bar a tad shorter please", ["B-TARGET", "B-CUE", "B-CUE",
                                  "B-GENERIC", "O"]),
    ("the panel should be wider", ["O", "B-TARGET", "O", "O",
                                   "B-GENERIC"]),
    ("make the dock bigger now", ["O", "O", "B-TARGET", "B-GENERIC", "O"]),
    ("bar smaller", ["B-TARGET", "B-GENERIC"]),
]


class CRFTests(unittest.TestCase):
    def _trained(self):
        tagger = CRFTagger()
        tagger.train(ROWS)
        return tagger

    def test_decodes_tags_and_spans(self):
        tagger = self._trained()
        result = tagger.tag("make the bar smaller")
        self.assertEqual(len(result["tags"]), len(result["tokens"]))
        self.assertIn("B-TARGET", result["tags"])
        self.assertTrue(result["spans"]["TARGET"])

    def test_marginals_are_normalized(self):
        tagger = self._trained()
        result = tagger.tag("bar smaller")
        for row in result["marginals"]:
            self.assertAlmostEqual(sum(row), 1.0, places=3)

    def test_confidence_is_a_probability(self):
        tagger = self._trained()
        for text, _gold in ROWS:
            result = tagger.tag(text)
            self.assertTrue(0.0 <= result["confidence"] <= 1.0)

    def test_trained_examples_counted(self):
        tagger = self._trained()
        self.assertEqual(tagger.trained_examples, len(ROWS))

    def test_state_round_trip(self):
        import json
        tagger = self._trained()
        restored = CRFTagger(json.loads(json.dumps(tagger.to_dict()))
                             if hasattr(tagger, "to_dict") else None)
        self.assertEqual(restored.trained_examples,
                         tagger.trained_examples)


class CRFGateTests(unittest.TestCase):
    def test_untrained_falls_back_to_grammar(self):
        result = tag_gated_crf("make the bar smaller", CRFTagger())
        self.assertEqual(result["used"], "grammar-fallback")
        self.assertIn("untrained", result["reason"])

    def test_gated_answer_when_calibrated(self):
        tagger = CRFTagger()
        tagger.train(ROWS)
        confidence = tagger.tag("make the bar smaller")["confidence"]
        calibrator = ConformalCalibrator()
        for _ in range(30):
            calibrator.observe(confidence, "approved")  # (score, outcome)
        result = tag_gated_crf("make the bar smaller", tagger, calibrator)
        self.assertEqual(result["used"], "crf-tagger")
        self.assertIn("conformal", result)

    def test_score_below_threshold_is_the_fallback(self):
        tagger = CRFTagger()
        tagger.train(ROWS)
        calibrator = ConformalCalibrator()
        for _ in range(30):
            calibrator.observe(0.99, "approved")   # the threshold is 0.99
            calibrator.observe(0.01, "rejected")
        result = tag_gated_crf("make the bar smaller", tagger, calibrator)
        # the CRF's 0.96 sits below the conformal threshold 0.99 -> the
        # grammar path answers, exactly like the perceptron's gate
        self.assertEqual(result["used"], "grammar-fallback")
        self.assertIn("below the conformal threshold",
                      str(result.get("reason")))


class CalibrationComparisonTests(unittest.TestCase):
    def test_measures_and_names_winner(self):
        result = calibration_comparison(ROWS)
        self.assertIsNotNone(result["crf"])
        self.assertIsNotNone(result["perceptron"])
        self.assertIn(result["winner"], ("crf", "perceptron", "tie"))
        self.assertTrue(0.0 <= result["crf"]["brier"] <= 1.0)
        self.assertIn("measured", result["note"])

    def test_thin_rows_refused(self):
        result = calibration_comparison(ROWS[:4], holdout=4)
        self.assertIsNone(result["crf"])
        self.assertIn("too few", result["note"])


class CorefTests(unittest.TestCase):
    def test_it_binds_to_just_mentioned_target(self):
        cache = PlanCache(turns=["open firefox"])
        report = cache.resolve_turn("pin it")
        self.assertEqual(report["rewritten"], "pin firefox")
        self.assertEqual(report["resolved"][0]["antecedent"], "firefox")
        self.assertEqual(report["resolved"][0]["source"], "prior-turn")

    def test_pending_plan_walked_first(self):
        cache = PlanCache(turns=["open firefox"])
        cache.compose([{"tool": "setPinned", "raw": "pin krunner window"}],
                      "pin it")
        report = cache.resolve_turn("and close it")
        self.assertEqual(report["resolved"][0]["source"], "pending-plan")
        # right-to-left walk over the raw label: "window" is the most
        # recent content word — the naive algorithm's recency order
        self.assertEqual(report["resolved"][0]["antecedent"], "window")
        # the pending plan outranks the older turn ("firefox") entirely
        self.assertNotEqual(report["resolved"][0]["antecedent"], "firefox")

    def test_them_wants_plural(self):
        report = resolve_turn("resize them", ["open the windows"])
        self.assertEqual(report["resolved"][0]["antecedent"], "windows")

    def test_them_against_singular_is_honest_dead_end(self):
        report = resolve_turn("resize them", ["open firefox"])
        self.assertEqual(report["resolved"], [])
        self.assertEqual(report["unresolved"][0]["antecedent"], None)
        self.assertEqual(report["rewritten"], "resize them")  # left as-is

    def test_pronoun_not_in_set_refused(self):
        report = resolve_pronoun("everyone", ["open firefox"])
        self.assertFalse(report["resolved"])
        self.assertIn("not a resolvable pronoun", report["note"])

    def test_no_context_refuses(self):
        report = resolve_turn("pin it", [])
        self.assertEqual(report["resolved"], [])
        self.assertIn("refusing to guess",
                      report["unresolved"][0]["note"])


if __name__ == "__main__":
    unittest.main()
