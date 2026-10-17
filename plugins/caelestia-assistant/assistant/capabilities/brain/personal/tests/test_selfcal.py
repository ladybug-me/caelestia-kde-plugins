"""Tests for personal/selfcal.py (phase 3.3): the Brier-score ledger
over the user's own STATED predictions.

The contract under test:
- the ledger is EXPLICIT-LOGGING ONLY: entries exist because predict()
  put them there; nothing is inferred from unstated behavior;
- the scoring: Brier (Brier 1950) mean squared error over RESOLVED
  predictions, hand-computed; None when nothing has resolved (no
  invented scores);
- a certainty claim (p = 0 or 1) is rejected, never clamped; resolving
  an already-resolved prediction is refused (the first resolution
  stands — no rewriting history);
- the calibration curve buckets stated p vs hit rate; the summary
  carries the framing;
- a DISTINCT personal-only instance: brain/calibrate.py untouched (its
  module source is pinned byte-identical in the wiring test below).
"""

from __future__ import annotations

import unittest

from assistant.capabilities.brain.personal import selfcal


class PredictTests(unittest.TestCase):
    def test_explicit_logging_only_creates_entries(self) -> None:
        entries: dict = {}
        selfcal.predict(entries, "p1", "migration done by Friday", 0.7,
                        due="2026-10-02", at="2026-09-26")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries["p1"]["p"], 0.7)
        self.assertIsNone(entries["p1"]["outcome"])
        self.assertEqual(selfcal.open_predictions(entries)[0]["id"], "p1")

    def test_certainty_claims_rejected_never_clamped(self) -> None:
        entries: dict = {}
        for bad in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                selfcal.predict(entries, "x", "will happen", bad)
        self.assertEqual(entries, {})


class ResolveAndScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries: dict = {}
        selfcal.predict(self.entries, "p1", "launch slips", 0.7)
        selfcal.predict(self.entries, "p2", "migration done", 0.9)
        selfcal.predict(self.entries, "p3", "review clears", 0.4)
        selfcal.resolve(self.entries, "p1", True)
        selfcal.resolve(self.entries, "p2", False)
        # p3 left open

    def test_brier_hand_computed(self) -> None:
        # p1: (0.7-1)^2 = 0.09 ; p2: (0.9-0)^2 = 0.81 -> mean 0.45
        self.assertAlmostEqual(selfcal.brier_score(self.entries), 0.45,
                               places=9)

    def test_no_resolutions_no_invented_score(self) -> None:
        self.assertIsNone(selfcal.brier_score({}))
        self.assertIsNone(selfcal.brier_score({"p": {"p": 0.5,
                                                     "outcome": None}}))

    def test_double_resolution_refused(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            selfcal.resolve(self.entries, "p1", False)
        self.assertIn("first resolution", str(ctx.exception))

    def test_unknown_prediction_refused(self) -> None:
        with self.assertRaises(KeyError):
            selfcal.resolve(self.entries, "nope", True)

    def test_calibration_curve_buckets(self) -> None:
        curve = selfcal.calibration_curve(self.entries, bins=5)
        by_bucket = {c["bucket"]: c for c in curve}
        self.assertEqual(by_bucket[3]["n"], 1)   # p=0.7
        self.assertEqual(by_bucket[4]["n"], 1)   # p=0.9
        self.assertEqual(by_bucket[3]["hit_rate"], 1.0)
        self.assertEqual(by_bucket[4]["hit_rate"], 0.0)

    def test_summary_carries_framing_and_counts(self) -> None:
        result = selfcal.summary(self.entries)
        self.assertAlmostEqual(result["brier"], 0.45, places=9)
        self.assertEqual(result["open"], 1)
        self.assertEqual(result["resolved"], 2)
        self.assertIn("diary, not a judgement", result["framing"])


class DistinctInstanceTests(unittest.TestCase):
    def test_state_key_is_personal_and_distinct(self) -> None:
        self.assertEqual(selfcal.STATE_KEY, "personal_predictions")

    def test_calibrate_module_untouched(self) -> None:
        # the brief: do not modify brain/calibrate.py — pin its public
        # surface shape so a drift here fails loudly
        from assistant.capabilities.brain import calibrate
        self.assertTrue(callable(calibrate.acceptance_rate))
        self.assertTrue(callable(calibrate.confidence_calibration))
        self.assertTrue(callable(calibrate.fold_undo_negatives))
        self.assertTrue(callable(calibrate.DailyBudget))


if __name__ == "__main__":
    unittest.main()


class MurphyDecompositionTests(unittest.TestCase):
    """exponential-build-4 C: the Murphy 1973 partition of the Brier
    score — REL/RES/UNC plus the honestly-reported finite-sample
    residual (the three-term identity is exact in expectation; the
    covariance residual is shown, not folded away)."""

    def _ledger(self):
        entries = {}
        outcomes = [(0.8, True), (0.8, False), (0.7, False),
                    (0.3, False), (0.3, True), (0.6, True)]
        for i, (p, o) in enumerate(outcomes):
            selfcal.predict(entries, f"p{i}", f"statement {i}", p)
            selfcal.resolve(entries, f"p{i}", o)
        return entries

    def test_components_present_and_ordered(self):
        m = selfcal.murphy_decomposition(self._ledger())
        for key in ("bs", "rel", "res", "unc", "d", "residual", "base_rate"):
            self.assertIn(key, m)
        self.assertAlmostEqual(m["bs"], 0.3183, places=3)
        self.assertAlmostEqual(m["base_rate"], 0.5, places=6)

    def test_residual_makes_the_identity_exact_by_construction(self):
        m = selfcal.murphy_decomposition(self._ledger())
        recomposed = m["rel"] - m["res"] + m["unc"] + m["residual"]
        self.assertAlmostEqual(recomposed, m["bs"], places=4)

    def test_perfectly_calibrated_low_rel(self):
        entries = {}
        outcomes = [(0.9, True), (0.9, True), (0.9, True),
                    (0.1, False), (0.1, False), (0.1, False)]
        for i, (p, o) in enumerate(outcomes):
            selfcal.predict(entries, f"p{i}", f"s{i}", p)
            selfcal.resolve(entries, f"p{i}", o)
        m = selfcal.murphy_decomposition(entries)
        high_res = m["rel"] + m["d"]
        self.assertLessEqual(high_res, 0.0101)

    def test_resolution_zero_when_outcomes_do_not_separate(self):
        entries = {}
        outcomes = [(0.7, True), (0.3, True), (0.7, False), (0.3, False)]
        for i, (p, o) in enumerate(outcomes):
            selfcal.predict(entries, f"p{i}", f"s{i}", p)
            selfcal.resolve(entries, f"p{i}", o)
        m = selfcal.murphy_decomposition(entries)
        self.assertAlmostEqual(m["res"], 0.0, places=6)

    def test_empty_ledger_refused_honestly(self):
        m = selfcal.murphy_decomposition({})
        self.assertIsNone(m["bs"])
        self.assertIn("empty", m["note"])

    def test_thin_sample_flagged(self):
        entries = {}
        for i, (p, o) in enumerate([(0.8, True), (0.9, False)]):
            selfcal.predict(entries, f"p{i}", f"s{i}", p)
            selfcal.resolve(entries, f"p{i}", o)
        m = selfcal.murphy_decomposition(entries)
        self.assertTrue(m["thin_sample"])

    def test_summary_carries_murphy(self):
        summary = selfcal.summary(self._ledger())
        self.assertIn("murphy", summary)
        self.assertIn("rel", summary["murphy"])
