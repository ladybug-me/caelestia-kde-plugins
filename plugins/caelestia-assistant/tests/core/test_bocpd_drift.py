"""Tests for exponential-build phase 1.3: BOCPD drift detection applied
to cortex's own routing accuracy.

The contract under test:

- the example log's rolling hit-rate feeds the EXISTING
  ``genius.data.bocpd`` primitive (Adams & MacKay 2007) — the same one
  the numeric telemetry uses — never a second implementation (pinned by
  a spy);
- a real shift in routing accuracy (a run of accepts followed by a run
  of rejects) is flagged ``drift-detected`` with the changepoint
  probability REPORTED, the same honest shape a CPU-load shift gets;
- a stable history is ``stable``; thin data is ``insufficient-data``
  (labelled thin, no numbers invented);
- the whole path is deterministic and additive: the existing halves
  heuristic (``drift_check``) is untouched.
"""
import unittest
from unittest import mock

from assistant.core.learn import CortexLearner, rolling_hit_rate


def _feed(learner: CortexLearner, outcomes) -> None:
    for outcome in outcomes:
        learner.observe("text", "setBarScale",
                        {"lex": 1.0, "sem": 1.0, "fuzz": 1.0,
                         "noun": 1.0, "cue": 1.0},
                        0.8, outcome)


class RollingHitRateTests(unittest.TestCase):
    def test_window_rate_in_arrival_order(self):
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        series = rolling_hit_rate(labels, window=5)
        # overlapping windows: the decline is gradual, the level shift real
        self.assertEqual(series, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        self.assertEqual(len(series), len(labels) - 5 + 1)

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            rolling_hit_rate([1, 0], window=0)


class BocpdDriftTests(unittest.TestCase):
    def test_real_shift_is_flagged_with_reported_probability(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 12 + ["rejected"] * 12)
        report = learner.bocpd_drift_check()
        self.assertEqual(report["status"], "drift-detected")
        self.assertGreaterEqual(report["max_changepoint_prob"], 0.5)
        self.assertIsNotNone(report["segment_mean_before"])
        self.assertLess(report["segment_mean_after"],
                        report["segment_mean_before"])
        self.assertEqual(report["examples"], 24)
        self.assertEqual(report["series_points"], 20)
        # the algorithm citation travels with the report
        self.assertIn("Adams & MacKay", report["algorithm"])

    def test_stable_history_stays_stable(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 24)
        report = learner.bocpd_drift_check()
        self.assertEqual(report["status"], "stable")
        self.assertLess(report["max_changepoint_prob"], 0.5)

    def test_thin_data_is_labelled_insufficient(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 6)  # series of 2 < bocpd's n >= 4
        report = learner.bocpd_drift_check()
        self.assertEqual(report["status"], "insufficient-data")
        self.assertEqual(report["series_points"], 2)
        # no invented numbers on thin evidence
        self.assertNotIn("max_changepoint_prob", report)

    def test_deterministic_across_calls(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 10 + ["rejected"] * 10)
        self.assertEqual(learner.bocpd_drift_check(),
                         learner.bocpd_drift_check())

    def test_reuses_the_single_bocpd_primitive(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 10 + ["rejected"] * 10)
        with mock.patch("assistant.capabilities.genius.data.bocpd",
                        wraps=__import__(
                            "assistant.capabilities.genius.data",
                            fromlist=["bocpd"]).bocpd) as spy:
            report = learner.bocpd_drift_check()
        self.assertEqual(spy.call_count, 1)  # the primitive was used...
        self.assertEqual(report["status"], "drift-detected")  # ...unwrapped

    def test_report_includes_the_bocpd_view(self):
        learner = CortexLearner()
        _feed(learner, ["applied"] * 12 + ["rejected"] * 12)
        full = learner.report()
        self.assertIn("drift_bocpd", full)
        self.assertEqual(full["drift_bocpd"], learner.bocpd_drift_check())
        # the halves heuristic is untouched beside it
        self.assertIn("status", full["drift"])


if __name__ == "__main__":
    unittest.main()
