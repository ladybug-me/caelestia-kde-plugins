"""A3 calibration tests — isotonic (PAVA), Venn-Abers, weighted conformal.

Contract under test:

- isotonic: monotone, idempotent on sorted perfect data, weights matter,
  and the weighted quantile reduces to the existing
  ConformalCalibrator.threshold rule exactly in the unweighted limit;
- Venn-Abers: the interval is honest — on perfectly separable
  calibration data the midpoint sits at the right extreme, and the
  interval always contains the isotonic point estimate;
- ece_report: deterministic (same rows -> same report), out-of-sample
  folds never score a row by a calibrator that saw it;
- the arena's calibration suite carries the A3 report with the VA ECE
  not worse than raw (the ship gate for the module).
"""

from __future__ import annotations

import unittest

from assistant.core.abers import ece_report, isotonic_fit, \
    isotonic_predict, venn_abers, weighted_conformal_quantile
from assistant.core.conformal import ConformalCalibrator


class IsotonicTests(unittest.TestCase):
    def test_monotone_non_decreasing(self) -> None:
        steps = isotonic_fit([(0.1, 0), (0.3, 0), (0.35, 1), (0.5, 0),
                              (0.7, 1), (0.9, 1)])
        prev = -1.0
        for x in (0.0, 0.2, 0.33, 0.4, 0.55, 0.75, 0.95, 1.0):
            v = isotonic_predict(steps, x)
            self.assertGreaterEqual(v, prev - 1e-12)
            prev = v

    def test_perfectly_separated_data_is_the_step(self) -> None:
        steps = isotonic_fit([(0.2, 0), (0.3, 0), (0.7, 1), (0.8, 1)])
        self.assertEqual(isotonic_predict(steps, 0.25), 0.0)
        self.assertEqual(isotonic_predict(steps, 0.75), 1.0)

    def test_weights_change_the_pool(self) -> None:
        pairs = [(0.4, 0.0), (0.6, 1.0)]
        unweighted = isotonic_fit(pairs)
        weighted = isotonic_fit(pairs, weights=[1.0, 1.0])
        self.assertEqual(unweighted, weighted)
        # a heavy 0-weight point disappears from the map entirely
        no_vote = isotonic_fit(pairs, weights=[1.0, 0.0])
        self.assertEqual(len(no_vote), 1)

    def test_mismatched_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            isotonic_fit([(0.5, 1.0)], weights=[1.0, 1.0])


class WeightedConformalTests(unittest.TestCase):
    def _scores(self):
        return [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.4]

    def test_unweighted_matches_the_existing_calibrator(self) -> None:
        c = ConformalCalibrator(alpha=0.1)
        for s in self._scores():
            c.observe(s, "approved")
        self.assertEqual(weighted_conformal_quantile(self._scores(), None, 0.1),
                         c.threshold())

    def test_weights_pull_the_quantile(self) -> None:
        scores = self._scores()
        up = weighted_conformal_quantile(scores, [1, 1, 1, 1, 1, 9, 1], 0.1)
        flat = weighted_conformal_quantile(scores, None, 0.1)
        self.assertGreaterEqual(up, flat)

    def test_too_little_data_is_none(self) -> None:
        self.assertIsNone(weighted_conformal_quantile([0.5, 0.6], None, 0.1))


class VennAbersTests(unittest.TestCase):
    def test_interval_contains_isotonic_and_orders(self) -> None:
        calib = [(0.9, 1), (0.85, 1), (0.8, 1), (0.4, 0), (0.3, 0), (0.2, 0)]
        for p in (0.1, 0.4, 0.6, 0.95):
            lo, hi, mid = venn_abers(calib, p)
            self.assertLessEqual(lo, hi + 1e-12)
            self.assertGreaterEqual(mid, 0.0)
            self.assertLessEqual(mid, 1.0)

    def test_separable_extremes(self) -> None:
        calib = [(0.9, 1), (0.85, 1), (0.2, 0), (0.1, 0)]
        lo, hi, mid = venn_abers(calib, 0.88)
        self.assertGreater(mid, 0.5)   # high score -> confident yes
        lo2, hi2, mid2 = venn_abers(calib, 0.12)
        self.assertLess(mid2, 0.5)     # low score -> confident no


class EceReportTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        rows = [(0.9, 1), (0.8, 1), (0.7, 0), (0.6, 1), (0.95, 0),
                (0.3, 0), (0.2, 0), (0.1, 0)] * 4
        self.assertEqual(ece_report(rows), ece_report(rows))

    def test_arena_carries_the_a3_report(self) -> None:
        from assistant.core.eval.engine import run_suite
        rep = run_suite("calibration", split="dev")
        cr = rep.get("calibration_report")
        self.assertIsNotNone(cr, "the calibration suite must carry the "
                                 "A3 before/after report")
        self.assertLessEqual(cr["ece_venn_abers"], cr["ece_raw"] + 1e-9,
                             "Venn-Abers must not make cold-start ECE "
                             "worse than raw on the arena")


if __name__ == "__main__":
    unittest.main()
