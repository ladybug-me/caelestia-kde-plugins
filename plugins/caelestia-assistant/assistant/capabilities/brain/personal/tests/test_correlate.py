"""Tests for personal/correlate.py (phase 3.1): habit signals vs task
completion, point-biserial (Tate 1954) and chi-square with Yates'
correction (Pearson 1900 / Yates 1934).

The contract under test:
- hand-computed statistics: r_pb from the classic formula with the
  n-1 sample sd; X² from the Yates-corrected 2x2 table;
- the honesty conventions: EVERY result carries the correlational-not-
  causal framing; thin evidence (n < 30, or an expected cell < 5) is
  LABELED thin while the number still computes;
- inputs are caller-paired records — nothing is imputed for a missing
  habit signal (skipped, never guessed);
- a zero-variance signal or a degenerate table refuses to fabricate a
  correlation.
"""

from __future__ import annotations

import math
import unittest

from assistant.capabilities.brain.personal import correlate


def _records():
    # completed group habit high, incomplete group low — hand-checkable
    return [{"habit": x, "completed": True} for x in (0.9, 0.8, 0.7, 1.0)] \
        + [{"habit": x, "completed": False} for x in (0.1, 0.2, 0.0, 0.3)]


class PointBiserialTests(unittest.TestCase):
    def test_hand_computed_r(self) -> None:
        result = correlate.point_biserial(_records())
        mean1 = 0.85          # mean habit | completed
        mean0 = 0.15          # mean habit | not completed
        xs = [0.9, 0.8, 0.7, 1.0, 0.1, 0.2, 0.0, 0.3]
        mean = 0.5
        var = sum((x - mean) ** 2 for x in xs) / 7   # n-1 = 7
        s = math.sqrt(var)
        expected = ((mean1 - mean0) / s) * math.sqrt(4 * 4 / 64)
        self.assertAlmostEqual(result["r"], expected, places=4)
        self.assertEqual(result["n"], 8)
        self.assertEqual(result["n_completed"], 4)

    def test_framing_is_always_correlational(self) -> None:
        result = correlate.point_biserial(_records())
        self.assertIn("correlational", result["framing"])
        self.assertIn("not causal", result["framing"])

    def test_thin_is_labeled_and_number_still_computes(self) -> None:
        result = correlate.point_biserial(_records())
        self.assertTrue(result["thin"])          # n=8 < 30
        self.assertIsNotNone(result["r"])        # computed anyway

    def test_large_sample_is_not_thin(self) -> None:
        records = [{"habit": 0.9 if i % 2 else 0.1,
                    "completed": i % 2 == 0} for i in range(40)]
        result = correlate.point_biserial(records)
        self.assertFalse(result["thin"])

    def test_degenerate_groups_refuse(self) -> None:
        all_done = [{"habit": 0.5, "completed": True} for _ in range(5)]
        result = correlate.point_biserial(all_done)
        self.assertIsNone(result["r"])
        self.assertTrue(result["thin"])
        zero_var = [{"habit": 0.5, "completed": i % 2 == 0}
                    for i in range(6)]
        self.assertIsNone(correlate.point_biserial(zero_var)["r"])

    def test_pair_records_skips_missing_signals(self) -> None:
        tasks = [{"id": "a", "completed": True, "streak": 3},
                 {"id": "b", "completed": False},          # no signal
                 {"id": "c", "done": True, "streak": None}]
        records = correlate.pair_records(
            tasks, lambda t: t.get("streak"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["habit"], 3.0)
        self.assertTrue(records[0]["completed"])


class ChiSquareTests(unittest.TestCase):
    def test_hand_computed_yates_chi2(self) -> None:
        records = ([{"habit": 1, "completed": True}] * 10
                   + [{"habit": 1, "completed": False}] * 0
                   + [{"habit": 0, "completed": True}] * 2
                   + [{"habit": 0, "completed": False}] * 8)
        result = correlate.chi_square(records)
        # table [[0,2],[10,8]] (row 0 = no habit, col 1 = completed)
        # E = [[1.2*? ...]]: rows 10/10, cols 12/8, n=20
        # E11(row1,col1)=6, E12=4, E21=6, E22=4
        # Yates: (3.5^2)/6 + (3.5^2)/4 + (3.5^2)/6 + (3.5^2)/4
        expected = 3.5 ** 2 * (1 / 6 + 1 / 4 + 1 / 6 + 1 / 4)
        self.assertAlmostEqual(result["chi2"], expected, places=3)
        self.assertEqual(result["completed_rate_with_habit"], 1.0)
        self.assertEqual(result["completed_rate_without_habit"], 0.2)

    def test_thin_labels_on_small_samples(self) -> None:
        records = ([{"habit": 1, "completed": True}] * 10
                   + [{"habit": 0, "completed": True}] * 2
                   + [{"habit": 0, "completed": False}] * 8)
        result = correlate.chi_square(records)
        self.assertTrue(result["thin"])  # n=20 < 30 AND an E cell < 5

    def test_binarization_is_stated_not_silent(self) -> None:
        records = [{"habit": 0.9 if i % 2 else 0.1,
                    "completed": i % 3 == 0} for i in range(40)]
        result = correlate.chi_square(records)
        self.assertEqual(result["binarized_at"], 0.5)
        result = correlate.chi_square(
            [{"habit": 1 if i % 2 else 0, "completed": i % 3 == 0}
             for i in range(40)])
        self.assertIsNone(result["binarized_at"])

    def test_summary_carries_both_and_the_framing(self) -> None:
        result = correlate.summary(_records())
        self.assertIn("point_biserial", result)
        self.assertIn("chi_square", result)
        self.assertIn("not causal", result["framing"])


if __name__ == "__main__":
    unittest.main()
