"""brain.seasonal tests — MP/SAX, PELT, Holt-Winters, the report."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import seasonal as S  # noqa: E402


def night_series(days=4):
    """Hourly 'notification counts': every day, hours 18-23 light up —
    a real per-phase rhythm, repeated so the init can see it."""
    return ([0.0] * 18 + [10.0, 12.0, 11.0, 13.0, 12.0, 10.0]) * days


class TestMatrixProfile(unittest.TestCase):
    def test_identical_repeats_are_motifs(self):
        m = [0.0] * 30 + [8.0, -8.0, 8.0] + [0.0] * 7
        r = S.motifs_and_discords(m, window=5)
        self.assertEqual(r["verdict"], "OK")
        self.assertGreater(r["discords"][0]["distance"],
                           r["motifs"][0]["distance"])
        # the discord must sit inside/near the spike region
        self.assertTrue(28 <= r["discords"][0]["index"] <= 34,
                        r["discords"][0])

    def test_abstains_past_cap(self):
        self.assertIsNone(S.matrix_profile([1.0] * 3000, 5))
        r = S.motifs_and_discords([1.0] * 3000, 5)
        self.assertEqual(r["verdict"], "ABSTAIN")

    def test_window_bounds_enforced(self):
        with self.assertRaises(ValueError):
            S.matrix_profile([1.0, 2.0], 5)


class TestSAX(unittest.TestCase):
    def test_flat_is_all_one_letter(self):
        self.assertEqual(S.sax_encode([1.0] * 20), "b" * 8)

    def test_rising_spans_alphabet(self):
        word = S.sax_encode(list(range(20)), word_size=8)
        self.assertTrue(word.startswith("a"))
        self.assertTrue(word.endswith("d"))

    def test_scale_invariance(self):
        """z-normalization makes SAX insensitive to a global scale
        change (2x the values, same shape, same word)."""
        shape = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 2
        self.assertEqual(S.sax_encode(shape),
                         S.sax_encode([2 * x for x in shape]))


class TestPELT(unittest.TestCase):
    def test_single_changepoint(self):
        self.assertEqual(S.pelt_changepoints([5.0] * 20 + [15.0] * 20,
                                             penalty=4.0), [20])

    def test_two_changepoints(self):
        self.assertEqual(
            S.pelt_changepoints([5.0] * 20 + [15.0] * 20 + [5.0] * 20,
                                penalty=4.0), [20, 40])

    def test_flat_series_has_none(self):
        self.assertEqual(S.pelt_changepoints([7.0] * 40, penalty=4.0), [])

    def test_deterministic(self):
        s = [float(i % 7) for i in range(60)]
        self.assertEqual(S.pelt_changepoints(s),
                         S.pelt_changepoints(s))


class TestHoltWinters(unittest.TestCase):
    def test_night_rhythm_detected(self):
        hw = S.holt_winters(night_series(), 24)
        self.assertTrue(hw["has_rhythm"])
        self.assertGreater(hw["seasonality_strength"], 0.6)

    def test_flat_series_has_no_rhythm(self):
        hw = S.holt_winters([5.0] * 96, 24)
        self.assertFalse(hw["has_rhythm"])

    def test_noise_has_no_rhythm(self):
        hw = S.holt_winters([(i * 7919 % 13) / 3.0 for i in range(96)],
                            24)
        self.assertFalse(hw["has_rhythm"])

    def test_forecast_is_one_period(self):
        hw = S.holt_winters(night_series(), 24)
        self.assertEqual(len(hw["forecast_next"]), 24)


class TestReport(unittest.TestCase):
    def test_full_report_shape(self):
        r = S.seasonality_report(night_series() + night_series()[:30],
                                 period=24, window=5)
        self.assertIn("seasonality", r)
        self.assertIn("changepoints", r)
        self.assertIn("profile", r)
        self.assertTrue(r["seasonality"]["has_rhythm"])

    def test_short_series_abstains_on_season(self):
        r = S.seasonality_report([1.0, 2.0, 3.0, 4.0, 5.0], period=24,
                                 window=3)
        self.assertEqual(r["seasonality"]["verdict"], "ABSTAIN")

    def test_deterministic(self):
        s = night_series()
        self.assertEqual(S.seasonality_report(s, period=24, window=5),
                         S.seasonality_report(s, period=24, window=5))


if __name__ == "__main__":
    unittest.main()
