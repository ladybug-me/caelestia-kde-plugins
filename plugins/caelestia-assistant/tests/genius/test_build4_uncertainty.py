"""tests for the exponential-build-4 H uncertainty pass — the audit
that either attaches an error bar (fed by Group B's dual-number
propagation where the estimate is arithmetic on a measurement) or an
explicit "no uncertainty model available" label.

Modules touched by this pass, each pinned here or in the module's own
tests: brain/personal/duration.py (posterior band), brain/personal/
health.py (explicit label), genius/units.py (dual-number conversion),
brain/forecast.py (Holt residual band + Kalman posterior interval),
genius/stats.py describe() (audited: already carried sem — unchanged).
"""
import unittest

from assistant.capabilities.brain.forecast import Kalman1D, holt, holt_with_uncertainty
from assistant.capabilities.brain.personal.duration import DurationModel
from assistant.capabilities.brain.personal.health import uncertainty_label
from assistant.capabilities.genius.units import UnitsError, convert_with_uncertainty


class DurationIntervalTests(unittest.TestCase):
    def test_posterior_band_brackets_the_median(self):
        model = DurationModel()
        for minutes in (25, 40, 35, 30):
            model.observe("chores", minutes)
        e = model.estimate("chores")
        self.assertLess(e["p20_min"], e["median_min"])
        self.assertGreater(e["p80_min"], e["median_min"])
        self.assertGreater(e["sigma_log"], 0.0)
        self.assertIn("posterior band", e["uncertainty"])

    def test_thin_category_still_labeled(self):
        model = DurationModel()
        model.observe("rare", 20)
        e = model.estimate("rare")
        self.assertIn("thin", e["uncertainty"])


class HealthLabelTests(unittest.TestCase):
    def test_label_without_curve(self):
        label = uncertainty_label()
        self.assertIn("no uncertainty model available", label)
        self.assertIn("no interval applies", label)

    def test_label_with_curve_names_the_gap(self):
        label = uncertainty_label([{"at_risk": 2, "revisited": 1}])
        self.assertIn("no uncertainty model available", label)
        self.assertIn("Greenwood", label)


class UnitsUncertaintyTests(unittest.TestCase):
    def test_multiplicative_factor_is_exact(self):
        r = convert_with_uncertainty(100.0, 2.0, "m", "ft")
        self.assertAlmostEqual(r["value"], 328.084, places=2)
        # sigma scales by the same exact factor
        self.assertAlmostEqual(r["sigma"], 2.0 * 3.28084, places=3)

    def test_affine_temperature_propagates(self):
        r = convert_with_uncertainty(20.0, 0.5, "degC", "degF")
        self.assertAlmostEqual(r["value"], 68.0, places=6)
        self.assertAlmostEqual(r["sigma"], 0.9, places=6)  # 0.5 * 9/5
        lo, hi = r["interval"]
        self.assertAlmostEqual(lo, 67.1, places=6)
        self.assertAlmostEqual(hi, 68.9, places=6)

    def test_dimension_mismatch_still_refused(self):
        with self.assertRaises(UnitsError):
            convert_with_uncertainty(1.0, 0.1, "m", "s")

    def test_negative_sigma_refused(self):
        with self.assertRaises(UnitsError):
            convert_with_uncertainty(1.0, -0.5, "m", "ft")


class ForecastUncertaintyTests(unittest.TestCase):
    def test_holt_band_covers_typical_variation(self):
        series = [2.0, 2.1, 1.9, 2.05, 1.95]
        r = holt_with_uncertainty(series, horizon=3)
        self.assertIsNotNone(r["sigma"])
        for (lo, hi), v in zip(r["band"], r["projections"]):
            self.assertLess(lo, v)
            self.assertGreater(hi, v)
        self.assertIn("iid-residual indication", r["note"])

    def test_thin_series_refuses_the_band(self):
        r = holt_with_uncertainty([5.0], horizon=2)
        self.assertIsNone(r["sigma"])
        self.assertIn("no residual sigma", r["note"])
        # the bare point path is unchanged for existing callers
        self.assertEqual(holt([5.0], horizon=2), [5.0, 5.0])

    def test_kalman_surfaces_its_tracked_variance(self):
        k = Kalman1D(q=0.01, r=0.5, x0=0.5, p0=1.0)
        for z in (0.5, 0.52, 0.48, 0.5):
            k.update(z)
        lo, hi = k.interval()
        self.assertLess(lo, k.x)
        self.assertGreater(hi, k.x)
        self.assertAlmostEqual((hi - lo) / 2.0,
                               (k.p ** 0.5), places=9)


if __name__ == "__main__":
    unittest.main()
