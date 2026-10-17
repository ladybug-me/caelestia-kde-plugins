"""Tests for exponential-build-3 E2 — the predictive battery/thermal
advisor (cortex/power_advisor.py).

Under test (every number hand-derived):

- holt_state on a perfectly linear series pins level/trend/residuals
  exactly (74 / -2 / [0,0,0]); the point forecasts are CROSS-CHECKED
  byte-identical against brain.forecast.holt (the two implementations
  of the same recursion cannot drift);
- advise_battery on that series: minutes_to_low = 810.0 exactly
  ((74-20)/2 samples x 30 min), trend -4.0%/h, forecast 72.0 with a
  zero-width interval (zero residuals);
- charging series get "no drain estimate" (never a negative
  time-to-low); already-low series say so instead of extrapolating;
  thin series (< 4) ABSTAIN; out-of-range values are refused, not
  clamped;
- the changepoint note: a sharp recent drain shift is flagged
  (detected [5], recent True); steady drain is clean; a GRADUAL
  acceleration is honestly NOT flagged (Holt's trend already
  re-weights it — the docstring's stated position);
- the thermal gross-shift calibration: a 25 degC step scores
  p~1.0 at the shipped 1-degC-squared variance floor where a sharper
  floor would underflow to p=0.0 everywhere and MISS it (the docstring
  documents this exact failure mode — the test pins it);
- thermal minutes-to-high is hand-derived: (85000-63000)/1000 x 5min
  = 110.0;
- every suggestion is an INERT SUGGESTED_NOT_EXECUTED string with a
  risk tier, naming the preset PREVIEW command — nothing executes;
- the CLI surface (`cortex power`) renders both halves and refuses
  usage errors (exit 2) and bad series (exit 1).
"""
import io
import sys
import unittest

from assistant.capabilities.brain.forecast import holt
from assistant.core import power_advisor
from assistant.core.cli import cmd_cortex
from assistant.core.power_advisor import (advise_battery, advise_thermal,
                                            advise, holt_state)


class HoltStateTests(unittest.TestCase):
    def test_linear_series_pins_everything(self):
        # hand-derived: [80,78,76,74] with alpha=.5 beta=.3 — every
        # step predicts exactly (residuals all zero), so level=74,
        # trend=-2, sigma=0
        st = holt_state([80, 78, 76, 74])
        self.assertEqual(st["level"], 74.0)
        self.assertEqual(st["trend"], -2.0)
        self.assertEqual(st["residuals"], [0.0, 0.0, 0.0])
        self.assertEqual(st["residual_std"], 0.0)

    def test_forecasts_match_brain_holt_byte_identically(self):
        # the duplication contract: this module's point forecasts ARE
        # brain.forecast.holt's — pinned so the two cannot drift
        for series in ([80, 79, 77.5, 74, 73, 70],
                       [10, 12, 11, 15, 14, 19],
                       [50, 50, 49, 47, 44, 40, 35]):
            st = holt_state(series)
            mine = [st["level"] + (h + 1) * st["trend"]
                    for h in range(7)]
            self.assertEqual(mine, holt(series, horizon=7), series)

    def test_refusals(self):
        with self.assertRaises(ValueError) as caught:
            holt_state([1.0])
        self.assertIn(">= 2 points", str(caught.exception))
        with self.assertRaises(ValueError):
            holt_state([1.0, 2.0], alpha=0.0)
        with self.assertRaises(ValueError):
            holt_state([1.0, 2.0], beta=1.5)


class BatteryAdviceTests(unittest.TestCase):
    def test_steady_drain_pins_minutes_to_low(self):
        r = advise_battery([80, 78, 76, 74], sample_minutes=30.0)
        self.assertTrue(r["available"])
        self.assertEqual(r["level"], 74.0)
        self.assertEqual(r["trend_per_sample"], -2.0)
        self.assertEqual(r["trend_pct_per_hour"], -4.0)
        self.assertEqual(r["minutes_to_low"], 810.0)
        self.assertFalse(r["charging"])
        self.assertEqual(r["forecast"][0],
                         {"step": 1, "point": 72.0, "low": 72.0,
                          "high": 72.0})
        self.assertIn("iid-residual", r["interval_note"])

    def test_suggestion_is_inert_and_names_the_gate(self):
        r = advise_battery([80, 78, 76, 74])
        self.assertTrue(r["suggestion"].startswith(
            "SUGGESTED_NOT_EXECUTED [STATE_CHANGING]"))
        self.assertIn("--preset battery-saver", r["suggestion"])
        self.assertIn("applies nothing", r["suggestion"])

    def test_charging_gets_no_drain_estimate(self):
        r = advise_battery([50, 60, 70, 80])
        self.assertTrue(r["charging"])
        self.assertIsNone(r["minutes_to_low"])
        self.assertIn("no drain estimate", r["suggestion"])
        self.assertIn("charging", r["suggestion"])

    def test_already_low_says_so(self):
        r = advise_battery([22, 20, 18, 16])
        self.assertIn("already at or below", r["suggestion"])

    def test_thin_series_abstains(self):
        r = advise_battery([90, 88])
        self.assertFalse(r["available"])
        self.assertIn("abstaining", r["reason"])

    def test_out_of_range_refuses(self):
        with self.assertRaises(ValueError) as caught:
            advise_battery([90, 105, 80, 70])
        self.assertIn("outside the plausible range",
                      str(caught.exception))
        self.assertIn("refusing rather than clamping",
                      str(caught.exception))

    def test_interval_widens_with_noise(self):
        steady = advise_battery([80, 78, 76, 74])
        wiggly = advise_battery([80, 70, 76, 66])
        self.assertEqual(
            steady["forecast"][0]["high"] - steady["forecast"][0]["low"],
            0.0)
        wig = wiggly["forecast"][0]
        self.assertGreater(wig["high"], wig["point"])
        self.assertLess(wig["low"], wig["point"])

    def test_recent_sharp_drain_shift_is_flagged(self):
        # -2/-2/-2/-2/-2 then -20/-20: the regime changed at the 6th
        # sample (BOCPD step 5 in the differences, 1-based)
        r = advise_battery([90, 88, 86, 84, 82, 80, 60, 40])
        cp = r["changepoint"]
        self.assertTrue(cp["checked"])
        self.assertEqual(cp["detected_changepoints"], [5])
        self.assertTrue(cp["recent"])
        self.assertIn("mixed history", cp["note"])

    def test_steady_drain_changepoint_clean(self):
        r = advise_battery([90, 88, 86, 84, 82, 80, 78, 76])
        self.assertFalse(r["changepoint"]["recent"])
        self.assertEqual(r["changepoint"]["detected_changepoints"], [])

    def test_gradual_acceleration_is_honestly_not_flagged(self):
        # drain doubles gradually: no single boundary exists, and
        # Holt's trend already re-weights recent points (the stated
        # position in the docstring — pinned so a "fix" that flags
        # this must also update the stated design)
        r = advise_battery([90, 88, 86, 84, 80, 74, 66, 54])
        self.assertFalse(r["changepoint"]["recent"])
        self.assertLess(
            r["changepoint"]["recent_changepoint_prob_max"], 0.3)

    def test_changepoint_skipped_below_five_points(self):
        r = advise_battery([80, 78, 76, 74])
        self.assertFalse(r["changepoint"]["checked"])
        self.assertIn("skipped", r["changepoint"]["note"])


class ThermalAdviceTests(unittest.TestCase):
    def test_steady_rise_pins_minutes_to_high(self):
        # (85000 - 63000)/1000 per-sample trend x 5 min = 110.0
        r = advise_thermal([60000, 61000, 62000, 63000],
                           sample_minutes=5.0)
        self.assertTrue(r["available"])
        self.assertEqual(r["level_mc"], 63000.0)
        self.assertEqual(r["trend_mc_per_sample"], 1000.0)
        self.assertEqual(r["minutes_to_high"], 110.0)
        self.assertTrue(r["suggestion"].startswith(
            "SUGGESTED_NOT_EXECUTED [STATE_CHANGING]"))
        self.assertIn("--preset minimal", r["suggestion"])

    def test_already_high_says_so(self):
        r = advise_thermal([60000, 61000, 62000, 63000, 64000, 65000,
                            90000, 92000])
        self.assertIn("already at or above", r["suggestion"])

    def test_gross_shift_detected_at_the_shipped_floor(self):
        # a 25 degC step: at the shipped 1-degC-squared variance floor
        # BOCPD scores it p~1.0 (detected); at a sharper 0.1-degC
        # floor the Gaussian underflows and scores 0.0 everywhere —
        # the docstring documents this exact calibration and the test
        # pins both sides of it
        r = advise_thermal([60000, 61000, 62000, 63000, 64000, 65000,
                            90000, 92000])
        self.assertTrue(r["changepoint"]["recent"])
        self.assertEqual(r["changepoint"]["detected_changepoints"], [5])
        self.assertGreaterEqual(
            r["changepoint"]["recent_changepoint_prob_max"], 0.9)
        # the counterfactual: the SAME series at the sharper floor
        # underflows and misses (the reason the floor is 1 degC^2) —
        # the shift step's probability is exactly 0.0 there (the
        # first step's 0.01 is the prior initialization, not evidence)
        from assistant.capabilities.genius.data import bocpd
        sharp = bocpd([1000, 1000, 1000, 1000, 1000, 25000, 20000],
                      variance=10_000.0)
        self.assertEqual(sharp["changepoints"], [])
        self.assertEqual(sharp["changepoint_prob"][4], 0.0)

    def test_out_of_range_refuses(self):
        with self.assertRaises(ValueError):
            advise_thermal([20000, 21000, 22000, 23000])  # below 25 C


class AdviseAndCliTests(unittest.TestCase):
    def test_combined_report_and_empty_note(self):
        both = advise(battery=[80, 78, 76, 74],
                      thermal=[60000, 61000, 62000, 63000])
        self.assertIn("battery", both)
        self.assertIn("thermal", both)
        empty = advise()
        self.assertIn("no series supplied", empty["note"])
        self.assertIn("caller-supplied", empty["note"])

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            rc = cmd_cortex(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return rc, out.getvalue(), err.getvalue()

    def test_cli_renders_battery_and_thermal(self):
        rc, out, _ = self._run(
            ["power", "--battery", "80,78,76,74"])
        self.assertEqual(rc, 0)
        self.assertIn("battery:", out)
        self.assertIn("level 74%", out)
        self.assertIn("~13.5 h", out)
        self.assertIn("SUGGESTED_NOT_EXECUTED", out)
        rc, out, _ = self._run(
            ["power", "--thermal", "60000,61000,62000,63000"])
        self.assertEqual(rc, 0)
        self.assertIn("thermal:", out)
        self.assertIn("~110 min", out)
        rc, out, _ = self._run(
            ["power", "--battery", "80,78,76,74",
             "--thermal", "60000,61000,62000,63000"])
        self.assertEqual(rc, 0)
        self.assertIn("battery:", out)
        self.assertIn("thermal:", out)

    def test_cli_usage_errors(self):
        rc, _out, err = self._run(["power"])
        self.assertEqual(rc, 2)
        self.assertIn("pass --battery", err)
        rc, _out, err = self._run(["power", "--battery", "80,abc"])
        self.assertEqual(rc, 2)
        self.assertIn("comma-separated numbers", err)

    def test_cli_bad_series_refuses(self):
        rc, _out, err = self._run(["power", "--battery", "90,105,80,70"])
        self.assertEqual(rc, 1)
        self.assertIn("outside the plausible range", err)

    def test_cli_thin_series_renders_the_abstention(self):
        rc, out, _ = self._run(["power", "--battery", "90,88"])
        self.assertEqual(rc, 0)
        self.assertIn("abstaining", out)


if __name__ == "__main__":
    unittest.main()
