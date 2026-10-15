"""Tests for the closed-loop cadence throttle (brain/dreamtime.py, 2.1).

The contract under test:
- the PI loop math is exact and hand-checkable (error = load - ceiling;
  cadence = base + Kp*error + Ki*integral, saturated to the actuator
  limits; integral bounded by anti-windup);
- telemetry is INJECTED (a fixture source callable returning synthetic
  snapshot dicts) — no test reads live /proc or /sys;
- an unavailable or failing telemetry probe is an honest no-op: the
  cadence holds where it was, observations do not advance, and the
  controller never brakes on a broken sensor;
- defaults are conservative: the default ceiling sits BELOW eligible()'s
  max_load gate, the cadence never drops below the floor even on a cold
  idle box, and out-of-range constructor values are rejected, never
  silently adjusted.
"""

from __future__ import annotations

import unittest

from assistant.capabilities.brain.dreamtime import (
    BASE_CADENCE_MIN,
    DEFAULT_CEILING_PERCENT,
    CadenceController,
    eligible,
    load_percent_from_snapshot,
)


def _fixture(available=True, load1=None):
    """A fixture telemetry source shaped like telemetry.read_loadavg()."""
    return {"available": available, "source": "/fixture/proc/loadavg",
            "load1": load1, "load5": load1, "load15": load1}


class LoopMathTests(unittest.TestCase):
    def controller(self, **kw):
        defaults = dict(ceiling_percent=25.0, base_cadence_min=15.0,
                        min_cadence_min=5.0, max_cadence_min=240.0,
                        kp=1.5, ki=0.25, cores=2)
        defaults.update(kw)
        return CadenceController(**defaults)

    def test_initial_cadence_is_the_base(self) -> None:
        c = self.controller()
        self.assertEqual(c.cadence_min, 15.0)
        self.assertEqual(c.report()["observations"], 0)

    def test_over_ceiling_raises_cadence_exactly(self) -> None:
        c = self.controller()
        # load1=1.0 on 2 cores = 50% -> error +25
        got = c.observe(_fixture(load1=1.0))
        # cadence = 15 + 1.5*25 + 0.25*25 = 58.75
        self.assertAlmostEqual(got, 58.75, places=6)
        self.assertAlmostEqual(c.integral, 25.0, places=6)

    def test_integral_accumulates_across_observations(self) -> None:
        c = self.controller()
        c.observe(_fixture(load1=1.0))          # integral 25
        got = c.observe(_fixture(load1=1.0))    # integral 50
        # cadence = 15 + 1.5*25 + 0.25*50 = 65
        self.assertAlmostEqual(got, 65.0, places=6)

    def test_under_ceiling_relaxes_toward_base(self) -> None:
        c = self.controller()
        c.observe(_fixture(load1=1.0))  # brake hard first
        got = c.observe(_fixture(load1=0.0))    # 0% -> error -25
        # raw: 15 + 1.5*(-25) + 0.25*(25-25) = -22.5 -> saturated to min
        self.assertAlmostEqual(got, 5.0, places=6)

    def test_sustained_overload_settles_bounded_and_deterministic(self) -> None:
        c = self.controller()
        for _ in range(40):
            got = c.observe(_fixture(load1=2.0))  # 100% on 2 cores
        # steady state, hand-computed: error 75 every step; the P term
        # alone gives 15 + 1.5*75 = 127.5; the integral anti-winds at
        # max-base = 225, adding 0.25*225 = 56.25 -> 183.75 (bounded,
        # below the 240 actuator max — exactly what the spec promises)
        self.assertAlmostEqual(got, 183.75, places=6)
        self.assertAlmostEqual(c.integral, 225.0, places=6)

    def test_anti_windup_bounds_the_integral(self) -> None:
        c = self.controller(integral_limit=40.0)
        for _ in range(20):
            c.observe(_fixture(load1=1.0))
        self.assertAlmostEqual(c.integral, 40.0, places=6)

    def test_exactly_at_ceiling_holds_base(self) -> None:
        c = self.controller()
        got = c.observe(_fixture(load1=0.5))    # 50% of 2 cores = 25%
        self.assertAlmostEqual(got, 15.0, places=6)


class TelemetryContractTests(unittest.TestCase):
    def test_unavailable_probe_is_a_noop(self) -> None:
        c = CadenceController(cores=2)
        got = c.observe(_fixture(available=False))
        self.assertEqual(got, c.base_cadence_min)
        self.assertEqual(c.report()["observations"], 0)

    def test_failing_source_is_a_noop(self) -> None:
        def broken():
            raise OSError("proc gone")

        c = CadenceController(cores=2, source=broken)
        got = c.observe()
        self.assertEqual(got, c.base_cadence_min)
        self.assertEqual(c.report()["observations"], 0)

    def test_source_is_polled_when_no_explicit_reading(self) -> None:
        calls = []

        def source():
            calls.append(1)
            return _fixture(load1=0.25)  # 12.5% of 2 cores

        c = CadenceController(cores=2, source=source)
        c.observe()
        self.assertEqual(len(calls), 1)
        self.assertLess(c.cadence_min, c.base_cadence_min)

    def test_load_percent_from_snapshot_shapes(self) -> None:
        # full snapshot shape (telemetry.snapshot) ...
        snap = {"loadavg": _fixture(load1=0.5), "meminfo": {}}
        self.assertAlmostEqual(
            load_percent_from_snapshot(snap, cores=2), 25.0, places=6)
        # ... and the bare loadavg dict shape
        self.assertAlmostEqual(
            load_percent_from_snapshot(_fixture(load1=0.5), cores=2),
            25.0, places=6)
        # unavailable -> None (hold, don't guess)
        self.assertIsNone(load_percent_from_snapshot(
            _fixture(available=False), cores=2))
        self.assertIsNone(load_percent_from_snapshot({"loadavg": {}}))

    def test_over_ceiling_flag(self) -> None:
        c = CadenceController(ceiling_percent=25.0, cores=2)
        self.assertTrue(c.over_ceiling(_fixture(load1=1.0)))
        self.assertFalse(c.over_ceiling(_fixture(load1=0.5)))
        self.assertFalse(c.over_ceiling(_fixture(available=False)))


class DefaultsAndValidationTests(unittest.TestCase):
    def test_default_ceiling_is_conservative_vs_eligibility_gate(self) -> None:
        # below eligible()'s own max_load=30: braking starts BEFORE the
        # window would lose eligibility
        self.assertLess(DEFAULT_CEILING_PERCENT, 30.0)
        self.assertLessEqual(DEFAULT_CEILING_PERCENT, 25.0)
        self.assertLessEqual(BASE_CADENCE_MIN, 15.0)

    def test_constructor_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            CadenceController(ceiling_percent=0.0)
        with self.assertRaises(ValueError):
            CadenceController(ceiling_percent=150.0)
        with self.assertRaises(ValueError):
            # min > base violates the actuator ordering
            CadenceController(min_cadence_min=30.0, base_cadence_min=15.0)

    def test_still_composes_with_the_existing_gate(self) -> None:
        # the throttle extends, never replaces: eligible() semantics
        # unchanged (regression pin for 2.1)
        self.assertTrue(eligible(idle_minutes=10, on_ac_power=True,
                                 cpu_load_percent=20))
        self.assertFalse(eligible(idle_minutes=10, on_ac_power=True,
                                  cpu_load_percent=31))


if __name__ == "__main__":
    unittest.main()
