"""Tests for the SI units engine (genius/units.py, phase 2.4).

The contract under test:
- parsing: quantities, prefixes (longest-first, "km" over "m", "mol"
  never splits), derived and accepted non-SI units, compound units
  (m/s^2, km*h^-1);
- dimension-checked arithmetic: + and - demand equal dimensions and
  keep the left unit for display; * / compose; ^ takes an integer;
  a MISMATCH is an explicit UnitsError — never silently coerced (the
  settings layer's reject-don't-clamp convention);
- conversion: same-dimension only; affine temperature (degC/degF)
  converts but refuses arithmetic;
- the meta-router wiring: unit-shaped requests dispatch to the units
  engine through the EXISTING dispatcher (convert shapes as their own
  domain, unit-ful arithmetic intercepted before plain math_eval);
- determinism and pure-function behavior throughout.
"""

from __future__ import annotations

import unittest

from assistant.capabilities.genius import meta, units
from assistant.capabilities.genius.units import UnitsError


class ParseUnitTests(unittest.TestCase):
    def test_base_derived_and_non_si_factors(self) -> None:
        self.assertEqual(units.parse_unit("m")["dims"][0], 1)
        self.assertEqual(units.parse_unit("cm")["factor"], 0.01)
        self.assertEqual(units.parse_unit("km")["factor"], 1000.0)
        self.assertEqual(units.parse_unit("min")["factor"], 60.0)
        # newton = kg*m*s^-2
        self.assertEqual(units.parse_unit("N")["dims"],
                         (1, 1, -2, 0, 0, 0, 0))
        # kg is its own base (never m+ol splitting)
        self.assertEqual(units.parse_unit("kg")["factor"], 1.0)
        self.assertEqual(units.parse_unit("kg")["dims"][1], 1)

    def test_compound_units(self) -> None:
        u = units.parse_unit("m/s^2")
        self.assertEqual(u["dims"], (1, 0, -2, 0, 0, 0, 0))
        u = units.parse_unit("km*h^-1")
        self.assertEqual(u["factor"], 1000.0 / 3600.0)
        self.assertEqual(u["dims"], (1, 0, -1, 0, 0, 0, 0))

    def test_unknown_unit_rejected(self) -> None:
        with self.assertRaises(UnitsError):
            units.parse_unit("flurb")
        with self.assertRaises(UnitsError):
            units.parse_unit("m//s")
        with self.assertRaises(UnitsError):
            units.parse_unit("m^")

    def test_parse_quantity_and_bare_numbers(self) -> None:
        value, unit = units.parse_quantity("9.81 m/s^2")
        self.assertEqual(value, 9.81)
        self.assertEqual(unit["dims"], (1, 0, -2, 0, 0, 0, 0))
        value, unit = units.parse_quantity("42")
        self.assertEqual(unit["name"], "1")

    def test_mol_never_splits(self) -> None:
        self.assertEqual(units.parse_unit("mol")["dims"][5], 1)
        self.assertEqual(units.parse_unit("mmol")["factor"], 1e-3)


class ConversionTests(unittest.TestCase):
    def test_linear_conversion(self) -> None:
        result = units.evaluate("convert 5 m to cm")
        self.assertEqual(result["value"], 500.0)
        self.assertEqual(result["unit"], "cm")
        self.assertEqual(result["dimension"], "m")

    def test_compound_conversion(self) -> None:
        result = units.evaluate("5 km/h to m/s")
        self.assertAlmostEqual(result["value"], 5 / 3.6, places=6)

    def test_affine_temperature_conversion(self) -> None:
        self.assertEqual(units.evaluate("convert 100 degC to degF")["value"],
                         212.0)
        self.assertEqual(units.evaluate("convert 0 degC to K")["value"],
                         273.15)

    def test_dimension_mismatch_rejected_never_coerced(self) -> None:
        with self.assertRaises(UnitsError) as ctx:
            units.evaluate("convert 5 m to s")
        self.assertIn("dimensions differ", str(ctx.exception))
        self.assertIn("rejected", str(ctx.exception))


class ArithmeticTests(unittest.TestCase):
    def test_addition_converts_to_left_unit(self) -> None:
        result = units.evaluate("3 km + 200 m")
        self.assertEqual(result["si_value"], 3200.0)
        self.assertEqual(result["value"], 3.2)
        self.assertEqual(result["unit"], "km")

    def test_subtraction(self) -> None:
        result = units.evaluate("5 m - 200 cm")
        self.assertAlmostEqual(result["value"], 3.0, places=9)
        self.assertAlmostEqual(result["si_value"], 3.0, places=9)

    def test_multiplication_composes_dimensions(self) -> None:
        result = units.evaluate("2 m * 3 m")
        self.assertEqual(result["si_value"], 6.0)
        self.assertEqual(result["dimension"], "m^2")
        # N = kg*m/s^2: force * distance = energy
        result = units.evaluate("2 N * 3 m")
        self.assertEqual(result["si_value"], 6.0)
        # the dimension display follows the fixed DIMS order (m first)
        self.assertEqual(result["dimension"], "m^2*kg*s^-2")

    def test_division_and_velocity(self) -> None:
        result = units.evaluate("9.81 m/s^2 * 2 s")
        self.assertAlmostEqual(result["si_value"], 19.62, places=9)
        self.assertEqual(result["dimension"], "m*s^-1")

    def test_mismatched_addition_is_explicit(self) -> None:
        with self.assertRaises(UnitsError) as ctx:
            units.evaluate("5 m + 3 s")
        self.assertIn("m vs s", str(ctx.exception))

    def test_affine_arithmetic_refused(self) -> None:
        with self.assertRaises(UnitsError) as ctx:
            units.evaluate("20 degC + 10 degC")
        self.assertIn("affine", str(ctx.exception))

    def test_integer_power_only(self) -> None:
        result = units.evaluate("2 m^2")
        self.assertEqual(result["dimension"], "m^2")
        with self.assertRaises(UnitsError):
            units.evaluate("2 m^(1+s)")


class MetaWiringTests(unittest.TestCase):
    def test_convert_routes_as_its_own_domain(self) -> None:
        result = meta.route_and_do("convert 5 m to cm")
        self.assertEqual(result["domain"], "units")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["value"], 500.0)

    def test_unit_arithmetic_intercepts_math_eval(self) -> None:
        result = meta.route_and_do("3 km + 200 m")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["si_value"], 3200.0)

    def test_mismatch_surfaces_as_an_honest_error(self) -> None:
        result = meta.route_and_do("5 m + 3 s")
        self.assertFalse(result["ok"])
        self.assertIn("dimensions differ", result["error"])

    def test_plain_arithmetic_still_reaches_mathengine(self) -> None:
        # the interception must NOT swallow unit-free expressions
        result = meta.route_and_do("what is 2^10")
        self.assertTrue(result["ok"])
        self.assertEqual(result["domain"], "math_eval")

    def test_looks_unitful_conservative(self) -> None:
        self.assertTrue(units.looks_unitful("3 km + 200 m"))
        self.assertTrue(units.looks_unitful("5 m"))
        self.assertFalse(units.looks_unitful("what is 2^10"))
        self.assertFalse(units.looks_unitful("3 commites"))  # no half-match


class DeterminismTests(unittest.TestCase):
    def test_same_input_same_output_bytes(self) -> None:
        import json
        a = json.dumps(units.evaluate("3 km + 200 m"), sort_keys=True)
        b = json.dumps(units.evaluate("3 km + 200 m"), sort_keys=True)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
