"""tests for genius.autodiff — dual-number forward-mode AD (Wengert 1964).

Pinned: exact chain-rule partials against hand-worked derivatives,
arithmetic overloads (including r-shifted forms and mixed constants),
the function table's closed-form derivatives, domain refusals (never
silently constant), jacobian layout, and the propagate_error contract
(first-order independent-error rule, per-input contributions,
non-negative sigmas, value/sigma length agreement).
"""
import math
import unittest

from assistant.capabilities.genius import autodiff as ad


class ArithmeticTests(unittest.TestCase):
    def test_add_sub_mul_div_gradients(self):
        x = ad.var(3.0, 0, 2)
        y = ad.var(4.0, 1, 2)
        f = x * y + x - y / x
        # d/dx = y + 1 + y/x^2 ; d/dy = x - 1/x
        self.assertAlmostEqual(f.grad[0], 4.0 + 1.0 + 4.0 / 9.0)
        self.assertAlmostEqual(f.grad[1], 3.0 - 1.0 / 3.0)

    def test_radd_rsub_rmul_rdiv_rpow_with_constants(self):
        x = ad.var(2.0, 0, 1)
        self.assertAlmostEqual((5.0 + x).grad[0], 1.0)
        self.assertAlmostEqual((5.0 - x).grad[0], -1.0)
        self.assertAlmostEqual((3.0 * x).grad[0], 3.0)
        self.assertAlmostEqual((12.0 / x).grad[0], -12.0 / 4.0)
        self.assertAlmostEqual((2.0 ** x).grad[0], 2.0 ** 2.0 * math.log(2.0))

    def test_pow_integer_and_negative_base(self):
        x = ad.var(2.0, 0, 1)
        self.assertAlmostEqual((x ** 3).grad[0], 12.0)
        y = ad.var(-2.0, 0, 1)
        self.assertAlmostEqual((y ** 2).grad[0], -4.0)
        with self.assertRaises(ad.AutodiffError):
            y ** 0.5

    def test_abs_sign_rule_and_kink_refusal(self):
        x = ad.var(-3.0, 0, 1)
        self.assertAlmostEqual(abs(x).grad[0], -1.0)
        z = ad.var(0.0, 0, 1)
        with self.assertRaises(ad.AutodiffError):
            abs(z)

    def test_comparisons_use_values_only(self):
        self.assertTrue(ad.var(1.0, 0, 1) < ad.var(2.0, 0, 1))
        self.assertTrue(ad.var(1.0, 0, 1) < 2.0)
        self.assertTrue(ad.var(2.0, 0, 1) == 2.0)

    def test_mixed_variable_counts_refused(self):
        with self.assertRaises(ad.AutodiffError):
            ad.var(1.0, 0, 2) + ad.var(1.0, 0, 3)


class FunctionTableTests(unittest.TestCase):
    def test_trig_and_exp_ln_chain(self):
        x = ad.var(0.7, 0, 1)
        f = ad.exp(x) * ad.sin(x) + ad.ln(x) * ad.cos(x)
        want = (math.exp(0.7) * math.sin(0.7)
                + math.exp(0.7) * math.cos(0.7)
                + math.cos(0.7) / 0.7
                - math.sin(0.7) * math.log(0.7))
        self.assertAlmostEqual(f.grad[0], want)

    def test_sqrt_log2_log10_tanh(self):
        x = ad.var(2.0, 0, 1)
        self.assertAlmostEqual(ad.sqrt(x).grad[0], 1.0 / (2.0 * math.sqrt(2.0)))
        self.assertAlmostEqual(ad.log2(x).grad[0], 1.0 / (2.0 * math.log(2.0)))
        self.assertAlmostEqual(ad.log10(x).grad[0], 1.0 / (2.0 * math.log(10.0)))
        t = ad.tanh(x)
        self.assertAlmostEqual(t.grad[0], 1.0 - math.tanh(2.0) ** 2)

    def test_domain_refusals_are_errors(self):
        x = ad.var(-1.0, 0, 1)
        with self.assertRaises(ad.AutodiffError):
            ad.ln(x)
        with self.assertRaises(ad.AutodiffError):
            ad.sqrt(x)
        edge = ad.var(1.0, 0, 1)
        with self.assertRaises(ad.AutodiffError):
            ad.asin(edge)  # outside the open interval

    def test_unsupported_function_named_and_refused(self):
        try:
            ad._apply("gamma", ad.var(2.0, 0, 1))
            self.fail("expected AutodiffError")
        except ad.AutodiffError as exc:
            self.assertIn("gamma", str(exc))
            self.assertIn("refusing", str(exc))


class DriverTests(unittest.TestCase):
    def test_derivative_matches_hand_value_no_step_size(self):
        f = lambda x: x * x + ad.exp(x)          # noqa: E731
        self.assertAlmostEqual(ad.derivative(f, 1.0), 2.0 + math.exp(1.0))

    def test_jacobian_layout_row_output_col_input(self):
        def fn(v):
            x, y = v
            return [x * y, x + y, x - y * x]
        jac = ad.jacobian(fn, [3.0, 5.0])
        self.assertEqual(len(jac), 3)
        self.assertEqual(len(jac[0]), 2)
        self.assertAlmostEqual(jac[0][0], 5.0)
        self.assertAlmostEqual(jac[0][1], 3.0)
        self.assertAlmostEqual(jac[1][0], 1.0)
        self.assertAlmostEqual(jac[2][1], -3.0)

    def test_propagate_error_quadratic_case(self):
        # f = x*y with x=2 (sigma 0.1), y=3 (sigma 0.2):
        # sigma_f = sqrt((3*0.1)^2 + (2*0.2)^2) = 0.5
        result = ad.propagate_error(lambda v: v[0] * v[1], [2.0, 3.0],
                                    [0.1, 0.2])
        self.assertAlmostEqual(result["values"][0], 6.0)
        self.assertAlmostEqual(result["sigmas"][0], 0.5)
        self.assertAlmostEqual(result["contributions"][0][0], 0.3)
        self.assertAlmostEqual(result["contributions"][0][1], 0.4)
        self.assertIn("correlations not modeled", result["method"])

    def test_propagate_error_rejects_bad_sigmas(self):
        with self.assertRaises(ValueError):
            ad.propagate_error(lambda v: v[0], [1.0], [-0.1])
        with self.assertRaises(ValueError):
            ad.propagate_error(lambda v: v[0], [1.0, 2.0], [0.1])

    def test_error_interval_carries_the_value(self):
        result = ad.propagate_error(lambda v: v[0] + 1.0, [4.0], [0.5])
        lo, hi = result["intervals"][0]
        self.assertAlmostEqual(lo, 4.5)
        self.assertAlmostEqual(hi, 5.5)


if __name__ == "__main__":
    unittest.main()
