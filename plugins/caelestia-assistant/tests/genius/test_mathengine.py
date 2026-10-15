"""Genius layer tests — every algorithm pinned to a known value."""
import math
import unittest

from assistant.capabilities.genius import mathengine as m
from assistant.capabilities.genius import probability as pr
from assistant.capabilities.genius import linalg as la


class TestMathEngine(unittest.TestCase):
    def test_eval_precedence(self):
        self.assertEqual(m.expression_info("2+3*4^2 - sqrt(16)")["value"], 46.0)

    def test_eval_functions_and_constants(self):
        self.assertAlmostEqual(m.expression_info("sin(pi/6)")["value"], 0.5, places=10)
        self.assertAlmostEqual(m.expression_info("ln(e)")["value"], 1.0, places=10)

    def test_eval_variables(self):
        info = m.expression_info("3*x^2 + 2*x - 5", {"x": 2})
        self.assertEqual(info["value"], 11.0)
        self.assertEqual(info["variables_used"], ["x"])

    def test_division_by_zero_raises(self):
        with self.assertRaises(m.CalcError):
            m.expression_info("1/0")

    def test_parse_error_is_honest(self):
        with self.assertRaises(m.CalcError):
            m.expression_info("2 +* 3")

    def test_symbolic_derivative_polynomial(self):
        d = m.simplify(m.differentiate(m.parse("x^3")))
        self.assertAlmostEqual(m.evaluate(d, {"x": 2}), 12.0, places=9)

    def test_symbolic_derivative_product(self):
        # d/dx sin(x)cos(x) = cos^2 - sin^2 ; at x=0 -> 1
        d = m.differentiate(m.parse("sin(x)*cos(x)"))
        self.assertAlmostEqual(m.evaluate(d, {"x": 0}), 1.0, places=9)

    def test_root_bisection_sqrt2(self):
        r = m.solve_root("x^2 - 2", method="bisection", lo=0, hi=2)
        self.assertAlmostEqual(r["root"], math.sqrt(2), places=8)

    def test_root_newton_dottie(self):
        r = m.solve_root("cos(x) - x", method="newton", x0=1)
        self.assertAlmostEqual(r["root"], 0.7390851332, places=9)

    def test_root_secant_cubic(self):
        r = m.solve_root("x^3 - x - 2", method="secant", x0=1, lo=2)
        self.assertAlmostEqual(r["root"], 1.5213797068, places=8)

    def test_root_same_sign_is_honest(self):
        with self.assertRaises(m.CalcError):
            m.solve_root("x^2 + 1", method="bisection", lo=0, hi=1)

    def test_integral_simpson_exact_quadratic(self):
        self.assertAlmostEqual(m.integrate("x^2", 0, 3)["value"], 9.0, places=9)

    def test_integral_sine(self):
        self.assertAlmostEqual(m.integrate("sin(x)", 0, math.pi)["value"], 2.0, places=6)

    def test_integral_adaptive_gauss(self):
        self.assertAlmostEqual(m.integrate("exp(-x^2)", 0, 5, method="adaptive")["value"],
                               math.sqrt(math.pi) / 2, places=5)

    def test_ode_rk4_exponential(self):
        r = m.ode_solve("y", 0, 1, 1, method="rk4")
        self.assertAlmostEqual(r["y_end"], math.e, places=8)

    def test_taylor_exp(self):
        t = m.taylor("exp(x)", order=4)
        coeffs = [term["coefficient"] for term in t["terms"]]
        self.assertTrue(abs(coeffs[0] - 1) < 1e-12)
        self.assertTrue(abs(coeffs[1] - 1) < 1e-12)
        self.assertTrue(abs(coeffs[2] - 0.5) < 1e-12)

    def test_interpolation_both_methods_agree(self):
        pts = [(1, 2), (3, 4), (5, 3)]
        a = m.interpolate(pts, 4)["value"]
        b = m.interpolate(pts, 4, method="newton")["value"]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 3.875, places=9)

    def test_percent_helpers(self):
        self.assertEqual(m.percent_of(15, 80)["value"], 12.0)
        self.assertAlmostEqual(m.percent_of(15, 80, reverse=True)["percent"], 18.75)


class TestLinalg(unittest.TestCase):
    A = [[2, 1, 1], [1, 3, 2], [1, 0, 0]]

    def test_solve(self):
        x = la.solve(self.A, [4, 5, 6])
        for row, b in zip(self.A, [4, 5, 6]):
            self.assertAlmostEqual(sum(r * xi for r, xi in zip(row, x)), b, places=9)

    def test_determinant(self):
        self.assertAlmostEqual(la.determinant(self.A), -1.0, places=9)

    def test_inverse_times_matrix_is_identity(self):
        prod = la.matmul(self.A, la.inverse(self.A))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0, places=9)

    def test_singular_raises(self):
        with self.assertRaises(la.LinAlgError):
            la.inverse([[1, 2], [2, 4]])

    def test_power_iteration(self):
        lam, _ = la.power_iteration([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
        self.assertAlmostEqual(lam, 3.0, places=9)

    def test_rank(self):
        self.assertEqual(la.rank([[1, 2], [2, 4], [3, 6]]), 1)
        self.assertEqual(la.rank([[1, 0], [0, 1]]), 2)

    def test_least_squares(self):
        beta, info = la.least_squares([[1, 1], [1, 2], [1, 3], [1, 4]], [2, 4, 5, 4])
        self.assertAlmostEqual(info["r2"], 0.5158, places=3)


class TestProbability(unittest.TestCase):
    def test_bayes_medical(self):
        b = pr.bayes(0.01, 0.95, 0.05)
        self.assertAlmostEqual(b["posterior"], 0.161017, places=6)

    def test_bayes_multi_ranks(self):
        r = pr.bayes_multi({"a": 0.3, "b": 0.5, "c": 0.2},
                           {"a": 0.9, "b": 0.3, "c": 0.1})
        self.assertEqual(r["best"], "a")
        self.assertAlmostEqual(sum(r["posterior"].values()), 1.0, places=9)

    def test_evidence_update_sequential(self):
        r = pr.evidence_update({"h": 0.5, "n": 0.5}, [{"h": 0.9, "n": 0.2}] * 3)
        self.assertGreater(r["posterior"]["h"], 0.98)

    def test_normal_quantile_wichura(self):
        self.assertAlmostEqual(pr.normal_quantile(0.975), 1.959964, places=6)
        self.assertAlmostEqual(pr.normal_quantile(0.5), 0.0, places=9)

    def test_binomial_cdf(self):
        self.assertAlmostEqual(pr.binomial_cdf(40, 100, 0.5), 0.0284, places=3)

    def test_combinatorics(self):
        self.assertEqual(pr.stars_and_bars(10, 3), 66)
        self.assertEqual(pr.catalan(5), 42)
        self.assertEqual(pr.derangements(4), 9)

    def test_monte_carlo_mean(self):
        mc = pr.monte_carlo(lambda rng: rng.random() ** 2, n=50000)
        self.assertAlmostEqual(mc["mean"], 1 / 3, places=2)

    def test_simulate_expression(self):
        se = pr.simulate_expression("u1 + 2*u2", n=20000)
        self.assertAlmostEqual(se["mean"], 1.5, places=2)

    def test_markov_stationary(self):
        st = pr.markov_stationary([[0.9, 0.1], [0.5, 0.5]])
        self.assertAlmostEqual(st["stationary"][0], 5 / 6, places=6)

    def test_markov_absorbing(self):
        ab = pr.markov_absorbing([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0, 1]])
        self.assertAlmostEqual(ab["expected_steps_to_absorption"][0], 8.0, places=6)
        self.assertAlmostEqual(ab["expected_steps_to_absorption"][1], 6.0, places=6)


class TestSymbolicIntegration(unittest.TestCase):
    """exponential-build-4 B: the bounded symbolic integrator — table
    hits verified differentially before shipping; table misses are the
    honest NO_CLOSED_FORM_IN_TABLE, never an invented answer."""

    def test_table_hits_verified_ok(self):
        cases = {
            "x": "(x ^ 2) / 2",
            "3*x^2": "x ^ 3",
            "exp(2*x)": "exp(2 * x) / 2",
            "ln(x)": "x * ln(x) - x",
            "sin(3*x)": "-cos(3 * x) / 3",
            "cos(x)": "sin(x)",
            "1/x": "ln(x)",
            "x*exp(x)": "x * exp(x) - exp(x)",        # by parts
            "x^2*exp(x)": None,                        # by parts, degree 2
            "x^2*sin(x)": None,                        # by parts, twice
            "x*exp(x^2)": "0.5 * exp(x ^ 2)",          # u-substitution
            "sqrt(x)": "(x ^ 1.5) / 1.5",
            "(2*x + 1)^3": "(2 * x + 1) ^ 4 / 8",      # chain power rule
        }
        for expr in cases:
            r = m.symbolic_integrate(expr)
            self.assertEqual(r["status"], "OK", msg=expr)
            self.assertTrue(r["verified"], msg=expr)
            self.assertTrue(r["method_trace"], msg=expr)

    def test_antiderivative_differentiates_back(self):
        for expr in ("x^3 - 2*x + 7", "tan(x)", "ln(2*x)", "5*x*exp(-x)"):
            r = m.symbolic_integrate(expr)
            self.assertEqual(r["status"], "OK", msg=expr)
            F = m.parse(r["antiderivative"])
            dF = m.simplify(m.differentiate(F, "x"))
            f = m.parse(expr)
            for x in (0.5, 1.3, 2.1):
                a = m.evaluate(dF, {"x": x})
                b = m.evaluate(f, {"x": x})
                self.assertAlmostEqual(a, b, places=5, msg=expr)

    def test_out_of_table_is_honest_refusal(self):
        for expr in ("exp(x^2)", "sin(x)*cos(x)", "exp(x)*sin(x)"):
            r = m.symbolic_integrate(expr)
            self.assertEqual(r["status"], "NO_CLOSED_FORM_IN_TABLE", msg=expr)
            self.assertIn("not a Risch", r["note"])

    def test_verification_gate_refuses_bad_candidates(self):
        # a forced bad candidate through the internal table: the gate
        # must refuse the engine's own wrong answer, never ship it
        integrand = m.parse("x*exp(x^2)")
        bad = m.parse("2 * exp(x^2)")   # wrong factor (k division bug)
        self.assertFalse(m._verify_antiderivative(bad, integrand, "x"))
        good = m.parse("0.5 * exp(x^2)")
        self.assertTrue(m._verify_antiderivative(good, integrand, "x"))

    def test_power_rule_negative_one_gives_ln(self):
        r = m.symbolic_integrate("1/x")
        self.assertEqual(r["status"], "OK")
        self.assertIn("ln", r["antiderivative"])


class TestRK45(unittest.TestCase):
    """exponential-build-4 B: adaptive Dormand-Prince 4(5) alongside the
    fixed-step solvers — same interface, honest step/error accounting."""

    def test_rk45_matches_exact_solution(self):
        r = m.ode_solve("y", 0.0, 1.0, 1.0, method="rk45", tol=1e-9)
        self.assertAlmostEqual(r["y_end"], math.e, places=8)
        self.assertEqual(r["method"], "rk45")
        self.assertIn("rejected_steps", r)
        self.assertLess(r["max_local_error"], 1e-8)

    def test_adaptive_steps_earn_their_keep(self):
        # y' = y on [0,1]: exact y(1) = e. Adaptive hits it with ~a dozen
        # steps; euler at the SAME step budget is off by ~0.1. The
        # comparison is against the same-budget euler, not rk4 (fixed
        # rk4 at h=1e-3 is also very accurate — 1000 steps of it —
        # which is the honest counterpoint: adaptivity buys steps, not
        # magic)
        adaptive = m.ode_solve("y", 0.0, 1.0, 1.0, method="rk45", tol=1e-8)
        euler_same_budget = m.ode_solve("y", 0.0, 1.0, 1.0, h=1.0 / adaptive["steps"],
                                        method="euler")
        self.assertLess(adaptive["steps"], 30)
        self.assertLess(abs(adaptive["y_end"] - math.e), 1e-7)
        self.assertGreater(abs(euler_same_budget["y_end"] - math.e), 0.05)

    def test_rejections_reported_on_stiff_problem(self):
        # y' = -1000y has time constant 1e-3; the initial step guess
        # (span/100) is beyond the stability region, so the controller
        # MUST reject and shrink — visibly, in the report. The mixed
        # one-knob tolerance bounds ABSOLUTE error, so a solution that
        # decays ~435 orders of magnitude is resolved only down to
        # ~tol*span — that envelope is the documented honest limit,
        # recorded in the build-4 findings
        r = m.ode_solve("-1000*y", 0.0, 1.0, 1.0, method="rk45", tol=1e-10)
        self.assertGreaterEqual(r["rejected_steps"], 1)
        self.assertLess(r["y_end"], 1e-9)

    def test_unknown_method_still_refused(self):
        with self.assertRaises(m.CalcError):
            m.ode_solve("y", 0.0, 1.0, 1.0, method="midpoint")

    def test_tol_must_be_positive(self):
        with self.assertRaises(m.CalcError):
            m.ode_solve("y", 0.0, 1.0, 1.0, method="rk45", tol=0.0)


if __name__ == "__main__":
    unittest.main()
