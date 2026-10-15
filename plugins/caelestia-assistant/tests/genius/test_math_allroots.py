"""F8 mathematics fixes (exponential-build-5, D6).

'solve x^2 - 2 = 0' returned ONE root (bisection, -1.414) with routing
metadata printed before the answer. Pinned behavior:

- polynomials in x solve for ALL roots: exact rational roots with
  multiplicity (fractions.Fraction), exact quadratic radicals
  ("sqrt(2)" / "-sqrt(2)"), Durand-Kerner + Newton polish for the rest;
- a Sturm sequence (exact) counts the real roots and the result says
  whether the root set is exhaustive;
- non-polynomial solve bracket-scans and reports EVERY sign-change root,
  labeled found-not-exhaustive;
- rk45 has the standard two-knob controller (rtol + atol) and survives
  the stiff decay y' = -1000y;
- merkle.diff walks iteratively (a depth-3000 crafted tree must diff,
  not RecursionError);
- the `do` text renderer shows the answer FIRST; routing metadata goes
  behind --verbose.
"""
from __future__ import annotations

import unittest

from assistant.capabilities.brain import merkle
from assistant.capabilities.genius import mathengine
from assistant.capabilities.genius.meta import _run_domain


class PolynomialAllRootsTests(unittest.TestCase):
    def solve(self, text):
        return _run_domain("solve_equation", text)

    def test_d6_x_squared_minus_2_has_both_roots_exact(self):
        out = self.solve("solve x^2 - 2 = 0")
        roots = out["roots"]
        values = sorted(round(r["value"], 9) for r in roots)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], -1.414213562, places=6)
        self.assertAlmostEqual(values[1], 1.414213562, places=6)
        exacts = {r["exact"] for r in roots}
        self.assertIn("sqrt(2)", exacts)
        self.assertIn("-sqrt(2)", exacts)
        self.assertEqual(out["real_roots_count"], 2)
        self.assertTrue(out.get("exhaustive"))

    def test_cubic_integer_roots_with_multiplicity(self):
        out = self.solve("solve (x-1)^2 * (x+2) = 0")
        by_value = {round(r["value"], 9): r for r in out["roots"]}
        self.assertEqual(by_value[1.0]["multiplicity"], 2)
        self.assertEqual(by_value[-2.0]["multiplicity"], 1)
        self.assertEqual(out["real_roots_count"], 2)  # distinct
        self.assertTrue(out["exhaustive"])

    def test_complex_roots_reported(self):
        out = self.solve("solve x^2 + 1 = 0")
        self.assertEqual(out["real_roots_count"], 0)
        self.assertEqual(len(out["roots"]), 2)
        self.assertTrue(all(abs(r["value"].imag) > 0.9 for r in out["roots"])
                        or all("i" in (r["exact"] or "") for r in out["roots"]))

    def test_quartic_mixed_roots(self):
        out = self.solve("solve x^4 - 16 = 0")
        values = sorted(round(r["value"].real if hasattr(r["value"], "real")
                              else r["value"], 6) for r in out["roots"])
        self.assertEqual(len(out["roots"]), 4)
        self.assertEqual(out["real_roots_count"], 2)

    def test_sturm_counts_and_labels_exhaustive(self):
        out = self.solve("solve x^3 - 6*x^2 + 11*x - 6 = 0")
        self.assertEqual(out["real_roots_count"], 3)
        vals = sorted(round(r["value"], 6) for r in out["roots"])
        self.assertEqual(vals, [1.0, 2.0, 3.0])


class NonPolynomialSolveTests(unittest.TestCase):
    def solve(self, text):
        return _run_domain("solve_equation", text)

    def test_all_sign_change_roots_found_not_exhaustive(self):
        out = self.solve("solve sin(x) = 0 from -5 to 5")
        roots = [r["value"] for r in out["roots"]]
        # sin(x)=0 on [-5,5] has roots at -3pi..3pi: -3.14, 0, 3.14 (and
        # -4.71? no — sin zeros are multiples of pi within [-5,5]: ±pi, ±2pi? 2pi=6.28>5)
        self.assertIn(0.0, [round(r, 6) for r in roots])
        self.assertGreaterEqual(len(roots), 3)
        self.assertFalse(out.get("exhaustive", True),
                         "bracket-scan roots must be labeled found-not-exhaustive")


class Rk45TwoKnobTests(unittest.TestCase):
    def test_stiff_decay_does_not_stall_or_blow_up(self):
        out = mathengine.ode_solve("-1000*y", x0=0.0, y0=1.0,
                                   x_end=0.05, method="rk45", tol=1e-6)
        # exact: e^(-50) ~ 2e-22 ~ 0
        self.assertLess(abs(out["y_end"]), 1e-6)
        self.assertGreater(out["steps"], 0)
        self.assertEqual(out.get("controller"), "rtol+atol")

    def test_two_knob_tolerances_reported(self):
        out = mathengine.ode_solve("-y", x0=0.0, y0=1.0, x_end=1.0,
                                   method="rk45", tol=1e-6)
        self.assertIn("rtol", out)
        self.assertIn("atol", out)
        self.assertAlmostEqual(out["y_end"], 0.36787944117144233, places=5)


class MerkleIterativeTests(unittest.TestCase):
    def test_diff_survives_depth_3000(self):
        def deep(depth, leaf_files=None):
            # content-derived hashes: a leaf change propagates to the
            # root, forcing a FULL traversal (no subtree pruning)
            node = {"hash": "leaf-" + ",".join(sorted((leaf_files or {"f": "h"}).values())),
                    "files": dict(leaf_files or {"f": "h"}), "dirs": {}}
            for i in range(depth):
                node = {"hash": f"d{i}+{node['hash']}", "files": {}, "dirs": {"c": node}}
            return node

        old = {"tree": deep(3000), "meta": {}}
        new = {"tree": deep(3000, {"f": "h", "g": "new"}), "meta": {}}
        result = merkle.diff(old, new)
        self.assertTrue(any(k.endswith("/g") or k == "g" for k in result["added"]))
        self.assertGreater(result["pruned"]["files_compared"], 0)


if __name__ == "__main__":
    unittest.main()
