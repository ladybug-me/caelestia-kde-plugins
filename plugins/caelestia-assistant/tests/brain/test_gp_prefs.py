"""brain.gp_prefs tests — Cholesky, probit pairs, validation."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import gp_prefs as G  # noqa: E402

ITEMS = [
    {"id": "s08", "tool": "setBarScale", "value": 0.8, "features": [0.8]},
    {"id": "s10", "tool": "setBarScale", "value": 1.0, "features": [1.0]},
    {"id": "s12", "tool": "setBarScale", "value": 1.2, "features": [1.2]},
    {"id": "s15", "tool": "setBarScale", "value": 1.5, "features": [1.5]},
]


class TestLinearAlgebra(unittest.TestCase):
    def test_chol_solve_recovers_identity(self):
        A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        L = G._chol(A)
        for col in range(3):
            e = [1.0 if i == col else 0.0 for i in range(3)]
            x = G._chol_solve(L, e)
            got = [sum(A[i][j] * x[j] for j in range(3))
                   for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(got[i], e[i], places=8)

    def test_chol_rejects_indefinite(self):
        with self.assertRaises(ValueError):
            G._chol([[1.0, 2.0], [2.0, 1.0]])

    def test_jitter_rescues_near_singular(self):
        L = G._chol([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
        self.assertEqual(L[0][0], 1.0)


class TestProbit(unittest.TestCase):
    def test_log_phi_stable_in_tails(self):
        self.assertTrue(math.isfinite(G._log_phi(-40.0)))
        self.assertTrue(math.isfinite(G._log_phi(40.0)))

    def test_score_is_positive_and_decays(self):
        self.assertGreater(G._probit_score(0.0), 0.0)
        self.assertLess(G._probit_score(10.0), G._probit_score(2.0))

    def test_curvature_near_mode_region(self):
        # strongly negative around the decision region; the far tail
        # may carry ~1e-10 float noise, which the GP prior dominates
        self.assertLess(G._probit_curvature(0.0), -0.1)
        self.assertLess(G._probit_curvature(1.0), -0.01)
        self.assertLess(abs(G._probit_curvature(-8.0)), 1e-6)


class TestModel(unittest.TestCase):
    def test_direct_winner_ranks_first(self):
        m = G.GPPreferenceModel(ITEMS)
        m.record("s12", "s08")
        m.record("s12", "s10")
        m.record("s12", "s15")
        r = m.fit()
        self.assertEqual(r["ranking"][0], "s12")

    def test_transitive_winners_fill_top(self):
        m = G.GPPreferenceModel(ITEMS)
        m.record("s15", "s08")
        m.record("s12", "s15")
        top2 = set(m.fit()["ranking"][:2])
        self.assertEqual(top2, {"s12", "s15"})

    def test_no_pairs_gives_prior_order(self):
        m = G.GPPreferenceModel(ITEMS)
        r = m.fit()
        self.assertEqual(len(r["ranking"]), 4)
        self.assertEqual(r["n_pairs"], 0)

    def test_next_question_avoids_asked_pairs(self):
        m = G.GPPreferenceModel(ITEMS)
        m.record("s12", "s08")
        q = m.next_question()
        self.assertIsNotNone(q)
        a, b = q
        self.assertNotEqual({a, b}, {"s08", "s12"})

    def test_item_cap(self):
        items = [{"id": str(i), "features": [float(i)]}
                 for i in range(60)]
        with self.assertRaises(ValueError):
            G.GPPreferenceModel(items)

    def test_unknown_item_rejected(self):
        m = G.GPPreferenceModel(ITEMS)
        with self.assertRaises(ValueError):
            m.record("s12", "ghost")

    def test_deterministic(self):
        m1 = G.GPPreferenceModel(ITEMS)
        m1.record("s12", "s08")
        m1.record("s12", "s10")
        m2 = G.GPPreferenceModel(ITEMS)
        m2.record("s12", "s08")
        m2.record("s12", "s10")
        self.assertEqual(m1.fit(), m2.fit())


class TestValidation(unittest.TestCase):
    def test_in_range_passes(self):
        v = G.validate_item({"tool": "setBarScale", "value": 1.2})
        self.assertTrue(v["validated"])

    def test_out_of_range_rejected_never_clamped(self):
        v = G.validate_item({"tool": "setBarScale", "value": 99.0})
        self.assertFalse(v["validated"])
        self.assertIn("never clamped", v["reason"])

    def test_unknown_tool_rejected(self):
        v = G.validate_item({"tool": "setNope", "value": 1})
        self.assertFalse(v["validated"])

    def test_enum_value_checked(self):
        v = G.validate_item({"tool": "setBarPosition", "value": "top"})
        self.assertTrue(v["validated"])
        v = G.validate_item({"tool": "setBarPosition",
                             "value": "diagonal"})
        self.assertFalse(v["validated"])


if __name__ == "__main__":
    unittest.main()
