"""Tests for exponential-build phase 3.3: the regret-vs-best-fixed-arm
audit (`brain/regret.py`, surfaced in `cortex report` and
`brain calibration`).

The contract under test:

- cumulative reward, per-arm means and the best-fixed-arm-in-hindsight
  estimate are derived from the EVIDENCE (the flat prior's own mass is
  never counted as plays or rewards);
- the best-fixed total and the regret are labeled ESTIMATES over
  unobserved rounds — the method string says so on every report;
- a bandit with one played arm has no comparison (regret 0 by
  definition, stated); an untouched bandit is an abstention, not a
  zero; determinism everywhere;
- both input shapes work: aggregate Beta posteriors and explicit
  per-round draw logs.
"""
import unittest

from assistant.capabilities.brain.regret import audit_from_arms, audit_from_draws


class ArmsAuditTests(unittest.TestCase):
    def test_cumulative_and_best_fixed_from_aggregates(self):
        # A: 8 wins / 2 losses (mean 0.8); B: 2/8 (0.25); C: 5/5 (0.5)
        arms = {"A": [1 + 8, 1 + 2], "B": [1 + 2, 1 + 8], "C": [1 + 5, 1 + 5]}
        out = audit_from_arms(arms)
        self.assertEqual(out["status"], "compared")
        self.assertEqual(out["horizon"], 30)
        self.assertAlmostEqual(out["cumulative_reward"], 15.0)
        self.assertEqual(out["best_fixed_arm"], "A")
        self.assertAlmostEqual(out["best_fixed_estimate"], 0.8 * 30)
        self.assertAlmostEqual(out["estimated_regret"], 9.0)

    def test_prior_mass_is_never_counted_as_evidence(self):
        # C never played: its flat prior must not inflate the horizon
        arms = {"A": [1 + 4, 1 + 0], "B": [1 + 0, 1 + 4], "C": [1.0, 1.0]}
        out = audit_from_arms(arms)
        self.assertEqual(out["horizon"], 8)
        by_arm = {a["arm"]: a for a in out["per_arm"]}
        self.assertEqual(by_arm["C"]["plays"], 0)
        self.assertIsNone(by_arm["C"]["mean_reward"])

    def test_single_played_arm_has_no_comparison(self):
        out = audit_from_arms({"A": [1 + 3, 1 + 1], "B": [1.0, 1.0]})
        self.assertEqual(out["status"], "single-arm")
        self.assertEqual(out["estimated_regret"], 0.0)
        self.assertIn("no policy comparison", out["note"])

    def test_untouched_bandit_abstains(self):
        out = audit_from_arms({"A": [1.0, 1.0], "B": [1.0, 1.0]})
        self.assertEqual(out["status"], "abstained")
        self.assertNotIn("estimated_regret", out)

    def test_method_caveat_travels_on_every_report(self):
        for out in (audit_from_arms({"A": [1 + 2, 1 + 2],
                                     "B": [1 + 1, 1 + 3]}),
                    audit_from_draws([("A", 1), ("B", 0)])):
            self.assertIn("estimate", out["method"].lower())


class DrawAuditTests(unittest.TestCase):
    def test_exact_counts_from_a_draw_log(self):
        draws = [("A", 1), ("B", 0), ("A", 1), ("A", 0), ("B", 0)]
        out = audit_from_draws(draws)
        self.assertEqual(out["horizon"], 5)
        self.assertAlmostEqual(out["cumulative_reward"], 2.0)
        self.assertEqual(out["best_fixed_arm"], "A")
        # best mean 2/3 x 5 rounds = 3.333; regret 3.333 - 2 = 1.333
        self.assertAlmostEqual(out["estimated_regret"], 4.0 / 3.0, places=3)

    def test_broken_arm_cannot_win_hindsight(self):
        draws = [("A", 1), ("A", 1), ("B", 0), ("B", 0)]
        out = audit_from_draws(draws)
        self.assertEqual(out["best_fixed_arm"], "A")
        self.assertAlmostEqual(out["estimated_regret"], 2.0)

    def test_empty_draw_log_abstains(self):
        out = audit_from_draws([])
        self.assertEqual(out["status"], "abstained")

    def test_deterministic(self):
        draws = [("A", 1), ("B", 0), ("A", 1)]
        self.assertEqual(audit_from_draws(draws), audit_from_draws(draws))
        arms = {"A": [1 + 2, 1 + 1], "B": [1 + 0, 1 + 3]}
        self.assertEqual(audit_from_arms(arms), audit_from_arms(arms))


class WiringTests(unittest.TestCase):
    def test_cortex_report_includes_the_strategy_bandit_audit(self):
        from assistant.core.learn import CortexLearner
        learner = CortexLearner()
        learner.reward_strategy("lexical", True)
        learner.reward_strategy("lexical", True)
        learner.reward_strategy("semantic", False)
        out = learner.report()["regret"]
        self.assertEqual(out["status"], "compared")
        self.assertEqual(out["best_fixed_arm"], "lexical")


if __name__ == "__main__":
    unittest.main()
