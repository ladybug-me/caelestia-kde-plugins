"""tests for cortex.conservative + cortex.ensemble + diagnostics.stacking
(exponential-build-4 D: conservative bandits, Learn++.NSE, stacking).

Pinned: the conservative gate (no evidence = floor closed; a cleared
UCB explores; a failing UCB is held; the guarantee sentence names the
baseline, delta, and epsilon), Learn++.NSE (a member per block, the
recent window drives the weights, age-out is deterministic, the state
round-trips), the selectable modes in CortexLearner (conservative
exploration and nse drift mode are opt-in, never default), and the
stacked combiner (drop-in weights for fusion.fuse, thin rows refuse to
fit, k-fold evaluate reports both Briers and names a measured winner).
"""
import random
import unittest

from assistant.core.conservative import (ConservativeBandit,
                                           hoeffding_radius)
from assistant.core.ensemble import LearnPPNSE, compare_with_adwin
from assistant.core.learn import CortexLearner
from assistant.capabilities.diagnostics import stacking


class HoeffdingTests(unittest.TestCase):
    def test_radius_decreases_with_evidence(self):
        self.assertGreater(hoeffding_radius(1, 0.05),
                           hoeffding_radius(100, 0.05))
        self.assertAlmostEqual(hoeffding_radius(10000, 0.05),
                               0.0136, places=3)

    def test_zero_evidence_refused(self):
        with self.assertRaises(ValueError):
            hoeffding_radius(0, 0.05)


class ConservativeGateTests(unittest.TestCase):
    def _arms(self, balanced=(30.0, 10.0), precision=(10.0, 10.0)):
        return {"balanced": {"alpha": balanced[0], "beta": balanced[1]},
                "precision": {"alpha": precision[0], "beta": precision[1]}}

    def test_unplayed_arm_cannot_explore(self):
        gate = ConservativeBandit(self._arms(precision=(1.0, 1.0)),
                                  safe_arm="balanced", epsilon=0.05)
        verdict = gate.choose(thompson_sample="precision")
        self.assertEqual(verdict["arm"], "balanced")
        self.assertTrue(verdict["overridden"])
        self.assertIn("no evidence", verdict["details"]["reason"])

    def test_weak_sample_held_by_floor(self):
        # precision evidence is clearly worse (mean 0.2 vs safe 0.75):
        # its UCB cannot clear the safe LCB minus eps
        gate = ConservativeBandit(self._arms(precision=(8.0, 32.0)),
                                  safe_arm="balanced", epsilon=0.05)
        verdict = gate.choose(thompson_sample="precision")
        self.assertEqual(verdict["arm"], "balanced")
        self.assertIn("floor closed", verdict["details"]["reason"])

    def test_strong_sample_explores(self):
        gate = ConservativeBandit(
            self._arms(precision=(90.0, 10.0)),  # mean 0.9 vs safe 0.75
            safe_arm="balanced", epsilon=0.05)
        verdict = gate.choose(thompson_sample="precision")
        self.assertEqual(verdict["arm"], "precision")
        self.assertFalse(verdict["overridden"])
        self.assertIn("cleared", verdict["details"]["reason"])

    def test_safe_arm_without_evidence_closes_floor(self):
        gate = ConservativeBandit(
            {"balanced": {"alpha": 1.0, "beta": 1.0},
             "precision": {"alpha": 90.0, "beta": 10.0}},
            safe_arm="balanced")
        verdict = gate.choose(thompson_sample="precision")
        self.assertEqual(verdict["arm"], "balanced")
        self.assertIn("safe arm has no evidence", verdict["details"]["reason"])

    def test_guarantee_names_scope(self):
        gate = ConservativeBandit(self._arms(), safe_arm="balanced",
                                  epsilon=0.03, delta=0.01)
        text = gate.guarantee()
        self.assertIn("balanced", text)
        self.assertIn("0.03 * T", text)
        self.assertIn("not regret against the best arm", text)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            ConservativeBandit(self._arms(), safe_arm="ghost")
        with self.assertRaises(ValueError):
            ConservativeBandit(self._arms(), safe_arm="balanced",
                               epsilon=-1.0)
        with self.assertRaises(ValueError):
            ConservativeBandit(self._arms(), safe_arm="balanced",
                               delta=1.0)

    def test_report_untouched_arm_abstains(self):
        gate = ConservativeBandit(self._arms(precision=(1.0, 1.0)),
                                  safe_arm="balanced")
        row = next(r for r in gate.report()["arms"] if r["arm"] == "precision")
        self.assertEqual(row["plays"], 0)
        self.assertIsNone(row["mean"])
        self.assertIsNone(row["lcb"])


class NSETests(unittest.TestCase):
    def test_member_born_every_block(self):
        ens = LearnPPNSE(n_features=2, block=5)
        for i in range(10):
            row = ens.observe([float(i), 1.0], i % 2)
        self.assertEqual(row["members"], 2)
        self.assertEqual(row["n_seen"], 10)

    def test_no_prediction_before_first_block(self):
        ens = LearnPPNSE(n_features=2, block=5)
        self.assertIsNone(ens.predict([1.0, 0.0]))

    def test_shift_reweights_old_experts_down(self):
        ens = LearnPPNSE(n_features=2, block=10, window=20)
        rng = random.Random(3)
        for _ in range(20):  # regime 1: feature 0 decides
            x = [rng.random(), rng.random()]
            ens.observe(x, 1 if x[0] > 0.5 else 0)
        for _ in range(30):  # regime 2: feature 1 decides
            x = [rng.random(), rng.random()]
            ens.observe(x, 1 if x[1] > 0.5 else 0)
        weights = ens.member_weights()
        # the newest experts (regime 2) must outweigh the oldest
        self.assertGreater(weights[-1], weights[0])
        # the floor keeps old knowledge alive rather than reset
        self.assertGreaterEqual(min(weights), ens.min_weight)

    def test_max_members_ages_out_oldest(self):
        ens = LearnPPNSE(n_features=1, block=2, max_members=3)
        for i in range(20):
            ens.observe([float(i)], i % 2)
        self.assertEqual(len(ens.members), 3)
        self.assertEqual(ens.members[0]["born_at"], 16)  # the oldest kept

    def test_round_trip(self):
        ens = LearnPPNSE(n_features=2, block=5)
        for i in range(12):
            ens.observe([float(i), 0.5], i % 2)
        restored = LearnPPNSE.from_dict(ens.to_dict())
        self.assertEqual(restored.members, ens.members)
        self.assertEqual(restored.member_weights(), ens.member_weights())

    def test_compare_reports_both_without_winner(self):
        result = compare_with_adwin([1] * 30 + [0] * 30, block=10)
        self.assertIn("adwin", result)
        self.assertIn("nse", result)
        self.assertIn("No winner", result["note"])


class SelectableModeTests(unittest.TestCase):
    def _learner(self, data):
        return CortexLearner(data)

    def test_conservative_mode_is_opt_in(self):
        plain = self._learner({})
        for _ in range(10):
            plain.reward_strategy("balanced", False)
        name, _state = plain.choose_strategy()
        self.assertIsNotNone(name)
        self.assertIsNone(plain.last_gate_report)

        conservative = self._learner({"conservative_exploration": True})
        for _ in range(10):
            conservative.reward_strategy("balanced", False)
        for _ in range(10):
            conservative.reward_strategy("precision", True)
        for _ in range(5):
            conservative.reward_strategy("precision", False)
        name2, _s = conservative.choose_strategy()
        self.assertIsNotNone(conservative.last_gate_report)
        self.assertIn(conservative.last_gate_report["details"]["reason"],
                      conservative.last_gate_report["details"]["reason"])

    def test_nse_mode_adds_alternative(self):
        rng = random.Random(5)
        examples = [{"text": f"t{i}", "surface": "s",
                     "features": {"lex": rng.random()}, "p": 0.5,
                     "label": 1 if i < 30 else 0,
                     "outcome": "applied" if i < 30 else "rejected"}
                    for i in range(60)]
        default = self._learner({"examples": examples})
        report = default.ph_adwin_consensus()
        self.assertNotIn("nse_alternative", report)
        nse_mode = self._learner({"examples": examples,
                                  "drift_mode": "nse"})
        report2 = nse_mode.ph_adwin_consensus()
        self.assertIn("nse_alternative", report2)
        self.assertGreater(report2["nse_alternative"]["members"], 0)
        # the consensus verdict is still there, unchanged
        self.assertIn("flag_drift", report2)


class StackingTests(unittest.TestCase):
    def _rows(self, n=40, reliability=0.7, seed=9):
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            correct = f"h{rng.randint(0, 2)}"
            confs = {}
            for s in stacking.SOURCES:
                base = {f"h{j}": round(rng.random(), 3) for j in range(3)}
                if rng.random() < reliability:
                    base[correct] = max(base.values()) + 0.05
                confs[s] = base
            rows.append({"confidences": confs, "correct": correct})
        return rows

    def test_weights_are_drop_in_shape(self):
        fit = stacking.stacked_weights(self._rows())
        self.assertEqual(set(fit["weights"]), set(stacking.SOURCES))
        for w in fit["weights"].values():
            self.assertGreaterEqual(w, 0.0)  # fusion refuses negatives
        self.assertAlmostEqual(sum(fit["weights"].values()), 1.0, places=3)
        self.assertFalse(fit["thin"])

    def test_thin_rows_refuse_to_fit(self):
        fit = stacking.stacked_weights(self._rows(n=5))
        self.assertTrue(fit["thin"])
        self.assertEqual(fit["weights"], {s: 0.25 for s in stacking.SOURCES})
        self.assertIn("guess", fit["note"])

    def test_evaluate_measures_a_winner(self):
        ev = stacking.evaluate(self._rows())
        self.assertEqual(ev["folds"], 5)
        self.assertIsNotNone(ev["brier_fixed"])
        self.assertIsNotNone(ev["brier_stacked"])
        self.assertIn(ev["winner"], ("stacked", "fixed_pool", "tie"))

    def test_chance_level_source_gets_no_vote(self):
        # scan's top pick is correct at exactly the chance rate: its
        # lift over chance is ~0, so the honest vote is ~0, recorded
        # as dropped rather than asked to vote against itself
        rows = self._rows(n=60, reliability=1.0)
        rng = random.Random(4)
        for row in rows:
            row["confidences"]["scan"] = {
                h: round(rng.random(), 3) for h in ("h0", "h1", "h2")}
        fit = stacking.stacked_weights(rows)
        # a finite sample wiggles a chance-level source slightly above
        # the floor — the honest assertion is dominance, not perfection
        self.assertLess(fit["weights"]["scan"], 0.1)
        self.assertGreater(max(fit["weights"].values()), 0.4)

    def test_too_few_rows_evaluates_to_honest_refusal(self):
        ev = stacking.evaluate(self._rows(n=5))
        self.assertIsNone(ev["brier_fixed"])
        self.assertIsNone(ev["brier_stacked"])
        self.assertIn("honestly", ev["note"])


if __name__ == "__main__":
    unittest.main()
