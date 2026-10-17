"""Tests for exponential-build phase 3.1: the shared hierarchical
partial-pooling utility (Efron & Morris 1975) in brain/pooling.py.

The contract under test:

- a NEW arm's hierarchical prior starts AT the pooled mean with k0
  pseudo-counts (not the flat prior), and only the pool's own spread
  sets how strong that pull is (method of moments, hand-checkable);
- a pool whose arms all agree pools FULLY; a pool with NO evidence
  anywhere honestly degenerates to the flat prior and says so; arms
  with no observed evidence are listed, never fabricated into the pool;
- every arm's shrunk estimate reports its shrinkage weight, which
  decays as its own evidence grows;
- deterministic, pure, hand-computed values pinned exactly.
"""
import unittest

from assistant.capabilities.brain.pooling import (hierarchical_prior, pool_arms,
                                     shrunk_mean)

# three arms with distinct evidence (all seeded from the flat prior):
#   A: [9, 3]   -> s=8, n=10, p=0.8
#   B: [3, 9]   -> s=2, n=10, p=0.2
#   C: [6, 6]   -> s=5, n=10, p=0.5
ARMS = {"A": [9.0, 3.0], "B": [3.0, 9.0], "C": [6.0, 6.0]}


class PoolingTests(unittest.TestCase):
    def test_hand_computed_moments(self):
        pooled = pool_arms(ARMS)
        self.assertTrue(pooled["pooled"])
        self.assertAlmostEqual(pooled["mu0"], 0.5)          # 15/30
        # var(0.8, 0.2, 0.5) = 0.06; sampling var mean = 0.25/10 = 0.025
        self.assertAlmostEqual(pooled["tau2"], 0.035, places=6)
        # k0 = 0.25 / 0.035 = 50/7
        self.assertAlmostEqual(pooled["k0"], 50.0 / 7.0, places=4)

    def test_new_arm_starts_at_the_pooled_mean(self):
        prior = hierarchical_prior(pool_arms(ARMS))
        self.assertAlmostEqual(prior["prior_mean"], 0.5, places=4)
        self.assertAlmostEqual(prior["alpha"], (50.0 / 7.0) * 0.5, places=4)
        self.assertAlmostEqual(prior["beta"], (50.0 / 7.0) * 0.5, places=4)
        self.assertFalse(prior["is_flat"])

    def test_agreeing_arms_pool_fully(self):
        arms = {"A": [9.0, 3.0], "B": [5.0, 2.0],   # p = 0.8 all three
                "C": [9.0, 3.0]}
        pooled = pool_arms(arms)
        # tau2 = 0 (identical rates): k0 caps at the total evidence
        self.assertAlmostEqual(pooled["k0"], pooled["total_evidence"])
        prior = hierarchical_prior(pooled)
        self.assertAlmostEqual(prior["prior_mean"], 0.8, places=4)

    def test_no_evidence_anywhere_degenerates_to_flat_honestly(self):
        arms = {"A": [1.0, 1.0], "B": [1.0, 1.0]}
        pooled = pool_arms(arms)
        self.assertFalse(pooled["pooled"])
        self.assertEqual(sorted(pooled["no_evidence_arms"]), ["A", "B"])
        prior = hierarchical_prior(pooled)
        self.assertTrue(prior["is_flat"])
        self.assertEqual((prior["alpha"], prior["beta"]), (1.0, 1.0))

    def test_unequal_evidence_weights_the_pool_mean(self):
        arms = {"big": [1.0 + 80.0, 1.0 + 20.0],   # p=0.8, n=100
                "tiny": [1.0, 1.0]}                # no evidence -> excluded
        pooled = pool_arms(arms)
        self.assertAlmostEqual(pooled["mu0"], 0.8)
        self.assertEqual(pooled["no_evidence_arms"], ["tiny"])
        # ...but an arm WITH evidence does pull the mean (81/102,
        # compared at the module's 6-decimal reporting precision)
        pooled2 = pool_arms({"big": [1.0 + 80.0, 1.0 + 20.0],
                             "tiny": [2.0, 2.0]})
        self.assertAlmostEqual(pooled2["mu0"], 81.0 / 102.0, places=5)

    def test_extreme_pool_stays_bounded(self):
        arms = {"A": [1.0 + 10.0, 1.0], "B": [1.0 + 5.0, 1.0]}
        pooled = pool_arms(arms)  # mu0 = 1.0
        self.assertEqual(pooled["mu0"], 1.0)
        prior = hierarchical_prior(pooled)
        # the 0.5 Jeffreys floor keeps the prior proper (never beta=0)
        self.assertGreaterEqual(prior["beta"], 0.5)
        self.assertGreater(prior["prior_mean"], 0.9)

    def test_empty_pool_refused(self):
        with self.assertRaises(ValueError):
            pool_arms({})


class ShrinkageTests(unittest.TestCase):
    def test_shrunk_mean_pulls_toward_the_pool(self):
        pooled = pool_arms(ARMS)
        # arm B (p=0.2, n=10) shrinks toward mu0=0.5
        out = shrunk_mean(3.0, 9.0, pooled)
        self.assertGreater(out["estimate"], 0.2)
        self.assertLess(out["estimate"], 0.5)
        self.assertAlmostEqual(out["shrinkage_weight"],
                               (50.0 / 7.0) / (10 + 50.0 / 7.0), places=4)

    def test_shrinkage_decays_as_evidence_grows(self):
        pooled = pool_arms(ARMS)
        small = shrunk_mean(1.0 + 1.0, 1.0 + 1.0, pooled)    # n=2
        large = shrunk_mean(1.0 + 80.0, 1.0 + 20.0, pooled)  # n=100
        self.assertGreater(small["shrinkage_weight"],
                           large["shrinkage_weight"])
        # ...and the large arm's own evidence dominates its estimate
        # (k0 ~ 7.1 still pulls ~2 points; its weight is ~6.7%)
        self.assertLess(abs(large["estimate"] - 0.8), 0.05)
        self.assertLess(large["shrinkage_weight"], 0.1)

    def test_deterministic(self):
        self.assertEqual(pool_arms(ARMS), pool_arms(ARMS))
        self.assertEqual(hierarchical_prior(pool_arms(ARMS)),
                         hierarchical_prior(pool_arms(ARMS)))


if __name__ == "__main__":
    unittest.main()
