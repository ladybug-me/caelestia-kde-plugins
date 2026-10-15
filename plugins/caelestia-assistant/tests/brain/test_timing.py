"""Tests for exponential-build phase 3.4: attention-aware suggestion
timing (`brain/timing.py`).

The contract under test:

- the timing bandit's arms are seeded from the POOLED prior of the
  user's existing proposal posteriors (the 3.1 consumer promise) — a
  user who approves ~80% overall gets timing arms centered near 0.8,
  not the flat 0.5;
- the recommendation biases WHEN, never WHAT: the output is a ranked
  list of buckets plus deferral advice, and the note says the caller
  still proposes through the ledger;
- the rhythm engine's activity z-scores and the historical decision
  latency both move the ranking in the stated direction;
- a falling Holt forecast of the acceptance rate advises deferring
  (reported, never enforced);
- deterministic under a seeded rng; bucket math is exact.
"""
import random
import unittest

from assistant.capabilities.brain.timing import (BUCKET_NAMES, bucket_of,
                                    suggest_surface_bucket)


class BucketMathTests(unittest.TestCase):
    def test_bucket_of_is_exact(self):
        self.assertEqual(bucket_of(0), 0)
        self.assertEqual(bucket_of(3), 0)
        self.assertEqual(bucket_of(4), 1)
        self.assertEqual(bucket_of(23), 5)
        self.assertEqual(bucket_of(24), 0)  # wraps like a clock


class PriorPoolingTests(unittest.TestCase):
    def test_arms_seed_from_the_pooled_prior_not_flat(self):
        # a user who approves ~80% of proposals across their tools
        seed = {"tool:a": [1 + 8, 1 + 2], "tool:b": [1 + 8, 1 + 2]}
        out = suggest_surface_bucket([], now_hour=10, seed_arms=seed,
                                     rng=random.Random(7))
        self.assertTrue(all(abs(r["prior_mean"] - 0.8) < 0.05
                            for r in out["ranked"]))
        self.assertFalse(out["prior_used"]["is_flat"])

    def test_no_seed_falls_back_to_flat_honestly(self):
        out = suggest_surface_bucket([], now_hour=10,
                                     rng=random.Random(7))
        self.assertTrue(out["prior_used"]["is_flat"])
        self.assertAlmostEqual(out["ranked"][0]["prior_mean"], 0.5)


class RankingTests(unittest.TestCase):
    def test_evening_history_beats_dormant_history(self):
        events = ([{"hour": h, "applied": True, "latency_hours": 0.5}
                   for h in (18, 19, 20, 21, 22, 23)]
                  + [{"hour": h, "applied": False, "latency_hours": 40.0}
                     for h in (2, 3, 4, 5, 6, 7)])
        out = suggest_surface_bucket(events, now_hour=17, rng=random.Random(3),
                                     horizon_hours=24)
        names = [r["name"] for r in out["ranked"]]
        self.assertLess(names.index("evening"), names.index("night"))

    def test_rhythm_activity_boosts_the_bucket(self):
        # heavy activity 18-23, quiet elsewhere
        activities = [18, 19, 20, 21, 22, 23] * 10
        out = suggest_surface_bucket([], now_hour=17, activities=activities,
                                     horizon_hours=24, rng=random.Random(5))
        evening = next(r for r in out["ranked"] if r["name"] == "evening")
        night = next(r for r in out["ranked"] if r["name"] == "night")
        self.assertGreater(evening["rhythm_multiplier"],
                           night["rhythm_multiplier"])

    def test_latency_history_boosts_the_attentive_bucket(self):
        events = ([{"hour": 18, "applied": True, "latency_hours": 0.2}] * 6
                  + [{"hour": 10, "applied": True, "latency_hours": 20.0}] * 6)
        out = suggest_surface_bucket(events, now_hour=17, horizon_hours=24,
                                     rng=random.Random(11))
        evening = next(r for r in out["ranked"] if r["name"] == "evening")
        midday = next(r for r in out["ranked"] if r["name"] == "midday")
        self.assertGreater(evening["latency_multiplier"],
                           midday["latency_multiplier"])


class TrendAndHonestyTests(unittest.TestCase):
    def test_falling_trend_advises_deferring(self):
        out = suggest_surface_bucket([], now_hour=10,
                                     recent_accept_series=[0.9, 0.7, 0.5,
                                                           0.35, 0.2],
                                     rng=random.Random(1))
        self.assertTrue(out["defer_advised"])
        self.assertIn("deferring", out["trend_note"])

    def test_rising_trend_does_not_defer(self):
        out = suggest_surface_bucket([], now_hour=10,
                                     recent_accept_series=[0.2, 0.4, 0.6,
                                                           0.8],
                                     rng=random.Random(1))
        self.assertFalse(out["defer_advised"])

    def test_output_biases_when_not_what(self):
        out = suggest_surface_bucket([], now_hour=10, rng=random.Random(2))
        self.assertIn("WHEN, not WHAT", out["note"])
        self.assertIn("ledger", out["note"])

    def test_invalid_hour_refused(self):
        with self.assertRaises(ValueError):
            suggest_surface_bucket([], now_hour=25, rng=random.Random(1))

    def test_seeded_rng_is_reproducible(self):
        a = suggest_surface_bucket([], now_hour=10, rng=random.Random(42))
        b = suggest_surface_bucket([], now_hour=10, rng=random.Random(42))
        self.assertEqual(a, b)

    def test_bucket_names_line_up(self):
        self.assertEqual(len(BUCKET_NAMES), 6)
        self.assertEqual(BUCKET_NAMES[bucket_of(20)], "late")
        self.assertEqual(BUCKET_NAMES[bucket_of(9)], "morning")


if __name__ == "__main__":
    unittest.main()
