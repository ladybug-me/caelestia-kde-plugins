"""brain.bursts tests — Hawkes fit, burst detection, honesty."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import bursts as B  # noqa: E402


def background_then_loop(seed=1, n_bg=30, loop_start=200.0, n_loop=20):
    """Deterministic mixture: a Poisson-ish background plus one tight
    crash loop — the canonical scenario."""
    rng = random.Random(seed)
    bg = sorted(rng.uniform(0.0, 100.0) for _ in range(n_bg))
    loop = [loop_start + i * 0.3 for i in range(n_loop)]
    return sorted([round(x, 3) for x in bg + loop])


class TestFit(unittest.TestCase):
    def test_fit_is_deterministic(self):
        ev = background_then_loop()
        a = B.fit_hawkes(ev)
        b = B.fit_hawkes(ev)
        self.assertEqual(a, b)

    def test_fit_shape_and_stability(self):
        fit = B.fit_hawkes(background_then_loop())
        for key in ("mu", "alpha", "beta", "branching_ratio"):
            self.assertIn(key, fit)
        self.assertGreater(fit["mu"], 0)
        self.assertTrue(fit["stable"],
                        "a 20-event loop over 30 background events "
                        "should stay subcritical")

    def test_too_few_events_raises(self):
        with self.assertRaises(ValueError):
            B.fit_hawkes([1.0, 2.0, 3.0])

    def test_events_are_sorted_internally(self):
        ev = background_then_loop()[::-1]
        a = B.fit_hawkes(list(reversed(ev)))
        b = B.fit_hawkes(ev)
        self.assertEqual(a, b)


class TestBursts(unittest.TestCase):
    def test_crash_loop_is_found_and_dominates(self):
        r = B.burst_report(background_then_loop())
        self.assertEqual(r["verdict"], "OK")
        self.assertGreaterEqual(r["n_bursts"], 1)
        top = max(r["bursts"], key=lambda b: b["over_base"])
        self.assertGreaterEqual(top["start"], 199.0)
        self.assertLessEqual(top["end"], 206.0)
        self.assertGreater(top["over_base"], 5.0,
                           "the loop must stand far above the base rate")

    def test_caveat_always_present(self):
        r = B.burst_report(background_then_loop())
        self.assertIn("CORRELATION, not causation", r["caveat"])

    def test_pure_poisson_has_weak_or_no_bursts(self):
        rng = random.Random(5)
        ev = sorted(round(rng.uniform(0, 200), 3) for _ in range(25))
        r = B.burst_report(ev, factor=4.0)
        top = max((b["over_base"] for b in r["bursts"]), default=0.0)
        self.assertLess(top, 15.0, "pure noise should not explode")

    def test_abstains_past_cap(self):
        r = B.burst_report([float(i) for i in range(6000)],
                           factor=2.0)
        self.assertEqual(r["verdict"], "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
