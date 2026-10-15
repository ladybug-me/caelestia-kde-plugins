"""cortex.conformal tests — coverage guarantee, committee disagreement, drift."""
import random
import unittest

from assistant.core.conformal import (ConformalCalibrator,
                                        PageHinkleyDrift, query_by_committee)


class TestConformal(unittest.TestCase):
    def _calibrated(self, seed=7):
        """Accepted routes score U(0.8, 1.0), rejected U(0.0, 0.6) — the
        conformal threshold must separate them with the promised coverage."""
        c = ConformalCalibrator(alpha=0.25)
        rng = random.Random(seed)
        for _ in range(100):
            c.observe(rng.uniform(0.8, 1.0), "applied")
            c.observe(rng.uniform(0.0, 0.6), "rejected")
        return c

    def test_threshold_separates_outcomes(self):
        c = self._calibrated()
        t = c.threshold()
        self.assertIsNotNone(t)
        good = c.verdict(min(1.0, t + 0.02))
        bad = c.verdict(max(0.0, t - 0.5))
        self.assertTrue(good["covered"])
        self.assertFalse(bad["covered"])
        self.assertIn("distribution-free", good["guarantee"])

    def test_empirical_accuracy_of_covered_routes(self):
        """THE coverage test: of fresh routes the calibrator says are
        covered, at least (1 - alpha - margin) must actually be accepted."""
        c = self._calibrated()
        rng = random.Random(99)
        covered_correct, covered_total = 0, 0
        for _ in range(400):
            accepted = rng.random() < 0.5
            score = rng.uniform(0.8, 1.0) if accepted else rng.uniform(0.0, 0.6)
            v = c.verdict(score)
            if v["covered"]:
                covered_total += 1
                covered_correct += accepted
        self.assertGreater(covered_total, 25)
        self.assertGreaterEqual(covered_correct / covered_total, 1.0 - 0.25 - 0.05)

    def test_insufficient_data_is_honest(self):
        c = ConformalCalibrator()
        for _ in range(3):
            c.observe(0.9, "applied")
        v = c.verdict(0.9)
        self.assertIsNone(v["covered"])
        self.assertIn("insufficient", v["reason"])

    def test_roundtrip(self):
        c = self._calibrated()
        c2 = ConformalCalibrator().from_dict(c.to_dict())
        self.assertEqual(c.verdict(0.9), c2.verdict(0.9))


class TestQbC(unittest.TestCase):
    def test_disagreement_ranked_first(self):
        cands = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        votes = [
            {0: 0.9, 1: 0.5, 2: 0.5},   # strategy 1
            {0: 0.2, 1: 0.5, 2: 0.5},   # strategy 2 disagrees on "a"
        ]
        out = query_by_committee(cands, votes)
        self.assertEqual(out[0]["text"], "a")
        dis = {row["text"]: row["disagreement"] for row in out}
        self.assertGreater(dis["a"], dis["c"])

    def test_unanimous_committee_no_disagreement(self):
        cands = [{"text": "x"}, {"text": "y"}]
        votes = [{0: 0.9, 1: 0.9}, {0: 0.9, 1: 0.9}]
        out = query_by_committee(cands, votes)
        self.assertEqual(out[0]["disagreement"], 0.0)
        self.assertEqual(out[1]["disagreement"], 0.0)

    def test_length_mismatch_refused(self):
        with self.assertRaises(ValueError):
            query_by_committee([{"text": "a"}], [{0: 1.0, 1: 1.0}])

    def test_both_low_is_not_disagreement(self):
        # the entropy failure mode: 0.05/0.05 is agreement, not confusion
        cands = [{"text": "low"}, {"text": "split"}]
        votes = [{0: 0.05, 1: 0.9}, {0: 0.05, 1: 0.2}]
        out = query_by_committee(cands, votes)
        self.assertEqual(out[0]["text"], "split")


class TestDrift(unittest.TestCase):
    def test_alarms_after_true_drop(self):
        d = PageHinkleyDrift(threshold=5.0, warmup=10)
        rng = random.Random(3)
        alarm_seen = False
        for i in range(300):
            accepted = rng.random() < (0.9 if i < 100 else 0.15)
            r = d.update(accepted)
            alarm_seen = alarm_seen or r["alarmed"]
        self.assertTrue(alarm_seen)
        self.assertTrue(d.status()["alarmed"])

    def test_quiet_when_stable(self):
        d = PageHinkleyDrift(threshold=20.0, warmup=30)
        rng = random.Random(5)
        for _ in range(500):
            d.update(rng.random() < 0.8)
        self.assertFalse(d.status()["alarmed"])

    def test_reset_clears_latch(self):
        d = PageHinkleyDrift(threshold=5.0, warmup=10)
        rng = random.Random(3)
        for i in range(300):
            d.update(rng.random() < (0.9 if i < 100 else 0.15))
        self.assertTrue(d.status()["alarmed"])
        d.reset()
        self.assertFalse(d.status()["alarmed"])
        self.assertEqual(d.status()["n"], 0)

    def test_roundtrip(self):
        d = PageHinkleyDrift()
        for _ in range(40):
            d.update(True)
        d2 = PageHinkleyDrift().from_dict(d.to_dict())
        self.assertEqual(d.status(), d2.status())


if __name__ == "__main__":
    unittest.main()
