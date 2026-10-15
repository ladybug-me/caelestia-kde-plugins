"""brain.sketch tests — Space-Saving hitters, t-digest quantiles."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.brain import sketch as S  # noqa: E402


class TestSpaceSaving(unittest.TestCase):
    def test_exact_when_under_capacity(self):
        ss = S.SpaceSaving(k=5)
        for i in range(10):
            ss.add("a", 1)
            ss.add("b", 2)
        hitters = {h["key"]: h for h in ss.heavy_hitters()}
        self.assertEqual(hitters["a"]["count"], 10)
        self.assertEqual(hitters["b"]["count"], 20)
        self.assertEqual(hitters["a"]["max_error"], 0)

    def test_top_dominant_key_survives_evictions(self):
        ss = S.SpaceSaving(k=3)
        ss.add("big", 1000)
        rng = random.Random(1)
        for i in range(50):
            ss.add(f"tiny{i}", 1)
        top = ss.heavy_hitters()[0]
        self.assertEqual(top["key"], "big")
        # Space-Saving guarantee: count is an overestimate, error bounded
        self.assertGreaterEqual(top["count"], 1000)
        self.assertLessEqual(top["count"], 1000 + 50)
        self.assertLessEqual(top["max_error"], 50)

    def test_merge_combines_counts(self):
        s1 = S.SpaceSaving(k=4)
        s1.add("a", 10)
        s2 = S.SpaceSaving(k=4)
        s2.add("a", 5)
        s2.add("b", 3)
        m = s1.merge(s2)
        hitters = {h["key"]: h["count"] for h in m.heavy_hitters()}
        self.assertEqual(hitters["a"], 15)
        self.assertEqual(m.n_total, 18)

    def test_zero_weight_ignored(self):
        ss = S.SpaceSaving(k=3)
        ss.add("x", 0)
        self.assertEqual(ss.n_items, 0)


class TestTDigest(unittest.TestCase):
    def test_exact_on_small_input(self):
        d = S.TDigest(buffer_cap=1000)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in vals:
            d.add(v)
        d._rebuild()
        self.assertEqual(d.quantile(0.0), 1.0)
        self.assertEqual(d.quantile(1.0), 10.0)
        self.assertAlmostEqual(d.quantile(0.5), 5.0, delta=0.6)

    def test_uniform_tail_accuracy(self):
        rng = random.Random(3)
        d = S.TDigest(buffer_cap=100000)
        vals = [rng.uniform(0, 1000) for _ in range(10000)]
        for v in vals:
            d.add(v)
        d._rebuild()
        srt = sorted(vals)
        for q in (0.5, 0.9, 0.95):
            exact = srt[int(q * len(srt))]
            self.assertLessEqual(abs(d.quantile(q) - exact) / 1000.0,
                                 0.01, f"q={q}")

    def test_min_max_survive(self):
        d = S.TDigest()
        for v in (3.0, 1.0, 9.0, 2.0):
            d.add(v)
        self.assertEqual(d.min_value, 1.0)
        self.assertEqual(d.max_value, 9.0)

    def test_merge_preserves_totals(self):
        d1 = S.TDigest()
        d1.add(5.0, 3)
        d2 = S.TDigest()
        d2.add(7.0, 2)
        m = d1.merge(d2)
        self.assertEqual(m.total, 5)
        self.assertEqual(m.n, 2)

    def test_zero_weight_ignored(self):
        d = S.TDigest()
        d.add(5.0, 0)
        self.assertEqual(d.n, 0)
        self.assertIsNone(d.quantile(0.5))


class TestSizingReport(unittest.TestCase):
    def test_card_shape(self):
        sizes = [100.0 * i for i in range(1, 101)]
        keys = [f"f{i:03d}" for i in range(1, 101)]
        r = S.sizing_report(sizes, keys, k=5)
        self.assertEqual(r["quantiles"]["verdict"], "EXACT")
        self.assertEqual(r["heavy_hitters"]["verdict"], "OK")
        self.assertEqual(len(r["heavy_hitters"]["hitters"]), 5)

    def test_key_size_mismatch_abstains(self):
        r = S.sizing_report([1.0, 2.0], ["only-one-key"])
        self.assertEqual(r["heavy_hitters"]["verdict"], "ABSTAIN")

    def test_tdigest_path_for_big_inputs(self):
        rng = random.Random(2)
        sizes = [rng.uniform(0, 5000) for _ in range(3000)]
        r = S.sizing_report(sizes)
        self.assertEqual(r["quantiles"]["verdict"], "TDIGEST")
        srt = sorted(sizes)
        exact95 = srt[int(0.95 * len(srt))]
        self.assertLess(abs(r["quantiles"]["values"]["0.95"]
                            - exact95) / 5000.0, 0.02)

    def test_no_keys_is_fine(self):
        r = S.sizing_report([1.0, 2.0, 3.0])
        self.assertNotIn("heavy_hitters", r)

    def test_deterministic(self):
        sizes = [float(i % 17) for i in range(300)]
        a = S.sizing_report(sizes, [f"k{i}" for i in range(300)])
        b = S.sizing_report(sizes, [f"k{i}" for i in range(300)])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
