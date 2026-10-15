"""scan tests — structures pinned against ground truth and their bounds."""
import random
import unittest

from assistant.capabilities.scan import (Automaton, BloomFilter, CountMinSketch,
                            HyperLogLog, Reservoir, EWMA, page_hinkley,
                            scan_stream, scan_text, render)


class TestAutomaton(unittest.TestCase):
    def test_matches_equal_regex_ground_truth(self):
        pats = ["quickshell", "pragma", "dbus timeout", "kwin"]
        auto = Automaton(pats)
        rng = random.Random(5)
        alphabet = "abcdefghijklmnopqrstuvwxyz _"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(120))
            text = text.replace("pragma", "pragma").replace("kwin", "kwin")
            if rng.random() < 0.5:
                pos = rng.randrange(0, 100)
                pat = rng.choice(pats)
                text = text[:pos] + pat + text[pos + len(pat):]
            expected = set()
            for p in pats:
                start = 0
                while True:
                    idx = text.casefold().find(p, start)
                    if idx < 0:
                        break
                    expected.add((idx + len(p), pats.index(p)))
                    start = idx + 1
            got = set(auto.scan(text))
            self.assertEqual(got, expected, text)

    def test_case_insensitive(self):
        auto = Automaton(["Error"])
        self.assertEqual(len(auto.scan("ERROR and error")), 2)

    def test_overlapping_dictionary_outputs(self):
        auto = Automaton(["he", "she", "her"])
        hits = auto.scan("usher")
        pids = {auto.pattern(p) for _, p in hits}
        self.assertEqual(pids, {"she", "he", "her"})


class TestBloom(unittest.TestCase):
    def test_no_false_negatives(self):
        bf = BloomFilter(capacity=10_000, error_rate=0.01)
        items = [f"line-{i}" for i in range(5000)]
        for it in items:
            bf.add(it)
        for it in items:
            self.assertIn(it, bf)

    def test_false_positives_within_bound(self):
        bf = BloomFilter(capacity=10_000, error_rate=0.01)
        for i in range(10_000):
            bf.add(f"line-{i}")
        fp = sum(1 for i in range(5000) if f"other-{i}" in bf)
        self.assertLess(fp / 5000, 0.05)  # generous 5x margin over the 1% target

    def test_serialisation_roundtrip(self):
        bf = BloomFilter(capacity=1000)
        bf.add("x")
        bf2 = BloomFilter.from_dict(bf.to_dict())
        self.assertIn("x", bf2)
        self.assertNotIn("y", bf2)


class TestCountMin(unittest.TestCase):
    def test_no_undercount_and_bounded_overcount(self):
        cms = CountMinSketch(epsilon=0.005, delta=0.001)
        true = {}
        rng = random.Random(9)
        for _ in range(50_000):
            tok = rng.choice(["error", "kwin", "dbus", "wayland", "vesktop"])
            cms.add(tok)
            true[tok] = true.get(tok, 0) + 1
        for tok, n in true.items():
            est = cms.estimate(tok)
            self.assertGreaterEqual(est, n)                      # never under
            self.assertLessEqual(est, n + 0.005 * cms.total + 1)  # eps bound

    def test_serialisation_roundtrip(self):
        cms = CountMinSketch()
        cms.add("q")
        cms2 = CountMinSketch.from_dict(cms.to_dict())
        self.assertEqual(cms2.estimate("q"), cms.estimate("q"))


class TestHLL(unittest.TestCase):
    def test_accuracy_within_expected_error(self):
        hll = HyperLogLog(precision=12)
        truth = {f"unit-{i}" for i in range(20_000)}
        for t in truth:
            hll.add(t)
        est = hll.count()
        # theoretical SE ~ 1.04/sqrt(4096) ≈ 1.6%; allow 5%
        self.assertLess(abs(est - len(truth)) / len(truth), 0.05)

    def test_duplicates_counted_once(self):
        hll = HyperLogLog()
        for _ in range(1000):
            hll.add("same")
        self.assertLess(hll.count(), 3)

    def test_merge(self):
        a, b = HyperLogLog(), HyperLogLog()
        for i in range(5000):
            a.add(f"u{i}")
        for i in range(5000, 9000):
            b.add(f"u{i}")
        a.merge(b)
        self.assertLess(abs(a.count() - 9000) / 9000, 0.05)


class TestReservoir(unittest.TestCase):
    def test_uniform_sample_of_unknown_stream(self):
        r = Reservoir(k=50, seed=3)
        stream = [f"item-{i}" for i in range(1000)]
        for s in stream:
            r.offer(s)
        self.assertEqual(len(r.sample), 50)
        self.assertEqual(r.n, 1000)
        # every sampled item must exist in the stream (uniformity is sampled,
        # so just verify membership + determinism)
        for s in r.sample:
            self.assertIn(s, stream)
        r2 = Reservoir(k=50, seed=3)
        for s in stream:
            r2.offer(s)
        self.assertEqual(r.sample, r2.sample)


class TestEWMAAndPH(unittest.TestCase):
    def test_ewma_tracks_level(self):
        e = EWMA(alpha=0.3)
        for _ in range(200):
            e.update(0.0)
        for _ in range(200):
            e.update(1.0)
        self.assertGreater(e.mean, 0.9)

    def test_ph_alarms_on_step_change(self):
        rng = random.Random(4)
        stream = [rng.gauss(0, 0.01) for _ in range(200)] + \
                 [rng.gauss(1.0, 0.01) for _ in range(200)]
        r = page_hinkley(stream, threshold=10.0, min_instances=20)
        self.assertIsNotNone(r["change_at"])
        self.assertGreaterEqual(r["change_at"], 150)
        self.assertLessEqual(r["change_at"], 260)

    def test_ph_quiet_on_stable_stream(self):
        rng = random.Random(4)
        stream = [rng.gauss(0.5, 0.01) for _ in range(1000)]
        r = page_hinkley(stream, threshold=50.0, min_instances=30)
        self.assertIsNone(r["change_at"])


class TestScanner(unittest.TestCase):
    def _auto(self):
        return Automaton(["quickshell has crashed", "kwin", "dbus timeout"])

    def test_counts_and_examples(self):
        lines = [
            "kernel: kwin_wayland Died",
            "quickshell has crashed twice today",
            "dbus timeout on com.canonical",
            "harmless log line",
            "kwin again",
        ]
        s = scan_stream(lines, self._auto())
        self.assertEqual(s["lines_scanned"], 5)
        self.assertEqual(s["lines_matching"], 4)
        pats = {h["pattern"]: h["hits"] for h in s["pattern_hits"]}
        self.assertEqual(pats.get("kwin"), 2)
        self.assertEqual(pats.get("quickshell has crashed"), 1)
        self.assertEqual(pats.get("dbus timeout"), 1)
        self.assertTrue(s["examples"])
        self.assertLessEqual(len(s["examples"]), 40)

    def test_rate_drift_detected(self):
        lines = ["boring"] * 1000
        lines[1500:2500] = ["quickshell has crashed"] * 1000
        s = scan_text("\n".join(lines), self._auto(), chunk_size=100)
        self.assertIsNotNone(s["rate"]["page_hinkley_change_at_chunk"])

    def test_render_plain_text(self):
        s = scan_text("kwin crashed\nerror dbus timeout", self._auto())
        text = render(s)
        self.assertIn("scanned 2 lines", text)
        self.assertIn("kwin", text)

    def test_bounded_memory_repr_serialisable(self):
        import json
        s = scan_text("kwin\nkwin\ndbus timeout", self._auto())
        json.dumps(s)  # must not raise
