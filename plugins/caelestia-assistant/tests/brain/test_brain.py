"""Tests for the shell-native brain engines: naive_bayes, minhash, forecast,
bandit, anomaly, ledger — plus the shell-native CLI surface.

(The personal-PKM engine tests live in assistant/brain/personal/tests/.)
"""
import io
import random
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import anomaly, bandit, forecast, ledger, minhash, naive_bayes
from assistant.capabilities.brain import cli
from assistant.capabilities.brain.nlp import tokens


class NaiveBayesTests(unittest.TestCase):
    def test_learns_tags_and_ranks_them(self):
        nb = naive_bayes.NaiveBayes()
        nb.train(tokens("kubernetes deploy pods cluster"), ["devops"])
        nb.train(tokens("sourdough flour yeast bake"), ["cooking"])
        nb.train(tokens("docker container image registry"), ["devops"])
        top = nb.rank(tokens("deploy the docker image to the cluster"))[0][0]
        self.assertEqual(top, "devops")

    def test_probabilities_sum_to_one(self):
        nb = naive_bayes.NaiveBayes()
        nb.train(["a"], ["x"])
        nb.train(["b"], ["y"])
        self.assertAlmostEqual(sum(p for _, p in nb.rank(["a"])), 1.0)

    def test_roundtrip(self):
        nb = naive_bayes.NaiveBayes()
        nb.train(["a", "b"], ["x"])
        nb2 = naive_bayes.NaiveBayes.from_dict(nb.to_dict())
        self.assertEqual(nb.rank(["a"]), nb2.rank(["a"]))


class MinHashTests(unittest.TestCase):
    def test_finds_near_duplicate_and_ignores_distinct(self):
        base = " ".join(f"word{i}" for i in range(80))
        docs = {
            "a.md": base,
            "b.md": base + " extra tail here",
            "c.md": " ".join(f"other{i}" for i in range(80)),
        }
        pairs = minhash.near_duplicates(docs, threshold=0.7)
        ids = {(a, b) for a, b, _ in pairs}
        self.assertIn(("a.md", "b.md"), ids)
        self.assertNotIn(("a.md", "c.md"), ids)

    def test_empty_text_is_skipped(self):
        self.assertEqual(minhash.near_duplicates({"e.md": "", "f.md": "!!!"}), [])


class ForecastTests(unittest.TestCase):
    def test_holt_extrapolates_linear_trend(self):
        out = forecast.holt([1, 2, 3, 4, 5, 6], alpha=0.9, beta=0.9, horizon=3)
        self.assertAlmostEqual(out[0], 7.0, delta=0.2)
        self.assertGreater(out[2], out[0])

    def test_kalman_converges_to_constant(self):
        k = forecast.Kalman1D(q=0.001, r=0.5, x0=0.0)
        for _ in range(60):
            x = k.update(10.0)
        self.assertAlmostEqual(x, 10.0, delta=0.5)


class BanditTests(unittest.TestCase):
    def test_learns_preferred_hour(self):
        b = bandit.HourBandit()
        for _ in range(80):
            b.reward(9, acted=True)
            b.reward(15, acted=False)
        rng = random.Random(0)
        picks = [b.choose(allowed=[9, 15], rng=rng) for _ in range(200)]
        self.assertGreater(picks.count(9), picks.count(15))

    def test_respects_allowed_window(self):
        self.assertIn(b_choose_in_window(), range(8, 12))


def b_choose_in_window():
    return bandit.HourBandit().choose(allowed=range(8, 12), rng=random.Random(1))


class AnomalyTests(unittest.TestCase):
    def test_zscore_flags_outlier(self):
        self.assertGreater(anomaly.zscore([10, 11, 9, 10, 10], 30), 5)

    def test_entropy_bits(self):
        self.assertAlmostEqual(anomaly.shannon_bits(["a", "b", "c", "d"]), 2.0)
        self.assertEqual(anomaly.shannon_bits(["a", "a", "a"]), 0.0)

    def test_deferral_flag(self):
        self.assertTrue(anomaly.deferral_flag(3))
        self.assertFalse(anomaly.deferral_flag(2))


class LedgerTests(unittest.TestCase):
    def test_propose_decide_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            led = ledger.Ledger(Path(d) / "l.json")
            pid = led.propose("tag", "n.md", {"add_tags": ["x"]}, "why", 0.9)
            self.assertEqual(len(led.pending()), 1)
            led.decide(pid, approve=True)
            self.assertEqual(led.pending(), [])
            self.assertEqual(len(led.labeled("tag")), 1)
            with self.assertRaises(ValueError):
                led.decide(pid, approve=False)
            # persisted
            self.assertEqual(ledger.Ledger(Path(d) / "l.json").items[0]["status"], "approved")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "state.json"
        self.led = self.root / "ledger.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out = io.StringIO()
        code = cli.main(["--state", str(self.state), "--ledger", str(self.led), *args], out=out)
        return code, out.getvalue()

    def test_focus_and_forecast(self):
        _, text = self.run_cli("focus", "a,b,a,c,b,d")
        self.assertIn("switch entropy", text)
        _, text = self.run_cli("forecast", "1,2,3,4,5,9")
        self.assertIn("anomaly", text)

    def test_ledger_list_empty(self):
        _, text = self.run_cli("ledger", "list")
        self.assertIn("no pending proposals", text)

    def test_personal_commands_are_not_on_this_surface(self):
        with self.assertRaises(SystemExit):
            self.run_cli("organize", "unused")
        with self.assertRaises(SystemExit):
            self.run_cli("plan", "unused")


if __name__ == "__main__":
    unittest.main()
