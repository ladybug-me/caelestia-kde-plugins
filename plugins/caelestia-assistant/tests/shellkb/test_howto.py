"""shellkb.howto tests — BM25 retrieval, time windows, license honesty."""
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.shellkb import howto  # noqa: E402

FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0)


class TestCorpus(unittest.TestCase):
    def test_corpus_loads_and_declares_license(self):
        corpus = howto._load_corpus()
        self.assertGreaterEqual(len(corpus["entries"]), 40)
        self.assertIn("CC0", corpus["license"])
        self.assertIn("original", corpus["license"])

    def test_every_entry_is_wellformed(self):
        for e in howto._load_corpus()["entries"]:
            self.assertTrue(e["id"].startswith("h"))
            self.assertTrue(e["task"])
            self.assertTrue(e["keywords"])
            self.assertTrue(e["steps"])
            for step in e["steps"]:
                self.assertNotIn("sudo rm", step)
            self.assertIsInstance(e.get("temporal", False), bool)

    def test_steps_reference_known_bins(self):
        """Every command step starts with a binary this repo or the
        shell actually ships — no invented commands."""
        known = {"caelestia", "caelestia-assist", "caelestia-shell-ipc",
                 "caelestia-color", "caelestia-record",
                 "caelestia-screenshot", "caelestia-update",
                 "caelestia-check-updates", "python3", "cat"}
        for e in howto._load_corpus()["entries"]:
            for step in e["steps"]:
                self.assertIn(step.split()[0], known,
                              f"{e['id']}: {step}")


class TestRetrieval(unittest.TestCase):
    def test_region_screenshot_ranks_first(self):
        data = howto.answer("how do I take a screenshot of just a region",
                            now=FIXED_NOW)
        self.assertEqual(data["results"][0]["id"], "h20")

    def test_stemming_makes_logging_meet_log(self):
        data = howto.answer("what was the shell logging last night",
                            now=FIXED_NOW)
        self.assertTrue(data["results"])
        self.assertEqual(data["results"][0]["id"], "h06")
        self.assertIsNotNone(data["window"])

    def test_window_is_parsed_and_reported(self):
        # '2 hours ago' — the reused nlhistory grammar takes digits (or
        # a/an) for ago-counts; number words are its steps grammar
        data = howto.answer("read the shell log from 2 hours ago",
                            now=FIXED_NOW)
        self.assertIsNotNone(data["window"])
        lo, hi = data["window"]["from"], data["window"]["to"]
        self.assertIn("2026-01-01T10:00", lo)
        self.assertIn("2026-01-01T12:00", hi)

    def test_garbage_query_is_honest(self):
        data = howto.answer("how do I fix the squid population "
                            "of the pacific ocean", now=FIXED_NOW)
        self.assertEqual(data["results"], [])
        self.assertIsNotNone(data["empty_note"])

    def test_every_step_is_labeled_not_executed(self):
        data = howto.answer("change the wallpaper", now=FIXED_NOW)
        for r in data["results"]:
            for step in r["steps"]:
                self.assertTrue(step.startswith("SUGGESTED_NOT_EXECUTED:"))

    def test_temporal_boost_applies_only_to_temporal_entries(self):
        plain = howto.answer("read the shell log two hours ago",
                             now=FIXED_NOW)
        for r in plain["results"]:
            if r["id"] == "h06":
                self.assertTrue(r["temporal"])

    def test_determinism(self):
        a = howto.answer("switch to dark mode", now=FIXED_NOW)
        b = howto.answer("switch to dark mode", now=FIXED_NOW)
        self.assertEqual(a, b)


class TestStemmer(unittest.TestCase):
    def test_doubling_rules(self):
        self.assertEqual(howto._stem("logging"), "log")
        self.assertEqual(howto._stem("recordings"), "record")
        self.assertEqual(howto._stem("missed"), "miss")   # ss never split
        self.assertEqual(howto._stem("telling"), "tell")  # ll never split
        self.assertEqual(howto._stem("schemes"), "scheme")


if __name__ == "__main__":
    unittest.main()
