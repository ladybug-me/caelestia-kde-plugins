"""B1 — sketch-divergence novelty detector (scan layer).

Contract under test (sketch.py NgramDivergence + scanner opt-in):

- SINGLE-PASS, BOUNDED: every line is consumed exactly once; state is two
  block-sized dicts + one CMS + one HLL, independent of stream length
  (the scanner's own charter: "a 2 GB log and a 2 KB log cost the same
  RAM" — the same bound now covers the novelty tracker).
- DETECTS THE UNMATCHED: the internal baseline covers EVERY line's
  n-grams (the scanner's own CMS only sees matching lines; a novelty
  detector's job is precisely the lines no signature matched).
- KL SEMANTICS: identical block vs reference distributions give KL = 0
  (up to self-comparison); a distribution shift yields a strictly larger
  KL; smoothing keeps it finite for unseen n-grams.
- The KL series feeds Page-Hinkley: a sustained shift alarms, a steady
  stream stays quiet.
- The opt-in is really opt-in: the default summary is key-for-key
  unchanged.
"""

from __future__ import annotations

import math
import random
import unittest

from assistant.capabilities.scan.ac import Automaton
from assistant.capabilities.scan.scanner import render, scan_stream
from assistant.capabilities.scan.sketch import NgramDivergence


def _stable_lines(n, vocab, seed=7):
    rng = random.Random(seed)
    return [" ".join(rng.choice(vocab) for _ in range(8)) for _ in range(n)]


class NgramDivergenceTests(unittest.TestCase):
    def test_identical_blocks_give_zero_kl(self) -> None:
        nd = NgramDivergence(block_lines=100)
        for line in _stable_lines(200, ["alpha", "beta", "gamma", "delta"]):
            nd.offer(line)
        # Block 2 was drawn from the same distribution as block 1 (same
        # seed discipline): KL is small (smoothing noise), never large.
        self.assertLess(nd.kl_series[1], 0.5)

    def test_distribution_shift_raises_kl(self) -> None:
        nd = NgramDivergence(block_lines=100)
        for line in _stable_lines(100, ["alpha", "beta"]):
            nd.offer(line)
        for line in _stable_lines(100, ["zeta", "eta", "theta"], seed=99):
            nd.offer(line)
        # Disjoint vocabulary: KL must be large (every gram unseen in the
        # reference block).
        self.assertGreater(nd.kl_series[1], 1.0)

    def test_kl_is_zero_for_first_block(self) -> None:
        nd = NgramDivergence(block_lines=10)
        for line in _stable_lines(10, ["a", "b"]):
            nd.offer(line)
        # No reference yet: the honest answer is 0, not a guess.
        self.assertEqual(nd.kl_series[0], 0.0)

    def test_kl_is_finite_and_nonnegative(self) -> None:
        rng = random.Random(3)
        nd = NgramDivergence(block_lines=50)
        for _ in range(150):
            nd.offer(" ".join(rng.choice("xyzw") for _ in range(6)))
        for value in nd.kl_series:
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)

    def test_offer_returns_report_only_on_block_boundaries(self) -> None:
        nd = NgramDivergence(block_lines=3)
        self.assertIsNone(nd.offer("a b"))
        self.assertIsNone(nd.offer("a b"))
        report = nd.offer("a b")
        self.assertIsNotNone(report)
        self.assertEqual(report["block_index"], 1)
        self.assertIsNone(nd.offer("a b"))

    def test_bigram_mode(self) -> None:
        nd = NgramDivergence(block_lines=5, n=2)
        for _ in range(5):
            nd.offer("a b c")
        # bigrams: "a b" x5, "b c" x5
        self.assertEqual(nd.kl_series, [0.0])  # no reference yet
        self.assertEqual(nd._block, {})  # closed and cleared

    def test_alarm_fires_on_sustained_shift(self) -> None:
        nd = NgramDivergence(block_lines=40)
        # Ten quiet blocks, then a hard vocabulary change.
        for line in _stable_lines(400, ["alpha", "beta", "gamma"]):
            nd.offer(line)
        for line in _stable_lines(400, ["nova", "kappa", "lambda"], seed=11):
            nd.offer(line)
        alarm = nd.alarm(threshold=0.5, min_blocks=3)
        self.assertIsNotNone(alarm["novelty_change_at_block"])
        self.assertGreater(alarm["novelty_magnitude"], 0.5)

    def test_alarm_quiet_on_steady_stream(self) -> None:
        nd = NgramDivergence(block_lines=40)
        for line in _stable_lines(800, ["alpha", "beta", "gamma"]):
            nd.offer(line)
        alarm = nd.alarm(threshold=0.5, min_blocks=3)
        self.assertIsNone(alarm["novelty_change_at_block"])

    def test_kl_vs_cms_reports_movers(self) -> None:
        nd = NgramDivergence(block_lines=50)
        for line in _stable_lines(150, ["alpha", "beta"]):
            nd.offer(line)
        # Current (empty) block after close: fill a partial block with novelties
        for _ in range(10):
            nd.offer("brandnewtoken other stuff")
        result = nd.kl_vs_cms()
        self.assertIn("kl_vs_cms", result)
        self.assertIn("movers", result)
        self.assertIn("distinct_ngrams_estimate", result)
        self.assertGreater(result["distinct_ngrams_estimate"], 0)
        # "brandnewtoken" was never in the long-run baseline: high lift.
        tops = [m["ngram"] for m in result["movers"]]
        self.assertIn("brandnewtoken", tops[:3])

    def test_serialisation_roundtrip(self) -> None:
        nd = NgramDivergence(block_lines=5)
        for line in _stable_lines(15, ["a", "b", "c"]):
            nd.offer(line)
        data = nd.to_dict()
        restored = NgramDivergence.from_dict(data)
        self.assertEqual(restored.kl_series, nd.kl_series)
        self.assertEqual(restored.block_lines, nd.block_lines)
        self.assertEqual(restored.to_dict()["cms"], data["cms"])

    def test_bounded_state_independent_of_stream_length(self) -> None:
        # The structural bound: block dict never exceeds block_lines lines'
        # worth of distinct n-grams regardless of how many lines flow.
        nd = NgramDivergence(block_lines=10)
        rng = random.Random(5)
        peak = 0
        for i in range(5000):
            nd.offer("t%d x%d" % (rng.randint(0, 30), rng.randint(0, 30)))
            peak = max(peak, len(nd._block) + len(nd._reference))
        # Two blocks of 10 lines x 2 tokens -> at most ~40 distinct grams
        # alive at once (20 per block), no matter that 5000 lines flowed.
        self.assertLessEqual(peak, 44)


class ScannerNoveltyWiringTests(unittest.TestCase):
    def _lines(self):
        rng = random.Random(1)
        return [" ".join(rng.choice(["quickshell", "crashed", "signal",
                                     "handoff", "pragma"]) for _ in range(6))
                for _ in range(1200)]

    def test_default_summary_has_no_novelty_key(self) -> None:
        summary = scan_stream(self._lines(), Automaton(["crashed"]))
        self.assertNotIn("novelty", summary)

    def test_opt_in_adds_novelty_section(self) -> None:
        lines = self._lines()
        # A vocabulary shift at line 600.
        rng = random.Random(2)
        shifted = [" ".join(rng.choice(["completely", "new", "failure",
                                        "mode", "unmatched"]) for _ in range(6))
                   for _ in range(600)]
        stream = list(lines[:600]) + list(shifted)
        summary = scan_stream(stream, Automaton(["crashed"]),
                              novelty=True, novelty_block_lines=200)
        self.assertIn("novelty", summary)
        self.assertEqual(len(summary["novelty"]["blocks"]), 6)
        self.assertIn("last_vs_cms", summary["novelty"])
        self.assertIn("alarm", summary["novelty"])
        # The post-shift blocks see the vocabulary change: their KL vs the
        # previous block is strictly higher than the steady-state ones.
        kls = [b["kl_vs_previous_block"] for b in summary["novelty"]["blocks"]]
        self.assertLess(kls[0], 0.5)     # first block: no reference
        self.assertGreater(kls[3], kls[1])  # block 3 = first shifted block

    def test_novelty_is_json_serialisable(self) -> None:
        import json
        summary = scan_stream(self._lines(), Automaton(["crashed"]),
                              novelty=True, novelty_block_lines=300)
        self.assertIsInstance(json.dumps(summary["novelty"]), str)

    def test_single_pass_token_discipline(self) -> None:
        # The divergence consumes each line exactly once — observable via
        # block cadence: 600 lines at block_lines=200 -> exactly 3 blocks.
        summary = scan_stream(self._lines()[:600], Automaton(["crashed"]),
                              novelty=True, novelty_block_lines=200)
        self.assertEqual(len(summary["novelty"]["blocks"]), 3)
        self.assertEqual(summary["novelty"]["blocks"][0]["block_tokens"] > 0, True)

    def test_render_still_works_with_novelty(self) -> None:
        summary = scan_stream(self._lines(), Automaton(["crashed"]),
                              novelty=True, novelty_block_lines=300)
        text = render(summary)
        self.assertIn("scanned", text)


if __name__ == "__main__":
    unittest.main()
