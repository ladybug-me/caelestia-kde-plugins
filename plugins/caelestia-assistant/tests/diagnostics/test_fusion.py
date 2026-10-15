"""Tests for diagnostics/fusion.py — weighted-Bayes evidence fusion.

The contract under test:
- the math: each source's confidence is one weight-scaled likelihood-
  ratio update from the prior in log-odds space (hand-computed expected
  values below); with equal weights it reduces to the naive-Bayes
  likelihood-ratio product;
- abstention: a source with no evidence for a hypothesis contributes
  NOTHING (never a silent 0.5 vote);
- rejection, never clamping: confidences outside (0,1), negative
  weights, unknown sources, duplicate votes -> ValueError;
- the collectors transform each engine's OWN output without modifying
  it (rule confidence rides through; BM25 and scan hits go through the
  fixed saturating transform);
- the convenience diagnose() runs the engines read-only and merges
  hypotheses only when the caller supplies the mapping.
"""

from __future__ import annotations

import math
import unittest

from assistant.capabilities.diagnostics import engine as diagnostics_engine
from assistant.capabilities.diagnostics.fusion import (
    DEFAULT_WEIGHTS,
    diagnose,
    from_diagnose,
    from_retrieval,
    from_scan,
    fuse,
)


def _combined(prior, weighted_updates):
    """Reference implementation of the documented math."""
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    x = logit(prior)
    for _, c, w in weighted_updates:
        x += w * (logit(c) - logit(prior))
    return 1.0 / (1.0 + math.exp(-x))


class FuseMathTests(unittest.TestCase):
    def test_equal_weights_reduce_to_likelihood_ratio_product(self) -> None:
        # two diagnostics votes at the same confidence, equal weights:
        # the posterior must equal one full Bayes update (the naive-
        # Bayes reduction the docstring promises)
        result = fuse([
            {"source": "diagnostics", "hypothesis": "H", "confidence": 0.9},
            {"source": "retrieval", "hypothesis": "H", "confidence": 0.9},
        ], weights={"diagnostics": 0.5, "retrieval": 0.5, "scan": 0.5})
        top = result["diagnoses"][0]
        self.assertAlmostEqual(top["confidence"],
                               _combined(0.5, [("d", 0.9, 0.5),
                                               ("r", 0.9, 0.5)]),
                               places=6)
        # hand-check the reduction: 0.5*(L(0.9)-L(0.5))*2 = L(0.9)-L(0.5)
        self.assertAlmostEqual(top["confidence"], 0.9, places=6)

    def test_weighted_combination_matches_reference_math(self) -> None:
        result = fuse([
            {"source": "diagnostics", "hypothesis": "H",
             "confidence": 0.8},
            {"source": "retrieval", "hypothesis": "H",
             "confidence": 0.7},
            {"source": "scan", "hypothesis": "H", "confidence": 0.3},
        ])
        expected = _combined(0.5, [
            ("d", 0.8, DEFAULT_WEIGHTS["diagnostics"]),
            ("r", 0.7, DEFAULT_WEIGHTS["retrieval"]),
            ("s", 0.3, DEFAULT_WEIGHTS["scan"])])
        top = result["diagnoses"][0]
        self.assertAlmostEqual(top["confidence"], expected, places=4)
        # odds multiplier = exp(total log-odds shift) vs the prior
        expected_mult = math.exp(sum(
            DEFAULT_WEIGHTS[s] * math.log(c / (1 - c))
            for s, c in [("diagnostics", 0.8), ("retrieval", 0.7),
                         ("scan", 0.3)]))
        self.assertAlmostEqual(top["odds_multiplier"], expected_mult,
                               places=2)

    def test_abstention_contributes_nothing(self) -> None:
        solo = fuse([{"source": "diagnostics", "hypothesis": "H",
                      "confidence": 0.8}])
        with_peer = fuse([
            {"source": "diagnostics", "hypothesis": "H",
             "confidence": 0.8},
            {"source": "scan", "hypothesis": "OTHER", "confidence": 0.6},
        ])
        self.assertEqual(
            solo["diagnoses"][0]["confidence"],
            with_peer["diagnoses"][0]["confidence"],
            "an unrelated source's evidence must not move H")

    def test_ranking_deterministic_with_tiebreak(self) -> None:
        result = fuse([
            {"source": "diagnostics", "hypothesis": "b", "confidence": 0.7},
            {"source": "diagnostics", "hypothesis": "a", "confidence": 0.7},
            {"source": "diagnostics", "hypothesis": "z", "confidence": 0.9},
        ])
        self.assertEqual([d["hypothesis"] for d in result["diagnoses"]],
                         ["z", "a", "b"])

    def test_out_of_range_rejected_never_clamped(self) -> None:
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                fuse([{"source": "diagnostics", "hypothesis": "H",
                       "confidence": bad}])
        with self.assertRaises(ValueError):
            fuse([{"source": "diagnostics", "hypothesis": "H",
                   "confidence": "high"}])
        with self.assertRaises(ValueError):
            fuse([{"source": "oracle", "hypothesis": "H",
                   "confidence": 0.5}])
        with self.assertRaises(ValueError):
            fuse([{"source": "diagnostics", "hypothesis": "H",
                   "confidence": 0.5}], weights={"oracle": 0.5})
        with self.assertRaises(ValueError):
            fuse([{"source": "diagnostics", "hypothesis": "H",
                   "confidence": 0.5}], weights={"diagnostics": -1.0})
        with self.assertRaises(ValueError):
            fuse([{"source": "diagnostics", "hypothesis": "H",
                   "confidence": 0.5}], prior=1.0)
        with self.assertRaises(ValueError):
            fuse([{"source": "diagnostics", "hypothesis": "H",
                   "confidence": 0.5},
                  {"source": "diagnostics", "hypothesis": "H",
                   "confidence": 0.6}])


class CollectorTests(unittest.TestCase):
    def test_from_diagnose_uses_match_score_not_tier_string(self) -> None:
        diag = diagnostics_engine.diagnose(
            "vesktop freezes when screen sharing turns on")
        snapshot = json_snapshot(diag)
        evidence = from_diagnose(diag)
        self.assertTrue(evidence)
        top_rule_id = diag["top"]["rule"]["id"]
        top_score = float(diag["top"]["score"])
        first = next(e for e in evidence
                     if e["hypothesis"] == top_rule_id)
        self.assertEqual(first["source"], "diagnostics")
        # the engine's own match score through the shared transform
        self.assertAlmostEqual(first["confidence"],
                               top_score / (top_score + 1.0), places=6)
        # the qualitative tier rides along as metadata, NOT as a number
        self.assertEqual(first["tier"],
                         diag["top"]["rule"]["confidence"])
        self.assertIsInstance(first["tier"], str)
        # the engine's own output is untouched by the collector
        self.assertEqual(json_snapshot(diag), snapshot)

    def test_from_retrieval_saturating_transform(self) -> None:
        hits = [{"doc_id": "ISS-120", "score": 3.0},
                {"doc_id": "zero", "score": 0.0}]
        evidence = from_retrieval(hits, scale=1.0)
        self.assertEqual(len(evidence), 1)  # zero-scored doc abstains
        self.assertEqual(evidence[0]["hypothesis"], "doc:ISS-120")
        self.assertAlmostEqual(evidence[0]["confidence"], 0.75, places=6)

    def test_from_scan_hit_density(self) -> None:
        scan = {"lines_scanned": 9,
                "pattern_hits": [{"pattern": "segfault", "hits": 1},
                                 {"pattern": "none", "hits": 0}]}
        evidence = from_scan(scan, scale=1.0)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["hypothesis"], "pattern:segfault")
        # density 1/9 -> c = (1/9)/(1/9 + 1) = 0.1
        self.assertAlmostEqual(evidence[0]["confidence"],
                               (1 / 9) / (1 / 9 + 1.0), places=6)


def json_snapshot(obj):
    import json
    return json.dumps(obj, sort_keys=True, default=str)


class DiagnoseConvenienceTests(unittest.TestCase):
    def test_runs_engines_readonly_and_fuses(self) -> None:
        from unittest import mock
        fake_hits = [{"doc_id": "ISS-999", "title": "t", "score": 1.0}]
        with mock.patch("assistant.capabilities.retrieval.search.search",
                        return_value=fake_hits):
            result = diagnose("vesktop freezes when screen sharing",
                              k=2,
                              hypothesis_map={"doc:ISS-999":
                                              "vesktop-screencast",
                                              "CL-runtime-shell-vesktop-001":
                                              "vesktop-screencast"})
        self.assertEqual(result["inputs"]["diagnostics_verdict"], "MATCH")
        self.assertEqual(result["inputs"]["retrieval_hits"], 1)
        self.assertFalse(result["inputs"]["scan_ran"])
        merged = [d for d in result["diagnoses"]
                  if d["hypothesis"] == "vesktop-screencast"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(merged[0]["sources"]),
                         ["diagnostics", "retrieval"])
        # the merged hypothesis carries ONE combined confidence
        self.assertTrue(0.0 < merged[0]["confidence"] < 1.0)

    def test_scan_joins_when_caller_supplies_inputs(self) -> None:
        from unittest import mock
        with mock.patch("assistant.capabilities.retrieval.search.search",
                        return_value=[]):
            result = diagnose(
                "vesktop freezes", k=1,
                stream_text="segfault\nsegfault\nsegfault\n",
                patterns=["segfault"],
                hypothesis_map={"pattern:segfault": "screencast-crash",
                                "CL-runtime-shell-vesktop-001":
                                    "screencast-crash"})
        self.assertTrue(result["inputs"]["scan_ran"])
        merged = [d for d in result["diagnoses"]
                  if d["hypothesis"] == "screencast-crash"]
        self.assertEqual(len(merged), 1)
        self.assertIn("scan", merged[0]["sources"])

    def test_no_mapping_means_no_fuzzy_join(self) -> None:
        from unittest import mock
        fake_hits = [{"doc_id": "ISS-999", "title": "t", "score": 1.0}]
        with mock.patch("assistant.capabilities.retrieval.search.search",
                        return_value=fake_hits):
            result = diagnose("vesktop freezes when screen sharing", k=2)
        hyps = [d["hypothesis"] for d in result["diagnoses"]]
        self.assertNotIn("vesktop-screencast", hyps)
        self.assertIn("doc:ISS-999", hyps)


if __name__ == "__main__":
    unittest.main()
