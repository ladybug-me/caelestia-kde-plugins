"""Tests for exponential-build-3 F3 — opt-in, coverage/accuracy-only
engine telemetry with the Laplace-DP export (cortex/
engine_telemetry.py, riding exponential-build-3 B4's cortex/dp.py).

Under test (values hand-derived from a fixed 6-example log):

- metrics(): per-surface n / decided / accepted / rejected /
  coverage / accuracy, with abstains counted in n but NOT in decided
  (abstaining is the honest no-answer, not a wrong one) and accuracy
  None (rendered '-') for surfaces with zero decided turns — never
  0.0, which would claim every decision was wrong;
- the exact artifact is byte-distinct from the noised artifact (the
  headers say which, so a noised export can never masquerade as
  exact) and contains NO text/phrases/features/timestamps;
- the DP pass: n and decided noised at sensitivity 1 (floored at 0,
  count reported), rates noised and clipped to [0,1] (counted), every
  row carrying an explicit (dp: epsilon=X) marker; a surface noised
  down to zero decided turns carries NO accuracy claim; epsilon <= 0
  refuses; the pass is reproducible for a fixed seed and the default
  seed is content-derived (same rows -> same noise, stated NOT
  independent across exports);
- the CLI: local report renders, --export prints the exact artifact,
  --export --dp prints the noised one with the composition warning;
- the capability gate: a disabled engine_telemetry manifest refuses.
"""
import io
import sys
import unittest
from unittest import mock

from assistant.core import engine_telemetry
from assistant.core.cli import cmd_cortex
from assistant.core.engine_telemetry import (export_text, metrics,
                                               noise_metrics)
from assistant.core.learn import CortexLearner

FEATURES = {"lex": 0.5, "sem": 0.2, "fuzz": 0.1, "noun": 0.3,
            "cue": 0.1}


def _learner() -> CortexLearner:
    learn = CortexLearner()
    for text, surface, outcome in [
        ("make bar thin", "setBarScale", "applied"),
        ("make bar thin again", "setBarScale", "applied"),
        ("thin bar", "setBarScale", "abstain"),
        ("color accent", "setBarPersistent", "rejected"),
        ("color accent two", "setBarPersistent", "rejected"),
        ("other", "setBarPosition", "applied"),
    ]:
        learn.observe(text, surface, FEATURES, 0.6, outcome)
    return learn


class MetricsTests(unittest.TestCase):
    def test_hand_derived_rows(self):
        # setBarScale: 3 turns, 1 abstain -> decided 2, accepted 2,
        # coverage 2/3, accuracy 1.0. setBarPersistent: 2/2 decided,
        # 0 accepted, accuracy 0.0 (a real all-rejected claim).
        # setBarPosition: 1/1, accuracy 1.0. Sorted by (-n, surface).
        rows = metrics(_learner())
        self.assertEqual(rows, [
            {"surface": "setBarScale", "n": 3, "decided": 2,
             "accepted": 2, "rejected": 0, "coverage": 0.6667,
             "accuracy": 1.0},
            {"surface": "setBarPersistent", "n": 2, "decided": 2,
             "accepted": 0, "rejected": 2, "coverage": 1.0,
             "accuracy": 0.0},
            {"surface": "setBarPosition", "n": 1, "decided": 1,
             "accepted": 1, "rejected": 0, "coverage": 1.0,
             "accuracy": 1.0},
        ])

    def test_all_abstain_surface_carries_no_accuracy_claim(self):
        learn = CortexLearner()
        for i in range(3):
            learn.observe(f"ambiguous {i}", "someTool", FEATURES, 0.4,
                          "abstain")
        rows = metrics(learn)
        self.assertEqual(rows[0]["decided"], 0)
        # accuracy: NO claim (None -> rendered '-'), never 0.0
        self.assertIsNone(rows[0]["accuracy"])
        # coverage: a real claim — zero of three turns decided
        self.assertEqual(rows[0]["coverage"], 0.0)

    def test_empty_log_is_empty(self):
        self.assertEqual(metrics(CortexLearner()), [])


class ExportTests(unittest.TestCase):
    def test_exact_artifact_shape_and_privacy_header(self):
        rows = metrics(_learner())
        text = export_text(rows, date="2026-09-27")
        self.assertIn("EXACT (local read-back; not for sharing)", text)
        self.assertIn("# date: 2026-09-27", text)
        self.assertIn("setBarScale | 3 | 2 | 2 | 0 | 0.6667 | 1.0",
                      text)
        self.assertIn("no text, no phrases, no features", text)
        self.assertIn("(thin)", text)  # all rows here are n < 4

    def test_exact_and_noised_are_byte_distinct(self):
        rows = metrics(_learner())
        noised = noise_metrics(rows, epsilon=1.0, seed=7)
        self.assertNotEqual(export_text(rows, date="d"),
                            export_text(noised["rows"], date="d",
                                        dp=noised))


class NoiseTests(unittest.TestCase):
    def test_seeded_noise_is_reproducible_and_marked(self):
        rows = metrics(_learner())
        first = noise_metrics(rows, epsilon=1.0, seed=7)
        second = noise_metrics(rows, epsilon=1.0, seed=7)
        self.assertEqual(first["rows"], second["rows"])
        for row in first["rows"]:
            self.assertEqual(row["dp_marker"], "(dp: epsilon=1)")
        # a different seed is different noise
        third = noise_metrics(rows, epsilon=1.0, seed=8)
        self.assertNotEqual(first["rows"], third["rows"])

    def test_default_seed_is_content_derived(self):
        rows = metrics(_learner())
        a = noise_metrics(rows, epsilon=1.0)
        b = noise_metrics(rows, epsilon=1.0)
        self.assertEqual(a["seed"], b["seed"])
        self.assertEqual(a["rows"], b["rows"])

    def test_invariants_hold_after_noising(self):
        rows = metrics(_learner())
        noised = noise_metrics(rows, epsilon=1.0, seed=7)
        for row in noised["rows"]:
            self.assertGreaterEqual(row["n"], 0)
            self.assertGreaterEqual(row["decided"], 0)
            self.assertLessEqual(row["decided"], row["n"])
            self.assertEqual(row["accepted"] + row["rejected"],
                             row["decided"])
            if row["accuracy"] is not None:
                self.assertTrue(0.0 <= row["accuracy"] <= 1.0)
            if row["coverage"] is not None:
                self.assertTrue(0.0 <= row["coverage"] <= 1.0)

    def test_zero_decided_noised_row_makes_no_accuracy_claim(self):
        rows = metrics(_learner())
        # LOW epsilon means STRONG noise (b = 1/epsilon): n=2 rows get
        # suppressed below the 0.5 rounding floor in most seeds; when a
        # row's noised decided count lands at zero, its accuracy must
        # be None (no claim), never 0.0
        found = 0
        for seed in range(60):
            noised = noise_metrics(rows, epsilon=0.5, seed=seed)
            for row in noised["rows"]:
                if row["surface"] == "setBarPersistent" \
                        and row["decided"] == 0:
                    self.assertIsNone(row["accuracy"])
                    found += 1
        # suppression is probabilistic; over 60 seeds at epsilon 0.5
        # it happens in the large majority (measured 32+/60) — require
        # a healthy margin so the test cannot flake near zero
        self.assertGreater(found, 10)

    def test_epsilon_and_sensitivity_refusals(self):
        rows = metrics(_learner())
        with self.assertRaises(ValueError) as caught:
            noise_metrics(rows, epsilon=0.0)
        self.assertIn("epsilon must be positive", str(caught.exception))
        with self.assertRaises(ValueError):
            noise_metrics(rows, epsilon=-1.0)


class CliTests(unittest.TestCase):
    def _run(self, argv, learner_state=None):
        out, err = io.StringIO(), io.StringIO()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            with mock.patch(
                    "assistant.core.cli._load_learner",
                    return_value=_learner()):
                rc = cmd_cortex(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return rc, out.getvalue(), err.getvalue()

    def test_local_report_renders(self):
        rc, out, _ = self._run(["telemetry"])
        self.assertEqual(rc, 0)
        self.assertIn("coverage/accuracy only", out)
        self.assertIn("setBarScale", out)
        self.assertIn("coverage=0.67", out)

    def test_export_exact_and_dp(self):
        rc, out, err = self._run(["telemetry", "--export"])
        self.assertEqual(rc, 0)
        self.assertIn("EXACT (local read-back; not for sharing)", out)
        self.assertIn("use --dp for the shareable", err)
        rc, out, err = self._run(["telemetry", "--export", "--dp",
                                  "1.0", "--dp-seed", "7"])
        self.assertEqual(rc, 0)
        self.assertIn("DP-NOISED (epsilon=1, seed=7", out)
        self.assertIn("(dp: epsilon=1)", out)
        self.assertIn("composition", err)
        # reproducible: the same seed exports the same bytes
        rc2, out2, _ = self._run(["telemetry", "--export", "--dp",
                                  "1.0", "--dp-seed", "7"])
        self.assertEqual(out, out2)

    def test_capability_gate_refuses_when_disabled(self):
        with mock.patch("assistant.core.features.enabled",
                        return_value=False):
            rc, _out, err = self._run(["telemetry"])
        self.assertEqual(rc, 1)
        self.assertIn("engine_telemetry is disabled", err)


if __name__ == "__main__":
    unittest.main()
