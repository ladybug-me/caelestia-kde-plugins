"""Tests for exponential-build 3 item B4: Laplace-mechanism
differential privacy for the lexicon-diff export path
(cortex/dp.py — Dwork, McSherry, Nissim & Smith 2006; composition per
Dwork & Roth 2014), plus the opt-in ``cortex lexicon export --dp`` CLI
wiring and the additive provenance marker in lexicon_diff's row
format.

The contract under test:

- laplace_noise is the EXACT inverse-CDF sampler: hand-derived values
  pinned for known uniforms (u=0.75/0.25/0.5, plus both tail edges),
  one uniform draw per call, deterministic given the rng;
- out-of-range epsilon (<= 0) and sensitivity (<= 0) are REJECTED with
  ValueError — never guessed;
- noised n is a non-negative integer (negative draws floored at zero
  AND counted in the report); with the floor off, the raw negative is
  kept and counted (mechanism-audit mode, whose artifact degrades
  honestly at import: lexicon_diff.parse skips the line with a
  warning — pinned);
- noised p is clipped to [0, 1] AND the clip count is reported (split
  low/high);
- EVERY noised row carries a "(dp: epsilon=X)" provenance marker, and
  the marker round-trips: lexicon_diff.parse accepts it (additive
  regex group) and records dp_epsilon row metadata;
- re-noising an already-noised diff is REJECTED (composing noise at
  the same stated epsilon would misreport the total);
- same (diff, epsilon, seed) -> byte-identical output; the default
  seed derives from the diff's content id (reproducible — tradeoff
  documented in the module docstring and pinned here);
- the DEFAULT export path (no --dp flag) is BYTE-IDENTICAL to
  lexicon_diff.render — pinned against the exact artifact;
- the epsilon documentation lives in the module docstring (the
  composition warning, the export-cap sensitivity rationale, and the
  honest noised-evidence-DP boundary) — pinned, the repo's
  policy-text convention.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from assistant.core import dp, lexicon_diff


class _FixedRng:
    """A rng stub whose single uniform is fixed — pins the inverse-CDF
    formula without any RNG state."""

    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


ROWS = [
    {"text": "frosted glass", "surface": "setBlurEnabled",
     "n": 2, "p": 0.88, "label": 1},
    {"text": "make the bar slim", "surface": "setBarScale",
     "n": 1, "p": 0.79, "label": 1},
    {"text": "enable the blur", "surface": "setBlurEnabled",
     "n": 1, "p": 0.40, "label": 0},
]


class LaplaceNoiseTests(unittest.TestCase):
    def test_inverse_cdf_hand_derived_values(self):
        # hand derivation (formula in dp.py's docstring):
        #   noise = -b * sign(u - 0.5) * ln(1 - 2|u - 0.5|),  b = s/eps
        # u = 0.75, b = 2: sign=+1, |u-0.5|=0.25, ln(0.5) = -ln2
        #   -> noise = -2 * 1 * (-ln2) = +2 ln2 = 1.3862943611198906...
        #      (check vs the CDF: F(+2ln2) = 1 - 0.5*exp(-ln2) = 0.75)
        self.assertEqual(dp.laplace_noise(10.0, 2.0, 1.0, _FixedRng(0.75)),
                         10.0 + 2.0 * math.log(2.0))
        # u = 0.25: sign=-1 -> noise = -2 ln2 (F(-2ln2) = 0.5*exp(-ln2) = 0.25)
        self.assertEqual(dp.laplace_noise(10.0, 2.0, 1.0, _FixedRng(0.25)),
                         10.0 - 2.0 * math.log(2.0))
        # u = 0.5 exactly: the median, noise exactly 0
        self.assertEqual(dp.laplace_noise(10.0, 2.0, 1.0, _FixedRng(0.5)),
                         10.0)
        # sensitivity scales b linearly: same u, s=4 -> twice the noise
        self.assertEqual(dp.laplace_noise(10.0, 4.0, 1.0, _FixedRng(0.75)),
                         10.0 + 4.0 * math.log(2.0))
        # epsilon halves b: same u, eps=2 -> half the noise
        self.assertEqual(dp.laplace_noise(10.0, 2.0, 2.0, _FixedRng(0.75)),
                         10.0 + 1.0 * math.log(2.0))

    def test_tail_edges_stay_finite(self):
        # u = 0.0 (possible: random() is [0,1)) and u = 1.0 would make
        # ln(0) undefined; the documented tail floor (1e-15) keeps the
        # draw finite at ~ +/- 34.5b — a tail treatment of the
        # mechanism's own sampling edge, pinned so it can't regress
        self.assertEqual(
            dp.laplace_noise(5.0, 1.0, 1.0, _FixedRng(0.0)),
            5.0 - math.log(1e15))
        self.assertEqual(
            dp.laplace_noise(5.0, 1.0, 1.0, _FixedRng(1.0)),
            5.0 + math.log(1e15))

    def test_bad_epsilon_and_sensitivity_are_rejected(self):
        rng = _FixedRng(0.75)
        for epsilon in (0.0, -0.5):
            with self.assertRaises(ValueError):
                dp.laplace_noise(1.0, 1.0, epsilon, rng)
        for sensitivity in (0.0, -1.0):
            with self.assertRaises(ValueError):
                dp.laplace_noise(1.0, sensitivity, 1.0, rng)

    def test_one_uniform_draw_per_call(self):
        # a rng that fails on the second draw proves the call consumes
        # exactly one uniform (the determinism contract's draw budget)
        class One:
            def __init__(self):
                self.left = 1

            def random(self):
                if self.left == 0:
                    raise AssertionError("second draw consumed")
                self.left -= 1
                return 0.75

        self.assertEqual(dp.laplace_noise(1.0, 1.0, 1.0, One()),
                         1.0 + math.log(2.0))


class NoiseDiffTests(unittest.TestCase):
    def test_pinned_output_and_counts(self):
        # fixture + epsilon=1.0 + seed=7, pinned byte-exactly
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=7, date="2026-09-26")
        self.assertEqual(report["text"],
                         "CAELESTIA LEXICON DIFF v1  (2026-09-26)\n"
                         "# (dp: epsilon=1.0) — counts and rates are "
                         "Laplace-noised (Dwork et al. 2006); presence is "
                         "exact; see cortex/dp.py\n"
                         "+frosted glass -> setBlurEnabled  (n=2, p=0.00) "
                         "(dp: epsilon=1.0)\n"
                         "+make the bar slim -> setBarScale  (n=1, p=0.00) "
                         "(dp: epsilon=1.0)\n"
                         "-enable the blur -> setBlurEnabled  (n=1, p=0.09) "
                         "(dp: epsilon=1.0)\n")
        # the report counts what happened: 0 floors, 2 low clips, 0 high
        self.assertEqual(report["n_floored_at_zero"], 0)
        self.assertEqual(report["p_clipped"], 2)
        self.assertEqual((report["p_clipped_low"], report["p_clipped_high"]),
                         (2, 0))
        self.assertEqual((report["rows_in"], report["rows_out"]), (3, 3))

    def test_noised_n_is_a_nonnegative_integer(self):
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=42)
        for row in report["rows"]:
            self.assertIsInstance(row["n"], int)
            self.assertGreaterEqual(row["n"], 0)
            self.assertIsInstance(row["p"], float)
            self.assertTrue(0.0 <= row["p"] <= 1.0)

    def test_negative_counts_floor_at_zero_and_are_counted(self):
        # seed 2 draws one negative noised count: floored at zero AND
        # counted in the report — REPORTED, never silent
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=2, date="2026-09-26")
        self.assertEqual(report["n_floored_at_zero"], 1)
        self.assertEqual([r["n"] for r in report["rows"]], [4, 0, 2])
        self.assertIn("(n=0, p=0.00) (dp: epsilon=1.0)", report["text"])

    def test_floor_off_keeps_the_raw_negative_and_degrades_honestly(self):
        # mechanism-audit mode: the raw negative survives (-1) and is
        # counted as n_negative_raw; the artifact it renders does NOT
        # re-parse (the row regex's n is \d+) — lexicon_diff.parse
        # skips it with a warning, the existing honest-degradation path
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=2,
                               min_count_floor=False)
        self.assertEqual(report["n_negative_raw"], 1)
        self.assertEqual(report["n_floored_at_zero"], 0)
        self.assertEqual([r["n"] for r in report["rows"]], [4, -1, 2])
        self.assertIn("(n=-1, p=0.00)", report["text"])
        parsed, warnings = lexicon_diff.parse(report["text"])
        self.assertEqual(len(parsed), 2)  # the negative row is skipped
        self.assertTrue(any("unparseable" in w for w in warnings))

    def test_p_out_of_range_clipped_and_reported(self):
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=11)
        # seed 11: one draw above 1 (clipped high) — the count says so
        self.assertEqual((report["p_clipped_low"], report["p_clipped_high"]),
                         (0, 1))
        self.assertEqual(report["p_clipped"], 1)
        self.assertIn("p=1.00", report["text"])

    def test_every_noised_row_carries_the_provenance_marker(self):
        for epsilon in (1.0, 0.5, 2.5):
            report = dp.noise_diff(ROWS, epsilon=epsilon, seed=7)
            marker = f"(dp: epsilon={epsilon})"
            row_lines = [ln for ln in report["text"].splitlines()
                         if ln[:1] in "+-"]
            self.assertEqual(len(row_lines), 3)
            for ln in row_lines:
                self.assertTrue(ln.endswith(marker), ln)
            for row in report["rows"]:
                self.assertEqual(row["dp_epsilon"], epsilon)

    def test_reproducibility_same_inputs_identical_bytes(self):
        a = dp.noise_diff(ROWS, epsilon=1.0, seed=7, date="2026-09-26")
        b = dp.noise_diff(ROWS, epsilon=1.0, seed=7, date="2026-09-26")
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(a, b)
        # a different seed is (pinned) different noise
        c = dp.noise_diff(ROWS, epsilon=1.0, seed=8, date="2026-09-26")
        self.assertNotEqual(a["text"], c["text"])

    def test_default_seed_derives_from_the_content_id(self):
        derived = dp.noise_diff(ROWS, epsilon=1.0)
        explicit = dp.noise_diff(
            ROWS, epsilon=1.0, seed=int(lexicon_diff.diff_id(ROWS), 16))
        self.assertEqual(derived["seed"], 12346500454137)  # pinned
        self.assertEqual(derived["text"], explicit["text"])
        # the tradeoff, pinned: SAME content -> SAME noise (reproducible
        # is NOT independent); CHANGED content -> different derived seed
        changed = [dict(ROWS[0], n=3)] + ROWS[1:]
        self.assertNotEqual(derived["seed"],
                            dp.noise_diff(changed, epsilon=1.0)["seed"])
        self.assertIn("NOT independent", derived["notes"][2])

    def test_text_input_matches_rows_input(self):
        # the rendered diff text noises identically to its rows —
        # render() sorts by text, so the rows must be sorted the same
        exact = lexicon_diff.render(ROWS, date="2026-09-26")
        sorted_rows = sorted(ROWS, key=lambda r: (r["text"], r["surface"]))
        from_text = dp.noise_diff(exact, epsilon=1.0, seed=7)
        from_rows = dp.noise_diff(sorted_rows, epsilon=1.0, seed=7,
                                  date="2026-09-26")
        self.assertEqual(from_text["text"], from_rows["text"])
        # the header date from the text input is preserved
        self.assertEqual(from_text["text"].splitlines()[0],
                         "CAELESTIA LEXICON DIFF v1  (2026-09-26)")

    def test_renoising_an_already_noised_diff_is_rejected(self):
        noised = dp.noise_diff(ROWS, epsilon=1.0, seed=7)
        with self.assertRaises(ValueError) as ctx:
            dp.noise_diff(noised["text"], epsilon=1.0, seed=8)
        self.assertIn("already noised", str(ctx.exception))
        with self.assertRaises(ValueError):
            dp.noise_diff(noised["rows"], epsilon=1.0, seed=8)

    def test_malformed_and_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            dp.noise_diff("garbage line with no arrow\n", 1.0)
        with self.assertRaises(ValueError):
            dp.noise_diff([], 1.0)
        with self.assertRaises(ValueError):
            dp.noise_diff(ROWS, epsilon=0.0)
        with self.assertRaises(ValueError):
            dp.noise_diff(ROWS, epsilon=1.0, seed="not-a-seed")

    def test_noised_diff_reparses_through_lexicon_diff(self):
        # the additive marker group: a noised artifact IMPORTS like any
        # other diff, carrying its dp provenance as row metadata
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=2, date="2026-09-26")
        parsed, warnings = lexicon_diff.parse(report["text"])
        self.assertEqual(warnings, [])
        self.assertEqual(len(parsed), 3)
        for row in parsed:
            self.assertEqual(row["dp_epsilon"], 1.0)
        # the EXACT format still parses with dp_epsilon None (additive)
        exact = lexicon_diff.parse(
            lexicon_diff.render(ROWS, date="2026-09-26"))[0]
        self.assertIsNone(exact[0]["dp_epsilon"])


class PresenceSubsamplingTests(unittest.TestCase):
    """The opt-in randomized row subsampling: honest AMPLIFICATION, not
    a row-presence epsilon-DP claim (a KEPT row is still exactly
    present) — the label is pinned, the drops are counted."""

    def test_rows_are_independently_kept_and_drops_counted(self):
        # seed 1, presence_keep=0.5: exactly one of the three rows
        # drops (deterministic given the seed), and it is counted
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=1,
                               presence_keep=0.5)
        self.assertEqual(report["rows_dropped"], 1)
        self.assertEqual([r["text"] for r in report["rows"]],
                         ["frosted glass", "make the bar slim"])
        self.assertEqual(report["rows_out"], 2)

    def test_keep_all_keeps_everything(self):
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=1, presence_keep=1.0)
        self.assertEqual(report["rows_dropped"], 0)
        self.assertEqual(report["rows_out"], 3)

    def test_bad_presence_keep_is_rejected(self):
        for bad in (0.0, -0.1, 1.5, 2.0):
            with self.assertRaises(ValueError):
                dp.noise_diff(ROWS, epsilon=1.0, seed=1,
                               presence_keep=bad)

    def test_the_guarantee_label_names_the_boundary(self):
        report = dp.noise_diff(ROWS, epsilon=1.0, seed=1, presence_keep=0.5)
        self.assertIn("noised-evidence DP", report["guarantee"])
        self.assertIn("not full row-level DP", report["guarantee"])


class EpsilonDocumentationTests(unittest.TestCase):
    """The module docstring IS the epsilon documentation (the repo's
    policy-text convention: pinned, so it cannot drift silently)."""

    DOC = dp.__doc__

    def test_composition_warning_is_documented(self):
        self.assertIn("composition", self.DOC.lower())
        self.assertIn("k * epsilon", self.DOC)
        self.assertIn("user-initiated and manual", self.DOC)

    def test_the_epsilon_rationale_cites_the_export_cap(self):
        self.assertIn("DEFAULT EPSILON = 1.0", self.DOC)
        self.assertIn("MAX_PAIRS = 200", self.DOC)  # lexicon_diff's cap
        self.assertIn("sensitivity", self.DOC.lower())

    def test_the_honest_boundary_is_stated(self):
        self.assertIn("NOISED-EVIDENCE DP, not full row-level DP", self.DOC)
        self.assertIn("presence is exact", self.DOC)
        self.assertIn("NOT independent across exports", self.DOC)

    def test_the_citations_are_present(self):
        self.assertIn("Dwork, McSherry, Nissim & Smith", self.DOC)
        self.assertIn("Calibrating Noise to Sensitivity", self.DOC)
        self.assertIn("Dwork & Roth 2014", self.DOC)
        self.assertIn("Algorithmic Foundations of Differential",
                      self.DOC)


class LexiconCliDpTests(unittest.TestCase):
    """`cortex lexicon export --dp [EPSILON] [--dp-seed N]` end to end,
    with an isolated brain state and a pinned date — and the DEFAULT
    export path byte-identical to the exact artifact."""

    EXAMPLES = [
        {"text": "frosted glass", "surface": "setBlurEnabled",
         "features": {}, "p": 0.86, "label": 1, "outcome": "applied"},
        {"text": "frosted glass", "surface": "setBlurEnabled",
         "features": {}, "p": 0.9, "label": 1, "outcome": "applied"},
        {"text": "make the bar slim", "surface": "setBarScale",
         "features": {}, "p": 0.79, "label": 1, "outcome": "applied"},
        {"text": "enable the blur", "surface": "setBlurEnabled",
         "features": {}, "p": 0.4, "label": 0, "outcome": "rejected"},
    ]

    class _FixedDatetime:
        @staticmethod
        def now():
            return datetime(2026, 9, 26)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state_path = Path(self._tmp.name) / "state.json"
        state = {"cortex_learn": {"examples": self.EXAMPLES}}
        self.state_path.write_text(json.dumps(state))

    def _cli(self, argv) -> str:
        from assistant.core.cli import cmd_cortex
        env = {"CAELESTIA_BRAIN_STATE": str(self.state_path)}
        old_stdin, sys.stdin = sys.stdin, io.StringIO("")
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                with unittest.mock.patch.dict(os.environ, env), \
                        unittest.mock.patch(
                            "assistant.core.cli.datetime",
                            self._FixedDatetime):
                    rc = cmd_cortex(["lexicon"] + argv)
        finally:
            sys.stdin = old_stdin
        self.rc = rc
        self.out = out.getvalue()
        self.err = err.getvalue()
        return rc

    def test_default_export_is_byte_identical_without_the_flag(self):
        # THE pin: no --dp flag -> the exact artifact, byte-identical
        # to lexicon_diff.render(export_rows(state), date) — the
        # default path is unchanged by the opt-in flag
        rows = lexicon_diff.export_rows(
            {"cortex_learn": {"examples": self.EXAMPLES}})
        expected = lexicon_diff.render(rows, date="2026-09-26")
        self.assertEqual(self._cli(["export"]), 0)
        self.assertEqual(self.out, expected)
        self.assertNotIn("(dp:", self.out)

    def test_dp_flag_produces_the_noised_artifact(self):
        rows = lexicon_diff.export_rows(
            {"cortex_learn": {"examples": self.EXAMPLES}})
        expected = dp.noise_diff(rows, epsilon=1.0, seed=None,
                                 date="2026-09-26")["text"]
        self.assertEqual(self._cli(["export", "--dp"]), 0)
        self.assertEqual(self.out, expected)  # byte-identical to the lib call
        self.assertTrue(self.out.endswith("(dp: epsilon=1.0)\n"))
        # the stderr report carries the honest counts + composition note
        self.assertIn("epsilon=1", self.err)
        self.assertIn("p clipped", self.err)
        self.assertIn("composition", self.err)
        self.assertIn("noised-evidence DP", self.err)

    def test_dp_epsilon_value_and_seed_are_respected(self):
        self.assertEqual(self._cli(["export", "--dp", "0.5",
                                     "--dp-seed", "7"]), 0)
        self.assertTrue(self.out.endswith("(dp: epsilon=0.5)\n"))
        rows = lexicon_diff.export_rows(
            {"cortex_learn": {"examples": self.EXAMPLES}})
        expected = dp.noise_diff(rows, epsilon=0.5, seed=7,
                                 date="2026-09-26")["text"]
        self.assertEqual(self.out, expected)

    def test_dp_output_is_reproducible_from_the_cli(self):
        self.assertEqual(self._cli(["export", "--dp", "--dp-seed", "42"]), 0)
        first = self.out
        self.assertEqual(self._cli(["export", "--dp", "--dp-seed", "42"]), 0)
        self.assertEqual(self.out, first)

    def test_bad_dp_epsilon_fails_honestly_not_with_a_traceback(self):
        # the repo's CLI convention for refused input: error line on
        # stderr, exit code 1 — the module's ValueError surfaces as a
        # message, never a traceback
        for bad in ("0", "-0.5"):
            self.assertEqual(self._cli(["export", "--dp", bad]), 1)
            self.assertIn("error:", self.err)
            self.assertIn("epsilon", self.err)


if __name__ == "__main__":
    unittest.main()
