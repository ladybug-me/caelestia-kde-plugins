"""A2-7 honesty + CLI fixes (exponential-build-5, D8, D10).

- D8: cold-start confidence is the flat Beta prior — the card says
  'uncalibrated (no history)', never a fake 0.50.
- D10: hub verbs accept flags before and after the text.

(The R4 lazy-import check — `--help` imports no engine — runs in the
bash harness, tests/test_assistant.sh: it needs a fresh interpreter.)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest


def hub(args):
    """Run the hub in-process with a scratch HOME; (code, stdout, stderr)."""
    from assistant import hub as hub_mod
    home = tempfile.mkdtemp(prefix="eb5-hub-")
    old_home, os.environ["HOME"] = os.environ.get("HOME"), home
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = hub_mod.main(list(args))
            except SystemExit as exc:  # argparse --help path
                code = int(exc.code or 0)
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
    return code, out.getvalue(), err.getvalue()


class UncalibratedConfidenceTests(unittest.TestCase):
    """Hermetic: drive the pipeline with NO learner (cold) and the card
    renderer directly — the CLI's learner path is frozen at first import
    in-process, so a suite-order HOME with history must not leak in."""

    def _cold_result(self):
        import tempfile
        from pathlib import Path
        from assistant.core.pipeline import process
        with tempfile.TemporaryDirectory() as tmp:
            return process("make the bar taller",
                           file_path=Path(tmp) / "shell.json")

    def test_cold_card_says_uncalibrated(self):
        result = self._cold_result()
        # no learner attached -> nothing behind the number: uncalibrated
        self.assertIs(False, result.calibrated)

    def test_card_renders_the_honest_label(self):
        from assistant.core import cli as cortex_cli
        result = self._cold_result()
        result.calibrated = False
        lines = cortex_cli._render_turn(result)
        self.assertTrue(any("uncalibrated (no history)" in ln for ln in lines))

    def test_calibrated_card_shows_the_number(self):
        from assistant.core import cli as cortex_cli
        result = self._cold_result()
        result.calibrated = True
        result.confidence = 0.83
        lines = cortex_cli._render_turn(result)
        self.assertTrue(any("confidence 0.83" in ln for ln in lines))


class FlagOrderTests(unittest.TestCase):
    """D10: flags before and after the text both parse."""

    def test_do_json_before_and_after_text(self):
        code1, out1, err1 = hub(["do", "--json", "solve x^2 - 2 = 0"])
        code2, out2, err2 = hub(["do", "solve x^2 - 2 = 0", "--json"])
        self.assertEqual(code1, 0, err1)
        self.assertEqual(code2, 0, err2)
        p1, p2 = json.loads(out1), json.loads(out2)
        self.assertEqual(p1["result"]["real_roots_count"], 2)
        self.assertEqual(p2["result"]["real_roots_count"], 2)

    def test_route_json_before_and_after_text(self):
        code1, out1, err1 = hub(["route", "--json", "make the bar taller"])
        code2, out2, err2 = hub(["route", "make the bar taller", "--json"])
        self.assertEqual(code1, 0, err1)
        self.assertEqual(code2, 0, err2)
        self.assertEqual(json.loads(out1)["resolved_text"],
                         json.loads(out2)["resolved_text"])

    def test_eval_json_before_suite(self):
        code, out, err = hub(["eval", "--json", "calibration"])
        self.assertEqual(code, 0, err)
        self.assertIn("brier", json.loads(out)["metrics"])


if __name__ == "__main__":
    unittest.main()
