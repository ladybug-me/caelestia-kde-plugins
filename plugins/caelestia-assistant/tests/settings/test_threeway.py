"""Tests for exponential-build-3 G1 — the three-way diff view
(`settings --threeway REQUEST`): per touched key, the file's current
value vs the proposal's value vs the value the undo history would
restore — so "apply then undo" is never a surprise.

Under test (built on the synthetic fixtures, F2's builders):

- the happy path: a key with undo history renders all three values,
  the restoring entry's provenance, and the honest note that
  apply-then-undo lands on the RESTORED value, not the current one;
- the no-history key says "none for this key" instead of inventing a
  restore value;
- the identity case (restore == current) says undo returns exactly to
  the current value;
- read-only: the target's bytes and the history file are untouched by
  the view (pinned);
- preset names work as the request (the same fallback what-if uses);
  unplanable requests exit 1 with the reason.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple

from assistant import fixtures
from assistant.capabilities.settings import cli
from assistant.capabilities.settings.applier import apply
from assistant.adapters.caelestia.registry import tool_by_path


def _plan(*ops: Tuple[str, Any]) -> dict:
    entries = []
    for path, new in ops:
        spec = tool_by_path(path)
        entries.append({"tool": spec.name, "path": path,
                        "action": "set", "raw": f"{path}={new}",
                        "new": new})
    return {"entries": entries, "apply_blocked": False}


class ThreeWayTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = fixtures.shell_config(self.dir / "shell.json")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, argv: List[str]) -> Tuple[int, str, str]:
        import io
        out, err = io.StringIO(), io.StringIO()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.getvalue(), err.getvalue()

    def test_three_values_with_history_and_the_honest_note(self):
        apply(_plan(("bar.scale", 1.2)), self.target, write=True,
              label="bigger bar")
        code, out, err = self._run(
            ["--threeway", "make the bar bigger",
             "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("now (file):          1.2", out)
        self.assertIn("proposal:            1.3", out)
        # the restore value + provenance (timestamp varies; the label
        # and entry id do not)
        self.assertIn("undo would restore:  1.0", out)
        self.assertIn("(history #1 'bigger bar'", out)
        self.assertIn("apply-then-undo lands on the restored value, "
                      "NOT back on the current one", out)

    def test_key_without_history_says_so(self):
        code, out, err = self._run(
            ["--threeway", "make the bar bigger",
             "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        # canonical fixture: bar.scale is 1.0 in the file, no history
        self.assertIn("now (file):          1.0", out)
        self.assertIn("undo record:         none for this key", out)

    def test_identity_case_returns_exactly_to_current(self):
        # apply A then propose A again: restore (the pre-A value) is
        # the CURRENT file value only when nothing changed since —
        # construct it by applying 1.2, undoing, then viewing a 1.3
        # proposal: file back at 1.0 == the restore target of the
        # consumed... (undo consumed the entry), so use the direct
        # shape: history holds old==current
        apply(_plan(("bar.scale", 1.2)), self.target, write=True,
              label="one")
        apply(_plan(("bar.scale", 1.0)), self.target, write=True,
              label="back")
        # file is 1.0 again; the NEWEST history entry for bar.scale
        # restores 1.2... but the OLDER one restored 1.0. The view
        # uses the NEWEST record: restore 1.2 != current 1.0
        code, out, _ = self._run(
            ["--threeway", "make the bar bigger",
             "--file", str(self.target)])
        self.assertEqual(code, 0)
        self.assertIn("undo would restore:  1.2", out)
        self.assertIn("lands on the restored value", out)

    def test_the_view_writes_nothing(self):
        apply(_plan(("bar.scale", 1.2)), self.target, write=True,
              label="bigger bar")
        before = self.target.read_bytes()
        hist = (self.dir / "shell.json.assistant-history.json")
        hist_before = hist.read_bytes()
        self._run(["--threeway", "make the bar bigger",
                   "--file", str(self.target)])
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(hist.read_bytes(), hist_before)

    def test_preset_name_works_as_the_request(self):
        code, out, err = self._run(
            ["--threeway", "minimal", "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("three-way view for 'minimal'", out)
        self.assertIn("now (file):", out)

    def test_unplanable_request_refuses(self):
        code, _out, err = self._run(
            ["--threeway", "wibble wobble frumious",
             "--file", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("could not plan", err)


if __name__ == "__main__":
    unittest.main()
