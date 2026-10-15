"""F24 tests — accessible output mode + the reduced-motion recipe
(settings/plain.py).

Under test:

- plain_requested: explicit flag wins; CAELESTIA_ASSISTANT_PLAIN
  accepts 1/true/yes/on (case-insensitive); unset/other values are
  False;
- to_linear_rows: one record per row, one fact per line, deterministic;
- the plain contract (FORBIDDEN_PLAIN_RE): ANSI, box-drawing and emoji
  are violations; normal text passes;
- --list-tools --plain renders linear records for every tool and
  passes the contract; the recipe command is inert and composes ONLY
  registry tools (validated by feeding its sources through
  profiles.parse_source + compose_ops).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.settings import cli, plain, profiles


def _cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


class PlainRequestedTests(unittest.TestCase):
    def tearDown(self) -> None:
        import os
        os.environ.pop("CAELESTIA_ASSISTANT_PLAIN", None)

    def test_env_var_forms(self) -> None:
        import os
        for value, expected in (("1", True), ("true", True),
                                ("YES", True), ("on", True),
                                ("", False), ("0", False), ("off", False)):
            os.environ["CAELESTIA_ASSISTANT_PLAIN"] = value
            self.assertEqual(plain.plain_requested(), expected, value)

    def test_explicit_flag_wins(self) -> None:
        import os
        os.environ["CAELESTIA_ASSISTANT_PLAIN"] = "1"
        self.assertFalse(plain.plain_requested(explicit=False))


class LinearRowsTests(unittest.TestCase):
    def test_one_fact_per_line(self) -> None:
        lines = plain.to_linear_rows(
            ("tool", "kind"),
            [("setBarScale", "float"), ("setBlurEnabled", "bool")])
        self.assertEqual(lines, [
            "record 1:", "  tool: setBarScale", "  kind: float",
            "record 2:", "  tool: setBlurEnabled", "  kind: bool",
        ])

    def test_forbidden_pattern_detector(self) -> None:
        plain.assert_plain_safe("plain text with tool=setBarScale")
        for bad in ("\x1b[1mbold\x1b[0m", "┌─box─┐", "\u2705 done"):
            with self.assertRaises(AssertionError, msg=bad):
                plain.assert_plain_safe(bad)


class CliPlainTests(unittest.TestCase):
    def test_list_tools_plain_is_linear_and_safe(self) -> None:
        rc, out, err = _cli(["--list-tools", "--plain"])
        self.assertEqual(rc, 0, err)
        plain.assert_plain_safe(out)
        self.assertIn("record 1:", out)
        self.assertIn("  tool: setBarScale", out)
        # every tool appears as its own record
        self.assertEqual(out.count("record "), 272)

    def test_env_var_only_activation(self) -> None:
        import os
        os.environ["CAELESTIA_ASSISTANT_PLAIN"] = "1"
        self.addCleanup(os.environ.pop, "CAELESTIA_ASSISTANT_PLAIN", None)
        rc, out, err = _cli(["--list-tools"])
        self.assertEqual(rc, 0, err)
        plain.assert_plain_safe(out)
        self.assertIn("record 1:", out)


class RecipeTests(unittest.TestCase):
    def test_recipe_command_is_inert_and_registry_valid(self) -> None:
        command = plain.recipe_command()
        self.assertTrue(command.startswith("SUGGESTED_NOT_EXECUTED"))
        # every knob in the recipe must be a registry tool whose value
        # parses as a source and composes cleanly
        tmp = Path(tempfile.mkdtemp(prefix="recipe-test-"))
        target = tmp / "shell.json"
        target.write_text("{}", encoding="utf-8")
        _, _, rest = command.partition("--profile-save reduced-motion ")
        sources = rest.split()
        parsed = [profiles.parse_source(s) for s in sources]
        ops, conflicts = profiles.compose_ops(target, parsed)
        self.assertEqual(len(ops), len(plain.REDUCED_MOTION_RECIPE))
        self.assertEqual(conflicts, [])

    def test_a11y_recipe_flag_renders(self) -> None:
        rc, out, err = _cli(["--a11y-recipe"])
        self.assertEqual(rc, 0, err)
        self.assertIn("reduced-motion recipe", out)
        self.assertIn("SUGGESTED_NOT_EXECUTED", out)
        self.assertIn("--profile-save reduced-motion", out)


if __name__ == "__main__":
    unittest.main()
