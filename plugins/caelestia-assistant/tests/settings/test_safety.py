"""Safety tests for the settings layer (the settings design notes (see module docstrings) §6, §9).

Three structural guarantees, re-checked here so they can never regress
quietly:

- the repo-wide import-policy lint is clean WITH the new
  assistant/settings/** modules present (they are scanned automatically —
  same call the existing diagnostics safety-lint test makes);
- the settings package's own sources contain no executor and no network
  capability: no subprocess/socket/... import and no os.system/os.popen
  attribute anywhere. Checked by AST, because the package __init__
  docstring literally spells out "no subprocess/os.system anywhere in this
  package" — a raw substring grep would false-positive on the guarantee
  text itself;
- the CLI exit-code contract (§5.1): dry-run exits 0 and writes NOTHING;
  invalid target JSON exits 1 with the file untouched; a blocked --apply
  exits 1 with nothing written; usage errors exit 2; --list-tools exits 0.
  cli.main() is called IN-PROCESS (never via subprocess — the assistant
  itself must stay executor-free).
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple

from assistant.capabilities.diagnostics import schema_lint
from assistant.capabilities.settings import cli

SETTINGS_DIR = Path(__file__).resolve().parent.parent


class _Recorder:
    """Duck-typed stdout/stderr sink (no io import needed)."""

    def __init__(self) -> None:
        self.chunks: List[str] = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self.chunks)


class NoExecutorNoNetworkTests(unittest.TestCase):
    """Belt-and-braces AST scan of the settings package's own modules."""

    def _implementation_files(self) -> List[Path]:
        return sorted(SETTINGS_DIR.glob("*.py"))  # __init__, registry, parser, planner, applier, cli, __main__

    def test_no_forbidden_imports_or_os_attributes_anywhere(self) -> None:
        problems: List[str] = []
        for path in self._implementation_files():
            # the documented per-module quarantine (pkgprobe,
            # dbus_surface) exempts exactly the named imports of the
            # named files — pinned by test in agent/tests and
            # diagnostics/schema_lint.py's own guard
            quarantined = schema_lint._QUARANTINED_IMPORTS.get(
                path.name, frozenset())
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in quarantined:
                            continue  # this module's documented carve-out
                        if root in schema_lint.FORBIDDEN_IMPORTS:
                            problems.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in quarantined:
                        continue
                    if root in schema_lint.FORBIDDEN_IMPORTS:
                        problems.append(f"{path.name}: from {node.module} import ...")
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "os":
                        if node.attr in schema_lint.FORBIDDEN_OS_ATTRS:
                            problems.append(f"{path.name}: os.{node.attr}")
        self.assertEqual(problems, [])

class CliExitCodeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run_main(self, argv: List[str]) -> Tuple[Any, str, str]:
        out, err = _Recorder(), _Recorder()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.text(), err.text()

    def test_dry_run_exits_zero_and_writes_nothing(self) -> None:
        target = self.dir / "shell.json"  # does not exist yet
        code, out, _ = self._run_main(["make the bar thinner", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertFalse((self.dir / "shell.json.assistant-backup").exists())
        self.assertFalse((self.dir / "shell.json.assistant-tmp").exists())
        self.assertIn("--apply", out)  # the copy-paste confirmation line (§5.2)

    def test_dry_run_on_existing_target_leaves_it_byte_identical(self) -> None:
        target = self.dir / "shell.json"
        original = '{"bar": {"scale": 1.0}, "other": 7}'
        target.write_text(original, encoding="utf-8")
        code, _, _ = self._run_main(["make the bar thinner", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])

    def test_invalid_json_target_exits_one_untouched(self) -> None:
        target = self.dir / "shell.json"
        target.write_text("oops {", encoding="utf-8")
        code, _, err = self._run_main(["set the bar scale to 1.4", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", err)
        self.assertEqual(target.read_bytes(), b"oops {")

    def test_blocked_apply_exits_one_and_writes_nothing(self) -> None:
        target = self.dir / "shell.json"
        target.write_text('{"bar": {"scale": 1.4}}', encoding="utf-8")
        code, _, err = self._run_main(
            ["set the bar scale to 9", "--apply", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("apply refused", err)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"bar": {"scale": 1.4}}')
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])

    def test_usage_errors_exit_two(self) -> None:
        for argv in ([], ["--apply", "--restore"]):
            with self.assertRaises(SystemExit) as ctx:
                self._run_main(argv)
            self.assertEqual(ctx.exception.code, 2)

    def test_list_tools_exits_zero_with_18_rows(self) -> None:
        code, out, _ = self._run_main(["--list-tools"])
        self.assertEqual(code, 0)
        # Rows are "  {idx:>2}. ...": whitespace, a number, a dot. The
        # grew the registry 14 -> 17 (pitchBlack, maxPopups, maxNotifs);
        # additions include setDockBadges (17 -> 18).
        self.assertEqual(len(re.findall(r"^\s+\d+\. \S", out, re.MULTILINE)), 18)
        self.assertIn("[global-only]", out)
        self.assertIn("setPitchBlack", out)
        self.assertIn("setNotifsMaxPopups", out)
        self.assertIn("setNotifsMaxNotifs", out)
        self.assertIn("setDockBadges", out)

    def test_restore_without_backup_exits_one(self) -> None:
        code, _, err = self._run_main(["--restore", "--file", str(self.dir / "shell.json")])
        self.assertEqual(code, 1)
        self.assertIn("no backup to restore", err)


class ExtendedCoreToolApplyRestoreTests(unittest.TestCase):
    """These three tools ride the SAME write protocol as the original 14:
    --apply writes only the target + backup sibling, --restore undoes it
    byte-identically, dry-run writes nothing (the settings design notes (see module docstrings) §5, §6; task: add a
    case that a new tool's --apply writes the key and --restore undoes it)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run_main(self, argv: List[str]) -> Tuple[Any, str, str]:
        out, err = _Recorder(), _Recorder()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.text(), err.text()

    def test_extended_core_tool_dry_run_apply_restore_round_trip(self) -> None:
        target = self.dir / "shell.json"
        original = '{"notifs": {"maxPopups": 8}, "unrelated": 7}'
        target.write_text(original, encoding="utf-8")

        # Dry-run: nothing written, plan lists the new tool, exit 0.
        code, out, _ = self._run_main(
            ["set the max notification popups to 5", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("this run writes nothing", out)
        self.assertIn("notifs.maxPopups: 8 -> 5", out)
        self.assertIn("setNotifsMaxPopups", out)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])

        # --apply: writes the key, creates the single backup, no tmp left.
        code, out, _ = self._run_main(
            ["set the max notification popups to 5", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("applied on", out)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["notifs"]["maxPopups"], 5)
        self.assertEqual(data["unrelated"], 7)
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            ["shell.json", "shell.json.assistant-backup",
             "shell.json.assistant-history.json"],
        )
        self.assertEqual(
            (self.dir / "shell.json.assistant-backup").read_text(encoding="utf-8"),
            original,
        )

        # --restore: byte-identical undo; the slot is consumed afterwards.
        code, _, _ = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        code, _, err = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("no backup to restore", err)

    def test_pitch_black_apply_creates_nested_path(self) -> None:
        # appearance.pitchBlack on a file with no appearance subtree: the
        # applier creates only the intermediate objects along the registry
        # path, and --restore returns the exact original bytes.
        target = self.dir / "shell.json"
        original = "{\"border\": {\"thickness\": 10}}\n"
        target.write_text(original, encoding="utf-8")
        code, out, _ = self._run_main(
            ["make the shell pitch black", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertIs(data["appearance"]["pitchBlack"], True)
        self.assertEqual(data["border"], {"thickness": 10})
        code, _, _ = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_extended_core_tool_out_of_range_apply_is_refused(self) -> None:
        # 300 popups is outside the shipped stepper range 0-30: REJECTED,
        # apply_blocked, nothing written, exit 1 (#120: not applied).
        target = self.dir / "shell.json"
        original = '{"notifs": {"maxPopups": 8}}'
        target.write_text(original, encoding="utf-8")
        code, _, err = self._run_main(
            ["set the max notification popups to 300", "--apply", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("apply refused", err)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])


class DockBadgesApplyRestoreTests(unittest.TestCase):
    """setDockBadges rides the SAME write protocol as the previous 17
    (the settings design notes (see module docstrings) §5, §6; [C36]): dry-run default writes nothing, --apply
    writes only the target + backup sibling, --restore undoes it
    byte-identically, and an out-of-type value (a number on the bool tool)
    is REJECTED and blocks the apply — never coerced."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run_main(self, argv: List[str]) -> Tuple[Any, str, str]:
        out, err = _Recorder(), _Recorder()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.text(), err.text()

    def test_dock_badges_dry_run_apply_restore_round_trip(self) -> None:
        target = self.dir / "shell.json"
        original = '{"bar": {"dock": {"iconSize": 32}}, "unrelated": 7}'
        target.write_text(original, encoding="utf-8")

        # Dry-run: nothing written, plan shows the new tool, exit 0.
        code, out, _ = self._run_main(
            ["turn off app badges", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("this run writes nothing", out)
        self.assertIn("bar.dock.showBadges: true -> false", out)
        self.assertIn("setDockBadges", out)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])

        # --apply: writes ONLY the badges key — the sibling iconSize key in
        # the same nested object is untouched — plus the single backup.
        code, out, _ = self._run_main(
            ["turn off app badges", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("applied on", out)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertIs(data["bar"]["dock"]["showBadges"], False)
        self.assertEqual(data["bar"]["dock"]["iconSize"], 32)
        self.assertEqual(data["unrelated"], 7)
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            ["shell.json", "shell.json.assistant-backup",
             "shell.json.assistant-history.json"],
        )
        self.assertEqual(
            (self.dir / "shell.json.assistant-backup").read_text(encoding="utf-8"),
            original,
        )

        # --restore: byte-identical undo; the slot is consumed afterwards.
        code, _, _ = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        code, _, err = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("no backup to restore", err)

    def test_dock_badges_apply_creates_nested_path(self) -> None:
        # bar.dock.showBadges on a file with no bar.dock subtree: the
        # applier creates only the intermediate objects along the registry
        # path, and --restore returns the exact original bytes. (Note the
        # wording must REQUEST a change: the registry default is true, so
        # "show app badges" on an absent key is an honest no-op — the
        # planner resolves absent leaves to the default, §4.2.)
        target = self.dir / "shell.json"
        original = "{\"notifs\": {\"maxPopups\": 8}}\n"
        target.write_text(original, encoding="utf-8")
        code, out, _ = self._run_main(
            ["turn off app badges", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertIs(data["bar"]["dock"]["showBadges"], False)
        self.assertEqual(data["notifs"], {"maxPopups": 8})
        code, _, _ = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_dock_badges_number_on_bool_is_rejected_not_coerced(self) -> None:
        # "3" on the on/off tool: a REJECTED entry ("on/off only; it has no
        # magnitude"), apply_blocked, nothing written, exit 1 — #120's
        # "simply isn't applied" (§3.4/§3.5/§4.5).
        target = self.dir / "shell.json"
        original = '{"bar": {"dock": {"showBadges": true}}}'
        target.write_text(original, encoding="utf-8")
        code, _, err = self._run_main(
            ["set the app badges to 3", "--apply", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("apply refused", err)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shell.json"])


if __name__ == "__main__":
    unittest.main()
