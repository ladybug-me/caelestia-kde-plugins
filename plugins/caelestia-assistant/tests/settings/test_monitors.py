"""F20 tests — monitor-aware planning (settings/monitors.py).

Upstream facts the module builds on (verified against the checkout):
every config type layers per-screen overrides from
``<configDir>/monitors/<screen>/<same file>`` (Root::forScreen,
monitorConfigDir), with the same schema as the global config.

Under test:

- resolve_target validates screen names (a single component: no '/',
  no '..') and yields monitors/<screen>/shell.json next to the global
  target;
- discover() reports only EXISTING override files with their
  managed/unknown key counts (it never invents screens);
- monitor_behavior_tools() returns the registry's monitor-behavior
  tools with citations (setMonitor among them);
- the CLI: --monitor routes a request/--call plan+apply at the screen's
  override file (backup sibling + history included, like any target),
  --monitor + --file is refused, --monitors renders the read-only
  report.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from assistant.capabilities.settings import cli, monitors


def _mk(initial=None) -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="monitors-test-"))
    target = tmp / "shell.json"
    target.write_text(json.dumps(initial or {}), encoding="utf-8")
    return tmp, target


def _cli(argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


class ResolveTests(unittest.TestCase):
    def test_resolve_layout_and_validation(self) -> None:
        tmp, target = _mk()
        resolved = monitors.resolve_target(target, "DP-1")
        self.assertEqual(resolved,
                         tmp / "monitors" / "DP-1" / "shell.json")
        for bad in ("../evil", "a/b", "", ".hidden"):
            with self.assertRaises(ValueError, msg=bad):
                monitors.resolve_target(target, bad)
        # a dot-containing but single-component name is fine
        ok = monitors.resolve_target(target, "DP-1.alt")
        self.assertIn("DP-1.alt", str(ok))

    def test_discover_sees_only_existing_files(self) -> None:
        tmp, target = _mk()
        self.assertEqual(monitors.discover(target), [])
        override = tmp / "monitors" / "DP-1" / "shell.json"
        override.parent.mkdir(parents=True)
        override.write_text(json.dumps(
            {"bar": {"scale": 0.8}, "ghost": {"key": 1}}), encoding="utf-8")
        found = monitors.discover(target)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["screen"], "DP-1")
        self.assertEqual(found[0]["managed_keys"], 1)
        self.assertEqual(found[0]["unknown_keys"], 1)


class BehaviorToolsTests(unittest.TestCase):
    def test_set_monitor_listed_with_citations(self) -> None:
        rows = monitors.monitor_behavior_tools()
        names = {row["name"] for row in rows}
        self.assertIn("setMonitor", names)
        row = next(r for r in rows if r["name"] == "setMonitor")
        self.assertTrue(row["citations"])
        self.assertIn("all", row["enum"])


class CliTests(unittest.TestCase):
    def test_monitor_target_routes_apply(self) -> None:
        # --monitor derives the override path from the DEFAULT target
        # (the config dir next to the user's shell.json), so the test
        # pins HOME; no --file is passed.
        import os
        tmp = Path(tempfile.mkdtemp(prefix="monitors-cli-"))
        old_home = os.environ.get("HOME")
        old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["HOME"] = str(tmp)
        # CI images (ubuntu-24.04) export XDG_CONFIG_HOME, which would
        # win over HOME; this test pins the HOME fallback path, so the
        # var must be cleared for the duration.
        os.environ.pop("XDG_CONFIG_HOME", None)

        def _restore_home():
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_xdg is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old_xdg
        self.addCleanup(_restore_home)
        rc, out, err = _cli(["set the bar scale to 0.8", "--apply",
                             "--confirm", "--monitor", "DP-1"])
        self.assertEqual(rc, 0, err)
        override = tmp / ".config" / "caelestia" / "monitors" / "DP-1" / "shell.json"
        self.assertTrue(override.exists(), "the override file is written")
        data = json.loads(override.read_text(encoding="utf-8"))
        self.assertEqual(data["bar"]["scale"], 0.8)
        # the override file has its own backup + history siblings
        names = sorted(p.name for p in override.parent.iterdir())
        self.assertIn("shell.json.assistant-backup", names)
        self.assertIn("shell.json.assistant-history.json", names)
        # the global config was never touched
        self.assertFalse((tmp / ".config" / "caelestia" / "shell.json").exists())

    def test_monitor_and_file_are_refused(self) -> None:
        rc, out, err = _cli(["--monitor", "DP-1", "--file", "/tmp/x.json",
                             "make the bar thinner"])
        self.assertEqual(rc, 2)

    def test_bad_screen_name_is_refused(self) -> None:
        rc, out, err = _cli(["--monitor", "../evil", "make the bar thinner"])
        self.assertEqual(rc, 2)
        self.assertIn("monitor names", err)

    def test_monitors_report_renders(self) -> None:
        tmp, target = _mk()
        rc, out, err = _cli(["--monitors", "--file", str(target)])
        self.assertEqual(rc, 0, err)
        self.assertIn("read-only report", out)
        self.assertIn("setMonitor", out)
        self.assertIn("rootnodes.cpp", out)


if __name__ == "__main__":
    unittest.main()
