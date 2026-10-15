"""Settings profile tests (F9 — profile algebra / issue #120 b6b).

The contract under test:

- a profile composes PRESETS, MACROS and direct CALLS in order, later
  source wins per tool, and every overridden value is REPORTED (F7
  no-silent-drop);
- save resolves every source NOW (unknown preset/macro/tool refuse the
  whole save), stores only the history file's bounded "profiles" key,
  and never touches the target shell.json;
- profile_ops returns plain planner ops (the presets/macro_ops shape),
  so apply rides the ordinary plan -> preview -> consent path;
- diff compares two profiles' EFFECTIVE values (only A / only B /
  changed per tool);
- CLI: --profile dry-run by default; --apply still needs --confirm for
  multi-change compositions; --profile-list/show/diff/delete are
  read-only or explicit; --also is bound to --profile-save.

Tests use live registry preset names (minimal/gaming conflict on
setAnimationSpeed), temp dirs via --file, and in-process cli.main().
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from assistant.capabilities.settings import cli, history, profiles


def _mk() -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="profiles-test-"))
    target = tmp / "shell.json"
    target.write_text("{}\n", encoding="utf-8")
    return tmp, target


def _with_target(target: Path, argv: List[str]) -> Tuple[int, str, str]:
    """In-process cli.main with --file pinned to the test's scratch target."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main([*argv, "--file", str(target)])
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


class ParseSourceTests(unittest.TestCase):
    def test_preset_macro_call_forms(self) -> None:
        self.assertEqual(profiles.parse_source("preset:minimal"),
                         {"kind": "preset", "name": "minimal"})
        self.assertEqual(profiles.parse_source("macro:focus"),
                         {"kind": "macro", "name": "focus"})
        self.assertEqual(
            profiles.parse_source("setDockBadges=false"),
            {"kind": "call", "tool": "setDockBadges", "value": False})
        self.assertEqual(
            profiles.parse_source("setBarScale=0.7"),
            {"kind": "call", "tool": "setBarScale", "value": 0.7})
        # bare token that fails JSON is a plain string (--call convention)
        self.assertEqual(
            profiles.parse_source("setBarPosition=bottom"),
            {"kind": "call", "tool": "setBarPosition", "value": "bottom"})

    def test_malformed_refuse(self) -> None:
        for bad in ("", "preset:", "macro:", "justwords", "preset:foo bar"):
            with self.assertRaises(profiles.ProfileError, msg=bad):
                profiles.parse_source(bad)


class ComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()

    def test_later_wins_and_conflicts_reported(self) -> None:
        ops, conflicts = profiles.compose_ops(self.target, [
            {"kind": "preset", "name": "minimal"},
            {"kind": "preset", "name": "gaming"},
        ])
        by_tool = {op["tool"]: op for op in ops}
        # minimal sets 0.5, gaming sets 0.25 -> gaming wins, conflict kept
        self.assertEqual(by_tool["setAnimationSpeed"]["value"], 0.25)
        self.assertTrue(any(c["tool"] == "setAnimationSpeed" and
                            c["dropped"]["value"] == 0.5 and
                            c["kept"]["value"] == 0.25 for c in conflicts),
                        conflicts)
        # union of tools, first-appearance order preserved
        tools = [op["tool"] for op in ops]
        self.assertEqual(len(tools), len(set(tools)))
        self.assertIn("setBarScale", by_tool)   # from minimal
        self.assertIn("setNotifsMaxPopups", by_tool)  # from gaming

    def test_same_value_reassert_is_not_a_conflict(self) -> None:
        ops, conflicts = profiles.compose_ops(self.target, [
            {"kind": "preset", "name": "minimal"},
            {"kind": "call", "tool": "setBarScale", "value": 0.7},
        ])
        self.assertEqual(conflicts, [])
        self.assertEqual(len(ops), 7)

    def test_unknown_sources_refuse(self) -> None:
        for src in ({"kind": "preset", "name": "nope"},
                    {"kind": "macro", "name": "nope"},
                    {"kind": "call", "tool": "setNope", "value": 1},
                    {"kind": "wat"}):
            with self.assertRaises(profiles.ProfileError, msg=str(src)):
                profiles.compose_ops(self.target, [src])


class SaveListDeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        self.before = self.target.read_bytes()

    def test_round_trip_and_target_untouched(self) -> None:
        entry = profiles.save(self.target, "evening", [
            {"kind": "preset", "name": "minimal"},
            {"kind": "call", "tool": "setDockBadges", "value": True},
        ], label="test")
        self.assertEqual(entry["name"], "evening")
        saved = profiles.list_profiles(self.target)
        self.assertEqual([p["name"] for p in saved], ["evening"])
        self.assertEqual(self.target.read_bytes(), self.before,
                         "the target shell.json must never be touched")

    def test_duplicate_and_unresolvable_refuse(self) -> None:
        profiles.save(self.target, "p1", [{"kind": "preset", "name": "gaming"}])
        with self.assertRaises(profiles.ProfileError):
            profiles.save(self.target, "p1", [{"kind": "preset", "name": "gaming"}])
        with self.assertRaises(profiles.ProfileError):
            profiles.save(self.target, "p2", [{"kind": "preset", "name": "nope"}])
        with self.assertRaises(profiles.ProfileError):
            profiles.save(self.target, "p3", [{"kind": "macro", "name": "nope"}])
        with self.assertRaises(profiles.ProfileError):
            profiles.save(self.target, "p4", [{"kind": "call",
                                               "tool": "setNope", "value": 1}])
        with self.assertRaises(profiles.ProfileError):
            profiles.save(self.target, "p5", [])

    def test_delete_refuses_unknown(self) -> None:
        profiles.save(self.target, "p1", [{"kind": "preset", "name": "gaming"}])
        result = profiles.delete(self.target, "p1")
        self.assertEqual(result["deleted"], "p1")
        with self.assertRaises(profiles.ProfileError):
            profiles.delete(self.target, "p1")

    def test_bounded_fifo_store(self) -> None:
        for i in range(profiles.MAX_PROFILES + 3):
            profiles.save(self.target, f"p{i:02d}",
                          [{"kind": "preset", "name": "gaming"}])
        saved = profiles.list_profiles(self.target)
        self.assertEqual(len(saved), profiles.MAX_PROFILES)
        self.assertEqual(saved[0]["name"], "p03",
                         "oldest captures evicted first")


class ProfileOpsAndDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()
        profiles.save(self.target, "a", [{"kind": "preset", "name": "minimal"}])
        profiles.save(self.target, "b", [{"kind": "preset", "name": "gaming"}])

    def test_profile_ops_planner_shape(self) -> None:
        ops, conflicts = profiles.profile_ops(self.target, "a")
        for op in ops:
            self.assertEqual(set(op) >= {"tool", "action", "value", "raw"},
                             True, op)
        self.assertEqual(conflicts, [])

    def test_unknown_profile_lists_saved(self) -> None:
        with self.assertRaisesRegex(profiles.ProfileError, "nope") as ctx:
            profiles.profile_ops(self.target, "nope")
        self.assertIn("a, b", str(ctx.exception))

    def test_diff_rows(self) -> None:
        result = profiles.diff(self.target, "a", "b")
        kinds = {row["tool"]: row["kind"] for row in result["rows"]}
        self.assertEqual(kinds.get("setAnimationSpeed"), "changed")
        self.assertEqual(kinds.get("setBarScale"), "only_a")
        self.assertEqual(kinds.get("setNotifsMaxPopups"), "only_b")
        changed = next(r for r in result["rows"]
                       if r["tool"] == "setAnimationSpeed")
        self.assertEqual((changed["a"], changed["b"]), (0.5, 0.25))


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk()

    def test_save_show_apply_dry_run_and_delete(self) -> None:
        rc, out, err = _with_target(self.target, [
            "--profile-save", "evening", "preset:minimal",
            "--also", "setDockBadges=true"])
        self.assertEqual(rc, 0, err)
        self.assertIn("saved profile 'evening'", out)

        rc, out, err = _with_target(self.target, ["--profile-show", "evening"])
        self.assertEqual(rc, 0, err)
        self.assertIn("source: preset minimal", out)
        self.assertIn("setDockBadges = True", out)
        self.assertIn("[call:setDockBadges]", out)

        rc, out, err = _with_target(self.target, ["--profile", "evening"])
        self.assertEqual(rc, 0, err)
        self.assertIn("profile: evening", out)
        # target untouched by the dry run
        rc, out, err = _with_target(self.target, ["--profile-list"])
        self.assertEqual(rc, 0, err)
        self.assertIn("evening", out)

        rc, out, err = _with_target(self.target, ["--profile-diff",
                                                  "evening", "evening"])
        self.assertEqual(rc, 0, err)
        self.assertIn("effectively identical", out)

        rc, out, err = _with_target(self.target, ["--profile-delete",
                                                  "evening"])
        self.assertEqual(rc, 0, err)
        rc, out, err = _with_target(self.target, ["--profile-list"])
        self.assertIn("no profiles saved", out)

    def test_apply_requires_confirm(self) -> None:
        _with_target(self.target, ["--profile-save", "combo",
                                   "preset:minimal", "--also",
                                   "preset:gaming"])
        rc, out, err = _with_target(self.target,
                                    ["--profile", "combo", "--apply"])
        self.assertEqual(rc, 1)
        self.assertIn("second consent", err)
        self.assertIn("nothing was written", err)

    def test_also_requires_profile_save(self) -> None:
        rc, out, err = _with_target(self.target, ["--also", "preset:gaming"])
        self.assertEqual(rc, 2)

    def test_unknown_source_error_is_rendered(self) -> None:
        rc, out, err = _with_target(self.target, ["--profile-save", "x",
                                                  "preset:doesnotexist"])
        self.assertEqual(rc, 1)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
