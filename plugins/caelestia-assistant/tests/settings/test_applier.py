"""Applier tests for the settings layer (the settings design notes (see module docstrings) §5, §9).

The applier is the ONLY module that writes, and only behind write=True.
Every test runs against fixture files in temp dirs and proves the write
protocol by DIRECTORY SNAPSHOT, not just by reading the target:

- dry-run (the default) writes NOTHING — no target, no backup, no tmp;
- apply writes exactly the target + its .assistant-backup sibling (the tmp
  is gone after the atomic os.replace) and preserves every other key;
- the backup slot is SINGLE (a second apply refreshes it);
- --restore is a byte-identical one-level undo; a missing backup is a
  clean error; a zero-byte backup restores the no-file state;
- a blocked (REJECTED) plan is refused wholesale — file untouched;
- a missing parent directory is an error, never a mkdir;
- a symlinked target is resolved so the link stays attached.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set

from assistant.capabilities.settings.applier import ApplierError, apply, restore
from assistant.capabilities.settings.parser import parse
from assistant.capabilities.settings.planner import plan

ORIGINAL = (
    "{\n"
    '    "bar": {"scale": 1.0},\n'
    '    "unrelated": {"keep": [1, 2], "nested": {"x": "y"}},\n'
    '    "solo": true\n'
    "}\n"
)


def _ops(sentence: str) -> List[Dict[str, Any]]:
    result = parse(sentence)
    assert result["verdict"] == "INTENT", sentence
    return result["ops"]


def _plan_for(sentence: str, target: Path) -> Dict[str, Any]:
    return plan(_ops(sentence), target)


class ApplierTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def target(self, content: str = ORIGINAL) -> Path:
        target = self.dir / "shell.json"
        target.write_text(content, encoding="utf-8")
        return target

    def snapshot(self) -> Set[str]:
        return {str(path.relative_to(self.dir)) for path in self.dir.rglob("*")}


class DryRunTests(ApplierTestCase):
    def test_dry_run_writes_nothing(self) -> None:
        # Existing target: unchanged, no backup, no tmp. Missing target:
        # still nothing at all is created.
        target = self.target()
        before = self.snapshot()
        result = apply(_plan_for("set the bar scale to 1.4", target), target, write=False)
        self.assertFalse(result["written"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

        missing = self.dir / "absent.json"
        result = apply(_plan_for("set the bar scale to 1.4", missing), missing, write=False)
        self.assertFalse(result["written"])
        self.assertEqual(self.snapshot(), before)  # nothing new appeared


class ApplyTests(ApplierTestCase):
    def test_apply_updates_target_and_preserves_every_other_key(self) -> None:
        target = self.target()
        result = apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        self.assertTrue(result["written"])
        self.assertEqual(result["changes"], 1)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["bar"]["scale"], 1.4)
        self.assertEqual(data["unrelated"], {"keep": [1, 2], "nested": {"x": "y"}})
        self.assertIs(data["solo"], True)

    def test_apply_serializes_with_four_space_indent_and_newline(self) -> None:
        target = self.target()
        apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        text = target.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertIn('\n    "bar"', text)  # 4-space indent, QJsonDocument style (§5.3)

    def test_apply_writes_backup_and_leaves_no_tmp(self) -> None:
        target = self.target()
        apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        backup = self.dir / "shell.json.assistant-backup"
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.read_bytes(), ORIGINAL.encode("utf-8"))
        self.assertFalse((self.dir / "shell.json.assistant-tmp").exists())

    def test_directory_snapshot_after_dry_run_and_apply(self) -> None:
        # The whole-write-scope proof: starting from an EMPTY directory
        # (target does not exist yet), a dry-run creates NOTHING; after an
        # apply the ONLY paths present are the target and its backup — the
        # tmp is consumed by the rename, and nothing else ever appears.
        target = self.dir / "shell.json"
        before = self.snapshot()
        apply(_plan_for("set the bar scale to 1.4", target), target, write=False)
        self.assertEqual(self.snapshot(), before)
        apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        self.assertEqual(
            self.snapshot(),
            {"shell.json", "shell.json.assistant-backup",
             # the bounded undo history sibling
             "shell.json.assistant-history.json"},
        )

    def test_apply_twice_keeps_a_single_backup_slot(self) -> None:
        target = self.target()
        apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        backup = self.dir / "shell.json.assistant-backup"
        self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))["bar"]["scale"], 1.0)
        apply(_plan_for("set the bar scale to 1.5", target), target, write=True)
        # The slot was REFRESHED to the second pre-apply state, not appended.
        self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))["bar"]["scale"], 1.4)
        self.assertEqual(
            [path.name for path in self.dir.iterdir() if "backup" in path.name],
            ["shell.json.assistant-backup"],
        )

    def test_no_op_plan_writes_nothing_and_no_backup(self) -> None:
        target = self.target(content='{"bar": {"scale": 1.0}}')
        before = self.snapshot()
        result = apply(_plan_for("set the bar scale to 1.0", target), target, write=True)
        self.assertFalse(result["written"])
        self.assertTrue(result["no_changes"])
        self.assertIn("no changes needed", result["message"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"bar": {"scale": 1.0}}')

    def test_blocked_plan_is_refused_wholesale(self) -> None:
        target = self.target()
        before = self.snapshot()
        blocked = _plan_for("set the bar scale to 9", target)  # REJECTED entry
        with self.assertRaises(ApplierError) as ctx:
            apply(blocked, target, write=True)
        self.assertIn("apply refused", str(ctx.exception))
        self.assertIn("nothing was written", str(ctx.exception))
        self.assertEqual(self.snapshot(), before)  # target AND backup untouched

    def test_apply_on_missing_target_writes_file_and_zero_byte_backup(self) -> None:
        target = self.dir / "shell.json"
        result = apply(_plan_for("set the bar scale to 1.2", target), target, write=True)
        self.assertTrue(result["written"])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["bar"]["scale"], 1.2)
        backup = self.dir / "shell.json.assistant-backup"
        self.assertEqual(backup.stat().st_size, 0)  # "there was no file" marker

    def test_apply_with_missing_parent_directory_is_an_error(self) -> None:
        target = self.dir / "no-such-dir" / "shell.json"
        with self.assertRaises(ApplierError) as ctx:
            apply(_plan_for("set the bar scale to 1.2", target), target, write=True)
        self.assertIn("not creating it", str(ctx.exception))
        self.assertFalse((self.dir / "no-such-dir").exists())

    def test_symlinked_target_is_resolved_not_detached(self) -> None:
        real = self.dir / "real.json"
        real.write_text('{"bar": {"scale": 1.0}}', encoding="utf-8")
        link = self.dir / "link.json"
        link.symlink_to(real)
        apply(_plan_for("set the bar scale to 1.3", link), link, write=True)
        self.assertTrue(link.is_symlink())  # the link survived
        self.assertEqual(json.loads(real.read_text(encoding="utf-8"))["bar"]["scale"], 1.3)
        # backup/tmp are siblings of the RESOLVED file (§5.3)
        self.assertTrue((self.dir / "real.json.assistant-backup").is_file())
        self.assertFalse((self.dir / "real.json.assistant-tmp").exists())
        self.assertFalse((self.dir / "link.json.assistant-backup").exists())


class RestoreTests(ApplierTestCase):
    def test_restore_is_byte_identical_and_consumes_the_slot(self) -> None:
        target = self.target()
        apply(_plan_for("set the bar scale to 1.4", target), target, write=True)
        target.write_text('{"bar": {"scale": 0.7}, "junk": 1}', encoding="utf-8")  # later writer
        result = restore(target)
        self.assertTrue(result["restored"])
        self.assertEqual(target.read_bytes(), ORIGINAL.encode("utf-8"))  # byte-identical
        with self.assertRaises(ApplierError) as ctx:
            restore(target)  # single slot already consumed
        self.assertIn("no backup to restore", str(ctx.exception))

    def test_restore_without_backup_is_a_clean_error(self) -> None:
        target = self.target()
        with self.assertRaises(ApplierError) as ctx:
            restore(target)
        self.assertIn("no backup to restore", str(ctx.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_restore_of_zero_byte_backup_deletes_the_target(self) -> None:
        # Apply onto a missing file, then restore -> back to the no-file
        # state, backup slot consumed, directory empty again.
        target = self.dir / "shell.json"
        apply(_plan_for("set the bar scale to 1.2", target), target, write=True)
        result = restore(target)
        self.assertTrue(result["restored"])
        self.assertIn("no-file state", result["message"])
        self.assertFalse(target.exists())
        self.assertEqual(self.snapshot(), {"shell.json.assistant-history.json"})


if __name__ == "__main__":
    unittest.main()
