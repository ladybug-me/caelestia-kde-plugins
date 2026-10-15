"""F17/F18 tests — environment snapshots, restore-as-plan, and the
review-only export/import migration (settings/environments.py).

Under test:

- save captures the target content + macros/profiles metadata into the
  history file's bounded "environments" key (FIFO at MAX_ENVIRONMENTS);
  the target shell.json is untouched; duplicate names refuse;
- integrity: the canonical content hash is pinned; a tampered snapshot
  FAILS verification and is never a restore/export source;
- restore is a PLAN, not a write: diff_to_ops turns registry-known
  differences into set ops (and set-to-registry-default ops for keys
  the snapshot lacks — the applier cannot remove keys, and the raw op
  says so); unknown keys are REPORTED split by direction and never
  planned; the CLI restore rides the ordinary preview/confirm gates;
- export writes ONLY the caller-chosen bundle; import is REVIEW-ONLY:
  schema/hash validated, drift measured, stored as imported, nothing
  applied, the target untouched.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from assistant.capabilities.settings import cli, environments


def _mk(initial: dict) -> Tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="env-test-"))
    target = tmp / "shell.json"
    target.write_text(json.dumps(initial), encoding="utf-8")
    return tmp, target


def _cli(target: Path, argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main([*argv, "--file", str(target)])
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue()


INITIAL = {"bar": {"scale": 1.0, "dock": {"iconSize": 46}},
           "custom": {"userKey": "keep me"}}


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk(INITIAL)
        self.before = self.target.read_bytes()

    def test_save_and_list(self) -> None:
        entry = environments.save(self.target, "clean")
        self.assertEqual(entry["name"], "clean")
        self.assertTrue(entry["content_hash"])
        saved = environments.list_environments(self.target)
        self.assertEqual([e["name"] for e in saved], ["clean"])
        self.assertNotIn("content", saved[0])
        # the target file was not touched
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_duplicate_and_bad_name_refuse(self) -> None:
        environments.save(self.target, "clean")
        with self.assertRaises(environments.EnvironmentError):
            environments.save(self.target, "clean")
        with self.assertRaises(environments.EnvironmentError):
            environments.save(self.target, "bad name with ! bang")

    def test_fifo_bound(self) -> None:
        for i in range(environments.MAX_ENVIRONMENTS + 2):
            environments.save(self.target, f"e{i:02d}")
        saved = environments.list_environments(self.target)
        self.assertEqual(len(saved), environments.MAX_ENVIRONMENTS)
        self.assertEqual(saved[0]["name"], "e02")

    def test_tampered_snapshot_fails_verification(self) -> None:
        environments.save(self.target, "clean")
        hist = self.target.parent / (self.target.name +
                                     ".assistant-history.json")
        data = json.loads(hist.read_text(encoding="utf-8"))
        data["environments"][0]["content"]["bar"]["scale"] = 0.1
        hist.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(environments.EnvironmentError,
                                    "integrity"):
            environments.get(self.target, "clean")
        with self.assertRaises(environments.EnvironmentError):
            environments.export_bundle(self.target, "clean")

    def test_unknown_refuses_with_listing(self) -> None:
        environments.save(self.target, "clean")
        with self.assertRaisesRegex(environments.EnvironmentError, "clean"):
            environments.get(self.target, "nope")


class RestorePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk(INITIAL)
        environments.save(self.target, "clean")

    def test_diff_produces_set_and_reset_ops(self) -> None:
        # drift the live file: change one managed key, drop another,
        # change the unknown key
        drifted = {"bar": {"scale": 1.4}, "custom": {"userKey": "changed"}}
        self.target.write_text(json.dumps(drifted), encoding="utf-8")
        diff = environments.diff_to_ops(self.target, "clean")
        by_tool = {op["tool"]: op for op in diff["set_ops"] + diff["reset_ops"]}
        # bar.dock.iconSize vanished live -> set back to the snapshot value
        self.assertIn("setDockIconSize", by_tool)
        self.assertEqual(by_tool["setDockIconSize"]["value"], 46)
        # bar.scale changed -> set back to the snapshot value
        self.assertIn("setBarScale", by_tool)
        self.assertEqual(by_tool["setBarScale"]["value"], 1.0)
        # the unknown custom.userKey is REPORTED, never planned
        self.assertIn("custom.userKey", diff["unknown_only_in_live"])

    def test_reset_to_default_when_snapshot_lacks_key(self) -> None:
        # a snapshot taken WITHOUT the dock key, then the live file
        # gains it: restore resets it to the registry default (the
        # applier cannot remove keys), and the raw op says so
        self.target.write_text(json.dumps({"bar": {"scale": 1.0}}),
                               encoding="utf-8")
        environments.save(self.target, "lean")
        self.target.write_text(json.dumps(
            {"bar": {"scale": 1.0, "dock": {"iconSize": 30}}}),
            encoding="utf-8")
        diff = environments.diff_to_ops(self.target, "lean")
        by_tool = {op["tool"]: op for op in diff["set_ops"] + diff["reset_ops"]}
        self.assertIn("setDockIconSize", by_tool)
        self.assertIn("registry default", by_tool["setDockIconSize"]["raw"])

    def test_identical_environment_is_honest_noop(self) -> None:
        diff = environments.diff_to_ops(self.target, "clean")
        self.assertTrue(diff["identical"])

    def test_cli_restore_rides_confirm_gate(self) -> None:
        self.target.write_text(json.dumps(
            {"bar": {"scale": 1.4}}), encoding="utf-8")
        rc, out, err = _cli(self.target, ["--env-restore", "clean",
                                          "--apply"])
        self.assertEqual(rc, 1, "multi-change restore without --confirm "
                                "must refuse")
        self.assertIn("second consent", err)
        # dry-run writes nothing
        rc, out, err = _cli(self.target, ["--env-restore", "clean"])
        self.assertEqual(rc, 0, err)
        self.assertIn("env restore: clean", out)

    def test_cli_restore_applies_with_confirm(self) -> None:
        self.target.write_text(json.dumps(
            {"bar": {"scale": 1.4}}), encoding="utf-8")
        rc, out, err = _cli(self.target, ["--env-restore", "clean",
                                          "--apply", "--confirm"])
        self.assertEqual(rc, 0, err)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data["bar"]["scale"], 1.0)


class ExportImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp, self.target = _mk(INITIAL)
        environments.save(self.target, "clean")
        self.before = self.target.read_bytes()

    def test_export_roundtrip_and_review_only_import(self) -> None:
        bundle = environments.export_bundle(self.target, "clean")
        self.assertEqual(bundle["kind"], "caelestia-assistant-environment")
        # import into a DIFFERENT target (the "other machine")
        other_tmp, other = _mk({"bar": {"scale": 2.0}})
        other_before = other.read_bytes()
        entry = environments.import_bundle(other, bundle, "from-laptop")
        self.assertTrue(entry["imported"])
        self.assertEqual(entry["import_drift"]["bundled_tools"],
                         entry["import_drift"]["local_tools"])
        # nothing was applied on the other machine
        self.assertEqual(other.read_bytes(), other_before)
        # the imported environment restores through the F17 path
        diff = environments.diff_to_ops(other, "from-laptop")
        self.assertFalse(diff["identical"])

    def test_import_rejects_tampered_bundle(self) -> None:
        bundle = environments.export_bundle(self.target, "clean")
        bundle["content"]["bar"]["scale"] = 0.01
        _other_tmp, other = _mk({})
        with self.assertRaisesRegex(environments.EnvironmentError,
                                    "integrity"):
            environments.import_bundle(other, bundle, "evil")

    def test_import_rejects_wrong_schema_and_kind(self) -> None:
        _other_tmp, other = _mk({})
        with self.assertRaisesRegex(environments.EnvironmentError, "kind"):
            environments.import_bundle(other, {"schema": 1}, "x")
        with self.assertRaisesRegex(environments.EnvironmentError, "schema"):
            environments.import_bundle(
                other, {"kind": "caelestia-assistant-environment",
                        "schema": 99, "content": {}, "content_hash": "x"},
                "x")

    def test_cli_export_writes_only_the_bundle(self) -> None:
        bundle_path = self.tmp / "bundle.json"
        rc, out, err = _cli(self.target, ["--env-export", "clean",
                                          "--to", str(bundle_path)])
        self.assertEqual(rc, 0, err)
        self.assertTrue(bundle_path.exists())
        self.assertEqual(self.target.read_bytes(), self.before)

    def test_cli_import_needs_as_name(self) -> None:
        bundle_path = self.tmp / "bundle.json"
        rc, _, _ = _cli(self.target, ["--env-export", "clean",
                                      "--to", str(bundle_path)])
        rc, out, err = _cli(self.target, ["--env-import", str(bundle_path)])
        self.assertEqual(rc, 2)
        rc, out, err = _cli(self.target, ["--env-import", str(bundle_path),
                                          "--as", "laptop"])
        self.assertEqual(rc, 0, err)
        self.assertIn("review-only", out)


if __name__ == "__main__":
    unittest.main()
