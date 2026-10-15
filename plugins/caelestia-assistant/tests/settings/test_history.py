"""Bounded undo history tests (issue #120).

Pins the issue's asks — "Undo the last change", "Restore yesterday's
theme", "Revert my last customization" — against a temp target:
- every --apply records one entry (id, timestamp, label, per-op old/new);
- the ring is bounded at history.MAX_ENTRIES (12 >= the required 10),
  oldest evicted first;
- undo(1) / undo(N) revert and consume entries;
- undo_by_id reverts a specific entry ("restore yesterday's theme");
- undo does not itself appear in the history;
- the single-slot --restore keeps working alongside.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.settings import history
from assistant.capabilities.settings.applier import apply


def _plan(path: str, new) -> dict:
    from assistant.adapters.caelestia.registry import tool_by_path
    spec = tool_by_path(path)
    assert spec is not None
    return {"entries": [{"tool": spec.name, "path": path, "action": "set",
                         "raw": f"{path}={new}", "new": new}],
            "apply_blocked": False}


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "shell.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _apply(self, path: str, new, label: str = "") -> dict:
        return apply(_plan(path, new), self.target, write=True, label=label)

    def _read(self, path: str):
        node = json.loads(self.target.read_text())
        for segment in path.split("."):
            node = node.get(segment) if isinstance(node, dict) else None
        return node

    def test_apply_records_entries_with_old_and_new(self) -> None:
        self._apply("bar.scale", 1.2, label="bigger bar")
        entries = history.entries(self.target)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["label"], "bigger bar")
        self.assertEqual(e["ops"], [{"path": "bar.scale", "old": None,
                                     "new": 1.2}])
        self.assertIn("at", e)
        self.assertIn("id", e)

    def test_undo_last_change(self) -> None:
        self._apply("bar.scale", 1.2)
        self._apply("bar.scale", 1.4)
        result = history.undo(self.target, 1)
        self.assertTrue(result.get("restored") or result.get("written")
                        or "undid" in result.get("message", ""))
        self.assertEqual(self._read("bar.scale"), 1.2)

    def test_undo_multiple_steps(self) -> None:
        self._apply("bar.scale", 1.2)
        self._apply("bar.scale", 1.4)
        self._apply("bar.scale", 1.6)
        history.undo(self.target, 2)
        self.assertEqual(self._read("bar.scale"), 1.2)

    def test_undo_does_not_record_itself(self) -> None:
        self._apply("bar.scale", 1.2)
        history.undo(self.target, 1)
        self.assertEqual(history.entries(self.target), [])

    def test_undo_by_id_restores_yesterdays_theme(self) -> None:
        self._apply("bar.scale", 1.2, label="monday theme")
        self._apply("bar.persistent", False, label="tuesday tweak")
        entries = history.entries(self.target)
        monday = next(e for e in entries if e["label"] == "monday theme")
        result = history.undo_by_id(self.target, monday["id"])
        # the monday apply CREATED bar.scale (old=None): its undo removes
        # the key again — the shell falls back to the C++ default (1.0).
        self.assertIsNone(self._read("bar.scale"))
        self.assertEqual(self._read("bar.persistent"), False)  # untouched
        remaining = history.entries(self.target)
        self.assertEqual([e["label"] for e in remaining], ["tuesday tweak"])

    def test_ring_is_bounded_oldest_evicted(self) -> None:
        for i in range(history.MAX_ENTRIES + 3):
            self._apply("bar.scale", 0.6 + 0.01 * i, label=f"step {i}")
        entries = history.entries(self.target)
        self.assertEqual(len(entries), history.MAX_ENTRIES)
        self.assertEqual(history.MAX_ENTRIES, 12)
        self.assertGreaterEqual(history.MAX_ENTRIES, 10)  # the required bar
        # oldest evicted: the newest entry is the last apply; the first
        # recorded ("step 0") is gone.
        self.assertEqual(entries[0]["label"], f"step {history.MAX_ENTRIES + 2}")
        self.assertNotIn("step 0", [e["label"] for e in entries])

    def test_undo_empty_history_is_a_noop(self) -> None:
        result = history.undo(self.target, 1)
        self.assertEqual(result["restored"], False)

    def test_undo_by_unknown_id_raises(self) -> None:
        self._apply("bar.scale", 1.2)
        with self.assertRaises(history.HistoryError):
            history.undo_by_id(self.target, 999)

    def test_history_file_is_target_sibling(self) -> None:
        self._apply("bar.scale", 1.2)
        expected = self.dir / "shell.json.assistant-history.json"
        self.assertTrue(expected.exists())
        data = json.loads(expected.read_text())
        self.assertIn("entries", data)
        self.assertIn("next_id", data)

    def test_corrupt_history_refuses_undo(self) -> None:
        self._apply("bar.scale", 1.2)
        h = self.dir / "shell.json.assistant-history.json"
        h.write_text("{not json", encoding="utf-8")
        with self.assertRaises(history.HistoryError):
            history.undo(self.target, 1)

    def test_core_tool_single_slot_restore_still_works(self) -> None:
        from assistant.capabilities.settings.applier import restore
        self._apply("bar.scale", 1.5)
        result = restore(self.target)
        self.assertTrue(result["restored"])
        # zero-byte backup marker: the apply CREATED the file, so --restore
        # returns to the no-file state.
        self.assertFalse(self.target.exists())

    def test_cli_undo_smoke(self) -> None:
        from assistant.capabilities.settings import cli
        import io
        import contextlib
        self._apply("bar.scale", 1.4)
        # capture stdout: a test that leaks CLI renders into the
        # runner's output hides real failures (fixed in
        # exponential-build-3 Phase 2's hygiene pass)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(["--undo", "--file", str(self.target)])
            self.assertEqual(rc, 0)
            # the apply created the key on an empty file; undo removes it.
            self.assertIsNone(self._read("bar.scale"))
            # and the EFFECTIVE value (file + default) reads back as 1.0:
            from assistant.capabilities.settings.explain import explain as explain_fn
            self.assertIn("set to 1 ",
                          explain_fn("bar.scale", self.target)["answer"])
            rc = cli.main(["--history", "--file", str(self.target)])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
