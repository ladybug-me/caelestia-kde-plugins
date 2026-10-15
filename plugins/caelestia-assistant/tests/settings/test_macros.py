"""Settings macro tests (exponential-build-3 C3 — capture an APPROVED
proposal sequence as a named, replayable template; Cypher (ed.) 1993
programming-by-demonstration pattern, with re-approval on every replay).

The contract under test:

- capture works ONLY from an undo-history entry (an apply that already
  passed preview -> consent -> apply once); the newest entry is the
  default, --from-id selects any entry in the ring;
- capture stores the APPLIED values {tool, path, value} with label and
  provenance, bounded FIFO at MAX_MACROS, never silently overwriting a
  used name;
- capture/delete write ONLY the history file's "macros" key (the A3
  undo_log precedent: same sibling path, same _save atomic write; the
  target file's bytes are untouched and the applier stays the only
  writer of shell.json);
- replay returns plain planner ops (the presets.preset_ops shape), so
  it rides the ordinary plan path: dry-run writes nothing, a replay of
  already-present values is an honest no-op, and a stale tool name is
  an error entry that blocks the apply;
- CLI consent discipline: --macro NAME is a dry-run preview by default;
  with --apply it STILL refuses without --confirm (single-change macros
  included — every replay is re-approved); with --apply --confirm it
  writes and records its OWN history entry (label "macro: NAME"), so
  bounded undo works per replay.

Tests use the 18-tool core registry names only (bar.scale,
bar.persistent, bar.position — the 2-pre contract pins their values),
temp dirs via --file, and in-process cli.main() calls (never a
subprocess — the assistant stays executor-free).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple

from assistant.capabilities.settings import cli, history, macros
from assistant.capabilities.settings.applier import apply
from assistant.adapters.caelestia.registry import tool_by_path
from tests.settings.test_cli_calls import _Recorder


def _plan(*ops: Tuple[str, Any]) -> dict:
    """A plan of set ops (path, value) — the same shape the planner
    emits, so apply() exercises the real writer path."""
    entries = []
    for path, new in ops:
        spec = tool_by_path(path)
        assert spec is not None, path
        entries.append({"tool": spec.name, "path": path,
                        "action": "set", "raw": f"{path}={new}",
                        "new": new})
    return {"entries": entries, "apply_blocked": False}


class MacroTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "shell.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _apply(self, *ops: Tuple[str, Any], label: str = "") -> dict:
        return apply(_plan(*ops), self.target, write=True, label=label)

    def _read(self, path: str) -> Any:
        node = json.loads(self.target.read_text())
        for segment in path.split("."):
            node = node.get(segment) if isinstance(node, dict) else None
        return node

    def _run_main(self, argv: List[str]) -> Tuple[Any, str, str]:
        out, err = _Recorder(), _Recorder()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.text(), err.text()


class CaptureTests(MacroTestCase):
    def test_capture_from_newest_entry(self) -> None:
        self._apply(("bar.scale", 1.2), label="bigger bar")
        self._apply(("bar.persistent", False), label="calm bar")
        macro = macros.save(self.target, "calm")
        # captured from the NEWEST entry, with provenance
        self.assertEqual(macro["name"], "calm")
        self.assertEqual(macro["label"], "calm bar")
        self.assertEqual(macro["from_id"], 2)
        self.assertEqual(macro["ops"],
                         [{"tool": "setBarPersistent",
                           "path": "bar.persistent",
                           "value": False}])
        self.assertIn("captured_at", macro)
        # and it is listed
        self.assertEqual([m["name"] for m in
                          macros.list_macros(self.target)], ["calm"])

    def test_capture_from_specific_id(self) -> None:
        self._apply(("bar.scale", 1.2), label="bigger bar")
        self._apply(("bar.persistent", False), label="calm bar")
        macro = macros.save(self.target, "bigger", from_id=1)
        self.assertEqual(macro["from_id"], 1)
        self.assertEqual(macro["label"], "bigger bar")
        self.assertEqual(macro["ops"],
                         [{"tool": "setBarScale", "path": "bar.scale",
                           "value": 1.2}])

    def test_capture_multi_op_sequence(self) -> None:
        # an approved SEQUENCE (multi-op plan) captures as a sequence
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        macro = macros.save(self.target, "evening mode")
        self.assertEqual([op["tool"] for op in macro["ops"]],
                         ["setBarScale", "setBarPersistent"])
        self.assertEqual([op["value"] for op in macro["ops"]],
                         [1.3, False])

    def test_empty_history_refuses(self) -> None:
        with self.assertRaises(macros.MacroError) as caught:
            macros.save(self.target, "nothing")
        self.assertIn("undo history is empty", str(caught.exception))
        self.assertIn("already approved", str(caught.exception))

    def test_unknown_id_refuses(self) -> None:
        self._apply(("bar.scale", 1.2))
        with self.assertRaises(macros.MacroError) as caught:
            macros.save(self.target, "x", from_id=99)
        self.assertIn("no history entry with id 99", str(caught.exception))

    def test_bad_names_refuse(self) -> None:
        self._apply(("bar.scale", 1.2))
        for bad in ("", " leading-space", "x" * 33, "bad/name",
                    "semi;colon"):
            with self.assertRaises(macros.MacroError, msg=repr(bad)):
                macros.save(self.target, bad)
        # a good maximal name passes (32 chars, starts alnum)
        ok = "a" + "b" * 31
        self.assertEqual(macros.save(self.target, ok)["name"], ok)

    def test_duplicate_name_never_silently_overwrites(self) -> None:
        self._apply(("bar.scale", 1.2), label="one")
        self._apply(("bar.scale", 1.4), label="two")
        macros.save(self.target, "focus", from_id=1)
        with self.assertRaises(macros.MacroError) as caught:
            macros.save(self.target, "focus", from_id=2)
        self.assertIn("already exists", str(caught.exception))
        self.assertIn("--macro-delete focus", str(caught.exception))
        # the FIRST capture is intact
        self.assertEqual(
            macros.list_macros(self.target)[0]["ops"][0]["value"], 1.2)

    def test_capture_writes_only_the_history_file(self) -> None:
        self._apply(("bar.scale", 1.2), label="bigger bar")
        before = self.target.read_bytes()
        files_before = sorted(p.name for p in self.dir.iterdir())
        macros.save(self.target, "focus")
        self.assertEqual(self.target.read_bytes(), before)
        files_after = sorted(p.name for p in self.dir.iterdir())
        # the history sibling is the ONLY new/changed path (it existed
        # already: the apply recorded into it); no new write surface
        self.assertEqual(files_after, files_before)
        # and the macros key lives INSIDE the history document
        doc = json.loads(
            (self.dir / "shell.json.assistant-history.json").read_text())
        self.assertIn("macros", doc)
        self.assertEqual(doc["macros"][0]["name"], "focus")

    def test_stale_path_refuses_the_whole_capture(self) -> None:
        self._apply(("bar.scale", 1.2), label="bigger bar")
        # a hand-edited history entry referencing a path the registry
        # no longer knows (the honest stale-registry scenario)
        hpath = self.dir / "shell.json.assistant-history.json"
        doc = json.loads(hpath.read_text())
        doc["entries"][0]["ops"].append(
            {"path": "bar.removedInAnUpdate", "old": None, "new": 3})
        hpath.write_text(json.dumps(doc))
        with self.assertRaises(macros.MacroError) as caught:
            macros.save(self.target, "stale")
        self.assertIn("no longer in the registry",
                      str(caught.exception))
        self.assertIn("bar.removedInAnUpdate", str(caught.exception))
        self.assertEqual(macros.list_macros(self.target), [])

    def test_store_survives_history_ring_churn(self) -> None:
        self._apply(("bar.scale", 1.2), label="one")
        macros.save(self.target, "keepme", from_id=1)
        # push MAX_ENTRIES applies through the ring: entries evict, the
        # macro store must survive inside the same document
        for i in range(history.MAX_ENTRIES + 2):
            self._apply(("bar.scale", 1.0 + (i % 3) * 0.1),
                        label=f"churn {i}")
        saved = macros.list_macros(self.target)
        self.assertEqual([m["name"] for m in saved], ["keepme"])
        self.assertEqual(saved[0]["ops"][0]["value"], 1.2)

    def test_fifo_eviction_at_max_macros(self) -> None:
        for i in range(macros.MAX_MACROS + 1):
            self._apply(("bar.scale", 1.0 + i / 10), label=f"c{i}")
            macros.save(self.target, f"macro{i:02d}")
        names = [m["name"] for m in macros.list_macros(self.target)]
        self.assertEqual(len(names), macros.MAX_MACROS)
        self.assertNotIn("macro00", names)  # oldest evicted first
        self.assertIn(f"macro{macros.MAX_MACROS:02d}", names)


class ReplayTests(MacroTestCase):
    def _capture_evening(self) -> None:
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        macros.save(self.target, "evening mode")

    def test_macro_ops_shape_matches_preset_ops(self) -> None:
        self._capture_evening()
        ops = macros.macro_ops(self.target, "evening mode")
        from assistant.capabilities.settings.presets import preset_ops
        preset_shaped = preset_ops("minimal")  # the shape reference
        for op, reference in zip(ops, preset_shaped + preset_shaped):
            self.assertEqual(sorted(op), sorted(reference))
            break  # one shape check is enough; keys are pinned below
        self.assertEqual(
            ops,
            [{"tool": "setBarScale", "action": "set", "value": 1.3,
              "raw": "macro evening mode: setBarScale=1.3"},
             {"tool": "setBarPersistent", "action": "set", "value": False,
              "raw": "macro evening mode: setBarPersistent=False"}])

    def test_replay_reapplies_after_a_different_change(self) -> None:
        self._capture_evening()
        # a later apply moves the values elsewhere
        self._apply(("bar.scale", 1.0), ("bar.persistent", True),
                    label="reset")
        self.assertEqual(self._read("bar.scale"), 1.0)
        # replay: plan the macro ops and apply through the REAL writer
        from assistant.capabilities.settings import planner
        plan = planner.plan(macros.macro_ops(self.target, "evening mode"),
                            self.target)
        self.assertFalse(plan["apply_blocked"])
        result = apply(plan, self.target, write=True,
                       label="macro: evening mode")
        self.assertTrue(result["written"])
        self.assertEqual(self._read("bar.scale"), 1.3)
        self.assertEqual(self._read("bar.persistent"), False)
        # the replay recorded its OWN history entry, so bounded undo
        # reverts the replay itself
        entries = history.entries(self.target)
        self.assertEqual(entries[0]["label"], "macro: evening mode")
        history.undo(self.target, 1)
        self.assertEqual(self._read("bar.scale"), 1.0)

    def test_replay_of_present_values_is_an_honest_no_op(self) -> None:
        self._capture_evening()
        from assistant.capabilities.settings import planner
        plan = planner.plan(macros.macro_ops(self.target, "evening mode"),
                            self.target)
        # every entry is a no-op: the values are already in place
        self.assertTrue(all(e.get("no_op")
                            for e in plan["entries"]))
        result = apply(plan, self.target, write=True,
                       label="macro: evening mode")
        self.assertFalse(result["written"])
        self.assertEqual(result["no_changes"], True)

    def test_stale_tool_name_is_an_error_that_blocks_apply(self) -> None:
        self._capture_evening()
        # hand-edit the stored tool to one the registry no longer knows
        hpath = self.dir / "shell.json.assistant-history.json"
        doc = json.loads(hpath.read_text())
        doc["macros"][0]["ops"][0]["tool"] = "removedInAnUpdate"
        hpath.write_text(json.dumps(doc))
        from assistant.capabilities.settings import planner
        plan = planner.plan(macros.macro_ops(self.target, "evening mode"),
                            self.target)
        self.assertTrue(plan["apply_blocked"])
        self.assertIn("removedInAnUpdate",
                      str(plan["errors"]))
        with self.assertRaises(Exception):
            apply(plan, self.target, write=True)

    def test_unknown_macro_lists_saved_ones(self) -> None:
        with self.assertRaises(macros.MacroError) as caught:
            macros.macro_ops(self.target, "ghost")
        self.assertIn("unknown macro 'ghost'", str(caught.exception))
        self.assertIn("no macros saved yet", str(caught.exception))
        self._capture_evening()
        with self.assertRaises(macros.MacroError) as caught:
            macros.macro_ops(self.target, "ghost")
        self.assertIn("evening mode", str(caught.exception))

    def test_delete_is_explicit_and_refuses_unknown_names(self) -> None:
        self._capture_evening()
        with self.assertRaises(macros.MacroError):
            macros.delete(self.target, "ghost")
        result = macros.delete(self.target, "evening mode")
        self.assertEqual(result, {"deleted": "evening mode",
                                  "from_id": 1, "ops": 2})
        self.assertEqual(macros.list_macros(self.target), [])
        with self.assertRaises(macros.MacroError):
            macros.macro_ops(self.target, "evening mode")


class MacroCliTests(MacroTestCase):
    """The CLI surface: --macro / --macro-save / --from-id /
    --macro-list / --macro-delete, and the stronger consent rule."""

    def test_cli_capture_list_delete_roundtrip(self) -> None:
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        code, out, err = self._run_main(
            ["--macro-save", "evening mode", "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("captured macro 'evening mode'", out)
        self.assertIn("2 change(s) from history entry #1", out)

        code, out, err = self._run_main(
            ["--macro-list", "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("evening mode", out)
        self.assertIn("setBarScale=1.3", out)
        self.assertIn("re-approved", out)

        code, out, err = self._run_main(
            ["--macro-delete", "evening mode", "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("deleted macro 'evening mode'", out)
        self.assertEqual(macros.list_macros(self.target), [])

    def test_cli_replay_dry_run_writes_nothing(self) -> None:
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        macros.save(self.target, "evening mode")
        # move the values away so a replay WOULD change things
        self._apply(("bar.scale", 1.0), ("bar.persistent", True),
                    label="reset")
        before = self.target.read_bytes()
        code, out, err = self._run_main(
            ["--macro", "evening mode", "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("macro: evening mode", out)
        self.assertEqual(self.target.read_bytes(), before)

    def test_cli_replay_apply_without_confirm_refuses(self) -> None:
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        macros.save(self.target, "evening mode")
        self._apply(("bar.scale", 1.0), ("bar.persistent", True),
                    label="reset")
        before = self.target.read_bytes()
        # --apply alone is NOT enough: every replay needs the second
        # consent, single-change macros included
        code, out, err = self._run_main(
            ["--macro", "evening mode", "--apply",
             "--file", str(self.target)])
        self.assertEqual(code, 1, out)
        self.assertIn("macro replays always need confirmation", err)
        self.assertEqual(self.target.read_bytes(), before)

    def test_cli_replay_apply_with_confirm_writes(self) -> None:
        self._apply(("bar.scale", 1.3), ("bar.persistent", False),
                    label="evening")
        macros.save(self.target, "evening mode")
        self._apply(("bar.scale", 1.0), ("bar.persistent", True),
                    label="reset")
        code, out, err = self._run_main(
            ["--macro", "evening mode", "--apply", "--confirm",
             "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertEqual(self._read("bar.scale"), 1.3)
        self.assertEqual(self._read("bar.persistent"), False)
        # the replay's own history entry enables bounded undo
        entries = history.entries(self.target)
        self.assertEqual(entries[0]["label"], "macro: evening mode")

    def test_cli_single_change_macro_also_needs_confirm(self) -> None:
        # the deliberate strengthening: the plain >1-change rule does
        # NOT apply here — a one-op macro replay is refused without
        # --confirm too (a replay's contents may be stale in memory)
        self._apply(("bar.scale", 1.3), label="one change")
        macros.save(self.target, "one")
        self._apply(("bar.scale", 1.0), label="reset")
        before = self.target.read_bytes()
        code, _out, err = self._run_main(
            ["--macro", "one", "--apply", "--file", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("always need confirmation", err)
        self.assertEqual(self.target.read_bytes(), before)

    def test_cli_unknown_macro_exits_one_with_hint(self) -> None:
        self._apply(("bar.scale", 1.3), label="x")
        macros.save(self.target, "focus")
        code, _out, err = self._run_main(
            ["--macro", "ghost", "--file", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("unknown macro 'ghost'", err)
        self.assertIn("focus", err)

    def test_cli_capture_from_id(self) -> None:
        self._apply(("bar.scale", 1.2), label="one")
        self._apply(("bar.scale", 1.4), label="two")
        code, out, err = self._run_main(
            ["--macro-save", "first", "--from-id", "1",
             "--file", str(self.target)])
        self.assertEqual(code, 0, err)
        self.assertIn("history entry #1", out)
        self.assertEqual(
            macros.list_macros(self.target)[0]["ops"][0]["value"], 1.2)

    def test_cli_from_id_requires_macro_save(self) -> None:
        # argparse usage errors raise SystemExit(2) (the CLI's
        # documented exit-code contract)
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(["--from-id", "1", "--file", str(self.target)])
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_empty_history_capture_explains_the_contract(self) -> None:
        code, _out, err = self._run_main(
            ["--macro-save", "ghosted", "--file", str(self.target)])
        self.assertEqual(code, 1)
        self.assertIn("undo history is empty", err)
        self.assertIn("already approved", err)


if __name__ == "__main__":
    unittest.main()
