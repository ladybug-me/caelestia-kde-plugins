"""Planner tests for the settings layer (the settings design notes (see module docstrings) §4, §9).

The planner resolves parser ops against the CURRENT target file (fixtures in
temp dirs — never the real ~/.config/caelestia/shell.json):

- missing file -> registry defaults + honest "file will be created" note;
- absolute out-of-range -> a REJECTED entry that blocks the whole plan
  (#120: "it simply isn't applied"); relative ops clamp with a notice;
- no coercion ever: a string/bool where a number belongs is a
  TYPE_MISMATCH entry error, never a silent cast;
- an existing-but-invalid (or non-object) target aborts the run UNTOUCHED;
- multi-entry preset plans resolve entry-by-entry with absent-key notes;
- per-monitor override files are READ once to warn about shadowing, and
  global-only tools skip that check (the loader quarantines them anyway).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from assistant.capabilities.settings.parser import parse
from assistant.capabilities.settings.planner import PlannerError, plan


def _ops(sentence: str) -> List[Dict[str, Any]]:
    result = parse(sentence)
    assert result["verdict"] == "INTENT", sentence
    return result["ops"]


class PlannerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def seed(self, name: str = "shell.json", content: str = '{"bar": {"scale": 1.0}}') -> Path:
        target = self.tmp / name
        target.write_text(content, encoding="utf-8")
        return target


class MissingFileTests(PlannerTestCase):
    def test_missing_file_uses_registry_defaults_with_notes(self) -> None:
        target = self.tmp / "does-not-exist.json"
        result = plan(_ops("set the bar scale to 1.4"), target)
        self.assertEqual(result["verdict"], "INTENT")
        self.assertFalse(result["apply_blocked"])
        self.assertIn(
            "target file does not exist; defaults assumed (file will be created on apply)",
            result["notes"],
        )
        entry = result["entries"][0]
        self.assertEqual(entry["old"], 1.0)  # the registry default, not an error
        self.assertEqual(entry["new"], 1.4)
        self.assertIn("key was absent; default assumed", entry["note"])


class ValidationTests(PlannerTestCase):
    def test_absolute_out_of_range_is_a_rejected_entry(self) -> None:
        target = self.seed(content='{"bar": {"scale": 1.4}}')
        result = plan(_ops("set the bar scale to 9"), target)
        self.assertTrue(result["apply_blocked"])
        self.assertEqual(len(result["errors"]), 1)
        entry = result["entries"][0]
        self.assertIsNone(entry["new"])  # never applied
        self.assertIn("outside the allowed range 0.6-1.6", entry["error"])
        self.assertIn("not applied", entry["error"])

    def test_relative_from_near_max_clamps_with_notice(self) -> None:
        # Stage-3 battery line 5: 1.4 x 1.2 = 1.68 -> clamped to 1.6.
        target = self.seed(content='{"bar": {"scale": 1.4}}')
        result = plan(_ops("make the bar 20% bigger"), target)
        self.assertFalse(result["apply_blocked"])
        entry = result["entries"][0]
        self.assertEqual((entry["old"], entry["new"]), (1.4, 1.6))
        self.assertTrue(entry["clamped"])
        self.assertIn("clamped to 1.6", entry["note"])

    def test_relative_from_near_min_clamps_with_notice(self) -> None:
        # §9: fixture bar.scale 0.65, "bar 20% smaller" -> 0.6 + clamped note.
        target = self.seed(content='{"bar": {"scale": 0.65}}')
        result = plan(_ops("make the bar 20% smaller"), target)
        entry = result["entries"][0]
        self.assertEqual((entry["old"], entry["new"]), (0.65, 0.6))
        self.assertTrue(entry["clamped"])
        self.assertIn("clamped to 0.6", entry["note"])

    def test_string_existing_value_is_type_mismatch_never_coerced(self) -> None:
        target = self.seed(content='{"bar": {"scale": "big"}}')
        result = plan(_ops("set the bar scale to 1.0"), target)
        self.assertTrue(result["apply_blocked"])
        entry = result["entries"][0]
        self.assertIsNone(entry["new"])
        self.assertIn("TYPE_MISMATCH", entry["error"])
        self.assertIn("refusing to coerce", entry["error"])

    def test_bool_existing_value_is_type_mismatch(self) -> None:
        # §4.3: a JSON number is not an acceptable bool AND bool is not an
        # acceptable number (json parses true to Python bool == 1).
        target = self.seed(content='{"bar": {"scale": true}}')
        result = plan(_ops("set the bar scale to 1.0"), target)
        self.assertTrue(result["apply_blocked"])
        self.assertIn("TYPE_MISMATCH", result["entries"][0]["error"])

    def test_bool_magnitude_is_rejected_in_plan(self) -> None:
        # The parser passes "blur 50%" through; the planner REJECTS it
        # (§3.4/§3.5) and blocks the whole plan.
        target = self.seed()
        result = plan(parse("blur 50%")["ops"], target)
        self.assertTrue(result["apply_blocked"])
        entry = result["entries"][0]
        self.assertIsNone(entry["new"])
        self.assertEqual(entry["error"], "this setting is on/off only; it has no magnitude")

    def test_no_change_is_a_no_op_entry(self) -> None:
        target = self.seed(content='{"bar": {"scale": 1.0}}')
        result = plan(_ops("set the bar scale to 1.0"), target)
        self.assertFalse(result["apply_blocked"])
        entry = result["entries"][0]
        self.assertTrue(entry["no_op"])
        self.assertEqual(entry["old"], entry["new"])

    def test_int_tool_step_uses_registry_step(self) -> None:
        target = self.seed(content='{"bar": {"dock": {"iconSize": 33}}}')
        result = plan(_ops("make the dock icons smaller"), target)
        entry = result["entries"][0]
        self.assertEqual((entry["old"], entry["new"]), (33, 29))  # -4, §1.2
        self.assertIn("step smaller (-4)", entry["note"])


class AbortTests(PlannerTestCase):
    def test_invalid_json_aborts_with_file_untouched(self) -> None:
        target = self.seed(content="not json {")
        before = target.read_bytes()
        with self.assertRaises(PlannerError) as ctx:
            plan(_ops("set the bar scale to 1.4"), target)
        self.assertIn("not valid JSON", str(ctx.exception))
        self.assertIn("refusing to touch it", str(ctx.exception))
        self.assertEqual(target.read_bytes(), before)

    def test_non_object_json_aborts(self) -> None:
        target = self.seed(content="[1, 2]")
        with self.assertRaises(PlannerError) as ctx:
            plan(_ops("set the bar scale to 1.4"), target)
        self.assertIn("not an object", str(ctx.exception))


class MultiEntryPlanTests(PlannerTestCase):
    def test_compact_preset_resolves_three_entries(self) -> None:
        target = self.seed(content='{"bar": {"scale": 1.0}}')
        result = plan(parse("make everything feel more compact")["ops"], target)
        self.assertFalse(result["apply_blocked"])
        self.assertEqual(len(result["entries"]), 3)
        resolved = [(e["tool"], e["old"], e["new"]) for e in result["entries"]]
        self.assertEqual(
            resolved,
            [
                ("setBarScale", 1.0, 0.85),
                ("setSpacingScale", 1, 0.9),   # absent in the file -> default 1
                ("setPaddingScale", 1, 0.9),
            ],
        )
        for entry in result["entries"][1:]:
            self.assertIn("key was absent; default assumed", entry["note"])


class MonitorOverrideTests(PlannerTestCase):
    def test_shadow_warning_only_for_non_global_tools(self) -> None:
        # §4.6: a per-monitor override of bar.position shadows the global
        # change -> warning note; the same monitor file even contains
        # appearance.blur, but global-only tools skip the check (the loader
        # quarantines global-only keys in overlays anyway).
        monitor = self.tmp / "monitors" / "DP-1"
        monitor.mkdir(parents=True)
        (monitor / "shell.json").write_text(
            json.dumps({"bar": {"position": "left"}, "appearance": {"blur": False}}),
            encoding="utf-8",
        )
        position_target = self.seed(content='{"bar": {"position": "bottom"}}')
        result = plan(_ops("move the bar to the left"), position_target)
        self.assertEqual(
            result["notes"],
            ["screen 'DP-1' has a per-monitor override for this key (bar.position); "
             "the global change will be shadowed there"],
        )
        blur_target = self.seed()
        result = plan(parse("turn off the blur")["ops"], blur_target)
        self.assertEqual(result["notes"], [])


if __name__ == "__main__":
    unittest.main()
