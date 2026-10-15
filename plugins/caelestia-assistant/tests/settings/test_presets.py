"""Preset bundle tests (issue #120).

Pins the rule this module sets: presets are expressed PURELY as
bundles of the same validated tool calls — never bespoke unvalidated code
paths. Every test here would break if a preset ever bypassed the planner,
the validation ranges or the confirmation gate.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from assistant.capabilities.settings import cli
from assistant.capabilities.settings.presets import (
    PresetError,
    describe_lines,
    preset_ops,
    presets,
)


class _Recorder:
    """Minimal stdout/stderr capture without new imports (the policy's
    answer to redirect_stdout — same trick test_cli_calls.py uses)."""

    def __init__(self) -> None:
        self._parts: list = []

    def write(self, text: str) -> None:
        self._parts.append(text)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self._parts)


EXPECTED_PRESETS = ("compact", "minimal", "gaming", "battery-saver", "macos-like")


class PresetDataTests(unittest.TestCase):
    def test_the_five_named_presets_exist(self) -> None:
        self.assertEqual(tuple(p["name"] for p in presets()), EXPECTED_PRESETS)

    def test_every_call_is_a_real_tool(self) -> None:
        from assistant.adapters.caelestia.registry import tool_by_name
        for preset in presets():
            for tool, value in preset["calls"]:
                spec = tool_by_name(str(tool))
                self.assertIsNotNone(spec, f"{preset['name']}: {tool}")
                assert spec is not None
                # the value must pass that tool's own validation
                if spec.kind == "bool":
                    self.assertIsInstance(value, bool)
                elif spec.kind == "enum":
                    self.assertIn(value, spec.enum or ())
                elif spec.kind == "string":
                    self.assertIsInstance(value, str)
                else:
                    self.assertIsInstance(value, (int, float))
                    self.assertGreaterEqual(value, spec.minimum)
                    self.assertLessEqual(value, spec.maximum)

    def test_every_preset_is_multi_change(self) -> None:
        # the issue's confirmation flow is for multi-setting requests;
        # presets are multi-op by construction.
        for preset in presets():
            self.assertGreater(len(preset["calls"]), 1)

    def test_preset_ops_carry_provenance(self) -> None:
        ops = preset_ops("minimal")
        self.assertTrue(all(op["raw"].startswith("preset minimal:") for op in ops))
        self.assertTrue(all(op["action"] == "set" for op in ops))

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(PresetError):
            preset_ops("not-a-preset")

    def test_describe_lines_renders_all(self) -> None:
        text = "\n".join(describe_lines())
        for name in EXPECTED_PRESETS:
            self.assertIn(name, text)


class PresetFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "shell.json"
        self.target.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, argv) -> Tuple[int, str, str]:
        out, err = _Recorder(), _Recorder()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli.main(argv + ["--file", str(self.target)])
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        return code, out.text(), err.text()

    def test_preset_dry_run_writes_nothing(self) -> None:
        code, out, _ = self._run(["--preset", "compact"])
        self.assertEqual(code, 0)
        self.assertIn("INTENT", out)
        self.assertEqual(self.target.read_text(), "{}")
        self.assertEqual(list(self.dir.iterdir()), [self.target])

    def test_preset_apply_requires_confirmation(self) -> None:
        code, _, err = self._run(["--preset", "compact", "--apply"])
        self.assertEqual(code, 1)
        self.assertIn("confirmation", err)
        self.assertEqual(self.target.read_text(), "{}")

    def test_preset_apply_with_confirm_writes_and_records_history(self) -> None:
        code, out, _ = self._run(["--preset", "compact", "--apply", "--confirm"])
        self.assertEqual(code, 0)
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["dock"]["iconSize"], 24)
        self.assertEqual(data["appearance"]["spacing"]["scale"], 0.8)
        history = json.loads(
            (self.dir / "shell.json.assistant-history.json").read_text())
        self.assertEqual(history["entries"][0]["label"], "preset: compact")

    def test_preset_undo_round_trip(self) -> None:
        self._run(["--preset", "minimal", "--apply", "--confirm"])
        code, _, _ = self._run(["--undo"])
        self.assertEqual(code, 0)
        # every key the preset created is gone (the file may keep the
        # applier's trailing newline; emptiness is what matters).
        self.assertEqual(json.loads(self.target.read_text()), {})

    def test_multi_call_requires_confirm_single_call_does_not(self) -> None:
        code, _, _ = self._run(["--call", "setBarScale=1.2",
                                "--call", "setBarPosition=top", "--apply"])
        self.assertEqual(code, 1)
        self.assertEqual(self.target.read_text(), "{}")
        code, _, _ = self._run(["--call", "setBarScale=1.2", "--apply"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.target.read_text())["bar"]["scale"], 1.2)


if __name__ == "__main__":
    unittest.main()
