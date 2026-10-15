"""CLI direct-tool-addressing tests for the settings layer.

Covers the cli.py direct-addressing surface on top of the untouched
natural-language pipeline:

- --call NAME=VALUE: VALUE is parsed as JSON (true/false, numbers, quoted
  strings; a bare unquoted token that fails JSON parsing is a plain
  string), composed as one {"tool", "action": "set", "value", "raw"} op
  and routed through the SAME planner/render/--apply/--restore path as
  NL requests — dry-run writes nothing, --apply writes only the target +
  its .assistant-backup sibling, --restore undoes byte-identically;
- error semantics for direct calls: out-of-range values, invalid enum
  values, unknown tool names, non-finite numbers (NaN/Infinity pass
  json.loads but can never be written) and string-valued tools are
  REJECTED entries that exit 1 EVEN IN DRY-RUN (a direct invocation with
  a bad name or value is a caller error, not an honest NL verdict), and
  --apply on such a plan is refused wholesale (all-or-nothing, §4.5);
- string-valued tools are rejected by a CLI pre-check BEFORE the planner
  is asked to resolve the op (the planner only validates bool/enum/
  number values);
- multiple --call flags compose one multi-op plan;
- --call and the positional request sentence are mutually exclusive
  (argparse usage error, exit 2);
- --tool NAME prints a detail card (exit 1 when unknown);
- --list-tools --group SLUG filters without crashing.

The tests are written to pass BOTH with the 18-tool core registry and with
the full generated registry: no tool-count pins anywhere; only the 18-tool core
tool names/paths/ranges are asserted, which the 2-pre contract guarantees
keep their exact values. cli.main() is called IN-PROCESS (never via
subprocess — the assistant itself must stay executor-free); all writes go
to temp dirs via --file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Tuple
from unittest import mock

from assistant.capabilities.settings import cli

from assistant.adapters.caelestia import registry
from assistant.adapters.caelestia.registry import ToolSpec

ORIGINAL = (
    "{\n"
    '    "bar": {"scale": 1.0, "persistent": true, "position": "bottom"},\n'
    '    "unrelated": {"keep": [1, 2]}\n'
    "}\n"
)


class _Recorder:
    """Duck-typed stdout/stderr sink."""

    def __init__(self) -> None:
        self.chunks: List[str] = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self.chunks)


def _fake_string_spec() -> Any:
    """A spec with kind="string" for the CLI pre-check test. The original ToolSpec
    has no group/citations fields; the current one adds them — build the
    real dataclass when possible and fall back to a duck-typed stand-in so
    the test works under either registry."""
    try:
        return ToolSpec(
            name="setFakeString", path="general.fakeString",
            kind="string", default="a",
        )
    except TypeError:  # current ToolSpec requires group/citations
        class _DuckStringSpec:
            name = "setFakeString"
            path = "general.fakeString"
            kind = "string"

        return _DuckStringSpec()


class CallCliTestCase(unittest.TestCase):
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

    def target(self, content: str = ORIGINAL) -> Path:
        target = self.dir / "shell.json"
        target.write_text(content, encoding="utf-8")
        return target

    def names(self) -> List[str]:
        return sorted(path.name for path in self.dir.iterdir())


class CallBoolToolTests(CallCliTestCase):
    def test_call_bool_dry_run_renders_plan_and_writes_nothing(self) -> None:
        target = self.target()
        code, out, err = self._run_main(
            ["--call", "setBarPersistent=false", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("this run writes nothing", out)
        self.assertIn("bar.persistent: true -> false", out)
        self.assertIn("setBarPersistent", out)
        # §5.2: the re-run command includes the --call flag verbatim.
        self.assertIn("--call", out)
        self.assertIn("--apply", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])

    def test_call_bool_apply_writes_json_path_and_restore_round_trips(self) -> None:
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarPersistent=false", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("applied on", out)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertIs(data["bar"]["persistent"], False)  # JSON path set correctly
        self.assertEqual(data["unrelated"], {"keep": [1, 2]})  # merge preserves
        self.assertEqual(self.names(), ["shell.json", "shell.json.assistant-backup",
                                   "shell.json.assistant-history.json"])
        self.assertEqual(
            (self.dir / "shell.json.assistant-backup").read_text(encoding="utf-8"),
            ORIGINAL,
        )
        # --restore: byte-identical undo, then the slot is consumed.
        code, _, _ = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        code, _, err = self._run_main(["--restore", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("no backup to restore", err)

    def test_call_bool_json_true_against_true_is_an_honest_no_op(self) -> None:
        # VALUE "true" parses as JSON bool True; against an existing true it
        # is a no-op that renders as such and writes nothing, even with
        # --apply (no pointless backup churn, §5.3 step 1).
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarPersistent=true", "--apply", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("[no change needed]", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])

    def test_call_number_on_bool_tool_is_rejected_not_coerced(self) -> None:
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarPersistent=3", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("on/off only", out)
        self.assertIn("REJECTED", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])


class CallValidationTests(CallCliTestCase):
    def test_call_numeric_out_of_range_is_rejected_and_exits_one(self) -> None:
        # Dry-run: the REJECTED entry renders (with the allowed range) and
        # the run exits 1 — direct calls report caller errors hard.
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarScale=9", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("REJECTED", out)
        self.assertIn("outside the allowed range 0.6-1.6", out)
        self.assertIn("apply_blocked", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])

    def test_call_numeric_out_of_range_apply_is_refused(self) -> None:
        target = self.target()
        code, _, err = self._run_main(
            ["--call", "setBarScale=9", "--apply", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("apply refused", err)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])

    def test_call_enum_invalid_value_is_rejected(self) -> None:
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarPosition=diagonal", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("REJECTED", out)
        self.assertIn("is not one of top|bottom|left|right", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_call_enum_accepts_bare_and_json_quoted_tokens(self) -> None:
        # A bare unquoted token that fails JSON parsing is a plain string:
        # both setBarPosition=top and setBarPosition="top" resolve to "top".
        for value in ("top", '"top"'):
            target = self.target()
            code, out, _ = self._run_main(
                ["--call", f"setBarPosition={value}", "--file", str(target)])
            self.assertEqual(code, 0)
            self.assertIn("bar.position: bottom -> top", out)
            self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_call_non_finite_value_is_rejected(self) -> None:
        # json.loads accepts NaN/Infinity, but the planner's range checks
        # cannot reject them (every NaN comparison is False) and NaN can
        # never be serialized as JSON — the CLI stops them as error entries.
        for value in ("NaN", "Infinity"):
            target = self.target()
            code, out, _ = self._run_main(
                ["--call", f"setBarScale={value}", "--file", str(target)])
            self.assertEqual(code, 1, msg=value)
            self.assertIn("not a finite number", out, msg=value)
            self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_call_unknown_tool_renders_error_entry_and_exits_one(self) -> None:
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBogusTool=1", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn("unknown tool", out)
        self.assertIn("setBogusTool", out)
        self.assertIn("REJECTED", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])

    def test_call_json_mode_emits_machine_readable_error_payload(self) -> None:
        target = self.target()
        code, out, _ = self._run_main(
            ["--call", "setBarScale=9", "--json", "--file", str(target)])
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertTrue(payload["apply_blocked"])
        self.assertIn("outside the allowed range", payload["entries"][0]["error"])

    def test_call_on_missing_target_uses_defaults_and_writes_nothing(self) -> None:
        target = self.dir / "absent.json"
        code, out, _ = self._run_main(
            ["--call", "setBarScale=1.2", "--file", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("bar.scale: 1.0 -> 1.2", out)  # registry default as old
        self.assertIn("does not exist", out)  # honest missing-file note
        self.assertEqual(self.names(), [])  # nothing at all was created


class CallMultiTests(CallCliTestCase):
    def test_multiple_calls_compose_one_multi_op_plan(self) -> None:
        target = self.target()
        code, out, _ = self._run_main([
            "--call", "setBarScale=1.2",
            "--call", "setDockIconSize=48",
            "--call", "setBarPosition=top",
            "--file", str(target),
        ])
        self.assertEqual(code, 0)
        self.assertIn("3 change(s) planned", out)
        self.assertIn("bar.scale: 1.0 -> 1.2", out)
        self.assertIn("bar.dock.iconSize: 32 -> 48", out)
        self.assertIn("bar.position: bottom -> top", out)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_multiple_calls_apply_writes_every_entry(self) -> None:
        target = self.target()
        code, _, _ = self._run_main([
            "--call", "setBarScale=1.2",
            "--call", "setDockIconSize=48",
            "--apply", "--confirm",  # multi-change: issue #120's gate
            "--file", str(target),
        ])
        self.assertEqual(code, 0)
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["bar"]["scale"], 1.2)
        self.assertEqual(data["bar"]["dock"]["iconSize"], 48)  # path created
        self.assertEqual(data["unrelated"], {"keep": [1, 2]})
        self.assertEqual(
            self.names(), ["shell.json", "shell.json.assistant-backup",
                           "shell.json.assistant-history.json"])

    def test_multi_call_plan_is_all_or_nothing_when_one_entry_is_rejected(self) -> None:
        # 400 is outside the dock icon stepper range 16-96: the whole plan
        # is blocked and --apply writes nothing (#120: "it simply isn't
        # applied" — for every entry, §4.5).
        target = self.target()
        code, _, err = self._run_main([
            "--call", "setBarScale=1.2",
            "--call", "setDockIconSize=400",
            "--apply", "--file", str(target),
        ])
        self.assertEqual(code, 1)
        self.assertIn("apply refused", err)
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.names(), ["shell.json"])


class CallUsageTests(CallCliTestCase):
    def test_call_with_positional_sentence_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run_main([
                "--call", "setBarScale=1.2",
                "make the bar thinner",
                "--file", str(self.target()),
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_call_without_equals_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(["--call", "setBarScale", "--file", str(self.target())])
        self.assertEqual(ctx.exception.code, 2)

    def test_call_with_empty_name_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(["--call", "=1.2", "--file", str(self.target())])
        self.assertEqual(ctx.exception.code, 2)

    def test_group_without_list_tools_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(["--group", "bar"])
        self.assertEqual(ctx.exception.code, 2)


class StringToolPrecheckTests(CallCliTestCase):
    """kind="string" tools are rejected by the CLI BEFORE the planner is
    asked to resolve the op (the planner only knows bool/enum/number)."""

    def test_string_kind_tool_is_rejected_before_the_planner(self) -> None:
        target = self.target()
        spec = _fake_string_spec()
        minimal_plan = {
            "verdict": "INTENT", "file": str(target), "entries": [],
            "notes": [], "errors": [], "apply_blocked": False,
        }
        with mock.patch.object(cli, "tool_by_name", return_value=spec), \
                mock.patch.object(cli.planner, "plan", return_value=minimal_plan) as plan_mock:
            code, out, _ = self._run_main(
                ["--call", "setFakeString=hello", "--file", str(target)])
        self.assertEqual(code, 1)
        self.assertIn(
            "string-valued tools are not settable through --call yet", out)
        self.assertIn("REJECTED", out)
        self.assertIn("apply_blocked", out)
        # The planner was called with ZERO ops for this run: the op never
        # reached it (the pre-check synthesized the error entry instead).
        self.assertEqual(plan_mock.call_args[0][0], [])
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)

    def test_string_kind_rejection_composes_with_a_valid_call(self) -> None:
        # A rejected string call and a valid bool call in one run: the
        # valid entry still resolves through the real planner, the string
        # entry stays an error, and the whole plan exits 1 / is blocked.
        target = self.target()
        spec = _fake_string_spec()
        with mock.patch.object(cli, "tool_by_name",
                               side_effect=lambda name: spec if name == spec.name
                               else registry.tool_by_name(name)):
            code, out, _ = self._run_main([
                "--call", f"{spec.name}=hello",
                "--call", "setBarPersistent=false",
                "--file", str(target),
            ])
        self.assertEqual(code, 1)
        self.assertIn("string-valued tools are not settable", out)
        self.assertIn("bar.persistent: true -> false", out)  # resolved anyway
        self.assertEqual(target.read_text(encoding="utf-8"), ORIGINAL)


class ToolCardTests(CallCliTestCase):
    def test_tool_detail_card_for_a_known_tool(self) -> None:
        code, out, _ = self._run_main(["--tool", "setBarScale"])
        self.assertEqual(code, 0)
        self.assertIn("setBarScale", out)
        self.assertIn("bar.scale", out)
        self.assertIn("citations", out)  # the citation section marker
        # guaranteed facts (the frozen core's own tests keep them unchanged).
        self.assertIn("float", out)
        self.assertIn("0.6-1.6", out)
        self.assertIn("bar|taskbar|panel", out)

    def test_tool_detail_card_writes_nothing(self) -> None:
        self._run_main(["--tool", "setBarScale"])
        self.assertEqual(self.names(), [])

    def test_tool_detail_unknown_tool_exits_one(self) -> None:
        code, out, err = self._run_main(["--tool", "setBogusTool"])
        self.assertEqual(code, 1)
        self.assertIn("unknown tool", err)
        self.assertEqual(self.names(), [])


class ListToolsGroupTests(CallCliTestCase):
    """--list-tools --group must not crash under EITHER registry: with the
    18-tool core (no group metadata) it degrades to the full table; with
    the full registry it filters. Only core tool names are asserted —
    never counts, never group layout."""

    def test_list_tools_unfiltered_still_lists_core_tools(self) -> None:
        code, out, _ = self._run_main(["--list-tools"])
        self.assertEqual(code, 0)
        self.assertIn("setBarScale", out)
        self.assertIn("setDockBadges", out)
        self.assertTrue(any(line.strip() for line in out.splitlines()))

    def test_list_tools_group_bar_does_not_crash_and_keeps_bar_tools(self) -> None:
        code, out, err = self._run_main(["--list-tools", "--group", "bar"])
        self.assertEqual(code, 0)
        self.assertTrue(out.strip(), "at least one line must be printed")
        self.assertIn("setBarScale", out)

    def test_list_tools_unknown_group_is_a_clean_error_when_groups_exist(self) -> None:
        if not hasattr(registry, "GROUPS"):
            self.skipTest(
                "registry predates group metadata; "
                "the pre-group-metadata bridge degrades --group to the full table"
            )
        code, out, err = self._run_main(
            ["--list-tools", "--group", "definitely-not-a-real-group"])
        self.assertEqual(code, 1)
        self.assertIn("unknown group", err)
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
