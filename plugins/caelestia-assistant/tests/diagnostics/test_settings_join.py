"""A4 — the diagnostics -> settings reverse-lookup join.

Contract under test (settings_join.py):

- EXACT, VOLUNTEERED-ONLY join: a rule's settings tools are only those it
  references explicitly (a doc anchor naming assistant/settings/tools.json,
  or exact tool names / registry paths in fix/notes text). Noun-vocabulary
  similarity deliberately does NOT join — CL-kde-thumbs-001 mentions
  "thumbnails" (a setLivePreviews noun) but its root cause is desktop
  cache staleness; joining would suggest a wrong fix confidently.
- 100% RESOLUTION: every cross-reference the join emits resolves to a
  real tools.json id — enforced across EVERY rule in rules.d, not a sample.
- UNRESOLVED mentions are surfaced, not dropped: the current corpus has
  none (pinned), so a future rule naming a nonexistent tool fails loudly.
- The closed loop is visible in the report: a matched rule with tools
  renders the "Settings tools addressing this root cause" line.
"""

from __future__ import annotations

import unittest

from assistant.capabilities.diagnostics import engine
from assistant.capabilities.diagnostics.settings_join import (
    rules_for_tool,
    settings_tools_for_rule,
)
from assistant.adapters.caelestia.registry import tool_by_name


class JoinResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = engine.load_rules()

    def test_every_cross_reference_resolves_to_a_real_tools_json_id(self) -> None:
        # The A4 verification bar: 100% resolution over the whole ruleset.
        total = 0
        for rule in self.rules:
            joined = settings_tools_for_rule(rule)
            for entry in joined["tools"]:
                spec = tool_by_name(entry["tool"])
                self.assertIsNotNone(spec, msg=rule["id"])
                self.assertEqual(spec.path, entry["path"])
                total += 1
        self.assertGreaterEqual(total, 1)  # the join is not vacuously green

    def test_no_unresolved_mentions_in_current_corpus(self) -> None:
        for rule in self.rules:
            joined = settings_tools_for_rule(rule)
            self.assertEqual(joined["unresolved"], [],
                             msg=f"{rule['id']}: {joined['unresolved']}")

    def test_deterministic_pure_join(self) -> None:
        for rule in self.rules:
            self.assertEqual(settings_tools_for_rule(rule),
                             settings_tools_for_rule(rule),
                             msg=rule["id"])

    def test_idle_throttle_rule_joins_both_tools(self) -> None:
        rule = next(r for r in self.rules if r["id"] == "CL-idle-throttle-001")
        joined = settings_tools_for_rule(rule)
        self.assertEqual([e["tool"] for e in joined["tools"]],
                         ["setAnimationSpeed", "setBlurEnabled"])
        self.assertEqual([e["path"] for e in joined["tools"]],
                         ["appearance.anim.durations.scale",
                          "appearance.blur"])

    def test_cache_rule_does_not_join_on_noun_similarity(self) -> None:
        # "Bar/taskbar thumbnails ... broken" mentions a setLivePreviews
        # NOUN, but the join must stay empty: the root cause is cache
        # staleness, not a settings value.
        rule = next(r for r in self.rules if r["id"] == "CL-kde-thumbs-001")
        self.assertEqual(settings_tools_for_rule(rule)["tools"], [])

    def test_rules_without_settings_surface_are_empty(self) -> None:
        rule = next(r for r in self.rules if r["id"] == "CL-kde-bluez-001")
        self.assertEqual(settings_tools_for_rule(rule)["tools"], [])

    def test_reverse_direction(self) -> None:
        self.assertIn("CL-idle-throttle-001",
                      rules_for_tool("setBlurEnabled", self.rules))
        self.assertEqual(rules_for_tool("setBlurEnabled", self.rules),
                         rules_for_tool("setBlurEnabled", self.rules))
        self.assertEqual(rules_for_tool("setNoSuchTool", self.rules), [])


class JoinWiringTests(unittest.TestCase):
    def test_diagnose_candidates_carry_the_join(self) -> None:
        text = ("battery drains while the machine sits idle "
                "throttle blur and animation")
        diagnosis = engine.diagnose(text)
        self.assertIn(diagnosis["verdict"], ("MATCH", "AMBIGUOUS"))
        found = False
        for candidate in diagnosis["candidates"]:
            if candidate["rule"]["id"] == "CL-idle-throttle-001":
                tools = [e["tool"] for e in candidate["settings_tools"]]
                self.assertIn("setBlurEnabled", tools)
                found = True
        self.assertTrue(found)

    def test_report_renders_the_settings_tools_line(self) -> None:
        text = ("battery drains while the machine sits idle "
                "throttle blur and animation")
        report = engine.render_report(engine.diagnose(text))
        self.assertIn("Settings tools addressing this root cause", report)
        self.assertIn("setBlurEnabled", report)

    def test_report_without_tools_renders_no_line(self) -> None:
        text = "bluetooth adapter turns on then immediately off nothing pairs"
        report = engine.render_report(engine.diagnose(text))
        self.assertNotIn("Settings tools addressing this root cause", report)

    def test_join_survives_a_fabricated_unresolved_mention(self) -> None:
        # A future rule naming a nonexistent tool: the join reports it in
        # `unresolved` and never emits it as a cross-reference.
        rule = {"id": "CL-test-fabricated-999", "fix": [
            {"text": "use setNoSuchTool to fix this"}],
            "references": []}
        joined = settings_tools_for_rule(rule)
        self.assertEqual(joined["tools"], [])
        self.assertEqual(joined["unresolved"], ["setNoSuchTool"])


if __name__ == "__main__":
    unittest.main()
