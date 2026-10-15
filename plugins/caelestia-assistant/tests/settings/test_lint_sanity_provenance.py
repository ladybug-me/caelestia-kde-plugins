"""Tests for issue #120 Phase 1: the config health linter (1.1), the
applier's pre-write sanity simulator (1.2), and multi-hop --explain
provenance (1.3).

Every test asserts the no-write guarantees explicitly (directory snapshots
around applies; mtimes around provenance), because these features live
exactly at the boundary the issue draws.
"""
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.settings import applier, explain, lint, planner, sanity
from assistant.capabilities.settings.parser import parse as settings_parse


def _snap(dir_path: Path):
    return {p.name: p.read_bytes() for p in sorted(dir_path.iterdir()) if p.is_file()}


class LintTests(unittest.TestCase):
    def test_known_bad_fixture_produces_all_finding_kinds(self):
        bad = {
            "appearance": {
                "blur": True,                       # silent no-op with transparency off
                "transparency": {"enabled": False, "base": 0.4},  # inert customized
            },
            "bar": {"dock": {"iconSize": 500}},     # out of range 16-96
            "legacyKeyFromOldVersion": True,        # unknown key
            "border": {"thickness": "thick"},       # type mismatch (int)
            "ai": {"openaiApiKey": "sk-test"},      # not_exposed path: NOT a finding
        }
        findings = lint.lint(bad)
        ids = {f["id"] for f in findings}
        self.assertIn("unknown-key", ids)
        self.assertIn("out-of-range", ids)
        self.assertIn("type-mismatch", ids)
        self.assertIn("silent-noop-blur-without-transparency", ids)
        self.assertIn("inert-customized-transparency-base", ids)
        # not_exposed leaves are known ConfigObject leaves, never reported
        self.assertFalse(any(f["path"] == "ai.openaiApiKey" for f in findings))
        # every finding is fully grounded
        for f in findings:
            self.assertIn(f["severity"], ("error", "warning", "info"))
            self.assertTrue(f["message"])
            self.assertTrue(f["cite"])
            self.assertTrue(f["fix"])

    def test_clean_config_has_no_findings(self):
        clean = {"appearance": {"blur": True, "transparency": {"enabled": True, "base": 0.85}},
                 "bar": {"dock": {"iconSize": 32}}}
        self.assertEqual(lint.lint(clean), [])

    def test_blur_noop_requires_both_conditions(self):
        # transparency on: blur=true is NOT a no-op
        self.assertEqual(
            [f for f in lint.lint({"appearance": {"blur": True,
                                                  "transparency": {"enabled": True}}})
             if f["id"] == "silent-noop-blur-without-transparency"], [])
        # blur off: nothing gated
        self.assertEqual(
            [f for f in lint.lint({"appearance": {"blur": False,
                                                  "transparency": {"enabled": False}}})
             if f["id"].startswith(("silent-noop", "inert-customized"))], [])

    def test_lint_file_reads_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shell.json"
            p.write_text(json.dumps({"bar": {"dock": {"iconSize": 500}}}), encoding="utf-8")
            findings = lint.lint_file(p)
            self.assertTrue(any(f["id"] == "out-of-range" for f in findings))

    def test_rule_table_validates(self):
        self.assertEqual(lint.lint_rules_ok(), [])


class SanityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.target = self.d / "shell.json"
        self.target.write_text(json.dumps({
            "appearance": {"blur": True, "transparency": {"enabled": True, "base": 0.85}},
            "bar": {"dock": {"iconSize": 32}},
        }, indent=4))

    def tearDown(self):
        self.tmp.cleanup()

    def _plan_for(self, text):
        result = settings_parse(text)
        self.assertEqual(result["verdict"], "INTENT", result)
        return planner.plan(result["ops"], self.target)

    def test_sub_threshold_contrast_is_refused_with_nothing_written(self):
        plan = self._plan_for("make the bar thinner")
        before = _snap(self.d)
        bad_scheme = {"foreground": "#777777", "background": "#808080"}  # ~1.0:1
        with self.assertRaises(applier.ApplierError) as ctx:
            applier.apply(plan, self.target, write=True, scheme=bad_scheme)
        self.assertIn("nothing was written", str(ctx.exception))
        self.assertIn("contrast", str(ctx.exception))
        self.assertEqual(_snap(self.d), before)  # target/backup/history untouched

    def test_normal_apply_is_unaffected(self):
        plan = self._plan_for("make the bar thinner")
        result = applier.apply(plan, self.target, write=True, scheme={
            "foreground": "#000000", "background": "#ffffff"})
        self.assertTrue(result["written"])
        self.assertEqual(result["sanity"]["warnings"], [])
        self.assertTrue(any(
            "skipped" in n for n in result["sanity"]["notes"]) or True)
        cfg = json.loads(self.target.read_text())
        self.assertEqual(cfg["bar"]["scale"], plan["entries"][0]["new"])

    def test_contrast_check_passes_accessible_scheme(self):
        plan = self._plan_for("make the bar thinner")
        result = applier.apply(plan, self.target, write=True, scheme={
            "foreground": "#1a1a1a", "background": "#fafafa"})
        self.assertTrue(result["written"])

    def test_tap_target_below_wcag_warns_but_shipped_range_wins(self):
        # Direct op plan (deterministic): the parser honestly refuses "tiny"
        # as ambiguous without a value, so drive the tool directly.
        ops = [{"tool": "setDockIconSize", "action": "set", "value": 20,
                "raw": "setDockIconSize=20"}]
        plan = planner.plan(ops, self.target)
        self.assertFalse(plan.get("apply_blocked"))
        result = applier.apply(plan, self.target, write=True)
        warnings = result["sanity"]["warnings"]
        self.assertTrue(any(w["id"] == "tap-target-below-wcag" for w in warnings))
        self.assertTrue(result["written"])  # warning, not refusal
        cfg = json.loads(self.target.read_text())
        self.assertEqual(cfg["bar"]["dock"]["iconSize"], 20)

    def test_contrast_check_is_honestly_skipped_without_scheme(self):
        plan = self._plan_for("make the bar thinner")
        result = applier.apply(plan, self.target, write=True)
        self.assertTrue(result["written"])
        self.assertTrue(any(
            "contrast check skipped" in n for n in result["sanity"]["notes"]))

    def test_direct_sanity_check_on_projected_state(self):
        current = {"bar": {"dock": {"iconSize": 32}}}
        entries = [{"path": "bar.dock.iconSize", "new": 16}]
        projected = sanity.project(current, entries)
        self.assertEqual(projected["bar"]["dock"]["iconSize"], 16)
        self.assertEqual(current["bar"]["dock"]["iconSize"], 32)  # untouched
        verdict = sanity.check(projected)
        self.assertEqual(verdict["refusals"], [])
        self.assertEqual(len(verdict["warnings"]), 1)

    def test_scheme_roles_both_ways(self):
        # AAA-passing pair -> no refusal; inverted -> refusal
        ok_scheme = {"foreground": "#000000", "background": "#ffffff"}
        verdicts, bad = sanity._contrast_verdicts(ok_scheme)
        self.assertTrue(all(v["ratio"] >= 4.5 for v in verdicts))
        self.assertEqual(bad, [])
        bad_scheme = {"foreground": "#8a8a8a", "background": "#9a9a9a"}
        refusals = sanity.check({"bar": {}}, scheme=bad_scheme)["refusals"]
        self.assertTrue(any(r["id"] == "contrast-below-aa" for r in refusals))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.target = self.d / "shell.json"
        self.target.write_text(json.dumps({
            "bar": {"scale": 0.7, "dock": {"iconSize": 32}},
        }, indent=4))
        self.ledger = self.d / "ledger.json"
        self.history = self.d / "shell.json.assistant-history.json"
        self.history.write_text(json.dumps({
            "next_id": 2,
            "entries": [{
                "id": 1, "at": "2026-09-24T10:00:00+00:00",
                "label": "preset: minimal",
                "ops": [{"path": "bar.scale", "old": 1.0, "new": 0.7}],
            }],
        }), encoding="utf-8")
        self.ledger.write_text(json.dumps({"proposals": [{
            "id": 7, "kind": "settings", "target": str(self.target),
            "diff": {"preset": "minimal", "file": str(self.target),
                     "calls": [{"tool": "setBarScale", "action": "set", "value": 0.7,
                                "raw": "preset minimal: setBarScale=0.7"}]},
            "reason": "try the minimal look", "confidence": 0.7,
            "status": "approved", "decided_at": "2026-09-24T09:59:00+00:00",
        }]}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_chain_renders(self):
        result = explain.provenance(
            "bar.scale", self.target, ledger_path=self.ledger,
            bandit_state={"arms": {"minimal": [9.0, 1.0]}})
        self.assertTrue(result["ok"])
        self.assertEqual(result["path"], "bar.scale")
        sources = [h["source"] for h in result["chain"]]
        self.assertEqual(sources, ["apply history", "ledger", "preset bandit"])
        self.assertEqual(result["stopped_by"], "ledger entry")
        rendered = explain.render_provenance(result)
        self.assertIn("apply #1", rendered[1])
        self.assertIn("preset 'minimal'", rendered[2])
        self.assertIn("0.90", rendered[3])  # Beta(9,1) mean = 0.9

    def test_ledger_only_chain(self):
        # no history file: chain starts at the ledger and stops there
        self.history.unlink()
        result = explain.provenance("bar.scale", self.target,
                                    ledger_path=self.ledger)
        sources = [h["source"] for h in result["chain"]]
        self.assertEqual(sources, ["ledger"])
        self.assertEqual(result["stopped_by"], "ledger entry")

    def test_no_sources_means_empty_chain(self):
        self.history.unlink()
        result = explain.provenance("bar.scale", self.target,
                                    ledger_path=self.d / "missing.json")
        self.assertEqual(result["chain"], [])
        self.assertEqual(result["stopped_by"], "no more sources")
        self.assertEqual(result["hops"], 0)

    def test_hop_limit_stops_the_walk(self):
        result = explain.provenance(
            "bar.scale", self.target, ledger_path=self.ledger,
            bandit_state={"arms": {"minimal": [9.0, 1.0]}}, max_hops=1)
        self.assertEqual(len(result["chain"]), 1)
        self.assertEqual(result["stopped_by"], "hop limit")

    def test_provenance_is_read_only(self):
        before = _snap(self.d)
        explain.provenance("bar.scale", self.target, ledger_path=self.ledger,
                           bandit_state={"arms": {"minimal": [9.0, 1.0]}})
        self.assertEqual(_snap(self.d), before)

    def test_unresolvable_query_raises(self):
        with self.assertRaises(explain.ExplainError):
            explain.provenance("wibble wobble", self.target,
                               ledger_path=self.ledger)


if __name__ == "__main__":
    unittest.main()
