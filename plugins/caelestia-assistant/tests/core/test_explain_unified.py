"""Tests for the unified why explainer (cortex/explain_unified.py).

The contract under test — ONE structured shape ({engine, headline,
lines, citations, confidence}) for every engine, with each engine's OWN
explanation output reused verbatim (templating, never synthesizing):

- diagnostics: the matched rule's own id/title/fix/references;
- cortex: dispatch.render_answer's own chat-card lines plus the
  conformal calibrator's own reason/guarantee sentence;
- settings: settings/explain.py's own answer string and cites;
- brain: calibrate's own Beta-Binomial posteriors over the ledger;
- wizard: settings/wizard.py's own render() lines;
- walk-back: the newest ledger proposal's kind names its engine and the
  item's own reason rides along; `why <id>` reuses the inbox id space.

One case per source engine, plus the walk-back and CLI surfaces.
Read-only everywhere: no test writes outside tmp fixtures.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain.ledger import Ledger
from assistant.core.explain_unified import (
    explain_brain,
    explain_cortex,
    explain_diagnostics,
    explain_inbox_id,
    explain_last,
    explain_ledger_item,
    explain_settings,
    explain_wizard,
    last_ledger_item,
    main,
    render,
)
from assistant.core.inbox import Inbox
from assistant.core.plans import PlanCache
from assistant.capabilities.settings import wizard as settings_wizard


class DiagnosticsEngineTests(unittest.TestCase):
    def test_matched_rule_explains_with_its_own_words(self) -> None:
        result = explain_diagnostics(
            "vesktop freezes when screen sharing turns on")
        self.assertEqual(result["engine"], "diagnostics")
        self.assertIn("MATCH", result["headline"])
        self.assertIn("CL-runtime-shell-vesktop-001", result["headline"])
        # the fix lines are the rule's own strings, verbatim
        self.assertTrue(result["lines"])
        self.assertIsInstance(result["citations"], list)
        self.assertIsNotNone(result["confidence"])

    def test_no_match_stays_honest(self) -> None:
        result = explain_diagnostics("the moon is made of gouda")
        self.assertEqual(result["engine"], "diagnostics")
        self.assertEqual(result["headline"], "NO_MATCH")
        self.assertIn("don't know", result["lines"][0])


class CortexEngineTests(unittest.TestCase):
    def test_router_path_and_conformal_interval(self) -> None:
        state = {"conformal": {"scores": [0.9, 0.92, 0.88, 0.91, 0.93,
                                          0.9, 0.89]}}
        result = explain_cortex("make the bar thinner", state=state)
        self.assertEqual(result["engine"], "cortex")
        self.assertTrue(result["headline"].startswith("PLAN"))
        # the conformal verdict's own guarantee sentence rides along
        self.assertTrue(any("historically" in line for line in result["lines"]))
        self.assertIsNotNone(result["confidence"])

    def test_without_calibration_data_says_so(self) -> None:
        result = explain_cortex("make the bar thinner", state={})
        self.assertTrue(any("insufficient calibration data" in line
                            for line in result["lines"]))


class SettingsEngineTests(unittest.TestCase):
    def test_settings_answer_reused_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text("{}", encoding="utf-8")
            result = explain_settings("setBarScale", target)
            self.assertEqual(result["engine"], "settings")
            # the module's own answer, byte for byte
            self.assertTrue(result["headline"].startswith(
                "The bar.scale is currently set to"))
            self.assertIn("shell/plugin/src/Caelestia/Config/barconfig.hpp",
                          result["citations"][0])


class BrainEngineTests(unittest.TestCase):
    def test_posterior_over_labeled_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "l.json"
            ledger = Ledger(lp)
            for _ in range(3):
                pid = ledger.propose("settings", "t", {}, "r", 0.5)
                ledger.decide(pid, True)
            pid = ledger.propose("settings", "t", {}, "r", 0.5)
            ledger.decide(pid, False)
            result = explain_brain(lp, kind="settings")
            self.assertEqual(result["engine"], "brain")
            self.assertEqual(result["confidence"], 0.667)
            self.assertIn("approved ~67% of the time (4 decisions)",
                          result["headline"])
            self.assertTrue(any("settings:" in line
                                for line in result["lines"]))

    def test_empty_history_says_prior_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = explain_brain(Path(tmp) / "none.json")
            self.assertIn("Beta(1,1) prior", result["headline"])


class WizardEngineTests(unittest.TestCase):
    def test_ahp_topsis_render_reused(self) -> None:
        answers = [1, 1, 1, 1, 1, 1]
        result = explain_wizard(answers)
        self.assertEqual(result["engine"], "wizard")
        self.assertIn("preset '", result["headline"])
        # the module's own render() lines, verbatim
        expected = settings_wizard.render(settings_wizard.run(answers))
        self.assertEqual(result["lines"], expected)


class WalkBackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.ledger_path = root / "l.json"
        self.target = root / "shell.json"
        self.target.write_text("{}", encoding="utf-8")

    def test_last_action_walks_to_its_engine(self) -> None:
        ledger = Ledger(self.ledger_path)
        ledger.propose("settings", "setBarScale",
                       {"file": str(self.target),
                        "calls": [{"name": "setBarScale", "value": 0.8}]},
                       "want it thinner", 0.8)
        result = explain_last(self.ledger_path)
        self.assertEqual(result["engine"], "settings")
        self.assertTrue(result["lines"][0].startswith(
            "last action: ledger proposal #1"))
        self.assertTrue(any(line.startswith("ledger reason: want it")
                            for line in result["lines"]))

    def test_last_action_on_empty_ledger(self) -> None:
        result = explain_last(self.ledger_path)
        self.assertEqual(result["engine"], "ledger")
        self.assertIn("no recorded action", result["headline"])

    def test_gap_kind_walks_to_cortex(self) -> None:
        ledger = Ledger(self.ledger_path)
        ledger.propose("ontology_gap", "gap-cluster:panel opacity",
                       {"cluster": {"label": "panel opacity"}},
                       "4 request(s) fell through", 0.9)
        item = last_ledger_item(self.ledger_path)
        result = explain_ledger_item(item)
        self.assertEqual(result["engine"], "cortex")
        self.assertEqual(result["lines"],
                         ["4 request(s) fell through"])

    def test_inbox_id_plan_explains_cache_contract(self) -> None:
        plan_path = Path(self._tmp.name) / "session.json"
        plan_path.write_text(json.dumps(PlanCache(
            [{"tool": "setBarScale", "value": 0.8}]).to_dict()))
        inbox = Inbox(self.ledger_path, None, plan_path)
        result = explain_inbox_id(inbox, "plan:setBarScale")
        self.assertEqual(result["engine"], "cortex")
        self.assertIn("standard planner", " ".join(result["lines"]))

    def test_inbox_id_agent_needs_goal(self) -> None:
        inbox = Inbox(self.ledger_path)
        result = explain_inbox_id(inbox, "agent:n2")
        self.assertIn("goal", result["lines"][0])


class RenderAndCliTests(unittest.TestCase):
    def test_render_is_one_consistent_shape(self) -> None:
        lines = render({"engine": "brain", "headline": "h",
                        "lines": ["a", "b"], "citations": ["x:1", "y:2"],
                        "confidence": 0.5})
        self.assertEqual(lines[0], "why: [brain] h")
        self.assertEqual(lines[1], "  a")
        self.assertEqual(lines[3], "  cites: x:1; y:2")
        self.assertEqual(lines[4],
                         "  confidence: 0.5 (the producing engine's own "
                         "value)")

    def test_cli_engine_wizard(self) -> None:
        code = main(["--engine", "wizard", "--answers", "1,1,1,1,1,1"])
        self.assertEqual(code, 0)

    def test_cli_last_action_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "l.json"
            Ledger(lp).propose("drift_added", "cfg.json", {}, "new", 0.9)
            code = main(["--ledger", str(lp), "--json"])
        self.assertEqual(code, 0)

    def test_cli_inbox_id_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "l.json"
            pid = Ledger(lp).propose("ontology_gap", "gap-cluster:x",
                                     {}, "fell through", 0.9)
            code = main([f"gap:{pid}", "--ledger", str(lp)])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
