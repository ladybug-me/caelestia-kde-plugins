"""Tests for the unified pending-decisions inbox (cortex/inbox.py).

The contract under test:
- AGGREGATION: the four sources (ledger, gap clusters, pending plan,
  agent consent nodes) appear in ONE ranked view; the ranking composes
  ONLY values that already exist (stated confidence, calibrate's
  Beta-Binomial kind posterior, the NamedBandit arm mean) with the
  class's own neutral priors where no history exists, and the order is
  deterministic (score desc, then SOURCE_ORDER, then id);
- DISPATCH: approve/reject call each source module's OWN entry point —
  Ledger.decide, dispatch.propose_gap_cluster, PlanCache.discard/commit
  behind the standard planner + applier, Agent's consent_fn gate — and
  nothing else writes;
- the inbox never invents a write path: an unproposed gap cluster's
  reject touches nothing, plan applies are dry-run without --write,
  agent approves without --write never execute.

Every source module is exercised through a fixture (tmp ledger/state/
plan payload) or a mock (Agent, the settings planner/applier).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assistant.core import dispatch as dispatch_mod
from assistant.core.inbox import SOURCE_ORDER, Inbox, main
from assistant.core.plans import PlanCache
from assistant.capabilities.brain.ledger import Ledger

GAPS_KEY = dispatch_mod.GAPS_KEY


def _gap_state() -> dict:
    """One qualifying gap cluster: support 4 (>= 3), purity 1.0 (>= 0.6)."""
    return {GAPS_KEY: [{"shape": ["panel", "opacity"],
                        "category": "router-abstain", "n": 4,
                        "first_at": "2026-09-26T10:00:00+00:00",
                        "last_at": "2026-09-26T11:00:00+00:00"}]}


class InboxAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.ledger_path = root / "ledger.json"
        self.state_path = root / "state.json"
        self.plan_path = root / "session.json"
        ledger = Ledger(self.ledger_path)
        # labeled history: kind "settings" seen 3 approved + 1 rejected
        # (posterior alpha=4, beta=2, mean 0.667); a second pending item
        # of a kind with NO history (posterior falls back to the flat
        # prior 0.5).
        for _ in range(3):
            pid = ledger.propose("settings", "tool:x", {}, "r", 0.5)
            ledger.decide(pid, True)
        pid = ledger.propose("settings", "tool:x", {}, "r", 0.5)
        ledger.decide(pid, False)
        self.settings_pid = ledger.propose(
            "settings", "setBarScale", {"tool": "setBarScale"},
            "want it", 0.8)
        self.drift_pid = ledger.propose("drift_added", "some-id", {},
                                        "drift", 0.6)
        self.state_path.write_text(json.dumps({
            "preset_bandit": {"tool:setBarScale": [3.0, 1.0]},  # mean 0.75
            **_gap_state(),
        }), encoding="utf-8")
        self.plan_path.write_text(json.dumps(PlanCache(
            [{"tool": "setBarScale", "value": 0.8}]).to_dict()),
            encoding="utf-8")

    def _inbox(self) -> Inbox:
        return Inbox(self.ledger_path, self.state_path, self.plan_path)

    def test_all_four_sources_aggregate(self) -> None:
        agent_projection = {"projection": [
            {"id": "n2", "title": "Apply plan", "risk": "STATE_CHANGING",
             "consent_required": True, "depends": [],
             "would": "ASK you before apply_plan"}]}

        class FakeAgent:
            def __init__(self, *a, **k):
                pass

            def simulate(self, goal):
                return dict(agent_projection, order=[], critical_path=[],
                            consent_required=True, ambiguous=False,
                            note="")

        with mock.patch("assistant.capabilities.agent.engine.Agent", FakeAgent):
            items = self._inbox().items(goal="tidy my downloads")
        sources = {i["source"] for i in items}
        self.assertEqual(
            sources, {"ledger", "gap", "plan", "agent"},
            f"all four sources must aggregate, got {sources}")
        ids = {i["id"] for i in items}
        self.assertIn(f"ledger:{self.settings_pid}", ids)
        self.assertIn(f"ledger:{self.drift_pid}", ids)
        self.assertIn("gap:panel opacity", ids)  # unproposed qualifying
        self.assertIn("plan:setBarScale", ids)
        self.assertIn("agent:n2", ids)

    def test_ranking_composes_existing_values_only(self) -> None:
        items = self._inbox().items()
        by_id = {i["id"]: i for i in items}
        # ledger settings item: stated 0.8 x kind 0.667 x arm 0.75
        self.assertEqual(by_id[f"ledger:{self.settings_pid}"]["rank_inputs"],
                         {"stated": 0.8, "kind_mean": 0.667,
                          "arm_mean": 0.75, "arm": "tool:setBarScale"})
        self.assertEqual(by_id[f"ledger:{self.settings_pid}"]["score"], 0.4)
        # plan op: no stated confidence (neutral 1.0), same kind and arm
        self.assertEqual(by_id["plan:setBarScale"]["rank_inputs"],
                         {"stated": 1.0, "kind_mean": 0.667,
                          "arm_mean": 0.75, "arm": "tool:setBarScale"})
        self.assertEqual(by_id["plan:setBarScale"]["score"], 0.5)
        # drift item: kind with no history -> the flat prior; no arm
        self.assertEqual(
            by_id[f"ledger:{self.drift_pid}"]["rank_inputs"]["kind_mean"],
            0.5)
        self.assertIsNone(
            by_id[f"ledger:{self.drift_pid}"]["rank_inputs"]["arm"])
        self.assertEqual(by_id[f"ledger:{self.drift_pid}"]["score"],
                         round(0.6 * 0.5 * 0.5, 3))
        # unproposed gap cluster: stated = purity, kind prior, no arm
        self.assertEqual(by_id["gap:panel opacity"]["score"],
                         round(1.0 * 0.5 * 0.5, 3))

    def test_order_deterministic_score_desc_then_source_then_id(self) -> None:
        items = self._inbox().items()
        keys = [(-i["score"], SOURCE_ORDER[i["source"]], i["id"])
                for i in items]
        self.assertEqual(keys, sorted(keys))
        scores = [i["score"] for i in items]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_sources_render_empty_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Inbox(Path(tmp) / "none.json")
            self.assertEqual(empty.items(), [])


class InboxLedgerDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.ledger_path = root / "ledger.json"
        self.inbox = Inbox(self.ledger_path)
        self.pid = Ledger(self.ledger_path).propose(
            "settings", "setBarScale", {"tool": "setBarScale"}, "want", 0.8)

    def test_approve_dispatches_to_ledger_decide(self) -> None:
        result = self.inbox.decide(f"ledger:{self.pid}", True)
        self.assertTrue(result["dispatched"])
        self.assertEqual(result["via"], "brain.ledger.Ledger.decide")
        self.assertEqual(result["status"], "approved")
        self.assertEqual(
            Ledger(self.ledger_path).items[0]["status"], "approved")

    def test_reject_dispatches_to_ledger_decide(self) -> None:
        result = self.inbox.decide(f"ledger:{self.pid}", False)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(
            Ledger(self.ledger_path).items[0]["status"], "rejected")

    def test_double_decide_reports_not_dispatched(self) -> None:
        self.inbox.decide(f"ledger:{self.pid}", True)
        result = self.inbox.decide(f"ledger:{self.pid}", True)
        self.assertFalse(result["dispatched"])
        self.assertIn("already", result["note"])


class InboxGapDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.ledger_path = root / "ledger.json"
        self.state_path = root / "state.json"
        self.state_path.write_text(json.dumps(_gap_state()), encoding="utf-8")
        self.inbox = Inbox(self.ledger_path, self.state_path)

    def test_approve_files_cluster_through_dispatch_entry_point(self) -> None:
        result = self.inbox.decide("gap:panel opacity", True)
        self.assertTrue(result["dispatched"])
        self.assertEqual(result["via"],
                         "cortex.dispatch.propose_gap_cluster")
        ledger = Ledger(self.ledger_path)
        self.assertEqual(len(ledger.pending()), 1)
        proposal = ledger.pending()[0]
        self.assertEqual(proposal["kind"], "ontology_gap")
        self.assertEqual(proposal["target"], "gap-cluster:panel opacity")
        # the decision STILL belongs to the ledger, not the inbox: the
        # filed proposal stays pending for the human's real call
        self.assertEqual(proposal["status"], "pending")

    def test_reject_unproposed_cluster_writes_nothing(self) -> None:
        result = self.inbox.decide("gap:panel opacity", False)
        self.assertTrue(result["dispatched"])
        self.assertEqual(result["status"], "not-proposed")
        self.assertFalse(self.ledger_path.exists(),
                         "rejecting an unproposed cluster must not write")

    def test_unknown_label_refuses_honestly(self) -> None:
        result = self.inbox.decide("gap:nonexistent", True)
        self.assertFalse(result["dispatched"])

    def test_surfaced_gap_pid_decides_in_ledger(self) -> None:
        ledger = Ledger(self.ledger_path)
        pid = ledger.propose("ontology_gap", "gap-cluster:panel opacity",
                             {"cluster": {}}, "why", 0.9)
        result = self.inbox.decide(f"gap:{pid}", True)
        self.assertEqual(result["via"], "brain.ledger.Ledger.decide")
        self.assertEqual(Ledger(self.ledger_path).items[0]["status"],
                         "approved")


class InboxPlanDispatchTests(unittest.TestCase):
    """The plan cache is a pure module: its payload file is the fixture,
    planner/applier are mocks — the inbox must call them, not re-implement
    them, and must never write without the explicit write flag."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.ledger_path = root / "ledger.json"
        self.plan_path = root / "session.json"
        self.target = root / "shell.json"
        self.ops = [{"tool": "setBarScale", "value": 0.8},
                    {"tool": "setDockIconSize", "value": 28}]
        self.plan_path.write_text(
            json.dumps(PlanCache(self.ops).to_dict()), encoding="utf-8")
        self.inbox = Inbox(self.ledger_path, None, self.plan_path)

    def _payload(self) -> dict:
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def test_reject_discards_the_whole_plan(self) -> None:
        result = self.inbox.decide("plan:setBarScale", False)
        self.assertTrue(result["dispatched"])
        self.assertEqual(result["via"], "cortex.plans.PlanCache.discard")
        self.assertIn("WHOLE", result["note"])
        self.assertEqual(self._payload()["ops"], [])

    def test_approve_without_target_refuses(self) -> None:
        result = self.inbox.decide("plan:setBarScale", True)
        self.assertFalse(result["dispatched"])
        self.assertIn("no target file", result["note"])

    def test_approve_dry_run_validates_through_planner(self) -> None:
        with mock.patch("assistant.capabilities.settings.planner.plan",
                        return_value={"entries": []}) as planner, \
                mock.patch("assistant.capabilities.settings.applier.apply",
                           return_value={"applied": 0}) as applier:
            result = self.inbox.decide("plan:setBarScale", True,
                                       target=self.target, write=False)
        planner.assert_called_once_with(
            [{"tool": "setBarScale", "value": 0.8}], self.target)
        applier.assert_called_once()
        self.assertEqual(applier.call_args.kwargs.get("write"), False)
        self.assertEqual(result["status"], "preview")
        # a dry-run closes nothing: the plan stays pending
        self.assertEqual(self._payload()["ops"], self.ops)

    def test_approve_write_applies_and_commits_cache(self) -> None:
        with mock.patch("assistant.capabilities.settings.planner.plan",
                        return_value={"entries": []}), \
                mock.patch("assistant.capabilities.settings.applier.apply",
                           return_value={"applied": 1}) as applier:
            result = self.inbox.decide("plan:setBarScale", True,
                                       target=self.target, write=True)
        self.assertEqual(applier.call_args.kwargs.get("write"), True)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["via"],
                         "settings.applier.apply (write=True, backed up)")
        # commit ONLY after an apply went through — the cache's own rule
        self.assertEqual(self._payload()["ops"], [])

    def test_unknown_tool_refuses(self) -> None:
        result = self.inbox.decide("plan:nope", True)
        self.assertFalse(result["dispatched"])


class _FakeAgentBase:
    def __init__(self, dispatchers=None, consent_fn=None, **kw):
        self.consent_fn = consent_fn
        type(self).last_consent = consent_fn

    def simulate(self, goal):
        return {"projection": [{"id": "n2", "action": "apply_plan",
                                "title": "Apply plan",
                                "risk": "STATE_CHANGING",
                                "consent_required": True, "depends": [],
                                "would": "ASK you before apply_plan"}],
                "order": ["n2"], "critical_path": [],
                "consent_required": ["n2"], "ambiguous": False, "note": ""}


class InboxAgentDispatchTests(unittest.TestCase):
    GOAL = "clean my downloads then apply the plan"

    def test_items_surface_consent_required_nodes_only(self) -> None:
        with mock.patch("assistant.capabilities.agent.engine.Agent", _FakeAgentBase):
            inbox = Inbox("unused-ledger")
            items = inbox.items(goal=self.GOAL)
        self.assertEqual([i["id"] for i in items], ["agent:n2"])
        self.assertIn("STATE_CHANGING", items[0]["why"])

    def test_approve_with_write_scripts_the_consent_gate(self) -> None:
        executed = {}

        class FakeAgentRun(_FakeAgentBase):
            def execute(self, goal):
                executed["goal"] = goal
                return {"summary": {"done": 1, "refused": 0,
                                    "consent_n2": self.consent_fn({"id": "n2"}),
                                    "consent_n9": self.consent_fn({"id": "n9"})},
                        "order": ["n2"]}

        with mock.patch("assistant.capabilities.agent.engine.Agent", FakeAgentRun):
            inbox = Inbox("unused-ledger")
            result = inbox.decide("agent:n2", True, goal=self.GOAL,
                                  write=True)
        self.assertEqual(executed["goal"], self.GOAL)
        self.assertEqual(result["status"], "consented-and-run")
        self.assertTrue(result["summary"]["consent_n2"],
                        "the decided node is consented")
        self.assertFalse(result["summary"]["consent_n9"],
                         "every OTHER node stays refused")

    def test_approve_without_write_never_executes(self) -> None:
        with mock.patch("assistant.capabilities.agent.engine.Agent", _FakeAgentBase):
            inbox = Inbox("unused-ledger")
            result = inbox.decide("agent:n2", True, goal=self.GOAL,
                                  write=False)
        self.assertEqual(result["via"], "agent.engine.Agent.simulate "
                                        "(read-only)")
        self.assertIn("projection", result)

    def test_reject_refuses_without_running(self) -> None:
        ran = []

        class FakeAgentNoRun(_FakeAgentBase):
            def execute(self, goal):
                ran.append(goal)
                return {}

        with mock.patch("assistant.capabilities.agent.engine.Agent", FakeAgentNoRun):
            inbox = Inbox("unused-ledger")
            result = inbox.decide("agent:n2", False, goal=self.GOAL,
                                  write=True)
        self.assertEqual(ran, [], "a refusal must not execute anything")
        self.assertIn("refused", result["status"])

    def test_agent_needs_the_goal(self) -> None:
        inbox = Inbox("unused-ledger")
        result = inbox.decide("agent:n2", True, goal=None, write=True)
        self.assertFalse(result["dispatched"])


class InboxCliTests(unittest.TestCase):
    def test_cli_list_json_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["list", "--ledger", str(Path(tmp) / "l.json"),
                         "--state", str(Path(tmp) / "s.json"), "--json"])
        self.assertEqual(code, 0)

    def test_cli_renders_ranked_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "l.json")
            ledger.propose("settings", "setBarScale", {}, "want", 0.8)
            code = main(["list", "--ledger", str(Path(tmp) / "l.json"),
                         "--state", str(Path(tmp) / "s.json")])
        self.assertEqual(code, 0)

    def test_cli_unknown_id_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["approve", "bogus:1",
                         "--ledger", str(Path(tmp) / "l.json"),
                         "--state", str(Path(tmp) / "s.json")])
        self.assertEqual(code, 1)

    def test_cli_approve_ledger_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "l.json"
            pid = Ledger(lp).propose("drift_added", "x", {}, "r", 0.5)
            code = main(["approve", f"ledger:{pid}",
                         "--ledger", str(lp),
                         "--state", str(Path(tmp) / "s.json")])
            self.assertEqual(code, 0)
            self.assertEqual(Ledger(lp).items[0]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
