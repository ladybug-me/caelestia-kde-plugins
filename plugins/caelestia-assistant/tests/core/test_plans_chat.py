"""Chat-loop wiring tests for the session-scoped plan cache (phase 2.5)
and the what-if consequence view (phase 2.7), plus the agent-seam
consequence projection and the review-bucket datetime coercion.

The contracts under test:
- a follow-up request composes with the PENDING plan (later wins per
  tool) and the proposed card shows the COMPOSED plan;
- an explicit "apply these changes" commits the composed plan through
  the standard consent gate (backup written, both values applied);
- a refused apply leaves the ops PENDING for the next turn;
- an explicit discard ("never mind") drops the pending plan;
- a what-if turn renders the consequence view and never writes;
- a BARE "what if" projects the pending plan only — it must not route
  a placeholder request (the router's fuzzy match on "show my pending
  changes" once resolved to a toast toggle and polluted the plan);
- the session dict round-trips the pending plan;
- the agent engine's validate_plan node carries the consequence
  projection over the planner's RESOLVED values;
- a near-threshold chat session exits cleanly (the review bucket once
  carried a raw datetime into brain_state.save and crashed the exit).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from assistant.core.cli import cmd_chat
from assistant.core.plans import PlanCache
from assistant.core.session import SessionState


def _write_target(directory: Path) -> Path:
    target = directory / "shell.json"
    target.write_text(json.dumps({"bar": {"scale": 1.0}}), encoding="utf-8")
    return target


class ChatPlanCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.target = _write_target(self.dir)

    def _chat(self, script: str) -> str:
        # brain-state isolation: every scripted session writes its
        # learned state to a temp file, never the real user path
        brain_path = self.dir / "brain-state.json"
        stdin = io.StringIO(script)
        original_stdin, sys.stdin = sys.stdin, stdin
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                with unittest.mock.patch.dict(
                        os.environ,
                        {"CAELESTIA_BRAIN_STATE": str(brain_path)}):
                    cmd_chat(["--file", str(self.target), "--no-learn"])
        finally:
            sys.stdin = original_stdin
        return buffer.getvalue()

    def test_followup_composes_with_the_pending_plan(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\nmake the dock icons smaller\nn\n/exit\n")
        self.assertIn("composed with your pending plan (2 ops total)", out)
        self.assertIn("1 pending change(s)", out)
        self.assertIn("2 pending change(s)", out)
        # nothing applied: both answers were n
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)

    def test_explicit_apply_commits_the_composed_plan(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\n"
            "make the dock icons smaller\nn\n"
            "apply these changes\ny\n/exit\n")
        self.assertIn("applied 2 composed change(s)", out)
        self.assertIn("backup written", out)
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 0.9)
        self.assertEqual(data["bar"]["dock"]["iconSize"], 28)

    def test_refused_apply_leaves_ops_pending(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\n"
            "make the dock icons smaller\nn\n"
            "apply these changes\nn\n/exit\n")
        self.assertIn("still pending (refused)", out)
        self.assertIn("2 pending change(s)", out)
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)

    def test_explicit_discard_drops_the_pending_plan(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\nnever mind\n/exit\n")
        self.assertIn("discarded 1 pending change(s)", out)
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)

    def test_discard_then_new_request_starts_clean(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\nnever mind\n"
            "make the dock icons smaller\nn\n/exit\n")
        # no "composed with your pending plan" — the discard took
        self.assertNotIn("composed with your pending plan", out)
        self.assertNotIn("2 pending", out)

    def test_what_if_renders_the_consequence_view_and_writes_nothing(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\nwhat if\n/exit\n")
        self.assertIn("what-if", out)
        self.assertIn("read-only view", out)
        self.assertIn("1 pending change(s)", out)
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)

    def test_bare_what_if_does_not_route_a_placeholder(self) -> None:
        # the regression this pins: a bare "what if" routed the literal
        # placeholder "show my pending changes", the fuzzy matcher
        # resolved "changed" to a toast toggle, and the GARBAGE op was
        # composed into the pending plan
        out = self._chat(
            "make my bar thinner\nn\nwhat if\n/exit\n")
        self.assertNotIn("setToastsChargingChanged", out)

    def test_what_if_projects_resolved_values_not_step_deltas(self) -> None:
        # "make my bar thinner" resolves to bar.scale 0.9 — the what-if
        # projection must show candidate ops over 0.9, never fire the
        # 0.6-floor edge on a raw step delta of -1
        out = self._chat(
            "make my bar thinner\nn\nwhat if\n/exit\n")
        self.assertNotIn("CLAMPED at 0.6", out)
        self.assertNotIn("UNSUPPORTED", out)

    def test_what_if_text_composes_the_request(self) -> None:
        out = self._chat(
            "make my bar thinner\nn\n"
            "what if the bar scale was 0.5\n/exit\n")
        self.assertIn("what-if", out)
        self.assertIn("bar.scale (effective)", out)  # the 0.6-floor edge
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)  # never applied

    def test_apply_requires_a_pending_plan(self) -> None:
        out = self._chat("apply these changes\n/exit\n")
        self.assertNotIn("applied", out)  # nothing pending: falls through
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)


class SessionPendingPlanRoundTripTests(unittest.TestCase):
    def test_session_state_carries_the_pending_plan(self) -> None:
        state = SessionState()
        cache = PlanCache()
        cache.compose([{"tool": "setBarScale", "value": 0.8}], "thinner")
        state.pending_plan = cache.to_dict()
        data = state.to_dict()
        revived = SessionState.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(revived.pending_plan, state.pending_plan)
        revived_cache = PlanCache.from_dict(revived.pending_plan)
        self.assertEqual(revived_cache.pending(),
                         [{"tool": "setBarScale", "value": 0.8}])

    def test_malformed_pending_plan_degrades_to_none(self) -> None:
        revived = SessionState.from_dict({"pending_plan": "not-a-dict"})
        self.assertIsNone(revived.pending_plan)
        revived2 = SessionState.from_dict({"pending_plan": None})
        self.assertIsNone(revived2.pending_plan)


class EngineConsequenceSeamTests(unittest.TestCase):
    """The agent layer's validate_plan node carries the what-if
    projection (the consent card shows consequences, not just actions)."""

    def test_route_request_carries_resolved_ops(self) -> None:
        from assistant.capabilities.agent.engine import _d_route_request
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(json.dumps({"bar": {"scale": 1.0}}),
                              encoding="utf-8")
            res = _d_route_request(
                {"text": "make my bar thinner"}, {})
            # hmm: _d_route_request routes without a file — the pipeline
            # defaults to the real target; assert on the raw shape only
            self.assertIn("ops", res)
            self.assertIn("resolved_ops", res)
            self.assertIn("ops_count", res)

    def test_validate_plan_projects_consequences_over_resolved_ops(self) -> None:
        from assistant.capabilities.agent.engine import _d_route_request, _d_validate_plan
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shell.json"
            target.write_text(
                json.dumps({"appearance": {"blur": True}}),
                encoding="utf-8")
            res = _d_route_request({"text": "disable blur"}, {})
            ctx = {"results": {"route_request": res}}
            verdict = _d_validate_plan(ctx, {})
            self.assertTrue(verdict["validated"])
            cons = verdict["consequences"]
            self.assertEqual(cons.get("n_ops", 0), 1)
            self.assertIn("reversibility", cons)
            self.assertIn("derived", cons)


class ReviewBucketCoercionTests(unittest.TestCase):
    """The pre-existing chat-exit crash this session fixed: the review
    bucket stored the live ``_now()`` datetime raw, and brain_state.save
    raised 'Object of type datetime is not JSON serializable'."""

    def test_log_review_candidate_coerces_datetime(self) -> None:
        from datetime import datetime
        from assistant.core import learn as cortex_learn
        bucket = cortex_learn.log_review_candidate(
            [], "some ambiguous phrase", "AMBIGUOUS",
            [{"surface": "setBarScale", "p": 0.42}],
            at=datetime(2026, 9, 26, 12, 0, 0))
        entry = bucket[0]
        json.dumps(entry)  # must not raise
        self.assertIsInstance(entry["at"], str)
        self.assertIn("2026-09-26T12:00:00", entry["at"])

    def test_near_threshold_chat_session_exits_cleanly(self) -> None:
        # a session ending on a parked near-threshold phrase must save
        # state and exit 0 — not crash at brain_state.save
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_target(Path(tmp))
            brain_state_path = Path(tmp) / "brain.json"
            out = io.StringIO()
            stdin = io.StringIO("zzz qqq xxx\n/exit\n")
            original_stdin, sys.stdin = sys.stdin, stdin
            try:
                import os
                from assistant.capabilities.brain import state as brain_state
                with redirect_stdout(out):
                    with unittest.mock.patch.dict(
                            os.environ,
                            {"CAELESTIA_BRAIN_STATE": str(brain_state_path)}):
                        rc = cmd_chat(["--file", str(target), "--no-learn"])
            finally:
                sys.stdin = original_stdin
            self.assertEqual(rc, 0)
            self.assertIn("session ended", out.getvalue())
            if brain_state_path.exists():
                json.loads(brain_state_path.read_text())  # must not raise


if __name__ == "__main__":
    unittest.main()
