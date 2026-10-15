"""Cortex integration tests: compound, nlhistory, session, memory,
learn, pipeline (end-to-end against a real temp shell.json), CLI, and
hub/bridge wiring.

Safety-critical behaviors pinned here:
- the pipeline NEVER writes (applies only happen through the CLI's
  confirmed paths, which these tests exercise against temp files);
- every apply produces history entries and a backup, and undo reverts;
- learning converges (calibration tracks the true acceptance rate) and
  round-trips through persistence;
- the chat REPL answers y/N prompts only with explicit consent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from assistant.core.compound import route_compound, split_clauses
from assistant.core.learn import CortexLearner, OnlineLogistic
from assistant.core.memory import (
    cooccurrence,
    followup_suggestion,
    habits,
    lift,
    new_episode,
    record,
    recall,
    surface_counts,
    summarize,
)
from assistant.core.nlhistory import parse_query, plan as plan_history
from assistant.core.pipeline import episode_for, ops_for_candidate, process
from assistant.core.session import SessionState, turn as session_turn
from assistant.capabilities.settings import history as settings_history


def _write_target(directory: Path, payload: dict | None = None) -> Path:
    target = directory / "shell.json"
    base = {"bar": {"scale": 1.0, "position": "bottom"},
            "appearance": {"blur": True, "transparency": {"base": 0.85}}}
    if payload:
        for key, value in payload.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key].update(value)
            else:
                base[key] = value
    target.write_text(json.dumps(base), encoding="utf-8")
    return target


class CompoundTests(unittest.TestCase):
    def test_single_intent_no_split(self):
        result = route_compound("make my bar thinner")
        self.assertTrue(result.single)

    def test_two_intents_split_and_routed(self):
        result = route_compound("make the bar thinner and move the dock to the left")
        self.assertFalse(result.single)
        surfaces = [r.top.surface for r in result.routes if r.top]
        self.assertIn("setBarScale", surfaces)
        self.assertIn("setBarPosition", surfaces)

    def test_three_intents_comma_and_and(self):
        result = route_compound("disable blur, faster animations and smaller corners")
        surfaces = [r.top.surface for r in result.routes if r.top]
        self.assertIn("setBlurEnabled", surfaces)
        self.assertIn("setAnimationSpeed", surfaces)
        self.assertIn("setRoundingScale", surfaces)

    def test_keep_clause_flagged(self):
        result = route_compound("make everything compact but keep the blur on")
        keep_clauses = [c for c in result.clauses if c.keep]
        self.assertTrue(keep_clauses)
        self.assertIn("keep-clauses are honored", " ".join(result.notes))

    def test_deterministic(self):
        self.assertEqual(split_clauses("a and b, but c"),
                         split_clauses("a and b, but c"))


class NlHistoryTests(unittest.TestCase):
    def test_undo_last_steps(self):
        query = parse_query("undo the last change")
        self.assertEqual(query.kind, "undo")
        self.assertEqual(query.steps, 1)

    def test_undo_two_steps(self):
        query = parse_query("undo my last two changes")
        self.assertEqual(query.steps, 2)

    def test_scoped_undo(self):
        query = parse_query("undo the bar changes")
        self.assertEqual(query.scope_words, ["bar"])

    def test_yesterday_window(self):
        now = datetime(2026, 9, 25, 15, 0)
        query = parse_query("restore yesterday's theme", now=now)
        self.assertIsNotNone(query.window)
        lo, hi = query.window
        self.assertEqual(lo.date(), (now - timedelta(days=1)).date())

    def test_listing_kind(self):
        self.assertEqual(parse_query("what did i change").kind, "list")

    def test_plan_undo_steps_against_real_entries(self):
        entries = [
            {"id": 2, "at": "2026-09-25T10:00:00", "label": "chat: b",
             "ops": [{"path": "bar.scale", "old": 0.9, "new": 1.0}]},
            {"id": 1, "at": "2026-09-24T10:00:00", "label": "chat: a",
             "ops": [{"path": "bar.scale", "old": None, "new": 0.9}]},
        ]
        query = parse_query("undo the last change")
        result = plan_history(query, entries)
        self.assertEqual(result.verdict, "UNDO")
        self.assertEqual(result.action, "undo")
        self.assertEqual(result.steps, 1)

    def test_scoped_not_found_is_honest(self):
        entries = [
            {"id": 1, "at": "2026-09-25T10:00:00", "label": "chat: a",
             "ops": [{"path": "notifs.maxPopups", "old": 8, "new": 5}]},
        ]
        query = parse_query("undo the bar changes")
        result = plan_history(query, entries)
        self.assertEqual(result.verdict, "NOT_FOUND")
        self.assertIsNone(result.action)  # never a silent fallback


class SessionTests(unittest.TestCase):
    def test_ellipsis_resolution(self):
        state = SessionState()
        session_turn("make the dock icons bigger", state)
        result = session_turn("a bit smaller now", state)
        self.assertIn("ellipsis", result.note)
        self.assertIn("dock", result.resolved_text)

    def test_pronoun_resolution(self):
        state = SessionState()
        session_turn("make the bar thinner", state)
        result = session_turn("actually make it much bigger", state)
        self.assertIn("pronoun", result.note)
        self.assertIn("bar", result.resolved_text)

    def test_pronoun_without_context_is_honest(self):
        state = SessionState()
        result = session_turn("make it bigger", state)
        self.assertIn("no earlier setting", result.note)

    def test_ordinal_answer_resolves_ambiguity(self):
        state = SessionState()
        first = session_turn("make the background nicer", state)
        if first.question:  # only meaningful when the turn was ambiguous
            second = session_turn("the second one", state)
            self.assertEqual(second.mode, "answer")
            self.assertIsNotNone(second.chosen_surface)

    def test_no_answer_drops_pending(self):
        state = SessionState()
        session_turn("make the background nicer", state)
        if state.pending_question:
            session_turn("no", state)
            self.assertIsNone(state.pending_question)

    def test_transcript_bounded(self):
        state = SessionState()
        for i in range(80):
            session_turn(f"make the bar thinner {i}", state)
        self.assertLessEqual(len(state.turns), 64)

    def test_state_round_trip(self):
        state = SessionState()
        session_turn("make the bar thinner", state)
        restored = SessionState.from_dict(state.to_dict())
        self.assertEqual(len(restored.turns), len(state.turns))
        self.assertEqual(restored.last_surfaces, state.last_surfaces)


class MemoryTests(unittest.TestCase):
    def _episodes(self):
        episodes = []
        base = datetime(2026, 9, 25, 12, 0, 0)
        # applied episodes that CO-CHANGE bar scale + spacing (+ padding);
        # one dock-only episode keeps P(bar) < 1 so lift is meaningful
        for i, (surfaces, days_ago) in enumerate([
            (["setBarScale", "setSpacingScale"], 0),
            (["setBarScale", "setSpacingScale"], 1),
            (["setBarScale", "setSpacingScale", "setPaddingScale"], 5),
            (["setBarScale"], 10),
            (["setDockIconSize"], 2),
        ]):
            episodes = record(episodes, new_episode(
                f"req {i}", f"resolved {i}", surfaces, "plan", "applied",
                at=base - timedelta(days=days_ago)))
        episodes = record(episodes, new_episode(
            "rejected", "rejected", ["setBlurEnabled"], "plan", "rejected", at=base))
        return episodes, base

    def test_record_bounds(self):
        episodes, _ = self._episodes()
        for _ in range(600):
            episodes = record(episodes, new_episode("x", "x", ["setBarScale"], "plan", "routed"))
        self.assertLessEqual(len(episodes), 500)

    def test_recall_ranked_by_decay(self):
        episodes, now = self._episodes()
        rows = recall(episodes, "setBarScale", now=now)
        self.assertTrue(rows)
        # the most recent setBarScale episode first
        self.assertEqual(rows[0]["at"], "2026-09-25T12:00:00")

    def test_cooccurrence_counts(self):
        episodes, _ = self._episodes()
        co = cooccurrence(episodes)
        self.assertIn(("setBarScale", "setSpacingScale"), co)

    def test_lift_positive_for_cochanged(self):
        episodes, _ = self._episodes()
        counts = surface_counts(episodes)
        total = 5  # applied episodes
        lifted = lift(cooccurrence(episodes), counts, total)
        pair = [row for row in lifted if row["a"] == "setBarScale" and row["b"] == "setSpacingScale"]
        self.assertTrue(pair)
        self.assertGreater(pair[0]["lift"], 1.0)

    def test_followup_suggestion(self):
        episodes, now = self._episodes()
        suggestions = followup_suggestion(episodes, "setBarScale", now=now)
        surfaces = [s["surface"] for s in suggestions]
        self.assertIn("setSpacingScale", surfaces)

    def test_habits_histogram(self):
        episodes, _ = self._episodes()
        habit = habits(episodes)
        self.assertEqual(sum(habit["hours"]), 5)

    def test_summarize_shape(self):
        episodes, _ = self._episodes()
        summary = summarize(episodes)
        self.assertIn("episodes", summary)
        self.assertIn("top_surfaces", summary)


class LearnTests(unittest.TestCase):
    def test_online_logistic_learns_direction(self):
        model = OnlineLogistic()
        # feature 'lex' perfectly correlates with the label
        for _ in range(40):
            model.update({"lex": 1.0, "sem": 0.0, "fuzz": 0.0, "noun": 0.0, "cue": 0.0}, 1)
            model.update({"lex": 0.0, "sem": 0.0, "fuzz": 0.0, "noun": 0.0, "cue": 0.0}, 0)
        # the separation is what matters (bias absorbs the base rate)
        self.assertGreater(model.predict({"lex": 1.0}), 0.6)
        self.assertLess(model.predict({"lex": 0.0}), 0.45)
        self.assertGreater(model.predict({"lex": 1.0}) - model.predict({"lex": 0.0}), 0.25)

    def test_calibration_converges_to_true_rate(self):
        learner = CortexLearner()
        for i in range(100):
            learner.observe("t", "setBarScale",
                            {"lex": 0.5, "sem": 0.5, "fuzz": 0.3, "noun": 1.0, "cue": 0.5},
                            0.6, "applied" if i % 4 == 0 else "rejected")
        # true rate 0.25; posterior mean must sit between 0.1 and 0.45
        self.assertLess(learner.calibration.mean, 0.45)
        self.assertGreater(learner.calibration.mean, 0.1)

    def test_persistence_round_trip(self):
        learner = CortexLearner()
        learner.observe("t", "setBarScale",
                        {"lex": 0.5, "sem": 0.5, "fuzz": 0.3, "noun": 1.0, "cue": 0.5},
                        0.6, "applied")
        learner.reward_strategy("lexical", True)
        data = learner.to_dict()
        restored = CortexLearner(data)
        self.assertEqual(restored.model.examples, 1)
        self.assertEqual(restored.to_dict(), data)

    def test_router_state_uses_strategy_below_evidence_threshold(self):
        learner = CortexLearner()
        state = learner.router_state()
        self.assertIsInstance(state.min_score, float)

    def test_drift_report_insufficient_data(self):
        learner = CortexLearner()
        self.assertEqual(learner.drift_check()["status"], "insufficient-data")

    def test_garbage_state_falls_back(self):
        learner = CortexLearner({"model": "not a dict", "examples": "nope"})
        self.assertEqual(learner.model.examples, 0)


class OpsResolutionTests(unittest.TestCase):
    def test_bool_cue_to_set(self):
        ops, notes = ops_for_candidate("setBlurEnabled", {"bool": False}, "turn off the blur")
        self.assertEqual(ops[0]["action"], "set")
        self.assertIs(ops[0]["value"], False)

    def test_direction_to_step(self):
        ops, _ = ops_for_candidate("setBarScale", {"direction": -1}, "thinner")
        self.assertEqual(ops[0]["action"], "step")
        self.assertEqual(ops[0]["value"], -1)

    def test_intense_direction_double_step(self):
        ops, _ = ops_for_candidate("setBarScale", {"direction": 1}, "much bigger")
        self.assertEqual(ops[0]["value"], 2)

    def test_percent_to_multiply(self):
        ops, _ = ops_for_candidate("setBarScale", {"direction": -1, "percent": 20.0}, "20% smaller")
        self.assertEqual(ops[0]["action"], "multiply")
        self.assertAlmostEqual(ops[0]["value"], 0.8)

    def test_transparency_inversion(self):
        # "less transparent" RAISES the opacity base
        ops, notes = ops_for_candidate("setTransparencyBase", {"direction": -1},
                                       "less transparent")
        self.assertEqual(ops[0]["value"], 1)
        self.assertTrue(any("more transparent" in n or "opacity" in n for n in notes))

    def test_number_to_set(self):
        ops, _ = ops_for_candidate("setBarScale", {"number": 1.3}, "bar height 1.3")
        self.assertEqual(ops[0]["action"], "set")
        self.assertEqual(ops[0]["value"], 1.3)

    def test_enum_position(self):
        ops, _ = ops_for_candidate("setBarPosition", {"position": "left"}, "to the left")
        self.assertEqual(ops[0]["value"], "left")

    def test_reset_to_default(self):
        ops, _ = ops_for_candidate("setBarScale", {"reset": True}, "reset the bar scale")
        self.assertEqual(ops[0]["value"], 1.0)

    def test_string_tool_is_honest(self):
        # The 2026-09-28 upstream resync removed the registry's only
        # string-kind tool (setFontMonoFamily), so this guard now builds a
        # synthetic string spec: the honesty property (string tools never
        # produce ops from free text; the note says so) must survive even
        # when the class is empty in the shipped registry.
        from unittest import mock
        from assistant.adapters.caelestia.registry import ToolSpec
        synthetic = ToolSpec(name="setTestString", path="test.string",
                             kind="string", default="")
        with mock.patch("assistant.core.pipeline.tool_by_name",
                        return_value=synthetic):
            ops, notes = ops_for_candidate("setTestString", {}, "set font")
        self.assertEqual(ops, [])
        self.assertTrue(any("string" in n for n in notes))

    def test_toggle_placeholder(self):
        ops, _ = ops_for_candidate("setDockBadges", {"toggle": True}, "toggle badges")
        self.assertEqual(ops[0]["value"], "TOGGLE")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.target = _write_target(self.dir)

    def test_plan_verdict_with_entries(self):
        result = process("make my bar thinner", file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        self.assertEqual(result.plan["entries"][0]["tool"], "setBarScale")
        self.assertEqual(result.plan["entries"][0]["new"], 0.9)

    def test_percent_multiply(self):
        result = process("make the bar 25% smaller", file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        entry = result.plan["entries"][0]
        self.assertEqual(entry["action"], "multiply")
        self.assertAlmostEqual(entry["new"], 0.75)

    def test_compound_request_composes(self):
        result = process("disable blur and make animations faster", file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        tools = {e["tool"] for e in result.plan["entries"]}
        self.assertEqual(tools, {"setBlurEnabled", "setAnimationSpeed"})

    def test_explain_verdict(self):
        result = process("why is my dock blurry", file_path=self.target)
        self.assertEqual(result.verdict, "EXPLAIN")
        self.assertIsNotNone(result.explain_answer)

    def test_undo_verdict(self):
        result = process("undo the last change", file_path=self.target)
        self.assertIn(result.verdict, ("UNDO", "LIST"))

    def test_inert_wallpaper(self):
        result = process("change my wallpaper", file_path=self.target)
        self.assertEqual(result.verdict, "INERT")
        self.assertTrue(result.suggestions)

    def test_delegate_diagnose(self):
        result = process("my shell crashed and nothing works", file_path=self.target)
        self.assertEqual(result.verdict, "DELEGATE")
        self.assertEqual(result.delegate, "diagnose")

    def test_never_writes(self):
        before = self.target.read_bytes()
        for text in ("make my bar thinner", "make everything compact",
                     "bar scale 1.4", "turn off the blur"):
            process(text, file_path=self.target)
        self.assertEqual(self.target.read_bytes(), before)

    def test_toggle_resolved_against_live_file(self):
        result = process("toggle the app badges", file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        entry = result.plan["entries"][0]
        # absent key -> registry default True (badges ship on) -> toggles OFF
        self.assertIs(entry["new"], False)

    def test_out_of_range_is_rejected_not_clamped(self):
        result = process("bar height 99", file_path=self.target)
        self.assertIn(result.verdict, ("PLAN", "QUESTION"))
        entry = result.plan["entries"][0]
        self.assertIsNotNone(entry.get("error"),
                             "absolute out-of-range must be an error entry, never a silent clamp")
        self.assertTrue(result.plan["apply_blocked"])

    def test_apply_through_applier_then_undo(self):
        from assistant.capabilities.settings import applier as settings_applier

        result = process("make my bar thinner", file_path=self.target)
        settings_applier.apply(result.plan, self.target, write=True, label="test")
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 0.9)
        entries = settings_history.entries(self.target)
        self.assertEqual(len(entries), 1)
        outcome = settings_history.undo(self.target, steps=1)
        # undo() returns the applier's result: "written" on success
        self.assertTrue(outcome.get("written") or outcome.get("restored"))
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)

    def test_learn_hook_present_with_learner(self):
        learner = CortexLearner()
        result = process("make my bar thinner", learner=learner, file_path=self.target)
        self.assertIsNotNone(result.learn_hook)
        self.assertEqual(result.learn_hook["surface"], "setBarScale")

    def test_episode_for_shape(self):
        result = process("make my bar thinner", file_path=self.target)
        episode = episode_for(result, "make my bar thinner", "routed")
        self.assertEqual(episode["outcome"], "routed")
        self.assertIn("setBarScale", episode["surfaces"])

    def test_session_integration(self):
        from assistant.core.session import SessionState

        session = SessionState()
        first = process("make the dock icons bigger", session=session, file_path=self.target)
        self.assertEqual(first.verdict, "PLAN")
        second = process("a bit smaller now", session=session, file_path=self.target)
        self.assertIn("ellipsis", second.session_note)
        tools = {e["tool"] for e in second.plan["entries"]}
        self.assertIn("setDockIconSize", tools)


class AgentDelegationTests(unittest.TestCase):
    """Phase 1.3: multi-step / goal-shaped requests route to the agent
    layer (a simulated task graph, consent untouched) instead of ABSTAIN
    or a settings misfire."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.target = _write_target(self.dir)

    def test_sequenced_goal_delegates_to_agent(self):
        result = process("clean my downloads then make the shell minimal",
                         file_path=self.target)
        self.assertEqual(result.verdict, "DELEGATE")
        self.assertEqual(result.delegate, "agent")
        self.assertIn("consent", " ".join(result.notes))

    def test_more_multi_step_phrases_delegate_to_agent(self):
        for text in (
            "tidy up my files then make the bar smaller",
            "first audit my packages then declutter the dock",
            "clean my downloads, after that make everything minimal",
            "back up my dotfiles then make the shell minimal",
        ):
            result = process(text, file_path=self.target)
            self.assertEqual(result.verdict, "DELEGATE", text)
            self.assertEqual(result.delegate, "agent", text)

    def test_single_step_goal_shape_delegates_to_agent(self):
        # No sequencing connective at all: the archetype vocabulary alone
        # (clean + downloads) routes to the agent surface.
        result = process("clean up my downloads folder", file_path=self.target)
        self.assertEqual(result.verdict, "DELEGATE")
        self.assertEqual(result.delegate, "agent")

    def test_pure_settings_sequence_stays_compound(self):
        # Sequencing over settings-addressable clauses is NOT agent-shaped:
        # the compound planner composes both ops into one gated plan.
        result = process("disable blur then make animations faster",
                         file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        self.assertIsNone(result.delegate)
        tools = {e["tool"] for e in result.plan["entries"]}
        self.assertEqual(tools, {"setBlurEnabled", "setAnimationSpeed"})

    def test_simultaneous_conjunction_is_not_agent_shaped(self):
        result = process("make the bar thinner and the dock smaller",
                         file_path=self.target)
        self.assertEqual(result.verdict, "PLAN")
        self.assertIsNone(result.delegate)

    def test_note_taking_vocabulary_stays_brain(self):
        # The agent doc is deliberately disjoint from the brain doc's
        # note vocabulary: "organize my notes" is a brain request.
        result = process("organize my notes", file_path=self.target)
        self.assertEqual(result.delegate, "brain")


class CliAndWiringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.target = _write_target(self.dir)
        # Hermeticity: the inline-run tests below exercise the full
        # pipeline, which reads learned state from HOME (episodes,
        # calibration). Pin HOME to this temp dir so an ambient HOME
        # with real learned state can never change routing outcomes
        # here (this suite once flipped depending on how many `route`
        # calls the developer had made).
        import os
        self._orig_home = os.environ.get("HOME")
        os.environ["HOME"] = self._tmp.name

        def _restore_home():
            if self._orig_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = self._orig_home
        self.addCleanup(_restore_home)

    def test_hub_routes_registered(self):
        from assistant.hub import ROUTES

        for name in ("chat", "route", "cortex"):
            self.assertIn(name, ROUTES)

    def test_route_command_json(self):
        import io
        from contextlib import redirect_stdout
        from assistant.core.cli import cmd_route

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_route(["make my bar thinner", "--file", str(self.target), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["verdict"], "PLAN")
        self.assertEqual(payload["ops"][0]["tool"], "setBarScale")

    # -- inline delegate runs (routing fix, phase 1.2/1.3) ------------------
    # One phrase per DELEGATE category: each must now produce an INLINE
    # ANSWER in the same turn, not the old "try: <command>" hint. The hint
    # itself stays as the fallback when the inline run itself fails.

    def _route_capture(self, text):
        import io
        from contextlib import redirect_stdout
        from assistant.core.cli import cmd_route

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_route([text, "--file", str(self.target)])
        return code, buffer.getvalue()

    def test_inline_genius_answers_instead_of_hint(self):
        code, out = self._route_capture("what is 2+2")
        self.assertEqual(code, 0)
        self.assertIn("genius ->", out)
        self.assertNotIn("try:", out)

    def test_inline_diagnose_answers_instead_of_hint(self):
        code, out = self._route_capture("my shell crashed and nothing works")
        self.assertEqual(code, 0)
        self.assertIn("diagnostics", out)
        self.assertNotIn("try:", out)

    def test_inline_search_answers_instead_of_hint(self):
        code, out = self._route_capture("how do i find the lockscreen docs")
        self.assertEqual(code, 0)
        self.assertIn("retrieval hits", out)
        self.assertNotIn("try:", out)

    def test_inline_brain_answers_instead_of_hint(self):
        code, out = self._route_capture("help me plan my tasks")
        self.assertEqual(code, 0)
        self.assertIn("BRIEF", out)
        self.assertNotIn("try:", out)

    def test_inline_issue_answers_instead_of_hint(self):
        code, out = self._route_capture("i want to draft a bug report")
        self.assertEqual(code, 0)
        self.assertIn("DRAFT — NOT SUBMITTED", out)
        self.assertNotIn("try:", out)

    def test_inline_agent_simulates_instead_of_hint(self):
        code, out = self._route_capture(
            "clean my downloads then make the shell minimal")
        self.assertEqual(code, 0)
        self.assertIn("simulation (nothing executed)", out)
        self.assertIn("[would ASK you]", out)
        self.assertNotIn("try:", out)

    def test_inline_run_failure_keeps_the_hint_fallback(self):
        # The safety net: when the inline run itself fails, the turn shows
        # the "try:" hint instead of crashing — never silence, never a crash.
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        from assistant.core import delegate as delegate_mod
        from assistant.core.cli import cmd_route

        original = delegate_mod.RUNNERS["diagnose"]
        with mock.patch.dict(delegate_mod.RUNNERS,
                             {"diagnose": mock.Mock(side_effect=RuntimeError("boom"))}):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cmd_route(["my shell crashed and nothing works",
                                  "--file", str(self.target)])
        self.assertEqual(code, 0)  # degraded, not crashed
        self.assertIn("try: caelestia-assist diagnose", buffer.getvalue())
        self.assertIn("inline diagnose run failed", buffer.getvalue())
        self.assertIs(delegate_mod.RUNNERS.get("diagnose"), original)

    def test_inline_delegate_json_shape(self):
        import io
        from contextlib import redirect_stdout
        from assistant.core.cli import cmd_route

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_route(["what is 2+2", "--file", str(self.target), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["verdict"], "DELEGATE")
        self.assertEqual(payload["delegate"], "genius")
        self.assertIn("genius", payload)
        self.assertEqual(payload["genius"]["domain"], "math_eval")

    def test_chat_repl_inline_delegate(self):
        # The REPL path answers delegates inline too (the same generalized
        # branch, both sites kept consistent — the way genius already was).
        import io
        import sys
        from contextlib import redirect_stdout

        from assistant.core.cli import cmd_chat

        stdin = io.StringIO("what is 2+2\n/exit\n")
        original_stdin, sys.stdin = sys.stdin, stdin
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                cmd_chat(["--file", str(self.target), "--no-learn"])
        finally:
            sys.stdin = original_stdin
        self.assertIn("genius ->", buffer.getvalue())
        self.assertNotIn("try:", buffer.getvalue())

    def test_route_command_json_explain_verdict(self):
        # Regression: an EXPLAIN verdict used to leak the raw ToolSpec
        # dataclass into the JSON payload ("TypeError: Object of type
        # ToolSpec is not JSON serializable"); the explain answer now
        # carries the spec as an explicit dict (settings.explain._spec_json).
        import io
        from contextlib import redirect_stdout
        from assistant.core.cli import cmd_route

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_route(
                ["why is my launcher so large", "--file", str(self.target), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())  # must not raise
        self.assertEqual(payload["verdict"], "EXPLAIN")
        self.assertTrue(payload["explain"]["answer"])
        spec = payload["explain"]["spec"]
        self.assertIsInstance(spec, dict)
        self.assertEqual(spec["name"], "setLauncherMaxShown")
        self.assertEqual(spec["path"], "launcher.maxShown")

    def test_bridge_route_op_explain_json_clean(self):
        # Same root cause via the JSON bridge (assistant.hub api): an
        # EXPLAIN route must survive json.dumps end to end.
        from assistant.capabilities.brain.bridge import handle

        response = handle({"op": "route", "text": "why is my launcher so large",
                           "file": str(self.target)},
                          state_path=str(self.dir / "state.json"))
        json.dumps(response)  # must not raise
        self.assertEqual(response["result"]["verdict"], "EXPLAIN")
        self.assertEqual(response["result"]["explain"]["spec"]["name"],
                         "setLauncherMaxShown")

    def test_explain_spec_is_json_dict(self):
        # The settings-layer contract itself: explain() never returns the
        # raw ToolSpec, so every consumer (cortex --json, brain bridge,
        # scripts) can serialize its output as-is.
        from assistant.capabilities.settings import explain as explain_mod

        result = explain_mod.explain("why is my launcher so large", self.target)
        json.dumps(result)  # must not raise
        self.assertEqual(result["spec"]["name"], "setLauncherMaxShown")
        self.assertEqual(result["spec"]["citations"][0][0],
                         "shell/plugin/src/Caelestia/Config/launcherconfig.hpp:44")

    def test_cortex_report_command(self):
        import io
        from contextlib import redirect_stdout
        from assistant.core.cli import cmd_cortex

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cmd_cortex(["report"])
        self.assertEqual(code, 0)
        self.assertIn("cortex", buffer.getvalue())

    def test_bridge_route_op(self):
        from assistant.capabilities.brain.bridge import handle

        response = handle({"op": "route", "text": "make my bar thinner",
                           "file": str(self.target)},
                          state_path=str(self.dir / "state.json"))
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["verdict"], "PLAN")

    def test_bridge_chat_turn_round_trip(self):
        from assistant.capabilities.brain.bridge import handle

        state_path = str(self.dir / "state.json")
        first = handle({"op": "chat_turn", "text": "make the dock icons bigger",
                        "file": str(self.target)}, state_path=state_path)
        self.assertTrue(first["ok"])
        session = first["result"]["session"]
        second = handle({"op": "chat_turn", "text": "a bit smaller now",
                         "session": session, "file": str(self.target)},
                        state_path=state_path)
        self.assertTrue(second["ok"])
        self.assertIn("ellipsis", second["result"]["result"]["session_note"])

    def test_bridge_learn_feedback_persists(self):
        from assistant.capabilities.brain.bridge import handle

        state_path = self.dir / "state.json"
        state_path.write_text("{}", encoding="utf-8")
        response = handle({"op": "learn_feedback", "text": "t", "surface": "setBarScale",
                           "features": {"lex": 0.5}, "p": 0.6, "outcome": "applied",
                           "strategy": "balanced"}, state_path=str(state_path))
        self.assertTrue(response["ok"])
        persisted = json.loads(state_path.read_text())
        self.assertEqual(persisted["cortex_learn"]["model"]["examples"], 1)

    def test_chat_repl_requires_consent(self):
        import io
        import sys
        from contextlib import redirect_stdout

        from assistant.core.cli import cmd_chat

        before = self.target.read_bytes()
        stdin = io.StringIO("make my bar thinner\nn\n/exit\n")
        original_stdin, sys.stdin = sys.stdin, stdin
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                cmd_chat(["--file", str(self.target)])
        finally:
            sys.stdin = original_stdin
        self.assertEqual(self.target.read_bytes(), before)  # said n: nothing written
        self.assertNotIn("applied (backup", buffer.getvalue())

    def test_chat_repl_applies_on_yes(self):
        import io
        import sys
        from contextlib import redirect_stdout

        from assistant.core.cli import cmd_chat

        stdin = io.StringIO("make my bar thinner\ny\n/exit\n")
        original_stdin, sys.stdin = sys.stdin, stdin
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                cmd_chat(["--file", str(self.target)])
        finally:
            sys.stdin = original_stdin
        data = json.loads(self.target.read_text())
        self.assertEqual(data["bar"]["scale"], 0.9)
        self.assertIn("applied (backup", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
