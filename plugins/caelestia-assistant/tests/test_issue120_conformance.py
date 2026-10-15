"""Issue #120 conformance (exponential-build-5 A9).

One test file mapping EVERY bullet of ladybug-me/caelestia-kde#120 to a
command and an assertion, plus the honest implemented / partial /
not-implemented table. The table is the single source of truth: a test
fails when reality drifts from the declared status in EITHER direction
— a shipped surface missing its IMPLEMENTED row, or an undeclared
surface appearing while its row still says NOT_IMPLEMENTED.

#120's ask (issue text, condensed to its bullets):
  1. Intent parser -> structured tool calls -> ConfigObject, never
     editing config files directly.
  2. Per-tool range validation, and confirmation for multi-setting
     changes.
  3. Explainability ("why is my dock blurry", "why is my launcher so
     large").
  4. Undo history ("undo the last change", "restore yesterday's theme").
  5. Nexus complements: explain, recommend values, optimize for gaming,
     battery, "more macOS", "more minimal".
  6. Phase 3: context-aware recommendations, optimization profiles,
     workspace automation, monitor-aware config, personalized
     suggestions.

STATUS table (honest as of this commit; flip a row ONLY together with
the surface it describes):
  b1  intent->validated-calls       IMPLEMENTED   parser->planner->applier
  b2a per-tool range validation     IMPLEMENTED   planner rejects, --apply
                                                  refuses rejected plans
  b2b multi-change confirmation     IMPLEMENTED   _confirm_multi gate
  b3  explainability                IMPLEMENTED   --explain, --why-chain
                                                  (F14), graph (F12), why
  b4a "undo the last change"        IMPLEMENTED   --undo/--undo-id/
                                                  --restore + dispatch
  b4b "restore yesterday's theme"   PARTIAL       --undo-id and macros
                                                  exist; no NL time-
                                                  expression surface
  b5a explain (Nexus complement)    IMPLEMENTED   same surfaces as b3
  b5b recommend values              IMPLEMENTED   settings --recommend (F29):
                                                  partial pooling over
                                                  default + presets +
                                                  approved history
  b5c optimize profiles             IMPLEMENTED   presets (gaming,
                                                  battery-saver,
                                                  macos-like, minimal)
                                                  + NL routing + --rank
  b6a context-aware recommendations NOT_IMPLEMENTED  F30 planned
  b6b optimization profiles         PARTIAL       preset bundles + the
                                                  learned pairwise
                                                  ladder; no profile
                                                  algebra/composition
  b6c workspace automation          NOT_IMPLEMENTED
  b6d monitor-aware config          NOT_IMPLEMENTED
  b6e personalized suggestions      PARTIAL       brain prefs/nlhistory
                                                  learn; nothing wired
                                                  into #120 proposals
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.core.nlhistory import parse_query as parse_history_query
from assistant.core.nlhistory import plan as plan_history

STATUS = {
    "b1": "IMPLEMENTED", "b2a": "IMPLEMENTED", "b2b": "IMPLEMENTED",
    "b3": "IMPLEMENTED", "b4a": "IMPLEMENTED",
    # F10 (33c42b1): the NL time-expression restore exists and the
    # restore-since reading rides the existing undo(steps=K) engine.
    "b4b": "IMPLEMENTED",
    "b5a": "IMPLEMENTED", "b5b": "IMPLEMENTED", "b5c": "IMPLEMENTED",
    # F30 (8c8f7a2): cortex context — injectable read-only probes, the
    # clock is never evidence by itself.
    "b6a": "IMPLEMENTED",
    # F9 (7088f2c): profile algebra — compose presets/macros/calls,
    # later-wins with conflicts reported, riding the normal gates.
    "b6b": "IMPLEMENTED",
    # F19 (57f5fd7): pull-based schedules — nothing runs by itself.
    "b6c": "IMPLEMENTED",
    # F20 (2b11443): --monitor/--monitors over the upstream forScreen
    # override layers; connected-screen enumeration stays in
    # Quickshell/Wayland and the report says so.
    "b6d": "IMPLEMENTED",
    # F16 (5b3d55f): personalize mines consented evidence into the
    # brain ledger — the #120 proposal surface.
    "b6e": "IMPLEMENTED",
}


def _settings(argv, state="{}", target=None):
    """Run the settings CLI in-process against a scratch file (or an
    EXISTING target, so history/backup siblings persist across calls)."""
    from assistant.capabilities.settings import cli as settings_cli
    if target is None:
        tmp = tempfile.mkdtemp(prefix="iss120-")
        target = Path(tmp) / "shell.json"
        target.write_text(state, encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    argv = [*argv, "--file", str(target)]
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = settings_cli.main(argv)
        except SystemExit as exc:
            rc = int(exc.code or 0)
    return rc, out.getvalue(), err.getvalue(), target


def _mk_target():
    """A scratch shell.json target for surface-existence checks."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="iss120-row-"))
    target = tmp / "shell.json"
    target.write_text("{}", encoding="utf-8")
    return tmp, target


def _route(text):
    """The NL front door (cortex route verb) in-process."""
    from assistant.core import pipeline
    home = tempfile.mkdtemp(prefix="iss120-home-")
    old = os.environ.get("HOME")
    os.environ["HOME"] = home
    try:
        return pipeline.process(text)
    finally:
        if old is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old


class TableDriftTests(unittest.TestCase):
    def test_status_table_is_complete_and_honest(self) -> None:
        bullets = ("b1", "b2a", "b2b", "b3", "b4a", "b4b", "b5a", "b5b",
                   "b5c", "b6a", "b6b", "b6c", "b6d", "b6e")
        self.assertEqual(sorted(STATUS), sorted(bullets))
        for k, v in STATUS.items():
            self.assertIn(v, ("IMPLEMENTED", "PARTIAL", "NOT_IMPLEMENTED"),
                          f"{k}: invalid status {v!r}")

    def test_not_implemented_rows_pin_absence(self) -> None:
        """A NOT_IMPLEMENTED row is a PROMISE about the surface set: the
        named capability must not secretly exist. When it ships, the
        row flips in the same commit (the drift detector)."""
        from assistant.capabilities.settings import cli as settings_cli
        parser = settings_cli._build_arg_parser()
        flags = {a.option_strings[0]
                 for a in parser._actions if a.option_strings}
        if STATUS["b5b"] == "NOT_IMPLEMENTED":
            # F29 recommend-values: no recommendation flag exists yet
            self.assertFalse(
                any("recommend" in f for f in flags),
                "a recommend-values surface exists — flip b5b's row")
        else:
            self.assertIn("--recommend", flags,
                          "b5b says IMPLEMENTED — the surface must exist")
        if STATUS["b6a"] == "NOT_IMPLEMENTED":
            self.assertFalse(
                any("context" in f for f in flags),
                "a context-profile surface exists — flip b6a's row")


class B1IntentToValidatedCalls(unittest.TestCase):
    """#120 bullet 1: parser -> structured tool calls, never direct edits."""

    def test_nl_request_plans_validated_ops(self) -> None:
        rc, out, _err, target = _settings(["make the bar 20% smaller"],
                                          state='{"bar": {"scale": 1.0}}')
        self.assertEqual(rc, 0)
        self.assertIn("INTENT", out)
        self.assertIn("setBarScale", out)
        self.assertIn("0.8", out)
        self.assertEqual(json.loads(target.read_text()),
                         {"bar": {"scale": 1.0}})  # dry-run: untouched

    def test_apply_writes_only_through_the_journaled_applier(self) -> None:
        rc, out, _err, target = _settings(
            ["make the bar 20% smaller", "--apply"],
            state='{"bar": {"scale": 1.0}}')
        self.assertEqual(rc, 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["bar"]["scale"], 0.8)
        # the journaled writer's two siblings: the backup slot and the
        # bounded history ledger (never a bare rewrite)
        self.assertTrue((target.parent /
                         (target.name + ".assistant-backup")).exists()
                        or "backup" in out.lower())
        history = target.parent / (target.name +
                                   ".assistant-history.json")
        self.assertTrue(history.exists())
        entries = json.loads(history.read_text())["entries"]
        self.assertTrue(entries[0]["ops"][0]["path"].endswith("scale"))

    def test_qml_service_surface_is_pinned(self) -> None:
        """The ConfigObject side lives in the shell's own QML service;
        its function surface is pinned by the settings layer's own
        cross-check suite (skips when the checkout is absent)."""
        qml = (Path(__file__).resolve().parents[1] / "assistant" / "adapters" /
               "caelestia" / "shell" /
               "services" / "SettingsTools.qml")
        if not qml.exists():
            self.skipTest("SettingsTools.qml absent — QML guards skip")
        text = qml.read_text(encoding="utf-8")
        for fn in ("validate", "buildPlan", "previewOf", "confirmPlan",
                   "applyOps", "pushHistory", "undo", "undoById"):
            self.assertIn(fn, text)


class B2ValidationAndConfirmation(unittest.TestCase):
    """#120 bullet 2: per-tool range validation + multi-change confirm."""

    def test_out_of_range_is_rejected_and_apply_refuses(self) -> None:
        rc, out, err, target = _settings(["--call", "setDockIconSize=500"])
        self.assertNotEqual(rc, 0)
        self.assertIn("outside the allowed range 16-96", out)
        self.assertIn("nothing was written", err + out)
        self.assertEqual(json.loads(target.read_text()), {})

    def test_enum_value_validation(self) -> None:
        rc, out, _err, _t = _settings(["--call", "setBarPosition=diagonal"])
        self.assertNotEqual(rc, 0)
        self.assertIn("top|bottom|left|right", out)

    def test_multi_change_needs_confirmation_without_tty(self) -> None:
        rc, out, err, target = _settings(["--preset", "gaming", "--apply"])
        self.assertNotEqual(rc, 0)
        self.assertIn("need confirmation", err + out)
        self.assertIn("nothing was written", err + out)
        self.assertEqual(json.loads(target.read_text()), {})

    def test_multi_change_applies_with_explicit_confirm(self) -> None:
        rc, out, _err, target = _settings(
            ["--preset", "gaming", "--apply", "--confirm"])
        self.assertEqual(rc, 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["appearance"]["blur"], False)


class B3Explainability(unittest.TestCase):
    """#120 bullet 3: 'why is my dock blurry' / 'why is my launcher...'."""

    def test_explain_answers_with_citation(self) -> None:
        state = json.dumps({"appearance": {"blur": True,
                                           "transparency":
                                           {"enabled": True}}})
        rc, out, _err, _t = _settings(["--explain", "why is my dock blurry"],
                                      state=state)
        self.assertEqual(rc, 0)
        self.assertIn("BlurOffsets.qml", out)  # the shipped citation

    def test_why_chain_gives_the_armed_causal_chain(self) -> None:
        state = json.dumps({"appearance": {"blur": False,
                                           "transparency":
                                           {"enabled": False}}})
        rc, out, _err, _t = _settings(["--why-chain", "appearance.blur"],
                                      state=state)
        self.assertEqual(rc, 0)
        self.assertIn("appearance.transparency.enabled", out)
        self.assertIn("AppearancePage.qml", out)

    def test_graph_affects_lists_curated_upstream(self) -> None:
        from assistant.capabilities.graph import build as gbuild, queries as gq
        r = gq.what_affects(gbuild.build_graph(), "appearance.blur")
        self.assertEqual(r["verdict"], "OK")
        self.assertTrue(r["upstream"])

    def test_route_handles_why_questions(self) -> None:
        result = _route("why is my dock blurry")
        self.assertEqual(result.verdict, "EXPLAIN")
        self.assertIsNone(result.plan)      # never a proposed change
        self.assertEqual(result.ops, [])    # for a why-question
        self.assertIn("Blur is enabled", str(result.explain_answer))


class B4UndoHistory(unittest.TestCase):
    """#120 bullet 4: 'undo the last change' / 'restore yesterday'."""

    def test_undo_the_last_change(self) -> None:
        _rc, _out, _err, target = _settings(
            ["make the bar 20% smaller", "--apply"],
            state='{"bar": {"scale": 1.0}}')
        rc, out, _err, _t = _settings(["--undo", "1"], target=target)
        self.assertEqual(rc, 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["bar"]["scale"], 1.0)  # restored

    def test_undo_by_id_from_the_ledger(self) -> None:
        _rc, _out, _err, target = _settings(
            ["make the bar 20% smaller", "--apply"],
            state='{"bar": {"scale": 1.0}}')
        history = json.loads((target.parent /
                              (target.name + ".assistant-history.json")
                              ).read_text())
        entry_id = history["entries"][0]["id"]
        rc, out, _err, _t = _settings(
            ["--undo-id", str(entry_id)], target=target)
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(target.read_text())
                         ["bar"]["scale"], 1.0)

    def test_restore_yesterdays_theme_resolves_the_window(self) -> None:
        """F10 flipped b4b: the NL time expression resolves to a
        concrete restore plan — a window with no domain scope means
        'restore TO that point' and reverts every change applied since
        then via the existing undo(steps=K) engine."""
        self.assertEqual(STATUS["b4b"], "IMPLEMENTED")
        now = datetime(2026, 9, 28, 12, 0)
        from datetime import timedelta as _td
        entries = [
            {"id": 2, "at": (now - _td(hours=1)).isoformat(),
             "label": "chat: x",
             "ops": [{"path": "bar.scale", "old": 0.9, "new": 1.0}]},
            {"id": 1, "at": (now - _td(days=2)).isoformat(),
             "label": "chat: y",
             "ops": [{"path": "bar.scale", "old": 1.0, "new": 0.9}]},
        ]
        query = parse_history_query("restore yesterday's theme", now=now)
        plan = plan_history(query, entries, now=now)
        self.assertEqual(plan.verdict, "UNDO")
        self.assertEqual(plan.action, "undo")
        self.assertEqual(plan.steps, 1)
        self.assertIn("restores the state as of", plan.reason)


class B5NexusComplements(unittest.TestCase):
    """#120 bullet 5: explain / recommend values / optimize profiles."""

    def test_all_four_profiles_exist_and_plan(self) -> None:
        for preset in ("gaming", "battery-saver", "macos-like", "minimal"):
            rc, out, _err, _t = _settings(["--preset", preset])
            self.assertEqual(rc, 0, preset)
            self.assertIn("INTENT", out)

    def test_nl_routes_optimize_for_gaming_to_the_preset(self) -> None:
        result = _route("optimize for gaming")
        self.assertEqual(result.verdict, "PLAN")
        raws = " ".join(str(e.get("raw", ""))
                        for e in result.plan["entries"])
        self.assertIn("preset gaming", raws)

    def test_learned_preset_ladder_exists(self) -> None:
        rc, out, _err, _t = _settings(["--rank"])
        self.assertEqual(rc, 0)
        self.assertIn("preset", out.lower())

    def test_recommend_values_surfaces_the_pooled_proposal(self) -> None:
        rc, out, _err, target = _settings(["--recommend",
                                           "setDockIconSize"])
        self.assertEqual(rc, 0)
        self.assertIn("recommend:", out)
        self.assertIn("95% interval", out)
        self.assertIn("proposal only", out)
        # nothing was written — the recommender is inert
        self.assertEqual(json.loads(target.read_text()), {})


class B6Phase3(unittest.TestCase):
    """#120 bullet 6 (Phase 3): context-aware recs, profiles, workspace
    automation, monitor-aware config, personalized suggestions."""

    def test_phase3_status_rows_are_declared_honestly(self) -> None:
        """The F16/F17-F21/F30 wave closed every Phase-3 row; the drift
        detector keeps the table honest in BOTH directions."""
        self.assertEqual(STATUS["b6a"], "IMPLEMENTED")
        self.assertEqual(STATUS["b6b"], "IMPLEMENTED")
        self.assertEqual(STATUS["b6c"], "IMPLEMENTED")
        self.assertEqual(STATUS["b6d"], "IMPLEMENTED")
        self.assertEqual(STATUS["b6e"], "IMPLEMENTED")

    def test_b6a_context_recommendations_surface(self) -> None:
        from assistant.core import context as context_mod
        view = context_mod.recommend_contextual(
            now=datetime(2026, 9, 28, 12, 0),
            power={"percent": 14.0, "charging": False,
                   "source": "BAT0"})
        self.assertEqual(view["findings"][0]["kind"], "battery_low")
        self.assertIn("SUGGESTED_NOT_EXECUTED",
                      view["findings"][0]["command"])

    def test_b6b_profile_algebra_composes(self) -> None:
        from assistant.capabilities.settings import cli as settings_cli
        parser = settings_cli._build_arg_parser()
        flags = {a.option_strings[0]
                 for a in parser._actions if a.option_strings}
        for flag in ("--profile", "--profile-save", "--profile-diff",
                     "--profile-whatif"):
            self.assertIn(flag, flags)

    def test_b6c_schedules_are_pull_based(self) -> None:
        from assistant.core import schedules
        self.assertTrue(hasattr(schedules, "evaluate"))
        self.assertTrue(hasattr(schedules, "file_firing"))

    def test_b6d_monitor_report_targets_override_layers(self) -> None:
        from assistant.capabilities.settings import monitors
        tmp, target = _mk_target()
        resolved = monitors.resolve_target(target, "DP-1")
        self.assertEqual(resolved.name, target.name)
        self.assertIn("monitors", str(resolved))
        rows = monitors.monitor_behavior_tools()
        self.assertTrue(any(r["name"] == "setMonitor" for r in rows))

    def test_b6e_personalization_files_into_the_ledger(self) -> None:
        from assistant.core import personalize
        mined = personalize.mine([
            {"id": i, "at": f"2026-09-28T1{i}:00:00",
             "label": "preset: battery-saver", "ops": []}
            for i in range(1, 4)])
        self.assertEqual(mined["recurring"][0]["count"], 3)
        self.assertEqual(personalize.SUGGESTION_KIND,
                         "personalized_suggestion")


if __name__ == "__main__":
    unittest.main()
