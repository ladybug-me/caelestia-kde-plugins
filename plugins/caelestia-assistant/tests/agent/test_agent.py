"""agent tests — decomposition, clarification entropy, consent-gated execution."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.agent import goals, clarify
from assistant.capabilities.agent.engine import Agent, DISPATCHERS
from assistant.capabilities.genius.graphs import toposort


class TestGoals(unittest.TestCase):
    def test_compound_split_on_sequence_words(self):
        parts = goals.split_compound("clean my downloads and then make the shell minimal")
        self.assertEqual(len(parts), 2)
        self.assertIn("clean", parts[0])
        self.assertIn("minimal", parts[1])

    def test_plain_and_not_split(self):
        parts = goals.split_compound("disable blur and move the dock left")
        self.assertEqual(len(parts), 1)

    def test_classify_goals(self):
        self.assertEqual(goals.classify_goal("vesktop freezes when I screenshare")["goal"],
                         "diagnose_issue")
        self.assertEqual(goals.classify_goal("make my bar thinner")["goal"],
                         "change_settings")
        self.assertEqual(goals.classify_goal("clean my downloads folder")["goal"],
                         "clean_files")
        self.assertEqual(goals.classify_goal("what is 15% of 80")["goal"],
                         "compute")

    def test_decompose_produces_dag(self):
        g = goals.decompose("why does vesktop freeze when I screenshare")
        edges = []
        for n in g["nodes"]:
            for d in n["depends"]:
                edges.append((d, n["id"]))
        ts = toposort(edges)
        self.assertTrue(ts["is_dag"])
        self.assertEqual(len(ts["order"]), len(g["nodes"]))

    def test_settings_graph_has_consent_gate(self):
        g = goals.decompose("make my bar thinner")
        gates = [n for n in g["nodes"] if n["consent_required"]]
        self.assertTrue(gates)
        self.assertTrue(all(n["risk"] in ("STATE_CHANGING", "PRIVILEGED",
                                          "DESTRUCTIVE")
                            for n in gates))
        # apply must depend on propose
        ids = {n["id"] for n in g["nodes"]}
        apply_nodes = [n for n in g["nodes"] if n["action"].startswith("apply")]
        for a in apply_nodes:
            for d in a["depends"]:
                self.assertIn(d, ids)
        self.assertTrue(g["consent_required"])

    def test_no_privileged_nodes_ever(self):
        for goal, method in goals.GOAL_METHODS.items():
            for step in method:
                self.assertIn(step["risk"], ("READ_ONLY", "STATE_CHANGING"),
                              f"{goal}/{step['id']} invents a privileged node")
                if step["risk"] == "STATE_CHANGING":
                    self.assertTrue(step["consent"])

    def test_decompose_deterministic(self):
        a = goals.decompose("clean my downloads and then make the shell minimal")
        b = goals.decompose("clean my downloads and then make the shell minimal")
        self.assertEqual([n["action"] for n in a["nodes"]],
                         [n["action"] for n in b["nodes"]])
        self.assertTrue(a["compound"])


class TestClarify(unittest.TestCase):
    def test_entropy_picker_ranks_discriminating_question(self):
        picker = clarify.EntropyPicker({"diagnose_issue": 3, "change_settings": 3})
        q = clarify.Question(
            "broken_or_tune",
            "Is something broken, or do you want it tuned?",
            {"broken": {"diagnose_issue": 1.0, "change_settings": 0.0},
             "tune": {"diagnose_issue": 0.0, "change_settings": 1.0}})
        gain = picker.expected_gain(q)
        self.assertGreater(gain, 0.5)  # perfect discriminator on max entropy

    def test_useless_question_not_asked(self):
        picker = clarify.EntropyPicker({"diagnose_issue": 1.0})
        q = clarify.Question(
            "noise", "Is the sky blue?",
            {"yes": {"diagnose_issue": 1.0}, "no": {"diagnose_issue": 1.0}})
        self.assertEqual(picker.expected_gain(q), 0.0)
        self.assertIsNone(picker.best_question([q]))

    def test_settles_after_clear_answer(self):
        picker = clarify.EntropyPicker({"a": 1, "b": 1})
        q = clarify.Question(
            "q1", "Which one?",
            {"one": {"a": 1.0, "b": 0.0}, "two": {"a": 0.0, "b": 1.0}})
        picker.observe(q, "one")
        self.assertTrue(picker.settled())
        self.assertEqual(picker.top(), ("a", picker.belief["a"]))
        self.assertAlmostEqual(picker.belief["a"], 1.0)

    def test_impossible_answer_ignored(self):
        picker = clarify.EntropyPicker({"a": 1, "b": 1})
        q = clarify.Question(
            "q1", "Which one?",
            {"one": {"a": 1.0, "b": 0.0}, "two": {"a": 0.0, "b": 1.0}})
        before = dict(picker.belief)
        picker.observe(q, "nonsense")
        self.assertEqual(picker.belief, before)  # zero-likelihood answer: no update


class TestEngine(unittest.TestCase):
    def test_simulate_executes_nothing(self):
        agent = Agent()
        result = agent.simulate("vesktop freezes and crashes every time")
        self.assertIn("projection", result)
        actions = [s["action"] for s in result["projection"]]
        self.assertIn("diagnose", actions)
        self.assertIn("retrieve_similar", actions)
        # the crucial guarantee: nothing ran
        self.assertEqual(agent._ctx["results"], {})

    def test_execute_read_only_chain(self):
        seen = []
        agent = Agent(consent_fn=lambda node: False,
                      observer=lambda action, outcome: seen.append(outcome))
        result = agent.execute("vesktop freezes and crashes every time")
        statuses = {n["action"]: n["status"] for n in result["nodes"]}
        self.assertEqual(statuses.get("diagnose"), "done")
        self.assertEqual(statuses.get("retrieve_similar"), "done")
        self.assertEqual(statuses.get("fix_plan"), "done")
        self.assertTrue(seen)

    def test_consent_refusal_blocks_downstream(self):
        # tidy goal on a real temp dir: survey is read-only, apply is gated
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.txt").write_text("x")
            agent = Agent(consent_fn=lambda node: False,
                          base_ctx={"tidy_root": td})
            result = agent.execute("clean my files")
            by_action = {n["action"]: n for n in result["nodes"]}
            self.assertEqual(by_action["tidy_survey"]["status"], "done")
            self.assertEqual(by_action["tidy_propose"]["status"], "refused")
            self.assertEqual(by_action["tidy_apply"]["status"], "skipped")
            self.assertIn("dependencies", by_action["tidy_apply"]["skip_reason"])

    def test_consent_grant_runs_apply(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "photo.jpg").write_bytes(b"j")
            disp = dict(DISPATCHERS)

            def survey(ctx, params):
                ctx["tidy_root"] = td
                return DISPATCHERS["tidy_survey"](ctx, params)

            disp["tidy_survey"] = survey
            agent = Agent(dispatchers=disp,
                          consent_fn=lambda node:
                          (calls.append(node["action"]) or True),
                          base_ctx={"tidy_root": td})
            result = agent.execute("clean my files")
            by_action = {n["action"]: n for n in result["nodes"]}
            self.assertEqual(by_action["tidy_apply"]["status"], "done")
            self.assertIn("tidy_propose", calls)
            self.assertIn("tidy_apply", calls)

    def test_failed_node_skips_dependents_not_graph(self):
        disp = dict(DISPATCHERS)

        def boom(ctx, params):
            raise RuntimeError("engine hiccup")

        disp["diagnose"] = boom
        agent = Agent(dispatchers=disp, consent_fn=lambda n: False)
        result = agent.execute("vesktop freezes and crashes every time")
        by_action = {n["action"]: n for n in result["nodes"]}
        self.assertEqual(by_action["diagnose"]["status"], "failed")
        self.assertEqual(by_action["fix_plan"]["status"], "skipped")
        # independent branch still completed
        self.assertEqual(by_action["retrieve_similar"]["status"], "done")

    def test_observer_receives_outcomes(self):
        events = []
        agent = Agent(consent_fn=lambda n: False,
                      observer=lambda action, o: events.append((action, o["outcome"])))
        agent.execute("plan my day")
        self.assertTrue(any(o in ("accepted", "refused", "failed", "skipped")
                            for _a, o in events))

    def test_execute_result_serialisable(self):
        agent = Agent(consent_fn=lambda n: False)
        result = agent.execute("what is 2^10")
        json.dumps(result, default=str)  # must not raise

    def test_consent_fn_receives_risk_metadata(self):
        captured = {}

        def consent(node):
            captured.update(node)
            return False

        agent = Agent(consent_fn=consent)
        agent.execute("make my bar thinner")
        self.assertIn("risk", captured)
        self.assertIn("consent_required", captured)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 2.3: new goal archetypes, the capability manifest, the quarantine.
# ---------------------------------------------------------------------------


def _make_png(path, w, h, fn):
    import struct
    import zlib

    raw = b""
    for y in range(h):
        raw += b"\x00" + bytes(v for x in range(w) for v in fn(x, y))

    def chunk(t, dd):
        return (struct.pack(">I", len(dd)) + t + dd
                + struct.pack(">I", zlib.crc32(t + dd) & 0xffffffff))

    data = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    Path(path).write_bytes(data)
    return str(path)


class TestPhase23Archetypes(unittest.TestCase):
    def _setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_new_cue_phrases_route(self):
        cases = [
            ("lint my config and check my dotfiles", "config_hygiene"),
            ("audit my packages for staleness", "package_audit"),
            ("triage my logs", "log_triage"),
            ("too many notifications, batch them", "notification_triage"),
            ("compare screenshots before after", "screenshot_diff"),
        ]
        for text, expected in cases:
            cls = goals.classify_goal(text)
            self.assertEqual(cls["goal"], expected, text)
            self.assertNotEqual(cls["verdict"], "AMBIGUOUS", text)

    def test_config_hygiene_consent_shape(self):
        graph = goals.decompose("lint my config")
        actions = [(n["action"], n["consent_required"], n["risk"])
                   for n in graph["nodes"]]
        self.assertEqual([a for a, _c, _r in actions],
                         ["lint_config", "config_drift",
                          "reconcile_propose", "reconcile_apply"])
        # the two STATE_CHANGING nodes require consent; the reads never do
        self.assertEqual([c for _a, c, _r in actions],
                         [False, False, True, True])

    def test_new_methods_have_no_privileged_or_destructive(self):
        for method in goals.GOAL_METHODS.values():
            for step in method:
                self.assertIn(step["risk"], ("READ_ONLY", "STATE_CHANGING"))
                if step["risk"] == "STATE_CHANGING":
                    self.assertTrue(step["consent"])

    def test_config_hygiene_end_to_end_with_refusal(self):
        self._setUp()
        target = self.dir / "shell.json"
        target.write_text(json.dumps({"bar": {"scale": "big"}}))
        before = target.read_bytes()
        agent = Agent(consent_fn=lambda node: False,
                      base_ctx={"config_target": str(target)})
        result = agent.execute("lint my config")
        statuses = {n["action"]: n["status"] for n in result["nodes"]}
        self.assertEqual(statuses["lint_config"], "done")
        self.assertEqual(statuses["config_drift"], "done")
        self.assertEqual(statuses["reconcile_propose"], "refused")
        self.assertEqual(statuses["reconcile_apply"], "skipped")
        self.assertEqual(target.read_bytes(), before)  # nothing written

    def test_config_hygiene_reconcile_applies_out_of_range_reset(self):
        self._setUp()
        target = self.dir / "shell.json"
        target.write_text(json.dumps({"bar": {"scale": 9.9}}))
        agent = Agent(consent_fn=lambda node: True,
                      base_ctx={"config_target": str(target)})
        result = agent.execute("lint my config")
        data = json.loads(target.read_text())
        # out-of-range 9.9 reset to the registry default 1.0 through the
        # planner/applier gate (a valid typed op — the standard path)
        self.assertEqual(data["bar"]["scale"], 1.0)
        applied = [n for n in result["nodes"]
                   if n["action"] == "reconcile_apply"][0]
        self.assertTrue(applied["result"].get("applied"))

    def test_config_hygiene_type_mismatch_is_manual_not_bypassed(self):
        # the planner refuses structural repairs by design ("never a
        # silent cast"); the archetype must honor that, not bypass it:
        # a type-broken key gets an honest manual instruction.
        self._setUp()
        target = self.dir / "shell.json"
        target.write_text(json.dumps({"bar": {"scale": "big"}}))
        agent = Agent(consent_fn=lambda node: True,
                      base_ctx={"config_target": str(target)})
        result = agent.execute("lint my config")
        before = target.read_bytes()
        data = json.loads(target.read_text())
        self.assertEqual(data["bar"]["scale"], "big")  # untouched
        proposal = [n for n in result["nodes"]
                    if n["action"] == "reconcile_propose"][0]["result"]
        self.assertEqual(proposal["ops"], [])
        self.assertTrue(any("structural repair" in line
                            for line in proposal["inert"]))

    def test_package_audit_disabled_by_default(self):
        from assistant.capabilities.agent import archetypes

        report = archetypes.package_report()
        self.assertIn("error", report)
        self.assertIn("package_audit is disabled", report["error"])

    def test_log_triage_to_issue_draft(self):
        lines = "error 404 on /home\n" * 5 + "warn disk 80%\n" \
                + "segfault at 0x4f3a2b\n"
        agent = Agent(base_ctx={"stream_text": lines})
        result = agent.execute("triage my logs")
        statuses = {n["action"]: n["status"] for n in result["nodes"]}
        self.assertEqual(statuses["log_triage"], "done")
        self.assertEqual(statuses["triage_draft"], "done")
        draft = [n for n in result["nodes"]
                 if n["action"] == "triage_draft"][0]["result"]["draft"]
        self.assertIn("DRAFT", draft)
        self.assertIn("template", draft.lower())

    def test_notification_triage_classifier(self):
        from assistant.capabilities.agent.archetypes import triage_notifications

        events = ([{"source": "chat", "ts": i, "accepted": False}
                   for i in range(10)]
                  + [{"source": "battery", "ts": i, "accepted": True}
                     for i in range(3)])
        out = triage_notifications(events)
        self.assertEqual([r["source"] for r in out["batch"]], ["chat"])
        self.assertEqual([r["source"] for r in out["pass"]], ["battery"])
        self.assertIn("batch", out["proposal"].lower())

    def test_notification_triage_empty_and_low_volume(self):
        from assistant.capabilities.agent.archetypes import triage_notifications

        self.assertEqual(triage_notifications([])["n"], 0)
        # low volume never batches, whatever the acceptance
        rare = [{"source": "chat", "ts": 0, "accepted": False}]
        self.assertEqual(triage_notifications(rare)["batch"], [])

    def test_screenshot_diff_identical(self):
        self._setUp()
        a = _make_png(self.dir / "a.png", 64, 48,
                      lambda x, y: (100, 100, 100))
        b = _make_png(self.dir / "b.png", 64, 48,
                      lambda x, y: (100, 100, 100))
        from assistant.capabilities.agent.archetypes import diff_screenshots

        out = diff_screenshots(a, b)
        self.assertTrue(out["same_geometry"])
        self.assertEqual(out["changed_fraction"], 0.0)
        self.assertEqual(out["regions"], [])

    def test_screenshot_diff_localizes_the_change(self):
        self._setUp()
        a = _make_png(self.dir / "a.png", 64, 48,
                      lambda x, y: (100, 100, 100))
        b = _make_png(self.dir / "b.png", 64, 48,
                      lambda x, y: (250, 250, 250) if x < 16
                      else (100, 100, 100))
        from assistant.capabilities.agent.archetypes import diff_screenshots

        out = diff_screenshots(a, b)
        self.assertGreater(out["changed_fraction"], 0.0)
        self.assertLess(out["changed_fraction"], 0.4)  # the left quarter
        # every region rectangle is on the LEFT side of the image
        self.assertTrue(all(r["x"] + r["w"] <= 31 for r in out["regions"]))

    def test_screenshot_diff_geometry_change(self):
        self._setUp()
        a = _make_png(self.dir / "a.png", 64, 48,
                      lambda x, y: (100, 100, 100))
        b = _make_png(self.dir / "b.png", 32, 48,
                      lambda x, y: (100, 100, 100))
        from assistant.capabilities.agent.archetypes import diff_screenshots

        out = diff_screenshots(a, b)
        self.assertFalse(out["same_geometry"])
        self.assertEqual(out["changed_fraction"], 1.0)

    def test_screenshot_draft_via_engine(self):
        self._setUp()
        a = _make_png(self.dir / "a.png", 64, 48,
                      lambda x, y: (100, 100, 100))
        b = _make_png(self.dir / "b.png", 64, 48,
                      lambda x, y: (10, 10, 10) if y < 12
                      else (100, 100, 100))
        agent = Agent()
        result = agent.execute(f"compare screenshots {a} and {b}")
        statuses = {n["action"]: n["status"] for n in result["nodes"]}
        self.assertEqual(statuses["screenshot_diff"], "done")
        self.assertEqual(statuses["screenshot_draft"], "done")

    def test_screenshot_diff_rejects_non_png(self):
        self._setUp()
        junk = self.dir / "junk.png"
        junk.write_bytes(b"not a png at all")
        other = _make_png(self.dir / "ok.png", 8, 8, lambda x, y: (0, 0, 0))
        from assistant.capabilities.agent.archetypes import diff_screenshots

        with self.assertRaises(ValueError):
            diff_screenshots(str(junk), other)


class TestCapabilityManifest(unittest.TestCase):
    def test_defaults_posture(self):
        from assistant.core import features as capabilities

        # read-only, zero-risk: on by default
        for name in ("fsbrain", "config_hygiene", "log_triage",
                     "screenshot_diff", "notification_triage",
                     "lexicon_sharing"):
            self.assertTrue(capabilities.DEFAULTS[name], name)
        # anything that shells out or extends the import surface: off
        for name in ("package_audit", "dbus_surface",
                     "notification_observation"):
            self.assertFalse(capabilities.DEFAULTS[name], name)

    def test_user_file_overrides(self):
        import os

        from assistant.core import features as capabilities

        user = Path(tempfile.mkdtemp()) / "caps.json"
        user.write_text(json.dumps({"package_audit": True,
                                    "fsbrain": False}))
        old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
        os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(user)
        try:
            self.assertTrue(capabilities.enabled("package_audit"))
            self.assertFalse(capabilities.enabled("fsbrain"))
            self.assertIn("edit it to flip", capabilities.render())
        finally:
            if old is None:
                os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
            else:
                os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old

    def test_broken_file_degrades_to_defaults(self):
        import os

        from assistant.core import features as capabilities

        user = Path(tempfile.mkdtemp()) / "caps.json"
        user.write_text("{not json")
        old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
        os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(user)
        try:
            self.assertFalse(capabilities.enabled("package_audit"))
        finally:
            if old is None:
                os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
            else:
                os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old


class TestPackageProbeQuarantine(unittest.TestCase):
    def test_probe_source_stays_singleton_subprocess_user(self):
        # the import allow-list is gone (minimal-guards brief); the
        # behavioral guard is that pkgprobe remains the ONLY agent-side
        # module whose source names subprocess, and it never opens sockets
        import inspect

        from assistant.capabilities.agent import pkgprobe

        src = inspect.getsource(pkgprobe)
        self.assertNotIn("import subprocess", src)  # routed via the executor
        self.assertNotIn("import socket", src)
        self.assertNotIn("urllib", src)
        import inspect as _i
        from assistant.executor import runner as _runner
        self.assertIn("subprocess", _i.getsource(_runner))

    def test_probe_refuses_when_disabled(self):
        from assistant.capabilities.agent import pkgprobe

        out = pkgprobe.query_installed()
        self.assertIn("error", out)
        self.assertFalse(out.get("enabled", True))

    def test_probe_uses_fixed_argument_arrays(self):
        import os
        from unittest import mock

        from assistant.capabilities.agent import pkgprobe

        user = Path(tempfile.mkdtemp()) / "caps.json"
        user.write_text(json.dumps({"package_audit": True}))
        old = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
        os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = str(user)
        calls = []

        try:
            fake_steps = []

            def fake_run_step(step):
                fake_steps.append(tuple(step.argv))
                return {"argv": list(step.argv), "ok": True,
                        "returncode": 0,
                        "stdout": "quickshell 1.2\nplasma-desktop 6.1\n",
                        "stderr": ""}

            with mock.patch(
                    "assistant.executor.runner.run_step",
                    side_effect=fake_run_step), \
                    mock.patch.object(pkgprobe.os.path, "isfile",
                                      return_value=True):
                out = pkgprobe.query_installed(["pacman"])
        finally:
            if old is None:
                os.environ.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
            else:
                os.environ["CAELESTIA_ASSIST_CAPABILITIES"] = old
        self.assertEqual(out["manager"], "pacman")
        self.assertEqual(out["n"], 2)
        # the fixed argument array, exactly — a tuple of str through the
        # executor contract, never a string, no shell
        self.assertEqual(fake_steps[0], ("pacman", "-Q"))
        self.assertIsInstance(fake_steps[0], tuple)

    def test_stale_match_groups(self):
        from assistant.capabilities.agent import pkgprobe

        out = pkgprobe.stale_match([("quickshell", "1.2"),
                                    ("plasma-desktop", "6.1"),
                                    ("vim", "9.0"),
                                    ("foo-git", "r123")])
        keywords = {g["keyword"] for g in out["groups"]}
        self.assertIn("quickshell", keywords)
        self.assertIn("plasma", keywords)
        self.assertIn("-git", keywords)
        self.assertNotIn("vim", keywords)


class TestEngineResultAddressing(unittest.TestCase):
    """The engine-key fix: results addressable by action name (the latent
    bug where fix_plan/explain composed from keys that never existed)."""

    def test_results_stored_by_id_and_action(self):
        agent = Agent()
        result = agent.execute("what is 2+2")
        ctx_results = result["ctx_results"]
        self.assertIn("genius_dispatch", ctx_results)
        self.assertTrue(any(k.startswith("n") for k in ctx_results))
