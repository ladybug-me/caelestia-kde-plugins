"""Tests for the personal CLI and service wiring (moved from brain/tests
during the issue #120 scope split). These cover the personal entry point
`python3 -m assistant.capabilities.brain.personal` only.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import ledger
from assistant.capabilities.brain.personal import cli, service


class PersonalServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.vault = self.d / "vault"
        self.vault.mkdir()
        (self.vault / "a.md").write_text("kubernetes deploy notes about clusters", encoding="utf-8")
        (self.vault / "b.md").write_text("cooking notes about sourdough bread baking", encoding="utf-8")
        self.state = str(self.d / "state.json")
        self.ledger = str(self.d / "ledger.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_links_suggest_via_service(self):
        result = service.links_suggest(str(self.vault), self.ledger, propose=True)
        self.assertIsInstance(result, list)

    def test_keywords_and_summarize_via_service(self):
        kws = service.note_keywords(str(self.vault), "a.md")
        self.assertIn("kubernetes", kws)
        lines = service.note_summarize(str(self.vault), "a.md")
        self.assertTrue(lines)

    def test_journal_record_resolve_report_roundtrip(self):
        service.journal_record("d1", "the deploy will work", 0.8, self.state)
        service.journal_resolve("d1", True, self.state)
        report = service.journal_report(self.state)
        self.assertAlmostEqual(report["brier_score"], 0.04)

    def test_health_report_via_service(self):
        rows = service.health_report(str(self.vault))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("score" in r for r in rows))

    def test_estimate_observe_then_query(self):
        service.estimate_observe("writing", 50, self.state)
        est = service.estimate_query("writing", self.state)
        self.assertEqual(est["n"], 1)


class PersonalCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        (self.vault / "kube.md").write_text(
            "tags: [devops]\nDeploy the kubernetes cluster pods with docker images and helm charts. [[deploy-notes]]")
        (self.vault / "kube-copy.md").write_text(
            "Deploy the kubernetes cluster pods with docker images and helm charts. [[deploy-notes]]")
        (self.vault / "bread.md").write_text(
            "tags: [cooking]\nSourdough flour yeast bake bread with starter.")
        (self.vault / "untagged.md").write_text(
            "docker images pods cluster deploy release notes")
        (self.vault / "lonely.md").write_text("no links here at all")
        self.state = self.root / "state.json"
        self.led = self.root / "ledger.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out = io.StringIO()
        code = cli.main(["--state", str(self.state), "--ledger", str(self.led), *args], out=out)
        return code, out.getvalue()

    def test_organize_proposes_and_reports(self):
        code, text = self.run_cli("organize", str(self.vault), "--dup", "0.6")
        self.assertEqual(code, 0)
        self.assertIn("duplicate pairs: 1", text)
        led = ledger.Ledger(self.led)
        kinds = {i["kind"] for i in led.pending()}
        self.assertIn("merge_duplicate", kinds)
        self.assertIn("orphan", kinds)

    def test_tag_command_ranks_devops_for_docker_text(self):
        code, text = self.run_cli("tag", str(self.vault), "docker pods deploy")
        self.assertEqual(code, 0)
        self.assertTrue(text.splitlines()[0].startswith("devops"))

    def test_plan_end_to_end_with_proposals(self):
        tasks = self.root / "tasks.json"
        tasks.write_text(json.dumps([
            {"id": "a", "title": "Write spec", "effort_min": 60, "importance": 0.9, "deadline_days": 1},
            {"id": "b", "title": "Read paper", "effort_min": 90, "importance": 0.3, "deps": ["a"]},
            {"id": "c", "title": "Email", "effort_min": 10, "importance": 0.4, "deadline_days": 30},
        ]))
        code, text = self.run_cli("plan", str(tasks), "--minutes", "100", "--propose", "--top", "2")
        self.assertEqual(code, 0)
        self.assertIn("ranked:", text)
        self.assertIn("critical path:", text)
        self.assertEqual(len(ledger.Ledger(self.led).pending()), 2)
        code, text = self.run_cli("ledger", "learn")
        self.assertIn("learned from 0", text)  # nothing decided yet -> no labeled examples

    def test_estimate_observe_then_query(self):
        self.run_cli("estimate", "observe", "writing", "50")
        _, text = self.run_cli("estimate", "query", "writing")
        self.assertEqual(json.loads(text)["n"], 1)

    def test_review_grade_and_due(self):
        self.run_cli("review", "grade", "card1", "3", "--days", "0")
        _, text = self.run_cli("review", "due", "--days", "30")
        self.assertIn("card1", text)

    def test_remind_feedback_then_choose_in_window(self):
        self.run_cli("remind", "feedback", "10", "1")
        _, text = self.run_cli("remind", "choose", "--allowed", "9-11")
        self.assertIn(int(text.strip()), range(9, 12))

    def test_remind_rejects_invalid_hour(self):
        with self.assertRaises(ValueError):
            service.remind_feedback(99, True, self.state)

    def test_cull(self):
        hist = self.root / "hist.json"
        hist.write_text(json.dumps({
            "durations": [1, 2, 3, 30, 40, 50],
            "observed": [True, True, True, False, False, False],
            "open": [{"id": "old", "age": 45}, {"id": "fresh", "age": 1}],
        }))
        _, text = self.run_cli("cull", str(hist))
        self.assertIn("cull? old", text)

    def test_spellcheck_and_ghosts_cli_smoke(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--state", str(self.state), "--ledger", str(self.led),
                             "spellcheck", str(self.vault)], out)
        self.assertEqual(code, 0)
        out2 = io.StringIO()
        code2 = cli.main(["--state", str(self.state), "--ledger", str(self.led),
                          "ghosts", str(self.vault)], out2)
        self.assertEqual(code2, 0)

    def test_links_cli_smoke(self):
        code, text = self.run_cli("links", str(self.vault))
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
