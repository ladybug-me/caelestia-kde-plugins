"""Tests for exponential-build phase 1.1: gap-cluster -> tool-template
stub drafting (``cortex gaps --draft-stubs``).

The contract under test:

- a DENSE (support >= 3, purity >= 0.6 — the workspace.py floors),
  UNADDRESSED (no ledger item of any status for its
  ``gap-cluster:<label>`` target) cluster drafts a reviewable stub file
  carrying a name, the cluster's cue words and a TODO body;
- a cluster the ledger already knows about — pending, approved OR
  rejected — is skipped as addressed (the ledger flow IS the address;
  no second review surface stacks on the same need);
- an existing stub file is never clobbered (it may carry human edits);
- NOTHING is registered or wired: the draft writes stub files only, and
  leaves the learned state, the gap bucket and the ledger untouched.
"""
import io
import json
import os
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from assistant.capabilities.brain import state as brain_state
from assistant.capabilities.brain.ledger import Ledger
from assistant.core import dispatch
from assistant.core.cli import cmd_cortex

GAPS_KEY = dispatch.GAPS_KEY


def _gap(shape, n, category="genius"):
    return {"shape": shape, "category": category, "n": n,
            "first_at": "2026-09-27T00:00:00",
            "last_at": "2026-09-27T00:00:00"}


class SlugTests(unittest.TestCase):
    def test_slug_is_deterministic_and_filesystem_safe(self):
        a = dispatch.tool_template_slug("convert png webp")
        b = dispatch.tool_template_slug("convert png webp")
        self.assertEqual(a, b)
        self.assertEqual(a, "convert_png_webp")
        self.assertEqual(dispatch.tool_template_slug("weird / label!!"),
                         "weird_label")
        # an empty label never produces an empty filename
        self.assertEqual(dispatch.tool_template_slug("///"), "unnamed_cluster")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = {GAPS_KEY: [_gap(["convert", "png", "webp"], 4)]}

    def test_dense_unaddressed_cluster_is_selected(self):
        sel = dispatch.select_stub_candidates(self.state)
        self.assertEqual(len(sel["unaddressed"]), 1)
        cand = sel["unaddressed"][0]
        self.assertEqual(cand["label"], "convert png webp")
        self.assertEqual(cand["support"], 4)
        self.assertEqual(cand["purity"], 1.0)
        self.assertEqual(sel["addressed"], [])
        # selection never mutates the state it reads
        self.assertEqual(len(self.state[GAPS_KEY]), 1)

    def test_scattered_gaps_select_nothing(self):
        self.state = {GAPS_KEY: [_gap(["a", "b"], 1), _gap(["c", "d"], 1),
                                 _gap(["e", "f"], 1)]}
        sel = dispatch.select_stub_candidates(self.state)
        self.assertEqual(sel["unaddressed"], [])
        self.assertEqual(sel["summary"]["candidates"], [])

    def test_ledger_known_cluster_is_addressed_whatever_its_status(self):
        path = Path(self.tmp.name) / "ledger.json"
        # pending counts as addressed
        ledger = Ledger(path)
        pid = ledger.propose("ontology_gap", "gap-cluster:convert png webp",
                             {}, "why", 0.9)
        sel = dispatch.select_stub_candidates(self.state, ledger)
        self.assertEqual(sel["unaddressed"], [])
        self.assertEqual([c["label"] for c in sel["addressed"]],
                         ["convert png webp"])
        # approved counts as addressed too — the human already decided
        ledger.decide(pid, True)
        sel = dispatch.select_stub_candidates(self.state, ledger)
        self.assertEqual(sel["unaddressed"], [])
        # and so does rejected (the ledger keeps its first resolution, so
        # a rejected need gets a FRESH proposal — the ledger's own
        # duplicate rule — not a stub alongside the dead one)
        rejected = Ledger(path)
        rejected.propose("ontology_gap", "gap-cluster:convert png webp",
                         {}, "why", 0.9)
        rejected.decide(max(p["id"] for p in rejected.items), False)
        sel = dispatch.select_stub_candidates(self.state, Ledger(path))
        self.assertEqual(sel["unaddressed"], [])

    def test_only_the_addressed_cluster_is_filtered(self):
        self.state = {GAPS_KEY: [_gap(["convert", "png", "webp"], 4),
                                 _gap(["rename", "batch", "files"], 4)]}
        ledger = Ledger(Path(self.tmp.name) / "ledger.json")
        ledger.propose("ontology_gap", "gap-cluster:convert png webp",
                       {}, "why", 0.9)
        sel = dispatch.select_stub_candidates(self.state, ledger)
        self.assertEqual([c["label"] for c in sel["unaddressed"]],
                         ["rename batch files"])
        self.assertEqual([c["label"] for c in sel["addressed"]],
                         ["convert png webp"])


class StubTextTests(unittest.TestCase):
    def test_stub_text_carries_name_cues_todo_and_draft_marker(self):
        cand = {"tokens": ["convert", "png", "webp"], "category": "genius",
                "support": 4, "purity": 0.75, "shapes": 2,
                "label": "convert png webp"}
        text = dispatch.tool_template_stub_text(cand, "2026-09-27T00:00:00")
        self.assertIn("# tool-template: convert_png_webp", text)
        self.assertIn("status: DRAFT", text)
        self.assertIn("cue words: convert, png, webp", text)
        self.assertIn("TODO:", text)
        self.assertIn("NOT registered in the router", text)
        self.assertIn("purity 0.75", text)
        self.assertIn("4 request(s)", text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_path = Path(self.tmp.name) / "state.json"
        self.ledger_path = Path(self.tmp.name) / "ledger.json"
        self.stubs_dir = Path(self.tmp.name) / "tool_templates"

    def _cli(self, argv):
        brain_state.save({GAPS_KEY: [_gap(["convert", "png", "webp"], 4)]},
                         self.state_path)
        env = {"CAELESTIA_BRAIN_STATE": str(self.state_path)}
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch.dict(os.environ, env):
                with unittest.mock.patch(
                        "assistant.capabilities.brain.cli.DEFAULT_LEDGER",
                        self.ledger_path):
                    rc = cmd_cortex(argv)
        return rc, out.getvalue()

    def test_draft_stubs_writes_one_reviewable_file(self):
        rc, printed = self._cli(
            ["gaps", "--draft-stubs", "--stubs-dir", str(self.stubs_dir)])
        self.assertEqual(rc, 0)
        stub = self.stubs_dir / "convert_png_webp.md"
        self.assertTrue(stub.exists())
        text = stub.read_text(encoding="utf-8")
        self.assertIn("# tool-template: convert_png_webp", text)
        self.assertIn("status: DRAFT", text)
        self.assertIn("cue words: convert, png, webp", text)
        self.assertIn("TODO:", text)
        self.assertIn("drafted 1 tool-template stub(s)", printed)
        # the ledger was never written by a draft-only run
        self.assertFalse(self.ledger_path.exists())

    def test_draft_stubs_leave_learned_state_untouched(self):
        brain_state.save({GAPS_KEY: [_gap(["convert", "png", "webp"], 4)]},
                         self.state_path)
        before = brain_state.load(self.state_path)
        rc, _ = self._cli(
            ["gaps", "--draft-stubs", "--stubs-dir", str(self.stubs_dir)])
        self.assertEqual(rc, 0)
        after = brain_state.load(self.state_path)
        self.assertEqual(before, after)
        self.assertEqual(after[GAPS_KEY],
                         [_gap(["convert", "png", "webp"], 4)])

    def test_stub_name_is_never_a_registry_tool(self):
        from assistant.adapters.caelestia.registry import TOOL_SPECS
        rc, _ = self._cli(
            ["gaps", "--draft-stubs", "--stubs-dir", str(self.stubs_dir)])
        self.assertEqual(rc, 0)
        self.assertNotIn("convert_png_webp", TOOL_SPECS)

    def test_existing_stub_file_is_never_clobbered(self):
        self.stubs_dir.mkdir(parents=True)
        stub = self.stubs_dir / "convert_png_webp.md"
        stub.write_text("HUMAN EDIT — keep", encoding="utf-8")
        rc, printed = self._cli(
            ["gaps", "--draft-stubs", "--stubs-dir", str(self.stubs_dir)])
        self.assertEqual(rc, 0)
        self.assertEqual(stub.read_text(encoding="utf-8"),
                         "HUMAN EDIT — keep")
        self.assertIn("never clobbered", printed)
        self.assertNotIn("drafted 1", printed)

    def test_propose_and_draft_stubs_do_both_in_one_run(self):
        rc, printed = self._cli(
            ["gaps", "--propose", "--draft-stubs", "--stubs-dir",
             str(self.stubs_dir)])
        self.assertEqual(rc, 0)
        self.assertTrue((self.stubs_dir / "convert_png_webp.md").exists())
        ledger = Ledger(self.ledger_path)
        self.assertEqual(len(ledger.pending()), 1)
        self.assertEqual(ledger.items[0]["target"],
                         "gap-cluster:convert png webp")
        self.assertIn("pending proposal(s) recorded", printed)
        self.assertIn("drafted 1 tool-template stub(s)", printed)

    def test_without_the_flag_nothing_is_drafted(self):
        rc, printed = self._cli(["gaps"])
        self.assertEqual(rc, 0)
        self.assertFalse(self.stubs_dir.exists())
        self.assertNotIn("tool-template stub", printed)


if __name__ == "__main__":
    unittest.main()
