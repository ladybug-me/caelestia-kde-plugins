"""settings_bridge + preset_bandit: proposals, approval writes, rejection
doesn't, and acceptance learning ranks presets the user actually keeps."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import cli, service
from assistant.capabilities.brain.ledger import Ledger
from assistant.capabilities.brain.preset_bandit import NamedBandit
from assistant.capabilities.brain import settings_bridge


class SettingsBridgeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.shell = self.dir / "shell.json"
        self.shell.write_text("{}", encoding="utf-8")
        self.ledger_path = self.dir / "ledger.json"
        self.state_path = self.dir / "state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_propose_never_writes(self):
        r = service.settings_propose(str(self.shell), self.ledger_path, preset="compact")
        self.assertIsNotNone(r["proposal_id"])
        self.assertFalse(r["blocked"])
        self.assertTrue(r["preview"])
        # dry-run: the target file is untouched, no backup, no history.
        self.assertEqual(self.shell.read_text(), "{}")
        self.assertFalse((self.dir / "shell.json.assistant-backup").exists())

    def test_approve_writes_through_the_real_applier(self):
        r = service.settings_propose(str(self.shell), self.ledger_path, preset="compact")
        outcome = service.settings_decide(r["proposal_id"], True, self.ledger_path,
                                          self.state_path)
        self.assertTrue(outcome["applied"])
        written = json.loads(self.shell.read_text())
        self.assertEqual(written["bar"]["dock"]["iconSize"], 24)
        # the applier's own safety guarantees still hold: backup + history.
        self.assertTrue((self.dir / "shell.json.assistant-backup").exists())
        self.assertTrue((self.dir / "shell.json.assistant-history.json").exists())

    def test_reject_writes_nothing(self):
        r = service.settings_propose(str(self.shell), self.ledger_path, preset="gaming")
        outcome = service.settings_decide(r["proposal_id"], False, self.ledger_path,
                                          self.state_path)
        self.assertFalse(outcome["applied"])
        self.assertEqual(self.shell.read_text(), "{}")

    def test_unknown_preset_raises_before_touching_the_ledger(self):
        with self.assertRaises(settings_bridge.SettingsBridgeError):
            settings_bridge.propose(Ledger(self.ledger_path), self.shell, preset="not-a-preset")
        self.assertEqual(Ledger(self.ledger_path).pending(), [])

    def test_explicit_calls_propose_like_a_preset_but_untagged(self):
        r = service.settings_propose(str(self.shell), self.ledger_path,
                                     calls=[{"tool": "setBarScale", "action": "set",
                                             "value": 0.8, "raw": "test"}])
        self.assertIsNotNone(r["proposal_id"])
        item = Ledger(self.ledger_path)._get(r["proposal_id"])
        self.assertIsNone(item["diff"]["preset"])

    def test_recommend_learns_from_approve_reject_history(self):
        bandit = NamedBandit()
        for _ in range(8):
            bandit.reward("compact", True)
        for _ in range(8):
            bandit.reward("gaming", False)
        ranked = {name: mean for name, _, mean in
                  bandit.rank(["compact", "gaming", "minimal"], rng=__import__("random").Random(1))}
        self.assertGreater(ranked["compact"], ranked["minimal"])
        self.assertGreater(ranked["minimal"], ranked["gaming"])

    def test_cli_round_trip_propose_decide_recommend(self):
        def run(*args):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(["--ledger", str(self.ledger_path),
                                 "--state", str(self.state_path)] + list(args))
            return code, out.getvalue()

        code, out = run("settings", "propose", str(self.shell), "--preset", "compact")
        self.assertEqual(code, 0)
        self.assertIn("proposal #1 pending", out)

        code, out = run("settings", "decide", "1", "approve")
        self.assertEqual(code, 0)
        self.assertIn("approved and written", out)

        code, out = run("settings", "recommend")
        self.assertEqual(code, 0)
        self.assertIn("compact", out)

    def test_confidence_defaults_to_flat_prior_with_no_history(self):
        r = service.settings_propose(str(self.shell), self.ledger_path, preset="compact")
        item = Ledger(self.ledger_path)._get(r["proposal_id"])
        self.assertEqual(item["confidence"], settings_bridge.DEFAULT_CONFIDENCE)

    def test_confidence_recalibrates_after_rejections(self):
        # Reject three settings proposals in a row: the ledger's own
        # acceptance_rate for kind="settings" should now be well under the
        # flat 0.7 prior, and a fresh propose() should reflect that -- an
        # honestly pessimistic default, not a stuck optimistic one.
        for _ in range(3):
            r = service.settings_propose(str(self.shell), self.ledger_path, preset="gaming")
            service.settings_decide(r["proposal_id"], False, self.ledger_path, self.state_path)
        r = service.settings_propose(str(self.shell), self.ledger_path, preset="minimal")
        item = Ledger(self.ledger_path)._get(r["proposal_id"])
        self.assertLess(item["confidence"], settings_bridge.DEFAULT_CONFIDENCE)

    def test_per_tool_learning_survives_raw_calls_not_just_presets(self):
        # Two raw --call proposals touching setBarScale: approve one,
        # reject the other elsewhere, then a preset that also touches
        # setBarScale should inherit that tool's track record.
        ops = [{"tool": "setBarScale", "action": "set", "value": 0.9, "raw": "t"}]
        r1 = service.settings_propose(str(self.shell), self.ledger_path, calls=ops)
        service.settings_decide(r1["proposal_id"], True, self.ledger_path, self.state_path)
        state = json.loads(self.state_path.read_text())
        self.assertIn("tool:setBarScale", state["preset_bandit"])
        self.assertEqual(state["preset_bandit"]["tool:setBarScale"], [2.0, 1.0])

    def test_recommend_tools_ranks_by_touched_tool_history(self):
        approved = [{"tool": "setBarScale", "action": "set", "value": 0.9, "raw": "t"}]
        rejected = [{"tool": "setDockIconSize", "action": "set", "value": 20, "raw": "t"}]
        r1 = service.settings_propose(str(self.shell), self.ledger_path, calls=approved)
        service.settings_decide(r1["proposal_id"], True, self.ledger_path, self.state_path)
        r2 = service.settings_propose(str(self.shell), self.ledger_path, calls=rejected)
        service.settings_decide(r2["proposal_id"], False, self.ledger_path, self.state_path)
        # rank() orders by one stochastic Thompson draw each, so compare the
        # stable "confidence" (posterior mean) field, not list position.
        by_tool = {r["tool"]: r["confidence"] for r in
                  service.settings_recommend_tools(self.state_path)}
        self.assertGreater(by_tool["setBarScale"], by_tool["setDockIconSize"])


if __name__ == "__main__":
    unittest.main()
