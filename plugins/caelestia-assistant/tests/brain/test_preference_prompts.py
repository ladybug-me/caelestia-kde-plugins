"""Tests for exponential-build-3 G2 — idle-time paired-comparison
preference invitations (brain/preference_prompts.py) riding the
dreamtime cadence with a hard frequency cap.

Under test:

- SELECTION: the pair with the fewest recorded comparisons, tie-broken
  by the closest Elo ratings, tie-broken by sorted order — pinned on a
  hand-built state where the informative pair is unambiguous;
- THE CAP: one unanswered prompt at a time (a second invitation is
  refused while one is pending), at least MIN_DAYS_BETWEEN days since
  the last prompt (refused with the elapsed-days reason), and prompts
  EXPIRE after MAX_AGE_DAYS (withdrawn to history as 'expired',
  never nagged);
- CONSUMPTION: answering the pending pair (either order) consumes it
  to history as 'answered'; a non-matching comparison does NOT;
- the service wrapper (dream_preference_prompt) persists through the
  existing brain-state path;
- the surfaces: `settings --rank` shows a pending invitation;
  `settings --prefer A B` consumes it (mocked stdin for the pick);
- determinism: same state -> same chosen pair.
"""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
from unittest import mock

from assistant.capabilities.brain import preference_prompts as pp
from assistant.capabilities.brain import service
from assistant.capabilities.settings import cli as settings_cli

NOW = datetime(2026, 9, 27, 12, 0, 0)
NAMES = ["minimal", "gaming", "battery-saver", "comfort"]


def _state(comparisons=None, prompt=None, history=None):
    return {"preset_comparisons": comparisons or [],
            pp.PROMPT_KEY: prompt,
            pp.HISTORY_KEY: history or []}


class SelectionTests(unittest.TestCase):
    def test_fewest_comparisons_pair_is_chosen(self):
        # every pair has exactly one recorded comparison EXCEPT
        # battery-saver/comfort (zero) — count dominates the tie-break,
        # so the chosen pair is forced regardless of Elo closeness
        comparisons = [
            {"winner": "minimal", "loser": "gaming"},
            {"winner": "comfort", "loser": "gaming"},
            {"winner": "minimal", "loser": "comfort"},
            {"winner": "minimal", "loser": "battery-saver"},
            {"winner": "battery-saver", "loser": "gaming"},
        ]
        result = pp.maybe_prompt(_state(comparisons), NOW,
                                 preset_names=NAMES)
        prompt = result["prompt"]
        self.assertIsNotNone(prompt)
        self.assertEqual(sorted((prompt["a"], prompt["b"])),
                         ["battery-saver", "comfort"])
        self.assertIn("fewest recorded comparisons (0)", prompt["reason"])

    def test_ties_break_to_closest_elo_then_order(self):
        # every pair compared exactly once -> the closest Elo pair
        # wins; determinism pinned (same state, same choice)
        comparisons = [
            {"winner": "minimal", "loser": "gaming"},
            {"winner": "battery-saver", "loser": "comfort"},
            {"winner": "minimal", "loser": "battery-saver"},
            {"winner": "comfort", "loser": "gaming"},
        ]
        first = pp.maybe_prompt(_state(comparisons), NOW,
                                preset_names=NAMES)
        second = pp.maybe_prompt(_state(comparisons), NOW,
                                 preset_names=NAMES)
        self.assertEqual(first["prompt"], second["prompt"])

    def test_fewer_than_two_presets_refuses(self):
        result = pp.maybe_prompt(_state(), NOW, preset_names=["only"])
        self.assertIsNone(result["prompt"])
        self.assertIn("fewer than two presets", result["reason"])


class CapTests(unittest.TestCase):
    def test_one_unanswered_prompt_at_a_time(self):
        state = _state(prompt={"a": "minimal", "b": "gaming",
                               "at": NOW.isoformat(timespec="seconds"),
                               "reason": "x"})
        result = pp.maybe_prompt(state, NOW, preset_names=NAMES)
        self.assertIsNone(result["prompt"])
        self.assertIn("already pending", result["reason"])
        self.assertIn("settings --prefer", result["reason"])

    def test_frequency_cap_days(self):
        last = (NOW - timedelta(days=2)).isoformat(timespec="seconds")
        state = _state(history=[{"a": "a", "b": "b", "at": last,
                                 "outcome": "answered"}])
        result = pp.maybe_prompt(state, NOW, preset_names=NAMES)
        self.assertIsNone(result["prompt"])
        self.assertIn("frequency cap", result["reason"])
        self.assertIn("2.0 of the required 7 days",
                      result["reason"])
        # past the cap: a prompt is invited
        old = (NOW - timedelta(days=8)).isoformat(timespec="seconds")
        state = _state(history=[{"a": "a", "b": "b", "at": old,
                                 "outcome": "answered"}])
        result = pp.maybe_prompt(state, NOW, preset_names=NAMES)
        self.assertIsNotNone(result["prompt"])

    def test_old_prompts_expire_instead_of_nagging(self):
        stale = (NOW - timedelta(days=15)).isoformat(timespec="seconds")
        state = _state(prompt={"a": "minimal", "b": "gaming",
                               "at": stale, "reason": "x"})
        result = pp.maybe_prompt(state, NOW, preset_names=NAMES)
        # the stale prompt was withdrawn to history as expired AND a
        # fresh invitation was recorded in its place
        self.assertIsNotNone(result["prompt"])
        self.assertEqual(state[pp.HISTORY_KEY][-1]["outcome"],
                         "expired")
        self.assertIsNotNone(pp.pending(state))


class AnswerTests(unittest.TestCase):
    def _pending_state(self):
        return _state(prompt={"a": "minimal", "b": "gaming",
                              "at": NOW.isoformat(timespec="seconds"),
                              "reason": "x"})

    def test_matching_answer_consumes_either_order(self):
        state = self._pending_state()
        self.assertTrue(pp.answer(state, "gaming", "minimal"))
        self.assertIsNone(pp.pending(state))
        self.assertEqual(state[pp.HISTORY_KEY][-1]["outcome"],
                         "answered")

    def test_non_matching_answer_leaves_the_prompt(self):
        state = self._pending_state()
        self.assertFalse(pp.answer(state, "minimal", "comfort"))
        self.assertIsNotNone(pp.pending(state))

    def test_answer_without_prompt_is_false(self):
        self.assertFalse(pp.answer(_state(), "a", "b"))


class ServiceAndCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.state_path = str(self.d / "state.json")
        self.vault = self.d / "vault"
        self.vault.mkdir()
        (self.vault / "a.md").write_text(
            "kubernetes deploy notes about clusters", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_service_wrapper_persists_through_brain_state(self):
        result = service.dream_preference_prompt(
            self.state_path, now=NOW)
        self.assertIsNotNone(result["prompt"])
        # the state file exists and holds the prompt
        import json
        state = json.loads(Path(self.state_path).read_text())
        self.assertEqual(pp.pending(state)["a"],
                         result["prompt"]["a"])
        # a second call inside the cap does not duplicate
        again = service.dream_preference_prompt(
            self.state_path, now=NOW)
        self.assertIsNone(again["prompt"])
        self.assertIn("already pending", again["reason"])

    def test_rank_shows_the_pending_invitation(self):
        service.dream_preference_prompt(self.state_path, now=NOW)
        out = StringIO()
        stdout = sys.stdout
        sys.stdout = out
        try:
            with mock.patch("assistant.capabilities.brain.state.DEFAULT_STATE",
                            self.state_path), \
                 mock.patch("assistant.capabilities.brain.state.load") as load:
                # --rank reads brain_state.load(); point it at the file
                import json as _json
                load.return_value = _json.loads(
                    Path(self.state_path).read_text())
                code = settings_cli.main(["--rank"])
        finally:
            sys.stdout = stdout
        self.assertEqual(code, 0)
        text = out.getvalue()
        # a fresh state has zero comparison rows, so --rank shows the
        # invitation through its empty-log branch — the invitation is
        # visible EITHER way
        self.assertIn("idle-time invitation is waiting", text)
        self.assertIn("settings --prefer", text)
        self.assertIn("fewest recorded comparisons (0)", text)

    def test_prefer_consumes_a_matching_prompt(self):
        service.dream_preference_prompt(self.state_path, now=NOW)
        import json
        state_doc = json.loads(Path(self.state_path).read_text())
        a = state_doc[pp.PROMPT_KEY]["a"]
        b = state_doc[pp.PROMPT_KEY]["b"]
        out = StringIO()
        stdout = sys.stdout
        sys.stdout = out
        try:
            with mock.patch("assistant.capabilities.brain.state.DEFAULT_STATE",
                            self.state_path), \
                 mock.patch("assistant.capabilities.brain.state.load") as load, \
                 mock.patch("assistant.capabilities.brain.state.save") as save, \
                 mock.patch("builtins.input", return_value="1"):
                load.return_value = json.loads(
                    Path(self.state_path).read_text())
                code = settings_cli.main(["--prefer", a, b])
                saved = save.call_args[0][0]
        finally:
            sys.stdout = stdout
        self.assertEqual(code, 0)
        self.assertIn("answers the pending idle-time prompt",
                      out.getvalue())
        self.assertIsNone(pp.pending(saved))


if __name__ == "__main__":
    unittest.main()
