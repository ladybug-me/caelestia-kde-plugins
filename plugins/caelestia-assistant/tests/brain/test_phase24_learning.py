"""Phase 2.4 self-learning upgrades: LinUCB, shared feature hashing,
the Ebbinghaus half-life setting, the shared ranking primitive, and
calibration surfacing."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from assistant.capabilities.brain.features import (DEFAULT_DIMENSIONS, hash_features,
                                      text_features, context_features)
from assistant.capabilities.brain.preset_bandit import LinUCBBandit, NamedBandit
from assistant.capabilities.brain.ranking import (EloLadder, bradley_terry, ladder_report)
from assistant.core.learn import CortexLearner
from assistant.core.memory import (
    HALFLIFE_DAYS,
    MEMORY_HALFLIFE_KEY,
    new_episode,
    recall,
    resolve_halflife,
)


class TestFeatureHashing(unittest.TestCase):
    def test_deterministic_and_pure(self):
        a = text_features("make my bar thinner")
        b = text_features("make my bar thinner")
        self.assertEqual(a, b)
        self.assertEqual(len(a), DEFAULT_DIMENSIONS)

    def test_l2_normalized(self):
        vec = text_features("the quick brown fox jumps repeatedly")
        norm = sum(v * v for v in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_different_texts_differ(self):
        self.assertNotEqual(text_features("bar thinner"),
                            text_features("wallpaper random"))

    def test_context_changes_features(self):
        self.assertNotEqual(context_features("tidy", hour=9),
                            context_features("tidy", hour=21))

    def test_wrong_dimension_rejected(self):
        with self.assertRaises(ValueError):
            hash_features(["a"], d=0)


class TestLinUCB(unittest.TestCase):
    def _seeded(self):
        """compact approves ~80% in morning text contexts, ~20% else."""
        rng = random.Random(7)
        bandit = LinUCBBandit(d=8)
        for day in range(60):
            hour = rng.choice([8, 9, 10]) if day % 2 == 0 else rng.choice([20, 21, 22])
            text = "make it compact" if day % 2 == 0 else "make it minimal"
            approved = rng.random() < (0.8 if day % 2 == 0 else 0.2)
            bandit.reward("compact" if day % 2 == 0 else "minimal", approved,
                          text=text, hour=hour)
        return bandit

    def test_learns_context_preference(self):
        bandit = self._seeded()
        morning = bandit.rank(["compact", "minimal"], text="make it tidy", hour=9)
        self.assertEqual(morning[0][0], "compact")

    def test_round_trip(self):
        bandit = self._seeded()
        clone = LinUCBBandit.from_dict(bandit.to_dict(), d=8)
        ctx = context_features("make it tidy", 9, d=8)
        self.assertEqual(bandit.rank(["compact", "minimal"], context=ctx),
                         clone.rank(["compact", "minimal"], context=ctx))

    def test_reward_shapes_match_paper(self):
        bandit = LinUCBBandit(d=4)
        ctx = [1.0, 0.0, 0.0, 0.0]
        bandit.reward("x", True, context=ctx)
        arm = bandit.arms["x"]
        # A += x x^T: only the (0,0) cell gains 1 beyond identity
        self.assertEqual(arm["A"][0][0], 2.0)
        self.assertEqual(arm["A"][1][1], 1.0)
        # b += r x with r=+1
        self.assertEqual(arm["b"][0], 1.0)

    def test_new_arms_start_neutral(self):
        bandit = LinUCBBandit(d=4)
        rows = bandit.rank(["never-seen"], context=[1, 0, 0, 0])
        # exploit = 0 for a fresh arm; only exploration uncertainty scores
        self.assertEqual(rows[0][2], 0.0)

    def test_named_bandit_unchanged(self):
        # the context-blind bandit keeps byte-identical reward semantics
        nb = NamedBandit()
        nb.reward("compact", True)
        self.assertEqual(nb.to_dict(), {"compact": [2.0, 1.0]})


class TestEbbinghausHalflife(unittest.TestCase):
    def test_default_and_override(self):
        self.assertEqual(resolve_halflife({}), HALFLIFE_DAYS)
        self.assertEqual(resolve_halflife({MEMORY_HALFLIFE_KEY: 60.0}), 60.0)

    def test_garbage_degrades_to_default(self):
        for bad in ("fast", -5, 99999, None, [7]):
            self.assertEqual(resolve_halflife({MEMORY_HALFLIFE_KEY: bad}),
                             HALFLIFE_DAYS, bad)

    def test_recall_respects_halflife(self):
        now = datetime(2026, 9, 26)
        old = new_episode("old request", "old", ["setBarScale"], "plan",
                          "applied", at=now - timedelta(days=30))
        recent = new_episode("new request", "new", ["setBlurEnabled"], "plan",
                             "applied", at=now - timedelta(days=1))
        episodes = [old, recent]
        short = recall(episodes, now=now, k=2, halflife_days=3.0)
        long_ = recall(episodes, now=now, k=2, halflife_days=365.0)
        # a 3-day half-life demotes the 30-day-old episode below the 1-day
        self.assertEqual([e["text"] for e in short][0], "new request")
        # both still recalled; ORDER of the long one also recent-first
        self.assertEqual(long_[0]["text"], "new request")

    def test_cli_halflife_read_and_bounds(self):
        import contextlib
        import io
        from assistant.core import cli as cortex_cli

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cortex_cli.main(["halflife"])
        self.assertEqual(code, 0)
        self.assertIn("memory decay half-life", out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cortex_cli.main(["halflife", "99999.0"])
        self.assertEqual(code, 1)
        self.assertIn("within", err.getvalue())


class TestRankingPrimitive(unittest.TestCase):
    TRUE = {"compact": 1.2, "minimal": 0.4, "gaming": 0.0,
            "macos-like": -0.5, "battery-saver": -1.0}

    def _pairs(self, n=300, seed=2026):
        rng = random.Random(seed)
        items = list(self.TRUE)
        pairs = []
        for _ in range(n):
            a, b = rng.sample(items, 2)
            p_a = 1.0 / (1.0 + 2.718281828 ** (self.TRUE[b] - self.TRUE[a]))
            winner = a if rng.random() < p_a else b
            pairs.append((winner, a if winner == b else b))
        return pairs

    def test_bradley_terry_recovers_ground_truth(self):
        # the proposal's own verification plan: Kendall tau >= 0.8
        bt = bradley_terry(self._pairs())
        order = sorted(self.TRUE, key=lambda i: -bt["strengths"][i])
        true_order = sorted(self.TRUE, key=lambda i: -self.TRUE[i])
        self.assertEqual(order, true_order)

    def test_elo_and_bt_agree(self):
        report = ladder_report(self._pairs())
        self.assertGreaterEqual(report["agreement_kendall_tau"], 0.8)

    def test_deterministic(self):
        self.assertEqual(bradley_terry(self._pairs()),
                         bradley_terry(self._pairs()))

    def test_sparsity_honesty(self):
        ladder = EloLadder()
        ladder.observe("a", "b")
        rows = ladder.ladder()
        self.assertTrue(all(not r["enough_data"] for r in rows))
        self.assertIn("not enough data", " ".join(
            "enough" if r["enough_data"] else "not enough data" for r in rows))

    def test_elo_update_moves_ratings_apart(self):
        ladder = EloLadder()
        ladder.observe("winner", "loser")
        self.assertGreater(ladder.ratings["winner"],
                           ladder.ratings["loser"])

    def test_self_comparison_rejected(self):
        with self.assertRaises(ValueError):
            EloLadder().observe("a", "a")

    def test_elo_round_trip(self):
        ladder = EloLadder()
        for winner, loser in self._pairs(50):
            ladder.observe(winner, loser)
        clone = EloLadder.from_dict(ladder.to_dict())
        self.assertEqual(clone.ratings, ladder.ratings)


class TestPresetRankingCli(unittest.TestCase):
    def test_prefer_records_and_rank_reads(self):
        import contextlib
        import io
        from assistant.capabilities.brain import state as brain_state
        from assistant.capabilities.settings import cli as settings_cli

        state_path = Path(tempfile.mkdtemp()) / "state.json"
        with mock.patch.object(brain_state, "load", return_value={}), \
                mock.patch.object(brain_state, "save") as save:
            # the pick arrives on stdin; nothing is applied
            stdin = io.StringIO("1\n")
            with mock.patch("sys.stdin", stdin), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                code = settings_cli.main(["--prefer", "compact", "minimal"])
            self.assertEqual(code, 0)
            saved = save.call_args[0][0]
            rows = saved["preset_comparisons"]
            self.assertEqual(rows[-1]["winner"], "compact")
            self.assertEqual(rows[-1]["loser"], "minimal")
        self.assertIn("recorded: compact > minimal", out.getvalue())
        self.assertIn("dry-run, nothing written", out.getvalue())

    def test_rank_reports_ladder(self):
        import contextlib
        import io
        from assistant.capabilities.brain import state as brain_state
        from assistant.capabilities.settings import cli as settings_cli

        state = {"preset_comparisons": [
            {"winner": "compact", "loser": "minimal"},
            {"winner": "compact", "loser": "minimal"},
            {"winner": "compact", "loser": "minimal"},
        ]}
        with mock.patch.object(brain_state, "load", return_value=state):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = settings_cli.main(["--rank"])
        self.assertEqual(code, 0)
        self.assertIn("preference ladder", out.getvalue())
        self.assertIn("compact", out.getvalue())

    def test_rank_with_no_data_is_honest(self):
        import contextlib
        import io
        from assistant.capabilities.brain import state as brain_state
        from assistant.capabilities.settings import cli as settings_cli

        with mock.patch.object(brain_state, "load", return_value={}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = settings_cli.main(["--rank"])
        self.assertEqual(code, 0)
        self.assertIn("no pairwise comparisons recorded", out.getvalue())


class TestCalibrationSurfacing(unittest.TestCase):
    def test_note_only_when_sample_supports_it(self):
        learner = CortexLearner()
        self.assertIsNone(learner.calibration_note(0.92))
        for i in range(25):
            learner.observe(f"request {i}", "setBarScale", {"lex": 0.9},
                            0.92, "applied")
        note = learner.calibration_note(0.92)
        self.assertIsNotNone(note)
        self.assertIn("% of the time", note)
        self.assertIn("25 decisions", note)

    def test_note_in_chat_card(self):
        import contextlib
        import io
        import sys

        from assistant.core import cli as cortex_cli
        from assistant.core.learn import CortexLearner

        learner = CortexLearner()
        # The note needs >= 5 observations in the bucket the LIVE route's
        # softmax p lands in. The pipeline's p depends on the sampled
        # strategy state and its route call, so the fixture observes into
        # EVERY bucket (0.2 / 0.4 / 0.6 / 0.9 cover p30-, p30-50, p50-75,
        # p75+): with history in all buckets the note must surface for
        # any honest route — no stale p pinning.
        for i in range(30):
            for p in (0.2, 0.4, 0.6, 0.9):
                learner.observe(f"make my bar thinner {i}", "setBarScale",
                                {"lex": 0.9}, p, "applied")
        target = Path(tempfile.mkdtemp()) / "shell.json"
        target.write_text(json.dumps({"bar": {"scale": 1.0}}))
        from assistant.capabilities.brain import state as brain_state
        with mock.patch.object(cortex_cli, "_load_learner", return_value=learner), \
                mock.patch.object(brain_state, "save"), \
                mock.patch("sys.stdin", io.StringIO("make my bar thinner\n/exit\n")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cortex_cli.cmd_chat(["--file", str(target)])
        self.assertIn("% of the time", out.getvalue())

    def test_note_in_dispatch_answer(self):
        from assistant.core.dispatch import dispatch

        state = {"cortex_learn": None}
        learner = CortexLearner()
        # observe into every bucket (see test_note_in_chat_card): the note
        # must surface wherever the live route's softmax p lands
        for i in range(30):
            for p in (0.2, 0.4, 0.6, 0.9):
                learner.observe(f"make my bar thinner {i}", "setBarScale",
                                {"lex": 0.9}, p, "applied")
        target = Path(tempfile.mkdtemp()) / "shell.json"
        target.write_text(json.dumps({"bar": {"scale": 1.0}}))
        state = {"cortex_learn": learner.to_dict()}
        outcome = dispatch("make my bar thinner", state=state,
                           file_path=str(target))
        self.assertEqual(outcome["action"], "local")
        self.assertTrue(any("% of the time" in line
                            for line in outcome["answer"]))


if __name__ == "__main__":
    unittest.main()
