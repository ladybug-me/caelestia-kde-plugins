"""Tests for the shell-native brain modules, second round: textmine,
spellfix, rhythm, placement, calibrate, drift, dreamtime — and the
service.py / bridge.py wiring that exposes them (shell-native ops only).

(The personal-PKM engine tests live in assistant/brain/personal/tests/.)
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.brain import (
    calibrate, drift, dreamtime, placement, rhythm, spellfix, textmine,
)
from assistant.capabilities.brain import bridge, cli, service


class TextMineTests(unittest.TestCase):
    def test_keywords_favor_distinctive_words(self):
        corpus = ["kubernetes cluster deploy pods"] * 3
        kws = textmine.keywords("kubernetes cluster kustomize overlays", corpus, top=3)
        self.assertTrue(any("kustomize" in k for k in kws))

    def test_tfidf_vectors_and_cosine(self):
        # phase 3.2 seam: whole-corpus TF-IDF vectors + cosine similarity
        vecs = textmine.tfidf_vectors([
            "sourdough starter flour water recipe",
            "sourdough starter feeding flour schedule",
            "quarterly budget spreadsheet totals",
        ])
        self.assertEqual(len(vecs), 3)
        similar = textmine.cosine(vecs[0], vecs[1])
        unlike = textmine.cosine(vecs[0], vecs[2])
        self.assertGreater(similar, unlike)
        self.assertGreater(similar, 0.0)
        self.assertLessEqual(similar, 1.0 + 1e-9)
        # empty vocabulary is not a similarity
        self.assertEqual(textmine.cosine({}, vecs[1]), 0.0)
        # determinism: same corpus, same vectors
        again = textmine.tfidf_vectors([
            "sourdough starter flour water recipe",
            "sourdough starter feeding flour schedule",
            "quarterly budget spreadsheet totals",
        ])
        self.assertEqual(vecs, again)

    def test_summarize_shorter_than_target_returns_everything(self):
        out = textmine.summarize("One short sentence only.", sentences_out=3)
        self.assertEqual(out, ["One short sentence only."])

    def test_summarize_picks_fewer_sentences_in_order(self):
        doc = ("First sentence here. Second sentence adds detail. "
               "Third sentence concludes. Fourth sentence drifts off topic.")
        out = textmine.summarize(doc, sentences_out=2)
        self.assertEqual(len(out), 2)
        self.assertIn("First", out[0])


class SpellFixTests(unittest.TestCase):
    def test_flags_rare_near_miss_of_common_word(self):
        texts = ["the quick brown fox", "the slow brown dog", "the quick red fox"] * 3
        idx = spellfix.SpellIndex().build(texts)
        rows = idx.suggest(top=5)
        self.assertTrue(all("likely" in r for r in rows))

    def test_no_suggestions_for_uniform_vocabulary(self):
        idx = spellfix.SpellIndex().build(["alpha beta", "alpha beta"])
        self.assertEqual(idx.suggest(top=5), [])


class RhythmTests(unittest.TestCase):
    def test_detects_skewed_day(self):
        weekdays = [1] * 20 + [0, 2, 3, 4, 5, 6]
        pattern = rhythm.day_of_week_pattern(weekdays)
        top = max(pattern, key=lambda r: r["z"])
        self.assertEqual(top["day"], "Tue")

    def test_notable_filters_small_deviations(self):
        pattern = [{"day": "Mon", "count": 5, "z": 0.1}, {"day": "Tue", "count": 5, "z": 3.0}]
        self.assertEqual(rhythm.notable(pattern, threshold=1.5), [pattern[1]])


class PlacementTests(unittest.TestCase):
    def test_ranks_closest_registry_entry(self):
        registry = [
            {"id": "bar.thickness", "description": "height and thickness of the top bar"},
            {"id": "blur.radius", "description": "background blur radius amount"},
        ]
        ranked = placement.suggest("make my bar thinner", registry)
        self.assertEqual(ranked[0]["id"], "bar.thickness")

    def test_no_match_returns_empty(self):
        registry = [{"id": "x", "description": "completely unrelated words here"}]
        self.assertEqual(placement.suggest("zzz qqq", registry), [])


class CalibrateTests(unittest.TestCase):
    def test_acceptance_rate_reflects_history(self):
        labeled = [{"kind": "tag", "status": "approved"}] * 8 + [{"kind": "tag", "status": "rejected"}] * 2
        rates = calibrate.acceptance_rate(labeled)
        self.assertGreater(rates["tag"]["mean"], 0.7)
        self.assertEqual(rates["tag"]["n"], 10)

    def test_daily_budget_roundtrip_and_reward(self):
        budget = calibrate.DailyBudget()
        budget.reward("more", True)
        budget.reward("less", False)
        restored = calibrate.DailyBudget.from_dict(budget.to_dict())
        self.assertEqual(restored.alpha["more"], budget.alpha["more"])
        self.assertEqual(restored.beta["less"], budget.beta["less"])


class DriftTests(unittest.TestCase):
    def test_detects_added_removed_and_changed(self):
        base = " ".join(f"word{i}" for i in range(60))
        old = {"a": base, "b": "will be removed " + base}
        new = {"a": base, "c": "brand new content entirely different words here"}
        result = drift.diff(old, new)
        self.assertEqual(result["added"], ["c"])
        self.assertEqual(result["removed"], ["b"])
        self.assertEqual(result["changed"], [])

    def test_reworded_item_flagged_as_changed(self):
        old = {"a": "one two three four five six seven eight nine ten"}
        new = {"a": "completely different words now unrelated to before at all"}
        result = drift.diff(old, new, similarity_threshold=0.5)
        self.assertEqual(result["changed"][0]["id"], "a")


class DreamtimeTests(unittest.TestCase):
    def test_not_eligible_on_battery(self):
        self.assertFalse(dreamtime.eligible(idle_minutes=30, on_ac_power=False, cpu_load_percent=5))

    def test_eligible_schedules_within_budget(self):
        jobs = [{"id": "organize", "minutes": 10, "value": 5},
                {"id": "health", "minutes": 10, "value": 8},
                {"id": "drift", "minutes": 10, "value": 1}]
        self.assertTrue(dreamtime.eligible(idle_minutes=30, on_ac_power=True, cpu_load_percent=5))
        chosen = dreamtime.schedule(jobs, budget_min=20)
        self.assertEqual(chosen, ["health", "organize"])


class ServiceAndCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.state = str(self.d / "state.json")
        self.ledger = str(self.d / "ledger.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_placement_propose_writes_ledger_entry(self):
        registry = [{"id": "bar.thickness", "description": "bar height thickness"}]
        ranked = service.placement_propose("make bar thinner", registry, self.ledger, propose=True)
        self.assertEqual(ranked[0]["id"], "bar.thickness")
        pending = service.ledger_list(self.ledger)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "placement")

    def test_calibration_report_after_some_decisions(self):
        from assistant.capabilities.brain.ledger import Ledger
        ledger = Ledger(self.ledger)
        pid1 = ledger.propose("settings", "shell.json", {}, "reason", 0.9)
        ledger.decide(pid1, True)
        report = service.calibration_report(self.ledger)
        self.assertIn("settings", report["acceptance_by_kind"])

    def test_dream_window_via_bridge(self):
        r = bridge.handle({"op": "dream_window", "idle_minutes": 30, "on_ac_power": True,
                           "cpu_load_percent": 5,
                           "jobs": [{"id": "health", "minutes": 10, "value": 5}],
                           "budget_min": 20},
                          state_path=self.state, ledger_path=self.ledger)
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"]["eligible"])
        self.assertEqual(r["result"]["chosen"], ["health"])

    def test_personal_ops_are_not_on_the_bridge(self):
        for op in ("organize", "tag", "plan", "review_grade", "estimate_observe",
                   "spellcheck", "ghosts", "journal_report", "health_report",
                   "links_suggest", "note_keywords", "ledger_learn"):
            r = bridge.handle({"op": op}, state_path=self.state, ledger_path=self.ledger)
            self.assertFalse(r["ok"], op)
            self.assertIn("unknown op", r["error"])

    def test_personal_cli_smoke_via_service_forecast(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--state", self.state, "--ledger", self.ledger,
                             "forecast", "1,2,3"], out)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
