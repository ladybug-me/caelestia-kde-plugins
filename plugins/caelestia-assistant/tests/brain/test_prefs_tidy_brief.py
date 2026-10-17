"""brain tests for prefs, tidy, brief — the new round-three modules."""
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.capabilities.brain import brief, prefs, tidy


class TestPrefs(unittest.TestCase):
    def test_beta_update_and_bias(self):
        m = prefs.PreferenceModel()
        for i in range(9):
            m.observe("bar", "increase", accepted=True, hour=20)
        m.observe("bar", "increase", accepted=False, hour=20)
        b = m.bias("bar", "increase", hour=20)
        self.assertEqual(b["n"], 10)
        self.assertGreater(b["p_accept"], 0.8)
        self.assertEqual(b["verdict"], "usually approved")
        lo, hi = b["ci95"]
        self.assertLess(lo, b["p_accept"])
        self.assertLess(b["p_accept"], hi)

    def test_no_data_is_honest(self):
        m = prefs.PreferenceModel()
        b = m.bias("dock", "increase", hour=3)
        self.assertEqual(b["n"], 0)
        self.assertEqual(b["verdict"], "unknown")
        self.assertAlmostEqual(b["p_accept"], 0.5, places=3)  # weak prior
        self.assertIn("no reliable preference", m.explain("dock", "increase"))

    def test_hour_buckets_separate(self):
        m = prefs.PreferenceModel()
        for _ in range(6):
            m.observe("bar", "increase", True, hour=20)     # evening approves
        for _ in range(6):
            m.observe("bar", "increase", False, hour=2)     # night rejects
        self.assertGreater(m.bias("bar", "increase", 20)["p_accept"],
                           m.bias("bar", "increase", 2)["p_accept"])

    def test_rank_reorders_by_learned_preference(self):
        m = prefs.PreferenceModel()
        for _ in range(8):
            m.observe("animations", "decrease", True, hour=None)
        for _ in range(8):
            m.observe("effects", "decrease", False, hour=None)
        cands = [
            {"group": "effects", "direction": "decrease", "confidence": 0.9},
            {"group": "animations", "direction": "decrease", "confidence": 0.9},
        ]
        ranked = m.rank(cands)
        self.assertEqual(ranked[0]["group"], "animations")
        self.assertIn("pref_score", ranked[0])

    def test_ledger_learning(self):
        m = prefs.PreferenceModel()
        items = [
            {"target": "bar.scale", "diff": {"old": 1.0, "new": 0.9},
             "status": "approved", "decided_at": "2026-09-20T20:00:00+00:00"},
            {"target": "bar.spacing", "diff": {"old": 4, "new": 2},
             "status": "rejected", "decided_at": "2026-09-21T20:30:00+00:00"},
            {"target": "bar.blur", "diff": {"old": True, "new": False},
             "status": "pending", "decided_at": None},
        ]
        n = m.from_ledger(items)
        self.assertEqual(n, 2)  # pending excluded
        self.assertEqual(m.bias("bar", "decrease", hour=20)["n"], 2)   # 1.0->0.9, 4->2
        self.assertEqual(m.bias("bar", "increase", hour=20)["n"], 0)

    def test_direction_inference(self):
        self.assertEqual(prefs._diff_direction({"old": 1.0, "new": 2.0}), "increase")
        self.assertEqual(prefs._diff_direction({"old": 2.0, "new": 1.0}), "decrease")
        self.assertEqual(prefs._diff_direction({"old": True, "new": False}), "toggle")
        self.assertEqual(prefs._diff_direction("weird"), "set")

    def test_serialisation_roundtrip(self):
        m = prefs.PreferenceModel()
        m.observe("bar", "increase", True, hour=9)
        m2 = prefs.PreferenceModel().from_dict(m.to_dict())
        self.assertEqual(m.bias("bar", "increase", 9), m2.bias("bar", "increase", 9))


class TestTidy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "downloads"
        self.root.mkdir()
        (self.root / "a.bin").write_bytes(b"same-bytes" * 100)
        (self.root / "b.bin").write_bytes(b"same-bytes" * 100)
        (self.root / "c.bin").write_bytes(b"different" * 100)
        (self.root / "photo.jpg").write_bytes(b"\xff\xd8jpegdata")
        (self.root / "notes.txt").write_text("hello")
        old = datetime.now().timestamp() - 200 * 86400
        os.utime(self.root / "notes.txt", (old, old))

    def tearDown(self):
        self.tmp.cleanup()

    def test_survey_finds_duplicates_and_routes_types(self):
        plan = tidy.survey(str(self.root))
        dup_paths = [mv["from"] for mv in plan["duplicates"]]
        self.assertEqual(len(dup_paths), 1)
        self.assertTrue(dup_paths[0].endswith(("a.bin", "b.bin")))
        moves = {Path(mv["from"]).name: mv["category"] for mv in plan["type_moves"]}
        self.assertEqual(moves.get("photo.jpg"), "images")
        self.assertEqual(moves.get("notes.txt"), "documents")
        self.assertGreater(plan["space_recoverable"], 0)
        self.assertTrue(all(s.startswith("SUGGESTED_NOT_EXECUTED:")
                            for s in plan["inert_suggestions"]))

    def test_unsafe_roots_refused(self):
        with self.assertRaises(ValueError):
            tidy.survey("/")
        with self.assertRaises(ValueError):
            tidy.survey("~")

    def test_apply_and_rollback_roundtrip(self):
        plan = tidy.survey(str(self.root))
        journal = Path(self.tmp.name) / "journal.json"
        moves = plan["type_moves"][:1]
        result = tidy.apply_moves(moves, str(self.root), str(journal))
        self.assertEqual(len(result["applied"]), 1)
        moved_to = Path(result["applied"][0]["to"])
        self.assertTrue(moved_to.exists())
        self.assertFalse(Path(moves[0]["from"]).exists())
        rb = tidy.rollback(str(journal))
        self.assertEqual(len(rb["undone"]), 1)
        self.assertTrue(Path(moves[0]["from"]).exists())
        self.assertFalse(moved_to.exists())

    def test_apply_refuses_outside_root(self):
        outside = Path(self.tmp.name) / "outside.bin"
        outside.write_bytes(b"x")
        res = tidy.apply_moves([{"from": str(outside), "to": str(self.root / "x.bin")}],
                               str(self.root))
        self.assertEqual(res["applied"], [])
        self.assertEqual(res["skipped"][0]["reason"], "outside root")

    def test_apply_never_overwrites_existing_target(self):
        src = self.root / "c.bin"
        existing = self.root / "images"
        existing.mkdir()
        (existing / "photo.jpg").write_bytes(b"occupied")
        res = tidy.apply_moves(
            [{"from": str(self.root / "photo.jpg"), "to": str(existing / "photo.jpg")}],
            str(self.root))
        self.assertEqual(len(res["applied"]), 1)
        self.assertNotEqual(Path(res["applied"][0]["to"]), existing / "photo.jpg")
        self.assertTrue((existing / "photo.jpg").exists())  # untouched

    def test_render_plan_plain(self):
        plan = tidy.survey(str(self.root))
        text = tidy.render_plan(plan)
        self.assertIn("tidy plan", text)
        self.assertIn("SUGGESTED_NOT_EXECUTED", text)

    def test_empty_dir_detection(self):
        (self.root / "void" / "deeper").mkdir(parents=True)
        plan = tidy.survey(str(self.root))
        self.assertIn(str(self.root / "void" / "deeper"), plan["empty_dirs"])


class TestBrief(unittest.TestCase):
    def test_compose_empty_is_honest(self):
        b = brief.compose()
        self.assertEqual(b["decisions"]["pending"], 0)
        self.assertIn("clear board", b["headline"])
        self.assertEqual(b["forecast"], {})

    def test_compose_sections(self):
        b = brief.compose(
            pending=[{"id": 1, "kind": "settings_plan", "target": "bar.scale",
                      "confidence": 0.7}],
            stuck=[{"title": "rewrite installer", "p_finish": 0.08}],
            today={"chosen": [{"title": "write report", "minutes_est": 45}]},
            forecast={"level": 3.2, "trend": 0.4, "forecast": 3.6},
            pref_lines=["you usually APPROVE bar increases in the evening (85% of 9)"],
            scan_alarm={"change_at": 14},
        )
        self.assertEqual(b["headline"],
                         "something changed in your logs mid-stream — "
                         "the rate detector alarmed")
        self.assertEqual(b["decisions"]["pending"], 1)
        self.assertEqual(b["decisions"]["needs_attention"], 1)
        text = brief.render(b)
        self.assertIn("today's plan", text)
        self.assertIn("rate change detected at chunk 14", text)
        json.dumps(b)  # serialisable

    def test_pending_heavy_headline(self):
        b = brief.compose(pending=[{"id": i} for i in range(7)])
        self.assertIn("7 decisions are waiting on you", b["headline"])


if __name__ == "__main__":
    unittest.main()
