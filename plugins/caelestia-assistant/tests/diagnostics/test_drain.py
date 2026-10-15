"""tests for diagnostics.drain — the streaming Drain template miner
(He, Zhu, He & Lyu, ICSM 2017) and its rule-engine front wiring.

Pinned behaviours: masking -> fixed-depth tree -> SimSeq clustering ->
template merge; bounded capacity (maxChildren fallback, maxClusters
overflow — honest, never silent eviction); caller-supplied dates
(nothing reads the clock); persistence round-trip; and the
augment_diagnosis contract (NO_MATCH/AMBIGUOUS only, MATCH untouched,
shapes-not-causes note carried).
"""
import unittest

from assistant.capabilities.diagnostics import drain, engine


class MaskTests(unittest.TestCase):
    def test_ip_number_hex_masked(self):
        patterns = drain.DEFAULT_MASK_PATTERNS
        self.assertEqual(drain.mask_token("10.0.0.7", patterns), "<*>")
        self.assertEqual(drain.mask_token("123", patterns), "<*>")
        self.assertEqual(drain.mask_token("deadbeef01", patterns), "<*>")
        self.assertEqual(drain.mask_token("kwin", patterns), "kwin")

    def test_masking_makes_shapes_cluster(self):
        miner = drain.DrainMiner()
        miner.update("connection to 10.0.0.1 failed")
        row = miner.update("connection to 192.168.3.9 failed")
        self.assertEqual(row["count"], 2)
        self.assertEqual(row["template"], "connection to <*> failed")


class ClusterTests(unittest.TestCase):
    def test_identical_lines_one_cluster(self):
        miner = drain.DrainMiner()
        for _ in range(5):
            row = miner.update("plasma-shell crashed at startup")
        self.assertEqual(row["count"], 5)
        self.assertEqual(len(miner.clusters), 1)

    def test_different_token_count_separate_clusters(self):
        miner = drain.DrainMiner()
        miner.update("connection to host failed")
        row = miner.update("connection to the remote host failed")
        self.assertEqual(row["count"], 1)
        self.assertEqual(len(miner.clusters), 2)

    def test_one_divergent_position_merges_to_mask(self):
        miner = drain.DrainMiner()
        miner.update("latency of disk sda is 120 ms")
        row = miner.update("latency of disk sdb is 120 ms")
        # both tokens diverge -> both positions masked
        self.assertEqual(row["template"],
                         "latency of disk <*> is <*> ms")
        self.assertEqual(row["count"], 2)

    def test_entirely_new_shape_below_threshold_new_cluster(self):
        miner = drain.DrainMiner()
        miner.update("a b c d e f")
        row = miner.update("a x x x x x")
        # similarity 1/6 < 0.4 -> new cluster, template NOT rewritten
        self.assertEqual(row["count"], 1)
        self.assertEqual(len(miner.clusters), 2)

    def test_first_seen_last_seen_caller_supplied(self):
        miner = drain.DrainMiner()
        miner.update("shape one two three", "2026-09-20")
        row = miner.update("shape one two three", "2026-09-28")
        self.assertEqual(row["first_seen"], "2026-09-20")
        self.assertEqual(row["last_seen"], "2026-09-28")

    def test_undated_runs_report_undated(self):
        miner = drain.DrainMiner()
        miner.update("shape one two three")
        row = miner.templates()[0]
        self.assertIsNone(row["first_seen"])

    def test_empty_line_refused(self):
        miner = drain.DrainMiner()
        with self.assertRaises(ValueError):
            miner.update("   ")

    def test_determinism_same_stream_same_state(self):
        lines = ["connection to 10.0.0.1 failed", "disk sda latency 120 ms",
                 "connection to 10.0.0.2 failed"] * 3
        a = drain.mine(lines)
        b = drain.mine(lines)
        self.assertEqual(a["templates"], b["templates"])

    def test_capacity_report(self):
        miner = drain.DrainMiner()
        miner.update("one two three")
        cap = miner.capacity()
        self.assertEqual(cap["clusters"], 1)
        self.assertEqual(cap["overflowed_lines"], 0)


class BoundedCapacityTests(unittest.TestCase):
    def test_max_children_fallback_keeps_bounded(self):
        miner = drain.DrainMiner(max_children=2, max_clusters=100)
        # 5 distinct first tokens with 4 children capacity worth of tree
        for i in range(5):
            miner.update(f"tok{i} a b c")
        cap = miner.capacity()
        self.assertLessEqual(cap["clusters"], 100)
        self.assertEqual(sum(r["count"] for r in miner.templates()),
                         miner.n_lines)

    def test_max_clusters_overflow_is_honest(self):
        miner = drain.DrainMiner(max_children=8, max_clusters=3)
        for i in range(5):
            miner.update(f"totally{i} distinct shape here now ok")
        rows = miner.templates()
        overflowed = [r for r in rows if r["overflowed"]]
        self.assertTrue(overflowed)
        self.assertEqual(miner.capacity()["overflowed_lines"], 2)
        self.assertEqual(
            drain.SHAPES_NOT_CAUSES in drain.mine(
                ["totally0 distinct shape here now ok"])["note"],
            True)


class PersistenceTests(unittest.TestCase):
    def test_round_trip_resumes_counts(self):
        miner = drain.DrainMiner()
        miner.update("alpha beta gamma delta", "2026-09-01")
        miner.update("alpha beta gamma delta", "2026-09-02")
        restored = drain.DrainMiner.from_dict(miner.to_dict())
        row = restored.update("alpha beta gamma delta", "2026-09-03")
        self.assertEqual(row["count"], 3)
        self.assertEqual(row["first_seen"], "2026-09-01")
        self.assertEqual(row["last_seen"], "2026-09-03")

    def test_round_trip_templates_equal(self):
        miner = drain.DrainMiner()
        for ln in ("x 1 2 3", "x 9 9 9", "y z w v u"):
            miner.update(ln)
        self.assertEqual(
            drain.DrainMiner.from_dict(miner.to_dict()).templates(),
            miner.templates())

    def test_state_key_registered(self):
        self.assertEqual(drain.STATE_KEY, "drain_miner")


class BatchTests(unittest.TestCase):
    def test_mine_skips_blank_lines_counts_rest(self):
        result = drain.mine(["line one alpha", "", "   ",
                             "line one beta"])
        self.assertEqual(result["n_lines"], 2)
        rows = result["templates"]
        self.assertEqual(rows[0]["count"], 2)

    def test_mine_templates_sorted_count_desc_then_template(self):
        result = drain.mine(["a b c", "a b c", "a b c", "d e f", "d e f",
                             "g h i"])
        counts = [r["count"] for r in result["templates"]]
        self.assertEqual(counts, sorted(counts, reverse=True))


class AugmentTests(unittest.TestCase):
    NO_MATCH_TEXT = ("totally unknown shape alpha beta gamma\n"
                     "totally unknown shape alpha beta gamma\n"
                     "widget frobnicator quux 2026")

    def test_no_match_gets_templates_and_note(self):
        diagnosis = engine.diagnose(self.NO_MATCH_TEXT)
        self.assertEqual(diagnosis["verdict"], "NO_MATCH")
        unmatched = diagnosis["unmatched_templates"]
        self.assertEqual(unmatched["n_lines"], 3)
        self.assertIn("shapes", unmatched["note"])
        self.assertTrue(unmatched["rows"][0]["count"] >= 2)

    def test_match_verdict_untouched(self):
        known = ("Symptom: every window has a visible shadow\n"
                 "the shadow follows the window everywhere\n")
        diagnosis = engine.diagnose(known)
        if diagnosis["verdict"] == "MATCH":
            self.assertNotIn("unmatched_templates", diagnosis)

    def test_render_block_note_present(self):
        diagnosis = engine.diagnose(self.NO_MATCH_TEXT)
        lines = engine._drain_lines(diagnosis)
        self.assertTrue(lines)
        self.assertIn("shapes, not causes", "\n".join(lines))

    def test_render_block_empty_when_match(self):
        diagnosis = {"verdict": "MATCH", "candidates": [], "top": None}
        self.assertEqual(engine._drain_lines(diagnosis), [])


if __name__ == "__main__":
    unittest.main()
