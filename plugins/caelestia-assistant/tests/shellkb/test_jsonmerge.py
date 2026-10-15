"""shellkb.jsonmerge tests — flatten, Myers, TED, three-way merge."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from assistant.capabilities.shellkb import jsonmerge as jm  # noqa: E402

BASE = {"bar": {"scale": 1.0, "persistent": True, "position": "bottom"},
        "blur": {"enabled": False}}
OURS = {"bar": {"scale": 1.2, "persistent": True, "position": "bottom"},
        "blur": {"enabled": True}}
THEIRS = {"bar": {"scale": 1.5, "persistent": True, "position": "bottom"},
          "blur": {"enabled": False},
          "clock": {"show_seconds": True}}


class TestFlatten(unittest.TestCase):
    def test_dotted_paths_sorted(self):
        rows = jm.flatten(BASE)
        paths = [p for p, _ in rows]
        self.assertEqual(paths, sorted(paths))
        self.assertIn("bar.scale", paths)

    def test_lists_indexed(self):
        paths = [p for p, _ in jm.flatten({"w": [1, [2]]})]
        self.assertEqual(paths, ["w.0", "w.1.0"])

    def test_real_null_is_a_value_not_absence(self):
        rows = dict(jm.flatten({"a": None}))
        self.assertEqual(rows["a"], None)


class TestMyers(unittest.TestCase):
    def test_identity_gives_equal_only(self):
        seq = ["a", "b", "c"]
        ops = jm.myers_ops(seq, list(seq))
        self.assertTrue(all(op == "=" for op, _, _ in ops))

    def test_insert_and_delete(self):
        a_seq = ["a", "b", "c"]
        ops = jm.myers_ops(a_seq, ["a", "x", "c"])
        script = "".join(op for op, _, _ in ops)
        self.assertIn("+", script)
        self.assertIn("-", script)
        kept = [a_seq[i] for op, i, _ in ops if op == "="]
        self.assertEqual(kept, ["a", "c"])

    def test_minimal_edit_count(self):
        # one substitution reads as one delete + one insert
        ops = jm.myers_ops(["a"], ["b"])
        self.assertEqual(sorted(op for op, _, _ in ops), ["+", "-"])


class TestTED(unittest.TestCase):
    def test_identical_documents_are_zero(self):
        self.assertEqual(jm.tree_edit_distance(BASE, BASE), 0)

    def test_leaf_change_is_one_edit(self):
        self.assertEqual(jm.tree_edit_distance(BASE, OURS), 2)

    def test_added_key_is_one_edit(self):
        bigger = {**BASE, "clock": {"show_seconds": True}}
        self.assertEqual(jm.tree_edit_distance(BASE, bigger), 2)

    def test_removed_subtree_counts_all_nodes(self):
        self.assertEqual(jm.tree_edit_distance(BASE, {"bar": BASE["bar"]}),
                         2)  # blur + blur.enabled

    def test_root_scalar(self):
        self.assertEqual(jm.tree_edit_distance(5, 6), 1)
        self.assertEqual(jm.tree_edit_distance(5, 5), 0)

    def test_node_cap_abstains(self):
        big = {"k": list(range(3000))}
        self.assertIsNone(jm.tree_edit_distance(big, big))


class TestDiff(unittest.TestCase):
    def test_diff_reports_changes_sorted(self):
        data = jm.diff_docs(BASE, OURS)
        self.assertEqual(data["n_changes"], 2)
        self.assertEqual([c["path"] for c in data["changes"]],
                         ["bar.scale", "blur.enabled"])

    def test_diff_identical(self):
        data = jm.diff_docs(BASE, json.loads(json.dumps(BASE)))
        self.assertEqual(data["n_changes"], 0)
        self.assertEqual(data["ted"], 0)


class TestMerge(unittest.TestCase):
    def test_clean_changes_taken_per_side(self):
        m = jm.merge_three_way(BASE, OURS, THEIRS)
        sources = {(t["path"], t["source"]) for t in m["taken"]}
        self.assertIn(("blur.enabled", "ours"), sources)
        self.assertIn(("clock.show_seconds", "theirs"), sources)

    def test_conflicting_change_keeps_base_in_merged(self):
        m = jm.merge_three_way(BASE, OURS, THEIRS)
        self.assertEqual(m["n_conflicts"], 1)
        c = m["conflicts"][0]
        self.assertEqual(c["kind"], "value-vs-value")
        self.assertEqual(c["ours"], 1.2)
        self.assertEqual(c["theirs"], 1.5)
        self.assertEqual(m["merged"]["bar"]["scale"], 1.0,
                         "conflicted value falls back to base in "
                         "the proposal")

    def test_merge_is_a_proposal_not_a_write(self):
        m = jm.merge_three_way(BASE, OURS, THEIRS)
        self.assertIn("PROPOSAL", m["note"])

    def test_identical_sides_agree(self):
        m = jm.merge_three_way(BASE, BASE, BASE)
        self.assertEqual(m["n_taken"], 0)
        self.assertEqual(m["n_conflicts"], 0)
        self.assertEqual(m["merged"], BASE)

    def test_add_vs_add_conflict(self):
        m = jm.merge_three_way({}, {"x": 1}, {"x": 2})
        self.assertEqual(m["conflicts"][0]["kind"], "add-vs-add")

    def test_delete_vs_edit_conflict(self):
        m = jm.merge_three_way({"a": 1}, {"a": 2}, {})
        self.assertEqual(m["conflicts"][0]["kind"], "delete-vs-edit")

    def test_null_value_distinct_from_deletion(self):
        m = jm.merge_three_way({"a": None}, {"a": None}, {})
        # theirs deleted the null key: that's a change, not a conflict
        self.assertEqual(m["n_conflicts"], 0)
        self.assertEqual(m["merged"], {})

    def test_list_element_merges(self):
        m = jm.merge_three_way({"w": [1, 2, 3]}, {"w": [1, 9, 3]},
                               {"w": [1, 2, 3, 4]})
        self.assertEqual(m["merged"]["w"], [1, 9, 3, 4])
        self.assertEqual(m["n_conflicts"], 0)


class TestCliFiles(unittest.TestCase):
    def test_diff_and_merge_over_temp_files(self):
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            for name, doc in (("base.json", BASE), ("ours.json", OURS),
                              ("theirs.json", THEIRS)):
                Path(tmp, name).write_text(json.dumps(doc))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = jm.main(["diff", f"{tmp}/base.json",
                              f"{tmp}/ours.json"])
            self.assertEqual(rc, 0)
            self.assertIn("bar.scale", out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = jm.main(["merge", "--base", f"{tmp}/base.json",
                              "--ours", f"{tmp}/ours.json",
                              "--theirs", f"{tmp}/theirs.json"])
            self.assertEqual(rc, 0)
            self.assertIn("PROPOSAL", out.getvalue())
            self.assertIn("CONFLICT", out.getvalue())


if __name__ == "__main__":
    unittest.main()
