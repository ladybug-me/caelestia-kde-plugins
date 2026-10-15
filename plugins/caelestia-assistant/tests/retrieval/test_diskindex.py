"""Tests for exponential-build phase 4.1: the disk-backed, bounded-
memory personal search index (`retrieval/diskindex.py`).

The contract under test:

- an arbitrary folder tree indexes into a term-sorted on-disk postings
  file via an external merge sort (heapq.merge), with runs spilled
  when the ceiling-derived character budget fills;
- queries load ONLY metadata — no postings dict exists on the searcher
  — and score with the SAME idf/k1/b as the in-memory Searcher;
- the footprint claim is MEASURED, not asserted: the build report
  carries peak RSS (VmHWM) before/after, the enforced run budget, and
  the run count; a multi-run build produces the same answers as a
  single-run build;
- binary files are skipped and counted; unreadable files are counted;
  deterministic builds are byte-identical.
"""
import tempfile
import unittest
from pathlib import Path

from assistant.capabilities.retrieval import diskindex
from assistant.capabilities.retrieval.diskindex import DiskSearcher, build_disk_index
from assistant.capabilities.retrieval.search import Searcher  # the in-memory baseline


def _tree(root: Path):
    (root / "notes").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "notes" / "kde.txt").write_text(
        "the kde compositor kwin handles the panel and the desktop effects",
        encoding="utf-8")
    (root / "notes" / "linux.txt").write_text(
        "the linux kernel schedules processes and the kde session sits on top",
        encoding="utf-8")
    (root / "logs" / "battery.log").write_text(
        "battery drain thermal kernel battery thermal\n" * 12,
        encoding="utf-8")
    # enough distinct terms that a 64 KB run budget spills several runs
    (root / "notes" / "bulk.txt").write_text(
        " ".join(f"termmmm{i:04d}tt shared" for i in range(3000)),
        encoding="utf-8")
    (root / "logs" / "blob.bin").write_bytes(b"\x00\x01\x02\x00binary")
    (root / "empty.txt").write_text("", encoding="utf-8")


class BuildTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        _tree(self.dir / "tree")

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_reports_measured_footprint(self):
        report = build_disk_index(str(self.dir / "tree"),
                                  str(self.dir / "idx"),
                                  ram_ceiling_mb=1)
        fp = report["footprint"]
        # the claim lives in numbers, not prose
        self.assertTrue(fp["measured"])
        self.assertEqual(fp["ram_ceiling_mb"], 1)
        self.assertIsNotNone(fp["hwm_start_kb"])
        self.assertIsNotNone(fp["hwm_end_kb"])
        self.assertGreater(fp["run_budget_chars"], 0)
        self.assertTrue(fp["budget_respected_by_construction"])
        self.assertIn("hwm_growth_within_ceiling", fp)
        self.assertIn("WHOLE PROCESS peak", fp["note"])

    def test_binary_and_empty_files_are_counted_not_indexed(self):
        report = build_disk_index(str(self.dir / "tree"),
                                  str(self.dir / "idx"))
        self.assertEqual(report["walk"]["binary_likely"], 1)
        # kde.txt, linux.txt, battery.log, bulk.txt, empty.txt
        # (blob.bin skipped): the empty file still yields a doc row
        self.assertEqual(report["n_docs"], 5)
        # the binary blob's content is not searchable
        searcher = DiskSearcher(str(self.dir / "idx"))
        self.assertEqual(searcher.search("binary"), [])

    def test_missing_directory_refused(self):
        with self.assertRaises(ValueError):
            build_disk_index(str(self.dir / "nope"), str(self.dir / "idx"))

    def test_nonpositive_ceiling_refused(self):
        with self.assertRaises(ValueError):
            build_disk_index(str(self.dir / "tree"), str(self.dir / "idx"),
                             ram_ceiling_mb=0)


class MultiRunMergeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        _tree(self.dir / "tree")

    def tearDown(self):
        self._tmp.cleanup()

    def test_tiny_budget_spills_runs_and_answers_identically(self):
        # a 64 KB budget forces several runs for even this small tree
        r_multi = build_disk_index(str(self.dir / "tree"),
                                   str(self.dir / "idx_multi"),
                                   ram_ceiling_mb=0.0625)
        r_single = build_disk_index(str(self.dir / "tree"),
                                    str(self.dir / "idx_single"),
                                    ram_ceiling_mb=64)
        self.assertGreater(r_multi["footprint"]["runs_spilled"], 1)
        # the big-ceiling build still flushes its one and only run
        self.assertEqual(r_single["footprint"]["runs_spilled"], 1)
        # the enforced budget was never crossed, in either build
        self.assertTrue(r_multi["footprint"]["budget_respected_by_construction"])
        self.assertTrue(r_single["footprint"]["budget_respected_by_construction"])
        # the merged answers are byte-identical to the single-run build
        multi = (Path(self.dir) / "idx_multi" / "postings.jsonl").read_text()
        single = (Path(self.dir) / "idx_single" / "postings.jsonl").read_text()
        self.assertEqual(multi, single)

    def test_deterministic_builds_are_byte_identical(self):
        build_disk_index(str(self.dir / "tree"), str(self.dir / "idx1"))
        build_disk_index(str(self.dir / "tree"), str(self.dir / "idx2"))
        a = (self.dir / "idx1" / "postings.jsonl").read_text()
        b = (self.dir / "idx2" / "postings.jsonl").read_text()
        self.assertEqual(a, b)


class QueryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        _tree(self.dir / "tree")
        build_disk_index(str(self.dir / "tree"), str(self.dir / "idx"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_finds_the_right_doc(self):
        s = DiskSearcher(str(self.dir / "idx"))
        hits = s.search("compositor kwin")
        self.assertTrue(hits)
        self.assertIn("kde.txt", hits[0]["path"])

    def test_no_postings_attribute_on_the_searcher(self):
        # the RAM-flat property, structurally: no postings dict exists
        s = DiskSearcher(str(self.dir / "idx"))
        self.assertFalse(hasattr(s, "postings"))
        self.assertFalse(hasattr(s, "df"))

    def test_scores_match_the_in_memory_searcher_semantics(self):
        # relative ordering by tf: the battery log repeats its terms
        s = DiskSearcher(str(self.dir / "idx"))
        hits = s.search("battery thermal")
        self.assertTrue(hits)
        self.assertIn("battery.log", hits[0]["path"])
        # a term absent from the corpus yields the honest empty answer
        self.assertEqual(s.search("xenon"), [])

    def test_missing_index_refused(self):
        with self.assertRaises(ValueError):
            DiskSearcher(str(self.dir / "nothing"))

    def test_k_limit_respected(self):
        s = DiskSearcher(str(self.dir / "idx"))
        hits = s.search("kernel", k=1)  # in two docs; 'the' is a stopword
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
